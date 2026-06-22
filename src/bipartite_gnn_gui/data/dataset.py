"""Dataset wrappers for GUI correction data.

Provides:
- GUIDataset: 2-pass cache-based dataset that parses raw VLM + GT JSON,
  runs Hungarian matching, normalises, extracts features, and caches as .pt.
- collate_variable_elements: dynamic N_max padding per batch.
- create_dataloader: standard PyTorch DataLoader wrapper.
- GUIDataModule: self-contained train/val/test split with lazy cache build.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import DataLoader, Dataset, Tensor, torch

from bipartite_gnn_gui.data.ground_truth import (
    GTElement,
    GroundTruth,
    GroundTruthParseError,
    load_ground_truth,
    match_predictions_to_ground_truth,
)
from bipartite_gnn_gui.data.preprocess import (
    extract_confidence_scores,
    extract_spatial_features,
    extract_type_embedding,
    train_val_test_split,
)
from bipartite_gnn_gui.data.vlm_output import (
    ELEMENT_TYPES,
    VLMOutput,
    VlmParseError,
    parse_minimax_output,
    parse_qwen_output,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_vlm_json(raw_data: Dict[str, Any]) -> VLMOutput:
    """Auto-detect and parse a VLM JSON dict (Qwen or MiniMax format).

    Detection is based on the first element's field names:
    - "label"  -> Qwen format
    - "category" -> MiniMax format
    - otherwise defaults to Qwen.
    """
    if not isinstance(raw_data, dict):
        raise VlmParseError(f"Expected dict, got {type(raw_data).__name__}")

    elements = raw_data.get("elements", [])
    if elements and isinstance(elements, list) and len(elements) > 0:
        first = elements[0]
        if isinstance(first, dict):
            if "label" in first:
                return parse_qwen_output(raw_data)
            elif "category" in first:
                return parse_minimax_output(raw_data)
    # Fall back to Qwen parser when unknown or empty
    return parse_qwen_output(raw_data)


def _resolve_gt_path(image_id: str, root_dir: Path) -> Optional[Path]:
    """Search for a ground-truth JSON file for *image_id*.

    Tries multiple standard locations and filename variants (with or without
    extension stripping).  Also searches RICO ``unique_uis/`` subdirectories
    when present.

    Args:
        image_id: Image identifier (may include file extension).
        root_dir: Root data directory containing ``gui360/``,
            ``screenspot/``, or ``rico/unique_uis/`` subdirectories.

    Returns:
        Path to the GT file, or ``None`` if not found.
    """
    base = Path(image_id)
    stem = base.stem  # Strip any extension like .png -> login_screen

    search_dirs = [
        root_dir / "gui360",
        root_dir / "screenspot",
        root_dir,
    ]

    # Also add RICO unique_uis subdirectories if present
    rico_dir = root_dir / "unique_uis"
    if rico_dir.is_dir():
        search_dirs.extend(sorted(rico_dir.iterdir()))

    candidates: List[Path] = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        # Try with and without explicit .json extension
        if not str(image_id).endswith(".json"):
            candidates.append(d / f"{image_id}.json")
            candidates.append(d / f"{stem}.json")
        else:
            candidates.append(d / image_id)
            candidates.append(d / base.name)

    for c in candidates:
        c = Path(c)
        if c.exists() and c.suffix == ".json":
            return c.resolve()

    return None


def _type_to_index(type_str: str, taxonomy: List[str]) -> int:
    """Map an element type string to its index in the taxonomy.

    Unknown types are mapped to index 0 (the catch-all).
    """
    try:
        return taxonomy.index(type_str)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# GUIDataset -- 2-pass cache pattern
# ---------------------------------------------------------------------------


class GUIDataset(Dataset):
    """Dataset for GUI correction data using a 2-pass cache pattern.

    **Build phase** (``_build_cache``):
    Scans raw data, parses VLM JSON + GT JSON, runs Hungarian matching
    via :func:`~bipartite_gnn_gui.data.ground_truth.match_predictions_to_ground_truth`,
    normalises, extracts features, and saves each sample as a ``.pt`` file.

    **Read phase** (``__getitem__``):
    Loads the cached ``.pt`` file from disk.  No full in-memory cache.

    Args:
        image_ids: List of image identifiers (filenames without ``.json``).
        vlm_dir: Directory containing VLM prediction ``{image_id}.json`` files.
        gt_dir: Directory containing ground-truth ``{image_id}.json`` files.
        cache_dir: Directory to store cached ``{image_id}.pt`` files.
            Created automatically if it does not exist.
        taxonomy: Ordered list of canonical element type names.  Defaults to
            the 20-type taxonomy from :data:`ELEMENT_TYPES`.
        force_rebuild: If ``True``, rebuild cache even if ``.pt`` files exist.

    Raises:
        FileNotFoundError: If *vlm_dir* does not exist.
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        vlm_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        cache_dir: Union[str, Path],
        taxonomy: Optional[List[str]] = None,
        force_rebuild: bool = False,
    ) -> None:
        self.image_ids = list(image_ids)
        self.vlm_dir = Path(vlm_dir)
        self.gt_dir = Path(gt_dir)
        self.cache_dir = Path(cache_dir)
        self.taxonomy = taxonomy or list(ELEMENT_TYPES.keys())
        self.force_rebuild = force_rebuild

        if not self.vlm_dir.is_dir():
            raise FileNotFoundError(f"VLM directory not found: {self.vlm_dir}")

        # Populated during build
        self._cached_ids: List[str] = []
        self._built: bool = False

    # ------------------------------------------------------------------
    # Cache build
    # ------------------------------------------------------------------

    def _build_cache(self) -> None:
        """Parse raw data, run matching, extract features, save .pt files.

        Skips samples whose cache file already exists (unless
        ``force_rebuild=True``).  Logs a warning and skips samples where
        the GT file cannot be found or parsed.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        num_types = len(self.taxonomy)

        for image_id in self.image_ids:
            cache_path = self.cache_dir / f"{image_id}.pt"
            if cache_path.exists() and not self.force_rebuild:
                self._cached_ids.append(image_id)
                continue

            # ---- Load VLM predictions ---------------------------------
            vlm_path = self.vlm_dir / f"{image_id}.json"
            if vlm_path.exists():
                try:
                    with vlm_path.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                    vlm_output = _parse_vlm_json(raw)
                except (VlmParseError, json.JSONDecodeError, Exception) as exc:
                    logger.warning("Failed to parse VLM JSON for %s: %s", image_id, exc)
                    vlm_output = VLMOutput(image_id=image_id)
            else:
                # Missing VLM JSON -> empty predictions
                logger.debug("VLM JSON not found for %s, using empty predictions", image_id)
                vlm_output = VLMOutput(image_id=image_id)

            # ---- Load GT ----------------------------------------------
            if self.gt_dir.is_dir():
                gt_path = _resolve_gt_path(image_id, self.gt_dir)
            else:
                gt_path = None

            if gt_path is None:
                logger.warning("No GT found for %s, skipping", image_id)
                continue

            try:
                gt: GroundTruth = load_ground_truth(gt_path)
            except (GroundTruthParseError, Exception) as exc:
                logger.warning("Failed to parse GT for %s: %s, skipping", image_id, exc)
                continue

            # ---- Hungarian matching -----------------------------------
            matched_pairs, fp_indices, fn_indices = match_predictions_to_ground_truth(
                vlm_output.elements, gt.elements
            )

            N = len(vlm_output.elements)
            N_gt = len(gt.elements)

            # ---- Build tensors ----------------------------------------
            if N > 0:
                # Boxes
                vlm_boxes = torch.tensor(
                    [list(e.bbox) for e in vlm_output.elements], dtype=torch.float32
                )
                gt_boxes = torch.zeros(N, 4, dtype=torch.float32)
                for pred_idx, gt_idx in matched_pairs:
                    gt_boxes[pred_idx] = torch.tensor(
                        list(gt.elements[gt_idx].bbox), dtype=torch.float32
                    )

                # Element types (index)
                element_types = torch.tensor(
                    [_type_to_index(e.element_type, self.taxonomy) for e in vlm_output.elements],
                    dtype=torch.long,
                )

                # Concatenated features: (N, 4 + num_types + 1)
                spatial = extract_spatial_features(vlm_boxes)  # (N, 4)
                type_emb = torch.stack(
                    [extract_type_embedding(e.element_type, self.taxonomy) for e in vlm_output.elements]
                )  # (N, num_types)
                conf = extract_confidence_scores(vlm_output.elements).unsqueeze(-1)  # (N, 1)
                element_features = torch.cat([spatial, type_emb, conf], dim=-1)

                # Matched mask
                matched_mask = torch.zeros(N, dtype=torch.bool)
                for pred_idx, _ in matched_pairs:
                    matched_mask[pred_idx] = True
            else:
                vlm_boxes = torch.zeros(0, 4, dtype=torch.float32)
                gt_boxes = torch.zeros(0, 4, dtype=torch.float32)
                element_types = torch.zeros(0, dtype=torch.long)
                element_features = torch.zeros(0, 4 + num_types + 1, dtype=torch.float32)
                matched_mask = torch.zeros(0, dtype=torch.bool)

            # GT present mask (all True -- used for FN counting)
            gt_present = torch.ones(N_gt, dtype=torch.bool)

            # Image size for denormalization
            image_size = torch.tensor(
                [gt.image_width, gt.image_height], dtype=torch.float32
            )

            sample = {
                "element_features": element_features,
                "vlm_boxes": vlm_boxes,
                "gt_boxes": gt_boxes,
                "element_types": element_types,
                "image_id": image_id,
                "image_size": image_size,
                "matched_mask": matched_mask,
                "gt_present": gt_present,
            }

            torch.save(sample, cache_path)
            self._cached_ids.append(image_id)

        self._built = True

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if not self._built:
            self._build_cache()
        return len(self._cached_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self._built:
            self._build_cache()
        image_id = self._cached_ids[index]
        cache_path = self.cache_dir / f"{image_id}.pt"
        return torch.load(cache_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_variable_elements(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-size GUI samples into a padded batch.

    Each sample has ``N`` elements; the batch is padded to ``N_max``
    (the largest ``N`` in the batch).  Tensor fields are padded with
    zeros (or -1 for ``element_types``).  A ``valid_mask`` boolean
    field is added to distinguish real elements from padding.

    Args:
        batch: List of sample dicts from :class:`GUIDataset`.

    Returns:
        Dict with keys:

        - ``element_features``: ``(B, N_max, D_feat)`` padded float32.
        - ``vlm_boxes``: ``(B, N_max, 4)`` padded float32.
        - ``gt_boxes``: ``(B, N_max, 4)`` padded float32.
        - ``element_types``: ``(B, N_max)`` padded long (-1 = padding).
        - ``valid_mask``: ``(B, N_max)`` bool (``True`` = real element).
        - ``matched_mask``: ``(B, N_max)`` bool.
        - ``image_ids``: ``list[str]`` of length B.
        - ``image_sizes``: ``(B, 2)`` float32.
        - ``gt_present``: ``list[Tensor]`` of varying lengths (for FN counting).
    """
    if not batch:
        return {}

    # Determine N_max from the batch
    N_max = max(s["element_features"].size(0) for s in batch)
    D_feat = batch[0]["element_features"].size(-1)

    # Tensor fields padded to N_max
    # Each entry: (key, dtype, pad_value)
    padded_fields: List[Tuple[str, torch.dtype, Any]] = [
        ("element_features", torch.float32, 0.0),
        ("vlm_boxes", torch.float32, 0.0),
        ("gt_boxes", torch.float32, 0.0),
        ("element_types", torch.long, -1),
        ("matched_mask", torch.bool, False),
    ]

    result: Dict[str, Any] = {}

    for key, dtype, pad_value in padded_fields:
        tensors: List[Tensor] = []
        for sample in batch:
            t: Tensor = sample[key]
            N = t.size(0)
            pad_size = N_max - N
            if pad_size > 0:
                pad_shape = (pad_size,) + t.shape[1:]
                pad = torch.full(pad_shape, pad_value, dtype=dtype)
                t = torch.cat([t, pad], dim=0)
            tensors.append(t)
        result[key] = torch.stack(tensors, dim=0)

    # valid_mask
    valid_masks: List[Tensor] = []
    for sample in batch:
        N = sample["element_features"].size(0)
        mask = torch.zeros(N_max, dtype=torch.bool)
        mask[:N] = True
        valid_masks.append(mask)
    result["valid_mask"] = torch.stack(valid_masks, dim=0)

    # Non-padded fields
    result["image_ids"] = [s["image_id"] for s in batch]
    result["image_sizes"] = torch.stack([s["image_size"] for s in batch], dim=0)
    result["gt_present"] = [s["gt_present"] for s in batch]

    return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` for variable-size GUI samples.

    Args:
        dataset: A :class:`GUIDataset` instance.
        batch_size: Number of samples per batch.
        shuffle: If ``True``, shuffle the data every epoch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Configured :class:`~torch.utils.data.DataLoader`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_elements,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# GUIDataModule -- self-contained train/val/test split
# ---------------------------------------------------------------------------


class GUIDataModule:
    """Self-contained data module with lazy cache build and split support.

    Scans raw data on first access (lazy), builds cache for all splits,
    and exposes train/val/test dataloaders.

    Args:
        root_dir: Root directory containing ``vlm_predictions/``,
            ``gui360/`` and/or ``screenspot/`` subdirectories.
        cache_dir: Directory for cached ``.pt`` files.  Defaults to
            ``<root_dir>/../processed/cache``.
        val_split: Fraction of data for validation (default 0.1).
        test_split: Fraction of data for testing (default 0.1).
        seed: Random seed for deterministic splitting (default 42).
        batch_size: Batch size for all dataloaders (default 1).
        force_rebuild: If ``True``, rebuild cache even if ``.pt`` files exist.

    Raises:
        FileNotFoundError: If *root_dir* does not exist.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        batch_size: int = 1,
        force_rebuild: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        if cache_dir is None:
            cache_dir = self.root_dir.parent / "processed" / "cache"
        self.cache_dir = Path(cache_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.batch_size = batch_size
        self.force_rebuild = force_rebuild

        # Lazy -- populated on first dataloader access
        self._train_dataset: Optional[GUIDataset] = None
        self._val_dataset: Optional[GUIDataset] = None
        self._test_dataset: Optional[GUIDataset] = None
        self._built: bool = False

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        """Scan, split, build cache, and create per-split datasets."""
        if self._built:
            return

        vlm_dir = self.root_dir / "vlm_predictions"
        if not vlm_dir.is_dir():
            logger.warning("vlm_predictions directory not found: %s", vlm_dir)
            self._built = True
            return

        # Scan for image IDs (sorted for determinism)
        image_ids = sorted([p.stem for p in vlm_dir.glob("*.json")])
        if not image_ids:
            logger.warning("No VLM prediction JSONs found in %s", vlm_dir)
            self._built = True
            return

        # Split
        train_ids, val_ids, test_ids = train_val_test_split(
            image_ids, val_split=self.val_split, test_split=self.test_split, seed=self.seed
        )

        # Shared args
        gt_dir = self.root_dir
        dataset_args = {
            "vlm_dir": vlm_dir,
            "gt_dir": gt_dir,
            "cache_dir": self.cache_dir,
            "force_rebuild": self.force_rebuild,
        }

        # Build all datasets (triggers cache build for each)
        self._train_dataset = GUIDataset(train_ids, **dataset_args)
        self._val_dataset = GUIDataset(val_ids, **dataset_args)
        self._test_dataset = GUIDataset(test_ids, **dataset_args)

        # Build cache for all three datasets
        _ = len(self._train_dataset)
        _ = len(self._val_dataset)
        _ = len(self._test_dataset)

        self._built = True

    # ------------------------------------------------------------------
    # Dataloader accessors
    # ------------------------------------------------------------------

    def train_dataloader(self) -> Optional[DataLoader]:
        """Get the training DataLoader (shuffled)."""
        if not self._built:
            self._ensure_built()
        if self._train_dataset is None or len(self._train_dataset) == 0:
            return None
        return create_dataloader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get the validation DataLoader (not shuffled)."""
        if not self._built:
            self._ensure_built()
        if self._val_dataset is None or len(self._val_dataset) == 0:
            return None
        return create_dataloader(
            self._val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test DataLoader (not shuffled)."""
        if not self._built:
            self._ensure_built()
        if self._test_dataset is None or len(self._test_dataset) == 0:
            return None
        return create_dataloader(
            self._test_dataset, batch_size=self.batch_size, shuffle=False
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> Optional[GUIDataset]:
        """The training :class:`GUIDataset` (triggers lazy build)."""
        if not self._built:
            self._ensure_built()
        return self._train_dataset

    @property
    def val_dataset(self) -> Optional[GUIDataset]:
        """The validation :class:`GUIDataset` (triggers lazy build)."""
        if not self._built:
            self._ensure_built()
        return self._val_dataset

    @property
    def test_dataset(self) -> Optional[GUIDataset]:
        """The testing :class:`GUIDataset` (triggers lazy build)."""
        if not self._built:
            self._ensure_built()
        return self._test_dataset
