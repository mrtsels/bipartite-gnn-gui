"""Coordinate and feature preprocessing helpers.

Provides:
- CoordinateNormalizer: Z-score normalization with fit/transform pattern.
- extract_spatial_features: xyxy -> (cx, cy, w, h) conversion.
- extract_type_embedding: one-hot encoding for element type labels.
- extract_confidence_scores: confidence extraction from element lists.
- train_val_test_split: deterministic dataset splitting.
- normalize_coordinates: simple [0,1] normalization for absolute coords.
- extract_element_features: legacy dict-based feature extraction.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from bipartite_gnn_gui.data.ground_truth import GTElement
from bipartite_gnn_gui.data.vlm_output import ELEMENT_TYPES, VLMOutputElement


# ---------------------------------------------------------------------------
# CoordinateNormalizer -- Z-score normalization (fit/transform pattern)
# ---------------------------------------------------------------------------


class CoordinateNormalizer:
    """Stateful normalizer with fit/transform pattern for Z-score normalization.

    Computes per-coordinate mean and standard deviation across a dataset,
    then applies Z-score normalization: ``(x - mean) / (std + eps)``.

    Args:
        bbox_format: Bounding box format, either ``"xyxy"`` or ``"cxcywh"``.
            Only used for input validation; the normalizer itself treats all
            four coordinates independently regardless of format.

    Raises:
        ValueError: If *bbox_format* is not ``"xyxy"`` or ``"cxcywh"``.

    Example:
        >>> normalizer = CoordinateNormalizer()
        >>> bboxes = torch.tensor([[10.0, 20.0, 100.0, 80.0],
        ...                        [30.0, 40.0, 120.0, 100.0]])
        >>> normalizer.fit(bboxes)
        >>> normed = normalizer.transform(bboxes)
        >>> restored = normalizer.inverse_transform(normed)
        >>> torch.allclose(restored, bboxes)
        True
    """

    eps: float = 1e-8

    def __init__(self, bbox_format: str = "xyxy") -> None:
        if bbox_format not in ("xyxy", "cxcywh"):
            raise ValueError(
                f"bbox_format must be 'xyxy' or 'cxcywh', got '{bbox_format}'"
            )
        self._bbox_format = bbox_format
        self._mean: Optional[Tensor] = None
        self._std: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> Tensor:
        """Per-coordinate means from the fitted dataset.

        Returns:
            Float32 tensor of shape ``(4,)``.

        Raises:
            RuntimeError: If the normalizer has not been fitted yet.
        """
        if self._mean is None:
            raise RuntimeError(
                "CoordinateNormalizer has not been fitted yet. "
                "Call fit() or fit_from_elements() first."
            )
        return self._mean

    @property
    def std(self) -> Tensor:
        """Per-coordinate standard deviations from the fitted dataset.

        Returns:
            Float32 tensor of shape ``(4,)``.

        Raises:
            RuntimeError: If the normalizer has not been fitted yet.
        """
        if self._std is None:
            raise RuntimeError(
                "CoordinateNormalizer has not been fitted yet. "
                "Call fit() or fit_from_elements() first."
            )
        return self._std

    @property
    def fitted(self) -> bool:
        """Whether the normalizer has been fitted (i.e. statistics computed)."""
        return self._mean is not None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, bboxes: Tensor) -> CoordinateNormalizer:
        """Compute per-coordinate mean and standard deviation.

        Args:
            bboxes: ``(N, 4)`` tensor of bounding boxes.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If *bboxes* is not a 2-D tensor with 4 columns.
        """
        if bboxes.dim() != 2 or bboxes.size(1) != 4:
            raise ValueError(
                f"Expected (N, 4) tensor, got {tuple(bboxes.shape)}"
            )
        self._mean = bboxes.mean(dim=0)
        self._std = bboxes.std(dim=0, unbiased=False)
        return self

    def fit_from_elements(
        self,
        elements: Sequence[Union[GTElement, VLMOutputElement]],
    ) -> CoordinateNormalizer:
        """Extract bounding boxes from element instances and fit.

        Each element's ``.bbox`` attribute is expected to be a 4-tuple
        of floats in ``(x1, y1, x2, y2)`` format.

        Args:
            elements: Sequence of ``GTElement`` or ``VLMOutputElement``.

        Returns:
            Self, for chaining.
        """
        bboxes = torch.tensor(
            [list(elem.bbox) for elem in elements], dtype=torch.float32
        )
        return self.fit(bboxes)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, bboxes: Tensor) -> Tensor:
        """Apply Z-score normalization.

        ``(x - mean) / (std + eps)``

        Args:
            bboxes: ``(..., 4)`` tensor of bounding boxes.

        Returns:
            Normalized tensor with the same shape and dtype.
        """
        return (bboxes - self.mean) / (self.std + self.eps)

    def inverse_transform(self, norm_bboxes: Tensor) -> Tensor:
        """Reverse Z-score normalization.

        ``x * (std + eps) + mean``

        Args:
            norm_bboxes: ``(..., 4)`` tensor of normalized bounding boxes.

        Returns:
            Tensor in the original coordinate space.
        """
        return norm_bboxes * (self.std + self.eps) + self.mean


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------


def extract_spatial_features(bbox_xyxy: Tensor) -> Tensor:
    """Convert xyxy bounding boxes to (cx, cy, w, h) spatial features.

    For each box:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

    Args:
        bbox_xyxy: ``(..., 4)`` tensor of xyxy bounding boxes.

    Returns:
        ``(..., 4)`` tensor of (cx, cy, w, h) values with the same dtype.
    """
    x1, y1, x2, y2 = bbox_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


# ---------------------------------------------------------------------------
# Type embedding
# ---------------------------------------------------------------------------


def extract_type_embedding(
    label: str,
    taxonomy: Optional[list[str]] = None,
) -> Tensor:
    """One-hot encode an element type label.

    Args:
        label: Element type string (matched case-insensitively).
        taxonomy: Ordered list of canonical type names. The first entry
            (index 0) serves as the catch-all for unrecognized labels.
            Defaults to the 20-type taxonomy from ``vlm_output.ELEMENT_TYPES``.

    Returns:
        Float32 tensor of shape ``(len(taxonomy),)`` with a single 1.0
        at the matching index, or index 0 if the label is unrecognized.
    """
    if taxonomy is None:
        taxonomy = list(ELEMENT_TYPES.keys())

    if not taxonomy:
        return torch.zeros(0, dtype=torch.float32)

    # Build a case-insensitive lookup for this call
    label_lower = label.strip().lower()
    idx = 0  # default: unrecognized -> index 0
    for i, t in enumerate(taxonomy):
        if t.lower() == label_lower:
            idx = i
            break

    emb = torch.zeros(len(taxonomy), dtype=torch.float32)
    emb[idx] = 1.0
    return emb


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------


def extract_confidence_scores(
    elements: Sequence[Union[GTElement, VLMOutputElement]],
) -> Tensor:
    """Extract confidence scores from a sequence of elements.

    ``VLMOutputElement`` has a ``.confidence`` attribute directly.
    ``GTElement`` does not carry a confidence score, so it defaults to 1.0.

    Args:
        elements: Sequence of ``GTElement`` or ``VLMOutputElement``.

    Returns:
        Float32 tensor of shape ``(N,)``, where N is the number of elements.
    """
    scores = [
        float(elem.confidence) if hasattr(elem, "confidence") else 1.0
        for elem in elements
    ]
    return torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------


def train_val_test_split(
    elements: list[Any],
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[list[Any], list[Any], list[Any]]:
    """Deterministically shuffle and split a list into train/val/test.

    The shuffle uses Python's ``random.Random`` with a fixed seed for
    reproducibility independent of external RNG state.

    Args:
        elements: List of items to split.
        val_split: Fraction of items for validation (default 0.1).
        test_split: Fraction of items for testing (default 0.1).
        seed: Random seed for deterministic shuffling (default 42).

    Returns:
        Tuple of ``(train, val, test)`` sublists.

    Raises:
        ValueError: If ``val_split + test_split >= 1.0``.
    """
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split + test_split must be < 1.0, "
            f"got {val_split + test_split}"
        )

    n = len(elements)
    indices = list(range(n))

    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = round(n * test_split)
    n_val = round(n * val_split)
    n_train = n - n_test - n_val

    train = [elements[i] for i in indices[:n_train]]
    val = [elements[i] for i in indices[n_train : n_train + n_val]]
    test = [elements[i] for i in indices[n_train + n_val :]]

    return train, val, test


# ---------------------------------------------------------------------------
# Legacy helpers  (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def normalize_coordinates(
    box: Sequence[float],
    width: float,
    height: float,
    fmt: str = "xyxy",
) -> list[float]:
    """Normalize absolute pixel coordinates to the [0, 1] range.

    Each coordinate is divided by the corresponding image dimension.

    Args:
        box: Four-element bounding box.
        width: Image width in pixels.
        height: Image height in pixels.
        fmt: Bounding box format -- ``"xyxy"`` (default) or ``"xywh"``.

    Returns:
        Normalized box preserving the input format:
        ``[x1/w, y1/h, x2/w, y2/h]`` for xyxy,
        ``[x/w, y/h, w/w, h/h]`` for xywh.

    Raises:
        ZeroDivisionError: If *width* or *height* is zero.
    """
    if fmt == "xywh":
        x, y, w, h = box
        return [x / width, y / height, w / width, h / height]

    # Default xyxy
    x1, y1, x2, y2 = box
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def extract_element_features(element: dict[str, Any]) -> Tensor:
    """Convert a GUI element payload into a 5-d feature tensor.

    Extracts the first 4 bbox values and the confidence score.

    Args:
        element: Dict with keys ``"bbox"`` and optionally ``"confidence"``.

    Returns:
        Float32 tensor of shape ``(5,)``: ``[b0, b1, b2, b3, conf]``.
    """
    bbox = element.get("bbox", [0.0, 0.0, 0.0, 0.0])
    confidence = float(element.get("confidence", 1.0))
    values = list(bbox)[:4] + [confidence]
    return torch.tensor([float(v) for v in values], dtype=torch.float32)
