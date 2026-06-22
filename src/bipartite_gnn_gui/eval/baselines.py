"""Heuristic baselines for comparison against the GNN corrector.

Each baseline implements the same ``correct_single`` / ``correct_batch``
interface as ``InferencePipeline``, so they can be plugged directly into
the ``Evaluator``.

Baselines
---------
- **NoOpBaseline** — Return VLM predictions as-is (no correction).
  Measures: how bad are the original VLM errors?
- **IdentityBaseline** — Return ground-truth positions (oracle upper bound).
  Measures: best possible performance.
- **RandomJitterBaseline** — Add uniform noise to each VLM bbox.
  Measures: does the GNN improve over random perturbation?
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_elements(vlm_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract elements from a raw VLM JSON dict, normalising bboxes.

    Matches the output format of ``InferencePipeline``:
    ``{"element_id", "bbox", "label", "confidence", "existence_score"}``.

    Args:
        vlm_json: Raw VLM output dictionary.

    Returns:
        List of normalised element dicts.
    """
    img_width = int(vlm_json.get("image_width", 0))
    img_height = int(vlm_json.get("image_height", 0))
    elements_raw = vlm_json.get("elements", [])

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(elements_raw):
        if not isinstance(item, dict):
            continue

        bbox_raw = item.get("bbox")
        if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox_raw)
        if img_width > 0:
            x1 = max(0.0, min(1.0, x1 / img_width))
            x2 = max(0.0, min(1.0, x2 / img_width))
        if img_height > 0:
            y1 = max(0.0, min(1.0, y1 / img_height))
            y2 = max(0.0, min(1.0, y2 / img_height))
        bbox = [max(0.0, min(1.0, v)) for v in [x1, y1, x2, y2]]

        label = str(item.get("label", "unknown"))
        confidence = float(item.get("confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))

        results.append({
            "element_id": i,
            "bbox": bbox,
            "label": label,
            "confidence": confidence,
            "existence_score": 1.0,
        })

    return results


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class NoOpBaseline:
    """Return VLM predictions unchanged (no correction).

    Used to measure how bad the original VLM errors are.

    Args:
        config: Configuration dict.
        device: Torch device.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def correct_single(self, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
        """Return the VLM predictions as-is in the standard output format.

        Args:
            vlm_json: Raw VLM output dictionary.

        Returns:
            Dict with ``"image_id"``, ``"elements"``, and ``"model"`` keys.
        """
        elements = _extract_elements(vlm_json)
        return {
            "image_id": vlm_json.get("image_id", ""),
            "elements": elements,
            "model": "noop",
        }

    def correct_batch(
        self, vlm_jsons: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correct multiple VLM outputs.

        Args:
            vlm_jsons: List of raw VLM output dictionaries.

        Returns:
            List of corrected JSON dicts.
        """
        return [self.correct_single(v) for v in vlm_jsons]


class IdentityBaseline:
    """Return ground-truth positions (oracle upper bound).

    Requires a ground-truth lookup dict that maps ``image_id`` to a list
    of element dicts (each with at least ``"bbox"`` and optionally
    ``"label"`` / ``"element_type"``).

    Used to measure best possible performance.

    Args:
        config: Configuration dict.
        device: Torch device.
        gt_lookup: Optional mapping from ``image_id`` to GT element list.
            If ``None`` (or an image_id is not found), falls back to the
            VLM input elements (acts like NoOp).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        gt_lookup: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.gt_lookup: Dict[str, List[Dict[str, Any]]] = gt_lookup or {}

    def set_gt_lookup(self, gt_lookup: Dict[str, List[Dict[str, Any]]]) -> None:
        """Set or replace the ground-truth lookup table."""
        self.gt_lookup = gt_lookup

    def correct_single(self, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
        """Return ground-truth positions for the given image.

        Args:
            vlm_json: Raw VLM output dictionary.

        Returns:
            Dict with ``"image_id"``, ``"elements"``, and ``"model"`` keys.
        """
        image_id = vlm_json.get("image_id", "")

        if image_id in self.gt_lookup:
            gt_elements = self.gt_lookup[image_id]
            elements: List[Dict[str, Any]] = []
            for i, gt_elem in enumerate(gt_elements):
                bbox_raw = gt_elem.get("bbox", [0, 0, 0, 0])
                bbox = [max(0.0, min(1.0, float(v))) for v in bbox_raw]
                # Accept both "label" (model output) and "element_type" (GT)
                label = str(gt_elem.get("label", gt_elem.get("element_type", "unknown")))
                confidence = float(gt_elem.get("confidence", 1.0))
                elements.append({
                    "element_id": i,
                    "bbox": bbox,
                    "label": label,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "existence_score": 1.0,
                })
        else:
            # Fallback: return VLM input as-is
            elements = _extract_elements(vlm_json)

        return {
            "image_id": image_id,
            "elements": elements,
            "model": "identity",
        }

    def correct_batch(
        self, vlm_jsons: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correct multiple VLM outputs.

        Args:
            vlm_jsons: List of raw VLM output dictionaries.

        Returns:
            List of corrected JSON dicts.
        """
        return [self.correct_single(v) for v in vlm_jsons]


class RandomJitterBaseline:
    """Add random noise to each VLM prediction's bbox coordinates.

    Noise is drawn from ``Uniform(-0.05, 0.05)`` in normalized
    coordinates.  The random seed is set at construction time so that
    results are reproducible.

    Used to measure whether the GNN improves over random perturbation.

    Args:
        config: Configuration dict (may contain ``"seed"``).
        device: Torch device.
        seed: Random seed.  If not provided, extracted from
            ``config["training"]["seed"]`` (default 42).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        if seed is None:
            seed = 42
            if isinstance(config, dict):
                seed = config.get("training", {}).get("seed", 42)
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    def correct_single(self, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
        """Add random jitter to each VLM element bbox.

        Args:
            vlm_json: Raw VLM output dictionary.

        Returns:
            Dict with ``"image_id"``, ``"elements"``, and ``"model"`` keys.
        """
        elements = _extract_elements(vlm_json)

        for elem in elements:
            # Uniform(-0.05, 0.05): rand [0,1) → scale to [0,0.1) → shift to [-0.05, 0.05)
            noise = (torch.rand(4, generator=self._rng) * 0.1) - 0.05
            bbox = elem["bbox"]
            for j in range(4):
                bbox[j] = max(0.0, min(1.0, bbox[j] + noise[j].item()))

        return {
            "image_id": vlm_json.get("image_id", ""),
            "elements": elements,
            "model": "random_jitter",
        }

    def correct_batch(
        self, vlm_jsons: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correct multiple VLM outputs.

        Args:
            vlm_jsons: List of raw VLM output dictionaries.

        Returns:
            List of corrected JSON dicts.
        """
        return [self.correct_single(v) for v in vlm_jsons]
