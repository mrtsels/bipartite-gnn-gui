"""End-to-end inference pipeline for GUI layout correction.

Takes raw VLM JSON and returns corrected element coordinates by:
    1. Parsing VLM output into element nodes.
    2. Extracting heuristic constraints.
    3. Building a bipartite HeteroData graph.
    4. Running the GNN model to predict coordinate deltas.
    5. Applying deltas and clamping results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from bipartite_gnn_gui.data.vlm_output import (
    VLMOutput,
    VLMOutputElement,
    normalize_bbox,
)
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode

from .model import BipartiteGNNCorrector

logger = logging.getLogger(__name__)


def _parse_elements_from_dict(vlm_json: Dict[str, Any]) -> VLMOutput:
    """Parse a raw VLM JSON dict into a ``VLMOutput``.

    Args:
        vlm_json: Raw dict with an ``"elements"`` key containing a list
            of element dicts. Each element dict should have ``"bbox"``,
            ``"label"``, and optionally ``"confidence"``.

    Returns:
        ``VLMOutput`` with normalized elements.
    """
    image_id = str(vlm_json.get("image_id", ""))
    img_width = int(vlm_json.get("image_width", 0))
    img_height = int(vlm_json.get("image_height", 0))
    elements_raw = vlm_json.get("elements", [])

    elements: List[VLMOutputElement] = []
    for i, item in enumerate(elements_raw):
        if not isinstance(item, dict):
            continue
        try:
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            bbox = normalize_bbox(
                list(bbox_raw),
                format="xyxy",
                img_width=img_width,
                img_height=img_height,
            )
            label = str(item.get("label", "unknown"))
            confidence = float(item.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))

            elem = VLMOutputElement(
                element_id=i,
                bbox=bbox,
                element_type=label,
                confidence=confidence,
            )
            elements.append(elem)
        except Exception as exc:
            logger.warning("Skipping invalid element %d: %s", i, exc)

    return VLMOutput(
        image_id=image_id,
        elements=elements,
        image_width=img_width,
        image_height=img_height,
    )


def _vlm_element_to_node(elem: VLMOutputElement) -> ElementNode:
    """Convert a ``VLMOutputElement`` to a graph ``ElementNode``.

    The bbox is expected to be in normalized ``(x1, y1, x2, y2)`` format.
    """
    return ElementNode(
        bbox=list(elem.bbox),
        label=elem.element_type,
        confidence=elem.confidence,
        element_id=str(elem.element_id),
    )


def _xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    """Convert xyxy bbox to xywh (center x, center y, width, height)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """Convert xywh bbox to xyxy."""
    cx, cy, w, h = bbox_xywh
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]


class InferencePipeline:
    """End-to-end inference pipeline for GUI layout correction.

    Takes raw VLM JSON predictions, builds a bipartite graph, runs the
    GNN corrector, applies predicted deltas, and returns corrected
    element coordinates.

    Args:
        model: Trained ``BipartiteGNNCorrector`` instance.
        device: Target device (auto-detected if ``None``).
        amp: Whether to use automatic mixed precision during inference.
        delta_clamp: Maximum absolute value for coordinate deltas.
    """

    def __init__(
        self,
        model: BipartiteGNNCorrector,
        device: torch.device | None = None,
        amp: bool = False,
        delta_clamp: float = 0.5,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.amp = amp
        self.delta_clamp = delta_clamp
        self.model = model.to(device)
        self.model.eval()
        self._builder = BipartiteGraphBuilder()

    def correct_single(
        self,
        vlm_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Correct a single VLM output.

        Full pipeline:
            1. Parse VLM JSON → element nodes.
            2. Extract constraints heuristically.
            3. Build ``HeteroData`` graph.
            4. Run model inference.
            5. Apply coordinate deltas to original bboxes.
            6. Clamp corrected bboxes to ``[0, 1]``.
            7. Filter low-confidence elements by existence score.

        Args:
            vlm_json: Raw VLM output dictionary with an ``"elements"`` key.

        Returns:
            Corrected JSON dict with refined ``"elements"`` list.
        """
        # Step 1: Parse VLM output.
        vlm_output = _parse_elements_from_dict(vlm_json)
        elements = vlm_output.elements

        if not elements:
            return {"image_id": vlm_output.image_id, "elements": []}

        # Convert to graph ElementNodes (xyxy bboxes).
        element_nodes = [_vlm_element_to_node(e) for e in elements]

        # Step 2: Extract constraints.
        constraints = extract_all_constraints(element_nodes)

        # Step 3: Build HeteroData graph.
        hetero_data = self._builder.build(element_nodes, constraints)
        hetero_data = self._to_device(hetero_data)

        # Step 4: Model inference.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp):  # type: ignore[attr-defined]
                outputs = self.model(hetero_data)

        # Step 5: Apply coordinate deltas.
        coord_deltas = outputs.get("coord")
        existence_scores = outputs.get("existence")

        corrected_elements = self._apply_deltas(
            element_nodes, coord_deltas, existence_scores
        )

        # Step 6: Serialize to JSON-compatible dict.
        return {
            "image_id": vlm_output.image_id,
            "elements": corrected_elements,
        }

    def correct_batch(
        self,
        vlm_jsons: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Correct multiple VLM outputs.

        Args:
            vlm_jsons: List of raw VLM output dictionaries.

        Returns:
            List of corrected JSON dicts, one per input.
        """
        return [self.correct_single(vlm_json) for vlm_json in vlm_jsons]

    def _apply_deltas(
        self,
        element_nodes: List[ElementNode],
        coord_deltas: Tensor | None,
        existence_scores: Tensor | None,
    ) -> List[Dict[str, Any]]:
        """Apply predicted deltas to original bboxes, clamp, and filter.

        Args:
            element_nodes: Original graph element nodes (xyxy bboxes).
            coord_deltas: ``(N, 4)`` tensor of predicted deltas in xywh
                format, or ``None``.
            existence_scores: ``(N, 1)`` tensor of existence probabilities,
                or ``None``.

        Returns:
            List of corrected element dicts.
        """
        results: List[Dict[str, Any]] = []

        for i, elem in enumerate(element_nodes):
            # Get original bbox in xywh format for delta application.
            original_xywh = _xyxy_to_xywh(elem.bbox)
            bbox_xywh = list(original_xywh)

            if coord_deltas is not None and i < coord_deltas.shape[0]:
                delta = coord_deltas[i].detach().cpu()  # (4,)
                # Clamp deltas to [-delta_clamp, delta_clamp].
                delta = torch.clamp(delta, -self.delta_clamp, self.delta_clamp)
                # Apply delta: corrected = original + delta.
                bbox_xywh[0] += delta[0].item()
                bbox_xywh[1] += delta[1].item()
                bbox_xywh[2] += delta[2].item()
                bbox_xywh[3] += delta[3].item()

            # Convert back to xyxy for output.
            bbox_xyxy = _xywh_to_xyxy(bbox_xywh)

            # Clamp to [0, 1].
            bbox_xyxy = [
                max(0.0, min(1.0, v)) for v in bbox_xyxy
            ]

            # Check existence score — skip elements below threshold.
            existence = 1.0
            if existence_scores is not None and i < existence_scores.shape[0]:
                existence = existence_scores[i].detach().cpu().item()
            if existence < 0.5:
                continue

            results.append({
                "element_id": i,
                "bbox": bbox_xyxy,
                "label": elem.label,
                "confidence": elem.confidence,
                "existence_score": existence,
            })

        return results

    def _to_device(self, hetero_data: Any) -> Any:
        """Move HeteroData tensors to the target device.

        Args:
            hetero_data: A ``HeteroData`` object from PyG.

        Returns:
            The same data object with all tensors on ``self.device``.
        """
        for store in hetero_data.node_stores:
            for key, value in list(store.items()):
                if isinstance(value, torch.Tensor):
                    store[key] = value.to(self.device)
        for store in hetero_data.edge_stores:
            for key, value in list(store.items()):
                if isinstance(value, torch.Tensor):
                    store[key] = value.to(self.device)
        return hetero_data


# Backward-compatible function.
def correct_layout(model: Any, data: Any) -> Any:
    """Run inference and return the model outputs.

    Backward-compatible wrapper around ``model.forward()``.

    Args:
        model: A ``BipartiteGNNCorrector`` (or compatible).
        data: A ``HeteroData`` graph (or dict).

    Returns:
        The model's forward output dict.
    """
    return model(data)
