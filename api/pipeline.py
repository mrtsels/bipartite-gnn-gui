"""Demo pipeline: VLM API call + GNN inference + bbox overlay rendering."""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — VLM
# ---------------------------------------------------------------------------

VLM_DEFAULT_MODEL = "qwen3-vl-flash"
VLM_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MAX_RETRIES = 3
VLM_RETRY_DELAY = 2.0
VLM_TIMEOUT = 60

VLM_SYSTEM_PROMPT = """You are a GUI element detector. Given a screenshot of a mobile app UI, identify all visible interactive and informative elements.

For each element, output a JSON object with these fields:
- "bbox_xyxy": [x1, y1, x2, y2] — the bounding box in **pixel coordinates** relative to the image dimensions. All values are integers.
- "label": one of: button, text, icon, image, input, container, checkbox, radio, switch, slider, tab, menu, divider, list, card, modal, toast, banner
- "text": the visible text content, or "" if none

Output a JSON array of elements. Example:
[
  {"bbox_xyxy": [10, 20, 100, 50], "label": "button", "text": "Submit"},
  {"bbox_xyxy": [10, 60, 300, 80], "label": "text", "text": "Welcome to the app"}
]

Rules:
1. Only include elements that are clearly visible
2. Use pixel coordinates (not normalized)
3. Be precise with bounding boxes — avoid overlap
4. Include ALL visible text, buttons, icons, inputs, images
5. Ignore background images and decorative elements
6. Output ONLY the JSON array, no other text"""

# ---------------------------------------------------------------------------
# Colours (RGBA tuples)
# ---------------------------------------------------------------------------

VLM_BBOX_COLOR = (255, 50, 50, 200)       # red
VLM_FILL_COLOR = (255, 50, 50, 40)        # red translucent
GNN_BBOX_COLOR = (50, 130, 255, 230)      # blue
GNN_FILL_COLOR = (50, 130, 255, 30)       # blue translucent
GNN_DASH = (6, 4)

# ---------------------------------------------------------------------------
# VLM API helpers
# ---------------------------------------------------------------------------


def _encode_image(img_bytes: bytes) -> str:
    """Encode image bytes to base64 data URL (JPEG)."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _guess_mime(img_bytes: bytes) -> str:
    """Guess MIME type from image bytes."""
    if img_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def _call_qwen_vl(
    image_b64: str,
    api_key: str,
    model: str = VLM_DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Call Qwen3-VL API with a single image.

    Returns dict with ``elements`` list and optional ``error``.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {
                        "type": "text",
                        "text": "Identify all GUI elements in this screenshot and return them as a JSON array.",
                    },
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    for attempt in range(1, VLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{VLM_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=VLM_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines if not line.startswith("```")
                )

            parsed = json.loads(content)
            if isinstance(parsed, list):
                return {"elements": parsed}
            if isinstance(parsed, dict) and "elements" in parsed:
                return {"elements": parsed["elements"]}
            return {"elements": parsed if isinstance(parsed, list) else [parsed]}

        except requests.exceptions.RequestException as e:
            logger.warning("VLM API error (attempt %d/%d): %s", attempt, VLM_MAX_RETRIES, e)
            if attempt < VLM_MAX_RETRIES:
                time.sleep(VLM_RETRY_DELAY * attempt)
            else:
                return {"error": str(e), "elements": []}

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("VLM parse error (attempt %d/%d): %s", attempt, VLM_MAX_RETRIES, e)
            if attempt < VLM_MAX_RETRIES:
                time.sleep(VLM_RETRY_DELAY * attempt)
            else:
                return {"error": str(e), "elements": []}

    return {"error": "Max retries exceeded", "elements": []}


# ---------------------------------------------------------------------------
# Element normalisation helpers
# ---------------------------------------------------------------------------


def _vlm_json_to_element_nodes(vlm_data: Dict[str, Any], img_w: int, img_h: int) -> List[ElementNode]:
    """Convert VLM JSON elements to ElementNode list with normalised bboxes."""
    raw_elements = vlm_data.get("elements", vlm_data.get("predictions", []))
    nodes: List[ElementNode] = []
    for i, item in enumerate(raw_elements):
        if not isinstance(item, dict):
            continue
        try:
            bbox_raw = item.get("bbox_xyxy") or item.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox_raw)
            if img_w > 0 and img_h > 0:
                x1 /= img_w
                y1 /= img_h
                x2 /= img_w
                y2 /= img_h
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(item.get("label", item.get("category", "unknown")))
            confidence = float(item.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))
            nodes.append(
                ElementNode(
                    bbox=[x1, y1, x2, y2],
                    label=label,
                    confidence=confidence,
                    element_id=str(i),
                )
            )
        except (ValueError, TypeError):
            continue
    return nodes


def _normalised_to_pixel(bbox_norm: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalised [x1,y1,x2,y2] to pixel coords."""
    x1, y1, x2, y2 = bbox_norm
    return (
        int(x1 * img_w),
        int(y1 * img_h),
        int(x2 * img_w),
        int(y2 * img_h),
    )


def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """Convert center-based xywh to xyxy."""
    cx, cy, w, h = bbox_xywh
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---------------------------------------------------------------------------
# Constraint type name mapping (for display)
# ---------------------------------------------------------------------------

CONSTRAINT_NAMES = {
    "align_left": "Align Left",
    "align_right": "Align Right",
    "align_top": "Align Top",
    "align_bottom": "Align Bottom",
    "center_x": "Center X",
    "center_y": "Center Y",
    "same_size": "Same Size",
    "spacing": "Spacing",
    "containment": "Containment",
    "grid": "Grid",
}

ELEMENT_TYPE_NAMES = [
    "button", "text", "image", "input", "icon", "container",
    "card", "checkbox", "radio", "slider", "switch", "label",
    "tab", "menu", "divider", "list", "modal", "toast", "banner", "other",
]


def _type_idx_to_name(idx: int) -> str:
    """Map type logit index to element type name."""
    if 0 <= idx < len(ELEMENT_TYPE_NAMES):
        return ELEMENT_TYPE_NAMES[idx]
    return "unknown"


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def _draw_bbox(draw: ImageDraw.Draw, bbox_px: Tuple[int, int, int, int],
               outline: Tuple[int, int, int, int],
               fill: Tuple[int, int, int, int],
               label: str = "",
               dashed: bool = False) -> None:
    """Draw a single bbox rectangle with optional label."""
    x1, y1, x2, y2 = bbox_px

    # Translucent fill
    draw.rectangle([x1, y1, x2, y2], fill=fill, outline=None)

    # Border
    if dashed:
        _draw_dashed_rect(draw, x1, y1, x2, y2, outline, width=2)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=None, outline=outline, width=2)

    # Label
    if label:
        # Semi-transparent label background
        bbox_h = y2 - y1
        label_font_size = max(10, min(13, int(bbox_h * 0.25)))
        # Simple white text — PIL's default font is fine for small labels
        draw.text((x1 + 3, max(y1 - 14, 0)), label, fill=(255, 255, 255, 255))


def _draw_dashed_rect(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
                      outline: Tuple[int, int, int, int], width: int = 2) -> None:
    """Draw a dashed rectangle using line segments."""
    dash_pattern = GNN_DASH
    # Top edge
    _dashed_line(draw, x1, y1, x2, y1, outline, dash_pattern, width)
    # Bottom edge
    _dashed_line(draw, x1, y2, x2, y2, outline, dash_pattern, width)
    # Left edge
    _dashed_line(draw, x1, y1, x1, y2, outline, dash_pattern, width)
    # Right edge
    _dashed_line(draw, x2, y1, x2, y2, outline, dash_pattern, width)


def _dashed_line(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int,
                 color: Tuple[int, int, int, int],
                 dash: Tuple[int, int], width: int) -> None:
    """Draw a dashed line segment."""
    dx, dy = x2 - x1, y2 - y1
    length = max(abs(dx), abs(dy))
    if length == 0:
        return
    step_x = dx / length
    step_y = dy / length
    on, off = dash
    pos = 0
    drawing = True
    while pos < length:
        end = min(pos + (on if drawing else off), length)
        if drawing:
            draw.line(
                [x1 + int(step_x * pos), y1 + int(step_y * pos),
                 x1 + int(step_x * end), y1 + int(step_y * end)],
                fill=color, width=width,
            )
        pos = end
        drawing = not drawing


# ---------------------------------------------------------------------------
# DemoPipeline
# ---------------------------------------------------------------------------


class DemoPipeline:
    """VLM + GNN demo pipeline: detect, analyse, render overlay.

    Args:
        checkpoint_path: Path to GNN model checkpoint.
        device: Torch device string (default ``"cpu"``).
        violation_threshold: Violation score threshold for proposal display.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        violation_threshold: float = 0.3,
    ) -> None:
        self.device = device
        self.violation_threshold = violation_threshold
        self._builder = BipartiteGraphBuilder()

        # Resolve checkpoint path
        if not checkpoint_path:
            # Auto-detect relative to this file
            here = Path(__file__).parent.parent
            default = here / "checkpoints" / "violation_detection_violation_only" / "best_model.pt"
            checkpoint_path = str(default)

        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch_load(checkpoint_path)
        self.model = BipartiteGNNCorrector(
            element_dim=5,
            constraint_dim=11,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
        )

        # Handle both raw state_dict and wrapped dict formats
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)

        self.model.to(device)
        self.model.eval()
        logger.info(
            "Model loaded: %s params, device=%s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_elements(self, img_bytes: bytes, api_key: str,
                        model: str = VLM_DEFAULT_MODEL) -> Dict[str, Any]:
        """Call VLM API to detect elements in a screenshot.

        Returns dict with ``elements`` (list of raw element dicts),
        ``time_ms``, and optional ``error``.
        """
        t0 = time.perf_counter()
        image_b64 = _encode_image(img_bytes)
        result = _call_qwen_vl(image_b64, api_key, model)
        elapsed = (time.perf_counter() - t0) * 1000

        result["time_ms"] = int(elapsed)
        if "error" in result and result["error"]:
            logger.error("VLM API error: %s", result["error"])
        else:
            logger.info("VLM detected %d elements in %d ms",
                        len(result.get("elements", [])), int(elapsed))
        return result

    def gnn_analyse(self, vlm_elements: List[Dict[str, Any]],
                    img_w: int = 0, img_h: int = 0) -> Dict[str, Any]:
        """Analyse VLM elements with GNN: extract constraints, detect violations, propose.

        Args:
            vlm_elements: Raw VLM element dicts.
            img_w: Original image pixel width (for bbox normalisation).
            img_h: Original image pixel height.

        Returns:
            Dict with ``constraints``, ``proposals``, ``existence_scores``,
            ``graph_stats``, and ``time_ms``.
        """
        t0 = time.perf_counter()

        # 1. Normalise elements
        vlm_data = {"elements": vlm_elements}
        element_nodes = _vlm_json_to_element_nodes(vlm_data, img_w, img_h)

        if not element_nodes:
            return {
                "constraints": [],
                "proposals": [],
                "existence_scores": [],
                "graph_stats": {"elements": 0, "constraints": 0},
                "time_ms": 0,
            }

        # 2. Extract constraints
        constraints = extract_all_constraints(element_nodes)

        # 3. Build graph
        graph = self._builder.build(element_nodes, constraints)
        graph.to(self.device)

        # 4. Model forward
        import torch
        with torch.no_grad():
            outputs = self.model(graph)

        existence = outputs.get("existence")  # (N_elem, 1)
        viol = outputs.get("violation")       # (N_con, 1)
        proposal = outputs.get("proposal")    # (N_con, 4) xywh
        proposal_type = outputs.get("proposal_type")  # (N_con, N_TYPES)

        # 5. Constraint info
        constraint_list = []
        for i, c in enumerate(constraints):
            score = float(viol[i].item()) if viol is not None and i < viol.shape[0] else 0.0
            constraint_list.append({
                "index": i,
                "type": c.constraint_type.value,
                "type_label": CONSTRAINT_NAMES.get(c.constraint_type.value, c.constraint_type.value),
                "violation_score": round(score, 4),
                "is_violated": score > self.violation_threshold,
                "element_indices": sorted(set(c.source_indices + c.target_indices)),
                "num_elements": len(set(c.source_indices + c.target_indices)),
            })

        # 6. Proposals (from violated constraints)
        proposals_list = []
        if proposal is not None and viol is not None:
            for i in range(proposal.shape[0]):
                score = float(viol[i].item())
                if score <= self.violation_threshold:
                    continue
                bbox_xywh = proposal[i].tolist()
                bbox_xyxy = _xywh_to_xyxy(bbox_xywh)
                # Clamp to [0, 1]
                bbox_xyxy = [max(0.0, min(1.0, v)) for v in bbox_xyxy]
                pred_type_idx = int(proposal_type[i].argmax().item()) if proposal_type is not None else 0
                proposals_list.append({
                    "bbox": bbox_xyxy,
                    "constraint_index": i,
                    "constraint_type": constraint_list[i]["type"] if i < len(constraint_list) else "",
                    "violation_score": round(score, 4),
                    "predicted_type": _type_idx_to_name(pred_type_idx),
                })

        # 7. Existence scores
        existence_scores = []
        if existence is not None:
            existence_scores = [round(float(existence[i].item()), 4) for i in range(existence.shape[0])]

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "constraints": constraint_list,
            "proposals": proposals_list,
            "existence_scores": existence_scores,
            "graph_stats": {
                "elements": len(element_nodes),
                "constraints": len(constraints),
                "num_violated": sum(1 for c in constraint_list if c["is_violated"]),
                "num_proposals": len(proposals_list),
            },
            "time_ms": round(elapsed, 1),
        }

    def build_corrected_json(
        self,
        vlm_elements: List[Dict[str, Any]],
        gnn_result: Dict[str, Any],
        img_w: int = 0,
        img_h: int = 0,
    ) -> Dict[str, Any]:
        """Merge VLM elements with GNN analysis into a corrected element list.

        Args:
            vlm_elements: Raw VLM-detected elements.
            gnn_result: Output from gnn_analyse().
            img_w: Original image pixel width (for bbox denormalisation).
            img_h: Original image pixel height.

        Returns:
            Dict with ``elements`` (the merged list), ``stats``.
        """
        corrected = []

        # Original VLM elements annotated with GNN existence scores
        existence_scores = gnn_result.get("existence_scores", [])
        for i, elem in enumerate(vlm_elements):
            cleaned = {
                "bbox": elem.get("bbox_xyxy") or elem.get("bbox", []),
                "label": elem.get("label", elem.get("category", "unknown")),
                "text": elem.get("text", ""),
                "confidence": elem.get("confidence", 0.0),
                "source": "vlm",
                "existence_score": existence_scores[i] if i < len(existence_scores) else None,
            }
            corrected.append(cleaned)

        # GNN proposals — suppressed in corrected JSON (research artifacts, not useful
        # for real VLM output where no elements are artificially masked). Existence
        # scores on VLM elements above are the valuable GNN signal for the demo.

        return {
            "elements": corrected,
            "total_count": len(corrected),
            "vlm_count": len(vlm_elements),
            "gnn_proposals_count": len(gnn_result.get("proposals", [])),
        }

    def render_overlay(self, img_bytes: bytes, vlm_elements: List[Dict[str, Any]],
                       gnn_proposals: List[Dict[str, Any]]) -> bytes:
        """Render screenshot with VLM bboxes (red) and GNN proposals (blue dashed).

        Returns PNG bytes.
        """
        # Open image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        img_w, img_h = img.size

        # Create overlay layer for translucent shapes
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw VLM bboxes (red)
        for elem in vlm_elements:
            bbox_raw = elem.get("bbox_xyxy") or elem.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                continue
            bbox_px = _normalised_to_pixel(
                [float(v) for v in bbox_raw], img_w, img_h
            ) if max(bbox_raw) <= 1.0 else (
                int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3])
            )
            label = str(elem.get("label", elem.get("category", "")))
            confidence = elem.get("confidence", 1.0)
            label_text = f"{label} {confidence:.2f}" if confidence else label
            _draw_bbox(overlay_draw, bbox_px, VLM_BBOX_COLOR, VLM_FILL_COLOR, label_text)

        # Draw GNN proposals (blue dashed)
        for prop in gnn_proposals:
            bbox_norm = prop.get("bbox")
            if not bbox_norm or len(bbox_norm) != 4:
                continue
            bbox_px = _normalised_to_pixel(bbox_norm, img_w, img_h)
            label_text = f"⚡ {prop.get('predicted_type', '')} ({prop.get('violation_score', 0):.2f})"
            _draw_bbox(overlay_draw, bbox_px, GNN_BBOX_COLOR, GNN_FILL_COLOR, label_text, dashed=True)

        # Composite
        combined = Image.alpha_composite(img, overlay)
        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        return buf.getvalue()

    def health(self) -> Dict[str, Any]:
        """Return health info."""
        return {
            "status": "ok",
            "params": sum(p.numel() for p in self.model.parameters()),
            "device": self.device,
            "model": type(self.model).__name__,
            "hidden_dim": self.model.hidden_dim,
            "violation_threshold": self.violation_threshold,
        }


def torch_load(path: str) -> Any:
    """Load a PyTorch checkpoint safely.

    ``weights_only=True`` by default for security; falls back to
    ``weights_only=False`` if the checkpoint contains non-tensor types.
    """
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)
