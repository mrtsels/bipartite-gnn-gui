"""FastAPI web demo — upload screenshot, detect with VLM, correct with GNN."""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# HEIC/HEIF support is optional — the app works without it (JPEG/PNG only).
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    logging.getLogger(__name__).warning(
        "pillow_heif not installed — HEIC/HEIF uploads unsupported (JPEG/PNG fine)."
    )

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

from pipeline import DemoPipeline, VLM_DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

app = FastAPI(title="GUI-GNN Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-init pipeline (avoids import-time GPU init)
_pipeline: Optional[DemoPipeline] = None

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "web")
DEMO_DATA = Path(__file__).resolve().parent.parent / "demo_data"
CASES_PATH = DEMO_DATA / "cases.json"


def _load_cases() -> List[Dict[str, Any]]:
    """Load pre-computed hero cases from ``demo_data/cases.json``.

    Returns an empty list (never raises) if the file is missing or malformed —
    the server should start and serve hero-case APIs with empty results.
    """
    if not CASES_PATH.exists():
        logger.warning(
            "cases.json not found — hero cases unavailable. "
            "Run scripts/prepare_demo_cases.py first."
        )
        return []
    try:
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("cases", [])
        return data if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001 — never crash on bad data
        logger.error("Failed to load cases.json: %s", e)
        return []


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the frontend single-page app (no-cache: dev iteration)."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(
                f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
    return HTMLResponse("<h1>GUI-GNN Demo</h1><p>Frontend not found.</p>", status_code=404)


def get_pipeline() -> DemoPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising DemoPipeline...")
        _pipeline = DemoPipeline(
            device="cpu",
            violation_threshold=0.60,
        )
        logger.info("Pipeline ready: %s", _pipeline.health())
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    p = get_pipeline()
    return JSONResponse({
        "status": "ok",
        "model": {
            "name": type(p.model).__name__,
            "params": sum(p.numel() for p in p.model.parameters()),
            "hidden_dim": p.model.hidden_dim,
        },
        "device": p.device,
        "violation_threshold": p.violation_threshold,
    })


# ---------------------------------------------------------------------------
# Capability-validation demo data (Tab 2: confidence scoring, Tab 3: completion)
# ---------------------------------------------------------------------------

CONFIDENCE_DIR = DEMO_DATA / "confidence"
COMPLETION_DIR = DEMO_DATA / "completion"


@app.get("/api/demo/confidence")
async def demo_confidence() -> JSONResponse:
    """Return pre-computed confidence-scoring demo data (synthetic imposters).

    Each item: {id, auroc, threshold, n_elements, n_real, n_imposter, elements}
    where ``elements`` entries carry {bbox, label, is_imposter, score}.
    """
    if not CONFIDENCE_DIR.is_dir():
        return JSONResponse({"error": "confidence demo data missing — "
                                      "run scripts/prepare_confidence_demo.py"}, status_code=404)
    out = []
    summary_path = CONFIDENCE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for f in sorted(CONFIDENCE_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        out.append({
            "id": d.get("id"),
            "auroc": d.get("auroc"),
            "threshold": d.get("threshold"),
            "n_elements": d.get("n_elements"),
            "n_real": d.get("n_real"),
            "n_imposter": d.get("n_imposter"),
            "imposter_ratio": d.get("imposter_ratio"),
            "elements": d.get("elements", []),
        })
    return JSONResponse({"summary": summary, "images": out})


@app.get("/api/demo/confidence/{img_id}")
async def demo_confidence_image(img_id: str):
    """Serve a confidence-demo overlay PNG (blue=real, red=imposter)."""
    path = CONFIDENCE_DIR / f"{img_id}.png"
    if not path.is_file():
        return JSONResponse({"error": f"confidence image {img_id} not found"}, status_code=404)
    return FileResponse(str(path))


@app.get("/api/demo/completion")
async def demo_completion() -> JSONResponse:
    """Return the structural-completion evaluation curve (GNN vs NN IoU per drop ratio)."""
    path = COMPLETION_DIR / "curve.json"
    if not path.is_file():
        return JSONResponse({"error": "completion curve missing — "
                                      "run scripts/prepare_completion_demo.py"}, status_code=404)
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/cases")
async def list_cases() -> JSONResponse:
    """Return summary list of all pre-computed hero cases.

    Each item is ``{"id", "name", "metrics"}`` — no bbox data, so the
    payload stays small.  Returns ``[]`` (not an error) when cases.json
    has not been generated yet.
    """
    cases = _load_cases()
    summary = [
        {
            "id": c.get("id"),
            "name": c.get("name") or f"Case {c.get('id')}",
            "metrics": c.get("metrics", {}),
        }
        for c in cases
    ]
    return JSONResponse(summary)


@app.get("/api/case/{case_id}")
async def get_case(case_id: str) -> JSONResponse:
    """Return the full pre-computed case (vlm_elements, proposals, metrics, img_w, img_h)."""
    cases = _load_cases()
    case = next((c for c in cases if str(c.get("id")) == case_id), None)
    if case is None:
        return JSONResponse({"error": f"Case {case_id} not found"}, status_code=404)
    return JSONResponse(case)


@app.get("/api/screenshot/{case_id}")
async def get_screenshot(case_id: str):
    """Serve a hero case's screenshot image from ``demo_data/screenshots/``."""
    cases = _load_cases()
    case = next((c for c in cases if str(c.get("id")) == case_id), None)
    candidates: List[str] = []
    if case and case.get("screenshot"):
        candidates.append(str(case["screenshot"]))
    candidates += [f"{case_id}.jpg", f"{case_id}.png", f"{case_id}.jpeg"]
    for fname in candidates:
        path = DEMO_DATA / "screenshots" / fname
        if path.is_file():
            return FileResponse(str(path))
    return JSONResponse(
        {"error": f"Screenshot for case {case_id} not found"},
        status_code=404,
    )


def _normalized_elements(vlm_elements: List[Dict[str, Any]],
                         img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Convert raw VLM elements to the hero-case JSON schema (normalised bbox).

    Mirrors ``demo_data/cases.json`` so the frontend can reuse the same
    rendering code for both hero cases and uploads.

    NOTE: bboxes are normalized against the qwen3-vl-flash fixed coordinate
    baseline (1080x960), NOT the original image size (img_w/img_h) — the
    VLM returns coords in that internal frame regardless of input size.
    """
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(vlm_elements):
        bbox_raw = e.get("bbox_xyxy") or e.get("bbox") or e.get("bbox_2d") or []
        if len(bbox_raw) == 4:
            bbox = [
                float(bbox_raw[0]) / 1080.0, float(bbox_raw[1]) / 960.0,
                float(bbox_raw[2]) / 1080.0, float(bbox_raw[3]) / 960.0,
            ]
        else:
            bbox = [float(v) for v in bbox_raw] if len(bbox_raw) == 4 else []
        out.append({
            "bbox": bbox,
            "label": e.get("label", e.get("category", "unknown")),
            "text": e.get("text", ""),
            "id": i,
        })
    return out


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload screenshot → VLM detection → GNN analysis → overlay.

    Args:
        file: Screenshot image (JPEG/PNG/HEIC).

    Returns:
        JSON in the hero-case schema (id/img_w/img_h/vlm_elements/proposals/
        metrics/vlm_time_ms/gnn_time_ms) plus backward-compat fields
        (vlm/gnn/corrected_json/overlay_b64).  No ``image_b64`` — the
        frontend renders the locally-selected file instead.
    """
    # Read uploaded file
    img_bytes = await file.read()
    if not img_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    # Get image dimensions
    try:
        pil_img = Image.open(BytesIO(img_bytes))
        img_w, img_h = pil_img.size
    except Exception as e:
        return JSONResponse({"error": f"Invalid image: {e}"}, status_code=400)

    if img_w <= 0 or img_h <= 0:
        return JSONResponse(
            {"error": "Invalid image dimensions (0x0)"},
            status_code=400,
        )

    # Resolve API key from environment
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        return JSONResponse(
            {
                "error": "VLM API key not configured. "
                         "Set DASHSCOPE_API_KEY in .env to enable upload mode. "
                         "Hero cases mode is still available."
            },
            status_code=400,
        )

    p = get_pipeline()

    # Step 1: VLM detection
    vlm_result = p.detect_elements(img_bytes, api_key=key, model=VLM_DEFAULT_MODEL)
    if "error" in vlm_result and vlm_result["error"]:
        logger.error("VLM detection failed: %s", vlm_result["error"])
        return JSONResponse({
            "error": f"VLM API error: {vlm_result['error']}",
            "vlm": {"elements": [], "count": 0, "time_ms": vlm_result.get("time_ms", 0)},
        }, status_code=502)

    vlm_elements = vlm_result.get("elements", [])

    # Step 2: GNN analysis
    gnn_result = p.gnn_analyse(vlm_elements, img_w=img_w, img_h=img_h)

    # Build corrected JSON
    corrected = p.build_corrected_json(vlm_elements, gnn_result, img_w=img_w, img_h=img_h)

    # Step 3: Render overlay
    try:
        overlay_bytes = p.render_overlay(img_bytes, vlm_elements, gnn_result["proposals"])
        overlay_b64 = base64.b64encode(overlay_bytes).decode("utf-8")
    except Exception as e:
        logger.error("Overlay rendering failed: %s", e)
        overlay_b64 = ""

    vlm_time_ms = vlm_result.get("time_ms", 0)
    gnn_time_ms = gnn_result.get("time_ms", 0)

    # Hero-case schema (frontend reuses case rendering); no image_b64 —
    # the frontend already has the local file preview via URL.createObjectURL.
    response = {
        "id": "upload",
        "img_w": img_w,
        "img_h": img_h,
        "vlm_elements": _normalized_elements(vlm_elements, img_w, img_h),
        "proposals": gnn_result["proposals"],
        "metrics": None,  # no ground truth for uploads
        "vlm_time_ms": vlm_time_ms,
        "gnn_time_ms": gnn_time_ms,
        "fallback": gnn_result.get("fallback"),
        # Backward-compat fields (old frontend)
        "vlm": {
            "elements": vlm_elements,
            "count": len(vlm_elements),
            "time_ms": vlm_time_ms,
        },
        "gnn": {
            "proposals": gnn_result["proposals"],
            "constraints_count": gnn_result["graph_stats"]["constraints"],
            "violations_count": gnn_result["graph_stats"]["num_violated"],
            "proposals_count": gnn_result["graph_stats"]["num_proposals"],
            "time_ms": gnn_time_ms,
        },
        "overlay_b64": f"data:image/png;base64,{overlay_b64}",
        "corrected_json": corrected,
        "dimensions": {"width": img_w, "height": img_h},
    }

    logger.info(
        "Predict done: VLM=%d elems, GNN=%d constraints/%d violations/%d proposals, %d+%dms",
        len(vlm_elements),
        gnn_result["graph_stats"]["constraints"],
        gnn_result["graph_stats"]["num_violated"],
        gnn_result["graph_stats"]["num_proposals"],
        vlm_time_ms,
        gnn_time_ms,
    )

    return JSONResponse(response)


@app.post("/api/gnn-only")
async def gnn_only(
    file: UploadFile = File(...),
    vlm_json: str = Form(...),
) -> JSONResponse:
    """Upload screenshot + VLM JSON → GNN analysis only (no VLM API call).

    Args:
        file: Screenshot image (for overlay).
        vlm_json: VLM prediction JSON string (list of element dicts).

    Returns:
        JSON with gnn, overlay_b64 fields.
    """
    img_bytes = await file.read()
    if not img_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    try:
        pil_img = Image.open(BytesIO(img_bytes))
        img_w, img_h = pil_img.size
    except Exception as e:
        return JSONResponse({"error": f"Invalid image: {e}"}, status_code=400)

    if img_w <= 0 or img_h <= 0:
        return JSONResponse(
            {"error": "Invalid image dimensions (0x0)"},
            status_code=400,
        )

    # Parse VLM JSON
    try:
        vlm_data = json.loads(vlm_json)
        if isinstance(vlm_data, list):
            vlm_elements = vlm_data
        elif isinstance(vlm_data, dict):
            vlm_elements = vlm_data.get("elements", vlm_data.get("predictions", []))
        else:
            return JSONResponse({"error": "vlm_json must be a list or dict"}, status_code=400)
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    p = get_pipeline()

    # GNN analysis
    gnn_result = p.gnn_analyse(vlm_elements, img_w=img_w, img_h=img_h)

    # Build corrected JSON
    corrected = p.build_corrected_json(vlm_elements, gnn_result, img_w=img_w, img_h=img_h)

    # Render overlay
    try:
        overlay_bytes = p.render_overlay(img_bytes, vlm_elements, gnn_result["proposals"])
        overlay_b64 = base64.b64encode(overlay_bytes).decode("utf-8")
    except Exception as e:
        logger.error("Overlay rendering failed: %s", e)
        overlay_b64 = ""

    response = {
        "id": "upload",
        "img_w": img_w,
        "img_h": img_h,
        "vlm_elements": _normalized_elements(vlm_elements, img_w, img_h),
        "proposals": gnn_result["proposals"],
        "metrics": None,  # no ground truth for uploads
        "gnn_time_ms": gnn_result.get("time_ms", 0),
        "fallback": gnn_result.get("fallback"),
        "gnn": {
            "proposals": gnn_result["proposals"],
            "constraints_count": gnn_result["graph_stats"]["constraints"],
            "violations_count": gnn_result["graph_stats"]["num_violated"],
            "proposals_count": gnn_result["graph_stats"]["num_proposals"],
            "time_ms": gnn_result["time_ms"],
        },
        "overlay_b64": f"data:image/png;base64,{overlay_b64}",
        "corrected_json": corrected,
        "dimensions": {"width": img_w, "height": img_h},
        "vlm": {"elements": vlm_elements, "count": len(vlm_elements)},
    }

    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
