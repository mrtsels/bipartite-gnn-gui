"""FastAPI web demo — upload screenshot, detect with VLM, correct with GNN."""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from PIL import Image

from pipeline import DemoPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the frontend single-page app."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>GUI-GNN Demo</h1><p>Frontend not found.</p>", status_code=404)


def get_pipeline() -> DemoPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising DemoPipeline...")
        _pipeline = DemoPipeline(
            device="cpu",
            violation_threshold=0.3,
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


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    vlm_model: str = Form("qwen3-vl-flash"),
    api_key: str = Form(None),
) -> JSONResponse:
    """Upload screenshot → VLM detection → GNN analysis → overlay.

    Args:
        file: Screenshot image (JPEG/PNG).
        vlm_model: Qwen3-VL model name.
        api_key: DashScope API key. Falls back to DASHSCOPE_API_KEY env var.

    Returns:
        JSON with vlm, gnn, overlay_b64 fields.
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

    # Resolve API key
    key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        return JSONResponse(
            {"error": "API key required. Pass api_key or set DASHSCOPE_API_KEY env var."},
            status_code=400,
        )

    p = get_pipeline()

    # Step 1: VLM detection
    vlm_result = p.detect_elements(img_bytes, api_key=key, model=vlm_model)
    if "error" in vlm_result and vlm_result["error"]:
        logger.error("VLM detection failed: %s", vlm_result["error"])
        return JSONResponse({
            "error": f"VLM API error: {vlm_result['error']}",
            "vlm": {"elements": [], "count": 0, "time_ms": vlm_result.get("time_ms", 0)},
        }, status_code=502)

    vlm_elements = vlm_result.get("elements", [])

    # Step 2: GNN analysis
    gnn_result = p.gnn_analyse(vlm_elements, img_w=img_w, img_h=img_h)

    # Step 3: Render overlay
    try:
        overlay_bytes = p.render_overlay(img_bytes, vlm_elements, gnn_result["proposals"])
        import base64
        overlay_b64 = base64.b64encode(overlay_bytes).decode("utf-8")
    except Exception as e:
        logger.error("Overlay rendering failed: %s", e)
        overlay_b64 = ""

    # Build response
    response = {
        "vlm": {
            "elements": vlm_elements,
            "count": len(vlm_elements),
            "time_ms": vlm_result.get("time_ms", 0),
        },
        "gnn": {
            "proposals": gnn_result["proposals"],
            "constraints_count": gnn_result["graph_stats"]["constraints"],
            "violations_count": gnn_result["graph_stats"]["num_violated"],
            "proposals_count": gnn_result["graph_stats"]["num_proposals"],
            "time_ms": gnn_result["time_ms"],
        },
        "overlay_b64": f"data:image/png;base64,{overlay_b64}",
        "dimensions": {"width": img_w, "height": img_h},
    }

    logger.info(
        "Predict done: VLM=%d elems, GNN=%d constraints/%d violations/%d proposals, %d+%dms",
        len(vlm_elements),
        gnn_result["graph_stats"]["constraints"],
        gnn_result["graph_stats"]["num_violated"],
        gnn_result["graph_stats"]["num_proposals"],
        vlm_result.get("time_ms", 0),
        gnn_result["time_ms"],
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

    # Render overlay
    try:
        overlay_bytes = p.render_overlay(img_bytes, vlm_elements, gnn_result["proposals"])
        import base64
        overlay_b64 = base64.b64encode(overlay_bytes).decode("utf-8")
    except Exception as e:
        logger.error("Overlay rendering failed: %s", e)
        overlay_b64 = ""

    response = {
        "gnn": {
            "proposals": gnn_result["proposals"],
            "constraints_count": gnn_result["graph_stats"]["constraints"],
            "violations_count": gnn_result["graph_stats"]["num_violated"],
            "proposals_count": gnn_result["graph_stats"]["num_proposals"],
            "time_ms": gnn_result["time_ms"],
        },
        "overlay_b64": f"data:image/png;base64,{overlay_b64}",
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
