"""Generate Qwen3-VL Flash predictions for all ScreenSpot images.

Adapted from generate_vlm_predictions.py for the ScreenSpot dataset.
Processes ALL 610 PC/web/mobile screenshots with qwen3-vl-flash.

Usage:
    source .env  # sets DASHSCOPE_API_KEY
    python scripts/generate_screenspot_predictions.py

Output: data/vlm_predictions/screenspot_qwen_flash/<stem>.json
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "qwen3-vl-flash"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_WORKERS = 4

IMAGES_DIR = Path("data/raw/screenspot/images")
OUTPUT_DIR = Path("data/vlm_predictions/screenspot_qwen_flash")

# ---------------------------------------------------------------------------
# Prompt — adapted for PC/web/mobile (not just mobile)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a GUI element detector. Given a screenshot of a user interface (PC desktop, web browser, or mobile app), identify all visible interactive and informative elements.

For each element, output a JSON object with these fields:
- "bbox_xyxy": [x1, y1, x2, y2] — the bounding box in **pixel coordinates** relative to the image dimensions. All values are integers.
- "label": one of: button, text, icon, image, input, container, checkbox, radio, switch, slider, tab, menu, divider, list, card, modal, toast, banner, toolbar, statusbar, scrollbar
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
4. Include ALL visible text, buttons, icons, inputs, images, menu items, tabs, toolbar elements
5. For PC and web screenshots, also detect window controls, system tray, taskbar, address bar
6. Ignore background images and decorative elements
7. Output ONLY the JSON array, no other text"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image(image_path: str | Path) -> str:
    """Read an image and return a base64 data URL (JPEG)."""
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_qwen_vl(
    image_b64: str,
    api_key: str,
    model: str = MODEL,
    timeout: int = 60,
) -> dict[str, Any]:
    """Call Qwen3-VL API with a single image. Returns parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": "Identify all GUI elements in this screenshot and return them as a JSON array."},
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            # The model may wrap JSON in ```json ... ``` or return raw JSON
            content = content.strip()
            if content.startswith("```"):
                # Extract from markdown code block
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )

            parsed = json.loads(content)
            if isinstance(parsed, list):
                return {"elements": parsed, "raw": content}
            if isinstance(parsed, dict) and "elements" in parsed:
                return {"elements": parsed["elements"], "raw": content}
            # Unexpected format — wrap as-is
            return {"elements": parsed if isinstance(parsed, list) else [parsed], "raw": content}

        except requests.exceptions.RequestException as e:
            logger.warning("API error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {"error": str(e)}

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Parse error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            raw_content = locals().get("content", "")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {"error": str(e), "raw": raw_content}

    return {"error": "Max retries exceeded"}


def _get_image_size(path: Path) -> tuple[int, int]:
    """Get image dimensions."""
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def process_one_image(
    image_path: Path,
    api_key: str,
    model: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Process a single image: call API, save result, return summary."""
    stem = image_path.stem
    out_path = output_dir / f"{stem}.json"

    if out_path.exists():
        return {"image": stem, "status": "skipped (exists)"}

    try:
        b64 = encode_image(image_path)
    except Exception as e:
        return {"image": stem, "status": f"encode error: {e}"}

    result = call_qwen_vl(b64, api_key, model)

    if "error" in result:
        return {"image": stem, "status": f"api error: {result['error']}"}

    # Build output
    img_w, img_h = _get_image_size(image_path)
    output = {
        "image_id": stem,
        "image_width": img_w,
        "image_height": img_h,
        "model_name": model,
        "elements": result["elements"],
        "raw_response": result.get("raw", ""),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_elems = len(result["elements"])
    return {"image": stem, "status": "ok", "elements": n_elems}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("API key required. Set DASHSCOPE_API_KEY env var (try: source .env)")
        sys.exit(1)

    if not IMAGES_DIR.is_dir():
        logger.error("Images directory not found: %s", IMAGES_DIR)
        sys.exit(1)

    images = sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        logger.error("No PNG images found in %s", IMAGES_DIR)
        sys.exit(1)

    # Filter out .DS_Store or any non-image files that snuck in
    images = [img for img in images if not img.name.startswith(".")]

    model = MODEL
    output_dir = OUTPUT_DIR

    logger.info("=" * 55)
    logger.info("ScreenSpot Qwen3-VL Flash Prediction Generator")
    logger.info("=" * 55)
    logger.info("Images:   %d", len(images))
    logger.info("Model:    %s", model)
    logger.info("Output:   %s", output_dir)
    logger.info("Workers:  %d", MAX_WORKERS)

    # Count by domain
    pc_count = len([i for i in images if i.name.startswith("pc_")])
    web_count = len([i for i in images if i.name.startswith("web_")])
    mobile_count = len([i for i in images if i.name.startswith("mobile_")])
    logger.info("  PC:      %d", pc_count)
    logger.info("  Web:     %d", web_count)
    logger.info("  Mobile:  %d", mobile_count)

    # Process
    logger.info("Processing %d images...", len(images))
    t0 = time.time()
    results = {"ok": 0, "skipped": 0, "errors": 0, "total_elements": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_image, img, api_key, model, output_dir): img
            for img in images
        }

        for future in as_completed(futures):
            img = futures[future]
            try:
                r = future.result()
            except Exception as e:
                logger.error("Unexpected error for %s: %s", img.name, e)
                results["errors"] += 1
                continue

            if r["status"] == "ok":
                results["ok"] += 1
                results["total_elements"] += r.get("elements", 0)
                if results["ok"] % 10 == 0:
                    logger.info("  Progress: %d/%d OK (%.0f elements so far)",
                                results["ok"], len(images), results["total_elements"])
            elif "skipped" in r["status"]:
                results["skipped"] += 1
            else:
                results["errors"] += 1
                logger.warning("  %s: %s", r["image"], r["status"])

    dt = time.time() - t0
    logger.info("=" * 55)
    logger.info("DONE in %.1fs", dt)
    logger.info("  OK:      %d", results["ok"])
    logger.info("  Skipped: %d", results["skipped"])
    logger.info("  Errors:  %d", results["errors"])
    ok = max(results["ok"], 1)
    logger.info("  Elements: %d (avg %.1f/img)",
                results["total_elements"],
                results["total_elements"] / ok)
    logger.info("  Speed:   %.1f img/min", results["ok"] / max(dt / 60, 0.01))
    logger.info("=" * 55)
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("")
    logger.info("Summary for task report:")
    logger.info("  Total images: %d", len(images))
    logger.info("  Successfully processed: %d", results["ok"])
    logger.info("  Skipped (already existed): %d", results["skipped"])
    logger.info("  Errors: %d", results["errors"])
    logger.info("  Total elements detected: %d", results["total_elements"])
    logger.info("  Average elements per image: %.1f", results["total_elements"] / ok)


if __name__ == "__main__":
    main()
