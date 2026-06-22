"""Generate VLM predictions for GUI screenshots using Qwen3-VL via Alibaba Cloud API.

Usage:
  1. Set DASHSCOPE_API_KEY in environment (or pass --api-key)
  2. Run on a batch of RICO images:
     python scripts/generate_vlm_predictions.py \
       --input data/rico_local/combined \
       --images data/rico_local/combined \
       --output data/vlm_predictions \
       --n 50 \
       --model qwen3-vl-plus

Output: Per-image JSON files in --output dir, matching the format expected by
the training pipeline (scripts/run_experiment.py).

API Reference:
  https://www.alibabacloud.com/help/zh/model-studio/qwen-api-via-openai-chat-completions
"""
from __future__ import annotations

import argparse
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

DEFAULT_MODEL = "qwen3-vl-plus"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_WORKERS = 4  # concurrent API calls (stay within rate limits)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a GUI element detector. Given a screenshot of a mobile app UI, identify all visible interactive and informative elements.

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
    model: str = DEFAULT_MODEL,
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
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {"error": str(e), "raw": content if "content" in locals() else ""}

    return {"error": "Max retries exceeded"}


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

    # Build output matching Qwen3.5-2B format
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


def _get_image_size(path: Path) -> tuple[int, int]:
    """Get image dimensions without loading full image."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except ImportError:
        # Fallback: parse JPEG headers
        with open(path, "rb") as f:
            data = f.read(2**16)
        # Simple JPEG SOF0 marker parser
        import struct
        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF and data[i + 1] == 0xC0:
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 1
        return 1440, 2560  # common RICO fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM predictions for GUI screenshots using Qwen3-VL API"
    )
    parser.add_argument("--input", required=True,
                        help="Directory containing screenshot images (JPG/PNG)")
    parser.add_argument("--output", required=True,
                        help="Directory for prediction JSON output")
    parser.add_argument("--api-key",
                        help="DashScope API key (default: DASHSCOPE_API_KEY env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Qwen VL model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of images to process (default: 50)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent API calls (default: {MAX_WORKERS})")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (skip first N images, default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List images that would be processed, then exit")
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("API key required. Set DASHSCOPE_API_KEY env var or pass --api-key")
        sys.exit(1)

    # Discover images
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not images:
        logger.error("No JPG/PNG images found in %s", input_dir)
        sys.exit(1)

    # Subset
    end = min(args.start + args.n, len(images))
    selected = images[args.start:end]
    output_dir = Path(args.output)

    logger.info("=" * 55)
    logger.info("Qwen3-VL Prediction Generator")
    logger.info("=" * 55)
    logger.info("Images:   %d/%d (%s)", len(selected), len(images), input_dir)
    logger.info("Model:    %s", args.model)
    logger.info("Output:   %s", output_dir)
    logger.info("Workers:  %d", args.workers)

    if args.dry_run:
        logger.info("Dry run — would process:")
        for img in selected:
            out_path = output_dir / f"{img.stem}.json"
            status = "EXISTS" if out_path.exists() else "NEW"
            logger.info("  [%s] %s", status, img.name)
        sys.exit(0)

    # Process
    logger.info("Processing %d images...", len(selected))
    t0 = time.time()
    results = {"ok": 0, "skipped": 0, "errors": 0, "total_elements": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_image, img, api_key, args.model, output_dir): img
            for img in selected
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
                                results["ok"], len(selected), results["total_elements"])
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
    logger.info("  Elements: %d (avg %.1f/img)",
                results["total_elements"],
                results["total_elements"] / max(results["ok"], 1))
    logger.info("  Speed:   %.1f img/min", results["ok"] / max(dt / 60, 0.01))
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
