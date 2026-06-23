"""Generate VLM predictions for GUI screenshots using Ollama-hosted VLMs.

Usage:
  python scripts/generate_ollama_predictions.py \
    --model llava \
    --input data/rico_local/combined \
    --output data/vlm_predictions/rico_llava \
    --n 50 --workers 2

Supported models: llava, moondream, ... (any vision model in Ollama)

Output format:
  {"image_id": "0", "image_width": W, "image_height": H,
   "model_name": "llava",
   "elements": [{"bbox_xyxy": [x1,y1,x2,y2], "label": "button", "text": ""}, ...]}
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROMPT = (
    "You are a GUI element detector. Analyze this mobile app screenshot and "
    "list ALL visible UI elements. For each element, output bounding box "
    "coordinates (normalized 0-1, format [x1,y1,x2,y2]) and type. "
    "Types: button, text, icon, input, image, menu, slider, switch, checkbox, "
    "radio, divider, modal, container, toast, banner, progress_bar, other.\n\n"
    "Return ONLY a valid JSON array. Example:\n"
    '[{"bbox_xyxy":[0.1,0.2,0.3,0.4],"label":"button","text":"Submit"}]\n\n'
    "No other text, no markdown formatting."
)


def query_ollama(img_path: str, model: str, timeout: int = 120) -> list[dict]:
    """Run Ollama vision model on image, return parsed elements."""
    try:
        r = subprocess.run(
            ["ollama", "run", model, img_path],
            input=PROMPT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = r.stdout.strip()
        # Extract JSON array from response
        if "[" in out:
            start = out.index("[")
            end = out.rindex("]") + 1
            return json.loads(out[start:end])
        return []
    except Exception as e:
        log.warning("  %s: %s", Path(img_path).stem, e)
        return []


def get_image_size(path: str) -> tuple[int, int]:
    """Get image dimensions from file."""
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VLM predictions via Ollama"
    )
    parser.add_argument("--model", default="llava",
                        help="Ollama model name (default: llava)")
    parser.add_argument("--input", default="data/rico_local/combined",
                        help="Input image directory")
    parser.add_argument("--output", default="data/vlm_predictions/rico_llava",
                        help="Output JSON directory")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of images to process")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent workers")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images already processed
    existing = {p.stem for p in output_dir.glob("*.json")}
    all_images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    to_process = [p for p in all_images if p.stem not in existing][:args.n]

    log.info("=" * 55)
    log.info("Ollama Prediction Generator")
    log.info("=" * 55)
    log.info("Images:   %d/%d (%s)", len(to_process), len(all_images), args.input)
    log.info("Model:    %s", args.model)
    log.info("Output:   %s", args.output)
    log.info("Workers:  %d", args.workers)
    log.info("Processing %d images...", len(to_process))

    t0 = time.time()
    ok_count = 0
    total_elements = 0
    skipped = 0

    if args.workers <= 1:
        for img in to_process:
            result = _process_one(img, output_dir, args.model)
            if result is None:
                skipped += 1
            else:
                ok, elems = result
                ok_count += ok
                total_elements += elems
            if ok_count % 10 == 0 and ok_count > 0:
                log.info("  Progress: %d/%d OK (%d elements so far)",
                         ok_count, len(to_process), total_elements)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, img, output_dir, args.model): img
                for img in to_process
            }
            for f in as_completed(futures):
                result = f.result()
                if result is None:
                    skipped += 1
                else:
                    ok, elems = result
                    ok_count += ok
                    total_elements += elems
                if ok_count % 10 == 0 and ok_count > 0:
                    log.info("  Progress: %d/%d OK (%d elements so far)",
                             ok_count, len(to_process), total_elements)

    dt = time.time() - t0
    log.info("=" * 55)
    log.info("DONE in %.1fs", dt)
    log.info("  OK:      %d", ok_count)
    log.info("  Skipped: %d", skipped)
    log.info("  Errors:  %d", len(to_process) - ok_count)
    log.info("  Elements: %d (avg %.1f/img)", total_elements,
             total_elements / max(ok_count, 1))
    log.info("  Speed:   %.1f img/min", ok_count / max(dt, 1) * 60)
    log.info("=" * 55)


def _process_one(
    img_path: Path, output_dir: Path, model: str
) -> tuple[int, int] | None:
    """Process one image: query model, save result."""
    elements = query_ollama(str(img_path), model)
    if not elements:
        return None

    # Normalize coordinates: if any coord > 1.0, divide by image dimensions
    img_w, img_h = get_image_size(str(img_path))
    all_norm = True
    for elem in elements:
        bbox = elem.get("bbox_xyxy") or elem.get("bbox")
        if not bbox:
            continue
        if any(v > 1.0 for v in bbox):
            all_norm = False
            break

    if not all_norm:
        for elem in elements:
            bbox = elem.get("bbox_xyxy") or elem.get("bbox")
            if bbox:
                elem["bbox_xyxy"] = [bbox[0] / img_w, bbox[1] / img_h,
                                     bbox[2] / img_w, bbox[3] / img_h]

    result = {
        "image_id": img_path.stem,
        "image_width": img_w,
        "image_height": img_h,
        "model_name": model,
        "elements": elements,
    }

    with open(output_dir / f"{img_path.stem}.json", "w") as f:
        json.dump(result, f)

    return (1, len(elements))


if __name__ == "__main__":
    main()
