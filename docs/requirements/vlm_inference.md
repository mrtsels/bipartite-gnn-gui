# VLM Inference Pipeline

## Overview

The bipartite GNN correction model needs **real VLM predictions** to train on
actual error patterns. This doc describes how to generate those predictions
using Qwen3-VL via the Alibaba Cloud DashScope API (OpenAI-compatible).

When you have VLM predictions for a set of screenshots, the training pipeline
(`scripts/run_experiment.py`) can consume them directly instead of using
simulated Gaussian noise — giving the GNN real error patterns to learn from.

## API Setup

1. Go to [Alibaba Cloud Model Studio](https://www.aliyun.com/product/bailian)
   (大模型服务平台百炼)
2. Create an API key in the console
3. Set the environment variable:

```bash
export DASHSCOPE_API_KEY="sk-..."
```

Or pass `--api-key sk-...` to the script (less secure).

### API Details

| Field | Value |
|---|---|
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Auth | `Authorization: Bearer <api-key>` |
| Format | OpenAI-compatible (same as `openai` Python package) |
| Docs | https://www.alibabacloud.com/help/zh/model-studio/qwen-api-via-openai-chat-completions |

### Available Vision Models

| Model | Description |
|---|---|
| `qwen3-vl-plus` | Qwen3 VL, balanced speed/quality |
| `qwen3-vl-max` | Higher quality, slower, more expensive |

Use `--model qwen3-vl-plus` for batch processing (default).

## Usage

### Dry Run — See What Would Be Processed

```bash
python scripts/generate_vlm_predictions.py \
  --input data/rico_local/combined \
  --output data/vlm_predictions \
  --n 10 \
  --dry-run
```

### Process 100 RICO Images

```bash
python scripts/generate_vlm_predictions.py \
  --input data/rico_local/combined \
  --output data/vlm_predictions \
  --n 100 \
  --model qwen3-vl-plus \
  --workers 4
```

### Resume / Continue from Index 100

```bash
python scripts/generate_vlm_predictions.py \
  --input data/rico_local/combined \
  --output data/vlm_predictions \
  --n 100 \
  --start 100
```

### Process a Different Dataset

```bash
python scripts/generate_vlm_predictions.py \
  --input data/raw/screenspot/images \
  --output data/vlm_predictions/screenspot \
  --n 50
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | (required) | Directory with screenshot images (JPG/PNG) |
| `--output` | (required) | Directory for prediction JSONs |
| `--api-key` | env `DASHSCOPE_API_KEY` | DashScope API key |
| `--model` | `qwen3-vl-plus` | Qwen VL model name |
| `--n` | 50 | Number of images to process |
| `--workers` | 4 | Concurrent API calls (stay ≤ 8 for rate limits) |
| `--start` | 0 | Skip first N images (resume support) |
| `--dry-run` | — | List images without calling API |

## Output Format

Each prediction JSON is saved as `<stem>.json` in `--output`, matching the
format expected by `scripts/run_experiment.py`:

```json
{
  "image_id": "12345",
  "image_width": 1440,
  "image_height": 2560,
  "model_name": "qwen3-vl-plus",
  "elements": [
    {
      "bbox_xyxy": [10, 20, 100, 50],
      "label": "button",
      "text": "Submit"
    }
  ],
  "raw_response": "..."
}
```

## Feeding Predictions Into Training

1. Generate predictions:
```bash
python scripts/generate_vlm_predictions.py \
  --input data/rico_local/combined \
  --output data/vlm_predictions/rico_qwen \
  --n 500
```

2. Train with real VLM predictions (not simulated noise):
```bash
python scripts/run_experiment.py \
  --vlm-dir data/vlm_predictions/rico_qwen \
  --vlm-gt data/rico_local/combined \
  --n 500 --epochs 50
```

Note: `run_experiment.py` currently uses simulated noise internally.
Integration of real VLM files as input is tracked in [TASK.md](../../TASK.md).

## Cost Estimate

| Model | ~1 image cost (¥) | 500 images (¥) | 1000 images (¥) |
|---|---|---|---|
| `qwen3-vl-plus` | ~0.003 | ~1.5 | ~3.0 |
| `qwen3-vl-max` | ~0.01 | ~5.0 | ~10.0 |

Prices are approximate at time of writing. Actual cost depends on image
resolution (resized internally by DashScope) and output token count.

## Rate Limits

- Default quota: ~10-20 requests/min for free tier
- Batch of 500 images at 4 workers: ~5-10 minutes
- Use `--workers 4` to stay safe; increase to 8 only if throughput confirmed

## Error Handling

- Failed API calls are retried up to 3 times with exponential backoff
- Images with existing outputs are skipped (safe to restart)
- Use `--start N` to resume from a specific index
- Results summary shows OK / Skipped / Errors at end of run
