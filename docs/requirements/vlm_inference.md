# VLM Inference Guide

## Overview

This guide explains how to generate VLM predictions for use in the training pipeline.

## Generating Predictions

When you have VLM predictions for a set of screenshots, the training pipeline
can use them as input. The script `scripts/generate_vlm_predictions.py`
handles the generation workflow.

### Dry Run — See What the Script Will Process

Run the script with the `--dry-run` flag to preview files:

```bash
python scripts/generate_vlm_predictions.py --dry-run
```

### Output Format

The script saves each prediction JSON to the output directory.
The output format follows the conventions in `docs/requirements/vlm_format.md`.

### Tracking

The pipeline tracks which real VLM files it uses as input. This ensures
reproducible experiment logs.

### Error Handling

- If an API call fails, the script retries it up to 3 times.
- The script skips images that already have an output file.
- Failed API calls after all retries go into a separate error log.
