# VLM Output Sample Data

This directory contains sample VLM JSON output files for testing and
documentation.  The JSON format follows the conventions defined in
`docs/requirements/vlm_format.md`.

## Qwen3.5-2B Format

Qwen3.5-2B outputs **normalized** bounding-box coordinates in `[0, 1]`.

```json
{
  "image_id": "screenshot_001.png",
  "elements": [
    {
      "bbox_xyxy": [0.12, 0.34, 0.28, 0.40],
      "label": "button",
      "text": "Submit",
      "confidence": 0.95
    },
    {
      "bbox_xyxy": [0.30, 0.10, 0.60, 0.16],
      "label": "text",
      "text": "Welcome to the dashboard"
    },
    {
      "bbox_xyxy": [0.05, 0.50, 0.40, 0.58],
      "label": "input",
      "text": "Enter your name"
    }
  ]
}
```

### Key points

- Coordinates are already normalized to `[0, 1]`.
- The bounding-box field name is `bbox_xyxy` (older versions may use `bbox`).
- The element type field is `label`.
- The OCR text field is `text`.
- `confidence` is optional; defaults to `1.0` when missing.
- Empty string `""` in `text` is treated as `None`.

## MiniMax-VL-01 Format

MiniMax-VL-01 outputs **pixel-value** bounding-box coordinates that require
normalization by dividing by `image_width` / `image_height`.

```json
{
  "image_id": "screen_42",
  "image_width": 1920,
  "image_height": 1080,
  "elements": [
    {
      "bbox": [100, 200, 400, 350],
      "category": "button",
      "confidence": 0.92,
      "text_content": "Click me",
      "attributes": {
        "role": "primary",
        "disabled": false
      }
    },
    {
      "bbox": [450, 50, 800, 120],
      "category": "text",
      "confidence": 0.98,
      "text_content": "User Profile Settings",
      "attributes": {}
    },
    {
      "bbox": [100, 400, 400, 440],
      "category": "input",
      "confidence": 0.87,
      "text_content": "username@example.com",
      "attributes": {
        "placeholder": "Enter email"
      }
    }
  ]
}
```

### Key points

- Coordinates are pixel values (absolute); divide by `image_width`/`image_height`
  to normalize to `[0, 1]`.
- The bounding-box field name is `bbox` (in `xyxy` order).
- The element type field is `category` (may include compound strings like
  `"button-primary"` — the parser extracts the prefix before the `-`).
- `attributes` is a free-form dict for metadata (role, disabled, placeholder, etc.).
- `confidence` is always present in MiniMax output.

## Shared Element Types

The canonical element type taxonomy (case-insensitive):

| Canonical Key | Aliases                                     |
| ------------- | ------------------------------------------- |
| `button`      | `btn`                                       |
| `text`        | `label`, `paragraph`, `span`                |
| `image`       | `img`, `picture`                            |
| `input`       | `textbox`, `search`, `textarea`, `textfield`|
| `icon`        | `glyph`                                     |
| `container`   | `div`, `section`, `frame`, `panel`          |
| `card`        | —                                           |
| `checkbox`    | `check`                                     |
| `radio`       | `radiobutton`                               |
| `slider`      | `range`                                     |
| `switch`      | `toggle`                                    |
| `label`       | —                                           |
| `tab`         | —                                           |
| `menu`        | `dropdown`, `nav`                           |
| `divider`     | `separator`, `hr`                           |
| `list`        | —                                           |
| `modal`       | `dialog`, `overlay`                         |
| `toast`       | `snackbar`, `notification`                  |
| `banner`      | `announcement`, `alertbar`                  |
| `other`       | Fallback for unrecognised types.            |

## Coordinate Convention

All parsed coordinates are stored internally in **normalized xyxy** format:

```
(x1, y1) — top-left corner, ∈ [0, 1]
(x2, y2) — bottom-right corner, ∈ [0, 1]
```

Origin is top-left of the image.  Values are clamped to `[0.0, 1.0]`.
