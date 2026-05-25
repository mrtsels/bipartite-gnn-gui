"""
Utility module — Configuration, logging, and miscellaneous helpers.

Provides shared infrastructure used across all modules:
    - YAML-based configuration loading and validation.
    - Structured logging (console + file) with experiment tracking integration.
    - Common helper functions (seeding, coordinate transforms,
      tensor type conversions, bounding box utilities).
"""

from .config import Config, load_config, validate_config
from .logging import setup_logger, get_logger
from .helpers import (
    set_seed,
    bbox_to_tensor,
    tensor_to_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
    compute_iou_pair,
)

__all__ = [
    "Config",
    "load_config",
    "validate_config",
    "setup_logger",
    "get_logger",
    "set_seed",
    "bbox_to_tensor",
    "tensor_to_bbox",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "compute_iou_pair",
]
