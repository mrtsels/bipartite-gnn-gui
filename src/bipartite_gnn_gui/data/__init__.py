"""
Data module — Dataset loading, preprocessing, and unified data interfaces.

Provides loaders for:
- VLM output JSON files (Qwen3.5-2B, MiniMax-VL-01 formats).
- Ground-truth annotations (GUI-360°, ScreenSpot).
- Preprocessing utilities (coordinate normalization, feature extraction).
- PyTorch Dataset / DataLoader wrappers for training and evaluation.
"""

from .vlm_output import (
    ELEMENT_TYPES,
    VLMOutput,
    VLMOutputElement,
    VlmParseError,
    normalize_bbox,
    normalize_element_type,
    parse_minimax_output,
    parse_qwen_output,
)
from .ground_truth import (
    GTElement,
    GroundTruth,
    GroundTruthParseError,
    load_ground_truth,
    load_gui360_annotation,
    load_screenspot_annotation,
    match_predictions_to_ground_truth,
)
from .preprocess import normalize_coordinates, extract_element_features
from .dataset import GUIDataset, GUIDataModule

__all__ = [
    "ELEMENT_TYPES",
    "VLMOutput",
    "VLMOutputElement",
    "VlmParseError",
    "normalize_bbox",
    "normalize_element_type",
    "parse_minimax_output",
    "parse_qwen_output",
    "GTElement",
    "GroundTruth",
    "GroundTruthParseError",
    "load_ground_truth",
    "load_gui360_annotation",
    "load_screenspot_annotation",
    "match_predictions_to_ground_truth",
    "normalize_coordinates",
    "extract_element_features",
    "GUIDataset",
    "GUIDataModule",
]
