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
from .preprocess import (
    CoordinateNormalizer,
    extract_confidence_scores,
    extract_element_features,
    extract_spatial_features,
    extract_type_embedding,
    normalize_coordinates,
    train_val_test_split,
)
from .dataset import GUIDataset, GUIDataModule

__all__ = [
    "CoordinateNormalizer",
    "ELEMENT_TYPES",
    "extract_confidence_scores",
    "extract_element_features",
    "extract_spatial_features",
    "extract_type_embedding",
    "GTElement",
    "GroundTruth",
    "GroundTruthParseError",
    "GUIDataModule",
    "GUIDataset",
    "load_ground_truth",
    "load_gui360_annotation",
    "load_screenspot_annotation",
    "match_predictions_to_ground_truth",
    "normalize_bbox",
    "normalize_coordinates",
    "normalize_element_type",
    "parse_minimax_output",
    "parse_qwen_output",
    "train_val_test_split",
    "VLMOutput",
    "VLMOutputElement",
    "VlmParseError",
]
