"""Tests for data preprocessing — CoordinateNormalizer, feature extraction,
type embedding, confidence extraction, train/val/test split, and coordinate
normalization."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from bipartite_gnn_gui.data.preprocess import (
    CoordinateNormalizer,
    extract_confidence_scores,
    extract_element_features,
    extract_spatial_features,
    extract_type_embedding,
    normalize_coordinates,
    train_val_test_split,
)
from bipartite_gnn_gui.data.vlm_output import ELEMENT_TYPES, VLMOutputElement
from bipartite_gnn_gui.data.ground_truth import GTElement


# ===================================================================
# CoordinateNormalizer
# ===================================================================


class TestCoordinateNormalizer:
    """Tests for CoordinateNormalizer — fit, transform, inverse_transform,
    fit_from_elements, fitted property, and error handling."""

    def test_fit_and_transform(self) -> None:
        """Basic fit and transform returns zero-mean unit-variance."""
        bboxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [50.0, 50.0, 150.0, 150.0],
                [100.0, 100.0, 200.0, 200.0],
            ],
            dtype=torch.float32,
        )
        normalizer = CoordinateNormalizer()
        normalizer.fit(bboxes)
        assert normalizer.fitted

        normed = normalizer.transform(bboxes)
        assert normed.shape == (3, 4)

        # Mean of normalized values should be near 0
        assert torch.allclose(normed.mean(dim=0), torch.zeros(4), atol=1e-6)

        # Std of normalized values should be near 1 (population std, unbiased=False)
        assert torch.allclose(normed.std(dim=0, unbiased=False), torch.ones(4), atol=1e-6)

    def test_inverse_transform_roundtrip(self) -> None:
        """inverse_transform(transform(x)) == x within tolerance."""
        bboxes = torch.tensor(
            [
                [10.0, 20.0, 200.0, 150.0],
                [30.0, 40.0, 300.0, 250.0],
                [50.0, 60.0, 100.0, 80.0],
            ],
            dtype=torch.float32,
        )
        normalizer = CoordinateNormalizer()
        normalizer.fit(bboxes)

        normed = normalizer.transform(bboxes)
        restored = normalizer.inverse_transform(normed)
        assert torch.allclose(restored, bboxes, atol=1e-6)

    def test_fit_from_elements_vlm(self) -> None:
        """fit_from_elements works with VLMOutputElement instances."""
        elements = [
            VLMOutputElement(element_id=0, bbox=(0.1, 0.2, 0.5, 0.8), element_type="other"),
            VLMOutputElement(element_id=1, bbox=(0.2, 0.3, 0.6, 0.9), element_type="other"),
            VLMOutputElement(element_id=2, bbox=(0.3, 0.4, 0.7, 1.0), element_type="other"),
        ]
        normalizer = CoordinateNormalizer()
        normalizer.fit_from_elements(elements)
        assert normalizer.fitted
        assert normalizer.mean.shape == (4,)
        assert normalizer.std.shape == (4,)

    def test_fit_from_elements_gt(self) -> None:
        """fit_from_elements works with GTElement instances."""
        elements = [
            GTElement(element_id="0", bbox=(0.0, 0.0, 100.0, 100.0), element_type="other"),
            GTElement(element_id="1", bbox=(50.0, 50.0, 150.0, 150.0), element_type="other"),
        ]
        normalizer = CoordinateNormalizer()
        normalizer.fit_from_elements(elements)
        assert normalizer.fitted

        normed = normalizer.transform(
            torch.tensor([[0.0, 0.0, 100.0, 100.0]], dtype=torch.float32)
        )
        assert normed.shape == (1, 4)

    def test_fit_from_elements_mixed(self) -> None:
        """fit_from_elements handles a mix of GTElement and VLMOutputElement."""
        elements: list = [
            VLMOutputElement(element_id=0, bbox=(0.1, 0.2, 0.5, 0.8), confidence=0.9, element_type="other"),
            GTElement(element_id="0", bbox=(0.2, 0.3, 0.6, 0.9), element_type="other"),
        ]
        normalizer = CoordinateNormalizer()
        normalizer.fit_from_elements(elements)
        assert normalizer.fitted
        assert normalizer.mean.shape == (4,)

    def test_fit_single_element(self) -> None:
        """fit with a single bbox gives std=0, transform handles eps."""
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
        normalizer = CoordinateNormalizer()
        normalizer.fit(bboxes)

        assert torch.allclose(normalizer.mean, bboxes[0])
        assert torch.allclose(normalizer.std, torch.zeros(4))

        # transform should not divide by zero (eps protects it)
        normed = normalizer.transform(bboxes)
        assert torch.allclose(normed, torch.zeros_like(normed), atol=1e-6)

    def test_not_fitted_error(self) -> None:
        """Accessing mean/std/transform before fit raises RuntimeError."""
        normalizer = CoordinateNormalizer()
        assert not normalizer.fitted

        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = normalizer.mean
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = normalizer.std
        with pytest.raises(RuntimeError, match="not been fitted"):
            normalizer.transform(torch.zeros((1, 4)))

    def test_invalid_bbox_format(self) -> None:
        """Invalid bbox_format raises ValueError."""
        with pytest.raises(ValueError, match="bbox_format"):
            CoordinateNormalizer(bbox_format="invalid")

    def test_fit_invalid_shape(self) -> None:
        """fit with non-(N,4) tensor raises ValueError."""
        normalizer = CoordinateNormalizer()
        with pytest.raises(ValueError, match="Expected"):
            normalizer.fit(torch.zeros((3, 5)))
        with pytest.raises(ValueError, match="Expected"):
            normalizer.fit(torch.zeros((4,)))

    def test_inverse_transform_identity(self) -> None:
        """Applying inverse_transform after transform recovers original."""
        bboxes = torch.tensor(
            [
                [5.0, 15.0, 50.0, 100.0],
                [10.0, 20.0, 60.0, 120.0],
            ],
            dtype=torch.float32,
        )
        normalizer = CoordinateNormalizer()
        normalizer.fit(bboxes)
        normed = normalizer.transform(bboxes)
        recovered = normalizer.inverse_transform(normed)
        assert torch.allclose(recovered, bboxes, atol=1e-6)

    def test_fit_chainable(self) -> None:
        """fit and fit_from_elements return self for chaining."""
        normalizer = CoordinateNormalizer()
        bboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        result = normalizer.fit(bboxes)
        assert result is normalizer


# ===================================================================
# extract_spatial_features
# ===================================================================


class TestExtractSpatialFeatures:
    """Tests for extract_spatial_features — single box, batched, edge cases."""

    def test_single_box(self) -> None:
        """Single xyxy box converted to (cx, cy, w, h)."""
        bbox = torch.tensor([0.0, 0.0, 100.0, 200.0], dtype=torch.float32)
        out = extract_spatial_features(bbox)
        expected = torch.tensor([50.0, 100.0, 100.0, 200.0], dtype=torch.float32)
        assert torch.allclose(out, expected)

    def test_batched_boxes(self) -> None:
        """Batch of xyxy boxes converted to (cx, cy, w, h)."""
        bboxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 200.0],
                [50.0, 50.0, 150.0, 250.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        )
        out = extract_spatial_features(bboxes)
        expected = torch.tensor(
            [
                [50.0, 100.0, 100.0, 200.0],
                [100.0, 150.0, 100.0, 200.0],
                [20.0, 30.0, 20.0, 20.0],
            ],
            dtype=torch.float32,
        )
        assert out.shape == (3, 4)
        assert torch.allclose(out, expected)

    def test_zero_size_box(self) -> None:
        """Zero-size box (x1 == x2 or y1 == y2) yields zero w or h."""
        bbox = torch.tensor([50.0, 50.0, 50.0, 100.0], dtype=torch.float32)
        out = extract_spatial_features(bbox)
        assert out[2].item() == 0.0  # w = 0
        assert out[3].item() == 50.0  # h = 50

    def test_negative_coordinates(self) -> None:
        """Boxes with negative coordinates are handled correctly."""
        bbox = torch.tensor([-10.0, -20.0, 10.0, 20.0], dtype=torch.float32)
        out = extract_spatial_features(bbox)
        expected = torch.tensor([0.0, 0.0, 20.0, 40.0], dtype=torch.float32)
        assert torch.allclose(out, expected)

    def test_float_dtype_preserved(self) -> None:
        """Output dtype matches input dtype."""
        bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        out = extract_spatial_features(bbox)
        assert out.dtype == torch.float64

    def test_3d_input(self) -> None:
        """Higher-dimensional inputs (..., 4) are handled."""
        bboxes = torch.zeros((2, 3, 4), dtype=torch.float32)
        bboxes[..., 2] = 100.0
        bboxes[..., 3] = 200.0
        out = extract_spatial_features(bboxes)
        assert out.shape == (2, 3, 4)
        # cx = (0 + 100) / 2 = 50, cy = (0 + 200) / 2 = 100
        assert torch.allclose(out[..., 0], torch.full((2, 3), 50.0))
        assert torch.allclose(out[..., 1], torch.full((2, 3), 100.0))


# ===================================================================
# extract_type_embedding
# ===================================================================


class TestExtractTypeEmbedding:
    """Tests for extract_type_embedding — known/unknown types,
    case-insensitivity, default/custom taxonomy."""

    def test_known_type(self) -> None:
        """Known type gets one-hot at its taxonomy index."""
        emb = extract_type_embedding("button")
        assert emb.shape == (len(ELEMENT_TYPES),)
        assert emb.dtype == torch.float32
        assert emb.sum().item() == pytest.approx(1.0)
        # "button" is first in ELEMENT_TYPES, index 0
        button_idx = list(ELEMENT_TYPES.keys()).index("button")
        assert emb[button_idx].item() == 1.0

    def test_unknown_type_goes_to_index_zero(self) -> None:
        """Unrecognized type maps to index 0."""
        emb = extract_type_embedding("nonexistent_type_xyz")
        assert emb[0].item() == 1.0
        assert emb.sum().item() == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        """Type matching is case-insensitive."""
        emb_upper = extract_type_embedding("BUTTON")
        emb_mixed = extract_type_embedding("Button")
        button_idx = list(ELEMENT_TYPES.keys()).index("button")
        assert emb_upper[button_idx].item() == 1.0
        assert emb_mixed[button_idx].item() == 1.0

    def test_custom_taxonomy(self) -> None:
        """Custom taxonomy is respected."""
        taxonomy = ["unknown", "button", "text", "image"]
        emb = extract_type_embedding("text", taxonomy=taxonomy)
        assert emb.shape == (4,)
        assert emb[2].item() == 1.0  # index 2 = "text"

    def test_custom_taxonomy_unknown(self) -> None:
        """Unrecognized label in custom taxonomy maps to index 0."""
        taxonomy = ["catchall", "type_a", "type_b"]
        emb = extract_type_embedding("type_c", taxonomy=taxonomy)
        assert emb[0].item() == 1.0  # index 0 = "catchall"

    def test_empty_taxonomy(self) -> None:
        """Empty taxonomy returns empty tensor."""
        emb = extract_type_embedding("anything", taxonomy=[])
        assert emb.shape == (0,)

    def test_type_with_whitespace(self) -> None:
        """Labels with surrounding whitespace are stripped before matching."""
        emb = extract_type_embedding("  button  ")
        button_idx = list(ELEMENT_TYPES.keys()).index("button")
        assert emb[button_idx].item() == 1.0

    def test_default_taxonomy_length(self) -> None:
        """Default taxonomy has 20 entries matching ELEMENT_TYPES."""
        emb = extract_type_embedding("text")
        assert len(emb) == len(ELEMENT_TYPES) == 20


# ===================================================================
# extract_confidence_scores
# ===================================================================


class TestExtractConfidenceScores:
    """Tests for extract_confidence_scores — mixed VLM/GT elements,
    empty list, etc."""

    def test_vlm_elements(self) -> None:
        """Confidence scores are taken from VLMOutputElement.confidence."""
        elements = [
            VLMOutputElement(element_id=0, bbox=(0, 0, 1, 1), confidence=0.9, element_type="other"),
            VLMOutputElement(element_id=1, bbox=(0, 0, 1, 1), confidence=0.75, element_type="other"),
        ]
        scores = extract_confidence_scores(elements)
        assert scores.shape == (2,)
        assert scores.dtype == torch.float32
        assert torch.allclose(scores, torch.tensor([0.9, 0.75]))

    def test_gt_elements_default_to_one(self) -> None:
        """GTElement does not have confidence; defaults to 1.0."""
        elements = [
            GTElement(element_id="0", bbox=(0.0, 0.0, 1.0, 1.0), element_type="other"),
            GTElement(element_id="1", bbox=(0.1, 0.1, 0.5, 0.5), element_type="other"),
        ]
        scores = extract_confidence_scores(elements)
        assert scores.shape == (2,)
        assert torch.allclose(scores, torch.ones(2))

    def test_mixed_elements(self) -> None:
        """Mixed VLM/GT elements yield appropriate confidence values."""
        elements = [
            VLMOutputElement(element_id=0, bbox=(0, 0, 1, 1), confidence=0.85, element_type="other"),
            GTElement(element_id="0", bbox=(0.0, 0.0, 1.0, 1.0), element_type="other"),
        ]
        scores = extract_confidence_scores(elements)
        expected = torch.tensor([0.85, 1.0])
        assert torch.allclose(scores, expected)

    def test_empty_list(self) -> None:
        """Empty list returns empty tensor."""
        scores = extract_confidence_scores([])
        assert scores.shape == (0,)
        assert scores.dtype == torch.float32

    def test_confidence_range(self) -> None:
        """Confidence values are passed through as-is."""
        elements = [
            VLMOutputElement(element_id=0, bbox=(0, 0, 1, 1), confidence=0.0, element_type="other"),
            VLMOutputElement(element_id=1, bbox=(0, 0, 1, 1), confidence=1.0, element_type="other"),
            VLMOutputElement(element_id=2, bbox=(0, 0, 1, 1), confidence=0.5, element_type="other"),
        ]
        scores = extract_confidence_scores(elements)
        assert torch.allclose(
            scores, torch.tensor([0.0, 1.0, 0.5])
        )


# ===================================================================
# train_val_test_split
# ===================================================================


class TestTrainValTestSplit:
    """Tests for train_val_test_split — split ratios, determinism,
    edge cases, and error handling."""

    def test_default_split(self) -> None:
        """Default split (0.1 val, 0.1 test) produces correct sizes."""
        elements = list(range(100))
        train, val, test = train_val_test_split(elements)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_ratios_respected(self) -> None:
        """Custom split ratios produce correct proportions."""
        elements = list(range(200))
        train, val, test = train_val_test_split(
            elements, val_split=0.2, test_split=0.3, seed=42
        )
        assert len(train) == 100  # 200 - 40 - 60
        assert len(val) == 40
        assert len(test) == 60

    def test_no_test_split(self) -> None:
        """test_split=0 produces no test set."""
        elements = list(range(50))
        train, val, test = train_val_test_split(
            elements, val_split=0.2, test_split=0.0
        )
        assert len(train) == 40
        assert len(val) == 10
        assert len(test) == 0

    def test_no_val_split(self) -> None:
        """val_split=0 produces no val set."""
        elements = list(range(50))
        train, val, test = train_val_test_split(
            elements, val_split=0.0, test_split=0.2
        )
        assert len(train) == 40
        assert len(val) == 0
        assert len(test) == 10

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical splits."""
        elements = list(range(100))
        train1, val1, test1 = train_val_test_split(elements, seed=42)
        train2, val2, test2 = train_val_test_split(elements, seed=42)
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds(self) -> None:
        """Different seeds produce different splits (high probability)."""
        elements = list(range(100))
        _, val1, test1 = train_val_test_split(elements, seed=42)
        _, val2, test2 = train_val_test_split(elements, seed=99)
        # Extremely unlikely that both val and test are identical
        assert val1 != val2 or test1 != test2

    def test_no_elements(self) -> None:
        """Empty list returns three empty lists."""
        train, val, test = train_val_test_split([])
        assert train == []
        assert val == []
        assert test == []

    def test_single_element(self) -> None:
        """Single element goes to train when val/test are 0."""
        train, val, test = train_val_test_split([42], val_split=0.0, test_split=0.0)
        assert train == [42]
        assert val == []
        assert test == []

    def test_splits_cover_all_elements(self) -> None:
        """Elements are partitioned without loss."""
        elements = list(range(73))
        train, val, test = train_val_test_split(
            elements, val_split=0.15, test_split=0.15
        )
        all_splits = train + val + test
        assert sorted(all_splits) == sorted(elements)
        assert len(all_splits) == len(elements)

    def test_invalid_split_sum(self) -> None:
        """val_split + test_split >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be < 1.0"):
            train_val_test_split([1, 2, 3], val_split=0.6, test_split=0.5)


# ===================================================================
# normalize_coordinates
# ===================================================================


class TestNormalizeCoordinates:
    """Tests for normalize_coordinates — basic, edge cases, format."""

    def test_xyxy_default(self) -> None:
        """Default xyxy format normalizes corners by width/height."""
        result = normalize_coordinates([100, 200, 500, 800], 1000, 1000)
        expected = [0.1, 0.2, 0.5, 0.8]
        assert result == pytest.approx(expected)

    def test_xywh_format(self) -> None:
        """xywh format normalizes (x, y, w, h) independently."""
        result = normalize_coordinates(
            [100, 200, 400, 600], 1000, 1000, fmt="xywh"
        )
        expected = [0.1, 0.2, 0.4, 0.6]
        assert result == pytest.approx(expected)

    def test_zero_width_raises(self) -> None:
        """Zero width raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            normalize_coordinates([0, 0, 10, 10], 0, 100)

    def test_zero_height_raises(self) -> None:
        """Zero height raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            normalize_coordinates([0, 0, 10, 10], 100, 0)

    def test_edge_pixel_values(self) -> None:
        """Box at image boundaries normalizes to 0 or 1."""
        result = normalize_coordinates([0, 0, 1920, 1080], 1920, 1080)
        assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_negative_coordinates(self) -> None:
        """Negative pixel values produce negative normalized values."""
        result = normalize_coordinates([-100, -100, 100, 100], 1000, 1000)
        expected = [-0.1, -0.1, 0.1, 0.1]
        assert result == pytest.approx(expected)


# ===================================================================
# extract_element_features (legacy)
# ===================================================================


class TestExtractElementFeatures:
    """Tests for the legacy extract_element_features helper."""

    def test_basic_extraction(self) -> None:
        """Basic extraction returns 5-d tensor with bbox + confidence."""
        element: Dict[str, Any] = {
            "bbox": [10.0, 20.0, 100.0, 80.0],
            "confidence": 0.95,
        }
        features = extract_element_features(element)
        assert features.shape == (5,)
        assert features.dtype == torch.float32
        expected = torch.tensor([10.0, 20.0, 100.0, 80.0, 0.95])
        assert torch.allclose(features, expected)

    def test_default_confidence(self) -> None:
        """Missing confidence defaults to 1.0."""
        element: Dict[str, Any] = {"bbox": [0.0, 0.0, 1.0, 1.0]}
        features = extract_element_features(element)
        assert features[-1].item() == 1.0

    def test_default_bbox(self) -> None:
        """Missing bbox defaults to zeros."""
        features = extract_element_features({})
        assert torch.allclose(features[:4], torch.zeros(4))
        assert features[-1].item() == 1.0


# ===================================================================
# Integration test
# ===================================================================


class TestPreprocessingPipeline:
    """End-to-end integration test combining multiple preprocessing steps."""

    def test_full_pipeline(self) -> None:
        """Pipeline: extract features -> normalize -> type embed -> split."""
        vlm_elements = [
            VLMOutputElement(
                element_id=0,
                bbox=(0.1, 0.2, 0.5, 0.8),
                element_type="button",
                confidence=0.95,
            ),
            VLMOutputElement(
                element_id=1,
                bbox=(0.2, 0.3, 0.6, 0.9),
                element_type="text",
                confidence=0.87,
            ),
            VLMOutputElement(
                element_id=2,
                bbox=(0.3, 0.4, 0.7, 1.0),
                element_type="input",
                confidence=0.92,
            ),
        ]

        # 1. Confidence scores
        conf = extract_confidence_scores(vlm_elements)
        assert conf.shape == (3,)

        # 2. Spatial features
        bboxes = torch.tensor([list(e.bbox) for e in vlm_elements])
        spatial = extract_spatial_features(bboxes)
        assert spatial.shape == (3, 4)

        # 3. Type embeddings
        for elem in vlm_elements:
            emb = extract_type_embedding(elem.element_type)
            assert emb.shape == (len(ELEMENT_TYPES),)
            assert emb.sum().item() == pytest.approx(1.0)

        # 4. Normalize
        normalizer = CoordinateNormalizer()
        normalizer.fit(bboxes)
        normed = normalizer.transform(bboxes)
        assert normed.shape == (3, 4)
        restored = normalizer.inverse_transform(normed)
        assert torch.allclose(restored, bboxes, atol=1e-6)

        # 5. Split
        train, val, test = train_val_test_split(
            vlm_elements, val_split=0.0, test_split=0.0
        )
        assert len(train) == 3
