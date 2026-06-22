"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.eval.metrics import (
    AlignmentError,
    ElementPrecision,
    ElementRecall,
    F1Score,
    MetricsBundle,
    PositionError,
    SizeError,
    _detect_alignments,
    compute_all_metrics,
    compute_iou,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boxes(x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_perfect_overlap(self) -> None:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        assert torch.allclose(compute_iou(b, b), torch.tensor([[1.0]]))

    def test_no_overlap(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        b2 = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
        iou = compute_iou(b1, b2)
        assert torch.allclose(iou, torch.tensor([[0.0]]))

    def test_n_m_matrix(self) -> None:
        b1 = torch.rand(3, 4)
        b2 = torch.rand(5, 4)
        iou = compute_iou(b1, b2)
        assert iou.shape == (3, 5)


# ---------------------------------------------------------------------------
# PositionError
# ---------------------------------------------------------------------------


class TestPositionError:
    def test_perfect_match(self) -> None:
        b = _boxes(0.1, 0.2, 0.3, 0.4)
        err = PositionError()(b, b)
        assert err.item() == pytest.approx(0.0, abs=1e-6)

    def test_offset(self) -> None:
        pred = _boxes(0.1, 0.1, 0.3, 0.3)
        tgt = _boxes(0.1, 0.2, 0.3, 0.4)
        err = PositionError()(pred, tgt)
        # top-left offset: sqrt((0.1-0.1)^2 + (0.1-0.2)^2) = 0.1
        assert err.item() == pytest.approx(0.1, abs=1e-4)

    def test_multiple_boxes(self) -> None:
        pred = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],
            [0.2, 0.3, 0.4, 0.5],
        ])
        tgt = torch.tensor([
            [0.0, 0.1, 0.5, 0.6],
            [0.2, 0.5, 0.4, 0.7],
        ])
        err = PositionError()(pred, tgt)
        # box 0: |(0,0) - (0,0.1)| = 0.1
        # box 1: |(0.2,0.3) - (0.2,0.5)| = 0.2
        # mean = 0.15
        assert err.item() == pytest.approx(0.15, abs=1e-4)

    def test_empty_pred(self) -> None:
        pred = torch.zeros(0, 4)
        tgt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        err = PositionError()(pred, tgt)
        assert err.item() == pytest.approx(0.0)

    def test_empty_target(self) -> None:
        pred = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        tgt = torch.zeros(0, 4)
        err = PositionError()(pred, tgt)
        assert err.item() == pytest.approx(0.0)

    def test_different_box_counts(self) -> None:
        pred = torch.rand(3, 4)
        tgt = torch.rand(5, 4)
        err = PositionError()(pred, tgt)
        # Should use min(3, 5) = 3 boxes
        assert err.item() >= 0.0


# ---------------------------------------------------------------------------
# SizeError
# ---------------------------------------------------------------------------


class TestSizeError:
    def test_perfect_match(self) -> None:
        b = _boxes(0.1, 0.2, 0.3, 0.4)  # w=0.2, h=0.2
        err = SizeError()(b, b)
        assert err.item() == pytest.approx(0.0, abs=1e-6)

    def test_size_diff(self) -> None:
        pred = _boxes(0.0, 0.0, 0.3, 0.4)  # w=0.3, h=0.4
        tgt = _boxes(0.0, 0.0, 0.5, 0.5)   # w=0.5, h=0.5
        err = SizeError()(pred, tgt)
        # size error = sqrt((0.3-0.5)^2 + (0.4-0.5)^2) = sqrt(0.05) ≈ 0.2236
        expected = (0.2 ** 2 + 0.1 ** 2) ** 0.5
        assert err.item() == pytest.approx(expected, abs=1e-4)

    def test_empty(self) -> None:
        err = SizeError()(torch.zeros(0, 4), torch.rand(3, 4))
        assert err.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _detect_alignments
# ---------------------------------------------------------------------------


class TestDetectAlignments:
    def test_no_elements(self) -> None:
        boxes = torch.zeros(0, 4)
        al = _detect_alignments(boxes)
        assert al == []

    def test_single_element(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        al = _detect_alignments(boxes)
        assert al == []

    def test_left_aligned(self) -> None:
        boxes = torch.tensor([
            [0.0, 0.1, 0.5, 0.3],  # x1=0.0
            [0.0, 0.5, 0.5, 0.7],  # x1=0.0 -> aligned left
        ])
        al = _detect_alignments(boxes, tolerance=0.01)
        names = {name for _, _, name in al}
        assert "align_left" in names

    def test_multiple_alignments(self) -> None:
        boxes = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],  # box 0
            [0.0, 0.5, 0.5, 0.7],  # box 1: left-aligned with 0
            [0.0, 0.8, 0.5, 1.0],  # box 2: left-aligned with 0,1
        ])
        al = _detect_alignments(boxes, tolerance=0.01)
        left_count = sum(1 for _, _, name in al if name == "align_left")
        # Pairs: (0,1), (0,2), (1,2) -> 3 left alignments
        assert left_count == 3

    def test_right_aligned(self) -> None:
        boxes = torch.tensor([
            [0.1, 0.0, 0.5, 0.3],
            [0.2, 0.5, 0.5, 0.7],  # x2=0.5 -> right-aligned
        ])
        al = _detect_alignments(boxes, tolerance=0.01)
        names = {name for _, _, name in al}
        assert "align_right" in names

    def test_center_aligned(self) -> None:
        boxes = torch.tensor([
            [0.0, 0.0, 0.4, 0.3],    # cx=0.2
            [0.05, 0.5, 0.35, 0.7],  # cx=0.2 -> center_x-aligned
        ])
        al = _detect_alignments(boxes, tolerance=0.01)
        names = {name for _, _, name in al}
        assert "center_x" in names


# ---------------------------------------------------------------------------
# AlignmentError
# ---------------------------------------------------------------------------


class TestAlignmentError:
    def test_perfect_alignment(self) -> None:
        boxes = torch.tensor([
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.5, 0.5, 0.7],
        ])
        err = AlignmentError(tolerance=0.01)(boxes, boxes)
        assert err.item() == pytest.approx(0.0, abs=1e-6)

    def test_deviated_alignment(self) -> None:
        # Target has left=0.0, right=0.5, center_x=0.25 for both boxes
        # → 3 alignment types detected
        tgt = torch.tensor([
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.5, 0.5, 0.7],
        ])
        # Prediction: only left alignment deviates (x1 shifts to 0.02)
        pred = torch.tensor([
            [0.0, 0.0, 0.5, 0.3],
            [0.02, 0.5, 0.5, 0.7],
        ])
        err = AlignmentError(tolerance=0.01)(pred, tgt)
        # 3 alignments: left dev=0.02, right dev=0.0, cx dev=0.01 → mean=0.01
        assert err.item() == pytest.approx(0.01, abs=1e-5)

    def test_empty_pred(self) -> None:
        pred = torch.zeros(0, 4)
        tgt = torch.rand(3, 4)
        err = AlignmentError()(pred, tgt)
        assert err.item() == pytest.approx(0.0)

    def test_empty_target(self) -> None:
        pred = torch.rand(3, 4)
        tgt = torch.zeros(0, 4)
        err = AlignmentError()(pred, tgt)
        assert err.item() == pytest.approx(0.0)

    def test_single_element(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        tgt = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        err = AlignmentError()(pred, tgt)
        assert err.item() == pytest.approx(0.0)

    def test_no_alignment_in_target(self) -> None:
        # Two boxes with no alignment relationships in target
        tgt = torch.tensor([
            [0.0, 0.0, 0.4, 0.4],
            [0.6, 0.6, 1.0, 1.0],  # far apart
        ])
        pred = torch.tensor([
            [0.0, 0.0, 0.4, 0.4],
            [0.6, 0.6, 1.0, 1.0],
        ])
        err = AlignmentError(tolerance=0.01)(pred, tgt)
        assert err.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ElementRecall
# ---------------------------------------------------------------------------


class TestElementRecall:
    def test_perfect(self) -> None:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        r = ElementRecall()(b, b)
        assert r.item() == pytest.approx(1.0)

    def test_none_matched(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        tgt = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
        r = ElementRecall()(pred, tgt)
        assert r.item() == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        pred = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.9, 0.9, 1.0, 1.0],
        ])
        tgt = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],   # matched by pred[0]
            [0.5, 0.5, 0.6, 0.6],   # no good match
        ])
        r = ElementRecall(iou_threshold=0.5)(pred, tgt)
        # GT[0] matched, GT[1] unmatched -> 0.5
        assert r.item() == pytest.approx(0.5)

    def test_empty(self) -> None:
        r = ElementRecall()(
            torch.zeros(0, 4), torch.rand(3, 4),
        )
        assert r.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ElementPrecision
# ---------------------------------------------------------------------------


class TestElementPrecision:
    def test_perfect(self) -> None:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        p = ElementPrecision()(b, b)
        assert p.item() == pytest.approx(1.0)

    def test_none_matched(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        tgt = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
        p = ElementPrecision()(pred, tgt)
        assert p.item() == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        pred = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.9, 0.9, 1.0, 1.0],
        ])
        tgt = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
        ])
        p = ElementPrecision(iou_threshold=0.5)(pred, tgt)
        # pred[0] matched, pred[1] unmatched -> 0.5
        assert p.item() == pytest.approx(0.5)

    def test_empty(self) -> None:
        p = ElementPrecision()(
            torch.zeros(0, 4), torch.rand(3, 4),
        )
        assert p.item() == pytest.approx(0.0)

    def test_empty_target(self) -> None:
        pred = torch.rand(5, 4)
        tgt = torch.zeros(0, 4)
        p = ElementPrecision()(pred, tgt)
        assert p.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F1Score
# ---------------------------------------------------------------------------


class TestF1Score:
    def test_perfect(self) -> None:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        f1 = F1Score()(b, b)
        assert f1.item() == pytest.approx(1.0)

    def test_zero(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        tgt = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
        f1 = F1Score()(pred, tgt)
        assert f1.item() == pytest.approx(0.0)

    def test_partial(self) -> None:
        pred = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.9, 0.9, 1.0, 1.0],
        ])
        tgt = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
        ])
        f1 = F1Score(iou_threshold=0.5)(pred, tgt)
        # recall = 1.0 (1/1 GT matched), precision = 0.5 (1/2 pred matched)
        # f1 = 2*1.0*0.5/(1.0+0.5) = 1.0/1.5 ≈ 0.6667
        assert f1.item() == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_empty(self) -> None:
        f1 = F1Score()(torch.zeros(0, 4), torch.rand(3, 4))
        assert f1.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MetricsBundle
# ---------------------------------------------------------------------------


class TestMetricsBundle:
    def test_defaults(self) -> None:
        mb = MetricsBundle()
        assert mb.recall == 0.0
        assert mb.precision == 0.0
        assert mb.f1 == 0.0

    def test_to_dict(self) -> None:
        mb = MetricsBundle(
            recall=0.9, precision=0.8, f1=0.85,
            position_error=0.01, size_error=0.02, alignment_error=0.03,
        )
        d = mb.to_dict()
        assert d["recall"] == 0.9
        assert d["f1"] == 0.85
        assert "alignment_error" in d

    def test_roundtrip(self) -> None:
        mb = MetricsBundle(
            recall=1.0, precision=0.5, f1=0.6667,
            position_error=0.1, size_error=0.05, alignment_error=0.03,
        )
        d = mb.to_dict()
        mb2 = MetricsBundle(**d)
        assert mb2.recall == mb.recall
        assert mb2.f1 == mb.f1


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_perfect_match(self) -> None:
        b = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.2, 0.2, 0.8, 0.8],
        ])
        result = compute_all_metrics(b, b)
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.position_error == pytest.approx(0.0)
        assert result.size_error == pytest.approx(0.0)

    def test_no_match(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        tgt = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
        result = compute_all_metrics(pred, tgt)
        assert result.recall == pytest.approx(0.0)
        assert result.precision == pytest.approx(0.0)
        assert result.f1 == pytest.approx(0.0)

    def test_empty_pred(self) -> None:
        pred = torch.zeros(0, 4)
        tgt = torch.rand(3, 4)
        result = compute_all_metrics(pred, tgt)
        assert result.recall == 0.0
        assert result.precision == 0.0
        assert result.f1 == 0.0

    def test_empty_target(self) -> None:
        pred = torch.rand(3, 4)
        tgt = torch.zeros(0, 4)
        result = compute_all_metrics(pred, tgt)
        assert result.recall == 0.0
        assert result.precision == 0.0

    def test_different_counts(self) -> None:
        pred = torch.rand(5, 4)
        tgt = torch.rand(3, 4)
        result = compute_all_metrics(pred, tgt)
        assert isinstance(result, MetricsBundle)
        assert 0.0 <= result.recall <= 1.0

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="2-d"):
            compute_all_metrics(torch.rand(4), torch.rand(3, 4))

    def test_invalid_last_dim(self) -> None:
        with pytest.raises(ValueError, match="last dim must be 4"):
            compute_all_metrics(torch.rand(3, 3), torch.rand(3, 4))

    def test_scalar_input(self) -> None:
        with pytest.raises(ValueError, match="2-d"):
            compute_all_metrics(torch.tensor(0.0), torch.rand(3, 4))

    def test_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-d"):
            compute_all_metrics(torch.rand(4), torch.rand(3, 4))
