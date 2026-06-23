"""Phase 5B — Integration tests for the structural completion pipeline (Phase 4.9)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bipartite_gnn_gui.data.masking import random_mask
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.heads import ElementProposalHead
from bipartite_gnn_gui.model.losses import compute_proposal_loss

ROOT = Path(__file__).resolve().parent.parent


# ── helpers ─────────────────────────────────────────────────────────────────


def _elem(x1: float, y1: float, x2: float, y2: float,
          label: str = "button") -> ElementNode:
    return ElementNode(bbox=[x1, y1, x2, y2], confidence=1.0, label=label)


def _make_minimal_layout() -> tuple[list[ElementNode], BipartiteGraphBuilder]:
    """Return a small layout with guaranteed constraints between elements."""
    elems = [
        _elem(0.0, 0.0, 0.3, 0.1, "button"),   # row 1 left
        _elem(0.35, 0.0, 0.65, 0.1, "button"),  # row 1 middle
        _elem(0.7, 0.0, 1.0, 0.1, "button"),    # row 1 right
        _elem(0.0, 0.2, 0.3, 0.35, "text"),      # row 2 left
        _elem(0.35, 0.2, 0.65, 0.35, "text"),    # row 2 middle
    ]
    builder = BipartiteGraphBuilder()
    return elems, builder


# ── 5B.1: build_violation_graph() ──────────────────────────────────────────


class TestBuildViolationGraph5B1:
    """Test the core data pipeline for structural completion."""

    def _call_build(self, elems, drop_ratio, seed=42):
        from scripts.train_violation import build_violation_graph
        builder = BipartiteGraphBuilder()
        return build_violation_graph(elems, builder, drop_ratio=drop_ratio, seed=seed)

    def test_drop_zero_no_violations(self) -> None:
        elems, _ = _make_minimal_layout()
        result = self._call_build(elems, drop_ratio=0.0)
        assert result is not None, "build returned None for drop=0"
        _, targets = result
        assert targets["violation"].sum().item() == 0.0, "expected no violations"
        assert targets["proposal_violation_mask"].sum().item() == 0

    def test_drop_all_returns_none(self) -> None:
        elems, _ = _make_minimal_layout()
        result = self._call_build(elems, drop_ratio=1.0)
        assert result is None, "expected None for drop=1"

    def test_drop_half_has_violations(self) -> None:
        elems, _ = _make_minimal_layout()
        result = self._call_build(elems, drop_ratio=0.5)
        assert result is not None
        data, targets = result

        # Some constraints should be violated.
        n_violated = targets["proposal_violation_mask"].sum().item()
        assert n_violated > 0, f"expected violations, got {n_violated}"

        # Proposal targets have correct shape and range.
        prop_tgt = targets["proposal_target"]
        assert prop_tgt.shape == (data["constraint"].x.shape[0], 5), \
            f"expected (N_con, 5), got {prop_tgt.shape}"
        assert prop_tgt[:, :4].min() >= 0.0
        assert prop_tgt[:, :4].max() <= 1.0
        # Type column (dim 4) should be in [0, N_TYPES)
        assert prop_tgt[:, 4].min() >= 0
        assert prop_tgt[:, 4].max() < 8

    def test_constraint_indices_are_valid(self) -> None:
        elems, _ = _make_minimal_layout()
        result = self._call_build(elems, drop_ratio=0.5)
        assert result is not None
        data, targets = result

        N_surv = data["element"].x.shape[0]
        edge = data["element", "to", "constraint"].edge_index
        # All indices should point to existing nodes.
        assert edge[0].max() < N_surv, "element index out of range"
        assert edge[1].max() < data["constraint"].x.shape[0]

    def test_survivor_count_matches_mask(self) -> None:
        elems, _ = _make_minimal_layout()
        result = self._call_build(elems, drop_ratio=0.5)
        assert result is not None
        data, _ = result
        expected = len(elems) // 2  # ~0.5 of 5 = 2-3
        assert abs(data["element"].x.shape[0] - expected) <= 1

    def test_very_small_layout_returns_none(self) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.55, 0.0, 1.0, 0.5)]
        result = self._call_build(elems, drop_ratio=0.5)
        # 2 elements, drop 0.5 → expected 1 survivor, need ≥2 for graph
        assert result is None


# ── 5B.2: random_mask() ───────────────────────────────────────────────────


class TestRandomMask5B2:
    """Test the feature masking pipeline."""

    def _build_full_graph(self):
        elems, _ = _make_minimal_layout()
        constraints = extract_all_constraints(elems)
        builder = BipartiteGraphBuilder()
        return builder.build(elems, constraints)

    def test_mask_exact_fraction(self) -> None:
        data = self._build_full_graph()
        N = data["element"].x.shape[0]
        masked, info = random_mask(data, mask_ratio=0.6, seed=42)
        actual_masked = info["mask"].sum().item()
        assert abs(actual_masked - round(N * 0.6)) <= 1, \
            f"expected ~{round(N * 0.6)} masked, got {actual_masked}"

    def test_masked_features_are_mask_token(self) -> None:
        data = self._build_full_graph()
        masked, info = random_mask(data, mask_ratio=0.6, seed=42)
        mask = info["mask"]
        masked_feats = masked["element"].x[mask]
        assert torch.allclose(masked_feats, torch.full_like(masked_feats, -1.0)), \
            "masked features should be MASK_TOKEN (-1.0)"

    def test_unmasked_features_preserved(self) -> None:
        data = self._build_full_graph()
        original = data["element"].x.clone()
        masked, info = random_mask(data, mask_ratio=0.6, seed=42)
        unmask = ~info["mask"]
        assert torch.allclose(masked["element"].x[unmask], original[unmask]), \
            "unmasked features should be unchanged"

    def test_target_features_match_original(self) -> None:
        data = self._build_full_graph()
        original = data["element"].x.clone()
        masked, info = random_mask(data, mask_ratio=0.6, seed=42)
        mask = info["mask"]
        assert torch.allclose(info["target"][mask], original[mask]), \
            "target features should equal original"

    def test_mask_ratio_zero_does_nothing(self) -> None:
        data = self._build_full_graph()
        masked, info = random_mask(data, mask_ratio=0.0)
        assert info["mask"].sum().item() == 0
        assert torch.allclose(masked["element"].x, data["element"].x)

    def test_mask_ratio_one_masks_all(self) -> None:
        data = self._build_full_graph()
        N = data["element"].x.shape[0]
        masked, info = random_mask(data, mask_ratio=1.0)
        assert info["mask"].sum().item() == N
        assert torch.allclose(
            masked["element"].x,
            torch.full_like(masked["element"].x, -1.0)
        )


# ── 5B.3: ElementProposalHead + compute_proposal_loss ─────────────────────


class TestElementProposalHead5B3:
    """Test the proposal head forward pass and loss computation."""

    def test_forward_output_shape_and_range(self) -> None:
        head = ElementProposalHead(input_dim=32)
        x = torch.randn(10, 32)
        out = head(x)
        assert out.shape == (10, 12), f"expected (10, 12), got {out.shape}"
        # First 4 dims: sigmoided bbox in [0, 1]
        assert out[:, :4].min() >= 0.0, f"min {out[:, :4].min().item()} < 0"
        assert out[:, :4].max() <= 1.0, f"max {out[:, :4].max().item()} > 1"
        # Last 8 dims: raw logits (unconstrained)
        assert out[:, 4:].shape == (10, 8)

    def test_loss_on_violated_constraints(self) -> None:
        pred = torch.rand(8, 4)
        tgt = torch.rand(8, 4)
        mask = torch.tensor([True, True, False, False, False, False, False, False])
        loss = compute_proposal_loss(pred, tgt, mask)
        assert loss.item() > 0.0
        # Check loss is computed only on the 2 violated entries.
        expected = torch.nn.functional.mse_loss(pred[mask], tgt[mask])
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_loss_returns_zero_when_no_violations(self) -> None:
        pred = torch.rand(8, 4)
        tgt = torch.rand(8, 4)
        mask = torch.zeros(8, dtype=torch.bool)
        loss = compute_proposal_loss(pred, tgt, mask)
        assert loss.item() == 0.0

    def test_loss_gradient_flows_only_to_violated(self) -> None:
        head = ElementProposalHead(input_dim=16)
        x = torch.randn(6, 16, requires_grad=True)
        out = head(x)
        tgt = torch.rand(6, 4)
        mask = torch.tensor([True, False, False, False, False, False])

        loss = compute_proposal_loss(out, tgt, mask)
        loss.backward()

        assert x.grad is not None
        # Gradient norm should be non-zero.
        assert x.grad.norm().item() > 0


# ── 5B.4: train_violation.py smoke test ────────────────────────────────────


@pytest.mark.slow
class TestTrainViolationSmoke5B4:
    """Run train_violation.py with minimal settings to catch crashes."""

    @pytest.fixture(scope="class")
    def rico_dir(self) -> Path:
        p = ROOT / "data" / "rico_local" / "combined"
        if not p.is_dir():
            pytest.skip(f"RICO data not found at {p}")
        return p

    def test_smoke_train(self, rico_dir: Path) -> None:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_violation.py"),
            "--n", "10",
            "--epochs", "2",
            "--hidden", "16",
            "--drop-ratio", "0.5",
            "--rico-dir", str(rico_dir),
            "--log-level", "ERROR",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, \
            f"train_violation.py failed:\n{result.stderr[-500:]}"


# ── 5B.5: evaluate_completion.py smoke test ────────────────────────────────


@pytest.mark.slow
class TestEvaluateCompletionSmoke5B5:
    """Run evaluate_completion.py with minimal settings."""

    @pytest.fixture(scope="class")
    def rico_dir(self) -> Path:
        p = ROOT / "data" / "rico_local" / "combined"
        if not p.is_dir():
            pytest.skip(f"RICO data not found at {p}")
        return p

    def test_smoke_evaluate(self, rico_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "test_eval.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_completion.py"),
            "--n", "10",
            "--epochs", "2",
            "--hidden", "16",
            "--drop-ratios", "0.4,0.6",
            "--seeds", "42",
            "--rico-dir", str(rico_dir),
            "--output", str(out),
            "--log-level", "ERROR",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, \
            f"evaluate_completion.py failed:\n{result.stderr[-500:]}"
        assert out.exists(), "output JSON not created"


# ── 5B.6: Baseline correctness ────────────────────────────────────────────


class TestBaselineCorrectness5B6:
    """Verify baselines in evaluate_completion.py produce valid results."""

    def test_nearest_neighbor_with_single_survivor(self) -> None:
        from scripts.evaluate_completion import baseline_nearest_neighbor
        targets = {
            "gt_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),  # xywh
            "proposal_target": torch.tensor([[0.0, 0.0, 0.01, 0.01]]),
            "proposal_violation_mask": torch.tensor([True]),
        }
        result = baseline_nearest_neighbor(targets)
        assert 0.0 <= result["mse"] <= 2.0, f"unexpected MSE: {result['mse']}"

    def test_center_baseline_returns_valid_metrics(self) -> None:
        from scripts.evaluate_completion import baseline_center
        targets = {
            "proposal_target": torch.tensor([[0.3, 0.3, 0.5, 0.5]]),
            "proposal_violation_mask": torch.tensor([True]),
        }
        result = baseline_center(targets, img_size=(1.0, 1.0))  # normalized coords
        assert 0.0 <= result["mse"] <= 2.0
        assert 0.0 <= result["iou"] <= 1.0

    def test_all_baselines_non_negative(self) -> None:
        from scripts.evaluate_completion import baseline_nearest_neighbor, baseline_center
        targets = {
            "gt_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            "proposal_target": torch.tensor([[0.3, 0.3, 0.5, 0.5]]),
            "proposal_violation_mask": torch.tensor([True]),
        }
        for fn in (baseline_nearest_neighbor, baseline_center):
            r = fn(targets)
            for v in r.values():
                assert v >= 0.0, f"{fn.__name__} returned negative: {v}"
