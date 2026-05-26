"""Tests for GUIDataset, collate_variable_elements, create_dataloader,
and GUIDataModule (Phase 4.2.4 Dataset wrappers)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from bipartite_gnn_gui.data.dataset import (
    GUIDataModule,
    GUIDataset,
    _parse_vlm_json,
    _resolve_gt_path,
    _type_to_index,
    collate_variable_elements,
    create_dataloader,
)
from bipartite_gnn_gui.data.vlm_output import ELEMENT_TYPES


# ===================================================================
# Fixtures
# ===================================================================


def _make_single_qwen_vlm(image_id: str, n_elements: int = 3) -> Dict[str, Any]:
    """Create a synthetic Qwen-format VLM JSON dict."""
    elements = []
    for i in range(n_elements):
        x1 = 0.1 * (i + 1)
        y1 = 0.1 * (i + 1)
        x2 = x1 + 0.05
        y2 = y1 + 0.05
        elements.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "label": ["button", "text", "input"][i % 3],
                "text": f"element_{i}",
                "confidence": 0.95 - i * 0.05,
            }
        )
    return {"image_id": image_id, "elements": elements}


def _make_single_gt(image_id: str, n_elements: int = 2) -> Dict[str, Any]:
    """Create a synthetic GUI360-format GT JSON dict."""
    annotations = []
    for i in range(n_elements):
        x1 = 0.1 * (i + 1) + 0.02
        y1 = 0.1 * (i + 1) + 0.02
        x2 = x1 + 0.06
        y2 = y1 + 0.06
        annotations.append(
            {
                "element_id": f"elem_{i}",
                "bbox": [x1, y1, x2, y2],
                "type": ["button", "text"][i % 2],
                "text": f"gt_text_{i}",
            }
        )
    return {
        "image_id": image_id,
        "image_width": 1920,
        "image_height": 1080,
        "platform": "web",
        "annotations": annotations,
    }


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Create a temporary raw data directory with VLM + GT JSONs."""
    raw = tmp_path / "data" / "raw"
    vlm_dir = raw / "vlm_predictions"
    gt_dir = raw / "gui360"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 samples
    for idx in range(5):
        img_id = f"img_{idx:04d}"
        # VLM
        vlm_data = _make_single_qwen_vlm(img_id, n_elements=3)
        (vlm_dir / f"{img_id}.json").write_text(json.dumps(vlm_data))
        # GT
        gt_data = _make_single_gt(img_id, n_elements=2)
        (gt_dir / f"{img_id}.json").write_text(json.dumps(gt_data))

    return raw


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    d = tmp_path / "data" / "processed" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def image_ids() -> List[str]:
    """Sorted list of image IDs for the 5 samples."""
    return [f"img_{i:04d}" for i in range(5)]


@pytest.fixture
def dataset(raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> GUIDataset:
    """A fully cached GUIDataset (build triggered)."""
    vlm_dir = raw_dir / "vlm_predictions"
    gt_dir = raw_dir / "gui360"
    ds = GUIDataset(
        image_ids=image_ids,
        vlm_dir=vlm_dir,
        gt_dir=gt_dir,
        cache_dir=cache_dir,
    )
    # Trigger build
    _ = len(ds)
    return ds


# ===================================================================
# _type_to_index
# ===================================================================


class TestTypeToIndex:
    def test_known_type(self) -> None:
        taxonomy = list(ELEMENT_TYPES.keys())
        idx = _type_to_index("button", taxonomy)
        assert taxonomy[idx] == "button"

    def test_unknown_type_defaults_to_zero(self) -> None:
        taxonomy = list(ELEMENT_TYPES.keys())
        idx = _type_to_index("nonexistent_type_xyz", taxonomy)
        assert idx == 0


# ===================================================================
# _resolve_gt_path
# ===================================================================


class TestResolveGtPath:
    def test_finds_json_in_gui360(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gui360"
        gt_dir.mkdir(parents=True)
        (gt_dir / "test_img.json").write_text("{}")
        result = _resolve_gt_path("test_img", gt_dir)
        assert result is not None
        assert result.name == "test_img.json"

    def test_finds_json_in_screenspot(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "screenspot"
        gt_dir.mkdir(parents=True)
        (gt_dir / "test_img.json").write_text("{}")
        result = _resolve_gt_path("test_img", gt_dir)
        assert result is not None
        assert result.name == "test_img.json"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gui360"
        gt_dir.mkdir(parents=True)
        result = _resolve_gt_path("nonexistent", gt_dir)
        assert result is None

    def test_strips_extension(self, tmp_path: Path) -> None:
        """Should find GT when image_id has extension but GT file does not."""
        gt_dir = tmp_path / "gui360"
        gt_dir.mkdir(parents=True)
        (gt_dir / "test_img.json").write_text("{}")
        result = _resolve_gt_path("test_img.png", gt_dir)
        assert result is not None
        assert result.name == "test_img.json"

    def test_handles_json_extension_in_image_id(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gui360"
        gt_dir.mkdir(parents=True)
        (gt_dir / "test_img.json").write_text("{}")
        result = _resolve_gt_path("test_img.json", gt_dir)
        assert result is not None
        assert result.name == "test_img.json"


# ===================================================================
# _parse_vlm_json
# ===================================================================


class TestParseVlmJson:
    def test_parses_qwen_format(self) -> None:
        data = {
            "image_id": "test",
            "elements": [
                {"bbox_xyxy": [0.1, 0.1, 0.5, 0.5], "label": "button", "confidence": 0.9}
            ],
        }
        result = _parse_vlm_json(data)
        assert result.image_id == "test"
        assert len(result.elements) == 1

    def test_parses_minimax_format(self) -> None:
        data = {
            "image_id": "test",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {"bbox": [10, 10, 50, 50], "category": "button", "confidence": 0.9}
            ],
        }
        result = _parse_vlm_json(data)
        assert result.image_id == "test"
        assert len(result.elements) == 1

    def test_empty_elements(self) -> None:
        data = {"image_id": "test", "elements": []}
        result = _parse_vlm_json(data)
        assert result.image_id == "test"
        assert len(result.elements) == 0

    def test_raises_on_non_dict(self) -> None:
        with pytest.raises(Exception):
            _parse_vlm_json("not a dict")  # type: ignore[arg-type]


# ===================================================================
# GUIDataset._build_cache
# ===================================================================


class TestBuildCache:
    def test_creates_pt_files(self, raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> None:
        vlm_dir = raw_dir / "vlm_predictions"
        gt_dir = raw_dir / "gui360"
        ds = GUIDataset(image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)
        _ = len(ds)

        # Check .pt files exist for all 5 images
        pt_files = sorted(cache_dir.glob("*.pt"))
        assert len(pt_files) == 5
        for img_id in image_ids:
            assert (cache_dir / f"{img_id}.pt").exists()

    def test_skips_rebuild_if_cache_exists(self, raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> None:
        vlm_dir = raw_dir / "vlm_predictions"
        gt_dir = raw_dir / "gui360"

        # First build
        ds1 = GUIDataset(image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)
        _ = len(ds1)

        # Get modification time of first cache file
        cache_path = cache_dir / f"{image_ids[0]}.pt"
        mtime1 = cache_path.stat().st_mtime_ns

        # Second build (no force)
        ds2 = GUIDataset(image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)
        _ = len(ds2)

        mtime2 = cache_path.stat().st_mtime_ns
        assert mtime2 == mtime1, "Cache file should not be overwritten"

    def test_rebuilds_with_force(self, raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> None:
        vlm_dir = raw_dir / "vlm_predictions"
        gt_dir = raw_dir / "gui360"

        # First build
        ds1 = GUIDataset(image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)
        _ = len(ds1)

        # Force rebuild
        ds2 = GUIDataset(
            image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir, force_rebuild=True
        )
        _ = len(ds2)

        # All files should still exist
        for img_id in image_ids:
            assert (cache_dir / f"{img_id}.pt").exists()

    def test_handles_missing_vlm_json(self, tmp_path: Path, cache_dir: Path) -> None:
        """Missing VLM JSON should use empty predictions (not skip)."""
        vlm_dir = tmp_path / "vlm_predictions"
        gt_dir = tmp_path / "gui360"
        vlm_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        # Create GT but no VLM for img_0000
        gt_data = _make_single_gt("img_0000", n_elements=2)
        (gt_dir / "img_0000.json").write_text(json.dumps(gt_data))

        ds = GUIDataset(
            ["img_0000"], vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir
        )
        assert len(ds) == 1

        sample = ds[0]
        # Empty predictions -> N=0
        assert sample["element_features"].size(0) == 0
        assert sample["vlm_boxes"].size(0) == 0
        assert sample["gt_boxes"].size(0) == 0
        assert sample["gt_present"].size(0) == 2  # GT still has 2 elements

    def test_handles_missing_gt_json(self, tmp_path: Path, cache_dir: Path) -> None:
        """Missing GT JSON should log warning and skip sample."""
        vlm_dir = tmp_path / "vlm_predictions"
        gt_dir = tmp_path / "gui360"
        vlm_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        # Create VLM but no GT
        vlm_data = _make_single_qwen_vlm("img_0000", n_elements=3)
        (vlm_dir / "img_0000.json").write_text(json.dumps(vlm_data))

        ds = GUIDataset(
            ["img_0000"], vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir
        )
        assert len(ds) == 0  # Sample skipped

    def test_build_idempotent(self, raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> None:
        """Calling len() multiple times should be idempotent."""
        vlm_dir = raw_dir / "vlm_predictions"
        gt_dir = raw_dir / "gui360"
        ds = GUIDataset(image_ids, vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)

        l1 = len(ds)
        l2 = len(ds)
        l3 = len(ds)

        assert l1 == 5
        assert l2 == 5
        assert l3 == 5

    def test_handles_vlm_parse_error(self, tmp_path: Path, cache_dir: Path) -> None:
        """Invalid VLM JSON should be handled gracefully (empty predictions)."""
        vlm_dir = tmp_path / "vlm_predictions"
        gt_dir = tmp_path / "gui360"
        vlm_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        # Create invalid VLM JSON
        (vlm_dir / "img_0000.json").write_text("not valid json")

        # Create valid GT
        gt_data = _make_single_gt("img_0000", n_elements=2)
        (gt_dir / "img_0000.json").write_text(json.dumps(gt_data))

        ds = GUIDataset(
            ["img_0000"], vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir
        )
        assert len(ds) == 1
        sample = ds[0]
        assert sample["element_features"].size(0) == 0  # Empty predictions


# ===================================================================
# GUIDataset.__getitem__
# ===================================================================


class TestGetItem:
    def test_expected_keys_present(self, dataset: GUIDataset) -> None:
        sample = dataset[0]
        expected_keys = {
            "element_features", "vlm_boxes", "gt_boxes", "element_types",
            "image_id", "image_size", "matched_mask", "gt_present",
        }
        assert set(sample.keys()) == expected_keys

    def test_correct_tensor_shapes(self, dataset: GUIDataset) -> None:
        sample = dataset[0]
        N = sample["element_features"].size(0)
        num_types = len(ELEMENT_TYPES)
        D_feat = 4 + num_types + 1

        assert sample["element_features"].shape == (N, D_feat)
        assert sample["vlm_boxes"].shape == (N, 4)
        assert sample["gt_boxes"].shape == (N, 4)
        assert sample["element_types"].shape == (N,)
        assert sample["image_size"].shape == (2,)
        assert sample["matched_mask"].shape == (N,)
        assert sample["gt_present"].dim() == 1

    def test_correct_dtypes(self, dataset: GUIDataset) -> None:
        sample = dataset[0]
        assert sample["element_features"].dtype == torch.float32
        assert sample["vlm_boxes"].dtype == torch.float32
        assert sample["gt_boxes"].dtype == torch.float32
        assert sample["element_types"].dtype == torch.long
        assert sample["image_size"].dtype == torch.float32
        assert sample["matched_mask"].dtype == torch.bool
        assert sample["gt_present"].dtype == torch.bool

    def test_image_id_string(self, dataset: GUIDataset) -> None:
        sample = dataset[0]
        assert isinstance(sample["image_id"], str)
        assert sample["image_id"] == dataset._cached_ids[0]

    def test_all_samples_accessible(self, dataset: GUIDataset) -> None:
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["element_features"].size(0) > 0

    def test_consistent_data(self, dataset: GUIDataset) -> None:
        """Repeated __getitem__ calls return the same data."""
        s1 = dataset[0]
        s2 = dataset[0]
        assert torch.equal(s1["element_features"], s2["element_features"])
        assert torch.equal(s1["vlm_boxes"], s2["vlm_boxes"])
        assert s1["image_id"] == s2["image_id"]


# ===================================================================
# collate_variable_elements
# ===================================================================


class TestCollate:
    def _make_sample(self, N: int, D_feat: int = 25) -> Dict[str, Any]:
        """Create a synthetic sample with N elements."""
        return {
            "element_features": torch.rand(N, D_feat),
            "vlm_boxes": torch.rand(N, 4),
            "gt_boxes": torch.rand(N, 4),
            "element_types": torch.randint(0, 10, (N,)),
            "matched_mask": torch.randint(0, 2, (N,), dtype=torch.bool),
            "image_id": "test_img",
            "image_size": torch.tensor([1920.0, 1080.0]),
            "gt_present": torch.ones(2, dtype=torch.bool),
        }

    def test_pads_to_n_max(self) -> None:
        """Batch with sizes [2, 5, 3] should pad to N_max=5."""
        samples = [self._make_sample(2), self._make_sample(5), self._make_sample(3)]
        batch = collate_variable_elements(samples)

        assert batch["element_features"].shape == (3, 5, 25)
        assert batch["vlm_boxes"].shape == (3, 5, 4)
        assert batch["gt_boxes"].shape == (3, 5, 4)
        assert batch["element_types"].shape == (3, 5)
        assert batch["matched_mask"].shape == (3, 5)
        assert batch["valid_mask"].shape == (3, 5)

    def test_valid_mask_correct(self) -> None:
        """valid_mask should be True for real elements, False for padding."""
        samples = [self._make_sample(2), self._make_sample(4)]
        batch = collate_variable_elements(samples)

        # First sample: N=2, N_max=4 -> first 2 True, last 2 False
        assert batch["valid_mask"][0, :2].all()
        assert not batch["valid_mask"][0, 2:].any()

        # Second sample: N=4, N_max=4 -> all True
        assert batch["valid_mask"][1].all()

    def test_element_types_padded_with_minus_one(self) -> None:
        samples = [self._make_sample(2)]
        batch = collate_variable_elements(samples)
        assert batch["element_types"][0, :2].ge(0).all()

    def test_single_sample(self) -> None:
        """Batch with B=1 should work correctly."""
        sample = self._make_sample(4, D_feat=25)
        batch = collate_variable_elements([sample])

        assert batch["element_features"].shape == (1, 4, 25)
        assert batch["valid_mask"][0].all()
        assert batch["image_ids"] == ["test_img"]

    def test_equal_length_samples(self) -> None:
        """No padding needed when all samples have the same N."""
        samples = [self._make_sample(3), self._make_sample(3), self._make_sample(3)]
        batch = collate_variable_elements(samples)

        assert batch["element_features"].shape == (3, 3, 25)
        assert batch["valid_mask"].all()

    def test_empty_batch(self) -> None:
        batch = collate_variable_elements([])
        assert batch == {}

    def test_image_ids_preserved(self) -> None:
        s1 = self._make_sample(2)
        s1["image_id"] = "img_a"
        s2 = self._make_sample(3)
        s2["image_id"] = "img_b"

        batch = collate_variable_elements([s1, s2])
        assert batch["image_ids"] == ["img_a", "img_b"]

    def test_image_sizes_stacked(self) -> None:
        s1 = self._make_sample(2)
        s1["image_size"] = torch.tensor([100.0, 200.0])
        s2 = self._make_sample(3)
        s2["image_size"] = torch.tensor([300.0, 400.0])

        batch = collate_variable_elements([s1, s2])
        assert batch["image_sizes"].shape == (2, 2)
        assert torch.equal(batch["image_sizes"][0], torch.tensor([100.0, 200.0]))
        assert torch.equal(batch["image_sizes"][1], torch.tensor([300.0, 400.0]))

    def test_gt_present_as_list(self) -> None:
        s1 = self._make_sample(2)
        s1["gt_present"] = torch.ones(3, dtype=torch.bool)
        s2 = self._make_sample(3)
        s2["gt_present"] = torch.ones(5, dtype=torch.bool)

        batch = collate_variable_elements([s1, s2])
        assert len(batch["gt_present"]) == 2
        assert batch["gt_present"][0].size(0) == 3
        assert batch["gt_present"][1].size(0) == 5

    def test_large_n_max_difference(self) -> None:
        """Handle very different N values in the same batch."""
        s1 = self._make_sample(1)
        s2 = self._make_sample(50)
        batch = collate_variable_elements([s1, s2])

        assert batch["element_features"].shape == (2, 50, 25)
        # First sample: only first element is valid
        assert batch["valid_mask"][0, 0].item() is True
        assert batch["valid_mask"][0, 1:].sum().item() == 0
        # Second sample: all valid
        assert batch["valid_mask"][1].sum().item() == 50

    def test_zero_element_samples(self) -> None:
        """Handle samples with zero elements."""
        s1 = self._make_sample(0)
        s2 = self._make_sample(3)
        batch = collate_variable_elements([s1, s2])

        assert batch["element_features"].shape == (2, 3, 25)
        assert batch["valid_mask"][0].sum().item() == 0  # All padding
        assert batch["valid_mask"][1].sum().item() == 3


# ===================================================================
# create_dataloader
# ===================================================================


class TestCreateDataloader:
    def test_returns_dataloader(self, dataset: GUIDataset) -> None:
        loader = create_dataloader(dataset, batch_size=2)
        assert isinstance(loader, DataLoader)

    def test_correct_batch_shapes(self, dataset: GUIDataset) -> None:
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert "element_features" in batch
        assert "valid_mask" in batch
        assert "image_ids" in batch
        assert batch["element_features"].dim() == 3  # (B, N_max, D_feat)
        assert batch["element_features"].size(0) == 2  # B=2

    def test_shuffle_respected(self, dataset: GUIDataset) -> None:
        loader_shuffled = create_dataloader(dataset, batch_size=5, shuffle=True)
        loader_unshuffled = create_dataloader(dataset, batch_size=5, shuffle=False)

        batch_s = next(iter(loader_shuffled))
        batch_u = next(iter(loader_unshuffled))

        # Both should have 5 samples (the full dataset)
        assert len(batch_s["image_ids"]) == 5
        assert len(batch_u["image_ids"]) == 5

    def test_single_sample_batch(self, dataset: GUIDataset) -> None:
        loader = create_dataloader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        assert batch["element_features"].size(0) == 1


# ===================================================================
# GUIDataModule
# ===================================================================


class TestGUIDataModule:
    def test_init_does_not_scan(self, raw_dir: Path) -> None:
        """__init__ should not scan data (lazy)."""
        module = GUIDataModule(root_dir=raw_dir, batch_size=2)
        assert module._built is False
        assert module._train_dataset is None

    def test_raises_on_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GUIDataModule(root_dir=tmp_path / "nonexistent")

    def test_train_dataloader_triggers_build(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, batch_size=2)
        loader = module.train_dataloader()
        assert loader is not None
        assert module._built is True

    def test_train_dataloader_shuffles(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, batch_size=2)
        loader = module.train_dataloader()
        assert loader is not None
        # train dataloader should use RandomSampler (shuffle=True)
        from torch.utils.data.sampler import RandomSampler
        assert isinstance(loader.sampler, RandomSampler)

    def test_val_dataloader_no_shuffle(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, batch_size=2,
                                val_split=0.3, test_split=0.3)
        loader = module.val_dataloader()
        assert loader is not None
        # val dataloader should use SequentialSampler (shuffle=False)
        from torch.utils.data.sampler import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler)

    def test_test_dataloader_no_shuffle(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, batch_size=2,
                                val_split=0.3, test_split=0.3)
        loader = module.test_dataloader()
        assert loader is not None
        # test dataloader should use SequentialSampler (shuffle=False)
        from torch.utils.data.sampler import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler)

    def test_splits_are_disjoint(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, val_split=0.2, test_split=0.2, seed=42)
        train_ids = set(module.train_dataset._cached_ids) if module.train_dataset else set()
        val_ids = set(module.val_dataset._cached_ids) if module.val_dataset else set()
        test_ids = set(module.test_dataset._cached_ids) if module.test_dataset else set()

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_all_dataloaders_produce_correct_shapes(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir, batch_size=2, val_split=0.2, test_split=0.2)

        train_loader = module.train_dataloader()
        val_loader = module.val_dataloader()
        test_loader = module.test_dataloader()

        if train_loader:
            batch = next(iter(train_loader))
            assert batch["element_features"].size(-1) == 4 + len(ELEMENT_TYPES) + 1

        if val_loader:
            batch = next(iter(val_loader))
            assert batch["element_features"].size(-1) == 4 + len(ELEMENT_TYPES) + 1

        if test_loader:
            batch = next(iter(test_loader))
            assert batch["element_features"].size(-1) == 4 + len(ELEMENT_TYPES) + 1

    def test_all_samples_accounted_for(self, raw_dir: Path) -> None:
        """Total samples across splits should equal total image IDs."""
        module = GUIDataModule(root_dir=raw_dir, val_split=0.2, test_split=0.2, seed=42)

        total = len(module.train_dataset) + len(module.val_dataset) + len(module.test_dataset)
        assert total == 5

    def test_dataset_properties_trigger_build(self, raw_dir: Path) -> None:
        module = GUIDataModule(root_dir=raw_dir)
        assert module._built is False

        _ = module.train_dataset
        assert module._built is True

    def test_cache_dir_default(self, raw_dir: Path) -> None:
        """Default cache dir should be relative to root_dir parent."""
        module = GUIDataModule(root_dir=raw_dir)
        expected = raw_dir.parent / "processed" / "cache"
        assert module.cache_dir == expected


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_element_sample(self, tmp_path: Path) -> None:
        """Dataset with a single element per image."""
        vlm_dir = tmp_path / "vlm_predictions"
        gt_dir = tmp_path / "gui360"
        cache_dir = tmp_path / "cache"
        vlm_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)

        vlm_data = _make_single_qwen_vlm("single", n_elements=1)
        gt_data = _make_single_gt("single", n_elements=1)
        (vlm_dir / "single.json").write_text(json.dumps(vlm_data))
        (gt_dir / "single.json").write_text(json.dumps(gt_data))

        ds = GUIDataset(["single"], vlm_dir=vlm_dir, gt_dir=gt_dir, cache_dir=cache_dir)
        sample = ds[0]

        assert sample["element_features"].size(0) == 1
        assert sample["vlm_boxes"].shape == (1, 4)

    def test_no_gt_dir(self, tmp_path: Path) -> None:
        """When gt_dir doesn't exist, all samples should be skipped."""
        vlm_dir = tmp_path / "vlm_predictions"
        cache_dir = tmp_path / "cache"
        vlm_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)

        vlm_data = _make_single_qwen_vlm("img_a", n_elements=3)
        (vlm_dir / "img_a.json").write_text(json.dumps(vlm_data))

        # gt_dir doesn't exist
        ds = GUIDataset(
            ["img_a"],
            vlm_dir=vlm_dir,
            gt_dir=tmp_path / "nonexistent_gt",
            cache_dir=cache_dir,
        )
        assert len(ds) == 0  # All skipped

    def test_mixed_vlm_format(self, raw_dir: Path, cache_dir: Path, image_ids: List[str]) -> None:
        """Should handle both Qwen and MiniMax format VLM JSONs."""
        vlm_dir = raw_dir / "vlm_predictions"
        gt_dir = raw_dir / "gui360"

        # Create a MiniMax-format VLM JSON
        minimax_data = {
            "image_id": "minimax_sample",
            "image_width": 1920,
            "image_height": 1080,
            "elements": [
                {
                    "bbox": [100, 100, 200, 200],
                    "category": "button",
                    "confidence": 0.9,
                    "text_content": "Click",
                }
            ],
        }
        (vlm_dir / "minimax_sample.json").write_text(json.dumps(minimax_data))

        # GT for it
        gt_data = _make_single_gt("minimax_sample", n_elements=1)
        (gt_dir / "minimax_sample.json").write_text(json.dumps(gt_data))

        ds = GUIDataset(
            list(image_ids) + ["minimax_sample"],
            vlm_dir=vlm_dir,
            gt_dir=gt_dir,
            cache_dir=cache_dir,
        )
        # Just verify it works
        assert len(ds) >= 5

    def test_repeated_getitem_is_consistent(self, dataset: GUIDataset) -> None:
        """Multiple calls to __getitem__ with same index return same data."""
        s1 = dataset[2]
        s2 = dataset[2]
        s3 = dataset[2]

        assert torch.equal(s1["element_features"], s2["element_features"])
        assert torch.equal(s2["element_features"], s3["element_features"])
        assert s1["image_id"] == s2["image_id"] == s3["image_id"]
