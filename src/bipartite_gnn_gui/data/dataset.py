"""Dataset wrappers for GUI correction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import DataLoader, Dataset


class GUIDataset(Dataset):
    """Simple dataset pairing VLM output with ground truth."""

    def __init__(self, samples: Sequence[dict[str, Any]] | None = None) -> None:
        self.samples = list(samples or [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_variable_elements(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-size samples as a list."""

    return batch


def create_dataloader(dataset: Dataset, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """Create a DataLoader for variable-size GUI samples."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_elements)


@dataclass
class GUIDataModule:
    """Lightweight container for train/val/test loaders."""

    train_dataset: GUIDataset | None = None
    val_dataset: GUIDataset | None = None
    test_dataset: GUIDataset | None = None
    batch_size: int = 1

    def train_dataloader(self) -> DataLoader | None:
        return None if self.train_dataset is None else create_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        return None if self.val_dataset is None else create_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        return None if self.test_dataset is None else create_dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
