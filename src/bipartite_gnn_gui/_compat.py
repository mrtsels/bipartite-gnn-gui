"""Compatibility shims for optional runtime dependencies.

These stubs keep the package importable in lightweight environments where
PyTorch or PyG are not installed yet. They are intentionally minimal and are
only meant to support module import and basic smoke tests.
"""

from __future__ import annotations

from types import SimpleNamespace


class _Tensor(list):
    """Very small list-backed tensor placeholder."""

    def __init__(self, values=None):
        super().__init__(values or [])

    def float(self):
        return self

    def long(self):
        return self

    def unsqueeze(self, *_args, **_kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def flatten(self):
        return self

    def tolist(self):
        return list(self)

    def item(self):
        return 0.0

    def mean(self):
        return 0.0

    def max(self, *_args, **_kwargs):
        return SimpleNamespace(values=self, dim=lambda *_a, **_k: self)


class _Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Linear(_Module):
    def __init__(self, *_args, **_kwargs):
        pass

    def forward(self, x):
        return x


class _ReLU(_Module):
    def forward(self, x):
        return x


class _Sequential(_Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x


class _ModuleList(list):
    pass


class _Dataset:
    pass


class _DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn


class _NNFunctional:
    @staticmethod
    def mse_loss(*_args, **_kwargs):
        return 0.0

    @staticmethod
    def binary_cross_entropy(*_args, **_kwargs):
        return 0.0


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(*_args, **_kwargs):
        return None


class _TorchNamespace(SimpleNamespace):
    def tensor(self, values, **_kwargs):
        return _Tensor(values)

    def as_tensor(self, values, **_kwargs):
        return _Tensor(values)

    def zeros(self, shape, **_kwargs):
        if isinstance(shape, int):
            return _Tensor([0.0] * shape)
        rows, cols = shape
        return _Tensor([_Tensor([0.0] * cols) for _ in range(rows)])

    def empty(self, shape, **_kwargs):
        return self.zeros(shape)

    def stack(self, tensors, **_kwargs):
        return _Tensor(list(tensors))

    def maximum(self, left, right):
        return left

    def minimum(self, left, right):
        return right

    def clamp(self, tensor, **_kwargs):
        return tensor

    def where(self, condition, left, right):
        return left if condition else right

    def norm(self, tensor, **_kwargs):
        return 0.0

    def abs(self, tensor):
        return tensor

    def relu(self, tensor):
        return tensor

    def sigmoid(self, tensor):
        return tensor

    def manual_seed(self, *_args, **_kwargs):
        return None

    def use_deterministic_algorithms(self, *_args, **_kwargs):
        return None


Tensor = _Tensor
nn = SimpleNamespace(Module=_Module, Linear=_Linear, ReLU=_ReLU, Sequential=_Sequential, ModuleList=_ModuleList)
F = _NNFunctional()
DataLoader = _DataLoader
Dataset = _Dataset
torch = _TorchNamespace(float32="float32", long="long", cuda=_Cuda(), Tensor=_Tensor)
