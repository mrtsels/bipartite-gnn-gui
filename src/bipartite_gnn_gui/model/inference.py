"""Inference helpers for GUI correction."""

from __future__ import annotations

from typing import Any


def correct_layout(model: Any, data: Any) -> Any:
    """Run inference and return the model outputs."""

    return model(data)
