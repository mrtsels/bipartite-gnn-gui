"""Baseline evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineNoCorrection:
    """Return inputs unchanged."""

    def __call__(self, data: Any) -> Any:
        return data


@dataclass
class BaselineRuleBased(BaselineNoCorrection):
    """Placeholder rule-based baseline."""


@dataclass
class BaselineMLPOnly(BaselineNoCorrection):
    """Placeholder MLP baseline."""
