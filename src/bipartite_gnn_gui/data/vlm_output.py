"""VLM output parsing helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class VLMOutputElement:
    """Single predicted GUI element."""

    bbox: list[float]
    label: str = "unknown"
    confidence: float = 1.0
    text: str | None = None
    element_id: str | None = None


@dataclass
class VLMOutput:
    """Container for parsed VLM predictions."""

    elements: list[VLMOutputElement] = field(default_factory=list)
    source: str | None = None
    image_size: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return {
            "elements": [asdict(element) for element in self.elements],
            "source": self.source,
            "image_size": self.image_size,
        }


def _parse_element(payload: Mapping[str, Any]) -> VLMOutputElement:
    bbox = list(payload.get("bbox", payload.get("box", [0.0, 0.0, 0.0, 0.0])))
    return VLMOutputElement(
        bbox=[float(value) for value in bbox],
        label=str(payload.get("label", payload.get("type", "unknown"))),
        confidence=float(payload.get("confidence", 1.0)),
        text=payload.get("text"),
        element_id=payload.get("id"),
    )


def load_vlm_output(source: str | Path | Mapping[str, Any]) -> VLMOutput:
    """Load a VLM output from a path or mapping."""

    if isinstance(source, Mapping):
        payload = source
        source_name = None
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source_name = str(path)

    elements = [_parse_element(element) for element in payload.get("elements", payload.get("predictions", []))]
    image_size = tuple(payload["image_size"]) if payload.get("image_size") else None
    return VLMOutput(elements=elements, source=source_name, image_size=image_size)


class VLMOutputLoader:
    """Simple callable loader wrapper."""

    def __call__(self, source: str | Path | Mapping[str, Any]) -> VLMOutput:
        return load_vlm_output(source)
