from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GeographicMeta:
    meta_type: str
    observation: Any
    confidence: float
    geographic_relevance: str
    source: str
    spatial: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
