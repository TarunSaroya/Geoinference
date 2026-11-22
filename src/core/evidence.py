from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Evidence:
    evidence_type: str
    value: Any
    confidence: float
    source: str

    geographic_relevance: str | None = None
    location: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
