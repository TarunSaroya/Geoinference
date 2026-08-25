from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractorResult:
    """
    Standard output returned by every extractor.
    """

    extractor: str
    success: bool
    confidence: float

    data: dict[str, Any] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    execution_time_ms: float = 0.0
