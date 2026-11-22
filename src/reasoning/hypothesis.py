from dataclasses import dataclass, field


@dataclass
class LocationHypothesis:
    region: str
    score: float
    reasons: list[str] = field(default_factory=list)
