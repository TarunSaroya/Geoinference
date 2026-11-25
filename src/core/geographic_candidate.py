from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class GeographicCandidate:
    latitude: float
    longitude: float
    score: float
    rank: int
    source: str

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("Invalid latitude")

        if not -180 <= self.longitude <= 180:
            raise ValueError("Invalid longitude")

        if not isfinite(self.score):
            raise ValueError("Score must be finite")

        if self.rank < 1:
            raise ValueError("Rank must be positive")

        if not self.source.strip():
            raise ValueError("Source cannot be empty")

    @property
    def map_url(self) -> str:
        return f"https://www.google.com/maps?q={self.latitude},{self.longitude}"
