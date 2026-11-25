from dataclasses import dataclass

from src.core.geographic_candidate import GeographicCandidate


@dataclass(frozen=True, slots=True)
class GeographicPrediction:
    source: str
    candidates: tuple[GeographicCandidate, ...]
    execution_time_ms: float
    model_loading_time_ms: float
    device: str
    model_version: str
    score_type: str

    @property
    def best_candidate(self) -> GeographicCandidate | None:
        if not self.candidates:
            return None

        return self.candidates[0]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)
