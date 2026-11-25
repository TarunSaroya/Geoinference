from dataclasses import dataclass

from src.core.evidence import Evidence
from src.core.meta import GeographicMeta
from src.core.prediction import GeographicPrediction
from src.core.result import ExtractorResult


@dataclass(frozen=True, slots=True)
class ApplicationResult:
    extractor_results: tuple[ExtractorResult, ...]
    evidence: tuple[Evidence, ...]
    geographic_metas: tuple[GeographicMeta, ...]
    geographic_prediction: GeographicPrediction
    execution_time_ms: float

    @property
    def successful_extractors(self) -> int:
        return sum(result.success for result in self.extractor_results)

    @property
    def failed_extractors(self) -> int:
        return sum(not result.success for result in self.extractor_results)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def meta_count(self) -> int:
        return len(self.geographic_metas)
