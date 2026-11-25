from src.core.application_result import ApplicationResult
from src.core.geographic_candidate import GeographicCandidate
from src.core.prediction import GeographicPrediction
from src.core.result import ExtractorResult


def test_application_result():
    prediction = GeographicPrediction(
        source="geoclip",
        candidates=(
            GeographicCandidate(
                latitude=26.9124,
                longitude=75.7873,
                score=0.8,
                rank=1,
                source="geoclip",
            ),
        ),
        execution_time_ms=100,
        model_loading_time_ms=0,
        device="cpu",
        model_version="test",
        score_type="test",
    )

    result = ApplicationResult(
        extractor_results=(
            ExtractorResult(
                extractor="test_success",
                success=True,
                confidence=0.9,
            ),
            ExtractorResult(
                extractor="test_failure",
                success=False,
                confidence=0.0,
            ),
        ),
        evidence=(),
        geographic_metas=(),
        geographic_prediction=prediction,
        execution_time_ms=200,
    )

    assert result.successful_extractors == 1
    assert result.failed_extractors == 1
    assert result.evidence_count == 0
    assert result.meta_count == 0
    assert result.geographic_prediction.candidate_count == 1

    candidate = result.geographic_prediction.best_candidate

    assert candidate is not None
    assert candidate.latitude == 26.9124
    assert candidate.longitude == 75.7873
