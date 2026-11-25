import pytest

from src.core.geographic_candidate import GeographicCandidate
from src.core.prediction import GeographicPrediction
from src.evaluation.metrics import (
    ImageEvaluation,
    evaluate_dataset,
    evaluate_prediction,
)


def test_evaluate_prediction():
    prediction = GeographicPrediction(
        source="geoclip",
        candidates=(
            GeographicCandidate(0, 1, 0.8, 1, "geoclip"),
            GeographicCandidate(0, 0, 0.2, 2, "geoclip"),
        ),
        execution_time_ms=100,
        model_loading_time_ms=0,
        device="cpu",
        model_version="test",
        score_type="test",
    )

    result = evaluate_prediction(prediction, 0, 0)

    assert result.top_1_error_km == pytest.approx(111.195, abs=0.01)
    assert result.best_candidate_error_km == 0
    assert result.best_candidate_rank == 2


def test_evaluate_dataset():
    evaluations = [
        ImageEvaluation(10, 5, 2, (10, 5)),
        ImageEvaluation(100, 20, 2, (100, 20)),
    ]

    result = evaluate_dataset(evaluations)

    assert result.sample_count == 2
    assert result.mean_error_km == 55
    assert result.median_error_km == 55
    assert result.accuracy_at_distance[25] == 0.5
    assert result.candidate_recall_at_distance[25] == 1.0


def test_empty_dataset():
    with pytest.raises(ValueError):
        evaluate_dataset([])
