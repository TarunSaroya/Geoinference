from pathlib import Path

import pytest

from src.core.geographic_candidate import GeographicCandidate
from src.core.prediction import GeographicPrediction
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.dataset import EvaluationSample


class MockPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, image_path, top_k):
        self.calls += 1

        candidates = tuple(
            GeographicCandidate(
                latitude=0,
                longitude=0 if rank == 10 else 10,
                score=1 / rank,
                rank=rank,
                source="mock",
            )
            for rank in range(1, top_k + 1)
        )

        return GeographicPrediction(
            source="mock",
            candidates=candidates,
            execution_time_ms=10,
            model_loading_time_ms=0,
            device="cpu",
            model_version="test",
            score_type="test",
        )


class FailingPredictor:
    def predict(self, image_path, top_k):
        raise RuntimeError("Test prediction failure")


def create_sample():
    return EvaluationSample(
        image_id="001",
        image_path=Path("test.jpg"),
        latitude=0,
        longitude=0,
        country="Test",
        source="test",
        source_id="1",
        split="smoke",
    )


def test_benchmark():
    predictor = MockPredictor()
    runner = BenchmarkRunner(predictor)

    report = runner.run((create_sample(),))

    assert predictor.calls == 1
    assert report["successful_samples"] == 1
    assert report["failed_samples"] == 0

    top_5 = report["metrics"]["5"]
    top_20 = report["metrics"]["20"]
    top_100 = report["metrics"]["100"]

    assert top_5["candidate_recall_at_distance"][25] == 0
    assert top_20["candidate_recall_at_distance"][25] == 1
    assert top_100["candidate_recall_at_distance"][25] == 1

    assert top_5["mean_error_km"] == top_20["mean_error_km"]


def test_empty_benchmark():
    runner = BenchmarkRunner(MockPredictor())

    with pytest.raises(ValueError):
        runner.run(())


def test_prediction_failure():
    runner = BenchmarkRunner(FailingPredictor())

    report = runner.run((create_sample(),))

    assert report["successful_samples"] == 0
    assert report["failed_samples"] == 1
    assert report["metrics"]["5"] is None

    assert report["failures"][0]["error_type"] == "RuntimeError"
