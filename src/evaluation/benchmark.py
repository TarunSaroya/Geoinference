from dataclasses import asdict, replace
from time import perf_counter
from typing import Any, Protocol

from src.core.prediction import GeographicPrediction
from src.evaluation.dataset import EvaluationSample
from src.evaluation.metrics import (
    ImageEvaluation,
    evaluate_dataset,
    evaluate_prediction,
)

CANDIDATE_COUNTS = (5, 20, 100)


class GeographicPredictor(Protocol):
    def predict(
        self,
        image_path: Any,
        top_k: int,
    ) -> GeographicPrediction: ...


class BenchmarkRunner:
    def __init__(
        self,
        predictor: GeographicPredictor,
    ) -> None:
        self.predictor = predictor

    def run(
        self,
        samples: tuple[EvaluationSample, ...],
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("Benchmark dataset is empty")

        start = perf_counter()

        evaluations: dict[int, list[ImageEvaluation]] = {
            count: [] for count in CANDIDATE_COUNTS
        }

        records = []
        failures = []

        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.image_id}")

            try:
                prediction = self.predictor.predict(
                    sample.image_path,
                    top_k=max(CANDIDATE_COUNTS),
                )

                if prediction.candidate_count < max(CANDIDATE_COUNTS):
                    raise ValueError("Insufficient geographic candidates")

                sample_evaluations = {}

                for count in CANDIDATE_COUNTS:
                    subset = replace(
                        prediction,
                        candidates=prediction.candidates[:count],
                    )

                    evaluation = evaluate_prediction(
                        subset,
                        sample.latitude,
                        sample.longitude,
                    )

                    sample_evaluations[count] = evaluation

                for count, evaluation in sample_evaluations.items():
                    evaluations[count].append(evaluation)

                records.append(
                    {
                        "image_id": sample.image_id,
                        "ground_truth": {
                            "latitude": sample.latitude,
                            "longitude": sample.longitude,
                            "country": sample.country,
                        },
                        "prediction": asdict(prediction),
                        "evaluations": {
                            str(count): asdict(evaluation)
                            for count, evaluation in sample_evaluations.items()
                        },
                    }
                )

            except Exception as error:
                failures.append(
                    {
                        "image_id": sample.image_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )

                print(f"    FAILED: {error}")

        metrics = {
            str(count): (asdict(evaluate_dataset(results)) if results else None)
            for count, results in evaluations.items()
        }

        return {
            "total_samples": len(samples),
            "successful_samples": len(records),
            "failed_samples": len(failures),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "execution_time_ms": (perf_counter() - start) * 1000,
            "metrics": metrics,
            "records": records,
            "failures": failures,
        }
