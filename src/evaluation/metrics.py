from dataclasses import dataclass
from statistics import mean, median

from src.core.prediction import GeographicPrediction
from src.evaluation.distance import haversine_distance

DISTANCE_THRESHOLDS_KM = (1, 25, 200, 750, 2500)


@dataclass(frozen=True, slots=True)
class ImageEvaluation:
    top_1_error_km: float
    best_candidate_error_km: float
    best_candidate_rank: int
    candidate_errors_km: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DatasetEvaluation:
    sample_count: int
    mean_error_km: float
    median_error_km: float
    mean_best_candidate_error_km: float
    accuracy_at_distance: dict[int, float]
    candidate_recall_at_distance: dict[int, float]


def evaluate_prediction(
    prediction: GeographicPrediction,
    actual_latitude: float,
    actual_longitude: float,
) -> ImageEvaluation:
    if not prediction.candidates:
        raise ValueError("Prediction contains no candidates")

    errors = tuple(
        haversine_distance(
            candidate.latitude,
            candidate.longitude,
            actual_latitude,
            actual_longitude,
        )
        for candidate in prediction.candidates
    )

    best_index = min(
        range(len(errors)),
        key=errors.__getitem__,
    )

    return ImageEvaluation(
        top_1_error_km=errors[0],
        best_candidate_error_km=errors[best_index],
        best_candidate_rank=prediction.candidates[best_index].rank,
        candidate_errors_km=errors,
    )


def evaluate_dataset(
    evaluations: list[ImageEvaluation],
) -> DatasetEvaluation:
    if not evaluations:
        raise ValueError("Evaluation dataset is empty")

    sample_count = len(evaluations)

    top_1_errors = [result.top_1_error_km for result in evaluations]

    best_errors = [result.best_candidate_error_km for result in evaluations]

    accuracy = {
        threshold: sum(error <= threshold for error in top_1_errors) / sample_count
        for threshold in DISTANCE_THRESHOLDS_KM
    }

    candidate_recall = {
        threshold: sum(error <= threshold for error in best_errors) / sample_count
        for threshold in DISTANCE_THRESHOLDS_KM
    }

    return DatasetEvaluation(
        sample_count=sample_count,
        mean_error_km=mean(top_1_errors),
        median_error_km=median(top_1_errors),
        mean_best_candidate_error_km=mean(best_errors),
        accuracy_at_distance=accuracy,
        candidate_recall_at_distance=candidate_recall,
    )
