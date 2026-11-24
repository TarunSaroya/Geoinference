import argparse

from src.evaluation.metrics import evaluate_prediction
from src.services.geoclip_service import GeoCLIPService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate geographic prediction accuracy"
    )

    parser.add_argument("image")
    parser.add_argument("latitude", type=float)
    parser.add_argument("longitude", type=float)

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        choices=range(1, 101),
        metavar="1-100",
    )

    args = parser.parse_args()

    if not -90 <= args.latitude <= 90:
        parser.error("Invalid ground-truth latitude")

    if not -180 <= args.longitude <= 180:
        parser.error("Invalid ground-truth longitude")

    service = GeoCLIPService()

    prediction = service.predict(
        args.image,
        top_k=args.top_k,
    )

    evaluation = evaluate_prediction(
        prediction,
        actual_latitude=args.latitude,
        actual_longitude=args.longitude,
    )

    print("\n=== GROUND TRUTH ===")
    print(f"Latitude: {args.latitude}")
    print(f"Longitude: {args.longitude}")

    print("\n=== CANDIDATE EVALUATION ===")

    for candidate, error in zip(
        prediction.candidates,
        evaluation.candidate_errors_km,
        strict=True,
    ):
        print(
            f"Rank {candidate.rank} | "
            f"Latitude: {candidate.latitude:.6f} | "
            f"Longitude: {candidate.longitude:.6f} | "
            f"Score: {candidate.score:.6f} | "
            f"Error: {error:.2f} km"
        )

    print("\n=== EVALUATION SUMMARY ===")

    print(f"Top-1 error: {evaluation.top_1_error_km:.2f} km")

    print(f"Best candidate error: {evaluation.best_candidate_error_km:.2f} km")

    print(f"Best candidate rank: {evaluation.best_candidate_rank}")

    improvement = evaluation.top_1_error_km - evaluation.best_candidate_error_km

    print(f"Potential reranking improvement: {improvement:.2f} km")


if __name__ == "__main__":
    main()
