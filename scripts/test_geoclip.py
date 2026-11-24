from pathlib import Path

from src.services.geoclip_service import GeoCLIPService


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "sample.jpg"

    service = GeoCLIPService()

    print(f"Device: {service.device}")
    print(f"Initially loaded: {service.is_loaded}")

    for run in range(1, 3):
        print(f"\n=== PREDICTION RUN {run} ===")

        prediction = service.predict(
            image_path,
            top_k=5,
        )

        print(f"Source: {prediction.source}")
        print(f"Model version: {prediction.model_version}")
        print(f"Device: {prediction.device}")
        print(f"Score type: {prediction.score_type}")

        print(f"Model loading: {prediction.model_loading_time_ms:.2f} ms")

        print(f"Execution time: {prediction.execution_time_ms:.2f} ms")

        for candidate in prediction.candidates:
            print(
                f"\nRank {candidate.rank}: "
                f"({candidate.latitude:.6f}, "
                f"{candidate.longitude:.6f})"
            )

            print(f"Score: {candidate.score:.6f}")
            print(f"Map: {candidate.map_url}")

    print(f"\nModel loaded: {service.is_loaded}")


if __name__ == "__main__":
    main()
