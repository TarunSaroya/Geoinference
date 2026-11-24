import argparse
from pathlib import Path
from time import perf_counter

from src.core.application import Application
from src.core.application_result import ApplicationResult
from src.utils.image_loader import ImageLoader


def print_result(
    result: ApplicationResult,
    verbose: bool,
) -> None:
    prediction = result.geographic_prediction

    print("\n=== GEOGRAPHIC PREDICTION ===")

    print(f"Source: {prediction.source}")
    print(f"Model version: {prediction.model_version}")
    print(f"Device: {prediction.device}")
    print(f"Score type: {prediction.score_type}")

    for candidate in prediction.candidates:
        print(f"\nCandidate {candidate.rank}")
        print(f"Latitude: {candidate.latitude:.6f}")
        print(f"Longitude: {candidate.longitude:.6f}")
        print(f"Model score: {candidate.score:.6f}")
        print(f"Map: {candidate.map_url}")

    print("\n=== EXTRACTOR RESULTS ===")

    for extractor in result.extractor_results:
        status = "SUCCESS" if extractor.success else "FAILED"

        print(
            f"[{status}] {extractor.extractor} | {extractor.execution_time_ms:.2f} ms"
        )

        for warning in extractor.warnings:
            print(f"    Warning: {warning}")

    print("\n=== GEOGRAPHIC EVIDENCE ===")

    for evidence in result.evidence:
        print(
            f"[{evidence.source}] "
            f"{evidence.value!r} | "
            f"confidence={evidence.confidence:.3f} | "
            f"relevance={evidence.geographic_relevance} | "
            f"classification="
            f"{evidence.metadata.get('classification', 'unknown')}"
        )

    print("\n=== GEOGRAPHIC METAS ===")

    for meta in result.geographic_metas:
        print(
            f"[{meta.meta_type}] "
            f"{meta.observation} | "
            f"confidence={meta.confidence:.3f} | "
            f"relevance={meta.geographic_relevance} | "
            f"source={meta.source}"
        )

        if verbose:
            for key, value in meta.metadata.items():
                print(f"    {key}: {value}")

            if meta.spatial:
                print(f"    spatial: {meta.spatial}")

    print("\n=== EXECUTION SUMMARY ===")

    print(f"Successful extractors: {result.successful_extractors}")
    print(f"Failed extractors: {result.failed_extractors}")
    print(f"Evidence count: {result.evidence_count}")
    print(f"Geographic metas: {result.meta_count}")
    print(f"GPS candidates: {prediction.candidate_count}")

    print(f"GeoCLIP loading time: {prediction.model_loading_time_ms:.2f} ms")
    print(f"GeoCLIP inference time: {prediction.execution_time_ms:.2f} ms")
    print(f"Total pipeline time: {result.execution_time_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geoinference visual geolocation system"
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="sample.jpg",
        help="Path to the input image",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        choices=range(1, 101),
        metavar="1-100",
        help="Number of geographic candidates",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed geographic metadata",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    print("Initializing Geoinference...")

    start = perf_counter()

    app = Application()

    initialization_time = (perf_counter() - start) * 1000

    print(f"Initialization time: {initialization_time:.2f} ms")

    image = ImageLoader.load(str(image_path))

    result = app.run(
        image,
        top_k=args.top_k,
    )

    print_result(
        result,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
