import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.dataset import EvaluationDataset
from src.services.geoclip_service import GeoCLIPService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GeoCLIP geographic accuracy"
    )

    parser.add_argument(
        "--manifest",
        default="data/evaluation/manifest.csv",
    )

    parser.add_argument(
        "--split",
        choices=["smoke", "development", "test"],
        default="smoke",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/benchmarks",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")

    dataset = EvaluationDataset(args.manifest)

    samples = dataset.get_split(args.split)

    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        parser.error("No samples available for selected split")

    print(f"Dataset: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(samples)}")

    service = GeoCLIPService()

    print(f"Device: {service.device}")

    runner = BenchmarkRunner(service)

    report = runner.run(samples)

    timestamp = datetime.now(timezone.utc)

    report["configuration"] = {
        "model": "geoclip",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "split": args.split,
        "limit": args.limit,
        "device": str(service.device),
        "timestamp": timestamp.isoformat(),
    }

    output_directory = Path(args.output_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = f"geoclip_{args.split}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"

    output_path = output_directory / filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    print("\n=== BENCHMARK SUMMARY ===")

    print(f"Total samples: {report['total_samples']}")
    print(f"Successful: {report['successful_samples']}")
    print(f"Failed: {report['failed_samples']}")

    for count, metrics in report["metrics"].items():
        if metrics is None:
            continue

        print(f"\n=== TOP-{count} ===")

        print(f"Mean top-1 error: {metrics['mean_error_km']:.2f} km")

        print(f"Median top-1 error: {metrics['median_error_km']:.2f} km")

        print(
            f"Mean best-candidate error: "
            f"{metrics['mean_best_candidate_error_km']:.2f} km"
        )

        print("\nTop-1 accuracy:")

        for threshold, accuracy in metrics["accuracy_at_distance"].items():
            print(f"    Within {threshold} km: {accuracy:.2%}")

        print("\nCandidate recall:")

        for threshold, recall in metrics["candidate_recall_at_distance"].items():
            print(f"    Within {threshold} km: {recall:.2%}")

    print(f"\nReport saved: {output_path}")

    if report["failed_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
