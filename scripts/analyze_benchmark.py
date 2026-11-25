import argparse
import json
from collections import Counter
from pathlib import Path


def classify_failure(
    top_1_error: float,
    best_error: float,
    threshold: float,
) -> str:
    if top_1_error <= threshold:
        return "successful_prediction"

    if best_error <= threshold:
        return "ranking_failure"

    return "candidate_generation_failure"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze geographic benchmark results")

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--split",
        choices=["smoke", "development", "test"],
        default="development",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
    )

    args = parser.parse_args()

    if args.threshold <= 0:
        parser.error("Threshold must be positive")

    report_path = args.report

    if report_path is None:
        reports = list(Path("reports/benchmarks").glob(f"geoclip_{args.split}_*.json"))

        if not reports:
            parser.error("No matching benchmark reports")

        report_path = max(
            reports,
            key=lambda path: path.stat().st_mtime,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    records = report["records"]

    if not records:
        parser.error("Report contains no successful predictions")

    failure_counts = Counter()
    country_counts = Counter()

    print(f"Report: {report_path}")
    print(f"Successful samples: {len(records)}")
    print(f"Failed samples: {report['failed_samples']}")
    print(f"Analysis threshold: {args.threshold} km")

    print("\n=== INDIVIDUAL PREDICTIONS ===")

    for record in records:
        evaluation = record["evaluations"]["100"]

        top_1_error = evaluation["top_1_error_km"]
        best_error = evaluation["best_candidate_error_km"]
        best_rank = evaluation["best_candidate_rank"]

        category = classify_failure(
            top_1_error,
            best_error,
            args.threshold,
        )

        failure_counts[category] += 1

        country = record["ground_truth"]["country"]

        country_counts[country] += 1

        print(
            f"{country:16} | "
            f"Top-1: {top_1_error:9.2f} km | "
            f"Best: {best_error:9.2f} km | "
            f"Rank: {best_rank:3} | "
            f"{category}"
        )

    print("\n=== FAILURE ANALYSIS ===")

    for category in (
        "successful_prediction",
        "ranking_failure",
        "candidate_generation_failure",
    ):
        count = failure_counts[category]

        print(f"{category}: {count} ({count / len(records):.2%})")

    print("\n=== COUNTRY DISTRIBUTION ===")

    for country, count in country_counts.most_common():
        print(f"{country}: {count}")

    metrics = report["metrics"]["100"]

    baseline_error = metrics["mean_error_km"]

    oracle_error = metrics["mean_best_candidate_error_km"]

    improvement = (
        (baseline_error - oracle_error) / baseline_error * 100
        if baseline_error > 0
        else 0.0
    )

    print("\n=== RERANKING POTENTIAL ===")

    print(f"Baseline mean error: {baseline_error:.2f} km")
    print(f"Oracle mean error: {oracle_error:.2f} km")
    print(f"Oracle error reduction: {improvement:.2f}%")

    print("\nNote: Oracle improvement assumes perfect candidate selection.")


if __name__ == "__main__":
    main()
