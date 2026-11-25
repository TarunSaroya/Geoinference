import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.datasets.mapillary import (
    MapillaryAPIError,
    MapillaryClient,
)
from src.evaluation.distance import haversine_distance


@dataclass(frozen=True, slots=True)
class SamplingRegion:
    name: str
    country: str
    latitude: float
    longitude: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        delta = 0.01

        return (
            self.longitude - delta,
            self.latitude - delta,
            self.longitude + delta,
            self.latitude + delta,
        )


SMOKE_REGIONS = (
    SamplingRegion("Jaipur", "India", 26.9124, 75.7873),
    SamplingRegion("Recife", "Brazil", -8.0476, -34.8770),
    SamplingRegion("Denver", "United States", 39.7392, -104.9903),
    SamplingRegion("Osaka", "Japan", 34.6937, 135.5023),
    SamplingRegion("Perth", "Australia", -31.9505, 115.8605),
    SamplingRegion("Cape Town", "South Africa", -33.9249, 18.4241),
    SamplingRegion("Lyon", "France", 45.7640, 4.8357),
    SamplingRegion("Bandung", "Indonesia", -6.9175, 107.6191),
    SamplingRegion("Santiago", "Chile", -33.4489, -70.6693),
    SamplingRegion("Tampere", "Finland", 61.4978, 23.7610),
)

DEVELOPMENT_REGIONS = (
    SamplingRegion("Delhi", "India", 28.6139, 77.2090),
    SamplingRegion("Bengaluru", "India", 12.9716, 77.5946),
    SamplingRegion("Curitiba", "Brazil", -25.4284, -49.2733),
    SamplingRegion("Fortaleza", "Brazil", -3.7319, -38.5267),
    SamplingRegion("Seattle", "United States", 47.6062, -122.3321),
    SamplingRegion("Austin", "United States", 30.2672, -97.7431),
    SamplingRegion("Toronto", "Canada", 43.6532, -79.3832),
    SamplingRegion("Mexico City", "Mexico", 19.4326, -99.1332),
    SamplingRegion("Tokyo", "Japan", 35.6762, 139.6503),
    SamplingRegion("Seoul", "South Korea", 37.5665, 126.9780),
    SamplingRegion("Bangkok", "Thailand", 13.7563, 100.5018),
    SamplingRegion("Jakarta", "Indonesia", -6.2088, 106.8456),
    SamplingRegion("Kuala Lumpur", "Malaysia", 3.1390, 101.6869),
    SamplingRegion("Melbourne", "Australia", -37.8136, 144.9631),
    SamplingRegion("Auckland", "New Zealand", -36.8509, 174.7645),
    SamplingRegion("Johannesburg", "South Africa", -26.2041, 28.0473),
    SamplingRegion("Nairobi", "Kenya", -1.2921, 36.8219),
    SamplingRegion("Berlin", "Germany", 52.5200, 13.4050),
    SamplingRegion("Madrid", "Spain", 40.4168, -3.7038),
    SamplingRegion("Rome", "Italy", 41.9028, 12.4964),
    SamplingRegion("Helsinki", "Finland", 60.1699, 24.9384),
    SamplingRegion("Stockholm", "Sweden", 59.3293, 18.0686),
    SamplingRegion("Warsaw", "Poland", 52.2297, 21.0122),
    SamplingRegion("Buenos Aires", "Argentina", -34.6037, -58.3816),
    SamplingRegion("Bogota", "Colombia", 4.7110, -74.0721),
)

MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "latitude",
    "longitude",
    "country",
    "source",
    "source_id",
    "split",
    "region",
    "sequence_id",
    "captured_at",
    "creator",
    "source_url",
]


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def save_manifest(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def is_spatially_separated(
    latitude: float,
    longitude: float,
    rows: list[dict],
    split: str,
    minimum_distance_km: float,
) -> bool:
    for row in rows:
        distance = haversine_distance(
            latitude,
            longitude,
            float(row["latitude"]),
            float(row["longitude"]),
        )

        threshold = minimum_distance_km if row["split"] == split else 25.0

        if distance < threshold:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire geotagged Mapillary images")

    parser.add_argument(
        "--manifest",
        default="data/evaluation/manifest.csv",
    )

    parser.add_argument(
        "--split",
        choices=["smoke", "development"],
        default="smoke",
    )

    parser.add_argument(
        "--per-region",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-distance-km",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not 1 <= args.per_region <= 10:
        parser.error("per-region must be between 1 and 10")

    if args.min_distance_km < 0:
        parser.error("Minimum distance cannot be negative")

    token = os.getenv("MAPILLARY_ACCESS_TOKEN")

    if not token:
        parser.error("MAPILLARY_ACCESS_TOKEN is not configured")

    regions = SMOKE_REGIONS if args.split == "smoke" else DEVELOPMENT_REGIONS

    manifest_path = Path(args.manifest).resolve()

    rows = load_manifest(manifest_path)

    seen_ids = {row["source_id"] for row in rows if row["source"] == "mapillary"}

    seen_sequences = {row.get("sequence_id") for row in rows if row.get("sequence_id")}

    client = MapillaryClient(token)

    downloaded = 0
    failures = []

    try:
        for region in regions:
            print(f"\n=== {region.name}, {region.country} ===")

            existing = sum(
                row.get("region") == region.name and row["split"] == args.split
                for row in rows
            )

            remaining = max(0, args.per_region - existing)

            if remaining == 0:
                print("Region already complete")
                continue

            try:
                candidates = client.search_images(
                    region.bbox,
                    limit=100,
                )

            except MapillaryAPIError as error:
                print(f"Search failed: {error}")

                failures.append(
                    {
                        "region": region.name,
                        "reason": str(error),
                    }
                )

                if error.status_code in (401, 403, 429):
                    print("Stopping acquisition due to API restriction")
                    break

                continue

            except Exception as error:
                print(f"Search failed: {type(error).__name__}")

                failures.append(
                    {
                        "region": region.name,
                        "reason": type(error).__name__,
                    }
                )

                continue

            print(f"Eligible candidates: {len(candidates)}")

            candidates.sort(key=lambda image: image.image_id)

            random.Random(f"{args.seed}:{region.name}").shuffle(candidates)

            acquired = 0

            for image in candidates:
                if acquired >= remaining:
                    break

                if image.image_id in seen_ids:
                    continue

                if image.sequence_id and image.sequence_id in seen_sequences:
                    continue

                if not is_spatially_separated(
                    image.latitude,
                    image.longitude,
                    rows,
                    args.split,
                    args.min_distance_km,
                ):
                    continue

                image_id = f"mapillary_{image.image_id}"

                relative_path = f"images/{image_id}.jpg"

                destination = manifest_path.parent / relative_path

                try:
                    client.download_image(
                        image,
                        destination,
                    )

                except Exception as error:
                    print(f"Download failed: {type(error).__name__}")
                    continue

                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": relative_path,
                        "latitude": image.latitude,
                        "longitude": image.longitude,
                        "country": region.country,
                        "source": "mapillary",
                        "source_id": image.image_id,
                        "split": args.split,
                        "region": region.name,
                        "sequence_id": image.sequence_id,
                        "captured_at": image.captured_at,
                        "creator": image.creator,
                        "source_url": (
                            f"https://www.mapillary.com/app/?pKey={image.image_id}"
                        ),
                    }
                )

                save_manifest(manifest_path, rows)

                seen_ids.add(image.image_id)

                if image.sequence_id:
                    seen_sequences.add(image.sequence_id)

                acquired += 1
                downloaded += 1

                print(f"Downloaded: {image_id}")

                print(f"GPS: {image.latitude:.6f}, {image.longitude:.6f}")

                time.sleep(0.5)

            if acquired < remaining:
                failures.append(
                    {
                        "region": region.name,
                        "reason": "Insufficient eligible images",
                    }
                )

            time.sleep(1)

    finally:
        client.close()

    print("\n=== ACQUISITION SUMMARY ===")
    print(f"Split: {args.split}")
    print(f"New images: {downloaded}")
    print(f"Manifest samples: {len(rows)}")

    split_count = sum(row["split"] == args.split for row in rows)

    print(f"Selected split samples: {split_count}")
    print(f"Regions with failures: {len(failures)}")

    if failures:
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
