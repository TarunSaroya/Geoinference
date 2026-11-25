import csv
from dataclasses import dataclass
from pathlib import Path

VALID_SPLITS = {"smoke", "development", "test"}

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "latitude",
    "longitude",
    "country",
    "source",
    "source_id",
    "split",
}


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    image_id: str
    image_path: Path
    latitude: float
    longitude: float
    country: str
    source: str
    source_id: str
    split: str

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("Image ID cannot be empty")

        if not -90 <= self.latitude <= 90:
            raise ValueError("Invalid latitude")

        if not -180 <= self.longitude <= 180:
            raise ValueError("Invalid longitude")

        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {self.split}")

        if not self.source.strip():
            raise ValueError("Source cannot be empty")

        if not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")


class EvaluationDataset:
    def __init__(
        self,
        manifest_path: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        self.samples = self._load()

    def _load(self) -> tuple[EvaluationSample, ...]:
        samples = []
        seen_ids = set()
        seen_sources = set()

        with self.manifest_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            columns = set(reader.fieldnames or [])

            missing = REQUIRED_COLUMNS - columns

            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                try:
                    sample = EvaluationSample(
                        image_id=row["image_id"].strip(),
                        image_path=(
                            self.manifest_path.parent / row["image_path"].strip()
                        ).resolve(),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        country=row["country"].strip(),
                        source=row["source"].strip(),
                        source_id=row["source_id"].strip(),
                        split=row["split"].strip(),
                    )

                    if sample.image_id in seen_ids:
                        raise ValueError("Duplicate image ID")

                    source_key = (
                        sample.source,
                        sample.source_id,
                    )

                    if source_key in seen_sources:
                        raise ValueError("Duplicate source image")

                    if not sample.image_path.is_file():
                        raise FileNotFoundError(sample.image_path)

                    seen_ids.add(sample.image_id)
                    seen_sources.add(source_key)
                    samples.append(sample)

                except (ValueError, OSError, AttributeError) as error:
                    raise ValueError(f"Manifest line {line_number}: {error}") from error

        return tuple(samples)

    def get_split(
        self,
        split: str,
    ) -> tuple[EvaluationSample, ...]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        return tuple(sample for sample in self.samples if sample.split == split)

    def __len__(self) -> int:
        return len(self.samples)
