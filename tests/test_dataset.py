from pathlib import Path

import pytest

from src.evaluation.dataset import EvaluationDataset

HEADER = "image_id,image_path,latitude,longitude,country,source,source_id,split\n"


def create_manifest(
    directory: Path,
    rows: str,
) -> Path:
    image = directory / "image.jpg"
    image.touch()

    manifest = directory / "manifest.csv"

    manifest.write_text(
        HEADER + rows,
        encoding="utf-8",
    )

    return manifest


def test_load_dataset(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,26.9,75.8,India,test,1,smoke\n",
    )

    dataset = EvaluationDataset(manifest)

    assert len(dataset) == 1

    sample = dataset.samples[0]

    assert sample.image_id == "001"
    assert sample.latitude == 26.9
    assert sample.longitude == 75.8
    assert sample.image_path.is_absolute()


def test_dataset_split(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,0,0,Test,test,1,smoke\n"
        "002,image.jpg,1,1,Test,test,2,development\n",
    )

    dataset = EvaluationDataset(manifest)

    assert len(dataset.get_split("smoke")) == 1
    assert len(dataset.get_split("development")) == 1
    assert len(dataset.get_split("test")) == 0


def test_duplicate_id(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,0,0,Test,test,1,smoke\n001,image.jpg,1,1,Test,test,2,test\n",
    )

    with pytest.raises(ValueError, match="Duplicate image ID"):
        EvaluationDataset(manifest)


def test_duplicate_source(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,0,0,Test,test,1,smoke\n002,image.jpg,0,0,Test,test,1,test\n",
    )

    with pytest.raises(ValueError, match="Duplicate source"):
        EvaluationDataset(manifest)


def test_invalid_coordinates(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,91,0,Test,test,1,smoke\n",
    )

    with pytest.raises(ValueError, match="Invalid latitude"):
        EvaluationDataset(manifest)


def test_missing_image(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,missing.jpg,0,0,Test,test,1,smoke\n",
    )

    with pytest.raises(ValueError):
        EvaluationDataset(manifest)


def test_invalid_split(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "001,image.jpg,0,0,Test,test,1,invalid\n",
    )

    with pytest.raises(ValueError, match="Invalid split"):
        EvaluationDataset(manifest)


def test_missing_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"

    manifest.write_text(
        "image_id,latitude\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing columns"):
        EvaluationDataset(manifest)
