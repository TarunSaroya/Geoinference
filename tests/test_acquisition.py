from scripts.acquire_dataset import is_spatially_separated
from scripts.analyze_benchmark import classify_failure


def test_reject_nearby_image():
    rows = [
        {
            "latitude": 0,
            "longitude": 0,
            "split": "development",
        }
    ]

    assert not is_spatially_separated(
        0,
        0.0001,
        rows,
        "development",
        0.15,
    )


def test_accept_distant_image():
    rows = [
        {
            "latitude": 0,
            "longitude": 0,
            "split": "development",
        }
    ]

    assert is_spatially_separated(
        0,
        1,
        rows,
        "development",
        0.15,
    )


def test_cross_split_separation():
    rows = [
        {
            "latitude": 0,
            "longitude": 0,
            "split": "smoke",
        }
    ]

    assert not is_spatially_separated(
        0,
        0.1,
        rows,
        "development",
        0.15,
    )


def test_successful_prediction():
    assert classify_failure(10, 5, 25) == ("successful_prediction")


def test_ranking_failure():
    assert classify_failure(500, 10, 25) == ("ranking_failure")


def test_generation_failure():
    assert classify_failure(500, 100, 25) == ("candidate_generation_failure")
