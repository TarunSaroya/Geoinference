import pytest

from src.core.geographic_candidate import GeographicCandidate


def test_valid_candidate():
    candidate = GeographicCandidate(
        latitude=26.9124,
        longitude=75.7873,
        score=0.75,
        rank=1,
        source="geoclip",
    )

    assert candidate.latitude == 26.9124
    assert candidate.longitude == 75.7873
    assert candidate.rank == 1
    assert candidate.source == "geoclip"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (91, 0),
        (-91, 0),
        (0, 181),
        (0, -181),
        (float("nan"), 0),
    ],
)
def test_invalid_coordinates(latitude, longitude):
    with pytest.raises(ValueError):
        GeographicCandidate(
            latitude=latitude,
            longitude=longitude,
            score=0.5,
            rank=1,
            source="geoclip",
        )


def test_invalid_rank():
    with pytest.raises(ValueError):
        GeographicCandidate(
            latitude=0,
            longitude=0,
            score=0.5,
            rank=0,
            source="geoclip",
        )


def test_invalid_score():
    with pytest.raises(ValueError):
        GeographicCandidate(
            latitude=0,
            longitude=0,
            score=float("nan"),
            rank=1,
            source="geoclip",
        )


def test_map_url():
    candidate = GeographicCandidate(
        latitude=10.0,
        longitude=20.0,
        score=0.5,
        rank=1,
        source="geoclip",
    )

    assert candidate.map_url == ("https://www.google.com/maps?q=10.0,20.0")
