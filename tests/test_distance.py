import pytest

from src.evaluation.distance import haversine_distance


def test_same_location():
    assert (
        haversine_distance(
            26.9124,
            75.7873,
            26.9124,
            75.7873,
        )
        == 0.0
    )


def test_one_degree_at_equator():
    distance = haversine_distance(0, 0, 0, 1)

    assert distance == pytest.approx(111.195, abs=0.01)


def test_distance_symmetry():
    forward = haversine_distance(26.9124, 75.7873, 28.6139, 77.2090)

    backward = haversine_distance(28.6139, 77.2090, 26.9124, 75.7873)

    assert forward == pytest.approx(backward)


def test_antipodal_distance():
    distance = haversine_distance(0, 0, 0, 180)

    assert distance == pytest.approx(20015.114, abs=0.01)


def test_crossing_antimeridian():
    distance = haversine_distance(0, 179, 0, -179)

    assert distance == pytest.approx(222.390, abs=0.01)


def test_invalid_latitude():
    with pytest.raises(ValueError):
        haversine_distance(91, 0, 0, 0)


def test_invalid_longitude():
    with pytest.raises(ValueError):
        haversine_distance(0, 181, 0, 0)
