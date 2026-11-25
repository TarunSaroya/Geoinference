from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from src.datasets.mapillary import (
    MapillaryAPIError,
    MapillaryClient,
    MapillaryImage,
)


@pytest.fixture
def client():
    instance = MapillaryClient("test-token")

    yield instance

    instance.close()


def test_empty_token():
    with pytest.raises(ValueError):
        MapillaryClient("")


def test_search_images(client):
    response = Mock()
    response.ok = True

    response.json.return_value = {
        "data": [
            {
                "id": "123",
                "geometry": {"coordinates": [75.8, 26.9]},
                "thumb_1024_url": "https://example.com/image.jpg",
                "sequence": "sequence-1",
                "captured_at": 1700000000000,
                "creator": {"username": "photographer"},
                "is_pano": False,
            }
        ]
    }

    client.session.get = Mock(return_value=response)

    images = client.search_images((75.7, 26.8, 75.9, 27.0))

    assert len(images) == 1
    assert images[0].image_id == "123"
    assert images[0].latitude == 26.9
    assert images[0].longitude == 75.8
    assert images[0].creator == "photographer"
    assert images[0].sequence_id == "sequence-1"


def test_skip_panorama(client):
    response = Mock()
    response.ok = True

    response.json.return_value = {
        "data": [
            {
                "id": "123",
                "is_pano": True,
            }
        ]
    }

    client.session.get = Mock(return_value=response)

    assert client.search_images((0, 0, 1, 1)) == []


def test_skip_invalid_coordinates(client):
    response = Mock()
    response.ok = True

    response.json.return_value = {
        "data": [
            {
                "id": "123",
                "geometry": {"coordinates": [0, 100]},
                "thumb_1024_url": "https://example.com/image.jpg",
            }
        ]
    }

    client.session.get = Mock(return_value=response)

    assert client.search_images((0, 0, 1, 1)) == []


def test_api_error(client):
    response = Mock()
    response.ok = False
    response.status_code = 429

    client.session.get = Mock(return_value=response)

    with pytest.raises(MapillaryAPIError) as error:
        client.search_images((0, 0, 1, 1))

    assert error.value.status_code == 429


def test_download_image(client, tmp_path):
    buffer = BytesIO()

    Image.new(
        "RGB",
        (640, 480),
    ).save(buffer, format="JPEG")

    response = Mock()
    response.ok = True
    response.content = buffer.getvalue()

    client.download_session.get = Mock(return_value=response)

    image = MapillaryImage(
        image_id="123",
        latitude=0,
        longitude=0,
        thumbnail_url="https://example.com/image.jpg",
        sequence_id="sequence-1",
        captured_at=None,
        creator="test",
    )

    destination = tmp_path / "image.jpg"

    client.download_image(image, destination)

    assert destination.is_file()

    with Image.open(destination) as saved:
        assert saved.size == (640, 480)

    assert not destination.with_suffix(".tmp").exists()


def test_download_session_has_no_token(client):
    assert "Authorization" in client.session.headers

    assert "Authorization" not in client.download_session.headers
