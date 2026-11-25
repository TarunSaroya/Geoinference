from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import sleep

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True, slots=True)
class MapillaryImage:
    image_id: str
    latitude: float
    longitude: float
    thumbnail_url: str
    sequence_id: str
    captured_at: int | None
    creator: str


class MapillaryAPIError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Mapillary API HTTP {status_code}")


class MapillaryClient:
    BASE_URL = "https://graph.mapillary.com"

    REQUEST_LIMIT = 20
    MAX_SEARCH_AREAS = 5
    REQUEST_DELAY = 2.0

    FIELDS = "id,geometry,thumb_1024_url,sequence,captured_at,creator,is_pano"

    def __init__(self, access_token: str) -> None:
        if not access_token.strip():
            raise ValueError("Mapillary access token is required")

        self.session = self._create_session()
        self.download_session = self._create_session()

        self.session.headers.update({"Authorization": f"OAuth {access_token}"})

    @staticmethod
    def _create_session() -> requests.Session:
        retry = Retry(
            total=1,
            connect=1,
            read=1,
            status=1,
            backoff_factor=2,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        session = requests.Session()

        session.mount(
            "https://",
            HTTPAdapter(max_retries=retry),
        )

        return session

    @staticmethod
    def _search_areas(
        bbox: tuple[float, float, float, float],
    ) -> list[tuple[float, float, float, float]]:
        west, south, east, north = bbox

        center_lon = (west + east) / 2
        center_lat = (south + north) / 2

        half_width = min(
            (east - west) / 4,
            0.005,
        )

        half_height = min(
            (north - south) / 4,
            0.005,
        )

        central_area = (
            center_lon - half_width,
            center_lat - half_height,
            center_lon + half_width,
            center_lat + half_height,
        )

        return [
            central_area,
            (west, south, center_lon, center_lat),
            (center_lon, south, east, center_lat),
            (west, center_lat, center_lon, north),
            (center_lon, center_lat, east, north),
        ]

    def _request_images(
        self,
        bbox: tuple[float, float, float, float],
        limit: int,
    ) -> list[MapillaryImage]:
        response = self.session.get(
            f"{self.BASE_URL}/images",
            params={
                "bbox": ",".join(map(str, bbox)),
                "fields": self.FIELDS,
                "limit": limit,
            },
            timeout=(10, 30),
        )

        if not response.ok:
            raise MapillaryAPIError(response.status_code)

        payload = response.json()

        images = []

        for item in payload.get("data", []):
            if item.get("is_pano"):
                continue

            geometry = item.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            thumbnail = item.get("thumb_1024_url")

            if len(coordinates) != 2 or not thumbnail:
                continue

            longitude, latitude = coordinates

            if not -90 <= latitude <= 90:
                continue

            if not -180 <= longitude <= 180:
                continue

            creator = item.get("creator") or {}

            if isinstance(creator, dict):
                creator_name = creator.get("username", "")
            else:
                creator_name = str(creator)

            sequence = item.get("sequence") or ""

            if isinstance(sequence, dict):
                sequence = sequence.get("id", "")

            images.append(
                MapillaryImage(
                    image_id=str(item["id"]),
                    latitude=float(latitude),
                    longitude=float(longitude),
                    thumbnail_url=thumbnail,
                    sequence_id=str(sequence),
                    captured_at=item.get("captured_at"),
                    creator=creator_name,
                )
            )

        return images

    def search_images(
        self,
        bbox: tuple[float, float, float, float],
        limit: int = 20,
    ) -> list[MapillaryImage]:
        if limit < 1:
            raise ValueError("Limit must be positive")

        candidates: dict[str, MapillaryImage] = {}
        last_error: Exception | None = None
        successful_requests = 0

        areas = self._search_areas(bbox)

        for index, area in enumerate(
            areas[: self.MAX_SEARCH_AREAS],
            start=1,
        ):
            request_limit = min(
                self.REQUEST_LIMIT,
                limit - len(candidates),
            )

            if request_limit <= 0:
                break

            if index > 1:
                sleep(self.REQUEST_DELAY)

            print(f"  Search area {index}/{len(areas)} (limit={request_limit})")

            try:
                images = self._request_images(
                    area,
                    request_limit,
                )

            except MapillaryAPIError as error:
                last_error = error

                if error.status_code in (401, 403, 429):
                    raise

                if error.status_code not in (500, 502, 503, 504):
                    raise

                print(f"  Area failed: {error}")
                continue

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as error:
                last_error = error

                print(f"  Area failed: {type(error).__name__}")

                continue

            successful_requests += 1

            for image in images:
                candidates[image.image_id] = image

            print(f"  Unique candidates: {len(candidates)}")

            unique_sequences = {
                image.sequence_id for image in candidates.values() if image.sequence_id
            }

            if len(candidates) >= limit:
                break

            if len(candidates) >= min(limit, 20) and len(unique_sequences) >= 6:
                break

        if successful_requests == 0 and last_error is not None:
            raise last_error

        return list(candidates.values())[:limit]

    def download_image(
        self,
        image: MapillaryImage,
        destination: Path,
    ) -> None:
        response = self.download_session.get(
            image.thumbnail_url,
            timeout=(10, 60),
        )

        if not response.ok:
            raise RuntimeError(f"Image download HTTP {response.status_code}")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = destination.with_suffix(".tmp")

        try:
            with Image.open(BytesIO(response.content)) as source:
                source.load()

                if source.width < 320 or source.height < 240:
                    raise ValueError("Image resolution too small")

                source.convert("RGB").save(
                    temporary_path,
                    format="JPEG",
                    quality=95,
                )

            temporary_path.replace(destination)

        finally:
            temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        self.session.close()
        self.download_session.close()
