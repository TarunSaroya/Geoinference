from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0088


def haversine_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    for latitude in (latitude_1, latitude_2):
        if not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude")

    for longitude in (longitude_1, longitude_2):
        if not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude")

    lat_1 = radians(latitude_1)
    lon_1 = radians(longitude_1)

    lat_2 = radians(latitude_2)
    lon_2 = radians(longitude_2)

    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1

    a = sin(delta_lat / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(delta_lon / 2) ** 2

    a = min(1.0, max(0.0, a))

    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))
