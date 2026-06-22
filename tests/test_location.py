from sky.location import resolve_location


def test_uses_fetched_coordinates():
    lat, lon = resolve_location(0.0, 0.0, fetch=lambda: {"lat": 40.7, "lon": -74.0})
    assert (round(lat, 1), round(lon, 1)) == (40.7, -74.0)


def test_falls_back_on_error():
    def boom():
        raise RuntimeError("offline")
    assert resolve_location(37.77, -122.42, fetch=boom) == (37.77, -122.42)


def test_falls_back_on_bad_payload():
    assert resolve_location(1.0, 2.0, fetch=lambda: {"nope": 1}) == (1.0, 2.0)
