from datetime import datetime, timezone

from sky.ephemeris import Ephemeris


def test_returns_expected_bodies():
    eph = Ephemeris(37.7749, -122.4194, 0.0)
    when = datetime(2026, 6, 21, 4, 0, 0, tzinfo=timezone.utc)
    names = {b.name for b in eph.bodies(when)}
    assert {"Sun", "Moon", "Mars", "Jupiter", "Venus", "Saturn", "Mercury"} <= names


def test_radec_ranges_valid():
    eph = Ephemeris(37.7749, -122.4194, 0.0)
    for b in eph.bodies(datetime(2026, 6, 21, 4, 0, 0, tzinfo=timezone.utc)):
        assert 0 <= b.ra_hours < 24
        assert -90 <= b.dec_deg <= 90
