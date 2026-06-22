"""Skyfield-backed positions for the Sun, Moon, and naked-eye planets."""

from dataclasses import dataclass
from datetime import datetime, timezone

from skyfield.api import load, wgs84


@dataclass
class Body:
    name: str
    ra_hours: float
    dec_deg: float
    magnitude: float
    kind: str


# Rough apparent magnitudes (good enough for sizing a billboard).
_PLANETS = {
    "Mercury": ("mercury", 0.0, "planet"),
    "Venus": ("venus", -4.0, "planet"),
    "Mars": ("mars", 0.7, "planet"),
    "Jupiter": ("jupiter barycenter", -2.2, "planet"),
    "Saturn": ("saturn barycenter", 0.5, "planet"),
}


class Ephemeris:
    def __init__(self, lat_deg: float, lon_deg: float, elevation_m: float):
        self._ts = load.timescale()
        self._eph = load("de421.bsp")
        self._observer = self._eph["earth"] + wgs84.latlon(lat_deg, lon_deg, elevation_m)

    def _radec(self, target, t):
        ra, dec, _ = self._observer.at(t).observe(target).apparent().radec()
        return ra.hours, dec.degrees

    def bodies(self, when: datetime | None = None) -> list[Body]:
        when = when or datetime.now(timezone.utc)
        t = self._ts.from_datetime(when)
        out: list[Body] = []

        ra, dec = self._radec(self._eph["sun"], t)
        out.append(Body("Sun", ra, dec, -26.7, "sun"))

        ra, dec = self._radec(self._eph["moon"], t)
        out.append(Body("Moon", ra, dec, -12.7, "moon"))

        for name, (key, mag, kind) in _PLANETS.items():
            ra, dec = self._radec(self._eph[key], t)
            out.append(Body(name, ra, dec, mag, kind))
        return out
