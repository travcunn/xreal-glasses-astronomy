"""Resolve observer location by IP geolocation, with a safe fallback."""

import json
import urllib.request


def _fetch_ipapi() -> dict:
    with urllib.request.urlopen("http://ip-api.com/json/", timeout=3) as resp:
        return json.load(resp)


def resolve_location(default_lat: float, default_lon: float, fetch=None) -> tuple[float, float]:
    fetch = fetch or _fetch_ipapi
    try:
        data = fetch()
        return float(data["lat"]), float(data["lon"])
    except Exception:
        return default_lat, default_lon
