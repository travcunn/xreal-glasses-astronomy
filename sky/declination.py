"""Magnetic declination (degrees east) from the World Magnetic Model."""

import config


def declination_deg(lat: float, lon: float, year: float, geomag_factory=None) -> float:
    try:
        if geomag_factory is None:
            from pygeomag import GeoMag
            geomag_factory = GeoMag
        geo = geomag_factory()
        result = geo.calculate(glat=lat, glon=lon, alt=0, time=year)
        return float(result.d)
    except Exception:
        return config.MAGNETIC_DECLINATION_DEG
