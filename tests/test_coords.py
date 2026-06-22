import numpy as np
from skyfield.api import load, wgs84, Star

from sky.coords import gmst_hours, lst_hours, radec_to_altaz, altaz_to_unit


# A fixed instant and place for deterministic comparison.
LAT, LON = 37.7749, -122.4194
RA_HOURS, DEC_DEG = 6.7525, -16.7161  # Sirius (ICRS)


def _skyfield_altaz():
    ts = load.timescale()
    t = ts.utc(2026, 6, 21, 4, 0, 0)
    eph = load("de421.bsp")
    observer = eph["earth"] + wgs84.latlon(LAT, LON)
    star = Star(ra_hours=RA_HOURS, dec_degrees=DEC_DEG)
    alt, az, _ = observer.at(t).observe(star).apparent().altaz()
    return t, alt.degrees, az.degrees


def test_altaz_matches_skyfield_within_tolerance():
    t, sf_alt, sf_az = _skyfield_altaz()
    lst = lst_hours(t.ut1, LON)
    alt, az = radec_to_altaz(RA_HOURS, DEC_DEG, LAT, lst)
    assert abs(alt - sf_alt) < 0.3
    assert abs(((az - sf_az + 180) % 360) - 180) < 0.3


def test_altaz_to_unit_is_unit_and_oriented():
    # Zenith -> straight up.
    assert np.allclose(altaz_to_unit(90, 0), [0, 0, 1], atol=1e-9)
    # North horizon -> +X; East horizon -> +Y.
    assert np.allclose(altaz_to_unit(0, 0), [1, 0, 0], atol=1e-9)
    assert np.allclose(altaz_to_unit(0, 90), [0, 1, 0], atol=1e-9)


def test_gmst_in_range():
    ts = load.timescale()
    t = ts.utc(2026, 6, 21, 4, 0, 0)
    g = gmst_hours(t.ut1)
    assert 0 <= g < 24
