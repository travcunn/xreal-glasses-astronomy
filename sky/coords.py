"""Pure coordinate transforms: sidereal time and equatorial <-> horizontal.

World frame convention: X = North, Y = East, Z = Up. Azimuth is measured from
North toward East.
"""

import numpy as np


def gmst_hours(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time in hours (IAU 1982 polynomial)."""
    d = jd_ut1 - 2451545.0
    t = d / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return (gmst_sec / 3600.0) % 24.0


def lst_hours(jd_ut1: float, longitude_deg: float) -> float:
    return (gmst_hours(jd_ut1) + longitude_deg / 15.0) % 24.0


def radec_to_altaz(ra_hours: float, dec_deg: float,
                   lat_deg: float, lst_h: float) -> tuple[float, float]:
    ha = np.radians((lst_h - ra_hours) * 15.0)  # hour angle
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    cos_az = (np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat))
    az = np.arccos(np.clip(cos_az, -1, 1))
    if np.sin(ha) > 0:
        az = 2 * np.pi - az
    return np.degrees(alt), np.degrees(az)


def altaz_to_unit(alt_deg: float, az_deg: float) -> np.ndarray:
    alt = np.radians(alt_deg)
    az = np.radians(az_deg)
    return np.array([
        np.cos(alt) * np.cos(az),  # North
        np.cos(alt) * np.sin(az),  # East
        np.sin(alt),               # Up
    ])


def radec_to_equatorial_unit(ra_hours: float, dec_deg: float) -> np.ndarray:
    ra = np.radians(ra_hours * 15.0)
    dec = np.radians(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def equatorial_to_horizontal_matrix(lat_deg: float, lst_h: float) -> np.ndarray:
    """3x3 mapping an equatorial unit vector to the (N, E, Up) world frame."""
    theta = np.radians(lst_h * 15.0)  # rotate by LST about equatorial Z
    lat = np.radians(lat_deg)
    # Rotate into the local meridian, then tilt by colatitude.
    rot_lst = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    colat = np.pi / 2 - lat
    tilt = np.array([
        [np.cos(colat), 0, -np.sin(colat)],
        [0, 1, 0],
        [np.sin(colat), 0, np.cos(colat)],
    ])
    # Result rows: North, East, Up. East is +Y after the LST rotation flips sign.
    m = tilt @ rot_lst
    m[1] = -m[1]  # orient azimuth North-through-East
    return m
