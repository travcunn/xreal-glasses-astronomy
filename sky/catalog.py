"""Load the Hipparcos star catalog and Stellarium constellation lines."""

from dataclasses import dataclass

import numpy as np
from skyfield.api import load
from skyfield.data import hipparcos

import config


# Common proper names for the brightest naked-eye stars, keyed by HIP id.
STAR_NAMES = {
    32349: "Sirius", 30438: "Canopus", 69673: "Arcturus", 91262: "Vega",
    24608: "Capella", 24436: "Rigel", 37279: "Procyon", 27989: "Betelgeuse",
    97649: "Altair", 21421: "Aldebaran", 80763: "Antares", 65474: "Spica",
    37826: "Pollux", 113368: "Fomalhaut", 102098: "Deneb", 49669: "Regulus",
    11767: "Polaris",
}


@dataclass
class StarData:
    ra_hours: np.ndarray
    dec_deg: np.ndarray
    magnitude: np.ndarray
    bv: np.ndarray
    hip_index: dict[int, int]


def load_stars(mag_limit: float = config.MAG_LIMIT) -> StarData:
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
    df = df[df["magnitude"].notnull() & (df["magnitude"] <= mag_limit)]
    df = df[df["ra_degrees"].notnull() & df["dec_degrees"].notnull()]
    ra_hours = df["ra_degrees"].to_numpy() / 15.0
    dec_deg = df["dec_degrees"].to_numpy()
    magnitude = df["magnitude"].to_numpy()
    bv = df["bv"].fillna(0.0).to_numpy() if "bv" in df else np.zeros(len(df))
    hip_index = {int(hip): i for i, hip in enumerate(df.index.to_numpy())}
    return StarData(ra_hours, dec_deg, magnitude, bv, hip_index)


def parse_constellation_lines(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        ids = [int(t) for t in tokens[2:]]
        for i in range(0, len(ids) - 1, 2):
            pairs.append((ids[i], ids[i + 1]))
    return pairs


def load_constellation_lines(path: str = "data/constellationship.fab") -> list[tuple[int, int]]:
    with open(path, encoding="utf-8") as fh:
        return parse_constellation_lines(fh.read())
