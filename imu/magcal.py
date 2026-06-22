"""Hard-iron magnetometer calibration: estimate and remove a fixed offset."""

import json

import numpy as np


class MagCalibration:
    def __init__(self, offset: np.ndarray | None = None):
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, float)
        self._min = np.full(3, np.inf)
        self._max = np.full(3, -np.inf)
        self.samples_count = 0

    def collect(self, sample: np.ndarray) -> None:
        s = np.asarray(sample, float)
        self._min = np.minimum(self._min, s)
        self._max = np.maximum(self._max, s)
        self.samples_count += 1

    def finish(self) -> None:
        if self.samples_count > 0 and np.all(np.isfinite(self._min)):
            self.offset = (self._min + self._max) / 2.0

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, float) - self.offset

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"offset": self.offset.tolist()}, fh)

    @classmethod
    def load(cls, path: str) -> "MagCalibration":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(offset=np.array(data["offset"], float))
