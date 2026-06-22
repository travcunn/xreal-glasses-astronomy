"""Complementary filter: gyro + accel -> orientation quaternion (gravity = up)."""

import numpy as np

import config
from mathlib import (
    quat_identity, quat_mul, quat_normalize, quat_from_rotvec,
    quat_to_matrix, rotate_vector, quat_slerp,
)


class ComplementaryFilter:
    def __init__(self, accel_gain: float = config.ACCEL_GAIN):
        self.q = quat_identity()
        self.accel_gain = accel_gain

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        # 1. Integrate gyro (body-frame angular velocity) into orientation.
        self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(gyro * dt)))

        # 2. Gravity correction (pitch/roll only). Skip if accel gain is zero or
        #    the reading is not ~1 g (during fast head motion accel is unreliable).
        norm = np.linalg.norm(accel)
        if self.accel_gain > 0 and norm > 1e-6:
            measured_up_world = rotate_vector(self.q, accel / norm)
            world_up = np.array([0.0, 0.0, 1.0])
            # Rotation that would bring the estimate's up onto true world up.
            axis = np.cross(measured_up_world, world_up)
            s = np.linalg.norm(axis)
            if s > 1e-9:
                angle = np.arctan2(s, measured_up_world @ world_up)
                correction = quat_from_rotvec(axis / s * angle * self.accel_gain)
                self.q = quat_normalize(quat_mul(correction, self.q))
        return self.q

    def matrix(self) -> np.ndarray:
        return quat_to_matrix(self.q)


class OrientationSmoother:
    """Low-pass filter on orientation: damps high-frequency head jitter.

    Each update slerps the held orientation a fraction of the way toward the new
    target. The fraction is derived from a time constant `tau` (seconds) so the
    feel is frame-rate independent: alpha = 1 - exp(-dt / tau). Small tau -> light
    smoothing with little lag; larger tau -> heavier smoothing, more lag. tau = 0
    is a pass-through (no smoothing).

    This is a deliberate jitter-vs-latency trade: a longer tau is steadier but
    lags real head motion, which in a head-locked view reads as the world
    "swimming." Keep it slight.
    """

    def __init__(self, tau: float):
        self.tau = tau
        self.q: np.ndarray | None = None

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self.q is None or self.tau <= 0.0:   # first sample / disabled: snap to target
            self.q = quat_normalize(target)
            return self.q
        alpha = 1.0 - np.exp(-dt / self.tau)
        self.q = quat_slerp(self.q, target, alpha)
        return self.q
