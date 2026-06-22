"""Complementary filter: gyro + accel -> orientation quaternion (gravity = up)."""

import numpy as np

import config
from mathlib import (
    quat_identity, quat_mul, quat_normalize, quat_from_rotvec,
    quat_to_matrix, rotate_vector,
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
