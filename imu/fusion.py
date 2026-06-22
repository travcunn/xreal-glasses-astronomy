"""Complementary filter: gyro + accel -> orientation quaternion (gravity = up)."""

import numpy as np

import config
from mathlib import (
    quat_identity, quat_mul, quat_normalize, quat_from_rotvec,
    quat_to_matrix, rotate_vector, quat_slerp, quat_conjugate, quat_to_rotvec,
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


def _one_euro_alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low-pass with the given cutoff (Hz)."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OrientationSmoother:
    """One Euro adaptive low-pass on orientation (Casiez et al., CHI 2012).

    A fixed low-pass forces a single jitter-vs-lag trade: smooth enough to kill
    hand/head tremor and real turns lag and "swim." The One Euro filter adapts the
    cutoff to how fast the signal is actually moving:

        cutoff = min_cutoff + beta * |angular velocity|
        out    = slerp(out, target, alpha(cutoff, dt))

    At rest / tiny shakes the angular velocity is ~0, so the cutoff sits at
    `min_cutoff` (heavy smoothing -> jitter dies). During an intentional turn the
    velocity is high, the cutoff rises, and lag shrinks (no swimming).

    The speed estimate is the *directional* angular velocity (a vector), low-passed
    at `d_cutoff`, so back-and-forth jitter cancels instead of reading as fast
    motion. `min_cutoff <= 0` disables smoothing (pass-through). Tuning: lower
    `min_cutoff` for more stillness, raise `beta` for more responsiveness in motion.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.q: np.ndarray | None = None        # filtered orientation
        self._prev_raw: np.ndarray | None = None
        self._omega = np.zeros(3)               # low-passed angular velocity (rad/s)

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = quat_normalize(target)
        if self.q is None or self.min_cutoff <= 0.0:   # first sample / disabled
            self.q = target
            self._prev_raw = target
            self._omega = np.zeros(3)
            return self.q

        # Directional angular velocity of the raw signal, low-passed. Using the
        # rotation-vector delta (not a scalar speed) means symmetric jitter cancels.
        delta = quat_to_rotvec(quat_mul(quat_conjugate(self._prev_raw), target))
        omega = delta / dt
        a_d = _one_euro_alpha(self.d_cutoff, dt)
        self._omega = a_d * omega + (1.0 - a_d) * self._omega
        self._prev_raw = target

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._omega))
        self.q = quat_slerp(self.q, target, _one_euro_alpha(cutoff, dt))
        return self.q
