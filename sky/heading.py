"""Heading helpers for magnetometer-based real-sky alignment (pure)."""

import numpy as np


def azimuth_of_magnetic_north(mag_world: np.ndarray) -> float:
    """World-frame azimuth (deg, North-through-East, [0,360)) of the mag vector.

    `mag_world` must already be rotated into the gravity-referenced world frame
    (X=North, Y=East, Z=Up), which makes this tilt-compensated.
    """
    north, east = float(mag_world[0]), float(mag_world[1])
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _wrap(angle_rad: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def slew_angle(current_rad: float, target_rad: float, gain: float) -> float:
    """Move `current` toward `target` by `gain` along the shortest path."""
    delta = _wrap(target_rad - current_rad)
    return _wrap(current_rad + gain * delta)


def azimuth_align_delta(gaze_world: np.ndarray, target_world: np.ndarray) -> float:
    """Yaw to add to the view (radians, about world up) so the camera's azimuth

    points at `target_world`. `gaze_world` is the current camera-forward direction
    and `target_world` is a known object's world direction (e.g. the Moon); both in
    the (North=x, East=y, Up=z) frame. Only azimuth is corrected (altitude is
    gravity-referenced and stays untouched); the result takes the shortest arc.
    """
    gaze_az = np.arctan2(gaze_world[1], gaze_world[0])
    target_az = np.arctan2(target_world[1], target_world[0])
    return _wrap(target_az - gaze_az)


def compute_yaw_target(mag_world: np.ndarray, declination_deg: float) -> float:
    """Radians to offset the rendered azimuth so true north -> screen az 0.

    Magnetic north sits at true azimuth = declination (east positive); its
    measured world-frame azimuth is `azimuth_of_magnetic_north`. To display it at
    azimuth `declination`, the offset must be (declination - measured).
    """
    mag_az = azimuth_of_magnetic_north(mag_world)
    return _wrap(np.radians(declination_deg) - np.radians(mag_az))
