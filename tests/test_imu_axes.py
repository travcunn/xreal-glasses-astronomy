"""Hardware gyro axis convention, measured on the worn glasses.

Fixture vectors are the real output of diagnose_imu_axes.py: the accelerometer
direction at rest (= +up, proper acceleration) and the raw gyro vector for each
of the three single-axis head motions. The bug these guard against: the gyro is
reported in a left-handed frame relative to the accelerometer, so integrating it
mirrors the orientation and the sky tracks the head instead of world-locking.
"""

import numpy as np

from imu.reader import to_body_gyro

ACCEL_UP = np.array([-0.062, -0.775, 0.629])   # accel at rest -> +up
YAW_RIGHT = np.array([0.019, 0.681, -0.732])   # raw gyro: turn head right
PITCH_UP = np.array([0.995, 0.075, -0.070])    # raw gyro: look up
ROLL_RIGHT = np.array([0.156, 0.810, 0.566])   # raw gyro: tilt right ear down


def _unit(v):
    return v / np.linalg.norm(v)


def test_raw_gyro_triple_is_left_handed():
    # Documents the defect: as reported, yaw x pitch points away from roll.
    assert np.cross(YAW_RIGHT, PITCH_UP) @ ROLL_RIGHT < 0


def test_corrected_gyro_triple_is_right_handed():
    y, p, r = to_body_gyro(YAW_RIGHT), to_body_gyro(PITCH_UP), to_body_gyro(ROLL_RIGHT)
    assert np.cross(y, p) @ r > 0


def test_correction_keeps_yaw_axis_aligned_with_gravity():
    # Yaw-right physically rotates about world-down (= -up). The correction must
    # not disturb that (it would, if it flipped Y or Z instead of X).
    world_down = -_unit(ACCEL_UP)
    assert _unit(to_body_gyro(YAW_RIGHT)) @ world_down > 0.95


def test_correction_only_flips_x():
    raw = np.array([1.0, 2.0, 3.0])
    assert np.allclose(to_body_gyro(raw), [-1.0, 2.0, 3.0])
