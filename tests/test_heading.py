import numpy as np
from sky.heading import (
    azimuth_of_magnetic_north, slew_angle, compute_yaw_target, azimuth_align_delta,
)


def test_azimuth_cardinals():
    assert abs(azimuth_of_magnetic_north(np.array([1.0, 0.0, 0.0])) - 0.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([0.0, 1.0, 0.0])) - 90.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([-1.0, 0.0, 0.0])) - 180.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([0.0, -1.0, 0.0])) - 270.0) < 1e-6


def test_azimuth_ignores_vertical():
    # Up component must not change the horizontal azimuth.
    assert abs(azimuth_of_magnetic_north(np.array([1.0, 1.0, 9.0])) - 45.0) < 1e-6


def test_slew_moves_fraction_toward_target():
    out = slew_angle(0.0, 1.0, 0.25)
    assert abs(out - 0.25) < 1e-9


def test_slew_takes_short_way_across_pi_seam():
    # current just below +pi, target just above -pi: shortest path crosses the seam.
    out = slew_angle(3.0, -3.0, 0.5)
    assert out > 3.0 or out < -3.0   # moved across the seam, not the long way


def test_yaw_target_puts_true_north_at_zero():
    # Magnetic north points due North in the world frame, declination +10 E.
    # Then magnetic north should display at azimuth +10 deg -> yaw_offset target = +10 deg.
    t = compute_yaw_target(np.array([1.0, 0.0, 0.0]), 10.0)
    assert abs(np.degrees(t) - 10.0) < 1e-6


def test_align_delta_rotates_gaze_azimuth_onto_target():
    # Looking North, target (Moon) is due East -> add +90 deg to swing the view east.
    gaze = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    assert abs(np.degrees(azimuth_align_delta(gaze, target)) - 90.0) < 1e-6


def test_align_delta_ignores_altitude():
    # Only the horizontal (azimuth) component matters; vertical offsets are ignored.
    gaze = np.array([1.0, 0.0, 0.6])      # looking N, tilted up
    target = np.array([0.0, 1.0, -0.4])   # E, below horizon
    assert abs(np.degrees(azimuth_align_delta(gaze, target)) - 90.0) < 1e-6


def test_align_delta_takes_short_way_across_seam():
    # Gaze az ~170 deg, target az ~-170 deg: shortest correction is +20, not -340.
    gaze = np.array([np.cos(np.radians(170)), np.sin(np.radians(170)), 0.0])
    target = np.array([np.cos(np.radians(-170)), np.sin(np.radians(-170)), 0.0])
    assert abs(np.degrees(azimuth_align_delta(gaze, target)) - 20.0) < 1e-6


def test_align_delta_zero_when_already_aligned():
    v = np.array([0.3, -0.7, 0.2])
    assert abs(azimuth_align_delta(v, v)) < 1e-9
