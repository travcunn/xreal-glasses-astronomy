import numpy as np
from sky.heading import azimuth_of_magnetic_north, slew_angle, compute_yaw_target


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
