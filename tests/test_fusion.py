import numpy as np
from imu.fusion import ComplementaryFilter, OrientationSmoother
from mathlib import rotate_vector, quat_identity, quat_from_rotvec, quat_to_matrix


def _angle_about_z(q):
    x = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    return np.degrees(np.arctan2(x[1], x[0]))


def _settle(f, gyro, accel, steps=2000, dt=0.001):
    for _ in range(steps):
        f.update(np.array(gyro, float), np.array(accel, float), dt)
    return f.q


def test_level_glasses_estimate_up_is_world_up():
    # Accel reads gravity straight up along body z; body should align with world.
    f = ComplementaryFilter(accel_gain=0.05)
    q = _settle(f, [0, 0, 0], [0, 0, 9.81])
    body_up_in_world = rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(body_up_in_world, [0, 0, 1], atol=1e-2)


def test_tilt_recovered_from_accel():
    # Glasses pitched so gravity splits between body z and body x.
    g = 9.81
    accel = [g * np.sin(np.radians(30)), 0.0, g * np.cos(np.radians(30))]
    f = ComplementaryFilter(accel_gain=0.05)
    q = _settle(f, [0, 0, 0], accel)
    up = rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    # Estimated up should be tilted ~30 deg from world up.
    tilt = np.degrees(np.arccos(np.clip(up @ np.array([0, 0, 1.0]), -1, 1)))
    assert abs(tilt - 30) < 3


def test_gyro_yaw_integrates():
    # Constant yaw rate about world up, gravity steady -> yaw advances ~ rate*time.
    f = ComplementaryFilter(accel_gain=0.0)  # no correction, pure integration
    rate = np.radians(20)  # 20 deg/s about body z
    _settle(f, [0, 0, rate], [0, 0, 9.81], steps=1000, dt=0.001)  # 1 s
    x_in_world = rotate_vector(f.q, np.array([1.0, 0.0, 0.0]))
    yaw = np.degrees(np.arctan2(x_in_world[1], x_in_world[0]))
    assert abs(yaw - 20) < 2


def test_smoother_first_sample_passes_through():
    # No history yet -> output is the input (no startup glide from identity).
    s = OrientationSmoother(tau=0.1)
    target = quat_from_rotvec(np.array([0.0, 0.0, 0.5]))
    out = s.update(target, dt=0.016)
    assert np.allclose(quat_to_matrix(out), quat_to_matrix(target), atol=1e-6)


def test_smoother_lags_then_converges_to_target():
    s = OrientationSmoother(tau=0.1)
    s.update(quat_identity(), dt=0.016)              # establish history at 0 deg
    target = quat_from_rotvec(np.array([0.0, 0.0, np.radians(30)]))
    out = s.update(target, dt=0.016)                 # one step toward 30 deg
    moved = _angle_about_z(out)
    assert 0.0 < moved < 30.0                        # lags: partway, not all the way
    for _ in range(400):                             # then settles onto the target
        out = s.update(target, dt=0.016)
    assert abs(_angle_about_z(out) - 30.0) < 0.1


def test_smoother_is_framerate_independent():
    # Same elapsed time, different step sizes -> nearly the same amount of catch-up.
    target = quat_from_rotvec(np.array([0.0, 0.0, np.radians(40)]))

    coarse = OrientationSmoother(tau=0.1)
    coarse.update(quat_identity(), dt=0.05)
    coarse.update(target, dt=0.05)

    fine = OrientationSmoother(tau=0.1)
    fine.update(quat_identity(), dt=0.01)
    for _ in range(5):                               # 5 * 0.01 == one 0.05 step
        fine.update(target, dt=0.01)

    assert abs(_angle_about_z(coarse.q) - _angle_about_z(fine.q)) < 1.5


def test_smoother_tau_zero_disables_smoothing():
    s = OrientationSmoother(tau=0.0)
    s.update(quat_identity(), dt=0.016)
    target = quat_from_rotvec(np.array([0.0, 0.0, 0.7]))
    assert np.allclose(quat_to_matrix(s.update(target, dt=0.016)), quat_to_matrix(target), atol=1e-6)
