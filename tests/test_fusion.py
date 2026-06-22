import numpy as np
from imu.fusion import ComplementaryFilter, OrientationSmoother
from mathlib import (
    rotate_vector, quat_identity, quat_from_rotvec, quat_to_matrix,
    quat_angle_between,
)


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
    s = OrientationSmoother(min_cutoff=1.0, beta=1.0)
    target = quat_from_rotvec(np.array([0.0, 0.0, 0.5]))
    out = s.update(target, dt=0.016)
    assert np.allclose(quat_to_matrix(out), quat_to_matrix(target), atol=1e-6)


def test_smoother_constant_input_stays_put():
    # A steady head pose must not be perturbed (no added wobble at rest).
    s = OrientationSmoother(min_cutoff=1.0, beta=1.0)
    target = quat_from_rotvec(np.array([0.0, 0.0, np.radians(15)]))
    out = None
    for _ in range(50):
        out = s.update(target, dt=0.016)
    assert quat_angle_between(out, target) < 1e-6


def test_smoother_attenuates_low_amplitude_jitter():
    # Tiny back-and-forth shake around a center should come out much smaller.
    s = OrientationSmoother(min_cutoff=1.0, beta=1.0)
    center = quat_identity()
    amp = np.radians(0.5)
    s.update(center, dt=0.016)
    out_dev = []
    for i in range(300):
        raw = quat_from_rotvec(np.array([0.0, 0.0, amp if i % 2 == 0 else -amp]))
        out = s.update(raw, dt=0.016)
        out_dev.append(quat_angle_between(out, center))
    # Input swings +/- amp every frame; output must be heavily damped.
    assert max(out_dev[-50:]) < 0.25 * amp


def test_smoother_tracks_fast_motion_with_less_lag_than_fixed():
    # Adaptivity: a sustained fast turn should lag less with beta>0 than beta=0.
    def run(beta):
        s = OrientationSmoother(min_cutoff=1.0, beta=beta)
        rate = 3.0  # rad/s about z
        ang = 0.0
        out = raw = quat_identity()
        for _ in range(120):
            ang += rate * 0.016
            raw = quat_from_rotvec(np.array([0.0, 0.0, ang]))
            out = s.update(raw, dt=0.016)
        return quat_angle_between(out, raw)  # steady-state lag

    fixed_lag = run(beta=0.0)
    adaptive_lag = run(beta=2.0)
    assert adaptive_lag < 0.5 * fixed_lag


def test_smoother_min_cutoff_zero_disables_smoothing():
    s = OrientationSmoother(min_cutoff=0.0, beta=1.0)
    s.update(quat_identity(), dt=0.016)
    target = quat_from_rotvec(np.array([0.0, 0.0, 0.7]))
    assert np.allclose(quat_to_matrix(s.update(target, dt=0.016)),
                       quat_to_matrix(target), atol=1e-6)
