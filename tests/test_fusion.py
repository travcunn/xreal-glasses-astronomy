import numpy as np
from imu.fusion import ComplementaryFilter
from mathlib import rotate_vector


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
