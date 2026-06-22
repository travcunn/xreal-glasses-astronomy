import numpy as np
from imu.magcal import MagCalibration


def _sphere_points(center, radius, n=400):
    pts = []
    for i in range(n):
        u = i / (n - 1)
        theta = 2 * np.pi * u
        phi = np.arccos(1 - 2 * ((i * 0.61803398875) % 1.0))
        d = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        pts.append(center + radius * d)
    return pts


def test_recovers_hard_iron_offset():
    center = np.array([12.0, -7.0, 30.0])
    cal = MagCalibration()
    for p in _sphere_points(center, 45.0):
        cal.collect(p)
    cal.finish()
    assert np.allclose(cal.offset, center, atol=2.0)


def test_apply_recenters():
    cal = MagCalibration(offset=np.array([5.0, 5.0, 5.0]))
    out = cal.apply(np.array([5.0, 6.0, 7.0]))
    assert np.allclose(out, [0.0, 1.0, 2.0])


def test_save_and_load(tmp_path):
    cal = MagCalibration(offset=np.array([1.0, 2.0, 3.0]))
    p = tmp_path / "cal.json"
    cal.save(str(p))
    loaded = MagCalibration.load(str(p))
    assert np.allclose(loaded.offset, [1.0, 2.0, 3.0])
