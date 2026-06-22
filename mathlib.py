"""Quaternion and vector helpers. Quaternions are [w, x, y, z] numpy arrays."""

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        return quat_identity()
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle in radians) -> quaternion."""
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return quat_identity()
    axis = v / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_right_offset(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Offset `o` such that `quat_mul(current, o) == target`.

    For unit quaternions, o = current^-1 * target = conj(current) * target. Used
    to calibrate a neutral pose: capture the live `current` head orientation and
    get the body-frame offset that makes it read as the canonical `target` view.
    """
    return quat_normalize(quat_mul(quat_conjugate(quat_normalize(current)), target))


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation from `a` to `b` by fraction `t` in [0, 1].

    Takes the shortest arc (handles quaternion double-cover) and falls back to
    normalized linear interpolation when the inputs are nearly parallel, where
    sin(theta) -> 0 would otherwise blow up.
    """
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(a @ b)
    if dot < 0.0:           # a and -b are the same rotation; flip to the short way
        b = -b
        dot = -dot
    if dot > 0.9995:        # nearly identical -> lerp + renormalize
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    wa = np.sin((1.0 - t) * theta) / sin_theta
    wb = np.sin(t * theta) / sin_theta
    return quat_normalize(wa * a + wb * b)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q) @ v


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3, proper) -> quaternion [w, x, y, z]."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))
