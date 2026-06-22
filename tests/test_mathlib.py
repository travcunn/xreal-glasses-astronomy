import numpy as np
from mathlib import (
    quat_identity, quat_mul, quat_normalize, quat_from_rotvec,
    quat_conjugate, quat_to_matrix, rotate_vector, quat_from_matrix,
)


def test_identity_rotates_nothing():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rotate_vector(quat_identity(), v), v)


def test_90deg_about_z_maps_x_to_y():
    q = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    out = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-6)


def test_matrix_is_orthonormal():
    q = quat_normalize(quat_from_rotvec(np.array([0.3, -0.7, 1.1])))
    m = quat_to_matrix(q)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-6)


def test_conjugate_inverts_rotation():
    q = quat_normalize(quat_from_rotvec(np.array([0.2, 0.5, -0.3])))
    v = np.array([0.4, -1.0, 2.0])
    back = rotate_vector(quat_conjugate(q), rotate_vector(q, v))
    assert np.allclose(back, v, atol=1e-6)


def test_mul_composes_rotations():
    qz = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    composed = quat_mul(qz, qz)  # 180 deg about z
    out = rotate_vector(composed, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-6)


def test_matrix_quat_roundtrip():
    q = quat_normalize(quat_from_rotvec(np.array([0.4, -0.9, 1.3])))
    m = quat_to_matrix(q)
    q2 = quat_from_matrix(m)
    # Quaternions are double-cover; compare via the matrix they produce.
    assert np.allclose(quat_to_matrix(q2), m, atol=1e-6)


def test_yaw_reflection_inverts_yaw_keeps_pitch():
    # Conjugating by diag(1,-1,1) must negate yaw (about Z) and preserve pitch (about Y).
    flip = np.diag([1.0, -1.0, 1.0])
    yaw = quat_from_rotvec(np.array([0.0, 0.0, 0.5]))
    reflected = quat_from_matrix(flip @ quat_to_matrix(yaw) @ flip)
    assert np.allclose(quat_to_matrix(reflected), quat_to_matrix(quat_from_rotvec(np.array([0.0, 0.0, -0.5]))), atol=1e-6)
    pitch = quat_from_rotvec(np.array([0.0, 0.6, 0.0]))
    reflected_p = quat_from_matrix(flip @ quat_to_matrix(pitch) @ flip)
    assert np.allclose(quat_to_matrix(reflected_p), quat_to_matrix(pitch), atol=1e-6)
