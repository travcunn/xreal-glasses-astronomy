import numpy as np
from mathlib import (
    quat_identity, quat_mul, quat_normalize, quat_from_rotvec,
    quat_conjugate, quat_to_matrix, rotate_vector, quat_from_matrix,
    quat_slerp, quat_right_offset, quat_angle_between, quat_to_rotvec,
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


def test_slerp_endpoints_return_inputs():
    a = quat_from_rotvec(np.array([0.0, 0.0, 0.0]))
    b = quat_from_rotvec(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(quat_to_matrix(quat_slerp(a, b, 0.0)), quat_to_matrix(a), atol=1e-6)
    assert np.allclose(quat_to_matrix(quat_slerp(a, b, 1.0)), quat_to_matrix(b), atol=1e-6)


def test_slerp_midpoint_is_half_the_rotation():
    a = quat_identity()
    b = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))  # 90 deg about z
    mid = quat_slerp(a, b, 0.5)                            # should be 45 deg about z
    out = rotate_vector(mid, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0], atol=1e-6)


def test_slerp_takes_shortest_arc_across_double_cover():
    # b and -b are the same rotation; slerp must pick the short way regardless of sign.
    a = quat_identity()
    b = quat_from_rotvec(np.array([0.0, 0.0, np.radians(20)]))
    mid_pos = quat_slerp(a, b, 0.5)
    mid_neg = quat_slerp(a, -b, 0.5)
    assert np.allclose(quat_to_matrix(mid_pos), quat_to_matrix(mid_neg), atol=1e-6)


def test_slerp_near_identical_inputs_stable():
    a = quat_from_rotvec(np.array([0.0, 0.0, 0.10]))
    b = quat_from_rotvec(np.array([0.0, 0.0, 0.10 + 1e-9]))
    out = quat_slerp(a, b, 0.5)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)  # no NaN from sin(theta)~0


def test_rotvec_roundtrips_through_quat():
    v = np.array([0.3, -0.7, 0.4])
    assert np.allclose(quat_to_rotvec(quat_from_rotvec(v)), v, atol=1e-6)


def test_rotvec_of_identity_is_zero():
    assert np.allclose(quat_to_rotvec(quat_identity()), [0.0, 0.0, 0.0], atol=1e-9)


def test_rotvec_takes_shortest_arc():
    # A rotation just past pi about z should read as a small negative rotation.
    v = np.array([0.0, 0.0, np.radians(350)])
    out = quat_to_rotvec(quat_from_rotvec(v))
    assert np.allclose(out, [0.0, 0.0, np.radians(-10)], atol=1e-6)


def test_angle_between_identical_is_zero():
    q = quat_normalize(quat_from_rotvec(np.array([0.3, -0.6, 0.2])))
    assert abs(quat_angle_between(q, q)) < 1e-9


def test_angle_between_is_rotation_angle():
    a = quat_identity()
    b = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))  # 90 deg
    assert abs(quat_angle_between(a, b) - np.pi / 2) < 1e-6


def test_angle_between_ignores_double_cover():
    a = quat_identity()
    b = quat_from_rotvec(np.array([0.0, 0.0, np.radians(30)]))
    assert abs(quat_angle_between(a, b) - quat_angle_between(a, -b)) < 1e-9


def test_right_offset_maps_current_onto_target():
    # The calibration invariant: current @ offset == target, for any orientation.
    current = quat_normalize(quat_from_rotvec(np.array([0.4, -1.1, 0.7])))
    target = quat_from_rotvec(np.array([np.pi / 2, 0.0, 0.0]))  # BASE_VIEW-like
    offset = quat_right_offset(current, target)
    composed = quat_mul(current, offset)
    assert np.allclose(quat_to_matrix(composed), quat_to_matrix(target), atol=1e-6)


def test_right_offset_of_self_is_identity_rotation():
    q = quat_normalize(quat_from_rotvec(np.array([0.2, 0.9, -0.5])))
    offset = quat_right_offset(q, q)
    assert np.allclose(quat_to_matrix(offset), np.eye(3), atol=1e-6)


def test_yaw_reflection_inverts_yaw_keeps_pitch():
    # Conjugating by diag(1,-1,1) must negate yaw (about Z) and preserve pitch (about Y).
    flip = np.diag([1.0, -1.0, 1.0])
    yaw = quat_from_rotvec(np.array([0.0, 0.0, 0.5]))
    reflected = quat_from_matrix(flip @ quat_to_matrix(yaw) @ flip)
    assert np.allclose(quat_to_matrix(reflected), quat_to_matrix(quat_from_rotvec(np.array([0.0, 0.0, -0.5]))), atol=1e-6)
    pitch = quat_from_rotvec(np.array([0.0, 0.6, 0.0]))
    reflected_p = quat_from_matrix(flip @ quat_to_matrix(pitch) @ flip)
    assert np.allclose(quat_to_matrix(reflected_p), quat_to_matrix(pitch), atol=1e-6)
