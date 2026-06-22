import numpy as np
import pytest

from render.scene import perspective, view_from_quat, magnitude_to_size, fovy_from_diagonal
from mathlib import quat_identity


def test_fovy_from_diagonal_one_pro():
    # XREAL One Pro: 57 deg DIAGONAL on a 16:9 panel -> ~29.8 deg vertical.
    assert abs(fovy_from_diagonal(57.0, 16 / 9) - 29.81) < 0.1


def test_fovy_is_smaller_than_diagonal_on_widescreen():
    assert fovy_from_diagonal(57.0, 16 / 9) < 57.0


def test_fovy_from_diagonal_roundtrips():
    # Reconstruct the diagonal from the derived vertical + aspect.
    aspect = 16 / 9
    fovy = fovy_from_diagonal(57.0, aspect)
    tan_v = np.tan(np.radians(fovy) / 2.0)
    diag = np.degrees(2.0 * np.arctan(tan_v * np.sqrt(1.0 + aspect**2)))
    assert abs(diag - 57.0) < 1e-6


def test_perspective_shape_and_clip():
    p = perspective(57.0, 16 / 9, 0.1, 100.0)
    assert p.shape == (4, 4)
    near_pt = np.array([0, 0, -0.1, 1.0])
    clip = p @ near_pt
    assert clip[3] > 0


def test_view_identity_is_identity_rotation():
    v = view_from_quat(quat_identity())
    assert np.allclose(v[:3, :3], np.eye(3), atol=1e-6)


def test_brighter_stars_are_bigger():
    big = magnitude_to_size(-1.5, 6.5)   # Sirius
    small = magnitude_to_size(6.0, 6.5)  # near the limit
    assert big > small
    assert small > 0


def test_offscreen_render_produces_pixels():
    """Compile shaders and render stars to an offscreen buffer (no display)."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as e:  # no GL available in this environment
        pytest.skip(f"standalone GL context unavailable: {e}")

    from render.scene import Scene
    fbo = ctx.simple_framebuffer((256, 256))
    fbo.use()
    scene = Scene(ctx, 256, 256)

    # Three bright stars straight ahead of the identity camera (looking down -Z).
    positions = np.array([[0, 0, -1.0], [0.1, 0, -1.0], [-0.1, 0.05, -1.0]], dtype="f4")
    sizes = np.array([14.0, 12.0, 12.0], dtype="f4")
    colors = np.ones((3, 3), dtype="f4")
    scene.load_stars(positions, sizes, colors)
    scene.set_camera(quat_identity())
    scene.render()

    data = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
    # Something brighter than the near-black background must have been drawn.
    assert data.max() > 30


def test_offscreen_full_scene_all_layers():
    """Exercise every render path (stars, lines, bodies, labels, horizon, splash)."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as e:
        pytest.skip(f"standalone GL context unavailable: {e}")

    from render.scene import Scene
    from render.labels import make_label_texture

    fbo = ctx.simple_framebuffer((256, 256))
    fbo.use()
    scene = Scene(ctx, 256, 256)
    scene.load_horizon()

    pos = np.array([[0, 0, -1.0], [0.1, 0, -1.0]], dtype="f4")
    scene.load_stars(pos, np.array([12.0, 10.0], "f4"), np.ones((2, 3), "f4"))
    scene.load_lines(np.array([[0, 0, -1.0], [0.1, 0, -1.0]], dtype="f4"))
    scene.load_bodies(pos, np.array([28.0, 12.0], "f4"),
                      np.array([[1, 1, 0.6], [1, 0.5, 0.3]], "f4"))
    tex, wh = make_label_texture(ctx, "Sirius")
    scene.load_labels([(np.array([0.0, 0.0, -1.0]), tex, wh)])
    scene.set_camera(quat_identity())
    scene.render()      # full layered path
    scene.draw_message(tex, wh)  # splash path
    # No exception means all shaders/uniforms/draws are valid.
