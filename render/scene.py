"""moderngl scene and camera for the celestial sphere.

Everything is drawn on a unit sphere (objects at distance 1 from the camera at
the origin), so no depth testing is needed; draw order + alpha blending handle
layering. Labels are screen-space billboards drawn last.
"""

import numpy as np
import moderngl

import config
from mathlib import quat_conjugate, quat_to_matrix
from sky.coords import altaz_to_unit


def fovy_from_diagonal(diagonal_deg: float, aspect: float) -> float:
    """Vertical FOV (deg) from a diagonal FOV (deg) for a given width/height aspect.

    Glasses quote a single DIAGONAL FOV (the One Pro's 57 deg). The projection
    needs the vertical FOV. On the rectilinear image plane the half-extents satisfy
    tan(d/2)^2 = tan(h/2)^2 + tan(v/2)^2 with tan(h/2) = aspect * tan(v/2), so:

        tan(v/2) = tan(d/2) / sqrt(1 + aspect^2)

    For 57 deg diagonal on 16:9 this gives ~29.8 deg vertical (~50.6 deg horizontal).
    Matching this to the real optics is what makes the sky hold still: a head turn
    then sweeps stars across the display by the correct angle.
    """
    tan_v = np.tan(np.radians(diagonal_deg) / 2.0) / np.sqrt(1.0 + aspect**2)
    return float(np.degrees(2.0 * np.arctan(tan_v)))


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def view_from_quat(q: np.ndarray) -> np.ndarray:
    """Camera at origin. World is rotated by the inverse of head orientation."""
    r = quat_to_matrix(quat_conjugate(q))
    v = np.eye(4)
    v[:3, :3] = r
    return v


def magnitude_to_size(mag: float, mag_limit: float) -> float:
    # Brighter (smaller mag) -> larger point. Clamp to a sensible pixel range.
    size = 1.0 + 3.0 * max(0.0, (mag_limit - mag))
    return float(min(size, 14.0))


_STAR_VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in float in_size;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    gl_PointSize = in_size;
    v_color = in_color;
}
"""

_STAR_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    float a = smoothstep(0.5, 0.0, r);
    f_color = vec4(v_color, a);
}
"""

_LINE_VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); }
"""
_LINE_FS = """
#version 330
uniform vec4 color;
out vec4 f_color;
void main() { f_color = color; }
"""

_LABEL_VS = """
#version 330
uniform mat4 mvp;
uniform vec3 anchor;
uniform vec2 offset_px;
uniform vec2 size_px;
uniform vec2 screen;
in vec2 in_corner;
out vec2 v_uv;
void main() {
    vec4 clip = mvp * vec4(anchor, 1.0);
    if (clip.w <= 0.0) {        // behind camera: cull off-screen
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_uv = vec2(0.0);
        return;
    }
    vec2 ndc = clip.xy / clip.w;
    vec2 px = (ndc * 0.5 + 0.5) * screen + offset_px + in_corner * size_px;
    vec2 out_ndc = (px / screen) * 2.0 - 1.0;
    gl_Position = vec4(out_ndc, 0.0, 1.0);
    // Texture bytes are uploaded vertically flipped (origin bottom-left), so the
    // top of the quad (corner.y=1) maps to the top of the text (v=1).
    v_uv = vec2(in_corner.x, in_corner.y);
}
"""
_LABEL_FS = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(tex, v_uv); }
"""


class Scene:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.aspect = width / height
        fovy = fovy_from_diagonal(config.FOV_DIAGONAL_DEG, self.aspect)
        self.proj = perspective(fovy, self.aspect, 0.01, 10.0)
        self.view = np.eye(4)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._star_prog = ctx.program(vertex_shader=_STAR_VS, fragment_shader=_STAR_FS)
        self._line_prog = ctx.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        self._label_prog = ctx.program(vertex_shader=_LABEL_VS, fragment_shader=_LABEL_FS)

        corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="f4")
        self._corner_vbo = ctx.buffer(corners.tobytes())
        self._label_vao = ctx.vertex_array(
            self._label_prog, [(self._corner_vbo, "2f", "in_corner")]
        )

        self._star_vao = None
        self._line_vao = None
        self._body_vao = None
        self._horizon_vao = None
        self._labels: list[tuple[np.ndarray, moderngl.Texture, tuple[int, int]]] = []

    def set_camera(self, quat: np.ndarray):
        self.view = view_from_quat(quat)

    def mvp(self) -> np.ndarray:
        # column-major for GLSL: transpose of (proj @ view)
        return (self.proj @ self.view).T.astype("f4")

    def _points_vao(self, positions, sizes, colors):
        data = np.hstack([
            positions.astype("f4"),
            sizes.reshape(-1, 1).astype("f4"),
            colors.astype("f4"),
        ])
        vbo = self.ctx.buffer(data.tobytes())
        return self.ctx.vertex_array(
            self._star_prog,
            [(vbo, "3f 1f 3f", "in_pos", "in_size", "in_color")],
        )

    def load_stars(self, positions: np.ndarray, sizes: np.ndarray, colors: np.ndarray):
        self._star_vao = self._points_vao(positions, sizes, colors)

    def load_bodies(self, positions: np.ndarray, sizes: np.ndarray, colors: np.ndarray):
        self._body_vao = self._points_vao(positions, sizes, colors)

    def load_lines(self, positions: np.ndarray):
        vbo = self.ctx.buffer(positions.astype("f4").tobytes())
        self._line_vao = self.ctx.vertex_array(self._line_prog, [(vbo, "3f", "in_pos")])

    def load_horizon(self, segments: int = 180):
        az = np.linspace(0, 360, segments, endpoint=False)
        pts = np.array([altaz_to_unit(0.0, a) for a in az], dtype="f4")
        vbo = self.ctx.buffer(pts.tobytes())
        self._horizon_vao = self.ctx.vertex_array(self._line_prog, [(vbo, "3f", "in_pos")])

    def load_labels(self, labels):
        """labels: list of (anchor_vec3, texture, (w, h))."""
        self._labels = labels

    def draw_message(self, tex: moderngl.Texture, wh: tuple[int, int]):
        """Clear and draw a single texture centered on screen (e.g. a splash)."""
        self.ctx.clear(0.0, 0.0, 0.03)
        identity = np.eye(4, dtype="f4")
        self._label_prog["mvp"].write(identity.tobytes())
        self._label_prog["screen"].value = (self.width, self.height)
        self._label_prog["offset_px"].value = (-wh[0] / 2.0, -wh[1] / 2.0)
        self._label_prog["anchor"].value = (0.0, 0.0, -1.0)
        self._label_prog["size_px"].value = (float(wh[0]), float(wh[1]))
        tex.use(0)
        self._label_prog["tex"].value = 0
        self._label_vao.render(moderngl.TRIANGLE_STRIP)

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.03)  # near-black sky
        mvp = self.mvp()

        if self._horizon_vao is not None and config.SHOW_HORIZON:
            self._line_prog["mvp"].write(mvp.tobytes())
            self._line_prog["color"].value = (0.4, 0.4, 0.45, 0.5)
            self._horizon_vao.render(moderngl.LINE_LOOP)

        if self._star_vao is not None:
            self._star_prog["mvp"].write(mvp.tobytes())
            self._star_vao.render(moderngl.POINTS)

        if self._line_vao is not None:
            self._line_prog["mvp"].write(mvp.tobytes())
            self._line_prog["color"].value = (0.3, 0.5, 0.8, 0.5)
            self._line_vao.render(moderngl.LINES)

        if self._body_vao is not None:
            self._star_prog["mvp"].write(mvp.tobytes())
            self._body_vao.render(moderngl.POINTS)

        if self._labels:
            self._label_prog["mvp"].write(mvp.tobytes())
            self._label_prog["screen"].value = (self.width, self.height)
            self._label_prog["offset_px"].value = (8.0, -8.0)
            for anchor, tex, (w, h) in self._labels:
                self._label_prog["anchor"].value = tuple(float(x) for x in anchor)
                self._label_prog["size_px"].value = (float(w), float(h))
                tex.use(0)
                self._label_prog["tex"].value = 0
                self._label_vao.render(moderngl.TRIANGLE_STRIP)
