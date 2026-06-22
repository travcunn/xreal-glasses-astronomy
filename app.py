"""Main loop: open a window (laptop dev or glasses), drive the camera, render.

Usage:
    uv run python app.py            # windowed laptop dev mode, mouse-drag to look
    uv run python app.py --glasses  # fullscreen on the glasses, head-tracked
"""

import os
import sys
import time

import numpy as np
import pygame
import moderngl
from skyfield.api import load as sf_load

import config
from mathlib import quat_mul, quat_from_rotvec, quat_to_matrix, rotate_vector, quat_from_matrix
from sky.catalog import load_stars, load_constellation_lines, STAR_NAMES
from sky.coords import (
    radec_to_equatorial_unit, equatorial_to_horizontal_matrix, lst_hours,
)
from sky.ephemeris import Ephemeris
from sky.location import resolve_location
from sky.declination import declination_deg
from sky.heading import compute_yaw_target, slew_angle
from imu.magcal import MagCalibration
from imu.fusion import OrientationSmoother
from render.scene import Scene, magnitude_to_size
from render.labels import make_label_texture

_BODY_COLORS = {
    "Sun": (1.0, 0.95, 0.6), "Moon": (0.9, 0.9, 0.9), "Mercury": (0.8, 0.8, 0.7),
    "Venus": (1.0, 1.0, 0.85), "Mars": (1.0, 0.5, 0.3),
    "Jupiter": (1.0, 0.85, 0.6), "Saturn": (0.9, 0.8, 0.6),
}

# Base camera tilt so a neutral pose looks at the horizon, not straight down.
# (The exact head->camera axis mapping is calibrated on the glasses; flip/swap
# these axes if motion feels mirrored.)
BASE_VIEW = quat_from_rotvec(np.array([np.pi / 2, 0.0, 0.0]))

# Reflect head orientation across the North-Up plane: inverts yaw (azimuth
# panning) while preserving pitch/roll and keeping a proper rotation (no image
# mirror). Makes the sky world-locked the right way: turn head -> reveal new sky.
_YAW_FLIP = np.diag([1.0, -1.0, 1.0])


def reflect_yaw(q: np.ndarray) -> np.ndarray:
    return quat_from_matrix(_YAW_FLIP @ quat_to_matrix(q) @ _YAW_FLIP)


def _pick_display(use_glasses: bool):
    sizes = pygame.display.get_desktop_sizes()
    if use_glasses:
        for i, size in enumerate(sizes):
            if tuple(size) == config.GLASSES_RESOLUTION:
                return i, tuple(size), True
        print("Glasses display (1920x1080) not found; using primary display.")
    return 0, (1280, 720), False


def main():
    use_glasses = "--glasses" in sys.argv
    pygame.init()
    display_index, size, fullscreen = _pick_display(use_glasses)
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else 0)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode(size, flags, display=display_index)
    ctx = moderngl.create_context()
    scene = Scene(ctx, *size)
    scene.load_horizon()

    # --- Observer location + magnetic declination ---
    if config.IP_GEOLOCATION:
        lat_deg, lon_deg = resolve_location(config.LATITUDE_DEG, config.LONGITUDE_DEG)
    else:
        lat_deg, lon_deg = config.LATITUDE_DEG, config.LONGITUDE_DEG
    print(f"observer location: {lat_deg:.3f}, {lon_deg:.3f}")
    declination = declination_deg(lat_deg, lon_deg, 2026.5)
    print(f"magnetic declination: {declination:+.1f} deg")

    # --- Star catalog (static equatorial unit vectors) ---
    stars = load_stars(config.MAG_LIMIT)
    star_eq = np.array([
        radec_to_equatorial_unit(ra, dec)
        for ra, dec in zip(stars.ra_hours, stars.dec_deg)
    ])
    star_sizes = np.array([magnitude_to_size(m, config.MAG_LIMIT) for m in stars.magnitude])
    star_colors = np.ones((len(stars.magnitude), 3), dtype="f4")  # white v1

    # --- Constellation line index ---
    pairs = load_constellation_lines()
    line_rows = [
        (stars.hip_index[a], stars.hip_index[b])
        for a, b in pairs
        if a in stars.hip_index and b in stars.hip_index
    ]
    line_a = np.array([r[0] for r in line_rows])
    line_b = np.array([r[1] for r in line_rows])

    # --- Bright-star name label textures (built once) ---
    star_label_tex = {}
    for hip, name in STAR_NAMES.items():
        if hip in stars.hip_index:
            star_label_tex[stars.hip_index[hip]] = (name, make_label_texture(ctx, name))

    # --- Ephemeris (Sun/Moon/planets) ---
    ephem = Ephemeris(lat_deg, lon_deg, config.ELEVATION_M)
    body_label_tex = {name: make_label_texture(ctx, name) for name in _BODY_COLORS}
    bodies = ephem.bodies()
    last_ephem = time.time()

    ts = sf_load.timescale()

    # --- Head orientation source ---
    reader = fusion = None
    if use_glasses:
        from imu.reader import IMUReader
        from imu.fusion import ComplementaryFilter
        reader = IMUReader()
        reader.start()
        fusion = ComplementaryFilter()

    # Light low-pass on the view orientation so small head jitter is not jagged.
    smoother = OrientationSmoother(config.VIEW_SMOOTHING_TAU)

    # --- Magnetometer calibration (loaded if present) ---
    mag_cal = (MagCalibration.load(config.MAG_CALIBRATION_PATH)
               if os.path.exists(config.MAG_CALIBRATION_PATH) else MagCalibration())
    calibrating = False

    leveling_tex = make_label_texture(ctx, "leveling...", size=36)

    yaw = pitch = 0.0          # dev-mode mouse look
    yaw_offset = 0.0           # re-center offset applied about world up
    current_az = 0.0           # current view azimuth (for the re-center key)
    mag_enabled = True         # M toggles the magnetometer heading anchor
    invert_azimuth = config.INVERT_AZIMUTH  # F toggles yaw panning direction
    start = time.time()
    last = start
    running = True

    while running:
        now = time.time()
        dt = max(now - last, 1e-3)
        last = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                yaw_offset -= current_az   # snap current heading to zero
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                if not calibrating:
                    mag_cal = MagCalibration()
                    calibrating = True
                    print("calibrating: rotate the glasses in all directions...")
                else:
                    mag_cal.finish()
                    mag_cal.save(config.MAG_CALIBRATION_PATH)
                    calibrating = False
                    print(f"calibration done: offset={mag_cal.offset}")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                mag_enabled = not mag_enabled
                print(f"magnetometer anchor: {'ON' if mag_enabled else 'OFF (free-look)'}")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                invert_azimuth = not invert_azimuth
                print(f"azimuth panning inverted: {invert_azimuth}")
            if event.type == pygame.MOUSEMOTION and event.buttons[0]:
                yaw += event.rel[0] * 0.005
                pitch += event.rel[1] * 0.005

        # Head orientation (gravity-referenced world frame).
        if reader is not None and reader.latest is not None:
            s = reader.latest
            head = fusion.update(s.gyro, s.accel, dt)
        else:
            head = quat_mul(quat_from_rotvec(np.array([0.0, 0.0, yaw])),
                            quat_from_rotvec(np.array([pitch, 0.0, 0.0])))

        # Smooth the orientation that drives the camera (damps high-frequency
        # jitter; intentional head turns still pass through).
        head = smoother.update(head, dt)

        # Flip yaw sense so the sky is world-locked the natural way.
        if invert_azimuth:
            head = reflect_yaw(head)

        # Magnetometer heading anchor (Approach A): slew yaw_offset toward true north.
        mag = reader.latest_mag if reader is not None else None
        if mag is not None:
            if calibrating:
                mag_cal.collect(mag)
            elif mag_enabled:
                mag_world = rotate_vector(head, mag_cal.apply(mag))
                target = compute_yaw_target(mag_world, declination)
                yaw_offset = slew_angle(yaw_offset, target, config.HEADING_GAIN)

        # Mount the camera on the head (right-multiply) so head motion maps to a
        # rigid, world-locked view; then apply the re-center offset about world up.
        cam_pre = quat_mul(head, BASE_VIEW)
        fwd = quat_to_matrix(cam_pre) @ np.array([0.0, 0.0, -1.0])
        current_az = float(np.arctan2(fwd[1], fwd[0]))  # N=x, E=y
        cam = quat_mul(quat_from_rotvec(np.array([0.0, 0.0, yaw_offset])), cam_pre)
        scene.set_camera(cam)

        # Rotate the static equatorial frame into the live horizontal frame.
        jd = ts.now().ut1
        lst = lst_hours(jd, config.LONGITUDE_DEG)
        rot = equatorial_to_horizontal_matrix(config.LATITUDE_DEG, lst)
        star_world = star_eq @ rot.T
        scene.load_stars(star_world, star_sizes, star_colors)

        if line_rows:
            seg = np.empty((len(line_rows) * 2, 3), dtype="f4")
            seg[0::2] = star_world[line_a]
            seg[1::2] = star_world[line_b]
            scene.load_lines(seg)

        if now - last_ephem > 1.0:
            bodies = ephem.bodies()
            last_ephem = now
        body_eq = np.array([radec_to_equatorial_unit(b.ra_hours, b.dec_deg) for b in bodies])
        body_world = body_eq @ rot.T
        body_sizes = np.array([28.0 if b.kind in ("sun", "moon") else 12.0 for b in bodies])
        body_colors = np.array([_BODY_COLORS[b.name] for b in bodies], dtype="f4")
        scene.load_bodies(body_world, body_sizes, body_colors)

        # Labels: bodies + bright stars.
        labels = []
        for b, pos in zip(bodies, body_world):
            tex, wh = body_label_tex[b.name]
            labels.append((pos, tex, wh))
        for idx, (_name, (tex, wh)) in star_label_tex.items():
            labels.append((star_world[idx], tex, wh))
        scene.load_labels(labels)

        # Leveling splash for the first second on glasses (filter settling).
        if use_glasses and now - start < 1.0:
            tex, wh = leveling_tex
            scene.draw_message(tex, wh)
        else:
            scene.render()

        pygame.display.flip()

    if reader is not None:
        reader.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
