# XREAL One Pro Sky — v1 Design

**Date:** 2026-06-21
**Status:** Approved (design); implementation plan pending
**Target hardware:** XREAL One Pro glasses + macOS

## Goal

A native macOS Python prototype that renders the night sky to XREAL One Pro
glasses. The user wears the glasses, looks around, and sees stars, constellation
lines, planets, the Sun, the Moon, and labels move correctly with their head.

This is an explicit **prototype**: chosen to validate the full pipeline (head
tracking → orientation → sky math → rendering) cheaply before deciding whether
to port to a "real" stack (e.g. Swift/Metal). Quality bar is "feels right and
proves the concept," not "shippable product."

## Validated foundation (already proven on the actual hardware, 2026-06-21)

Head tracking works over a plain TCP socket — no SDK, no Nebula, no kernel
driver. Confirmed live on the user's One Pro + Mac:

- **Transport:** TCP to `169.254.2.1:52998` over the glasses' USB-ethernet
  link-local interface. No handshake/subscribe; the device streams immediately.
  Requires "developer mode" / Ethernet enabled in the glasses OSD (it was
  already on).
- **Framing:** fixed **134-byte records**, each starting with header
  `28 36 00 00 00 80`. (The community demo's HEADER..FOOTER delimiter framing
  does NOT apply to this firmware — no footer is sent.)
- **Payload:** six little-endian `float32` at **fixed offset 34** in each
  record = gyro x,y,z (rad/s) then accel x,y,z (m/s²).
- **IMU record tag:** marker `00 40 1f 00 00 40` near offset 78; only records
  with it carry IMU data. Other 134-byte record types are skipped.
- **Rate:** ~1000 Hz IMU samples. Verified accel magnitude = 9.79 m/s² (gravity)
  while stationary; gyro ≈ 0 at rest. Correctly scaled physical units.

Reference implementation of this parser already exists at
`/Users/tcunningham/xreal/probe_imu.py`, and a real capture is saved for use as
a test fixture (`scratchpad/capture.bin`, ~563 KB, 3000 IMU records).

Display side: the One Pro is a plug-and-play external monitor (1920×1080) over
USB-C DisplayPort; video "just works" with stock macOS drivers. Only the IMU
needs the socket.

## Scope decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Content | "The works": stars, constellation lines, planets, Sun, Moon, labels, optional Milky Way |
| Heading alignment | **Free-look** — geometrically correct sky, arbitrary compass heading (no magnetometer, no alignment ritual) |
| Vertical reference | Gravity-absolute: altitude (pitch/roll) is physically correct; only azimuth is free |
| Time | Live "now", updating continuously |
| Location | Configurable default latitude/longitude (drives which stars are overhead + planet positions) |
| Stereo | **Mono** for v1 (at infinity ≈ stereo); SBS/3D Mode deferred |
| Language/stack | Python prototype |
| Rendering | `moderngl` + `pygame` (3D scene, Approach A) |
| Astronomy data | `skyfield` (ephemeris + Hipparcos catalog) + Stellarium `constellationship.fab` |

## Architecture

### Module layout
```
xreal/
  pyproject.toml          # uv-managed
  config.py               # default lat/lon, magnitude limit, FOV, display index, toggles
  imu/
    reader.py             # TCP socket -> frame 134-byte records -> (gyro, accel) samples
    fusion.py             # gyro+accel -> orientation quaternion (gravity = up)
  sky/
    coords.py             # RA/Dec <-> alt/az, unit vectors (pure functions, unit-tested)
    catalog.py            # Hipparcos stars + constellation lines (HIP id pairs)
    ephemeris.py          # Skyfield: Sun/Moon/planets RA/Dec for now + location
  render/
    scene.py              # moderngl scene: stars, lines, planets, milky way, camera
    labels.py             # text -> texture billboards
  app.py                  # main loop: window on glasses, orientation -> camera, draw
  tests/
```

### Unit responsibilities
- **`imu/reader.py`** — owns the socket. Runs in a background thread, frames
  records, decodes IMU samples, publishes the latest `(gyro, accel, timestamp)`.
  Auto-reconnects on drop. Depends on: stdlib `socket`, `struct`.
- **`imu/fusion.py`** — pure-ish filter turning the gyro/accel stream into an
  orientation quaternion in a gravity-referenced world frame. v1 default is a
  **complementary filter**: integrate the gyro for fast response and nudge
  pitch/roll toward the accelerometer's gravity vector to kill drift (yaw is left
  free, matching free-look). Madgwick is an equivalent drop-in if the
  complementary filter proves too noisy. Depends on: `numpy`.
- **`sky/coords.py`** — pure coordinate math: equatorial (RA/Dec) → horizontal
  (alt/az) given latitude + local sidereal time; (alt, az) → unit vector;
  equatorial→horizontal rotation matrix. No I/O. Depends on: `numpy`.
- **`sky/catalog.py`** — loads Hipparcos stars (via Skyfield's loader) and the
  Stellarium constellation-line file (pairs of HIP ids). Produces star arrays
  (RA, Dec, magnitude, B–V color, HIP id) and a line index. Depends on:
  `skyfield`, `numpy`.
- **`sky/ephemeris.py`** — Skyfield wrapper for Sun/Moon/planets RA/Dec at a
  given time + location. Depends on: `skyfield`.
- **`render/scene.py`** — all GPU work: VBOs for stars/lines, billboards for
  planets/labels, optional Milky Way background sphere, camera with configurable
  FOV. Depends on: `moderngl`, `numpy`.
- **`render/labels.py`** — rasterizes text to textures for billboard labels.
  Depends on: `pygame` (font) or `freetype`, `moderngl`.
- **`app.py`** — wiring + main loop, window/display management, dev-mode input.

### Coordinate mapping (the crux)
The fusion quaternion expresses head orientation in a world frame where:
- **world up = gravity** (from the accelerometer), so altitude is physically
  correct; and
- **az = 0 = the heading at startup** (arbitrary — this is the free-look choice).

The sky is built in that same frame: each object's (altitude, azimuth) → unit
vector with zenith along world-up. The render camera orientation is the fusion
quaternion after a **fixed axis-remap** from the IMU's axis convention to GL's
(calibrated empirically; the probe capture shows how gravity projects onto the
accel axes when the glasses are tilted). Net behavior: look up → high-altitude
sky; spin → azimuth scrolls freely; roll → horizon tilts with the head.

### Data flow & timing
- IMU thread (~1000 Hz): socket → fusion → publish latest orientation (lock).
- Render thread (~60 fps): snapshot orientation → set camera → draw.
- Star positions: compute each star's fixed **equatorial** unit vector once at
  load. Each frame, build the equatorial→horizontal rotation from current local
  sidereal time + latitude (single 3×3) and apply it (CPU or vertex shader).
  This gives smooth Earth rotation with no per-frame catalog recompute.
- Planets/Sun/Moon (~10 objects): recompute RA/Dec via Skyfield ~once/second.

## Rendering details
- **Stars:** point sprites; size and brightness mapped from magnitude (down to a
  configurable limit, default ~6.5); color from Hipparcos B–V.
- **Constellation lines:** GL line segments between the HIP star vectors named in
  `constellationship.fab`.
- **Planets / Sun / Moon:** labeled billboard disks. Moon is a plain disk in v1
  (phase rendering deferred).
- **Labels:** brightest stars, planets, and constellation names as camera-facing
  texture quads. Kept sparse to avoid clutter.
- **Milky Way (optional, toggle):** equirectangular texture on a background
  sphere.
- **Camera FOV:** set to the glasses' real field of view (~57° for the One Pro,
  config-tunable) so angular separations between objects look correct.
- **Horizon:** faint horizon ring at altitude 0 (meaningful because altitude is
  physical); objects below the horizon dimmed or hidden.
- **Output:** single mono view filling 1920×1080.

## Display / output
Enumerate displays, detect the 1920×1080 glasses panel, open a borderless
fullscreen `pygame`/`moderngl` window on that display index. `config.py` allows
overriding the display index and selecting a **windowed laptop dev mode**.

## Error handling
- **Socket:** connect failures and mid-stream drops trigger auto-reconnect with
  backoff. Clear diagnostics if the glasses aren't reachable (developer/Ethernet
  mode off, interface down).
- **No glasses present:** fall back to **mouse/keyboard look** so the renderer
  and sky math are fully developable without hardware.
- **Astronomy data:** Skyfield downloads the DE421 ephemeris and Hipparcos
  catalog on first run and caches them locally; handle the offline case
  gracefully (cached data; clear message if missing).
- **Fusion startup:** brief "leveling" state while the filter settles from the
  first accelerometer readings.

## Testing
- **`sky/coords.py`:** unit tests vs Skyfield reference — known star RA/Dec →
  alt/az at a fixed time and location must match within tolerance.
- **`imu/reader.py`:** parse the real captured stream (`scratchpad/capture.bin`)
  and assert known sample values (e.g. accel magnitude ≈ 9.79 m/s², ~3000 IMU
  records). High-value regression fixture, costs nothing.
- **`imu/fusion.py`:** synthetic gravity vector with zero gyro → expected
  pitch/roll; constant gyro integrates to expected yaw delta.
- **Manual/integration:** run with glasses; confirm the sky tracks head motion
  and looks geometrically right (constellations recognizable, gravity-down
  correct).

## Project setup
`uv` + `pyproject.toml` per house Python standard. Dependencies: `skyfield`,
`moderngl`, `pygame`, `numpy`. The existing `probe_imu.py` parser logic folds
into `imu/reader.py`.

## Out of scope for v1 (future work)
Stereo / SBS 3D Mode rendering; magnetometer compass and true-north alignment;
time-scrubbing ("time machine"); Moon phase rendering; object search / go-to;
deep-sky catalogs (galaxies, nebulae) beyond the Milky Way texture.

## Open risks / notes for implementation
- **IMU axis-remap** to GL is empirical; expect a short calibration step where we
  confirm head-up maps to sky-up and motion directions are not mirrored.
- **Yaw drift:** gyro-only yaw will drift slowly over minutes. Acceptable for
  free-look v1; a "re-center" key can reset the azimuth offset on demand.
- **Label clutter / performance** with full Hipparcos + lines + labels: mitigate
  with magnitude-limited label rendering and instanced/batched draws.
