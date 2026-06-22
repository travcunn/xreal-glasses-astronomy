# Real-Sky Alignment (Magnetometer Compass) — Design

**Date:** 2026-06-21
**Status:** Approved (design); implementation plan pending
**Builds on:** `2026-06-21-xreal-sky-design.md` (the working free-look prototype)

## Goal

Make the virtual sky correspond to the real sky. When the wearer looks at the
actual Polaris, Orion, or the Moon, the app shows it in that same direction.
This replaces free-look (arbitrary heading) with a true-north-anchored heading,
while keeping the gravity-referenced pitch/roll that already works.

## Spike result (verified on the captured stream, 2026-06-21)

The magnetometer is live and healthy — no reverse-engineering or enable command
needed. The XREAL One Pro emits two report types on the same TCP stream
(confirmed against `xr-tools` source and `tests/fixtures/imu_capture.bin`):

- **IMU report**, `report_type = 0x0B` at body offset `0x18` (record offset 30):
  gyro xyz (rad/s) + accel xyz (m/s²) valid; mx/my/mz are an invalid `-3200.0`
  sentinel here. ~1000 Hz.
- **Magnetometer report**, `report_type = 0x04`: mx/my/mz (body `0x34..0x40`,
  record offset 58) valid; gyro/accel invalid. ~400 Hz.

Both share the 6-byte header (magic `0x28 0x36`) + 128-byte body = our 134-byte
records. Captured mag vector ≈ `(-14.3, -13.2, -40.9)`, magnitude **45.3 µT**
(std 0.45) — a clean Earth-field reading. Temperature reads ~33 °C at body
`0x40`. The earlier "`-3200` sentinel" conclusion was wrong: it was the invalid
mag field inside IMU records; the real mag is in the 0x04 records we filtered out.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Intent | Match the real sky (anchor heading to true north) |
| Heading source | Magnetometer compass (spike proved it feasible) |
| Fusion integration | **Approach A**: mag anchors yaw; keep the working gyro/accel filter and slew `yaw_offset` toward true north |
| Location | IP auto-detect at startup, config override, offline fallback |
| Declination | `pygeomag` (WMM) from lat/lon/date, config fallback |
| Calibration | Hard-iron offset via a key-triggered ~15 s rotation, persisted; works uncalibrated (less accurate) |

## Why Approach A (not full MARG)

The current orientation feel ("works nicely" on-device) comes from the
gyro+accel complementary filter plus the `BASE_VIEW` axis mapping. Folding mag
into the quaternion (full MARG) would re-open that axis calibration and risk the
feel for no visible benefit. Approach A is a bolt-on: gyro/accel are untouched;
the magnetometer only drives the azimuth anchor, replacing the arbitrary
free-look heading with a true-north one. Short-term response stays with the
gyro; the mag removes long-term yaw drift and fixes "where is north".

## Architecture

### New / changed units

- **`imu/reader.py` (change):** discriminate records by `report_type` (record
  offset 30: `0x0B` IMU, `0x04` mag) — the principled replacement for the
  offset-78 marker check. Decode mag (mx/my/mz at offset 58) and publish a
  `latest_mag: np.ndarray(3) | None` alongside `latest` (IMU). `decode_record`
  keeps returning an `IMUSample` for IMU records (same set as before, so existing
  tests hold); add `decode_mag(rec) -> np.ndarray | None` and
  `report_type(rec) -> int`.
- **`imu/magcal.py` (new):** `MagCalibration` with `offset: np.ndarray(3)`.
  `collect(sample)` accumulates per-axis min/max; `finish()` sets
  `offset = (min + max) / 2`. `apply(raw) -> calibrated`. `save(path)` /
  `load(path)` JSON persistence. Identity (zero offset) when uncalibrated.
- **`sky/heading.py` (new):** one pure function.
  - `azimuth_of_magnetic_north(mag_world) -> deg` — given the calibrated mag
    vector already rotated into the gravity-referenced world frame (N=x, E=y,
    Up=z), return the world-frame azimuth of magnetic north as
    `degrees(atan2(E, N))` in `[0, 360)`. (This is the tilt-compensated result,
    since rotating by the gravity-referenced orientation removes pitch/roll.)
    Declination is applied in the app's anchor formula below, not here, to keep
    one unambiguous definition of "where magnetic north points".
- **`sky/location.py` (new):** `resolve_location(default_lat, default_lon)` —
  one `urllib` GET to a keyless IP-geolocation endpoint (e.g. `ip-api.com/json`)
  with a short timeout; returns `(lat, lon)`; on any failure returns the defaults.
- **`sky/declination.py` (new):** `declination_deg(lat, lon, year) -> float`
  via `pygeomag` (WMM). Falls back to `config.MAGNETIC_DECLINATION_DEG` if
  `pygeomag` is unavailable or errors.
- **`app.py` (change):** wire the above. Each frame: read `latest_mag`,
  calibrate, rotate into the world frame using the current filter orientation,
  compute magnetic heading, add declination → true-north target azimuth.
  Complementary-slew `yaw_offset` toward the value that makes the rendered
  azimuth of magnetic north equal the declination (so true north → az 0).
  Keys: `C` runs/stops calibration (with an on-screen prompt to rotate); `R`
  still hard-snaps; `Esc` quits.
- **`config.py` (change):** add `MAGNETIC_DECLINATION_DEG` (fallback),
  `HEADING_GAIN` (yaw-anchor slew rate), `IP_GEOLOCATION = True`.

### Coordinate / math notes

- The complementary filter quaternion `head` maps device→world with gravity as
  up and an (initially arbitrary) yaw. Rotating the calibrated mag vector by
  `head` puts it in that world frame; its horizontal (N/E) projection points to
  magnetic north at azimuth `mag_az`.
- Target: with east-positive declination `D`, magnetic north sits at true
  azimuth `D` (true north at displayed az 0). `azimuth_of_magnetic_north` gives
  the world-frame azimuth `mag_az` of magnetic north. Displayed az = world az +
  `yaw_offset`, so to put magnetic north at displayed azimuth `D`:
  `yaw_offset_target = radians(D) - radians(mag_az)`.
- Slew, don't snap: `yaw_offset += HEADING_GAIN * wrap(yaw_offset_target - yaw_offset)`
  each frame, so the gyro keeps short-term motion and the mag pulls the anchor in
  smoothly. `wrap` handles the ±180° seam.
- `BASE_VIEW` (look-at-horizon tilt) is unchanged.

### Data flow

IMU thread publishes `latest` (IMU) and `latest_mag`. Render thread: fusion from
IMU → orientation; calibrate+rotate mag → magnetic heading; +declination → anchor
target; slew `yaw_offset`; render. Location + declination resolved once at startup.

## Error handling

- **No mag yet / `latest_mag is None`:** behave as free-look (no anchor slew)
  until mag arrives; never block.
- **Uncalibrated:** use raw mag (zero offset); heading may be biased — the `C`
  routine fixes it. State is visible via an on-screen hint.
- **IP geolocation offline/blocked:** fall back to `config` lat/lon, print a notice.
- **`pygeomag` missing/errors:** fall back to `config.MAGNETIC_DECLINATION_DEG`.
- **Magnetic noise spikes:** the slew gain low-passes them; a sample whose
  magnitude deviates wildly from the calibrated norm is skipped.

## Testing

- **Reader:** parse the real `imu_capture.bin` — assert ~1200 mag (0x04) records,
  mean |mag| ≈ 45 µT, and IMU count unchanged (~3000).
- **`magcal`:** feed synthetic samples on a sphere offset by a known hard-iron
  vector → recovered `offset` matches within tolerance; `apply` re-centers.
- **`heading`:** synthetic world-frame mag pointing at a known azimuth →
  `tilt_compensated_heading` returns it; `magnetic_to_true` adds declination.
- **`location`:** monkeypatch the fetch to a canned JSON → parsed lat/lon; force
  an exception → returns defaults.
- **`declination`:** `pygeomag` value for SF is positive and ~10–14° (sanity);
  fallback path returns the config value when forced.

## Out of scope (future)

Full MARG quaternion fusion; soft-iron (ellipsoid) calibration; automatic
continuous recalibration; on-screen compass rose / cardinal labels; using the
glasses' own factory calibration values from the control channel.

## Open risks

- Indoor magnetic distortion (monitors, steel) can bias heading; mitigated by
  calibration + low-pass slew, but accuracy indoors is inherently limited.
- Mag and accel must share the device axis frame; if heading is consistently
  rotated/mirrored, the mag axis mapping needs the same kind of one-line fix as
  the `BASE_VIEW` calibration (verify on-device).
- `pygeomag` dependency: confirm it is maintained and ships WMM coefficients; if
  not, keep the config-constant declination as the default path.
