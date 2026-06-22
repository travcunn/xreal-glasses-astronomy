# Real-Sky Alignment (Magnetometer Compass) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anchor the sky's heading to true north using the One Pro's magnetometer, so the virtual sky matches the real sky (look at real Polaris, see Polaris).

**Architecture:** Approach A — leave the working gyro/accel orientation filter untouched; read the magnetometer (the 0x04 report records), calibrate hard-iron, compute the world-frame azimuth of magnetic north, add declination, and complementary-slew the existing `yaw_offset` so true north maps to screen-azimuth 0. Location is auto-detected by IP; declination from WMM (`pygeomag`). Both have config fallbacks.

**Tech Stack:** Python 3.12+, `uv`, `numpy`, `pygeomag` (new), existing `skyfield`/`moderngl`/`pygame`, `pytest`.

## Global Constraints

- `uv` + `pyproject.toml`; run via `uv run`. No pip/Poetry.
- Verified report protocol (do not re-derive): 134-byte records, 6-byte header (magic `0x28 0x36`) + 128-byte body; `report_type` is a little-endian u32 at **record offset 30** (`0x0B`=IMU, `0x04`=magnetometer); IMU floats (gyro xyz rad/s, accel xyz m/s²) at offset 34; magnetometer floats (mx,my,mz µT) at offset 58. Mag stream ≈400 Hz, |B|≈45 µT.
- Approach A only: do NOT fold mag into the orientation quaternion; it only drives `yaw_offset`.
- World frame convention (unchanged): X=North, Y=East, Z=Up; azimuth from North toward East.
- All tunables in `config.py`.
- **Version control:** project has a `uv`-created git repo but the user manages VCS and bars agent git writes. Each "Checkpoint" = run the task's tests green; commit only if the user asks.
- Existing tests must stay green (`uv run pytest`).

## File Structure

```
imu/reader.py     (modify)  parse 0x04 mag records; publish latest_mag
imu/magcal.py     (new)     hard-iron calibration: collect/finish/apply/save/load
sky/heading.py    (new)     azimuth_of_magnetic_north + slew_angle (pure)
sky/location.py   (new)     IP geolocation with fallback
sky/declination.py(new)     WMM declination via pygeomag with fallback
config.py         (modify)  MAGNETIC_DECLINATION_DEG, HEADING_GAIN, IP_GEOLOCATION
app.py            (modify)  wire mag->calibrate->heading->declination->slew yaw_offset; C key
tests/test_reader.py        (modify) mag-record tests
tests/test_magcal.py        (new)
tests/test_heading.py       (new)
tests/test_location.py      (new)
tests/test_declination.py   (new)
```

---

### Task 1: Read the magnetometer (`imu/reader.py`)

**Files:**
- Modify: `imu/reader.py`
- Test: `tests/test_reader.py`

**Interfaces:**
- Produces: constants `REPORT_TYPE_OFFSET = 30`, `REPORT_IMU = 0x0B`, `REPORT_MAG = 0x04`, `MAG_OFFSET = 58`; `report_type(rec) -> int`; `decode_mag(rec) -> np.ndarray | None` (shape (3,), µT); `decode_record` unchanged signature (returns `IMUSample` for IMU records); `IMUReader.latest_mag -> np.ndarray | None`.

- [ ] **Step 1: Write failing tests for mag decoding**

Append to `tests/test_reader.py`:
```python
import numpy as np
from imu.reader import report_type, decode_mag, REPORT_IMU, REPORT_MAG


def test_report_types_present_in_capture():
    buf = FIXTURE.read_bytes()
    records, _ = split_records(buf)
    types = [report_type(r) for r in records]
    assert types.count(REPORT_IMU) > 2000      # ~3000 IMU
    assert types.count(REPORT_MAG) > 800        # ~1200 mag


def test_decode_mag_reads_earth_field():
    buf = FIXTURE.read_bytes()
    records, _ = split_records(buf)
    mags = [decode_mag(r) for r in records]
    mags = [m for m in mags if m is not None]
    assert 800 < len(mags) < 1500
    mag_norms = [float(np.linalg.norm(m)) for m in mags]
    mean = sum(mag_norms) / len(mag_norms)
    assert 30.0 < mean < 65.0      # Earth's field magnitude in microtesla


def test_decode_mag_returns_none_for_imu_record():
    buf = FIXTURE.read_bytes()
    records, _ = split_records(buf)
    imu_rec = next(r for r in records if report_type(r) == REPORT_IMU)
    assert decode_mag(imu_rec) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_reader.py -k "report_type or decode_mag" -v`
Expected: FAIL — `ImportError: cannot import name 'report_type'`.

- [ ] **Step 3: Implement report-type discrimination + mag decode**

In `imu/reader.py`, replace the constants block and `decode_record` and add the new functions. Change the constants near the top to:
```python
HEADER = bytes.fromhex("283600000080")
RECORD_LEN = 134
PAYLOAD_OFFSET = 34       # gyro xyz, accel xyz (IMU records)
MAG_OFFSET = 58           # mx, my, mz (magnetometer records)
REPORT_TYPE_OFFSET = 30   # little-endian u32: 0x0B IMU, 0x04 magnetometer
REPORT_IMU = 0x0B
REPORT_MAG = 0x04
```
(Delete the now-unused `SENSOR` constant.) Add:
```python
def report_type(rec: bytes) -> int:
    if len(rec) < REPORT_TYPE_OFFSET + 4:
        return -1
    return struct.unpack_from("<I", rec, REPORT_TYPE_OFFSET)[0]
```
Replace `decode_record` with the report-type-based version (same IMU records as before):
```python
def decode_record(rec: bytes) -> IMUSample | None:
    if len(rec) < RECORD_LEN or report_type(rec) != REPORT_IMU:
        return None
    try:
        gx, gy, gz, ax, ay, az = struct.unpack_from("<6f", rec, PAYLOAD_OFFSET)
    except struct.error:
        return None
    return IMUSample(gx, gy, gz, ax, ay, az)


def decode_mag(rec: bytes) -> np.ndarray | None:
    if len(rec) < RECORD_LEN or report_type(rec) != REPORT_MAG:
        return None
    try:
        mx, my, mz = struct.unpack_from("<3f", rec, MAG_OFFSET)
    except struct.error:
        return None
    return np.array([mx, my, mz])
```

- [ ] **Step 4: Run to verify the new + existing reader tests pass**

Run: `uv run pytest tests/test_reader.py -v`
Expected: PASS (existing 5 + 3 new = 8). The IMU tests still pass because `report_type==0x0B` selects exactly the records the old marker did.

- [ ] **Step 5: Publish `latest_mag` from the reader thread**

In `imu/reader.py`, `IMUReader.__init__` add `self._latest_mag = None`. Add the property:
```python
    @property
    def latest_mag(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_mag
```
In `_run`, replace the per-record loop body with:
```python
                for rec in records:
                    sample = decode_record(rec)
                    if sample is not None:
                        with self._lock:
                            self._latest = sample
                        continue
                    mag = decode_mag(rec)
                    if mag is not None:
                        with self._lock:
                            self._latest_mag = mag
```

- [ ] **Step 6: Extend the fake-socket test to assert mag is published**

Append to `tests/test_reader.py`:
```python
def test_reader_publishes_latest_mag_from_fake_socket():
    data = FIXTURE.read_bytes()
    reader = IMUReader(connect_fn=lambda: _FakeSocket(data))
    reader.start()
    try:
        deadline = _time.time() + 2.0
        while reader.latest_mag is None and _time.time() < deadline:
            _time.sleep(0.01)
        assert reader.latest_mag is not None
        assert 30.0 < float(np.linalg.norm(reader.latest_mag)) < 65.0
    finally:
        reader.stop()
```

- [ ] **Step 7: Run**

Run: `uv run pytest tests/test_reader.py -v`
Expected: PASS (9 tests).

- [ ] **Step 8: Checkpoint** — reader tests green. (Commit only if the user asks.)

---

### Task 2: Hard-iron calibration (`imu/magcal.py`)

**Files:**
- Create: `imu/magcal.py`
- Test: `tests/test_magcal.py`

**Interfaces:**
- Produces: `MagCalibration(offset: np.ndarray = zeros(3))` with `collect(sample: np.ndarray) -> None`, `finish() -> None` (sets `offset = (min+max)/2`), `apply(raw: np.ndarray) -> np.ndarray`, `save(path: str) -> None`, `MagCalibration.load(path: str) -> MagCalibration`, and `samples_count -> int`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_magcal.py`:
```python
import json
import numpy as np
from imu.magcal import MagCalibration


def _sphere_points(center, radius, n=400):
    rng = np.linspace(0, 1, n)
    pts = []
    for i, u in enumerate(rng):
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_magcal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imu.magcal'`.

- [ ] **Step 3: Implement `imu/magcal.py`**

```python
"""Hard-iron magnetometer calibration: estimate and remove a fixed offset."""

import json

import numpy as np


class MagCalibration:
    def __init__(self, offset: np.ndarray | None = None):
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, float)
        self._min = np.full(3, np.inf)
        self._max = np.full(3, -np.inf)
        self.samples_count = 0

    def collect(self, sample: np.ndarray) -> None:
        s = np.asarray(sample, float)
        self._min = np.minimum(self._min, s)
        self._max = np.maximum(self._max, s)
        self.samples_count += 1

    def finish(self) -> None:
        if self.samples_count > 0 and np.all(np.isfinite(self._min)):
            self.offset = (self._min + self._max) / 2.0

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, float) - self.offset

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"offset": self.offset.tolist()}, fh)

    @classmethod
    def load(cls, path: str) -> "MagCalibration":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(offset=np.array(data["offset"], float))
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_magcal.py -v`
Expected: PASS (3).

- [ ] **Step 5: Checkpoint** — magcal tests green.

---

### Task 3: Heading math (`sky/heading.py`)

**Files:**
- Create: `sky/heading.py`
- Test: `tests/test_heading.py`

**Interfaces:**
- Produces: `azimuth_of_magnetic_north(mag_world: np.ndarray) -> float` (degrees in [0,360), `atan2(E, N)`), and `slew_angle(current_rad: float, target_rad: float, gain: float) -> float` (move `current` toward `target` along the shortest angular path by `gain` fraction; result wrapped to (-pi, pi]).

- [ ] **Step 1: Write failing tests**

Create `tests/test_heading.py`:
```python
import numpy as np
from sky.heading import azimuth_of_magnetic_north, slew_angle


def test_azimuth_cardinals():
    assert abs(azimuth_of_magnetic_north(np.array([1.0, 0.0, 0.0])) - 0.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([0.0, 1.0, 0.0])) - 90.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([-1.0, 0.0, 0.0])) - 180.0) < 1e-6
    assert abs(azimuth_of_magnetic_north(np.array([0.0, -1.0, 0.0])) - 270.0) < 1e-6


def test_azimuth_ignores_vertical():
    # Up component must not change the horizontal azimuth.
    assert abs(azimuth_of_magnetic_north(np.array([1.0, 1.0, 9.0])) - 45.0) < 1e-6


def test_slew_moves_fraction_toward_target():
    out = slew_angle(0.0, 1.0, 0.25)
    assert abs(out - 0.25) < 1e-9


def test_slew_takes_short_way_across_pi_seam():
    # current just below +pi, target just above -pi: shortest path crosses the seam.
    out = slew_angle(3.0, -3.0, 0.5)
    # shortest delta is +(2pi-6) ~ +0.283; half of it ~ +0.14 -> wraps near -pi
    assert out > 3.0 or out < -3.0   # moved across the seam, not the long way
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_heading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sky.heading'`.

- [ ] **Step 3: Implement `sky/heading.py`**

```python
"""Heading helpers for magnetometer-based real-sky alignment (pure)."""

import numpy as np


def azimuth_of_magnetic_north(mag_world: np.ndarray) -> float:
    """World-frame azimuth (deg, North-through-East, [0,360)) of the mag vector.

    `mag_world` must already be rotated into the gravity-referenced world frame
    (X=North, Y=East, Z=Up), which makes this tilt-compensated.
    """
    north, east = float(mag_world[0]), float(mag_world[1])
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _wrap(angle_rad: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def slew_angle(current_rad: float, target_rad: float, gain: float) -> float:
    """Move `current` toward `target` by `gain` along the shortest path."""
    delta = _wrap(target_rad - current_rad)
    return _wrap(current_rad + gain * delta)
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_heading.py -v`
Expected: PASS (4).

- [ ] **Step 5: Checkpoint** — heading tests green.

---

### Task 4: Location auto-detect (`sky/location.py`)

**Files:**
- Create: `sky/location.py`
- Test: `tests/test_location.py`

**Interfaces:**
- Produces: `resolve_location(default_lat: float, default_lon: float, fetch=None) -> tuple[float, float]`. `fetch` is an injectable `() -> dict` returning a mapping with `lat`/`lon`; default does a real IP lookup. Any failure returns the defaults.

- [ ] **Step 1: Write failing tests**

Create `tests/test_location.py`:
```python
from sky.location import resolve_location


def test_uses_fetched_coordinates():
    lat, lon = resolve_location(0.0, 0.0, fetch=lambda: {"lat": 40.7, "lon": -74.0})
    assert (round(lat, 1), round(lon, 1)) == (40.7, -74.0)


def test_falls_back_on_error():
    def boom():
        raise RuntimeError("offline")
    assert resolve_location(37.77, -122.42, fetch=boom) == (37.77, -122.42)


def test_falls_back_on_bad_payload():
    assert resolve_location(1.0, 2.0, fetch=lambda: {"nope": 1}) == (1.0, 2.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_location.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sky.location'`.

- [ ] **Step 3: Implement `sky/location.py`**

```python
"""Resolve observer location by IP geolocation, with a safe fallback."""

import json
import urllib.request


def _fetch_ipapi() -> dict:
    with urllib.request.urlopen("http://ip-api.com/json/", timeout=3) as resp:
        return json.load(resp)


def resolve_location(default_lat: float, default_lon: float, fetch=None) -> tuple[float, float]:
    fetch = fetch or _fetch_ipapi
    try:
        data = fetch()
        return float(data["lat"]), float(data["lon"])
    except Exception:
        return default_lat, default_lon
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_location.py -v`
Expected: PASS (3).

- [ ] **Step 5: Checkpoint** — location tests green.

---

### Task 5: Declination via WMM (`sky/declination.py` + dependency + config)

**Files:**
- Create: `sky/declination.py`
- Modify: `config.py`, `pyproject.toml` (add `pygeomag`)
- Test: `tests/test_declination.py`

**Interfaces:**
- Produces: `declination_deg(lat: float, lon: float, year: float, geomag_factory=None) -> float`. Uses `pygeomag` by default; on any error returns `config.MAGNETIC_DECLINATION_DEG`. `geomag_factory` is injectable for testing.

- [ ] **Step 1: Add the dependency and config values**

Run: `uv add pygeomag`
Expected: installs `pygeomag`.

Append to `config.py`:
```python
# Magnetometer heading anchor
MAGNETIC_DECLINATION_DEG = 13.0   # fallback if WMM lookup fails (SF ~ +13 E)
HEADING_GAIN = 0.02               # how fast the compass pulls yaw toward true north
IP_GEOLOCATION = True             # auto-detect observer location at startup
MAG_CALIBRATION_PATH = "mag_calibration.json"
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_declination.py`:
```python
from sky.declination import declination_deg


def test_san_francisco_declination_is_easterly():
    d = declination_deg(37.7749, -122.4194, 2026.0)
    assert 8.0 < d < 16.0      # SF declination is ~ +13 deg East


def test_fallback_on_geomag_error():
    def boom():
        raise RuntimeError("no coefficients")
    import config
    assert declination_deg(0.0, 0.0, 2026.0, geomag_factory=boom) == config.MAGNETIC_DECLINATION_DEG
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_declination.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sky.declination'`.

- [ ] **Step 4: Implement `sky/declination.py`**

```python
"""Magnetic declination (degrees east) from the World Magnetic Model."""

import config


def declination_deg(lat: float, lon: float, year: float, geomag_factory=None) -> float:
    try:
        if geomag_factory is None:
            from pygeomag import GeoMag
            geomag_factory = GeoMag
        geo = geomag_factory()
        result = geo.calculate(glat=lat, glon=lon, alt=0, time=year)
        return float(result.d)
    except Exception:
        return config.MAGNETIC_DECLINATION_DEG
```

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_declination.py -v`
Expected: PASS (2). If `result.d` raises an `AttributeError` (older pygeomag uses `.dec`), adjust to `getattr(result, "d", None) or result.dec`; the SF-range test will confirm the correct attribute.

- [ ] **Step 6: Checkpoint** — declination tests green.

---

### Task 6: App integration + on-device verification (`app.py`)

**Files:**
- Modify: `app.py`
- Test: `tests/test_heading.py` (add the anchor-target helper test)

**Interfaces:**
- Consumes: everything above.
- Produces: `compute_yaw_target(mag_world: np.ndarray, declination_deg: float) -> float` (radians) in `sky/heading.py`, used by the loop; `C` key toggles calibration; the mag continuously slews `yaw_offset`.

- [ ] **Step 1: Write a failing test for the anchor target**

Append to `tests/test_heading.py`:
```python
from sky.heading import compute_yaw_target


def test_yaw_target_puts_true_north_at_zero():
    # Magnetic north points due North in the world frame, declination +10 E.
    # Then magnetic north should display at azimuth +10 deg -> yaw_offset target = +10 deg.
    t = compute_yaw_target(np.array([1.0, 0.0, 0.0]), 10.0)
    assert abs(np.degrees(t) - 10.0) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_heading.py::test_yaw_target_puts_true_north_at_zero -v`
Expected: FAIL — `ImportError: cannot import name 'compute_yaw_target'`.

- [ ] **Step 3: Add `compute_yaw_target` to `sky/heading.py`**

```python
def compute_yaw_target(mag_world: np.ndarray, declination_deg: float) -> float:
    """Radians to offset the rendered azimuth so true north -> screen az 0.

    Magnetic north sits at true azimuth = declination (east positive); its
    measured world-frame azimuth is `azimuth_of_magnetic_north`. To display it at
    azimuth `declination`, the offset must be (declination - measured).
    """
    mag_az = azimuth_of_magnetic_north(mag_world)
    return _wrap(np.radians(declination_deg) - np.radians(mag_az))
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_heading.py -v`
Expected: PASS (5).

- [ ] **Step 5: Wire location + declination at startup in `app.py`**

In `app.py`, after the imports add:
```python
from imu.magcal import MagCalibration
from sky.location import resolve_location
from sky.declination import declination_deg
from sky.heading import compute_yaw_target, slew_angle
from mathlib import rotate_vector
```
Replace the fixed-location setup (where `Ephemeris(...)` and the star rotation use `config.LATITUDE_DEG/LONGITUDE_DEG`) so the resolved location is used. Right after `pygame.init()` / before building the catalog, add:
```python
    if config.IP_GEOLOCATION:
        lat_deg, lon_deg = resolve_location(config.LATITUDE_DEG, config.LONGITUDE_DEG)
    else:
        lat_deg, lon_deg = config.LATITUDE_DEG, config.LONGITUDE_DEG
    print(f"observer location: {lat_deg:.3f}, {lon_deg:.3f}")
    declination = declination_deg(lat_deg, lon_deg, 2026.5)
    print(f"magnetic declination: {declination:+.1f} deg")
```
Then use `lat_deg`/`lon_deg` in place of `config.LATITUDE_DEG`/`config.LONGITUDE_DEG` for the `Ephemeris(...)` constructor and in the per-frame `lst_hours(jd, lon_deg)` and `equatorial_to_horizontal_matrix(lat_deg, lst)` calls.

- [ ] **Step 6: Add calibration state and the `C` key in `app.py`**

Where the head-orientation source is set up (the `if use_glasses:` block), add after it:
```python
    import os
    mag_cal = (MagCalibration.load(config.MAG_CALIBRATION_PATH)
               if os.path.exists(config.MAG_CALIBRATION_PATH) else MagCalibration())
    calibrating = False
```
In the event loop, add a handler:
```python
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
```

- [ ] **Step 7: Slew `yaw_offset` from the magnetometer each frame**

In `app.py`, replace the orientation block so that after computing `head` and before `cam_pre`, the magnetometer anchors yaw:
```python
        # Magnetometer heading anchor (Approach A): slew yaw_offset toward true north.
        mag = reader.latest_mag if reader is not None else None
        if mag is not None:
            if calibrating:
                mag_cal.collect(mag)
            mag_world = rotate_vector(head, mag_cal.apply(mag))
            if not calibrating:
                target = compute_yaw_target(mag_world, declination)
                yaw_offset = slew_angle(yaw_offset, target, config.HEADING_GAIN)
```
(Place this immediately after `head = ...` is computed and before `cam_pre = quat_mul(BASE_VIEW, head)`. The existing `R` key and `current_az` logic stay; `R` still hard-snaps.)

- [ ] **Step 8: Run the full unit suite**

Run: `uv run pytest`
Expected: PASS (all tests across the project, including the prior 26 + the new reader/magcal/heading/location/declination tests).

- [ ] **Step 9: Headless smoke check of the integrated math**

Run:
```bash
uv run python -c "
import numpy as np
from sky.heading import compute_yaw_target, slew_angle, azimuth_of_magnetic_north
from imu.magcal import MagCalibration
from imu.reader import split_records, decode_mag
m = [x for x in (decode_mag(r) for r in split_records(open('tests/fixtures/imu_capture.bin','rb').read())[0]) if x is not None][0]
cal = MagCalibration()
print('raw mag', np.round(m,1), 'az', round(azimuth_of_magnetic_north(m),1))
print('yaw target deg', round(np.degrees(compute_yaw_target(m, 13.0)),1))
print('slew', round(slew_angle(0.0, 1.0, 0.02),4))
print('OK')
"
```
Expected: prints sensible numbers and `OK` (no exceptions).

- [ ] **Step 10: On-device verification (glasses)**

Run: `uv run python app.py --glasses`
Expected, wearing the glasses:
1. Startup prints the detected location and declination.
2. Press `C`, rotate the glasses through all orientations for ~15 s, press `C` again — it prints the calibration offset and saves `mag_calibration.json`.
3. The sky now holds a fixed heading: physically turn around and the same real-world direction shows the same stars; it no longer free-floats/drifts.
4. Look toward real north — the app's northern sky (e.g. Polaris near the celestial pole at altitude ≈ your latitude) is there.
- If the sky is rotated by a constant amount or spins the wrong way, this is the heading sign/axis calibration: flip the sign in `compute_yaw_target` (return `-_wrap(...)`) or add `np.pi`, mirroring how `BASE_VIEW` was calibrated. Re-verify.

- [ ] **Step 11: Checkpoint** — suite green + on-device heading holds north.

---

## Self-Review (spec coverage)

- Read magnetometer (0x04 records), publish alongside IMU → Task 1. ✓
- Approach A (mag anchors yaw, filter untouched) → Task 6 Step 7 (slew `yaw_offset`; gyro/accel filter unchanged). ✓
- Hard-iron calibration, key-triggered, persisted, works uncalibrated → Task 2 + Task 6 Steps 6–7. ✓
- Tilt-compensated azimuth of magnetic north → Task 3 (`azimuth_of_magnetic_north` on world-frame mag). ✓
- Declination via WMM with config fallback → Task 5. ✓
- IP auto-location with offline/config fallback → Task 4 + Task 6 Step 5. ✓
- `yaw_offset_target = radians(D) - radians(mag_az)`; slew not snap → Task 6 Step 3/7. ✓
- Config keys (`MAGNETIC_DECLINATION_DEG`, `HEADING_GAIN`, `IP_GEOLOCATION`, `MAG_CALIBRATION_PATH`) → Task 5 Step 1. ✓
- Testing (reader vs fixture |B|≈45; calibration recovers offset; heading; location; declination) → Tasks 1–5. ✓
- Error handling: `latest_mag is None` → no slew (Task 6 Step 7 guards on `mag is not None`); uncalibrated uses raw (zero offset); location/declination fallbacks (Tasks 4/5). ✓
- New dep `pygeomag` → Task 5 Step 1. ✓

Open risks carried from spec: heading sign/axis is on-device-calibrated (Task 6 Step 10); indoor magnetic distortion limits accuracy (calibration + slew low-pass mitigate); `pygeomag` `.d` vs `.dec` attribute confirmed by the SF-range test (Task 5 Step 5).
