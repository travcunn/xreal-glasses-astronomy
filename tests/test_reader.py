import math
from pathlib import Path

from imu.reader import split_records, decode_record, RECORD_LEN

FIXTURE = Path(__file__).parent / "fixtures" / "imu_capture.bin"


def _decoded_samples():
    buf = FIXTURE.read_bytes()
    records, _leftover = split_records(buf)
    return [s for s in (decode_record(r) for r in records) if s is not None]


def test_parses_about_3000_imu_records():
    samples = _decoded_samples()
    assert 2900 <= len(samples) <= 3100


def test_stationary_gravity_magnitude():
    samples = _decoded_samples()
    mags = [math.sqrt(s.ax**2 + s.ay**2 + s.az**2) for s in samples]
    mean = sum(mags) / len(mags)
    assert 9.5 <= mean <= 10.1   # gravity, glasses at rest


def test_gyro_small_at_rest():
    samples = _decoded_samples()
    assert all(abs(s.gx) < 1.0 and abs(s.gy) < 1.0 and abs(s.gz) < 1.0 for s in samples)


def test_split_records_leftover_is_partial_tail():
    buf = FIXTURE.read_bytes()
    records, leftover = split_records(buf)
    assert all(len(r) == RECORD_LEN for r in records)
    assert len(leftover) < RECORD_LEN


import time as _time
from imu.reader import IMUReader


class _FakeSocket:
    def __init__(self, data: bytes, chunk: int = 4096):
        self._data = data
        self._chunk = chunk
        self._pos = 0

    def recv(self, n):
        if self._pos >= len(self._data):
            _time.sleep(0.01)
            return b""
        end = min(self._pos + self._chunk, len(self._data))
        out = self._data[self._pos:end]
        self._pos = end
        return out

    def close(self):
        pass


def test_reader_publishes_latest_from_fake_socket():
    data = FIXTURE.read_bytes()
    reader = IMUReader(connect_fn=lambda: _FakeSocket(data))
    reader.start()
    try:
        deadline = _time.time() + 2.0
        while reader.latest is None and _time.time() < deadline:
            _time.sleep(0.01)
        assert reader.latest is not None
        s = reader.latest
        assert 9.0 <= math.sqrt(s.ax**2 + s.ay**2 + s.az**2) <= 10.5
    finally:
        reader.stop()
