"""Read the XREAL One Pro IMU stream over TCP and decode it to samples.

Wire format (verified on real hardware): fixed 134-byte records, each starting
with HEADER. Six little-endian float32 at offset 34 = gyro xyz (rad/s) then
accel xyz (m/s^2). IMU records carry SENSOR near offset 78; others are skipped.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

import config

HEADER = bytes.fromhex("283600000080")
SENSOR = bytes.fromhex("00401f000040")
RECORD_LEN = 134
PAYLOAD_OFFSET = 34


@dataclass
class IMUSample:
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gx, self.gy, self.gz])

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])


def split_records(buf: bytes) -> tuple[list[bytes], bytes]:
    """Frame fixed-length records starting at each HEADER. Returns (records, leftover)."""
    records = []
    i = 0
    while True:
        h = buf.find(HEADER, i)
        if h == -1:
            return records, buf[i:]
        if h + RECORD_LEN > len(buf):
            return records, buf[h:]
        records.append(buf[h:h + RECORD_LEN])
        i = h + RECORD_LEN


def decode_record(rec: bytes) -> IMUSample | None:
    if len(rec) < RECORD_LEN or SENSOR not in rec[70:90]:
        return None
    try:
        gx, gy, gz, ax, ay, az = struct.unpack_from("<6f", rec, PAYLOAD_OFFSET)
    except struct.error:
        return None
    return IMUSample(gx, gy, gz, ax, ay, az)


class IMUReader:
    def __init__(self, host: str = config.IMU_HOST, port: int = config.IMU_PORT,
                 connect_fn=None):
        self._host = host
        self._port = port
        self._connect_fn = connect_fn or self._default_connect
        self._latest: IMUSample | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _default_connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((self._host, self._port))
        s.settimeout(2)
        return s

    @property
    def latest(self) -> IMUSample | None:
        with self._lock:
            return self._latest

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        buf = b""
        sock = None
        while not self._stop.is_set():
            try:
                if sock is None:
                    sock = self._connect_fn()
                    buf = b""
                data = sock.recv(65536)
                if not data:
                    continue
                buf += data
                records, buf = split_records(buf)
                for rec in records:
                    sample = decode_record(rec)
                    if sample is not None:
                        with self._lock:
                            self._latest = sample
            except (OSError, socket.timeout):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                sock = None
                time.sleep(0.5)  # backoff before reconnect
        if sock is not None:
            sock.close()
