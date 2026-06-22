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
RECORD_LEN = 134
PAYLOAD_OFFSET = 34       # gyro xyz, accel xyz (IMU records)
MAG_OFFSET = 58           # mx, my, mz (magnetometer records)
REPORT_TYPE_OFFSET = 30   # little-endian u32: 0x0B IMU, 0x04 magnetometer
REPORT_IMU = 0x0B
REPORT_MAG = 0x04


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


def report_type(rec: bytes) -> int:
    if len(rec) < REPORT_TYPE_OFFSET + 4:
        return -1
    return struct.unpack_from("<I", rec, REPORT_TYPE_OFFSET)[0]


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


class IMUReader:
    def __init__(self, host: str = config.IMU_HOST, port: int = config.IMU_PORT,
                 connect_fn=None):
        self._host = host
        self._port = port
        self._connect_fn = connect_fn or self._default_connect
        self._latest: IMUSample | None = None
        self._latest_mag: np.ndarray | None = None
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

    @property
    def latest_mag(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_mag

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
                        continue
                    mag = decode_mag(rec)
                    if mag is not None:
                        with self._lock:
                            self._latest_mag = mag
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
