#!/usr/bin/env python3
"""
probe_imu.py - throwaway spike to confirm the XREAL One Pro streams IMU data
to this Mac over its USB-ethernet TCP interface.

Goal: answer one question. Does the socket at 169.254.2.1:52998 talk to us,
and can we parse gyro/accelerometer samples out of it? If yes, the whole
astronomy-app project is green-lit. If no, we find out before writing app code.

No dependencies, stdlib only. Protocol/framing mirrors the working community
demo (github.com/SamiMitwalli/One-Pro-IMU-Retriever-Demo).

Usage:
    python3 probe_imu.py              # run for ~10s, print rate + samples
    python3 probe_imu.py --duration 30
    python3 probe_imu.py --raw        # also hexdump raw bytes (use if no frames parse)
"""

import argparse
import socket
import struct
import sys
import time

IP = "169.254.2.1"
PORT = 52998
CONNECT_TIMEOUT = 5  # seconds to establish the TCP connection

# Wire framing as observed on this One Pro firmware (verified by capture).
#
#   Fixed-length 134-byte records, each starting with HEADER. No footer
#   delimiter (the community demo's footer is absent on this firmware).
#   The IMU payload is six little-endian float32 at a FIXED offset 34 from
#   the header start: gyro (x,y,z) in rad/s, then accel (x,y,z) in m/s^2.
#   IMU records carry the SENSOR marker around offset 78; other record types
#   (also 134 bytes) lack it and are skipped. IMU stream runs at ~1000 Hz.
#
#   record layout (134 bytes):
#     [0:6]    HEADER 28 36 00 00 00 80
#     [6:14]   session/counter
#     [14:34]  timestamp + status
#     [34:58]  6x float32  <-- gyro xyz (rad/s), accel xyz (m/s^2)
#     [58:..]  aux fields (mag placeholder, etc.)
#     [78:84]  SENSOR marker 00 40 1f 00 00 40  (present only on IMU records)
HEADER = bytes.fromhex("283600000080")
SENSOR = bytes.fromhex("00401f000040")
RECORD_LEN = 134
PAYLOAD_OFFSET = 34


def decode_record(rec: bytes):
    """Return (gx, gy, gz, ax, ay, az) for an IMU record, or None."""
    if len(rec) < RECORD_LEN or SENSOR not in rec[70:90]:
        return None  # not an IMU-type record
    try:
        gx, gy, gz, ax, ay, az = struct.unpack_from("<6f", rec, PAYLOAD_OFFSET)
    except struct.error:
        return None
    return gx, gy, gz, ax, ay, az


def hexdump(buf: bytes, limit: int = 256) -> str:
    chunk = buf[:limit]
    out = []
    for i in range(0, len(chunk), 16):
        row = chunk[i : i + 16]
        hexpart = " ".join(f"{b:02x}" for b in row)
        out.append(f"  {i:04x}  {hexpart}")
    if len(buf) > limit:
        out.append(f"  ... ({len(buf) - limit} more bytes)")
    return "\n".join(out)


def connect() -> socket.socket:
    print(f"Connecting to {IP}:{PORT} (timeout {CONNECT_TIMEOUT}s)...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(CONNECT_TIMEOUT)
    sock.connect((IP, PORT))
    print("TCP connected. Reading stream...\n")
    sock.settimeout(2)  # per-recv timeout once connected
    return sock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to sample")
    ap.add_argument("--raw", action="store_true", help="hexdump raw bytes too")
    args = ap.parse_args()

    try:
        sock = connect()
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"FAILED to connect: {e}\n")
        print("Troubleshooting:")
        print(f"  - Confirm an interface has a 169.254.2.x address: ifconfig | grep 169.254.2")
        print("  - In the glasses OSD, open the developer menu and confirm Ethernet is ENABLED.")
        print("  - Re-plug the USB-C cable; give macOS a few seconds to assign the link-local IP.")
        print(f"  - Sanity check reachability: ping {IP}")
        return 1

    start = time.time()
    recv_buffer = b""
    raw_total = 0
    frames = 0
    last_print = 0.0
    first_sample = None

    try:
        while time.time() - start < args.duration:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                print("Connection closed by device.")
                break

            raw_total += len(data)
            recv_buffer += data

            if args.raw and raw_total <= 4096:
                print(f"[raw] received {len(data)} bytes:")
                print(hexdump(data))

            # Frame fixed-length 134-byte records starting at each HEADER.
            while True:
                h = recv_buffer.find(HEADER)
                if h == -1 or len(recv_buffer) < h + RECORD_LEN:
                    break
                rec = recv_buffer[h : h + RECORD_LEN]
                recv_buffer = recv_buffer[h + RECORD_LEN :]

                imu = decode_record(rec)
                if imu is None:
                    continue

                frames += 1
                if first_sample is None:
                    first_sample = imu
                now = time.time()
                if now - last_print >= 0.25:  # throttle console to ~4 lines/sec
                    last_print = now
                    rate = frames / (now - start) if now > start else 0
                    gx, gy, gz, ax, ay, az = imu
                    print(
                        f"[{frames:06d}] {rate:6.1f}Hz | "
                        f"G=({gx:+.3f},{gy:+.3f},{gz:+.3f}) "
                        f"A=({ax:+.3f},{ay:+.3f},{az:+.3f})"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sock.close()

    elapsed = time.time() - start
    print("\n----- summary -----")
    print(f"raw bytes received : {raw_total}")
    print(f"IMU frames parsed  : {frames}")
    if frames:
        print(f"avg rate           : {frames / elapsed:.1f} Hz")
        print(f"first sample (G/A) : {first_sample}")
        print("\nRESULT: SUCCESS - the One Pro is streaming parseable IMU data to this Mac.")
        return 0
    if raw_total:
        print("\nRESULT: PARTIAL - bytes are flowing but no frames parsed.")
        print("The socket works; the framing differs from the reference protocol")
        print("(firmware version drift). Re-run with --raw and share the hexdump.")
        return 2
    print("\nRESULT: NO DATA - connected but received nothing.")
    print("Check that the glasses' developer/Ethernet mode is on and content is active.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
