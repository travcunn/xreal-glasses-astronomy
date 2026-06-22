#!/usr/bin/env python3
"""Diagnose the XREAL One Pro IMU axis convention while the glasses are worn.

World-locking needs the body (IMU) frame mapped correctly onto the GL/sky frame.
The capture-on-a-desk fixture can't tell us how the axes sit when worn, so this
walks through four short moves and reports, in body coordinates:

  - which way gravity points when you look level + forward  (-> body "up")
  - the gyro axis + sign for yaw / pitch / roll head motions (-> the 3 head axes)

Run it on the Mac with the glasses on and the IMU reachable, then paste the
final summary block back. From it we can build the exact remap (no guessing).

    uv run python diagnose_imu_axes.py
"""

import time

import numpy as np

from imu.reader import IMUReader

AXES = ["X", "Y", "Z"]


def _dominant(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    return f"{'+' if v[i] >= 0 else '-'}{AXES[i]}"


def _capture(reader: IMUReader, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_gyro, mean_accel) over `seconds`, sampling latest at ~200 Hz."""
    gyros, accels = [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        s = reader.latest
        if s is not None:
            gyros.append(s.gyro.copy())
            accels.append(s.accel.copy())
        time.sleep(0.005)
    if not gyros:
        raise RuntimeError("no IMU samples received; is the glasses link up?")
    return np.mean(gyros, axis=0), np.mean(accels, axis=0)


def _move(reader: IMUReader, name: str, instruction: str) -> np.ndarray:
    input(f"\n[{name}] {instruction}\n  Press Enter, then START MOVING for ~2.5 s ... ")
    print("  capturing ...")
    mean_gyro, _ = _capture(reader, 2.5)
    deg = np.degrees(mean_gyro)
    print(f"  mean gyro = ({deg[0]:+.1f}, {deg[1]:+.1f}, {deg[2]:+.1f}) deg/s"
          f"  -> dominant axis {_dominant(mean_gyro)}")
    return mean_gyro


def main() -> None:
    reader = IMUReader()
    reader.start()
    print("Connecting to IMU ... waiting for first sample.")
    t0 = time.time()
    while reader.latest is None:
        if time.time() - t0 > 8:
            print("No IMU data after 8 s. Check the glasses link (see probe_imu.py).")
            reader.stop()
            return
        time.sleep(0.1)
    print("IMU streaming. Put the glasses on for the steps below.")

    input("\n[level] Look straight ahead at the horizon, head level and still.\n"
          "  Press Enter, then HOLD STILL for ~2 s ... ")
    print("  capturing ...")
    _, grav = _capture(reader, 2.0)
    g_dir = grav / np.linalg.norm(grav)
    print(f"  gravity (down) in body frame = ({g_dir[0]:+.2f}, {g_dir[1]:+.2f}, "
          f"{g_dir[2]:+.2f})  -> body 'down' ~ {_dominant(g_dir)}")

    yaw = _move(reader, "yaw", "Turn your head RIGHT (look right), like saying 'no'.")
    pitch = _move(reader, "pitch", "Look UP toward the ceiling (nod up).")
    roll = _move(reader, "roll", "Tilt your head so your RIGHT ear goes toward your shoulder.")

    reader.stop()

    print("\n================= PASTE THIS BLOCK BACK =================")
    print(f"down  (gravity)      : {np.round(g_dir, 3).tolist()}  [{_dominant(g_dir)}]")
    print(f"yaw-right  gyro dir  : {np.round(yaw / (np.linalg.norm(yaw) or 1), 3).tolist()}  [{_dominant(yaw)}]")
    print(f"pitch-up   gyro dir  : {np.round(pitch / (np.linalg.norm(pitch) or 1), 3).tolist()}  [{_dominant(pitch)}]")
    print(f"roll-right gyro dir  : {np.round(roll / (np.linalg.norm(roll) or 1), 3).tolist()}  [{_dominant(roll)}]")
    print("========================================================")


if __name__ == "__main__":
    main()
