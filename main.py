import time
import numpy as np
from ahrs.filters import Madgwick

from multiplex import Multiplexer, read_mpu_on_channel


# =========================
# Math utils
# =========================
def quaternion_to_rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - w*z), 2.0 * (x*z + w*y)],
        [2.0 * (x*y + w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - w*x)],
        [2.0 * (x*z - w*y), 2.0 * (y*z + w*x), 1.0 - 2.0 * (x*x + y*y)],
    ])


def rot_x_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def rot_y_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def rot_z_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def frame_transformation(target, reference):
    return reference.T @ target


def rotation_matrix_to_euler_zyx(R):
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.rad2deg([roll, pitch, yaw])


# =========================
# Config
# =========================
BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 3, 2]

DT = 1 / 50
BETA = 0.2
CALIB_SAMPLES = 10

YAW_THRESHOLD = np.deg2rad(10)
YAW_RESET_SAMPLES = 1000

OUT_PATH = "euler_angles.txt"


# =========================
# Mounting correction
# =========================
MOUNT_CORRECTION = {
    0: np.eye(3),
    2: rot_y_deg(90),   # ändra till -90 om fel håll
    3: rot_y_deg(90),
}


# =========================
# Init
# =========================
mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)

filters = {ch: Madgwick(sampletime=DT, beta=BETA) for ch in MUX_CHANNELS}
quats = {ch: np.array([1.0, 0.0, 0.0, 0.0]) for ch in MUX_CHANNELS}
gyro_bias = {ch: np.zeros(3) for ch in MUX_CHANNELS}
last_update = {ch: None for ch in MUX_CHANNELS}

yaw_offset = {ch: 0.0 for ch in MUX_CHANNELS}
latest_yaw = {ch: 0.0 for ch in MUX_CHANNELS}
inactive_counter = 0

latest_output = {ch: None for ch in MUX_CHANNELS}

t0 = time.monotonic()


# =========================
# Gyro calibration
# =========================
for ch in MUX_CHANNELS:
    acc = np.zeros(3)
    count = 0
    for _ in range(CALIB_SAMPLES):
        res = read_mpu_on_channel(mux, ch, MPU_ADDR)
        if res is None:
            continue
        _, gyro = res
        acc += np.deg2rad([gyro["x"], gyro["y"], gyro["z"]])
        count += 1
        time.sleep(DT)
    if count > 0:
        gyro_bias[ch] = acc / count


# =========================
# Main loop
# =========================
try:
    with open(OUT_PATH, "w", buffering=1) as f:

        R0_world = None
        R2_world = None

        while True:
            frame_start = time.monotonic()

            for ch in MUX_CHANNELS:

                res = read_mpu_on_channel(mux, ch, MPU_ADDR)
                if res is None:
                    continue

                accel, gyro = res

                accel = np.array([accel["x"], accel["y"], accel["z"]])
                gyro = np.deg2rad([gyro["x"], gyro["y"], gyro["z"]]) - gyro_bias[ch]

                now = time.monotonic()
                dt = DT if last_update[ch] is None else max(1e-4, now - last_update[ch])
                last_update[ch] = now
                filters[ch].Dt = dt

                # ===== orientation =====
                quats[ch] = filters[ch].updateIMU(quats[ch], gyro, accel)

                R_world = quaternion_to_rotation_matrix(quats[ch])
                R_body = R_world @ MOUNT_CORRECTION[ch]

                # ===== yaw reset =====
                _, _, yaw_world = rotation_matrix_to_euler_zyx(R_body)
                latest_yaw[ch] = yaw_world

                if abs(gyro[2]) < YAW_THRESHOLD:
                    inactive_counter += 1
                else:
                    inactive_counter = 0

                if inactive_counter >= YAW_RESET_SAMPLES:
                    for k in MUX_CHANNELS:
                        yaw_offset[k] = latest_yaw[k]
                    inactive_counter = 0

                # applicera yaw offset
                R_body = R_body @ rot_z_deg(-yaw_offset[ch])

                # ===== relative chain =====
                if ch == 0:
                    R0_world = R_body
                    R_out = R_body

                elif ch == 2:
                    if R0_world is None:
                        continue
                    R2_world = R_body
                    R_out = frame_transformation(R2_world, R0_world)

                else:  # ch3
                    if R0_world is None or R2_world is None:
                        continue
                    R3_world = R_body
                    R_out = frame_transformation(R3_world, R2_world)

                roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_out)

                t_rel = time.monotonic() - t0
                epoch = time.time()

                latest_output[ch] = (t_rel, epoch, roll, pitch, yaw)

            # ===== write ALL channels at once =====
            if all(latest_output[ch] is not None for ch in MUX_CHANNELS):
                f.seek(0)
                for ch in MUX_CHANNELS:
                    t_rel, epoch, r, p, y = latest_output[ch]
                    f.write(f"{ch},{t_rel:.6f},{epoch:.6f},{r:.6f},{p:.6f},{y:.6f}\n")
                f.truncate()
                f.flush()

            # håll ~50 Hz
            elapsed = time.monotonic() - frame_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Exiting...")