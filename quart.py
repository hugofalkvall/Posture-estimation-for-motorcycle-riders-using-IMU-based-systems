import time
import numpy as np
from ahrs.filters import Madgwick

from multiplex import Multiplexer, read_mpu_on_channel


# =========================
# Math utils
# =========================
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3)

    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),     2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),     1.0 - 2.0 * (x * x + y * y)],
    ])


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return q


def rot_x_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,  -s],
        [0.0, s,   c],
    ])


def rot_y_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rot_z_deg(a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def frame_transformation(target, reference):
    """
    Express target frame in the reference frame coordinate system.
    Both inputs are 3x3 rotation matrices in the same parent frame.
    """
    return reference.T @ target


def yaw_from_rotation_matrix_zyx(R):
    """
    Extract yaw (degrees) from a ZYX rotation matrix.
    """
    return np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))


# =========================
# Config
# =========================
BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 3, 2]

DT = 1 / 50
BETA = 0.2
CALIB_SAMPLES = 100

# Yaw reset heuristic
YAW_THRESHOLD_RAD = np.deg2rad(10.0)
YAW_RESET_SAMPLES = 1000

OUT_PATH = "quaternions.txt"

# =========================
# Mounting correction
# =========================
# ch0: z-axis up
# ch2/ch3: x-axis up
#
# If the signs are wrong in your setup, test rot_y_deg(-90) instead.
MOUNT_CORRECTION = {
    0: np.eye(3),
    2: rot_y_deg(90),
    3: rot_y_deg(90),
}


# =========================
# Init
# =========================
mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)

filters = {ch: Madgwick(sampletime=DT, beta=BETA) for ch in MUX_CHANNELS}
quats = {ch: np.array([1.0, 0.0, 0.0, 0.0], dtype=float) for ch in MUX_CHANNELS}
gyro_bias = {ch: np.zeros(3, dtype=float) for ch in MUX_CHANNELS}
last_update = {ch: None for ch in MUX_CHANNELS}

# Global yaw offsets, applied after mounting correction
yaw_offset_deg = {ch: 0.0 for ch in MUX_CHANNELS}
latest_yaw_deg = {ch: 0.0 for ch in MUX_CHANNELS}

# Latest transformed output per channel:
# ch -> (t_rel, epoch, qw, qx, qy, qz)
latest_output = {ch: None for ch in MUX_CHANNELS}

inactive_counter = 0
t0 = time.monotonic()


# =========================
# Gyro bias calibration
# =========================
for ch in MUX_CHANNELS:
    bias_sum = np.zeros(3, dtype=float)
    count = 0

    for _ in range(CALIB_SAMPLES):
        res = read_mpu_on_channel(mux, ch, MPU_ADDR)
        if res is None:
            time.sleep(DT)
            continue

        _, gyro = res
        bias_sum += np.deg2rad([gyro["x"], gyro["y"], gyro["z"]])
        count += 1
        time.sleep(DT)

    if count > 0:
        gyro_bias[ch] = bias_sum / count


# =========================
# Main loop
# =========================
try:
    with open(OUT_PATH, "w", buffering=1) as f:
        while True:
            frame_start = time.monotonic()

            # Raw/corrected world-aligned body matrices for this frame
            body_world_raw = {}
            body_world_corrected = {}
            gyro_frame = {}

            # -------- Read and update all channels first --------
            for ch in MUX_CHANNELS:
                res = read_mpu_on_channel(mux, ch, MPU_ADDR)
                if res is None:
                    continue

                accel, gyro = res

                accel = np.array([accel["x"], accel["y"], accel["z"]], dtype=float)
                gyro = np.deg2rad([gyro["x"], gyro["y"], gyro["z"]]) - gyro_bias[ch]
                gyro_frame[ch] = gyro

                now = time.monotonic()
                dt = DT if last_update[ch] is None else max(1e-4, now - last_update[ch])
                last_update[ch] = now
                filters[ch].Dt = dt

                # Update Madgwick
                quats[ch] = filters[ch].updateIMU(quats[ch], gyro, accel)

                # Quaternion -> world matrix
                R_world = quaternion_to_rotation_matrix(quats[ch])

                # Apply fixed sensor->body mounting correction
                R_body_world = R_world @ MOUNT_CORRECTION[ch]
                body_world_raw[ch] = R_body_world

                # Extract yaw only for reset heuristic
                latest_yaw_deg[ch] = yaw_from_rotation_matrix_zyx(R_body_world)

            # Need all channels present before doing chained transforms / file write
            if not all(ch in body_world_raw for ch in MUX_CHANNELS):
                elapsed = time.monotonic() - frame_start
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # -------- Global yaw inactivity detection --------
            # Reset only if ALL channels are yaw-inactive in this frame
            if all(abs(gyro_frame[ch][2]) < YAW_THRESHOLD_RAD for ch in MUX_CHANNELS):
                inactive_counter += 1
            else:
                inactive_counter = 0

            if inactive_counter >= YAW_RESET_SAMPLES:
                for ch in MUX_CHANNELS:
                    yaw_offset_deg[ch] = latest_yaw_deg[ch]
                inactive_counter = 0

            # -------- Apply yaw offsets on matrix level --------
            for ch in MUX_CHANNELS:
                body_world_corrected[ch] = body_world_raw[ch] @ rot_z_deg(-yaw_offset_deg[ch])

            # -------- Relative chain --------
            # ch0: world/body reference
            R0_world = body_world_corrected[0]

            # ch2: relative to ch0
            R2_world = body_world_corrected[2]
            R2_rel = frame_transformation(R2_world, R0_world)

            # ch3: relative to ch2
            R3_world = body_world_corrected[3]
            R3_rel = frame_transformation(R3_world, R2_world)

            out_matrices = {
                0: R0_world,
                2: R2_rel,
                3: R3_rel,
            }

            # -------- Write quaternion output for all 3 at once --------
            t_rel = time.monotonic() - t0
            epoch = time.time()

            for ch in MUX_CHANNELS:
                q_out = rotation_matrix_to_quaternion(out_matrices[ch])
                latest_output[ch] = (t_rel, epoch, q_out[0], q_out[1], q_out[2], q_out[3])

            f.seek(0)
            for ch in MUX_CHANNELS:
                t_rel_ch, epoch_ch, qw, qx, qy, qz = latest_output[ch]
                f.write(
                    f"{ch},{t_rel_ch:.6f},{epoch_ch:.6f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n"
                )
            f.truncate()
            f.flush()

            # Keep approximately 50 Hz
            elapsed = time.monotonic() - frame_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Exiting...")