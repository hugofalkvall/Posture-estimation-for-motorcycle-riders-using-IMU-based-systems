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


def wrap_angle_deg(a):
    """Wrap angle in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def twist_angle_deg_from_quaternion(q, axis):
    """
    Signed twist angle around a given axis (x/y/z) from quaternion [w, x, y, z].
    This is an axial rotation measure, not an Euler decomposition.
    """
    q = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return 0.0
    q = q / q_norm

    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        return 0.0
    axis = axis / axis_norm

    w = q[0]
    v = q[1:]
    s = float(np.dot(v, axis))

    angle_rad = 2.0 * np.arctan2(s, w)
    return wrap_angle_deg(np.rad2deg(angle_rad))


def axial_angles_zyx_deg_from_rotation_matrix(R):
    """
    Return axial angles [Z, Y, X] in degrees from a rotation matrix.
    These are per-axis twist angles and are not Euler angles.
    """
    q = rotation_matrix_to_quaternion(R)
    az = twist_angle_deg_from_quaternion(q, np.array([0.0, 0.0, 1.0]))
    ay = twist_angle_deg_from_quaternion(q, np.array([0.0, 1.0, 0.0]))
    ax = twist_angle_deg_from_quaternion(q, np.array([1.0, 0.0, 0.0]))
    return np.array([az, ay, ax], dtype=float)


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
MAX_FILTER_DT = DT * 3

# Yaw reset heuristic
YAW_THRESHOLD_RAD = np.deg2rad(10.0)
YAW_RESET_SAMPLES = 1000

OUT_PATH = "quaternions.txt"
ANGLE_OUT_PATH = "axial_angles.txt"
CAPTURED_ANGLE_OUT_PATH = "captured_axial_angles.txt"

CAPTURE_AXIAL_DATA = False
CAPTURE_START_FRAME = 500

# =========================
# Mounting correction
# =========================
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

yaw_offset_deg = {ch: 0.0 for ch in MUX_CHANNELS}
latest_yaw_deg = {ch: 0.0 for ch in MUX_CHANNELS}

latest_output = {ch: None for ch in MUX_CHANNELS}
latest_angle_output = {ch: None for ch in MUX_CHANNELS}
neutral_axial_deg = {ch: None for ch in MUX_CHANNELS}

inactive_counter = 0
frame_counter = 0
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
capture_file = open(CAPTURED_ANGLE_OUT_PATH, "w", buffering=1) if CAPTURE_AXIAL_DATA else None

try:
    with open(OUT_PATH, "w", buffering=1) as f, open(ANGLE_OUT_PATH, "w", buffering=1) as f_angles:
        while True:
            frame_start = time.monotonic()
            frame_counter += 1

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
                dt = DT if last_update[ch] is None else min(MAX_FILTER_DT, max(1e-4, now - last_update[ch]))
                last_update[ch] = now
                filters[ch].Dt = dt

                quats[ch] = filters[ch].updateIMU(quats[ch], gyro, accel)

                R_world = quaternion_to_rotation_matrix(quats[ch])
                R_body_world = R_world @ MOUNT_CORRECTION[ch]
                body_world_raw[ch] = R_body_world

                latest_yaw_deg[ch] = yaw_from_rotation_matrix_zyx(R_body_world)

            # -------- Global yaw inactivity detection --------
            if all(ch in gyro_frame for ch in MUX_CHANNELS) and all(
                abs(gyro_frame[ch][2]) < YAW_THRESHOLD_RAD for ch in MUX_CHANNELS
            ):
                inactive_counter += 1
            else:
                inactive_counter = 0
            
            if inactive_counter >= YAW_RESET_SAMPLES:
                for ch in MUX_CHANNELS:
                    yaw_offset_deg[ch] = latest_yaw_deg[ch]
                inactive_counter = 0

            # -------- Apply yaw offsets on matrix level --------
            for ch in body_world_raw:
                body_world_corrected[ch] = rot_z_deg(-yaw_offset_deg[ch]) @ body_world_raw[ch]

            # -------- Relative chain --------
            R0_world = body_world_corrected.get(0)

            R2_world = body_world_corrected.get(2)
            R2_rel = frame_transformation(R2_world, R0_world) if (R2_world is not None and R0_world is not None) else None

            R3_world = body_world_corrected.get(3)
            R3_rel = frame_transformation(R3_world, R2_world) if (R3_world is not None and R2_world is not None) else None

            out_matrices = {
                0: R0_world,
                2: R2_rel,
                3: R3_rel,
            }

            t_rel = time.monotonic() - t0
            epoch = time.time()

            for ch in MUX_CHANNELS:
                if out_matrices[ch] is None:
                    latest_output[ch] = (t_rel, epoch, np.nan, np.nan, np.nan, np.nan)
                    latest_angle_output[ch] = (t_rel, epoch, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                    continue

                q_out = rotation_matrix_to_quaternion(out_matrices[ch])
                latest_output[ch] = (t_rel, epoch, q_out[0], q_out[1], q_out[2], q_out[3])

                abs_zyx = axial_angles_zyx_deg_from_rotation_matrix(out_matrices[ch])

                if neutral_axial_deg[ch] is None:
                    neutral_axial_deg[ch] = abs_zyx.copy()

                rel_zyx = np.array(
                    [wrap_angle_deg(abs_zyx[i] - neutral_axial_deg[ch][i]) for i in range(3)],
                    dtype=float,
                )

                latest_angle_output[ch] = (
                    t_rel,
                    epoch,
                    abs_zyx[0],
                    abs_zyx[1],
                    abs_zyx[2],
                    rel_zyx[0],
                    rel_zyx[1],
                    rel_zyx[2],
                )

            # -------- Quaternion stream: always current 3-line snapshot --------
            f.seek(0)
            for ch in MUX_CHANNELS:
                t_rel_ch, epoch_ch, qw, qx, qy, qz = latest_output[ch]
                f.write(
                    f"{ch},{t_rel_ch:.6f},{epoch_ch:.6f},"
                    f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n"
                )
            f.truncate()
            f.flush()

            # -------- Axial angle output after startup delay --------
            if frame_counter >= CAPTURE_START_FRAME:
                f_angles.seek(0)

                for ch in MUX_CHANNELS:
                    (
                        t_rel_ch,
                        epoch_ch,
                        abs_z,
                        abs_y,
                        abs_x,
                        rel_z,
                        rel_y,
                        rel_x,
                    ) = latest_angle_output[ch]

                    line = (
                        f"{ch},{epoch_ch:.6f},{t_rel_ch:.6f},"
                        f"{abs_z:.6f},{abs_y:.6f},{abs_x:.6f},"
                        f"{rel_z:.6f},{rel_y:.6f},{rel_x:.6f}\n"
                    )

                    # Realtime streaming file: always overwritten to exactly 3 rows
                    f_angles.write(line)

                    # Local capture file: append full history
                    if CAPTURE_AXIAL_DATA and capture_file is not None:
                        capture_file.write(line)

                f_angles.truncate()
                f_angles.flush()

                if CAPTURE_AXIAL_DATA and capture_file is not None:
                    capture_file.flush()

            elapsed = time.monotonic() - frame_start

            # -------- Performance test section --------
            # if frame_counter % 50 == 0:   # every ~1 second at 50 Hz
            #     frame_ms = elapsed * 1000.0

            #     if elapsed > 0:
            #         loop_hz = 1.0 / elapsed
            #     else:
            #         loop_hz = 0.0

            #     lag = "YES" if elapsed > DT else "NO"

            #     print(
            #         f"[TEST] Frame {frame_counter} | "
            #         f"time: {frame_ms:.2f} ms | "
            #         f"rate: {loop_hz:.1f} Hz | "
            #         f"target: {1/DT:.1f} Hz | "
            #         f"missed deadline: {lag}"
            #     )

            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            

except KeyboardInterrupt:
    print("Exiting...")

finally:
    if capture_file is not None:
        capture_file.close()