import time
from collections import deque
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
from scipy.spatial.transform import Rotation

from multiplex import Multiplexer, read_mpu_on_channel
def euler2rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # Compute individual rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    return R

def frame_transformation(frame1, frame2):

    # Inverse
    inverse_frame = np.linalg.inv(frame2)

    # Transform
    transformed_frame = inverse_frame @ frame1
    return transformed_frame

def rotation_matrix2euler(matrix):
    # Extract 3x3 rotation matrix to euler angles
    rot = Rotation.from_matrix(matrix)
    return rot.as_euler('zyx', degrees=True)


# Configuration variables 
BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 2, 3]

DT = 1 / 50  # 50 Hz loop delay target (real per-channel dt is measured below)
BETA = 0.2  # Madgwick filter gain (lower is usually better for IMU-only yaw)
RAD2DEG = 180.0 / np.pi
CALIB_SAMPLES = 0
MAX_PROCESS_LATENCY_MS = 100.0

OUT_PATH = "euler_angles.txt"
KEEP_LAST = 50  # per channel
LATENCY_WINDOW = 300  # moving average window over latest samples
LATENCY_PRINT_EVERY = 5.0  # seconds
STREAM_REFRESH_EVERY = 0.1  # seconds

# Initilaziation of multiplexer 
mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)

# Create a filter, quaternion, and history for each channel
filters = {ch: Madgwick(sampletime=DT, beta=BETA) for ch in MUX_CHANNELS}
quats = {ch: np.array([1.0, 0.0, 0.0, 0.0]) for ch in MUX_CHANNELS}
history = {ch: deque(maxlen=KEEP_LAST) for ch in MUX_CHANNELS}
gyro_bias = {ch: np.zeros(3, dtype=float) for ch in MUX_CHANNELS}
last_update = {ch: None for ch in MUX_CHANNELS}
start_due = time.monotonic()
phase_step = DT / max(1, len(MUX_CHANNELS))
next_due = {ch: start_due + (i * phase_step) for i, ch in enumerate(MUX_CHANNELS)}
sample_times = {ch: deque(maxlen=200) for ch in MUX_CHANNELS}
latency_window = deque(maxlen=LATENCY_WINDOW)
latency_total_ms = 0.0
latency_total_count = 0
latency_min_ms = float("inf")
latency_max_ms = 0.0

# Runtime reference time
t0 = time.monotonic()
last_print = 0.0
last_stream_refresh = 0.0

# Calibrate gyro bias per channel while sensors are stationary.
for ch in MUX_CHANNELS:
    acc_bias = np.zeros(3, dtype=float)
    count = 0
    for _ in range(CALIB_SAMPLES):
        result = read_mpu_on_channel(mux, ch, MPU_ADDR)
        if result is None:
            time.sleep(DT)
            continue
        _, gyro = result
        acc_bias += np.deg2rad(np.array([gyro["x"], gyro["y"], gyro["z"]], dtype=float))
        count += 1
        time.sleep(DT)
    if count > 0:
        gyro_bias[ch] = acc_bias / count

# Main loop
try:
    with open(OUT_PATH, "w", buffering=1) as f:
        frame_matrix=None

        counter=0

        while True:
            now = time.monotonic()
            due_channels = [ch for ch in MUX_CHANNELS if now >= next_due[ch]]

            if not due_channels:
                sleep_for = min(next_due.values()) - now
                time.sleep(max(0.0005, min(sleep_for, DT / 2)))
                continue

            # Process only channels that are due; this enforces 50 Hz target per channel.
            for ch in due_channels:
                sample_start = time.perf_counter()
                result = read_mpu_on_channel(mux, ch, MPU_ADDR)
                process_end = time.monotonic()

                # Keep the schedule aligned to real time and avoid drift/catch-up bursts.
                next_due[ch] += DT
                if process_end > next_due[ch]:
                    missed = int((process_end - next_due[ch]) / DT) + 1
                    next_due[ch] += missed * DT

                if result is None:
                    continue

                # Unpack acceleration and gyro data in a result tuple
                accel, gyro = result

                # Structure the data into arrays, converting gyro deg/s -> rad/s
                accel = np.array([accel["x"], accel["y"], accel["z"]], dtype=float)
                gyro = np.array([
                    gyro["x"],
                    gyro["y"],
                    gyro["z"],
                ], dtype=float)
                gyro = np.deg2rad(gyro) - gyro_bias[ch]

                # Use measured per-channel dt.
                now_ch = process_end
                dt_ch = DT if last_update[ch] is None else max(1e-4, now_ch - last_update[ch])
                last_update[ch] = now_ch
                if hasattr(filters[ch], "Dt"):
                    filters[ch].Dt = dt_ch

                # Update the filter and convert to Euler angles
                quats[ch] = filters[ch].updateIMU(quats[ch], gyro, accel)
                roll, pitch, yaw = q2euler(quats[ch])

                # Convert radians to degrees
                roll_deg = float(roll * RAD2DEG)
                pitch_deg = float(pitch * RAD2DEG)
                yaw_deg = float(yaw * RAD2DEG)

                if(ch==0):
                    frame_matrix=euler2rotation_matrix(roll_deg,pitch_deg,yaw_deg)
                else:
                    R = euler2rotation_matrix(roll_deg, pitch_deg, yaw_deg)
                    R_trans = frame_transformation(R,frame_matrix)
                    angles = rotation_matrix2euler(R_trans)
                    roll_deg=angles[0]
                    pitch_deg=angles[1]
                    yaw_deg=angles[2]

                # Calculate time
                t_rel = time.monotonic() - t0      # starts at 0
                epoch = time.time()                # for latency if clocks are synced

                # Store values and time
                history[ch].append((t_rel, epoch, roll_deg, pitch_deg, yaw_deg))
                sample_times[ch].append(now_ch)

                # End-to-end latency per channel sample in ms (read -> filter -> store).
                sample_latency_ms = (time.perf_counter() - sample_start) * 1000.0
                latency_window.append(sample_latency_ms)
                latency_total_ms += sample_latency_ms
                latency_total_count += 1
                latency_min_ms = min(latency_min_ms, sample_latency_ms)
                latency_max_ms = max(latency_max_ms, sample_latency_ms)

            now = time.monotonic()
            if (now - last_print) >= LATENCY_PRINT_EVERY and latency_total_count > 0:
                avg_window_ms = sum(latency_window) / len(latency_window)
                avg_total_ms = latency_total_ms / latency_total_count
                win_min_ms = min(latency_window)
                win_max_ms = max(latency_window)
                spike_high_ms = win_max_ms - avg_window_ms
                spike_low_ms = avg_window_ms - win_min_ms
                rate_parts = []
                for ch in MUX_CHANNELS:
                    if len(sample_times[ch]) >= 2:
                        elapsed = sample_times[ch][-1] - sample_times[ch][0]
                        hz = (len(sample_times[ch]) - 1) / elapsed if elapsed > 0 else 0.0
                        rate_parts.append(f"ch{ch}:{hz:.1f}Hz")
                    else:
                        rate_parts.append(f"ch{ch}:n/a")

                last_print = now

            # Keep streamed plot input bounded in size to avoid growing redraw cost.
            if (now - last_stream_refresh) >= STREAM_REFRESH_EVERY:
                f.seek(0)
                for ch in MUX_CHANNELS:
                    for t_rel, epoch, r, p, y in history[ch]:
                        f.write(f"{ch},{t_rel:.6f},{epoch:.6f},{r:.6f},{p:.6f},{y:.6f}\n")
                f.truncate()
                f.flush()
                last_stream_refresh = now

except KeyboardInterrupt:
    print("Exiting...")
    pass

