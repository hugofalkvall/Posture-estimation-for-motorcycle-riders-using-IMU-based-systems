import time
from collections import deque
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler

from multiplex import Multiplexer, read_mpu_on_channel

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] into a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw in degrees to a 3x3 rotation matrix."""
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)

    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    cy = np.cos(y)
    sy = np.sin(y)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

def frame_transformation(target_frame, reference_frame):
    """Express target_frame in the reference_frame coordinate system."""
    return reference_frame.T @ target_frame


def rotation_matrix_to_euler_zyx(matrix):
    """Return roll, pitch, yaw in degrees for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    matrix = np.asarray(matrix, dtype=float)
    pitch = np.arcsin(-np.clip(matrix[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        # Gimbal-lock fallback: keep yaw defined and collapse roll to zero.
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])

    return np.rad2deg([roll, pitch, yaw])
# Configuration variables 
BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 3, 2]

DT = 1 / 50  
BETA = 0.2  # Madgwick filter gain (lower is usually better for IMU-only yaw)
RAD2DEG = 180.0 / np.pi
CALIB_SAMPLES = 10
MAX_PROCESS_LATENCY_MS = 100.0
LATENCY_WINDOW = 200  # number of recent samples to track for latency stats
LATENCY_PRINT_EVERY = 5.0  # seconds
yaw_deg_calib_ch0 = 0
yaw_deg_calib_ch2 = 0 
yaw_deg_calib_ch3 = 0

OUT_PATH = "euler_angles.txt"
KEEP_LAST = 1  # per channel
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
        reference_matrix_bike = None
        reference_matrix_ch2 = None 

        counter=0
        inactive_counter=0
        reset_counter=0

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
                threshold = np.deg2rad(10)
                
                if abs(gyro[2]) < threshold:
                    inactive_counter = inactive_counter +1
                else:
                    inactive_counter = 0

                if(inactive_counter >= 1000):
                    
                    if(ch==0):
                        yaw_deg_calib_ch0 = yaw_deg
                        reset_counter = reset_counter +1
                    if(ch==2):
                        yaw_deg_calib_ch2 = yaw_deg
                        reset_counter = reset_counter +1
                    if(ch==3):
                        yaw_deg_calib_ch3 = yaw_deg
                        reset_counter = reset_counter +1

                if(reset_counter == 3):
                    inactive_counter = 0
                    reset_counter = 0

                if(ch==0):
                    yaw_deg -= yaw_deg_calib_ch0
                if(ch==2):
                    yaw_deg -= yaw_deg_calib_ch2
                if(ch==3):
                    yaw_deg -= yaw_deg_calib_ch3 

                sensor_matrix = euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg)

                if(ch==0):
                    reference_matrix_bike = sensor_matrix
                elif(ch==2):
                    if reference_matrix_bike is None:
                        continue
                    relative_matrix = frame_transformation(sensor_matrix, reference_matrix_bike)
                    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_euler_zyx(relative_matrix)
                    reference_matrix_ch2 = relative_matrix
                else: 
                    if reference_matrix_bike is None or reference_matrix_ch2 is None:
                        continue
                   
                    relative_matrix = frame_transformation(sensor_matrix,reference_matrix_ch2)
                    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_euler_zyx(relative_matrix)

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
