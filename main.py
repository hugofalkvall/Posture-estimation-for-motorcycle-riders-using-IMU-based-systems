import time
from collections import deque
from sensor import sensor
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler

imu_sensor = sensor(address=0x68)  # change to 0x69 if needed
dt = 1/60
madgwick = Madgwick(sampletime=dt, beta=0.02)
q = np.array([1.0, 0.0, 0.0, 0.0])

t = 0.0
out_path = "euler_angles.txt"

# Keep only the last 10 samples in memory
last50 = deque(maxlen=10)

try:
    while True:
        result = imu_sensor.read_sensor_data()
        if result is None:
            time.sleep(2)
            print("Retrying sensor read after error...")
            continue

        accel, gyro = result
        accel = np.array([accel["x"], accel["y"], accel["z"]], dtype=float)
        gyro = np.array([
            gyro["x"] / 131.0 * np.pi / 180.0,
            gyro["y"] / 131.0 * np.pi / 180.0,
            gyro["z"] / 131.0 * np.pi / 180.0
        ], dtype=float)

        q = madgwick.updateIMU(q, gyro, accel)
        roll, pitch, yaw = q2euler(q)

        roll_deg = float(roll * 180.0 / np.pi)
        pitch_deg = float(pitch * 180.0 / np.pi)
        yaw_deg = float(yaw * 180.0 / np.pi)

        last50.append((t, roll_deg, pitch_deg, yaw_deg))

        # Overwrite the file each iteration with only the last 10 lines
        with open(out_path, "w") as f:
            for tt, r, p, y in last50:
                pi_epoch = time.time()
                f.write(f"{pi_epoch:.6f},{roll_deg:.6f},{pitch_deg:.6f},{yaw_deg:.6f}\n")

        t += dt
        time.sleep(dt)

except KeyboardInterrupt:
    pass