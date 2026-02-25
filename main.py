import time
from collections import deque
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler

from multiplex import Multiplexer, read_mpu_on_channel

# Configuration variables 
BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 2, 7]

DT = 1 / 60 # 60 Hz update rate
BETA = 2.5  # Madgwick filter gain

OUT_PATH = "euler_angles.txt"
KEEP_LAST = 10  # per channel

# Initilaziation of multiplexer 
mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)

# Create a filter, quaternion, and history for each channel
filters = {ch: Madgwick(sampletime=DT, beta=BETA) for ch in MUX_CHANNELS}
quats = {ch: np.array([1.0, 0.0, 0.0, 0.0]) for ch in MUX_CHANNELS}
history = {ch: deque(maxlen=KEEP_LAST) for ch in MUX_CHANNELS}

# Runtime reference time
t0 = time.monotonic()

# Main loop
try:
    while True:

        # Read from each channel on mux, update the filter, and store the results
        for ch in MUX_CHANNELS:
            result = read_mpu_on_channel(mux, ch, MPU_ADDR)
            if result is None:
                continue

            # Unpack acceleration and gyro data in a result tuple 
            accel, gyro = result

            # Structure the data into arrays, converting gyro to radians/sec
            accel = np.array([accel["x"], accel["y"], accel["z"]], dtype=float)
            gyro = np.array([
                gyro["x"] / 131.0 * np.pi / 180.0,
                gyro["y"] / 131.0 * np.pi / 180.0,
                gyro["z"] / 131.0 * np.pi / 180.0,
            ], dtype=float)

            # Update the filter and convert to Euler angles
            quats[ch] = filters[ch].updateIMU(quats[ch], gyro, accel)
            roll, pitch, yaw = q2euler(quats[ch])

            # Convert radians to degrees
            roll_deg = float(roll * 180.0 / np.pi)
            pitch_deg = float(pitch * 180.0 / np.pi)
            yaw_deg = float(yaw * 180.0 / np.pi)

            # Calculate time 
            t_rel = time.monotonic() - t0      # starts at 0
            epoch = time.time()                # for latency if clocks are synced

            # Store values and time 
            history[ch].append((t_rel, epoch, roll_deg, pitch_deg, yaw_deg))        

        # Write to file: ch,t_rel,epoch,roll,pitch,yaw
        with open(OUT_PATH, "w") as f:
            for ch in MUX_CHANNELS:
                for t_rel, epoch, r, p, y in history[ch]:
                    f.write(f"{ch},{t_rel:.6f},{epoch:.6f},{r:.6f},{p:.6f},{y:.6f}\n")

        # Frequency control
        time.sleep(DT)

except KeyboardInterrupt:
    print("Exiting...")
    pass