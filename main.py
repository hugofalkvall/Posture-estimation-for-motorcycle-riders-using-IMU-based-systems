import time
from sensor import sensor
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
import matplotlib.pyplot as plt

plt.figure()

imu_sensor = sensor(address=0x68)  # change to 0x69 if needed
dt = 1/60
madgwick = Madgwick(sampletime=dt,beta=0.02)
q = np.array([1,0,0,0])

roll_list = []
pitch_list = []
yaw_list = []
time_list = []

t = 0

while True:

    # Read sensor data from the IMU sensor
    result = imu_sensor.read_sensor_data()

    if result is None:
        print("Failed to read sensor data.")
        time.sleep(2)
        continue
    
    accel, gyro = result
    accel = np.array([accel['x'], accel['y'], accel['z']])
    gyro = np.array([gyro['x']/131.0*np.pi/180, gyro['y']/131.0*np.pi/180, gyro['z']/131.0*np.pi/180])  # Convert raw gyroscope values to radians per second


    q = madgwick.updateIMU(q,gyro,accel)
    roll, pitch, yaw = q2euler(q)

    roll_list.append(roll * 180/np.pi)
    pitch_list.append(pitch * 180/np.pi)
    yaw_list.append(yaw * 180/np.pi)

    time_list.append(t)

    t += dt

    print("Accelerometer data:", accel)
    print("Gyroscope data:", gyro)
    print("q=", q)
    print("-" * 30)

    plt.plot(time_list, roll_list, label="Roll")
    plt.plot(time_list, pitch_list, label="Pitch")
    plt.plot(time_list, yaw_list, label="Yaw")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title("IMU Orientation")
    plt.legend()
    plt.grid()

    plt.show()

    # Update frequency of data reading
    time.sleep(dt)