import time
from sensor import sensor
import numpy as np
from ahrs.filters import Madgwick


imu_sensor = sensor(address=0x68)  # change to 0x69 if needed
dt = 1/60
madgwick = Madgwick(sampletime=dt,beta=0.02)
q = np.array([1,0,0,0])

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

    print("Accelerometer data:", accel)
    print("Gyroscope data:", gyro)
    print("q=", q)
    print("-" * 30)

    # Update frequency of data reading
    time.sleep(dt)