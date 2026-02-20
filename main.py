import time
from sensor import sensor

imu_sensor = sensor(address=0x68)
dt = 1/60

while True:

    # Read sensor data from the IMU sensor
    result = imu_sensor.read_sensor_data()

    if result is None:
        print("Failed to read sensor data.")
        time.sleep(2)
        continue

    # Add accelerometer and gyroscope data to the result list 
    accel, gyro = result 
    print("Accelerometer data:", accel)
    print("Gyroscope data:", gyro)
    print("-" * 30)

    # Update frequency of data reading
    time.sleep(dt)