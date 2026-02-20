import time
from sensor import sensor

imu_sensor = sensor(address=0x68)  # change to 0x69 if needed

while True:
    result = imu_sensor.read_sensor_data()

    if result is None:
        print("Failed to read sensor data.")
        time.sleep(2)
        continue

    accel, gyro = result
    print("Accelerometer data:", accel)
    print("Gyroscope data:", gyro)
    print("-" * 30)

    time.sleep(1/60)