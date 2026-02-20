import time
from sensor import sensor
from ahrs.filters import Madgwick

<<<<<<< HEAD

imu_sensor = sensor(address=0x68)  # change to 0x69 if needed
=======
imu_sensor = sensor(address=0x68)
>>>>>>> db6232c3ff42ffa530043a5acc99d6bd69287d68
dt = 1/60
<<<<<<< HEAD
madgwick = Madgwick(sampletime=dt)
q = np.array([1,0,0,0])
=======

>>>>>>> db6232c3ff42ffa530043a5acc99d6bd69287d68
while True:

    # Read sensor data from the IMU sensor
    result = imu_sensor.read_sensor_data()

    if result is None:
        print("Failed to read sensor data.")
        time.sleep(2)
        continue

<<<<<<< HEAD
    
    accel, gyro = result
    accel = np.array(accel)
    gyro = np.array(gyro)
    q = madgwick.updateIMU(q=q,gyr=gyro,acc=accel)
=======
    # Add accelerometer and gyroscope data to the result list 
    accel, gyro = result 
>>>>>>> db6232c3ff42ffa530043a5acc99d6bd69287d68
    print("Accelerometer data:", accel)
    print("Gyroscope data:", gyro)
    print("-" * 30)
    print("q=", q)

    # Update frequency of data reading
    time.sleep(dt)