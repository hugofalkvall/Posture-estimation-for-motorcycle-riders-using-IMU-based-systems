from mpu6050 import mpu6050

class sensor:
    def __init__(self, address=0x68):
        self.address = address
        self.mpu = None

        try:
            self.mpu = mpu6050(self.address)
        except OSError as e:
            print("I2C error when initializing MPU6050:", e)
        except Exception as e:
            print("Unexpected error when initializing MPU6050:", e)


    def read_sensor_data(self):
        if self.mpu is None:
            return None

        try:
            accelerometer_data = self.mpu.get_accel_data()
            gyroscope_data = self.mpu.get_gyro_data()
            return accelerometer_data, gyroscope_data

        except OSError as e:
            print("I2C read/write error:", e)
            return None
        except Exception as e:
            print("Unexpected error while reading sensor:", e)
            return None