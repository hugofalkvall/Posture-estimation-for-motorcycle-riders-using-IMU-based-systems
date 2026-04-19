from smbus2 import SMBus

# MPU6050 registers
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B


def _to_int16(msb, lsb):
    value = (msb << 8) | lsb
    return value - 65536 if value > 32767 else value


# Sensor class to interface with the MPU6050 IMU sensor
class sensor:
    # Construct the sensor with the given I2C address
    def __init__(self, address=0x68, bus=None, bus_num=1, channel=None):
        self.address = address
        self.bus = bus if bus is not None else SMBus(bus_num)
        self._owns_bus = bus is None
        self._initialized = False
        self.channel = channel
        self._init_sensor()

    def _sensor_label(self):
        if self.channel is None:
            return ""
        return f" ({self.channel} Ch sensor)"

    def _init_sensor(self):
        try:
            # Wake the MPU6050 and keep default full-scale ranges:
            # accel +/-2g, gyro +/-250 deg/s.
            self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
            self.bus.write_byte_data(self.address, SMPLRT_DIV, 0x00)
            self.bus.write_byte_data(self.address, CONFIG, 0x03)
            self.bus.write_byte_data(self.address, GYRO_CONFIG, 0x00)
            self.bus.write_byte_data(self.address, ACCEL_CONFIG, 0x00)
            self._initialized = True
        except OSError as e:
            self._initialized = False
            print(f"I2C error when initializing MPU6050{self._sensor_label()}:", e)
        except Exception as e:
            self._initialized = False
            print(f"Unexpected error when initializing MPU6050{self._sensor_label()}:", e)

    # Read accelerometer and gyroscope data from the sensor
    def read_sensor_data(self):
        if not self._initialized:
            self._init_sensor()
            if not self._initialized:
                return None

        try:
            # Single burst read for accel, temp, gyro in one I2C transaction.
            data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)

            ax_raw = _to_int16(data[0], data[1])
            ay_raw = _to_int16(data[2], data[3])
            az_raw = _to_int16(data[4], data[5])
            gx_raw = _to_int16(data[8], data[9])
            gy_raw = _to_int16(data[10], data[11])
            gz_raw = _to_int16(data[12], data[13])

            # Scale to g and deg/s to match previous code expectations.
            accelerometer_data = {
                "x": ax_raw / 16384.0,
                "y": ay_raw / 16384.0,
                "z": az_raw / 16384.0,
            }
            gyroscope_data = {
                "x": gx_raw / 131.0,
                "y": gy_raw / 131.0,
                "z": gz_raw / 131.0,
            }
            return accelerometer_data, gyroscope_data

        except OSError as e:
            print(f"I2C read/write error{self._sensor_label()}:", e)
            self._initialized = False
            return None
        except Exception as e:
            print(f"Unexpected error while reading sensor{self._sensor_label()}:", e)
            return None

    def __del__(self):
        if self._owns_bus:
            try:
                self.bus.close()
            except Exception:
                pass
