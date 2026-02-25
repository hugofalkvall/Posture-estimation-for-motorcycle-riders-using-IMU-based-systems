import time
from smbus2 import SMBus
from sensor import sensor

# Multiplexer class to control the TCA9548A I2C multiplexer with sensor reading functionality
class Multiplexer:

    # Construct the multiplexer with the given I2C bus number and address
    def __init__(self, bus_num=1, address=0x70):
        self.bus = SMBus(bus_num)
        self.address = address

    # Select a specific channel on the multiplexer by writing the appropriate byte to the control register
    def select_channel(self, channel: int):
        action = 1 << channel
        for _ in range(3):
            try:
                self.bus.write_byte(self.address, action)
                time.sleep(0.01)
                return True
            except OSError as e:
                time.sleep(0.05)
        return False


# Function to read sensor data from a specific channel
def read_mpu_on_channel(mux: Multiplexer, channel: int, mpu_addr: int = 0x68):
    mux.select_channel(channel)
    time.sleep(0.01) # Short delay to stabilize channel selection

    # Read sensor data using the sensor class
    imu = sensor(address=mpu_addr)
    result = imu.read_sensor_data()
    return result
