import time
from smbus2 import SMBus
from sensor import sensor

# Multiplexer class to control the TCA9548A I2C multiplexer with sensor reading functionality
class Multiplexer:

    # Construct the multiplexer with the given I2C bus number and address
    def __init__(self, bus_num=1, address=0x70):
        self.bus_num = bus_num
        self.bus = SMBus(bus_num)
        self.address = address
        self._sensors = {}
        self._channel_failures = {}

    # Select a specific channel on the multiplexer by writing the appropriate byte to the control register
    def select_channel(self, channel: int):
        action = 1 << channel
        for _ in range(3):
            try:
                self.bus.write_byte(self.address, action)
                # Channel switch is effectively immediate; avoid large fixed delay.
                time.sleep(0.001)
                return True
            except OSError as e:
                time.sleep(0.005)
        return False

    def recover_bus(self):
        try:
            self.bus.close()
        except Exception:
            pass

        self.bus = SMBus(self.bus_num)
        self._sensors.clear()
        self._channel_failures.clear()


# Function to read sensor data from a specific channel
def read_mpu_on_channel(mux: Multiplexer, channel: int, mpu_addr: int = 0x68):
    SENSOR_RESET_FAILS = 3
    BUS_RECOVERY_FAILS = 9

    if not mux.select_channel(channel):
        fails = mux._channel_failures.get(channel, 0) + 1
        mux._channel_failures[channel] = fails
        if fails >= BUS_RECOVERY_FAILS:
            print(f"I2C bus recovery triggered ({channel} Ch sensor)")
            mux.recover_bus()
        return None

    # Reuse one sensor instance per channel to avoid expensive re-initialization.
    if channel not in mux._sensors:
        mux._sensors[channel] = sensor(address=mpu_addr, bus=mux.bus, channel=channel)

    imu = mux._sensors[channel]
    result = imu.read_sensor_data()
    if result is None:
        fails = mux._channel_failures.get(channel, 0) + 1
        mux._channel_failures[channel] = fails

        if fails == SENSOR_RESET_FAILS:
            print(f"Reinitializing sensor after repeated read failures ({channel} Ch sensor)")
            mux._sensors.pop(channel, None)
        elif fails >= BUS_RECOVERY_FAILS:
            print(f"I2C bus recovery triggered after repeated read failures ({channel} Ch sensor)")
            mux.recover_bus()
    else:
        mux._channel_failures[channel] = 0

    return result
