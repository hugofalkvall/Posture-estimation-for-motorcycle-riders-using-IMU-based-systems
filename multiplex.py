import time
from smbus2 import SMBus
from sensor import sensor


class Multiplexer:
    def __init__(self, bus_num=1, address=0x70):
        self.bus = SMBus(bus_num)
        self.address = address

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



def read_mpu_on_channel(mux: Multiplexer, channel: int, mpu_addr: int = 0x68):
    mux.select_channel(channel)
    time.sleep(0.01)  # liten delay så buss/stabilisering hinner ske

    imu = sensor(address=mpu_addr)  # init efter kanalval är enklast/säkraste
    result = imu.read_sensor_data()
    return result


if __name__ == "__main__":
    BUS = 1
    MUX_ADDR = 0x70
    MPU_ADDR = 0x68  # ändra till 0x69 om AD0 är HIGH på din MPU

    mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)

    while True:
        for ch in [0, 2]:
            result = read_mpu_on_channel(mux, ch, MPU_ADDR)

            if result is None:
                print(f"CH{ch}: read failed")
            else:
                accel, gyro = result
                print(f"CH{ch}: accel={accel} gyro={gyro}")

        print("-" * 40)
        time.sleep(0.1)
