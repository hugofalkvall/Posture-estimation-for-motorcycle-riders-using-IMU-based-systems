import time
from smbus2 import SMBus

MUX_ADDR = 0x70
MPU_ADDR = 0x68
CHANNELS = [0, 2, 3]


def select_channel(bus, ch):
    try:
        bus.write_byte(MUX_ADDR, 1 << ch)
        time.sleep(0.01)  # viktig delay
        return True
    except Exception as e:
        print(f"[MUX ERROR] channel {ch}:", e)
        return False


def test_channel(bus, ch):
    if not select_channel(bus, ch):
        return

    try:
        who_am_i = bus.read_byte_data(MPU_ADDR, 0x75)
        print(f"[OK] CH{ch} WHO_AM_I = {who_am_i}")
    except Exception as e:
        print(f"[FAIL] CH{ch}:", e)


def main():
    print("Starting I2C debug...\n")

    bus = SMBus(1)

    # Test multiplexer exists
    try:
        bus.write_byte(MUX_ADDR, 0x00)
        print("[OK] Multiplexer responding at 0x70\n")
    except Exception as e:
        print("[FATAL] Multiplexer not responding:", e)
        return

    # Test each channel
    for ch in CHANNELS:
        test_channel(bus, ch)

    print("\nDone.")


if __name__ == "__main__":
    main()