import time
import argparse
from smbus2 import SMBus

MUX_ADDR = 0x70
MPU_ADDR = 0x68
CHANNELS = [0, 2, 3]
EXPECTED_WHO_AM_I = 0x68
DEFAULT_SAMPLES = 200
SAMPLE_DELAY_S = 0.01


def select_channel(bus, ch):
    try:
        bus.write_byte(MUX_ADDR, 1 << ch)
        time.sleep(0.01)  # viktig delay
        return True
    except Exception as e:
        print(f"[MUX ERROR] channel {ch}:", e)
        return False


def test_channel_once(bus, ch):
    if not select_channel(bus, ch):
        return False

    try:
        who_am_i = bus.read_byte_data(MPU_ADDR, 0x75)
        return who_am_i == EXPECTED_WHO_AM_I
    except Exception as e:
        print(f"[FAIL] CH{ch}:", e)
        return False


def test_channel_samples(bus, ch, samples):
    failures = 0
    for _ in range(samples):
        if not test_channel_once(bus, ch):
            failures += 1
        time.sleep(SAMPLE_DELAY_S)
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="I2C multiplexer/channel stability check for MPU WHO_AM_I reads."
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of samples to test per channel (default: {DEFAULT_SAMPLES})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = max(1, args.samples)

    print("Starting I2C debug...\n")
    print(f"Samples per channel: {samples}\n")

    bus = SMBus(1)

    # Test multiplexer exists
    try:
        bus.write_byte(MUX_ADDR, 0x00)
        print("[OK] Multiplexer responding at 0x70\n")
    except Exception as e:
        print("[FATAL] Multiplexer not responding:", e)
        return

    # Test each channel over X samples and summarize result
    for ch in CHANNELS:
        failures = test_channel_samples(bus, ch, samples)
        fail_pct = (failures / samples) * 100.0
        status = "OK" if failures == 0 else "NOT OK"
        print(
            f"[{status}] CH{ch}: {samples - failures}/{samples} passed, "
            f"{fail_pct:.2f}% fails"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
