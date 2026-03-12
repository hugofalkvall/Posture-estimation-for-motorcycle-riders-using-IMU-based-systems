import argparse
import csv
import time
from collections import deque
from statistics import median

import numpy as np

from multiplex import Multiplexer, read_mpu_on_channel


BUS = 1
MUX_ADDR = 0x70
MPU_ADDR = 0x68
MUX_CHANNELS = [0, 2, 7]
DT = 1 / 50
CALIB_SAMPLES = 100
FIXED_BETA = 0.1
RAD2DEG = 180.0 / np.pi


def require_ahrs():
    try:
        from ahrs.common.orientation import q2euler as _q2euler
        from ahrs.filters import Madgwick as _Madgwick
        return _Madgwick, _q2euler
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: ahrs. Install with: pip install ahrs"
        ) from e


def calibrate_gyro_bias(mux, dt):
    bias = {ch: np.zeros(3, dtype=float) for ch in MUX_CHANNELS}
    print("Calibrating gyro bias (keep sensors still)...")
    for ch in MUX_CHANNELS:
        acc_bias = np.zeros(3, dtype=float)
        count = 0
        for _ in range(CALIB_SAMPLES):
            result = read_mpu_on_channel(mux, ch, MPU_ADDR)
            if result is None:
                time.sleep(dt)
                continue
            _, gyro = result
            acc_bias += np.deg2rad(np.array([gyro["x"], gyro["y"], gyro["z"]], dtype=float))
            count += 1
            time.sleep(dt)
        if count > 0:
            bias[ch] = acc_bias / count
    return bias


def run_single_test(mux, beta, duration_s, gyro_bias):
    Madgwick, q2euler = require_ahrs()
    filters = {ch: Madgwick(sampletime=DT, beta=beta) for ch in MUX_CHANNELS}
    quats = {ch: np.array([1.0, 0.0, 0.0, 0.0]) for ch in MUX_CHANNELS}
    last_update = {ch: None for ch in MUX_CHANNELS}

    start_due = time.monotonic()
    phase_step = DT / max(1, len(MUX_CHANNELS))
    next_due = {ch: start_due + (i * phase_step) for i, ch in enumerate(MUX_CHANNELS)}

    sample_times = {ch: deque(maxlen=2000) for ch in MUX_CHANNELS}
    latency_ms = []

    t_end = time.monotonic() + duration_s

    while time.monotonic() < t_end:
        now = time.monotonic()
        due_channels = [ch for ch in MUX_CHANNELS if now >= next_due[ch]]

        if not due_channels:
            sleep_for = min(next_due.values()) - now
            time.sleep(max(0.0005, min(sleep_for, DT / 2)))
            continue

        for ch in due_channels:
            sample_start = time.perf_counter()
            result = read_mpu_on_channel(mux, ch, MPU_ADDR)
            process_end = time.monotonic()

            next_due[ch] += DT
            if process_end > next_due[ch]:
                missed = int((process_end - next_due[ch]) / DT) + 1
                next_due[ch] += missed * DT

            if result is None:
                continue

            accel, gyro = result
            accel_arr = np.array([accel["x"], accel["y"], accel["z"]], dtype=float)
            gyro_arr = np.array([gyro["x"], gyro["y"], gyro["z"]], dtype=float)
            gyro_arr = np.deg2rad(gyro_arr) - gyro_bias[ch]

            dt_ch = DT if last_update[ch] is None else max(1e-4, process_end - last_update[ch])
            last_update[ch] = process_end
            if hasattr(filters[ch], "Dt"):
                filters[ch].Dt = dt_ch

            quats[ch] = filters[ch].updateIMU(quats[ch], gyro_arr, accel_arr)
            _roll, _pitch, _yaw = q2euler(quats[ch])
            _ = (_roll * RAD2DEG, _pitch * RAD2DEG, _yaw * RAD2DEG)

            sample_times[ch].append(process_end)
            latency_ms.append((time.perf_counter() - sample_start) * 1000.0)

    hz_per_channel = {}
    for ch in MUX_CHANNELS:
        if len(sample_times[ch]) < 2:
            hz_per_channel[ch] = 0.0
            continue
        elapsed = sample_times[ch][-1] - sample_times[ch][0]
        hz_per_channel[ch] = (len(sample_times[ch]) - 1) / elapsed if elapsed > 0 else 0.0

    return {
        "samples": len(latency_ms),
        "lat_mean_ms": float(np.mean(latency_ms)) if latency_ms else float("nan"),
        "lat_median_ms": float(median(latency_ms)) if latency_ms else float("nan"),
        "lat_min_ms": float(np.min(latency_ms)) if latency_ms else float("nan"),
        "lat_max_ms": float(np.max(latency_ms)) if latency_ms else float("nan"),
        "hz_ch0": hz_per_channel.get(0, 0.0),
        "hz_ch2": hz_per_channel.get(2, 0.0),
        "hz_ch7": hz_per_channel.get(7, 0.0),
        "hz_avg": float(np.mean(list(hz_per_channel.values()))) if hz_per_channel else 0.0,
    }


def save_csv(path, rows):
    fields = [
        "run",
        "beta",
        "samples",
        "lat_mean_ms",
        "lat_median_ms",
        "lat_min_ms",
        "lat_max_ms",
        "hz_ch0",
        "hz_ch2",
        "hz_ch7",
        "hz_avg",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows):
    import matplotlib.pyplot as plt

    runs = [r["run"] for r in rows]
    lat_max = [r["lat_max_ms"] for r in rows]
    lat_mean = [r["lat_mean_ms"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax1 = axes[0]
    ax1.plot(runs, lat_max, marker="o", color="tab:red", label="Max latency")
    ax1.set_title("Max Latency vs Run Iteration")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(runs, lat_mean, marker="o", color="tab:blue", label="Mean latency")
    ax2.set_title("Mean Latency vs Run Iteration")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.savefig(path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description="Run repeated latency benchmarks at fixed beta and plot per-run stability.")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration per run in seconds")
    parser.add_argument("--runs", type=int, default=7, help="Number of repeated runs")
    parser.add_argument("--beta", type=float, default=FIXED_BETA, help="Fixed beta value for all runs")
    parser.add_argument("--csv", default="beta_fixed_run_iterations.csv", help="Output CSV path")
    parser.add_argument("--plot", default="beta_fixed_run_iterations.png", help="Output plot path")
    args = parser.parse_args()

    require_ahrs()

    mux = Multiplexer(bus_num=BUS, address=MUX_ADDR)
    gyro_bias = calibrate_gyro_bias(mux, DT)

    results = []
    for run_idx in range(1, args.runs + 1):
        print(f"Running iteration {run_idx}/{args.runs} with beta={args.beta:.3f} for {args.duration:.0f}s...")
        row = run_single_test(mux, args.beta, args.duration, gyro_bias)
        row["run"] = run_idx
        row["beta"] = args.beta
        results.append(row)
        print(
            f"run={run_idx} | beta={args.beta:.3f} | "
            f"mean={row['lat_mean_ms']:.2f} ms | min/max={row['lat_min_ms']:.2f}/{row['lat_max_ms']:.2f} ms | "
            f"hz_avg={row['hz_avg']:.1f}"
        )

    save_csv(args.csv, results)
    save_plot(args.plot, results)

    print(f"Saved CSV: {args.csv}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
