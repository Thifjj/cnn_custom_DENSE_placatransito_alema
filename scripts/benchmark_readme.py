import os
import time
import json
from pathlib import Path

# ============================================================
# CONFIG
# Change only DEVICE to benchmark CPU or GPU.
# ============================================================

DEVICE = "gpu"          # "gpu" or "cpu"

WARMUP = 1000
RUNS = 5
RUN_SECONDS = 60

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "modelo" / "TrafficSignNet_FP32.h5"
TEST_DIR = PROJECT_DIR / "data" / "gtsrb" / "split" / "test"
RESULTS_DIR = PROJECT_DIR / "results"

# CPU must be forced before TensorFlow is imported.
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# ============================================================
# LOAD MODEL
# ============================================================

model = keras.models.load_model(MODEL_PATH, compile=False)

if DEVICE == "gpu":
    device_name = "/GPU:0"
else:
    device_name = "/CPU:0"


# ============================================================
# PREPARE ONE INPUT
#
# Decode, resize and normalization happen ONCE and OUTSIDE
# the benchmark. Timed region = CNN + output .numpy().
# ============================================================

image_path = next(TEST_DIR.rglob("*.png"))

with Image.open(image_path) as image:
    image = image.convert("RGB")
    image = image.resize((32, 32), Image.Resampling.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0

image = np.expand_dims(image, axis=0)
input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)


# ============================================================
# WARMUP
# ============================================================

print("\n========================================")
print("TrafficSignNet single-stream benchmark")
print("========================================")
print(f"Device       : {DEVICE.upper()}")
print("Batch        : 1")
print(f"Warmup       : {WARMUP}")
print(f"Runs         : {RUNS}")
print(f"Seconds/run  : {RUN_SECONDS}")
print("Mode         : model-only")
print("========================================")

print("\nWarmup...")

with tf.device(device_name):
    for _ in range(WARMUP):
        output = model(input_tensor, training=False)
        output.numpy()


# ============================================================
# BENCHMARK
# ============================================================

all_runs = []

with tf.device(device_name):

    for run_id in range(1, RUNS + 1):

        latencies_ns = []
        count = 0

        run_start_ns = time.perf_counter_ns()
        run_deadline_ns = run_start_ns + RUN_SECONDS * 1_000_000_000

        while time.perf_counter_ns() < run_deadline_ns:

            start_ns = time.perf_counter_ns()

            output = model(input_tensor, training=False)

            # Synchronization:
            # forces CPU/GPU inference to finish.
            output.numpy()

            end_ns = time.perf_counter_ns()

            latencies_ns.append(end_ns - start_ns)
            count += 1

        run_end_ns = time.perf_counter_ns()

        elapsed_s = (run_end_ns - run_start_ns) / 1e9

        latencies_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6

        result = {
            "run": run_id,
            "inferences": count,
            "elapsed_s": elapsed_s,
            "fps": count / elapsed_s,
            "mean_ms": float(np.mean(latencies_ms)),
            "median_ms": float(np.median(latencies_ms)),
            "std_ms": float(np.std(latencies_ms)),
            "min_ms": float(np.min(latencies_ms)),
            "max_ms": float(np.max(latencies_ms)),
            "p90_ms": float(np.percentile(latencies_ms, 90)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "p999_ms": float(np.percentile(latencies_ms, 99.9)),
        }

        all_runs.append(result)

        print(
            f"Run {run_id}: "
            f"{result['fps']:.2f} FPS | "
            f"mean {result['mean_ms']:.4f} ms | "
            f"P99 {result['p99_ms']:.4f} ms"
        )


# ============================================================
# AGGREGATE
# README: average the metric calculated in each run.
# ============================================================

def average(name):
    return float(np.mean([run[name] for run in all_runs]))


fps_values = [run["fps"] for run in all_runs]

summary = {
    "device": DEVICE,
    "batch_size": 1,
    "warmup": WARMUP,
    "runs": RUNS,
    "seconds_per_run": RUN_SECONDS,
    "mode": "model-only",
    "fps_mean": average("fps"),
    "fps_median_runs": float(np.median(fps_values)),
    "fps_min": float(np.min(fps_values)),
    "fps_max": float(np.max(fps_values)),
    "latency_mean_ms": average("mean_ms"),
    "latency_median_ms": average("median_ms"),
    "latency_std_ms": average("std_ms"),
    "latency_min_ms": average("min_ms"),
    "latency_max_ms": average("max_ms"),
    "p90_ms": average("p90_ms"),
    "p95_ms": average("p95_ms"),
    "p99_ms": average("p99_ms"),
    "p999_ms": average("p999_ms"),
    "runs_data": all_runs,
}


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FINAL RESULT")
print("========================================")
print(f"FPS mean             : {summary['fps_mean']:.2f}")
print(f"FPS median runs      : {summary['fps_median_runs']:.2f}")
print(f"FPS min / max        : {summary['fps_min']:.2f} / {summary['fps_max']:.2f}")
print(f"Mean latency         : {summary['latency_mean_ms']:.4f} ms")
print(f"Median latency       : {summary['latency_median_ms']:.4f} ms")
print(f"P90                  : {summary['p90_ms']:.4f} ms")
print(f"P95                  : {summary['p95_ms']:.4f} ms")
print(f"P99                  : {summary['p99_ms']:.4f} ms")
print(f"P99.9                : {summary['p999_ms']:.4f} ms")
print("========================================")


# ============================================================
# SAVE JSON
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

result_path = RESULTS_DIR / f"benchmark_{DEVICE}_single_stream.json"

with open(result_path, "w") as file:
    json.dump(summary, file, indent=4)

print(f"\nSaved: {result_path}")
