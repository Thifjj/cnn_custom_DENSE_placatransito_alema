import os
import time
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DEVICE = "gpu"          # "gpu" or "cpu"
WARMUP = 1000
RUNS = 5
RUN_SECONDS = 60

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "modelo" / "TrafficSignNet_FP32.h5"
TEST_DIR = PROJECT_DIR / "data" / "gtsrb" / "split" / "test"

if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


model = keras.models.load_model(MODEL_PATH, compile=False)

image_path = next(TEST_DIR.rglob("*.png"))

with Image.open(image_path) as image:
    image = image.convert("RGB")
    image = image.resize((32, 32), Image.Resampling.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0

x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)


# IMPORTANT:
# Compile the complete model call into one TensorFlow graph.
# This removes most Python/Keras eager overhead from every inference.
@tf.function(
    input_signature=[tf.TensorSpec(shape=(1, 32, 32, 3), dtype=tf.float32)],
    reduce_retracing=True,
)
def infer(input_tensor):
    return model(input_tensor, training=False)


print("========================================")
print("TrafficSignNet compiled single-stream")
print("========================================")
print(f"Device      : {DEVICE.upper()}")
print("Batch       : 1")
print("Execution   : tf.function graph")
print(f"Warmup      : {WARMUP}")
print(f"Runs        : {RUNS}")
print(f"Seconds/run : {RUN_SECONDS}")
print("========================================")

print("\nWarmup...")

for _ in range(WARMUP):
    infer(x).numpy()


run_results = []

for run in range(1, RUNS + 1):
    latencies = []
    count = 0

    run_start = time.perf_counter_ns()
    deadline = run_start + RUN_SECONDS * 1_000_000_000

    while time.perf_counter_ns() < deadline:
        start = time.perf_counter_ns()

        y = infer(x)
        y.numpy()  # synchronization

        end = time.perf_counter_ns()

        latencies.append((end - start) / 1e6)
        count += 1

    elapsed = (time.perf_counter_ns() - run_start) / 1e9
    latencies = np.asarray(latencies)

    result = {
        "fps": count / elapsed,
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "p90": np.percentile(latencies, 90),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "p999": np.percentile(latencies, 99.9),
    }

    run_results.append(result)

    print(
        f"Run {run}: "
        f"{result['fps']:.2f} FPS | "
        f"{result['mean']:.4f} ms | "
        f"P99 {result['p99']:.4f} ms"
    )


def mean(key):
    return float(np.mean([r[key] for r in run_results]))


print("\n========================================")
print("FINAL RESULT")
print("========================================")
print(f"FPS mean       : {mean('fps'):.2f}")
print(f"Latency mean   : {mean('mean'):.4f} ms")
print(f"Median         : {mean('median'):.4f} ms")
print(f"P90            : {mean('p90'):.4f} ms")
print(f"P95            : {mean('p95'):.4f} ms")
print(f"P99            : {mean('p99'):.4f} ms")
print(f"P99.9          : {mean('p999'):.4f} ms")
print("========================================")
