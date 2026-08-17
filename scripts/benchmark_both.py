from pathlib import Path
import os
import sys
import time

# ============================================================
# ARGUMENTS
# Examples:
#   python3 benchmark.py -gpu -batch1
#   python3 benchmark.py -gpu -batch64
#   python3 benchmark.py -cpu -batch1
# ============================================================

DEVICE = None
BATCH_SIZE = None

for arg in sys.argv[1:]:
    if arg == "-gpu":
        DEVICE = "gpu"
    elif arg == "-cpu":
        DEVICE = "cpu"
    elif arg.startswith("-batch"):
        BATCH_SIZE = int(arg.replace("-batch", ""))

if DEVICE not in ("gpu", "cpu") or BATCH_SIZE is None:
    print("Usage:")
    print("  python3 benchmark.py -gpu -batch1")
    print("  python3 benchmark.py -gpu -batch64")
    print("  python3 benchmark.py -cpu -batch1")
    sys.exit(1)

if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "modelo" / "TrafficSignNet_FP32.h5"
TEST_DIR = PROJECT_DIR / "data" / "gtsrb" / "split" / "test"

IMAGE_SIZE = (32, 32)
WARMUP = 100

# ============================================================
# DEVICE
# ============================================================

if DEVICE == "gpu":
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected.")
        sys.exit(1)
    DEVICE_NAME = "/GPU:0"
else:
    DEVICE_NAME = "/CPU:0"

print(f"\nDevice     : {DEVICE.upper()}")
print(f"Batch size : {BATCH_SIZE}")

# ============================================================
# LOAD MODEL
# ============================================================

model = keras.models.load_model(MODEL_PATH)

# ============================================================
# DATASET FOR MODEL-ONLY
# Decoding, resize and normalization happen OUTSIDE timing.
# ============================================================

model_only_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

model_only_ds = model_only_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

# ============================================================
# WARMUP
# ============================================================

warmup_images, _ = next(iter(model_only_ds))

print(f"Warmup     : {WARMUP}")
print("Test set   : full official GTSRB test set")

with tf.device(DEVICE_NAME):
    for _ in range(WARMUP):
        out = model(warmup_images, training=False)
        out.numpy()

# ============================================================
# 1) MODEL-ONLY BENCHMARK
# ============================================================

model_times = []
model_total_images = 0
model_correct = 0

with tf.device(DEVICE_NAME):
    for images, labels in model_only_ds:
        batch_images = int(images.shape[0])

        start = time.perf_counter()

        output = model(images, training=False)
        output_np = output.numpy()

        end = time.perf_counter()

        model_times.append(end - start)
        model_total_images += batch_images

        predictions = np.argmax(output_np, axis=1)
        model_correct += np.sum(predictions == labels.numpy())

model_times = np.array(model_times)

model_total_time = model_times.sum()
model_throughput = model_total_images / model_total_time
model_mean_batch_ms = model_times.mean() * 1000
model_median_batch_ms = np.median(model_times) * 1000
model_min_batch_ms = model_times.min() * 1000
model_max_batch_ms = model_times.max() * 1000
model_p90 = np.percentile(model_times, 90) * 1000
model_p95 = np.percentile(model_times, 95) * 1000
model_p99 = np.percentile(model_times, 99) * 1000
model_mean_image_ms = (model_total_time / model_total_images) * 1000
model_accuracy = model_correct / model_total_images

# ============================================================
# 2) END-TO-END BENCHMARK
#
# Includes:
#   disk read
#   PNG decode
#   resize
#   float32 conversion
#   normalization
#   batching
#   inference
#   output synchronization
# ============================================================

raw_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

e2e_times = []
e2e_total_images = 0
e2e_correct = 0

iterator = iter(raw_ds)

while True:
    start = time.perf_counter()

    try:
        images, labels = next(iterator)
    except StopIteration:
        break

    images = tf.cast(images, tf.float32) / 255.0

    with tf.device(DEVICE_NAME):
        output = model(images, training=False)
        output_np = output.numpy()

    end = time.perf_counter()

    batch_images = int(images.shape[0])

    e2e_times.append(end - start)
    e2e_total_images += batch_images

    predictions = np.argmax(output_np, axis=1)
    e2e_correct += np.sum(predictions == labels.numpy())

e2e_times = np.array(e2e_times)

e2e_total_time = e2e_times.sum()
e2e_throughput = e2e_total_images / e2e_total_time
e2e_mean_batch_ms = e2e_times.mean() * 1000
e2e_median_batch_ms = np.median(e2e_times) * 1000
e2e_min_batch_ms = e2e_times.min() * 1000
e2e_max_batch_ms = e2e_times.max() * 1000
e2e_p90 = np.percentile(e2e_times, 90) * 1000
e2e_p95 = np.percentile(e2e_times, 95) * 1000
e2e_p99 = np.percentile(e2e_times, 99) * 1000
e2e_mean_image_ms = (e2e_total_time / e2e_total_images) * 1000
e2e_accuracy = e2e_correct / e2e_total_images

# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("MODEL-ONLY")
print("========================================")
print(f"Device               : {DEVICE.upper()}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Total batches        : {len(model_times)}")
print(f"Total images         : {model_total_images}")
print(f"Accuracy             : {model_accuracy * 100:.2f}%")
print(f"Throughput           : {model_throughput:.2f} images/s")
print(f"Mean latency/batch   : {model_mean_batch_ms:.4f} ms")
print(f"Median latency/batch : {model_median_batch_ms:.4f} ms")
print(f"Min latency/batch    : {model_min_batch_ms:.4f} ms")
print(f"Max latency/batch    : {model_max_batch_ms:.4f} ms")
print(f"P90 latency/batch    : {model_p90:.4f} ms")
print(f"P95 latency/batch    : {model_p95:.4f} ms")
print(f"P99 latency/batch    : {model_p99:.4f} ms")
print(f"Mean latency/image   : {model_mean_image_ms:.6f} ms")

print("\n========================================")
print("END-TO-END")
print("========================================")
print(f"Device               : {DEVICE.upper()}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Total batches        : {len(e2e_times)}")
print(f"Total images         : {e2e_total_images}")
print(f"Accuracy             : {e2e_accuracy * 100:.2f}%")
print(f"Throughput           : {e2e_throughput:.2f} images/s")
print(f"Mean latency/batch   : {e2e_mean_batch_ms:.4f} ms")
print(f"Median latency/batch : {e2e_median_batch_ms:.4f} ms")
print(f"Min latency/batch    : {e2e_min_batch_ms:.4f} ms")
print(f"Max latency/batch    : {e2e_max_batch_ms:.4f} ms")
print(f"P90 latency/batch    : {e2e_p90:.4f} ms")
print(f"P95 latency/batch    : {e2e_p95:.4f} ms")
print(f"P99 latency/batch    : {e2e_p99:.4f} ms")
print(f"Mean latency/image   : {e2e_mean_image_ms:.6f} ms")
print("========================================")
