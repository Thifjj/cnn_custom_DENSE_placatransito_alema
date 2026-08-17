# TrafficSignNet benchmark methodology

This repository uses `scripts/benchmark_suite.py` as the reference benchmark for the original FP32 TensorFlow/Keras model **before** weights are exported/rebuilt for Vitis AI.

The benchmark deliberately separates latency, sustained throughput, real fixed workloads, maximum batched throughput, and startup cost. These numbers answer different questions and must not be mixed.

## Primary rule for CPU/GPU/FPGA/DPU comparisons

For a fair comparison against Vitis AI DPU or HLS4ML, use the two **batch-1 synchronous** sections:

1. `single_stream_model_only`
2. `single_stream_host_to_host`

Both run one request at a time and force completion with `.numpy()` before stopping the per-inference timer. Batch-sweep results are secondary maximum-throughput numbers and should not be compared directly with batch-1 FPGA latency.

## Benchmark sections

### `single_stream_model_only`

Starts with one real GTSRB image already decoded, resized to `32x32`, converted to `float32`, normalized to `[0,1]`, and already represented as a TensorFlow tensor.

Timed region:

```text
prepared float32 tensor
        -> tf.function model execution
        -> CPU/GPU execution
        -> output synchronization (.numpy())
```

Excluded: disk I/O, PNG decode, resize, normalization, argmax, model loading.

This is the primary measure of model/runtime latency and steady-state batch-1 throughput.

### `single_stream_host_to_host`

Starts from an RGB `uint8` NumPy image already decoded/resized in host RAM.

Timed region:

```text
NumPy uint8 image in RAM
        -> TensorFlow tensor conversion
        -> float32 cast + /255 normalization
        -> CPU/GPU model inference
        -> argmax
        -> synchronization
```

Excluded: disk I/O, image decode, resize, model loading.

This is the primary application-compute benchmark and is the closest CPU/GPU counterpart to the host-to-host benchmark used in the Vitis AI deployment.

### `fixed_workload_memory`

Selects `--workload-size` different official GTSRB test images (100 by default) with a deterministic seed, loads them into RAM **before** the timer, then processes all images sequentially at batch 1.

It answers literally:

> If 100 real images are already in memory, how many seconds until the 100th result is complete?

The JSON stores each closed-workload run separately, including total elapsed seconds, FPS, accuracy, and per-image latency percentiles.

### `fixed_workload_end_to_end`

Processes the same closed workload, but starts each image from its file on disk.

Timed region:

```text
file open/read
 -> PNG decode
 -> RGB conversion
 -> resize 32x32
 -> NumPy/TensorFlow conversion
 -> normalization
 -> inference
 -> argmax
 -> synchronization
```

This is the most direct local-file answer to "I gave the application 100 images; when was the last result ready?"

The operating-system page cache is **not** forcibly cleared. Later repetitions can therefore benefit from cached file data; the generated JSON/Markdown explicitly records this caveat.

### `full_test_end_to_end`

Runs the same serial end-to-end path over the complete official GTSRB test split. It reports total images, correct predictions, accuracy, elapsed time, images/s, and latency distribution.

### `batch_throughput_sweep`

Uses preprocessed real images and evaluates several batch sizes (`1,8,32,64,128` by default). It measures model-only throughput and reports both batch latency and images/s.

This is a **maximum platform throughput** experiment. A larger batch can greatly increase GPU utilization, but the result is not a fair replacement for batch-1 latency.

### `cold_start`

Launches a fresh Python subprocess for each run. The wall-clock measurement includes process startup, TensorFlow import/runtime initialization, CUDA initialization when GPU is selected, model loading, one real image decode/resize, graph tracing + first inference, synchronization, and process exit.

Cold start is intentionally separate from steady-state inference.

## Output format

Each execution writes two files under `results/benchmark_suite/`:

```text
benchmark_cpu_YYYYMMDD_HHMMSS.json
benchmark_cpu_YYYYMMDD_HHMMSS.md
```

or the equivalent `benchmark_gpu_...` names.

The JSON is the canonical machine-readable result. Every benchmark section contains:

- `description`
- `purpose`
- `timed_region_includes`
- `timed_region_excludes`
- `caveats`
- `result`

The companion Markdown is generated automatically from the same JSON data and is intended for quick reading/reporting.

## Recommended full runs

CPU:

```bash
python3 scripts/benchmark_suite.py --device cpu
```

GPU:

```bash
python3 scripts/benchmark_suite.py --device gpu
```

Default workload:

```text
batch-1 warmup:       1000 inferences
steady-state runs:    5
seconds per run:      60
fixed workload:       100 different images
fixed-workload runs:  10
batch sweep:          1,8,32,64,128
batch sweep time:     10 s per batch
cold-start runs:      3
```

## Quick smoke test

Use this only to verify the environment; do not use these numbers in the final comparison.

```bash
python3 scripts/benchmark_suite.py \
  --device gpu \
  --warmup 10 \
  --runs 1 \
  --seconds 2 \
  --workload-size 20 \
  --workload-runs 1 \
  --batch-sizes 1,8 \
  --batch-warmup 2 \
  --batch-seconds 1 \
  --cold-runs 1 \
  --skip-full-test
```

## Reproducibility notes

- The model is `modelo/TrafficSignNet_FP32.h5`.
- Model input is RGB `32x32x3` normalized by `/255.0`, matching `scripts/train.py`.
- Prediction is `argmax` over 43 logits.
- Real workload images come from `data/gtsrb/split/test`.
- Fixed workloads are selected deterministically with seed `42` unless overridden.
- CPU mode hides CUDA before TensorFlow import.
- GPU mode refuses to run when TensorFlow cannot see a GPU and verifies that the model output is placed on GPU, avoiding silent CPU fallback.
- `time.perf_counter_ns()` is used for latency timing.
- `.numpy()` forces TensorFlow device work to complete before a timed inference ends.

## Interpreting throughput correctly

If a fixed workload of 100 images takes `0.080 s`, then its measured throughput is `100 / 0.080 = 1250 images/s`. That is a directly observed closed-workload result, not `1000 / average_latency` inferred indirectly.

For the final CPU/GPU/Vitis-AI/HLS4ML comparison, preserve the same definitions and batch size in all platforms. Use batched results only in a separate maximum-throughput comparison.
