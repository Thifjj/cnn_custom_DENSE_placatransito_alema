# cnn_custom_DENSE_placatransito_alema

TrafficSignNet / GTSRB project with dataset preparation, training, FP32 TensorFlow model, and reproducible CPU/GPU benchmarking before Vitis AI weight export/rebuild.

## Main flow

```text
scripts/dataset_fixed_classes.py
        -> data/gtsrb/split/{train,val,test}
        -> scripts/train.py
        -> modelo/TrafficSignNet_FP32.h5
        -> scripts/benchmark_suite.py
```

## Benchmark

The reference benchmark is:

```bash
python3 scripts/benchmark_suite.py --device cpu
python3 scripts/benchmark_suite.py --device gpu
```

It reports separate sections for:

- synchronous batch-1 model-only latency/throughput;
- synchronous batch-1 host-to-host latency/throughput;
- fixed workload of 100 different images already in RAM;
- fixed workload of 100 files end-to-end from disk to prediction;
- full official test-set end-to-end accuracy/throughput;
- maximum-throughput batch sweep;
- fresh-process cold start.

Each run generates a canonical JSON plus a human-readable Markdown report under `results/benchmark_suite/`. Every section records exactly what the timer includes and excludes.

See [`BENCHMARK.md`](BENCHMARK.md) for the complete methodology and interpretation rules.
