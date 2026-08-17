#!/usr/bin/env python3
"""Comprehensive CPU/GPU benchmark suite for TrafficSignNet.

Primary comparison tests use synchronous batch-1 execution. Secondary tests
measure fixed real workloads, full end-to-end dataset processing, batch-sweep
maximum throughput, and process cold-start cost.
"""
from __future__ import annotations

SCRIPT_START_NS = __import__("time").perf_counter_ns()

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Device selection must happen before TensorFlow is imported.
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
_early_args, _ = _early.parse_known_args()
if _early_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_DIR / "modelo" / "TrafficSignNet_FP32.h5"
DEFAULT_TEST_DIR = PROJECT_DIR / "data" / "gtsrb" / "split" / "test"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "benchmark_suite"
IMAGE_SUFFIXES = {".png", ".ppm", ".jpg", ".jpeg", ".bmp"}
INPUT_SHAPE = (1, 32, 32, 3)
COLD_SENTINEL = "__TRAFFICSIGNNET_COLD_JSON__="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--workload-size", type=int, default=100)
    parser.add_argument("--workload-runs", type=int, default=10)
    parser.add_argument("--batch-sizes", default="1,8,32,64,128")
    parser.add_argument("--batch-warmup", type=int, default=50)
    parser.add_argument("--batch-seconds", type=float, default=10.0)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-full-test", action="store_true")
    parser.add_argument("--skip-batch-sweep", action="store_true")
    parser.add_argument("--skip-cold-start", action="store_true")

    parser.add_argument("--_cold-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_cold-image", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1 or args.seconds <= 0:
        parser.error("use --warmup >= 0, --runs >= 1 and --seconds > 0")
    if args.workload_size < 1 or args.workload_runs < 1:
        parser.error("workload size/runs must be >= 1")
    if args.batch_warmup < 0 or args.batch_seconds <= 0:
        parser.error("batch warmup must be >= 0 and batch seconds > 0")
    if args.cold_runs < 1:
        parser.error("--cold-runs must be >= 1")
    return args


def percentile_summary(values_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(values_ms.mean()),
        "median_ms": float(np.median(values_ms)),
        "std_ms": float(values_ms.std()),
        "min_ms": float(values_ms.min()),
        "max_ms": float(values_ms.max()),
        "p90_ms": float(np.percentile(values_ms, 90)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "p99_9_ms": float(np.percentile(values_ms, 99.9)),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fps = np.asarray([run["fps"] for run in runs], dtype=np.float64)
    result: dict[str, Any] = {
        "fps_mean": float(fps.mean()),
        "fps_median": float(np.median(fps)),
        "fps_min": float(fps.min()),
        "fps_max": float(fps.max()),
        "runs_data": runs,
    }
    if runs and "latency_ms" in runs[0]:
        for key in (
            "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
            "p90_ms", "p95_ms", "p99_ms", "p99_9_ms",
        ):
            result["latency_" + key] = float(
                np.mean([run["latency_ms"][key] for run in runs])
            )
    return result


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def read_text_if_exists(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata(device: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": getattr(keras, "__version__", "bundled-with-tensorflow"),
        "numpy": np.__version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": cpu_model_name(),
        "cpu_governor": read_text_if_exists(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "requested_device": device,
        "tensorflow_cuda_build": bool(tf.test.is_built_with_cuda()),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], PROJECT_DIR),
    }
    if device == "gpu":
        meta["nvidia_smi"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,power.limit",
                "--format=csv,noheader,nounits",
            ]
        )
        meta["tensorflow_gpus"] = [
            tf.config.experimental.get_device_details(gpu)
            for gpu in tf.config.list_physical_devices("GPU")
        ]
    return meta


def configure_device(device: str) -> str:
    if device == "cpu":
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError(
                "CPU mode still sees a GPU. CUDA_VISIBLE_DEVICES must be set before TensorFlow import."
            )
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU mode requested, but TensorFlow detected no GPU.")
    selected = gpus[0]
    try:
        tf.config.set_visible_devices(selected, "GPU")
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.set_soft_device_placement(False)
    except RuntimeError as exc:
        raise RuntimeError("GPU was initialized before benchmark configuration") from exc
    return "/GPU:0"


def collect_samples(test_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = int(class_dir.name)
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, label))
    if not samples:
        raise RuntimeError(f"No labeled images found under {test_dir}")
    return samples


def choose_workload(
    samples: list[tuple[Path, int]], size: int, seed: int
) -> list[tuple[Path, int]]:
    if size > len(samples):
        raise ValueError(f"workload-size {size} exceeds test-set size {len(samples)}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))[:size]
    return [samples[int(index)] for index in indices]


def load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((32, 32), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def load_model(model_path: Path, device_name: str) -> keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with tf.device(device_name):
        return keras.models.load_model(model_path, compile=False)


def make_inference_functions(
    model: keras.Model, device_name: str
) -> tuple[Callable[[tf.Tensor], tf.Tensor], Callable[[tf.Tensor], tf.Tensor]]:
    @tf.function(
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)],
        reduce_retracing=True,
    )
    def model_only(value: tf.Tensor) -> tf.Tensor:
        with tf.device(device_name):
            return model(value, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.uint8)],
        reduce_retracing=True,
    )
    def host_to_host(value: tf.Tensor) -> tf.Tensor:
        with tf.device(device_name):
            normalized = tf.cast(value, tf.float32) / 255.0
            logits = model(normalized, training=False)
            return tf.argmax(logits, axis=1, output_type=tf.int32)

    return model_only, host_to_host


def timed_repeated(
    call: Callable[[], Any], warmup: int, run_count: int, seconds: float
) -> dict[str, Any]:
    for _ in range(warmup):
        call()

    runs: list[dict[str, Any]] = []
    for run_id in range(1, run_count + 1):
        latencies_ns: list[int] = []
        run_start = time.perf_counter_ns()
        deadline = run_start + int(seconds * 1e9)
        while time.perf_counter_ns() < deadline:
            before = time.perf_counter_ns()
            call()
            latencies_ns.append(time.perf_counter_ns() - before)
        elapsed = (time.perf_counter_ns() - run_start) / 1e9
        latencies_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
        run = {
            "run": run_id,
            "inferences": len(latencies_ns),
            "elapsed_seconds": elapsed,
            "fps": float(len(latencies_ns) / elapsed),
            "latency_ms": percentile_summary(latencies_ms),
        }
        runs.append(run)
        print(
            f"  run {run_id}: {run['fps']:.2f} FPS | "
            f"mean {run['latency_ms']['mean_ms']:.4f} ms | "
            f"P99 {run['latency_ms']['p99_ms']:.4f} ms"
        )
    return aggregate_runs(runs)


def fixed_workload_memory(
    host_images: np.ndarray,
    labels: np.ndarray,
    host_to_host: Callable[[tf.Tensor], tf.Tensor],
    run_count: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_id in range(1, run_count + 1):
        correct = 0
        per_image_ns: list[int] = []
        started = time.perf_counter_ns()
        for image, expected in zip(host_images, labels):
            before = time.perf_counter_ns()
            value = tf.convert_to_tensor(image[None, ...], dtype=tf.uint8)
            predicted = int(host_to_host(value).numpy()[0])
            per_image_ns.append(time.perf_counter_ns() - before)
            correct += int(predicted == int(expected))
        elapsed = (time.perf_counter_ns() - started) / 1e9
        latency_ms = np.asarray(per_image_ns, dtype=np.float64) / 1e6
        runs.append(
            {
                "run": run_id,
                "images": int(len(host_images)),
                "correct": correct,
                "accuracy": float(correct / len(host_images)),
                "elapsed_seconds": elapsed,
                "fps": float(len(host_images) / elapsed),
                "latency_ms": percentile_summary(latency_ms),
            }
        )
    return aggregate_runs(runs)


def fixed_workload_end_to_end(
    workload: list[tuple[Path, int]],
    host_to_host: Callable[[tf.Tensor], tf.Tensor],
    run_count: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_id in range(1, run_count + 1):
        correct = 0
        per_image_ns: list[int] = []
        started = time.perf_counter_ns()
        for path, expected in workload:
            before = time.perf_counter_ns()
            image = load_rgb_uint8(path)
            value = tf.convert_to_tensor(image[None, ...], dtype=tf.uint8)
            predicted = int(host_to_host(value).numpy()[0])
            per_image_ns.append(time.perf_counter_ns() - before)
            correct += int(predicted == expected)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        latency_ms = np.asarray(per_image_ns, dtype=np.float64) / 1e6
        runs.append(
            {
                "run": run_id,
                "images": len(workload),
                "correct": correct,
                "accuracy": float(correct / len(workload)),
                "elapsed_seconds": elapsed,
                "fps": float(len(workload) / elapsed),
                "latency_ms": percentile_summary(latency_ms),
            }
        )
    return aggregate_runs(runs)


def full_test_end_to_end(
    samples: list[tuple[Path, int]], host_to_host: Callable[[tf.Tensor], tf.Tensor]
) -> dict[str, Any]:
    correct = 0
    latencies_ns: list[int] = []
    started = time.perf_counter_ns()
    for index, (path, expected) in enumerate(samples, 1):
        before = time.perf_counter_ns()
        image = load_rgb_uint8(path)
        value = tf.convert_to_tensor(image[None, ...], dtype=tf.uint8)
        predicted = int(host_to_host(value).numpy()[0])
        latencies_ns.append(time.perf_counter_ns() - before)
        correct += int(predicted == expected)
        if index % 1000 == 0 or index == len(samples):
            print(f"  full-test progress: {index}/{len(samples)}")
    elapsed = (time.perf_counter_ns() - started) / 1e9
    latency_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
    return {
        "images": len(samples),
        "correct": correct,
        "accuracy": float(correct / len(samples)),
        "elapsed_seconds": elapsed,
        "fps": float(len(samples) / elapsed),
        "latency_ms": percentile_summary(latency_ms),
    }


def batch_sweep(
    model: keras.Model,
    device_name: str,
    float_images: np.ndarray,
    batch_sizes: list[int],
    warmup: int,
    seconds: float,
) -> dict[str, Any]:
    sweep: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size > len(float_images):
            raise ValueError(
                f"batch size {batch_size} requires at least {batch_size} preloaded images"
            )
        batch_tensor = tf.convert_to_tensor(
            float_images[:batch_size], dtype=tf.float32
        )

        @tf.function(
            input_signature=[
                tf.TensorSpec((batch_size, 32, 32, 3), tf.float32)
            ],
            reduce_retracing=True,
        )
        def infer_batch(value: tf.Tensor) -> tf.Tensor:
            with tf.device(device_name):
                return model(value, training=False)

        for _ in range(warmup):
            infer_batch(batch_tensor).numpy()

        latencies_ns: list[int] = []
        calls = 0
        started = time.perf_counter_ns()
        deadline = started + int(seconds * 1e9)
        while time.perf_counter_ns() < deadline:
            before = time.perf_counter_ns()
            infer_batch(batch_tensor).numpy()
            latencies_ns.append(time.perf_counter_ns() - before)
            calls += 1
        elapsed = (time.perf_counter_ns() - started) / 1e9
        latency_ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
        item = {
            "batch_size": batch_size,
            "calls": calls,
            "images": calls * batch_size,
            "elapsed_seconds": elapsed,
            "images_per_second": float(calls * batch_size / elapsed),
            "batches_per_second": float(calls / elapsed),
            "batch_latency_ms": percentile_summary(latency_ms),
            "effective_mean_ms_per_image": float(
                latency_ms.mean() / batch_size
            ),
        }
        sweep.append(item)
        print(
            f"  batch {batch_size}: {item['images_per_second']:.2f} images/s | "
            f"batch mean {item['batch_latency_ms']['mean_ms']:.4f} ms"
        )
    best = max(sweep, key=lambda item: item["images_per_second"])
    return {"best": best, "sweep": sweep}


def cold_child(args: argparse.Namespace) -> None:
    if args._cold_image is None:
        raise SystemExit("--_cold-image is required in cold-child mode")
    device_name = configure_device(args.device)

    load_started = time.perf_counter_ns()
    model = load_model(args.model.resolve(), device_name)
    model_load_ms = (time.perf_counter_ns() - load_started) / 1e6

    prepare_started = time.perf_counter_ns()
    image = load_rgb_uint8(args._cold_image.resolve())
    value = tf.convert_to_tensor(image[None, ...], dtype=tf.uint8)
    image_prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6

    _, host_to_host = make_inference_functions(model, device_name)
    infer_started = time.perf_counter_ns()
    prediction = int(host_to_host(value).numpy()[0])
    first_inference_ms = (time.perf_counter_ns() - infer_started) / 1e6

    payload = {
        "script_start_to_result_ms": (time.perf_counter_ns() - SCRIPT_START_NS) / 1e6,
        "model_load_ms": model_load_ms,
        "image_decode_resize_tensor_ms": image_prepare_ms,
        "first_compiled_inference_ms": first_inference_ms,
        "prediction": prediction,
    }
    print(COLD_SENTINEL + json.dumps(payload, separators=(",", ":")))


def cold_start_benchmark(
    args: argparse.Namespace, sample_path: Path
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    runs: list[dict[str, Any]] = []
    for run_id in range(1, args.cold_runs + 1):
        command = [
            sys.executable,
            str(script_path),
            "--device", args.device,
            "--model", str(args.model.resolve()),
            "--test-dir", str(args.test_dir.resolve()),
            "--_cold-child",
            "--_cold-image", str(sample_path.resolve()),
        ]
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        started = time.perf_counter_ns()
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        process_wall_ms = (time.perf_counter_ns() - started) / 1e6
        if proc.returncode != 0:
            raise RuntimeError(
                "cold-start child failed:\n" + proc.stderr[-4000:]
            )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(COLD_SENTINEL):
                payload = json.loads(line[len(COLD_SENTINEL):])
        if payload is None:
            raise RuntimeError("cold-start child returned no benchmark payload")
        payload["run"] = run_id
        payload["process_wall_ms"] = process_wall_ms
        runs.append(payload)
    return {
        "runs_data": runs,
        "process_wall_mean_ms": float(statistics.mean(r["process_wall_ms"] for r in runs)),
        "script_to_result_mean_ms": float(
            statistics.mean(r["script_start_to_result_ms"] for r in runs)
        ),
        "model_load_mean_ms": float(statistics.mean(r["model_load_ms"] for r in runs)),
        "first_compiled_inference_mean_ms": float(
            statistics.mean(r["first_compiled_inference_ms"] for r in runs)
        ),
    }


def section(
    description: str,
    included: list[str],
    excluded: list[str],
    purpose: str,
    result: dict[str, Any],
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "description": description,
        "purpose": purpose,
        "timed_region_includes": included,
        "timed_region_excludes": excluded,
        "caveats": caveats or [],
        "result": result,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    def metric(section_name: str, key: str, default: str = "-") -> str:
        try:
            value = report["benchmarks"][section_name]["result"][key]
        except KeyError:
            return default
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "# TrafficSignNet benchmark report",
        "",
        f"- Device: **{report['environment']['requested_device'].upper()}**",
        f"- Generated: `{report['generated_at']}`",
        f"- TensorFlow: `{report['environment']['tensorflow']}`",
        f"- Model: `{report['model']['path']}`",
        f"- Parameters: `{report['model']['parameters']}`",
        "",
        "## Primary batch-1 comparison",
        "",
        "These are the sections to compare directly against other batch-1 synchronous platforms such as a DPU or HLS4ML implementation.",
        "",
        "| Benchmark | FPS mean | Mean latency (ms) | P99 (ms) |",
        "|---|---:|---:|---:|",
        f"| Model-only | {metric('single_stream_model_only','fps_mean')} | {metric('single_stream_model_only','latency_mean_ms')} | {metric('single_stream_model_only','latency_p99_ms')} |",
        f"| Host-to-host | {metric('single_stream_host_to_host','fps_mean')} | {metric('single_stream_host_to_host','latency_mean_ms')} | {metric('single_stream_host_to_host','latency_p99_ms')} |",
        "",
    ]

    for name, bench in report["benchmarks"].items():
        lines.extend([
            f"## {name}",
            "",
            bench["description"],
            "",
            f"**Purpose:** {bench['purpose']}",
            "",
            "**Timed region includes:** " + "; ".join(bench["timed_region_includes"]),
            "",
            "**Timed region excludes:** " + "; ".join(bench["timed_region_excludes"]),
            "",
        ])
        if bench.get("caveats"):
            lines.append("**Caveats:** " + "; ".join(bench["caveats"]))
            lines.append("")
        result = bench["result"]
        if "fps_mean" in result:
            lines.append(f"- FPS mean: **{result['fps_mean']:.2f}**")
        if "latency_mean_ms" in result:
            lines.append(f"- Mean latency: **{result['latency_mean_ms']:.4f} ms**")
        if "elapsed_seconds" in result:
            lines.append(f"- Total time: **{result['elapsed_seconds']:.6f} s**")
        if "accuracy" in result:
            lines.append(f"- Accuracy: **{result['accuracy'] * 100:.2f}%**")
        if name.startswith("fixed_workload") and result.get("runs_data"):
            elapsed = [r["elapsed_seconds"] for r in result["runs_data"]]
            lines.append(
                f"- Closed workload mean total time: **{statistics.mean(elapsed):.6f} s**"
            )
        if name == "batch_throughput_sweep":
            best = result["best"]
            lines.append(
                f"- Best batch: **{best['batch_size']}**, **{best['images_per_second']:.2f} images/s**"
            )
        if name == "cold_start":
            lines.append(
                f"- Fresh-process wall time mean: **{result['process_wall_mean_ms']:.2f} ms**"
            )
        lines.append("")

    lines.extend([
        "## Interpretation rules",
        "",
        "- Do not compare batch-sweep maximum throughput directly with batch-1 FPGA/DPU latency numbers.",
        "- `.numpy()` is used at the end of timed TensorFlow inference calls to force completion/synchronization.",
        "- Fixed-workload tests process different real GTSRB images, not one repeated synthetic tensor.",
        "- End-to-end file tests can benefit from the operating-system page cache after the first pass; this is recorded as a caveat rather than hidden.",
        "- Cold-start is intentionally separate from steady-state inference because model loading, TensorFlow startup, and CUDA initialization answer a different question.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args._cold_child:
        cold_child(args)
        return

    model_path = args.model.resolve()
    test_dir = args.test_dir.resolve()
    results_dir = args.results_dir.resolve()
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    if not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("--batch-sizes must contain positive integers")

    samples = collect_samples(test_dir)
    workload = choose_workload(samples, args.workload_size, args.seed)
    preload_count = max(args.workload_size, max(batch_sizes))
    preload_samples = choose_workload(samples, preload_count, args.seed)

    cold_result = None
    if not args.skip_cold_start:
        print("\n[COLD START]")
        cold_result = cold_start_benchmark(args, workload[0][0])

    device_name = configure_device(args.device)
    print(f"\nDevice: {args.device.upper()} ({device_name})")
    model = load_model(model_path, device_name)
    model_only, host_to_host = make_inference_functions(model, device_name)

    preloaded_uint8 = np.stack([load_rgb_uint8(path) for path, _ in preload_samples])
    preloaded_float = preloaded_uint8.astype(np.float32) / 255.0
    workload_uint8 = preloaded_uint8[: args.workload_size]
    workload_labels = np.asarray([label for _, label in preload_samples[: args.workload_size]])

    model_input = tf.convert_to_tensor(preloaded_float[0:1], dtype=tf.float32)
    host_input_np = preloaded_uint8[0]

    probe = model_only(model_input)
    expected_marker = "GPU" if args.device == "gpu" else "CPU"
    if expected_marker not in probe.device.upper():
        raise RuntimeError(
            f"Model output is on {probe.device}, expected {expected_marker}. "
            "Benchmark aborted instead of silently measuring another device."
        )
    probe.numpy()

    benchmarks: dict[str, Any] = {}

    print("\n[SINGLE STREAM MODEL-ONLY]")
    model_only_result = timed_repeated(
        lambda: model_only(model_input).numpy(),
        args.warmup,
        args.runs,
        args.seconds,
    )
    benchmarks["single_stream_model_only"] = section(
        description=(
            "Synchronous batch-1 steady-state inference with an already prepared "
            "float32 [0,1] tensor. The model call is compiled with tf.function."
        ),
        purpose="Primary accelerator/model latency and sustained-throughput comparison.",
        included=["tf.function model execution", "device execution", "output synchronization via .numpy()"],
        excluded=["disk I/O", "image decode", "resize", "uint8-to-float conversion", "normalization", "argmax", "model loading"],
        result=model_only_result,
    )

    print("\n[SINGLE STREAM HOST-TO-HOST]")
    host_result = timed_repeated(
        lambda: host_to_host(
            tf.convert_to_tensor(host_input_np[None, ...], dtype=tf.uint8)
        ).numpy(),
        args.warmup,
        args.runs,
        args.seconds,
    )
    benchmarks["single_stream_host_to_host"] = section(
        description=(
            "Synchronous batch-1 steady-state path starting from an in-memory NumPy "
            "RGB uint8 image. It converts the host image to a TensorFlow tensor, "
            "normalizes /255, runs the model, performs argmax, and synchronizes."
        ),
        purpose="Primary application-compute comparison without storage/decode effects.",
        included=["NumPy uint8 to TensorFlow tensor", "normalization /255", "device transfer/placement as required", "tf.function model execution", "argmax", "output synchronization"],
        excluded=["disk I/O", "image decode", "resize", "model loading"],
        result=host_result,
    )

    print(f"\n[FIXED WORKLOAD MEMORY: {args.workload_size} IMAGES]")
    fixed_mem = fixed_workload_memory(
        workload_uint8, workload_labels, host_to_host, args.workload_runs
    )
    benchmarks["fixed_workload_memory"] = section(
        description=(
            f"Closed workload of {args.workload_size} different real GTSRB images "
            "already decoded/resized in RAM, processed sequentially at batch 1."
        ),
        purpose=(
            f"Answer literally: how long does the running application take to finish "
            f"{args.workload_size} real in-memory images?"
        ),
        included=["all images in the closed workload", "NumPy uint8 to TensorFlow conversion", "normalization", "device execution", "argmax", "synchronization"],
        excluded=["disk I/O", "PNG decode", "resize", "model loading"],
        result=fixed_mem,
    )

    print(f"\n[FIXED WORKLOAD END-TO-END: {args.workload_size} IMAGES]")
    fixed_e2e = fixed_workload_end_to_end(workload, host_to_host, args.workload_runs)
    benchmarks["fixed_workload_end_to_end"] = section(
        description=(
            f"Closed workload of {args.workload_size} different real GTSRB files. "
            "Each timed image is opened from storage, decoded, converted to RGB, "
            "resized to 32x32, normalized, inferred, argmaxed, and synchronized."
        ),
        purpose=(
            f"Closest local-file test to 'I gave the application {args.workload_size} "
            "images; when was the last result ready?'"
        ),
        included=["file open/read", "image decode", "RGB conversion", "resize 32x32", "NumPy/TensorFlow conversion", "normalization", "device inference", "argmax", "synchronization"],
        excluded=["model loading", "application process startup"],
        result=fixed_e2e,
        caveats=["Later runs may benefit from the operating-system page cache; caches are not forcibly dropped."],
    )

    if not args.skip_full_test:
        print("\n[FULL OFFICIAL TEST SET END-TO-END]")
        full_result = full_test_end_to_end(samples, host_to_host)
        benchmarks["full_test_end_to_end"] = section(
            description=(
                "One serial pass over every labeled image in data/gtsrb/split/test, "
                "including file decode/resize and batch-1 inference."
            ),
            purpose="Report real test-set accuracy and serial end-to-end throughput using the same inference path.",
            included=["entire test set", "file I/O", "decode/resize", "normalization", "batch-1 model inference", "argmax", "synchronization", "accuracy counting"],
            excluded=["model loading", "process startup"],
            result=full_result,
            caveats=["Filesystem cache state is not controlled."],
        )

    if not args.skip_batch_sweep:
        print("\n[MAX THROUGHPUT BATCH SWEEP]")
        batch_result = batch_sweep(
            model,
            device_name,
            preloaded_float,
            batch_sizes,
            args.batch_warmup,
            args.batch_seconds,
        )
        benchmarks["batch_throughput_sweep"] = section(
            description=(
                "Secondary model-only throughput test using multiple batch sizes and "
                "preprocessed real images. This intentionally trades per-request latency "
                "for hardware utilization."
            ),
            purpose="Find maximum platform throughput; not the primary batch-1 fairness metric.",
            included=["preprocessed batch", "tf.function model execution", "device synchronization"],
            excluded=["file I/O", "decode/resize", "normalization", "argmax", "model loading"],
            result=batch_result,
            caveats=["Do not compare batch>1 throughput directly with a batch-1 FPGA/DPU latency result."],
        )

    if cold_result is not None:
        benchmarks["cold_start"] = section(
            description=(
                "Fresh Python subprocess measurement. Wall time includes process startup, "
                "TensorFlow import/runtime initialization, model load, one real image "
                "decode/resize, graph tracing/first inference, synchronization, and exit."
            ),
            purpose="Quantify startup responsiveness separately from steady-state inference.",
            included=["fresh process startup", "TensorFlow import", "CPU/CUDA runtime initialization", "model load", "one image decode/resize", "first compiled inference", "synchronization", "process exit"],
            excluded=["steady-state warmup"],
            result=cold_result,
            caveats=["Cold-start is a different metric and must not be mixed with steady-state FPS."],
        )

    report = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_goal": (
            "Measure both fair synchronous batch-1 latency/throughput and real closed-workload "
            "completion time, while keeping maximum batched throughput and cold-start separate."
        ),
        "environment": environment_metadata(args.device),
        "model": {
            "path": str(model_path),
            "name": model.name,
            "parameters": int(model.count_params()),
            "input_shape": [1, 32, 32, 3],
            "input_semantics": "RGB, float32 normalized to [0,1] for model-only",
            "output_classes": 43,
            "output_semantics": "43 logits; prediction = argmax(logits)",
        },
        "configuration": {
            "batch_size_primary": 1,
            "warmup": args.warmup,
            "runs": args.runs,
            "seconds_per_run": args.seconds,
            "workload_size": args.workload_size,
            "workload_runs": args.workload_runs,
            "batch_sizes": batch_sizes,
            "batch_warmup": args.batch_warmup,
            "batch_seconds": args.batch_seconds,
            "cold_runs": args.cold_runs,
            "seed": args.seed,
            "test_images": len(samples),
        },
        "methodology": {
            "primary_comparison_rule": (
                "Use single_stream_model_only and single_stream_host_to_host for fair "
                "batch-1 synchronous comparisons across CPU, GPU, Vitis-AI DPU, and HLS4ML."
            ),
            "synchronization": (
                "Every timed TensorFlow inference is materialized with .numpy() so asynchronous "
                "device work completes before the timer stops."
            ),
            "throughput_definition": "completed images / measured wall-clock seconds",
            "latency_clock": "time.perf_counter_ns()",
            "fixed_workload_sampling": (
                "Different official GTSRB test images chosen once with a deterministic NumPy "
                f"permutation using seed {args.seed}."
            ),
        },
        "benchmarks": benchmarks,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"benchmark_{args.device}_{timestamp}"
    json_path = results_dir / f"{stem}.json"
    md_path = results_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print("\n========================================")
    print("FINAL PRIMARY RESULTS")
    print("========================================")
    print(
        "MODEL-ONLY   : {:.2f} FPS | {:.4f} ms".format(
            model_only_result["fps_mean"], model_only_result["latency_mean_ms"]
        )
    )
    print(
        "HOST-TO-HOST: {:.2f} FPS | {:.4f} ms".format(
            host_result["fps_mean"], host_result["latency_mean_ms"]
        )
    )
    fixed_times = [run["elapsed_seconds"] for run in fixed_e2e["runs_data"]]
    print(
        f"{args.workload_size} FILES E2E: "
        f"{statistics.mean(fixed_times):.6f} s mean total"
    )
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
