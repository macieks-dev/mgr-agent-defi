
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

DEFAULT_MODEL = Path("models/ppo_v4_final_final.zip")
DEFAULT_OUTPUT = Path("results/inference_benchmark.json")
DEFAULT_ITERATIONS = 1000
DEFAULT_WARMUP_ITERATIONS = 50
BLOCK_TIME_MS = 12_000
TX_SUBMISSION_BUDGET_MS = 6_000
NETWORK_LATENCY_ESTIMATE_MS = 200
FEATURE_COMPUTATION_ESTIMATE_MS = 1


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=positive_int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=positive_int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    return parser.parse_args()


def devices_to_benchmark(selection: str) -> list[str]:
    if selection == "cpu":
        return ["cpu"]
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        return ["cuda"]

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_device(
    model_path: Path,
    device: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, float]:
    model = PPO.load(str(model_path), device=device)
    observation_shape = model.observation_space.shape
    if not observation_shape:
        raise ValueError("The loaded model does not have a vector observation space.")
    observation = np.zeros(observation_shape, dtype=np.float32)

    for _ in range(warmup_iterations):
        model.predict(observation, deterministic=True)
    synchronize(device)

    elapsed_ms = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        synchronize(device)
        start_ns = time.perf_counter_ns()
        model.predict(observation, deterministic=True)
        synchronize(device)
        elapsed_ms[index] = (time.perf_counter_ns() - start_ns) / 1_000_000

    return {
        "mean_ms": float(elapsed_ms.mean()),
        "median_ms": float(np.median(elapsed_ms)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "p99_ms": float(np.percentile(elapsed_ms, 99)),
        "max_ms": float(elapsed_ms.max()),
        "std_ms": float(elapsed_ms.std()),
    }


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Choose another --output or pass --overwrite."
        )

    try:
        devices = devices_to_benchmark(args.device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results = {
        "model": str(model_path),
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "observation": "zero vector matching the model observation shape",
        "assumptions": {
            "block_time_ms": BLOCK_TIME_MS,
            "tx_submission_budget_ms": TX_SUBMISSION_BUDGET_MS,
            "network_latency_estimate_ms": NETWORK_LATENCY_ESTIMATE_MS,
            "feature_computation_estimate_ms": FEATURE_COMPUTATION_ESTIMATE_MS,
        },
        "devices": {},
    }

    for device in devices:
        metrics = benchmark_device(
            model_path,
            device,
            args.iterations,
            args.warmup,
        )
        estimated_pipeline_ms = (
            metrics["p99_ms"]
            + NETWORK_LATENCY_ESTIMATE_MS
            + FEATURE_COMPUTATION_ESTIMATE_MS
        )
        results["devices"][device] = {
            "hardware": device_name(device),
            **metrics,
            "estimated_pipeline_p99_ms": estimated_pipeline_ms,
            "fits_submission_budget": estimated_pipeline_ms < TX_SUBMISSION_BUDGET_MS,
        }

        print(f"{device.upper()} inference on {device_name(device)}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        print(f"  estimated pipeline P99: {estimated_pipeline_ms:.3f} ms")
        print(f"  fits submission budget: {estimated_pipeline_ms < TX_SUBMISSION_BUDGET_MS}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write("\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
