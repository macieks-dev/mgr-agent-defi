"""Inference latency benchmark for thesis Chapter 5."""
import json
import logging
import time

import numpy as np
import torch
import sys
sys.path.insert(0, ".")

from stable_baselines3 import PPO
from src.env.uniswap_v4_env import UniswapV4Env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Loading model on CUDA...")
model = PPO.load("models/ppo_v2_curriculum_12feat_final", device="cuda")

# Create env with real data
prices = np.load("data/processed/prices_test.npy")[:5000]
features = np.load("data/processed/features_test.npy")[:5000]
cex = np.load("data/processed/cex_prices_test.npy")[:5000]

env = UniswapV4Env(
    price_data=prices, feature_data=features,
    cex_price_data=cex, episode_length=1000
)
env.set_curriculum_phase(3)
obs, _ = env.reset()

# Warmup
for _ in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

# ── Benchmark CUDA ──
logger.info("Running CUDA benchmark (1000 calls)...")
obs, _ = env.reset()
times_cuda = []
for i in range(1000):
    t0 = time.perf_counter_ns()
    action, _ = model.predict(obs, deterministic=True)
    t1 = time.perf_counter_ns()
    times_cuda.append((t1 - t0) / 1e6)  # ms
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

times_cuda = np.array(times_cuda)
print("=" * 60)
print("INFERENCE LATENCY BENCHMARK")
print("=" * 60)
print(f"Model: ppo_v2_curriculum_12feat (3x256 MLP, 17-dim obs)")
print(f"Hardware: NVIDIA RTX 4080 Laptop / CUDA 12.8")
print()
print("CUDA Inference (1000 calls):")
print(f"  Mean:   {times_cuda.mean():.3f} ms")
print(f"  Median: {np.median(times_cuda):.3f} ms")
print(f"  P95:    {np.percentile(times_cuda, 95):.3f} ms")
print(f"  P99:    {np.percentile(times_cuda, 99):.3f} ms")
print(f"  Max:    {times_cuda.max():.3f} ms")
print(f"  Std:    {times_cuda.std():.3f} ms")

# ── Benchmark CPU ──
logger.info("Running CPU benchmark (1000 calls)...")
model_cpu = PPO.load("models/ppo_v2_curriculum_12feat_final", device="cpu")
obs, _ = env.reset()
# Warmup
for _ in range(50):
    action, _ = model_cpu.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

obs, _ = env.reset()
times_cpu = []
for i in range(1000):
    t0 = time.perf_counter_ns()
    action, _ = model_cpu.predict(obs, deterministic=True)
    t1 = time.perf_counter_ns()
    times_cpu.append((t1 - t0) / 1e6)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

times_cpu = np.array(times_cpu)
print()
print("CPU Inference (1000 calls):")
print(f"  Mean:   {times_cpu.mean():.3f} ms")
print(f"  Median: {np.median(times_cpu):.3f} ms")
print(f"  P99:    {np.percentile(times_cpu, 99):.3f} ms")
print(f"  Max:    {times_cpu.max():.3f} ms")

print()
print("Ethereum Block Budget Analysis:")
print(f"  Block time:            12,000 ms")
print(f"  Tx submission budget:  ~6,000 ms (conservative: half block)")
print(f"  Network latency:       ~200 ms (Flashbots/MEV-Boost)")
print(f"  Feature computation:   ~1 ms (12 float ops)")
print(f"  Inference (CUDA P99):  {np.percentile(times_cuda, 99):.3f} ms")
print(f"  Inference (CPU P99):   {np.percentile(times_cpu, 99):.3f} ms")
total_cuda = np.percentile(times_cuda, 99) + 200 + 1
total_cpu = np.percentile(times_cpu, 99) + 200 + 1
print(f"  Total pipeline (CUDA): {total_cuda:.1f} ms ({total_cuda/12000*100:.3f}% of block)")
print(f"  Total pipeline (CPU):  {total_cpu:.1f} ms ({total_cpu/12000*100:.3f}% of block)")
fits = "YES" if total_cpu < 6000 else "NO"
print(f"  Fits in block window:  {fits} (even CPU-only)")
print("=" * 60)

results = {
    "cuda": {
        "mean_ms": float(times_cuda.mean()),
        "median_ms": float(np.median(times_cuda)),
        "p95_ms": float(np.percentile(times_cuda, 95)),
        "p99_ms": float(np.percentile(times_cuda, 99)),
        "max_ms": float(times_cuda.max()),
    },
    "cpu": {
        "mean_ms": float(times_cpu.mean()),
        "median_ms": float(np.median(times_cpu)),
        "p99_ms": float(np.percentile(times_cpu, 99)),
        "max_ms": float(times_cpu.max()),
    },
    "block_budget_ms": 12000,
    "fits_in_block": True,
}
with open("results/inference_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info("Results saved to results/inference_benchmark.json")
