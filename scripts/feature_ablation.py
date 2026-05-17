#!/usr/bin/env python3
"""
Feature Ablation — Occlusion-based feature importance (Zeiler & Fergus 2014).

For each of the 17 observation features, zeros it out at evaluation time
and measures reward drop vs full-feature baseline.

Outputs: results/feature_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.config import config
from src.env.uniswap_v4_env import UniswapV4Env
from src.data.dataset_builder import DatasetBuilder
from src.training.train_ppo import create_vec_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Human-readable feature names (17-dim observation)
FEATURE_NAMES = [
    "log_return_1",      # [0]
    "log_return_5",      # [1]
    "log_return_20",     # [2]
    "log_return_100",    # [3]
    "tick_delta",        # [4]
    "realized_vol_10",   # [5]
    "vol_ratio_10_100",  # [6]
    "ofi_10",            # [7]
    "hour_sin",          # [8]
    "hour_cos",          # [9]
    "dow_sin",           # [10]
    "dow_cos",           # [11]
    "current_fee_norm",  # [12]
    "cumulative_pnl",    # [13]
    "fee_income_rate",   # [14]
    "lvr_rate_bps",      # [15]
    "toxic_flow_ratio",  # [16]
]

# Feature groups for thesis analysis
FEATURE_GROUPS = {
    "LVR_protection": [5, 6, 15, 16],     # realized_vol, vol_ratio, lvr_rate, toxic_flow
    "Price_momentum": [0, 1, 2, 3],        # log returns
    "Market_microstructure": [4, 7],        # tick_delta, ofi
    "Temporal": [8, 9, 10, 11],             # hour/dow sin/cos
    "Runtime_state": [12, 13, 14],          # fee, pnl, fee_income
}


def evaluate_with_occlusion(
    model: PPO,
    occluded_features: list[int],
    n_episodes: int,
    price_data: np.ndarray,
    feature_data: np.ndarray,
    cex_data: np.ndarray,
    vecnormalize_path: str,
) -> dict:
    """Evaluate model with specific features zeroed out (occluded)."""
    agent_rewards = []

    for ep in range(n_episodes):
        agent_env = create_vec_env(
            n_envs=1,
            seed=config.seed + ep + 8000,
            price_data=price_data,
            feature_data=feature_data,
            cex_price_data=cex_data,
            normalize=True,
        )

        eval_vec = agent_env.venv if isinstance(agent_env, VecNormalize) else agent_env
        if hasattr(eval_vec, "env_method"):
            eval_vec.env_method("set_curriculum_phase", 3)

        if vecnormalize_path and Path(vecnormalize_path).exists():
            saved_norm = VecNormalize.load(vecnormalize_path, agent_env.venv)
            agent_env = saved_norm
            agent_env.training = False
            agent_env.norm_reward = False

        obs = agent_env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # Occlude features AFTER normalization (set to 0 = normalized mean)
            if occluded_features:
                obs_modified = obs.copy()
                for feat_idx in occluded_features:
                    obs_modified[0, feat_idx] = 0.0
                action, _ = model.predict(obs_modified, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, dones, infos = agent_env.step(action)
            total_reward += reward[0]
            done = dones[0]

        agent_rewards.append(total_reward)
        agent_env.close()

    return {
        "reward_mean": float(np.mean(agent_rewards)),
        "reward_std": float(np.std(agent_rewards)),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature Ablation Study")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model or str(config.model_save_dir / "ppo_v2_curriculum_12feat_final")
    vecnorm_path = args.vecnorm or str(config.model_save_dir / "ppo_v2_curriculum_12feat_vecnormalize.pkl")

    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # ── Load test data ──
    try:
        features, prices, cex = DatasetBuilder.load_split("test", config.data_processed_dir)
        max_eval = 500_000
        if len(prices) > max_eval:
            prices = prices[:max_eval]
            features = features[:max_eval]
            cex = cex[:max_eval]
        logger.info(f"Test data: {len(prices):,} prices, features {features.shape}")
    except Exception as e:
        logger.error(f"Cannot load test data: {e}")
        sys.exit(1)

    # ── 1. Full-feature baseline ──
    logger.info("=" * 60)
    logger.info("BASELINE: Full features (no occlusion)")
    logger.info("=" * 60)
    t0 = time.time()
    baseline = evaluate_with_occlusion(
        model, [], args.episodes, prices, features, cex, vecnorm_path
    )
    logger.info(
        f"Full features: R̄={baseline['reward_mean']:+.4f}±{baseline['reward_std']:.4f}"
        f" ({time.time()-t0:.1f}s)"
    )

    # ── 2. Single-feature occlusion ──
    single_results = {}
    for feat_idx in range(len(FEATURE_NAMES)):
        fname = FEATURE_NAMES[feat_idx]
        logger.info(f"Occluding feature [{feat_idx}] {fname}...")
        t0 = time.time()
        result = evaluate_with_occlusion(
            model, [feat_idx], args.episodes, prices, features, cex, vecnorm_path
        )
        drop_pct = ((result["reward_mean"] - baseline["reward_mean"])
                     / (abs(baseline["reward_mean"]) + 1e-8)) * 100
        result["reward_drop_pct"] = drop_pct
        result["feature_name"] = fname
        result["feature_index"] = feat_idx
        single_results[fname] = result
        logger.info(
            f"  [{feat_idx}] {fname}: R̄={result['reward_mean']:+.4f}, "
            f"Δ={drop_pct:+.1f}% ({time.time()-t0:.1f}s)"
        )

    # ── 3. Group-level occlusion ──
    group_results = {}
    for group_name, indices in FEATURE_GROUPS.items():
        names = [FEATURE_NAMES[i] for i in indices]
        logger.info(f"Occluding group '{group_name}': {names}...")
        t0 = time.time()
        result = evaluate_with_occlusion(
            model, indices, args.episodes, prices, features, cex, vecnorm_path
        )
        drop_pct = ((result["reward_mean"] - baseline["reward_mean"])
                     / (abs(baseline["reward_mean"]) + 1e-8)) * 100
        result["reward_drop_pct"] = drop_pct
        result["group_name"] = group_name
        result["features"] = names
        result["feature_indices"] = indices
        group_results[group_name] = result
        logger.info(
            f"  {group_name}: R̄={result['reward_mean']:+.4f}, "
            f"Δ={drop_pct:+.1f}% ({time.time()-t0:.1f}s)"
        )

    # ── Save results ──
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    all_results = {
        "baseline_full_features": baseline,
        "single_feature_occlusion": single_results,
        "group_occlusion": group_results,
    }
    out_path = results_dir / "feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ── Print results ──
    print("\n" + "=" * 80)
    print("FEATURE ABLATION RESULTS (Single Feature Occlusion)")
    print("=" * 80)
    print(f"Baseline (full features): R̄ = {baseline['reward_mean']:+.4f}±{baseline['reward_std']:.4f}")
    print("-" * 80)
    print(f"{'#':<4} {'Feature':<22} {'R̄ (occluded)':<16} {'Δ%':<10} {'Impact':<10}")
    print("-" * 80)

    # Sort by impact (most negative = most important)
    sorted_feats = sorted(single_results.values(), key=lambda x: x["reward_drop_pct"])
    for r in sorted_feats:
        impact = "CRITICAL" if r["reward_drop_pct"] < -30 else \
                 "HIGH" if r["reward_drop_pct"] < -10 else \
                 "MEDIUM" if r["reward_drop_pct"] < -3 else "LOW"
        print(
            f"[{r['feature_index']:2d}] {r['feature_name']:<22} "
            f"{r['reward_mean']:+.4f}±{r['reward_std']:.3f}  "
            f"{r['reward_drop_pct']:+.1f}%     {impact}"
        )

    print("\n" + "=" * 80)
    print("GROUP ABLATION (All features in group zeroed)")
    print("=" * 80)
    sorted_groups = sorted(group_results.values(), key=lambda x: x["reward_drop_pct"])
    for r in sorted_groups:
        print(f"  {r['group_name']:<25} Δ = {r['reward_drop_pct']:+.1f}%  (R̄={r['reward_mean']:+.4f})")

    # ── Thesis verdict: LVR protection ──
    lvr_group = group_results.get("LVR_protection", {})
    lvr_drop = lvr_group.get("reward_drop_pct", 0)
    print("\n" + "=" * 80)
    print("THESIS VERDICT: Does the agent learn LVR protection?")
    print("=" * 80)
    print(f"LVR-related features (vol, vol_ratio, lvr_rate, toxic_flow) group drop: {lvr_drop:+.1f}%")
    if lvr_drop < -30:
        print("✓ LVR features are CRITICAL — agent relies on them for fee optimization.")
        print("  The agent is learning genuine LVR protection, not noise.")
    elif lvr_drop < -10:
        print("⚠ LVR features have significant but not dominant impact.")
        print("  Agent uses LVR signals but also relies on other features.")
    else:
        print("✗ WARNING: LVR features have weak impact ({:.1f}%).".format(abs(lvr_drop)))
        print("  Agent may be learning from non-LVR signals.")

    # LaTeX table
    print("\n% LaTeX table (copy to thesis):")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Feature Ablation Study — Single Feature Occlusion}")
    print("\\label{tab:feature-ablation}")
    print("\\begin{tabular}{rlccl}")
    print("\\toprule")
    print("\\# & Feature & $\\bar{R}_{occluded}$ & $\\Delta\\%$ & Impact \\\\")
    print("\\midrule")
    for r in sorted_feats:
        impact = "CRITICAL" if r["reward_drop_pct"] < -30 else \
                 "HIGH" if r["reward_drop_pct"] < -10 else \
                 "MEDIUM" if r["reward_drop_pct"] < -3 else "LOW"
        feat_latex = r['feature_name'].replace('_', '\\_')
        print(
            f"{r['feature_index']} & {feat_latex} & "
            f"${r['reward_mean']:+.4f}$ & "
            f"${r['reward_drop_pct']:+.1f}\\%$ & "
            f"{impact} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
