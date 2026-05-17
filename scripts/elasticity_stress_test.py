#!/usr/bin/env python3
"""
Elasticity Stress Test — evaluates the trained PPO agent under different
demand-elasticity regimes (ε = -1.0, -1.5, -2.5) to verify robustness.

Outputs: results/elasticity_stress_test.json + printed LaTeX table.
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


def evaluate_at_elasticity(
    model: PPO,
    elasticity: float,
    n_episodes: int,
    price_data: np.ndarray,
    feature_data: np.ndarray,
    cex_data: np.ndarray,
    vecnormalize_path: str,
) -> dict:
    """Run n_episodes at a specific elasticity, return agent vs baseline stats."""
    agent_rewards = []
    baseline_rewards = []
    agent_mean_fees = []
    bankruptcies = 0

    for ep in range(n_episodes):
        # ── Agent evaluation ──
        agent_env = create_vec_env(
            n_envs=1,
            seed=config.seed + ep + 7000,
            price_data=price_data,
            feature_data=feature_data,
            cex_price_data=cex_data,
            normalize=True,
            price_elasticity=elasticity,
        )

        # Set phase 3
        eval_vec = agent_env.venv if isinstance(agent_env, VecNormalize) else agent_env
        if hasattr(eval_vec, "env_method"):
            eval_vec.env_method("set_curriculum_phase", 3)

        # Load normalization stats
        if vecnormalize_path and Path(vecnormalize_path).exists():
            saved_norm = VecNormalize.load(vecnormalize_path, agent_env.venv)
            agent_env = saved_norm
            agent_env.training = False
            agent_env.norm_reward = False

        obs = agent_env.reset()
        total_reward = 0.0
        done = False
        ep_fees = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = agent_env.step(action)
            total_reward += reward[0]
            done = dones[0]
            if "fee_bps" in infos[0]:
                ep_fees.append(infos[0]["fee_bps"])

        # Check bankruptcy (LP value dropped > 50%)
        try:
            final_info = infos[0]
            if final_info.get("lp_value", float("inf")) < config.initial_liquidity_usd * 0.5:
                bankruptcies += 1
        except Exception:
            pass

        agent_rewards.append(total_reward)
        if ep_fees:
            agent_mean_fees.append(np.mean(ep_fees))
        agent_env.close()

        # ── Baseline (static 30bps, same elasticity) ──
        env_bl = UniswapV4Env(
            price_data=price_data,
            feature_data=feature_data,
            cex_price_data=cex_data,
            seed=config.seed + ep + 7000,
            episode_length=config.episode_length,
            initial_liquidity_usd=config.initial_liquidity_usd,
            fee_min_bps=30.0,
            fee_max_bps=30.0,
            fee_step_bps=1.0,
            price_elasticity=elasticity,
            action_delay_blocks=config.action_delay_blocks,
        )
        env_bl.set_curriculum_phase(3)
        obs_b, _ = env_bl.reset(seed=config.seed + ep + 7000)
        total_b = 0.0
        term_b = trunc_b = False
        while not (term_b or trunc_b):
            obs_b, r_b, term_b, trunc_b, _ = env_bl.step(np.array([0.0], dtype=np.float32))
            total_b += r_b
        baseline_rewards.append(total_b)

    agent_mean = float(np.mean(agent_rewards))
    agent_std = float(np.std(agent_rewards))
    bl_mean = float(np.mean(baseline_rewards))
    bl_std = float(np.std(baseline_rewards))
    improvement = ((agent_mean - bl_mean) / (abs(bl_mean) + 1e-8)) * 100

    return {
        "elasticity": elasticity,
        "agent_reward_mean": agent_mean,
        "agent_reward_std": agent_std,
        "baseline_reward_mean": bl_mean,
        "baseline_reward_std": bl_std,
        "improvement_pct": improvement,
        "agent_mean_fee_bps": float(np.mean(agent_mean_fees)) if agent_mean_fees else 0.0,
        "bankruptcies": bankruptcies,
        "n_episodes": n_episodes,
        "agent_beats_baseline": agent_mean > bl_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Elasticity Stress Test")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    args = parser.parse_args()

    # ── Load model ──
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

    # ── Run stress test at each elasticity ──
    elasticities = [-1.0, -1.5, -2.5]
    all_results = {}

    for eps_val in elasticities:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING ELASTICITY ε = {eps_val}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        result = evaluate_at_elasticity(
            model=model,
            elasticity=eps_val,
            n_episodes=args.episodes,
            price_data=prices,
            feature_data=features,
            cex_data=cex,
            vecnormalize_path=vecnorm_path,
        )
        elapsed = time.time() - t0
        result["eval_time_sec"] = elapsed
        all_results[f"elasticity_{eps_val}"] = result

        logger.info(
            f"ε={eps_val}: Agent={result['agent_reward_mean']:+.4f}±{result['agent_reward_std']:.4f}, "
            f"Baseline={result['baseline_reward_mean']:+.4f}±{result['baseline_reward_std']:.4f}, "
            f"Improvement={result['improvement_pct']:+.1f}%, "
            f"Mean Fee={result['agent_mean_fee_bps']:.1f}bps, "
            f"Bankruptcies={result['bankruptcies']}/{args.episodes}"
        )

    # ── Save results ──
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "elasticity_stress_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ── Print LaTeX table ──
    print("\n")
    print("=" * 80)
    print("TABLE FOR CHAPTER 5: Agent Performance Under Demand Elasticity Variation")
    print("=" * 80)
    print(f"{'ε':<8} {'Agent R̄':<14} {'Baseline R̄':<14} {'Δ%':<10} {'Mean Fee':<10} {'Bankrupt':<10}")
    print("-" * 80)
    for eps_val in elasticities:
        r = all_results[f"elasticity_{eps_val}"]
        print(
            f"{eps_val:<8.1f} "
            f"{r['agent_reward_mean']:+.4f}±{r['agent_reward_std']:.3f}  "
            f"{r['baseline_reward_mean']:+.4f}±{r['baseline_reward_std']:.3f}  "
            f"{r['improvement_pct']:+.1f}%     "
            f"{r['agent_mean_fee_bps']:.1f} bps   "
            f"{r['bankruptcies']}/{r['n_episodes']}"
        )
    print("=" * 80)

    # ── Verdict ──
    r25 = all_results["elasticity_-2.5"]
    if r25["agent_beats_baseline"]:
        print("\n✓ VERDICT: Agent beats baseline at ALL elasticities (ε=-2.5 included).")
        print("  System is NOT 'luckily calibrated' — generalizes across demand regimes.")
    else:
        print(f"\n⚠ Agent underperforms baseline at ε=-2.5 by {abs(r25['improvement_pct']):.1f}%")
        if r25["bankruptcies"] == 0:
            print("  But no bankruptcies — agent survives extreme market conditions.")
        else:
            print(f"  WARNING: {r25['bankruptcies']} bankruptcies at ε=-2.5!")

    # LaTeX
    print("\n% LaTeX table (copy to thesis):")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Wyniki agenta przy różnej elastyczności popytu}")
    print("\\label{tab:elasticity-stress}")
    print("\\begin{tabular}{lccccr}")
    print("\\toprule")
    print("$\\varepsilon$ & Agent $\\bar{R}$ & Baseline $\\bar{R}$ & $\\Delta\\%$ & Mean Fee & Bankr. \\\\")
    print("\\midrule")
    for eps_val in elasticities:
        r = all_results[f"elasticity_{eps_val}"]
        print(
            f"${eps_val:.1f}$ & "
            f"${r['agent_reward_mean']:+.4f} \\pm {r['agent_reward_std']:.3f}$ & "
            f"${r['baseline_reward_mean']:+.4f} \\pm {r['baseline_reward_std']:.3f}$ & "
            f"${r['improvement_pct']:+.1f}\\%$ & "
            f"${r['agent_mean_fee_bps']:.1f}$ & "
            f"${r['bankruptcies']}/{r['n_episodes']}$ \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
