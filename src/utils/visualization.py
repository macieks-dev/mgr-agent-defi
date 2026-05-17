"""
Visualization and monitoring utilities for the RL Hook project.

Provides plotting functions for:
  - Training curves (reward, loss, explained variance)
  - Fee dynamics over episodes
  - LVR analysis
  - Agent vs. baseline comparison
  - Action distribution analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available — plotting disabled")


# ══════════════════════════════════════════════════════════════════════════
#                     FEE DYNAMICS PLOT
# ══════════════════════════════════════════════════════════════════════════

def plot_episode_dynamics(
    episode_infos: list[dict],
    save_path: Optional[Path] = None,
    title: str = "Episode Dynamics",
):
    """Plot fee, price, LVR, and reward dynamics for a single episode."""
    if not HAS_MATPLOTLIB:
        logger.warning("Cannot plot: matplotlib not installed")
        return

    df = pd.DataFrame(episode_infos)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    steps = df["step"]

    # 1. Price
    axes[0].plot(steps, df["external_price"], label="External Price", linewidth=0.8)
    axes[0].plot(steps, df["amm_price"], label="AMM Price", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("Price (USDC)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Price Dynamics")

    # 2. Fee
    axes[1].plot(steps, df["fee_bps"], color="purple", linewidth=1.0)
    axes[1].axhline(y=30.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (30 bps)")
    axes[1].set_ylabel("Fee (bps)")
    axes[1].legend(loc="upper left")
    axes[1].set_title("Dynamic Fee (Agent)")

    # 3. LVR vs Fee Income
    axes[2].plot(steps, df["lvr"], label="LVR", color="red", linewidth=0.8)
    axes[2].plot(steps, df["fee_income"], label="Fee Income", color="green", linewidth=0.8)
    axes[2].set_ylabel("Value (USD)")
    axes[2].legend(loc="upper left")
    axes[2].set_title("LVR vs Fee Income")

    # 4. Cumulative PnL
    axes[3].plot(steps, df["cumulative_pnl"], color="blue", linewidth=1.0)
    axes[3].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[3].set_ylabel("Cumulative PnL")
    axes[3].set_xlabel("Step")
    axes[3].set_title("Agent Cumulative P&L")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#                AGENT VS BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def plot_evaluation_comparison(
    eval_results: dict,
    save_path: Optional[Path] = None,
):
    """Bar chart + histogram comparing agent vs. static fee baseline."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Agent vs. Static Fee Baseline", fontsize=14, fontweight="bold")

    # 1. Reward distribution
    agent_rewards = eval_results.get("agent_rewards", [])
    baseline_rewards = eval_results.get("baseline_rewards", [])

    if agent_rewards and baseline_rewards:
        axes[0].hist(agent_rewards, bins=30, alpha=0.7, label="Agent", color="blue")
        axes[0].hist(baseline_rewards, bins=30, alpha=0.7, label="Baseline", color="gray")
        axes[0].set_xlabel("Episode Reward")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Reward Distribution")
        axes[0].legend()

    # 2. Bar chart: mean rewards
    means = [eval_results["agent_reward_mean"], eval_results["baseline_reward_mean"]]
    stds = [eval_results["agent_reward_std"], eval_results["baseline_reward_std"]]
    labels = ["Agent", "Baseline (0.30%)"]
    colors = ["blue", "gray"]

    axes[1].bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title(f"Performance Comparison\nImprovement: {eval_results['improvement_pct']:+.1f}%")

    # 3. Summary text
    axes[2].axis("off")
    summary_text = (
        f"Agent Mean Reward: {eval_results['agent_reward_mean']:+.4f}\n"
        f"Agent Std: {eval_results['agent_reward_std']:.4f}\n\n"
        f"Baseline Mean Reward: {eval_results['baseline_reward_mean']:+.4f}\n"
        f"Baseline Std: {eval_results['baseline_reward_std']:.4f}\n\n"
        f"Improvement: {eval_results['improvement_pct']:+.2f}%\n"
        f"Episodes: {eval_results['n_episodes']}"
    )
    axes[2].text(
        0.1, 0.5, summary_text,
        fontsize=12, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[2].set_title("Summary Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#                   LVR ANALYSIS PLOT
# ══════════════════════════════════════════════════════════════════════════

def plot_lvr_analysis(
    lvr_history: list[dict],
    save_path: Optional[Path] = None,
    title: str = "LVR Analysis",
):
    """Detailed LVR analysis plot from LVRCalculator history."""
    if not HAS_MATPLOTLIB:
        return

    df = pd.DataFrame(lvr_history)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Price divergence
    axes[0, 0].plot(df["block_number"], df["price_divergence"] * 10000, linewidth=0.5)
    axes[0, 0].set_ylabel("Divergence (bps)")
    axes[0, 0].set_title("AMM vs External Price Divergence")

    # 2. Instantaneous LVR
    axes[0, 1].plot(df["block_number"], df["instantaneous_lvr_bps"], linewidth=0.5, color="red")
    axes[0, 1].set_ylabel("LVR (bps)")
    axes[0, 1].set_title("Instantaneous LVR")

    # 3. Fee earned
    axes[1, 0].plot(df["block_number"], df["instantaneous_fee_bps"], linewidth=0.5, color="green")
    axes[1, 0].set_ylabel("Fee (bps)")
    axes[1, 0].set_title("Instantaneous Fee Income")

    # 4. Cumulative LVR vs Fees
    axes[1, 1].plot(df["block_number"], df["cumulative_lvr"], label="Cumulative LVR", color="red")
    axes[1, 1].plot(df["block_number"], df["cumulative_fee"], label="Cumulative Fees", color="green")
    axes[1, 1].set_ylabel("USD")
    axes[1, 1].set_title("Cumulative LVR vs Fees")
    axes[1, 1].legend()

    # 5. Net LVR
    axes[2, 0].plot(df["block_number"], df["net_cumulative_lvr"], color="orange")
    axes[2, 0].axhline(y=0, color="gray", linestyle="--")
    axes[2, 0].set_ylabel("Net LVR (USD)")
    axes[2, 0].set_xlabel("Block")
    axes[2, 0].set_title("Net Cumulative LVR (LVR − Fees)")

    # 6. LP Value
    axes[2, 1].plot(df["block_number"], df["lp_value"], label="LP Value", color="blue")
    axes[2, 1].plot(df["block_number"], df["rebalancing_value"], label="Rebalancing", color="gray", linestyle="--")
    axes[2, 1].set_ylabel("Value (USD)")
    axes[2, 1].set_xlabel("Block")
    axes[2, 1].set_title("LP Value vs Rebalancing Portfolio")
    axes[2, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#               TRAINING METRICS FROM TENSORBOARD
# ══════════════════════════════════════════════════════════════════════════

def load_tensorboard_metrics(log_dir: Path | str) -> Optional[pd.DataFrame]:
    """Load training metrics from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard not installed — cannot load metrics")
        return None

    log_dir = Path(log_dir)
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return None

    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {e.step: e.value for e in events}

    return pd.DataFrame(data)
