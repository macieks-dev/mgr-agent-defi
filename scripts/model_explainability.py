"""
V4 explainability and robustness evaluation for chapters 5 and 6.

Outputs include the benchmark table, fee response to oracle staleness,
epsilon return-to-prior, stress-test summary, and TensorBoard training curves.
All figures are written to results/figures/.
"""

import argparse
import sys
import json
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats
from collections import defaultdict

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.env.uniswap_v4_env import UniswapV4Env
from src.training.train_ppo import create_vec_env
from src.data.dataset_builder import DatasetBuilder
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STALE_FIGURES = [
    "action_heatmap.png",
    "benchmark_comparison.png",
    "cumulative_lvr.png",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = str(config.model_save_dir / "ppo_v4_final_final.zip")
VECNORM_PATH = str(config.model_save_dir / "ppo_v4_final_vecnormalize.pkl")
RISK_METRIC_EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V4 explainability, benchmark, and robustness figure generation.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Overwrite results/benchmark_table.json if it already exists.",
    )
    return parser.parse_args()


def write_results_json(path: Path, data: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logger.warning(
            "Preserving existing results file: %s. Use --overwrite-results to replace it.",
            path,
        )
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to: {path}")


# Helper functions

def run_agent_episode_detailed(model, price_data, feature_data, cex_data,
                                seed=42, episode_length=1000):
    """Run one episode collecting per-step telemetry for explainability plots."""
    env = UniswapV4Env(
        price_data=price_data,
        feature_data=feature_data,
        cex_price_data=cex_data,
        seed=seed,
        episode_length=episode_length,
        initial_liquidity_usd=config.initial_liquidity_usd,
        fee_min_bps=config.fee_min_bps,
        fee_max_bps=config.fee_max_bps,
        fee_step_bps=config.fee_step_bps,
        price_elasticity=config.price_elasticity,
        domain_randomize=False,
        action_delay_blocks=config.action_delay_blocks,
    )
    env.set_curriculum_phase(3)

    # Wrap for normalization
    vec_env = create_vec_env(
        n_envs=1, seed=seed,
        price_data=price_data,
        feature_data=feature_data,
        cex_price_data=cex_data,
        normalize=True,
    )
    eval_vec = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    if hasattr(eval_vec, "env_method"):
        eval_vec.env_method("set_curriculum_phase", 3)

    if Path(VECNORM_PATH).exists():
        saved_norm = VecNormalize.load(VECNORM_PATH, vec_env.venv)
        vec_env = saved_norm
        vec_env.training = False
        vec_env.norm_reward = False

    obs = vec_env.reset()

    # Also run standalone env for detailed data
    obs_raw, _ = env.reset(seed=seed)
    steps_data = []

    for step in range(episode_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_norm, dones, infos = vec_env.step(action)

        # Step the raw env with same action
        obs_raw, reward, terminated, truncated, info = env.step(action[0])

        steps_data.append({
            "step": step + 1,
            "price": info.get("external_price", 0),
            "fee_bps": info.get("fee_bps", 0),
            "volume": info.get("volume", 0),
            "fee_income": info.get("fee_income", 0),
            "lvr": info.get("lvr", 0),
            "is_arb": info.get("is_arbitrage", False),
            "cumulative_pnl": info.get("cumulative_pnl", 0),
            "lp_value": info.get("lp_value", 0),
            "baseline_lp_value": info.get("baseline_lp_value", 0),
            "slippage_tax_active": info.get("slippage_tax_active", False),
            "oracle_stale_count": info.get("oracle_stale_count", 0),
            "reward": reward,
            "obs_epsilon": obs_raw[6] if len(obs_raw) > 6 else 0,
            "obs_volatility": obs_raw[0] if len(obs_raw) > 0 else 0,
            "obs_fee_norm": obs_raw[3] if len(obs_raw) > 3 else 0,
        })

        if dones[0] or terminated or truncated:
            break

    vec_env.close()
    return steps_data


def compute_max_drawdown_pct(cumulative_pnl, initial_liquidity):
    """Return non-negative max drawdown in percent of the running equity peak."""
    if len(cumulative_pnl) == 0:
        return None

    equity_curve = initial_liquidity + np.asarray(cumulative_pnl, dtype=float)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = np.maximum(0.0, (running_peak - equity_curve) / np.maximum(running_peak, RISK_METRIC_EPS))
    return float(np.max(drawdowns) * 100.0)


def compute_sharpe_ratio(returns):
    """Return Sharpe ratio, or None when variance collapses."""
    returns_arr = np.asarray(returns, dtype=float)
    if returns_arr.size == 0:
        return None

    std = float(np.std(returns_arr))
    if std <= RISK_METRIC_EPS:
        return None

    return float(np.mean(returns_arr) / std)


def compute_sortino_ratio(returns):
    """Return Sortino ratio, or None when downside deviation is undefined."""
    returns_arr = np.asarray(returns, dtype=float)
    if returns_arr.size == 0:
        return None

    downside = np.minimum(returns_arr, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev <= RISK_METRIC_EPS:
        return None

    return float(np.mean(returns_arr) / downside_dev)


def format_metric(value, precision=4, suffix=""):
    if value is None:
        return "—"
    return f"{value:.{precision}f}{suffix}"


def cleanup_stale_figures():
    for name in STALE_FIGURES:
        path = FIGURES_DIR / name
        if path.exists():
            path.unlink()
            logger.info(f"Removed stale figure: {path.name}")


def run_baseline_episode_detailed(price_data, feature_data, cex_data,
                                   fee_bps=30.0, seed=42, episode_length=1000):
    """Run one static-fee episode, collecting per-step data."""
    env = UniswapV4Env(
        price_data=price_data,
        feature_data=feature_data,
        cex_price_data=cex_data,
        seed=seed,
        episode_length=episode_length,
        initial_liquidity_usd=config.initial_liquidity_usd,
        fee_min_bps=fee_bps,
        fee_max_bps=fee_bps,
        fee_step_bps=1.0,
        price_elasticity=config.price_elasticity,
        domain_randomize=False,
        action_delay_blocks=config.action_delay_blocks,
    )
    env.set_curriculum_phase(3)
    obs, _ = env.reset(seed=seed)

    steps_data = []
    for step in range(episode_length):
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        steps_data.append({
            "step": step + 1,
            "price": info.get("external_price", 0),
            "fee_bps": info.get("fee_bps", 0),
            "volume": info.get("volume", 0),
            "fee_income": info.get("fee_income", 0),
            "lvr": info.get("lvr", 0),
            "is_arb": info.get("is_arbitrage", False),
            "cumulative_pnl": info.get("cumulative_pnl", 0),
            "lp_value": info.get("lp_value", 0),
            "reward": reward,
        })

        if terminated or truncated:
            break

    summary = env.get_episode_summary()
    return steps_data, summary


# ═══════════════════════════════════════════════════════════════════
# 1. FEE vs ORACLE STALENESS (Slippage Tax proof)
# ═══════════════════════════════════════════════════════════════════

def plot_fee_vs_staleness(steps_data):
    """Show how fee escalates with oracle staleness (Slippage Tax proof)."""
    logger.info("[1] Fee vs Oracle Staleness...")

    stale_counts = [s["oracle_stale_count"] for s in steps_data]
    fees = [s["fee_bps"] for s in steps_data]
    tax_active = [s["slippage_tax_active"] for s in steps_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Time series
    ax = axes[0]
    steps = range(len(stale_counts))
    ax.plot(steps, fees, color="#2196F3", linewidth=1.2, label="Bieżąca opłata (bps)", alpha=0.9)
    ax2 = ax.twinx()
    ax2.fill_between(steps, 0, stale_counts, alpha=0.2, color="#FF5722", label="Starość wyroczni")
    ax2.plot(steps, stale_counts, color="#FF5722", linewidth=0.8, alpha=0.7)

    # Mark tax activations
    for i, active in enumerate(tax_active):
        if active and (i == 0 or not tax_active[i-1]):
            ax.axvline(i, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Krok")
    ax.set_ylabel("Opłata (bps)", color="#2196F3")
    ax2.set_ylabel("Nieaktualne bloki wyroczni", color="#FF5722")
    ax.set_title("Slippage Tax: eskalacja opłat przy niaktualnej wyroczni", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Theoretical curve
    ax3 = axes[1]
    K = UniswapV4Env.SLIPPAGE_TAX_K
    threshold = 5
    stale_range = np.arange(0, 60)
    base_fee = 30.0
    effective_fees = []
    for s in stale_range:
        if s < threshold:
            effective_fees.append(base_fee)
        else:
            excess = s - threshold
            effective_fees.append(base_fee * np.exp(K * excess))

    ax3.plot(stale_range, effective_fees, color="#E91E63", linewidth=2.5)
    ax3.axhline(200, color="gray", linestyle="--", alpha=0.5, label="Limit opłaty (200 bps)")
    ax3.axvline(threshold, color="orange", linestyle="--", alpha=0.5,
                label=f"Próg ({threshold} bloków)")

    # Annotate key points
    for s_val in [10, 20, 30, 50]:
        if s_val >= threshold:
            f_val = base_fee * np.exp(K * (s_val - threshold))
            if f_val <= 220:
                ax3.annotate(f"{f_val:.0f} bps", xy=(s_val, f_val),
                           xytext=(s_val+3, f_val+10),
                           fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))

    ax3.set_xlabel("Kolejne nieaktualne bloki")
    ax3.set_ylabel("Efektywna opłata (bps)")
    ax3.set_title(f"$f_{{eff}} = f_{{base}} \\times e^{{k \\cdot \\Delta t}}$  (k={K})",
                  fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 300)

    plt.tight_layout()
    path = FIGURES_DIR / "fee_vs_staleness.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 2. EPSILON RETURN-TO-PRIOR (Bayesian Decay proof)
# ═══════════════════════════════════════════════════════════════════

def plot_epsilon_return_to_prior(steps_data):
    """Show epsilon decaying to prior during quiet periods."""
    logger.info("[2] Epsilon Return-to-Prior...")

    epsilon_vals = [s["obs_epsilon"] for s in steps_data]
    fees = [s["fee_bps"] for s in steps_data]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    steps = range(len(epsilon_vals))
    prior_norm = UniswapV4Env.EPSILON_PRIOR / 5.0

    # Panel 1: Epsilon over time
    ax = axes[0]
    ax.plot(steps, epsilon_vals, color="#2196F3", linewidth=1.2, label="ε̂ (obserwacja agenta)")
    ax.axhline(prior_norm, color="gray", linestyle=":", linewidth=1.5,
               label=f"Prior = {prior_norm:.2f}")
    ax.axhline(0, color="red", linestyle="-", alpha=0.3, linewidth=1)
    ax.text(len(epsilon_vals)*0.02, 0.01, "ε=0 (stary estymator: kłamstwo)", fontsize=8, color="red", alpha=0.6)

    # Mark quiet periods (fee unchanged for 10+ steps)
    quiet_starts = []
    quiet_ends = []
    in_quiet = False
    quiet_count = 0
    for i in range(1, len(fees)):
        if abs(fees[i] - fees[i-1]) < 0.01:
            quiet_count += 1
            if quiet_count >= 10 and not in_quiet:
                quiet_starts.append(i - quiet_count)
                in_quiet = True
        else:
            if in_quiet:
                quiet_ends.append(i)
                in_quiet = False
            quiet_count = 0
    if in_quiet:
        quiet_ends.append(len(fees))

    for qs, qe in zip(quiet_starts[:5], quiet_ends[:5]):
        ax.axvspan(qs, qe, alpha=0.08, color="red")

    ax.set_ylabel("Znormalizowane ε̂")
    ax.set_title("Bayesowski zanik do prioru: ε̂ w okresach aktywnych i cichych", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Fee over time (for context)
    ax2 = axes[1]
    ax2.plot(steps, fees, color="#4CAF50", linewidth=1.0)
    ax2.set_xlabel("Krok")
    ax2.set_ylabel("Opłata (bps)")
    ax2.set_title("Decyzje agenta dot. opłat", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "epsilon_return_to_prior.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 3. ACTION HEATMAP (Volume × Volatility → Fee)
# ═══════════════════════════════════════════════════════════════════

def plot_action_heatmap(all_steps_data):
    """Heatmap: X=Volume regime, Y=Volatility regime, Color=Avg Fee."""
    logger.info("[3] Action Heatmap...")

    volumes = np.array([s["volume"] for s in all_steps_data])
    vols = np.array([s["obs_volatility"] for s in all_steps_data])
    fees = np.array([s["fee_bps"] for s in all_steps_data])

    # Bin into quantiles
    n_bins = 15
    vol_pcts = np.percentile(vols[vols > 0], np.linspace(0, 100, n_bins + 1))
    volume_pcts = np.percentile(volumes[volumes > 0], np.linspace(0, 100, n_bins + 1))

    heatmap = np.full((n_bins, n_bins), np.nan)
    counts = np.zeros((n_bins, n_bins))

    for i in range(len(fees)):
        v = vols[i]
        vol = volumes[i]
        if v <= 0 or vol <= 0:
            continue

        vi = min(np.searchsorted(vol_pcts[1:], v), n_bins - 1)
        voli = min(np.searchsorted(volume_pcts[1:], vol), n_bins - 1)

        if np.isnan(heatmap[vi, voli]):
            heatmap[vi, voli] = fees[i]
            counts[vi, voli] = 1
        else:
            heatmap[vi, voli] += fees[i]
            counts[vi, voli] += 1

    mask = counts > 0
    heatmap[mask] /= counts[mask]

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list("fee_cmap", ["#1565C0", "#4CAF50", "#FF9800", "#E91E63"])
    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap=cmap,
                   interpolation="nearest")

    # Labels
    vol_labels = [f"{vol_pcts[i]:.4f}" for i in range(0, n_bins + 1, max(1, n_bins // 5))]
    volume_labels = [f"{volume_pcts[i]:.0f}" for i in range(0, n_bins + 1, max(1, n_bins // 5))]

    tick_pos = np.linspace(0, n_bins - 1, len(vol_labels)).astype(int)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(vol_labels, fontsize=8)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(volume_labels, fontsize=8, rotation=45)

    ax.set_xlabel("Wolumen (USD)", fontsize=12)
    ax.set_ylabel("Zrealizowana zmienność (σ₁₀)", fontsize=12)
    ax.set_title("Mapa cieplna polityki opłat agenta\n(Wolumen × Zmienność → Średnia opłata)", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Średnia opłata (bps)", fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "action_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 4. BENCHMARK TABLE (Agent vs Static vs HODL)
# ═══════════════════════════════════════════════════════════════════

def compute_benchmark_table(
    model,
    price_data,
    feature_data,
    cex_data,
    n_episodes=20,
    save_plot=False,
    overwrite_results=False,
):
    """Compute comprehensive benchmark comparison."""
    logger.info("[4] Benchmark Table...")

    results = {
        "RL Agent (V4)": {"returns": [], "drawdowns": [], "lvr_total": [], "fee_total": []},
        "Static 0.30%":  {"returns": [], "drawdowns": [], "lvr_total": [], "fee_total": []},
        "Static 0.05%":  {"returns": [], "drawdowns": [], "lvr_total": [], "fee_total": []},
    }

    for ep in range(n_episodes):
        if ep % 5 == 0:
            logger.info(f"    Benchmark episode {ep+1}/{n_episodes}...")

        seed = config.seed + ep + 8000

        # ── RL Agent ──
        agent_data = run_agent_episode_detailed(
            model, price_data, feature_data, cex_data,
            seed=seed, episode_length=config.episode_length,
        )
        if agent_data:
            rewards = [s["reward"] for s in agent_data]
            cum_reward = np.cumsum(rewards)
            # Return based on cumulative PnL (reward) relative to initial liquidity
            total_pnl = cum_reward[-1]
            ret_pct = total_pnl / config.initial_liquidity_usd * 100
            drawdown = compute_max_drawdown_pct(cum_reward, config.initial_liquidity_usd)
            results["RL Agent (V4)"]["returns"].append(ret_pct)
            results["RL Agent (V4)"]["drawdowns"].append(drawdown)
            results["RL Agent (V4)"]["lvr_total"].append(sum(s["lvr"] for s in agent_data))
            results["RL Agent (V4)"]["fee_total"].append(sum(s["fee_income"] for s in agent_data))

        # ── Static 0.30% ──
        baseline_data, _ = run_baseline_episode_detailed(
            price_data, feature_data, cex_data,
            fee_bps=30.0, seed=seed, episode_length=config.episode_length,
        )
        if baseline_data:
            bl_rewards = [s["reward"] for s in baseline_data]
            bl_cum = np.cumsum(bl_rewards)
            bl_pnl = bl_cum[-1]
            bl_ret = bl_pnl / config.initial_liquidity_usd * 100
            dd_bl = compute_max_drawdown_pct(bl_cum, config.initial_liquidity_usd)
            results["Static 0.30%"]["returns"].append(bl_ret)
            results["Static 0.30%"]["drawdowns"].append(dd_bl)
            results["Static 0.30%"]["lvr_total"].append(sum(s["lvr"] for s in baseline_data))
            results["Static 0.30%"]["fee_total"].append(sum(s["fee_income"] for s in baseline_data))

        # ── Static 0.05% ──
        bl5_data, _ = run_baseline_episode_detailed(
            price_data, feature_data, cex_data,
            fee_bps=5.0, seed=seed, episode_length=config.episode_length,
        )
        if bl5_data:
            bl5_rewards = [s["reward"] for s in bl5_data]
            bl5_cum = np.cumsum(bl5_rewards)
            bl5_pnl = bl5_cum[-1]
            b5_ret = bl5_pnl / config.initial_liquidity_usd * 100
            dd_b5 = compute_max_drawdown_pct(bl5_cum, config.initial_liquidity_usd)
            results["Static 0.05%"]["returns"].append(b5_ret)
            results["Static 0.05%"]["drawdowns"].append(dd_b5)
            results["Static 0.05%"]["lvr_total"].append(sum(s["lvr"] for s in bl5_data))
            results["Static 0.05%"]["fee_total"].append(sum(s["fee_income"] for s in bl5_data))

    # ── HODL (no LP, just hold 50/50 ETH + USDC) ──
    # HODL return = price change of ETH portion
    hodl_returns = []
    for ep in range(n_episodes):
        seed = config.seed + ep + 8000
        rng = np.random.default_rng(seed)
        max_start = len(price_data) - config.episode_length - 1
        start = rng.integers(0, max(1, max_start))
        p_start = price_data[start]
        p_end = price_data[min(start + config.episode_length, len(price_data) - 1)]
        # 50% in ETH, 50% in USDC
        eth_return = (p_end - p_start) / p_start
        hodl_ret = 0.5 * eth_return * 100  # only ETH portion changes
        hodl_returns.append(hodl_ret)

    # ── Compute metrics ──
    rl_ret = np.mean(results["RL Agent (V4)"]["returns"])
    s30_ret = np.mean(results["Static 0.30%"]["returns"])
    s05_ret = np.mean(results["Static 0.05%"]["returns"])
    hodl_ret_mean = np.mean(hodl_returns)

    rl_dd = np.mean(results["RL Agent (V4)"]["drawdowns"])
    s30_dd = np.mean(results["Static 0.30%"]["drawdowns"])
    s05_dd = np.mean(results["Static 0.05%"]["drawdowns"])

    rl_sharpe = compute_sharpe_ratio(results["RL Agent (V4)"]["returns"])
    hodl_sharpe = compute_sharpe_ratio(hodl_returns)
    rl_excess_sharpe = compute_sharpe_ratio(
        np.asarray(results["RL Agent (V4)"]["returns"], dtype=float)
        - np.asarray(results["Static 0.30%"]["returns"], dtype=float)
    )

    # Static-fee policies are reported without Sharpe/Sortino because near-zero
    # dispersion makes these ratios non-informative for deterministic strategies.
    s30_sharpe = None
    s05_sharpe = None

    rl_sortino = compute_sortino_ratio(results["RL Agent (V4)"]["returns"])
    s30_sortino = None
    s05_sortino = None

    rl_lvr = np.mean(results["RL Agent (V4)"]["lvr_total"])
    s30_lvr = np.mean(results["Static 0.30%"]["lvr_total"])
    s05_lvr = np.mean(results["Static 0.05%"]["lvr_total"])
    rl_lvr_bps = rl_lvr / config.initial_liquidity_usd * 10_000
    s30_lvr_bps = s30_lvr / config.initial_liquidity_usd * 10_000
    s05_lvr_bps = s05_lvr / config.initial_liquidity_usd * 10_000

    rl_fee = np.mean(results["RL Agent (V4)"]["fee_total"])
    s30_fee = np.mean(results["Static 0.30%"]["fee_total"])
    s05_fee = np.mean(results["Static 0.05%"]["fee_total"])

    # ── Display table ──
    print("\n    ╔══════════════════════╦═══════════════╦═══════════════╦═══════════════╦═══════════════╗")
    print("    ║ Metric               ║ RL Agent (V4) ║ Static 0.30%  ║ Static 0.05%  ║ Passive HODL  ║")
    print("    ╠══════════════════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╣")
    print(f"    ║ Cum. PnL Return (%)  ║ {rl_ret:+12.4f}  ║ {s30_ret:+12.4f}  ║ {s05_ret:+12.4f}  ║ {hodl_ret_mean:+12.4f}  ║")
    print(f"    ║ Max Drawdown         ║ {format_metric(rl_dd, precision=4, suffix='%'):>12}  ║ {format_metric(s30_dd, precision=4, suffix='%'):>12}  ║ {format_metric(s05_dd, precision=4, suffix='%'):>12}  ║      —        ║")
    print(f"    ║ Sharpe Ratio         ║ {format_metric(rl_sharpe, precision=4):>12}  ║ {format_metric(s30_sharpe, precision=4):>12}  ║ {format_metric(s05_sharpe, precision=4):>12}  ║ {format_metric(hodl_sharpe, precision=4):>12}  ║")
    print(f"    ║ Sortino Ratio        ║ {format_metric(rl_sortino, precision=4):>12}  ║ {format_metric(s30_sortino, precision=4):>12}  ║ {format_metric(s05_sortino, precision=4):>12}  ║      —        ║")
    print(f"    ║ Excess Sharpe        ║ {format_metric(rl_excess_sharpe, precision=4):>12}  ║      —        ║      —        ║      —        ║")
    print(f"    ║ LVR (bps)            ║ {rl_lvr_bps:12.4f}  ║ {s30_lvr_bps:12.4f}  ║ {s05_lvr_bps:12.4f}  ║      —        ║")
    print(f"    ║ Fee Income (USD)     ║ {rl_fee:12.2f}  ║ {s30_fee:12.2f}  ║ {s05_fee:12.2f}  ║      —        ║")
    print(f"    ║ Fee/LVR Ratio        ║ {rl_fee/(rl_lvr+1e-8):12.2f}  ║ {s30_fee/(s30_lvr+1e-8):12.2f}  ║ {s05_fee/(s05_lvr+1e-8):12.2f}  ║      —        ║")
    print("    ╚══════════════════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╝")

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        strategies = ["RL Agent\n(V4)", "Static\n0.30%", "Static\n0.05%", "HODL"]
        colors = ["#2196F3", "#FF5722", "#FF9800", "#9E9E9E"]

        ax = axes[0]
        vals = [rl_fee, s30_fee, s05_fee, 0]
        bars = ax.bar(strategies, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Dochód z opłat (USD)")
        ax.set_title("Łączny dochód z opłat na epizod", fontweight="bold")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                        f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        ax2 = axes[1]
        vals2 = [
            0.0 if rl_sharpe is None else rl_sharpe,
            0.0 if s30_sharpe is None else s30_sharpe,
            0.0 if s05_sharpe is None else s05_sharpe,
            0.0 if hodl_sharpe is None else hodl_sharpe,
        ]
        bars2 = ax2.bar(strategies, vals2, color=colors, edgecolor="white", linewidth=1.5)
        ax2.set_ylabel("Wsp. Sharpe'a")
        ax2.set_title("Zwrot skorygowany o ryzyko (Sharpe)", fontweight="bold")
        ax2.axhline(0, color="black", linewidth=0.5)
        for bar, raw_val in zip(bars2, [rl_sharpe, s30_sharpe, s05_sharpe, hodl_sharpe]):
            val = 0.0 if raw_val is None else raw_val
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (max(vals2) - min(vals2))*0.03 if val >= 0 else bar.get_height() - (max(vals2) - min(vals2))*0.06,
                     "—" if raw_val is None else f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = axes[2]
        vals3 = [rl_ret, s30_ret, s05_ret, hodl_ret_mean]
        bars3 = ax3.bar(strategies, vals3, color=colors, edgecolor="white", linewidth=1.5)
        ax3.set_ylabel("Skumulowany zwrot PnL (%)")
        ax3.set_title("Zwrot oparty na PnL na epizod", fontweight="bold")
        ax3.axhline(0, color="black", linewidth=0.5)
        for bar, val in zip(bars3, vals3):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (max(vals3) - min(vals3))*0.03 if val >= 0 else bar.get_height() - (max(vals3) - min(vals3))*0.06,
                     f"{val:.4f}%", ha="center", fontsize=9, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Porównanie benchmarków: Agent RL vs strategie statyczne", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = FIGURES_DIR / "benchmark_comparison.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"    Saved: {path}")

    # Save as JSON
    table_data = {
        "RL Agent (V4)": {
            "cum_pnl_return_pct": float(rl_ret), "max_drawdown": float(rl_dd),
            "sharpe_ratio": None if rl_sharpe is None else float(rl_sharpe),
            "sortino_ratio": None if rl_sortino is None else float(rl_sortino),
            "excess_sharpe_ratio_vs_static30": None if rl_excess_sharpe is None else float(rl_excess_sharpe),
            "lvr_bps": float(rl_lvr_bps), "fee_income_usd": float(rl_fee),
        },
        "Static 0.30%": {
            "cum_pnl_return_pct": float(s30_ret), "max_drawdown": float(s30_dd),
            "sharpe_ratio": None, "sortino_ratio": None,
            "lvr_bps": float(s30_lvr_bps), "fee_income_usd": float(s30_fee),
        },
        "Static 0.05%": {
            "cum_pnl_return_pct": float(s05_ret), "max_drawdown": float(s05_dd),
            "sharpe_ratio": None, "sortino_ratio": None,
            "lvr_bps": float(s05_lvr_bps), "fee_income_usd": float(s05_fee),
        },
        "HODL": {
            "total_return_pct": float(hodl_ret_mean),
            "sharpe_ratio": None if hodl_sharpe is None else float(hodl_sharpe),
        },
    }
    write_results_json(
        PROJECT_ROOT / "results" / "benchmark_table.json",
        table_data,
        overwrite=overwrite_results,
    )

    return results, table_data


# ═══════════════════════════════════════════════════════════════════
# 5. CUMULATIVE LVR COMPARISON
# ═══════════════════════════════════════════════════════════════════

def plot_cumulative_lvr(agent_data, baseline_data):
    """Show agent minimizes LVR better than static fee."""
    logger.info("[5] Cumulative LVR Comparison...")

    agent_cum_lvr = np.cumsum([s["lvr"] for s in agent_data])
    baseline_cum_lvr = np.cumsum([s["lvr"] for s in baseline_data])
    agent_cum_fee = np.cumsum([s["fee_income"] for s in agent_data])
    baseline_cum_fee = np.cumsum([s["fee_income"] for s in baseline_data])

    steps = range(len(agent_cum_lvr))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Cumulative LVR
    ax = axes[0]
    ax.plot(steps, agent_cum_lvr, color="#2196F3", linewidth=2, label="Agent (V4)")
    ax.plot(steps[:len(baseline_cum_lvr)], baseline_cum_lvr, color="#FF5722",
            linewidth=2, label="Static 0.30%")
    ax.set_xlabel("Krok")
    ax.set_ylabel("Skumulowane LVR (USD)")
    ax.set_title("Skumulowane LVR: Agent vs Baseline", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    final_agent_lvr = agent_cum_lvr[-1] if len(agent_cum_lvr) > 0 else 0
    final_bl_lvr = baseline_cum_lvr[-1] if len(baseline_cum_lvr) > 0 else 0
    reduction = (1 - final_agent_lvr / (final_bl_lvr + 1e-8)) * 100
    ax.text(0.98, 0.02, f"Redukcja LVR: {reduction:+.1f}%",
            transform=ax.transAxes, fontsize=10, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # Panel 2: Net PnL (Fee Income - LVR)
    ax2 = axes[1]
    agent_net = agent_cum_fee - agent_cum_lvr
    baseline_net = baseline_cum_fee[:len(baseline_cum_lvr)] - baseline_cum_lvr
    ax2.plot(steps, agent_net, color="#2196F3", linewidth=2, label="Agent (V4)")
    ax2.plot(steps[:len(baseline_net)], baseline_net, color="#FF5722",
             linewidth=2, label="Static 0.30%")
    ax2.fill_between(steps[:len(baseline_net)],
                     agent_net[:len(baseline_net)], baseline_net, alpha=0.12, color="#4CAF50")
    ax2.set_xlabel("Krok")
    ax2.set_ylabel("Zysk netto: Dochód z opłat − LVR (USD)")
    ax2.set_title("Zysk netto LP po LVR", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "cumulative_lvr.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 6. STRESS TESTS
# ═══════════════════════════════════════════════════════════════════

def run_stress_tests(model):
    """Run flash crash and low liquidity stress tests."""
    logger.info("[6] Stress Tests...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── 6a. FLASH CRASH: -20% in ~50 blocks ──
    logger.info("[6a] Flash Crash scenario...")
    n = 300
    prices_crash = np.zeros(n + 1)
    prices_crash[0] = 2000.0
    for i in range(1, n + 1):
        if 100 <= i <= 150:
            # Crash: -20% over 50 blocks
            prices_crash[i] = prices_crash[i-1] * np.exp(-0.004)
        elif 150 < i <= 200:
            # Partial recovery
            prices_crash[i] = prices_crash[i-1] * np.exp(0.002)
        else:
            prices_crash[i] = prices_crash[i-1] * np.exp(np.random.normal(0, 0.001))

    crash_env = UniswapV4Env(
        price_data=prices_crash, episode_length=n,
        initial_liquidity_usd=config.initial_liquidity_usd,
        domain_randomize=False, action_delay_blocks=1,
    )
    crash_env.set_curriculum_phase(3)

    vec_crash = create_vec_env(n_envs=1, seed=42, price_data=prices_crash, normalize=True)
    eval_vec = vec_crash.venv if isinstance(vec_crash, VecNormalize) else vec_crash
    if hasattr(eval_vec, "env_method"):
        eval_vec.env_method("set_curriculum_phase", 3)
    if Path(VECNORM_PATH).exists():
        saved = VecNormalize.load(VECNORM_PATH, vec_crash.venv)
        vec_crash = saved
        vec_crash.training = False
        vec_crash.norm_reward = False

    obs_v = vec_crash.reset()
    obs_r, _ = crash_env.reset(seed=42)
    crash_fees = []
    crash_prices_rec = []
    for step in range(n):
        action, _ = model.predict(obs_v, deterministic=True)
        obs_v, _, dones, _ = vec_crash.step(action)
        obs_r, _, t1, t2, info = crash_env.step(action[0])
        crash_fees.append(info["fee_bps"])
        crash_prices_rec.append(info["external_price"])
        if dones[0] or t1 or t2:
            break
    vec_crash.close()

    ax = axes[0, 0]
    ax.plot(crash_prices_rec, color="#333", linewidth=1.5, label="Price")
    ax.set_ylabel("Cena ETH (USD)", color="#333")
    ax.set_title("Flash Crash: −20% w 50 blokach", fontweight="bold")
    ax_fee = ax.twinx()
    ax_fee.plot(crash_fees, color="#E91E63", linewidth=1.5, label="Opłata agenta")
    ax_fee.set_ylabel("Opłata (bps)", color="#E91E63")
    ax.axvspan(100, 150, alpha=0.1, color="red", label="Strefa krachu")
    ax.legend(loc="upper left", fontsize=8)
    ax_fee.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 6b. LOW LIQUIDITY ──
    logger.info("[6b] Low Liquidity scenario...")
    low_liq_env = UniswapV4Env(
        episode_length=200,
        initial_liquidity_usd=10_000.0,  # 100x less than normal
        domain_randomize=False, action_delay_blocks=1,
        seed=123,
    )
    low_liq_env.set_curriculum_phase(3)

    vec_low = create_vec_env(
        n_envs=1, seed=123, normalize=True,
    )
    eval_vec_l = vec_low.venv if isinstance(vec_low, VecNormalize) else vec_low
    if hasattr(eval_vec_l, "env_method"):
        eval_vec_l.env_method("set_curriculum_phase", 3)
    if Path(VECNORM_PATH).exists():
        saved_l = VecNormalize.load(VECNORM_PATH, vec_low.venv)
        vec_low = saved_l
        vec_low.training = False
        vec_low.norm_reward = False

    obs_l = vec_low.reset()
    obs_lr, _ = low_liq_env.reset(seed=123)
    low_fees = []
    low_pnl = []
    for step in range(200):
        action, _ = model.predict(obs_l, deterministic=True)
        obs_l, _, dones_l, _ = vec_low.step(action)
        obs_lr, _, t1, t2, info_l = low_liq_env.step(action[0])
        low_fees.append(info_l["fee_bps"])
        low_pnl.append(info_l["cumulative_pnl"])
        if dones_l[0] or t1 or t2:
            break
    vec_low.close()

    ax2 = axes[0, 1]
    ax2.plot(low_fees, color="#2196F3", linewidth=1.5)
    ax2.set_xlabel("Krok")
    ax2.set_ylabel("Opłata (bps)")
    ax2.set_title("Niska płynność ($10K TVL): Zachowanie agenta", fontweight="bold")
    ax2_r = ax2.twinx()
    ax2_r.plot(low_pnl, color="#4CAF50", linewidth=1.2, alpha=0.7)
    ax2_r.set_ylabel("Skumulowany PnL", color="#4CAF50")
    ax2.grid(True, alpha=0.3)

    # ── 6c. ABLATION: with vs without Slippage Tax ──
    logger.info("[6c] Ablation: Slippage Tax enabled vs disabled...")
    # Create oracle-stale scenario (same prices repeated for 30 blocks)
    n_abl = 200
    prices_abl = np.zeros(n_abl + 1)
    prices_abl[0] = 2000.0
    cex_abl = np.zeros(n_abl + 1)
    cex_abl[0] = 2000.0
    for i in range(1, n_abl + 1):
        if 50 <= i <= 100:
            # CEX price frozen (oracle outage)
            prices_abl[i] = prices_abl[i-1] * np.exp(np.random.normal(0, 0.002))
            cex_abl[i] = cex_abl[49]  # frozen
        else:
            prices_abl[i] = prices_abl[i-1] * np.exp(np.random.normal(0, 0.001))
            cex_abl[i] = prices_abl[i] * (1 + np.random.normal(0, 0.0005))

    # WITH Slippage Tax
    env_with = UniswapV4Env(
        price_data=prices_abl, cex_price_data=cex_abl, episode_length=n_abl,
        initial_liquidity_usd=config.initial_liquidity_usd,
        domain_randomize=False, action_delay_blocks=1, seed=42,
    )
    env_with.set_curriculum_phase(3)
    obs_w, _ = env_with.reset(seed=42)
    fees_with = []
    pnl_with = []
    tax_active_arr = []
    for step in range(n_abl):
        obs_w, r, t1, t2, info = env_with.step(np.array([0.0], dtype=np.float32))
        fees_with.append(info["fee_bps"])
        pnl_with.append(info["cumulative_pnl"])
        tax_active_arr.append(info["slippage_tax_active"])
        if t1 or t2:
            break

    # WITHOUT Slippage Tax (set threshold very high)
    env_without = UniswapV4Env(
        price_data=prices_abl, cex_price_data=cex_abl, episode_length=n_abl,
        initial_liquidity_usd=config.initial_liquidity_usd,
        domain_randomize=False, action_delay_blocks=1, seed=42,
    )
    env_without.oracle_stale_threshold = 99999  # effectively disabled
    env_without.set_curriculum_phase(3)
    obs_wo, _ = env_without.reset(seed=42)
    fees_without = []
    pnl_without = []
    for step in range(n_abl):
        obs_wo, r, t1, t2, info = env_without.step(np.array([0.0], dtype=np.float32))
        fees_without.append(info["fee_bps"])
        pnl_without.append(info["cumulative_pnl"])
        if t1 or t2:
            break

    ax3 = axes[1, 0]
    ax3.plot(fees_with, color="#2196F3", linewidth=1.5, label="Ze Slippage Tax")
    ax3.plot(fees_without, color="#FF5722", linewidth=1.5, linestyle="--", label="Bez Slippage Tax")
    ax3.axvspan(50, 100, alpha=0.1, color="red", label="Awaria wyroczni")
    ax3.set_xlabel("Krok")
    ax3.set_ylabel("Opłata (bps)")
    ax3.set_title("Ablacja: efekt Slippage Tax przy awarii wyroczni", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(pnl_with, color="#2196F3", linewidth=2, label="Ze Slippage Tax")
    ax4.plot(pnl_without, color="#FF5722", linewidth=2, linestyle="--", label="Bez Slippage Tax")
    ax4.axvspan(50, 100, alpha=0.1, color="red", label="Awaria wyroczni")
    ax4.set_xlabel("Krok")
    ax4.set_ylabel("Skumulowany PnL")
    ax4.set_title("Wpływ Slippage Tax na PnL", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    pnl_w_final = pnl_with[-1] if pnl_with else 0
    pnl_wo_final = pnl_without[-1] if pnl_without else 0
    diff = pnl_w_final - pnl_wo_final
    ax4.text(0.02, 0.98, f"ΔPnL = {diff:+.2f}\n(Tax chroni {abs(diff):.2f})" if diff > 0 else f"ΔPnL = {diff:+.2f}",
             transform=ax4.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = FIGURES_DIR / "stress_tests.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 7. TENSORBOARD TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════

def plot_training_curves():
    """Extract and plot TensorBoard training curves."""
    logger.info("[7] TensorBoard Training Curves...")

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("TensorBoard not available, skipping.")
        return

    # Find the latest event file
    tb_dirs = sorted((PROJECT_ROOT / "runs").iterdir())
    v4_dirs = [d for d in tb_dirs if "v4_final" in d.name]
    if not v4_dirs:
        logger.warning("No V4 TensorBoard logs found, skipping.")
        return

    tb_dir = v4_dirs[-1]
    logger.info(f"Loading from: {tb_dir.name}")

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    logger.info(f"Available tags: {len(tags)}")

    # Target metrics
    metrics = {
        "train/explained_variance": "Wyjaśniona wariancja",
        "train/policy_gradient_loss": "Strata polityki",
        "train/value_loss": "Strata wartości",
        "train/entropy_loss": "Strata entropii",
        "rollout/ep_rew_mean": "Średnia nagroda epizodu",
        "train/learning_rate": "Współczynnik uczenia",
    }

    available = {k: v for k, v in metrics.items() if k in tags}

    if not available:
        # Try without prefix
        for tag in tags:
            for key in metrics:
                if key.split("/")[-1] in tag:
                    available[tag] = metrics[key]

    if not available:
        logger.warning(f"No matching metrics found. Tags: {tags[:20]}")
        return

    n_plots = len(available)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (tag, label) in enumerate(available.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        ax.plot(steps, values, color="#2196F3", linewidth=0.8, alpha=0.4)
        # Smoothed
        if len(values) > 20:
            window = max(len(values) // 50, 5)
            smoothed = np.convolve(values, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smoothed, color="#1565C0", linewidth=2)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Kroki treningowe")
        ax.grid(True, alpha=0.3)

        # Phase boundaries
        phase2 = int(config.total_timesteps * config.curriculum_phase2_frac)
        phase3 = int(config.total_timesteps * config.curriculum_phase3_frac)
        ax.axvline(phase2, color="orange", linestyle="--", alpha=0.4, linewidth=1)
        ax.axvline(phase3, color="red", linestyle="--", alpha=0.4, linewidth=1)
        if idx == 0:
            ax.text(phase2, ax.get_ylim()[1], " P2", fontsize=7, color="orange")
            ax.text(phase3, ax.get_ylim()[1], " P3", fontsize=7, color="red")

    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("Krzywe treningowe V4 (10M kroków)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("Model Explainability & Defense Preparation")
    logger.info("=" * 65)

    logger.info("[1] Loading V4 model...")
    model = PPO.load(MODEL_PATH)

    # Load test data
    logger.info("[2] Loading test data...")
    tf, tp, tc = DatasetBuilder.load_split("test", config.data_processed_dir)
    max_eval = 500_000
    if len(tp) > max_eval:
        tp = tp[:max_eval]
        if tf is not None: tf = tf[:max_eval]
        if tc is not None: tc = tc[:max_eval]
    logger.info(f"Test data: {len(tp):,} prices")
    cleanup_stale_figures()

    logger.info("[3] Running detailed agent episode for appendix plots...")
    agent_data = run_agent_episode_detailed(
        model, tp, tf, tc, seed=42, episode_length=config.episode_length,
    )
    logger.info(f"Collected {len(agent_data)} steps")

    # Run baseline for comparison
    baseline_data, _ = run_baseline_episode_detailed(
        tp, tf, tc, fee_bps=30.0, seed=42, episode_length=config.episode_length,
    )

    logger.info("[4] Generating selected outputs...")
    plot_fee_vs_staleness(agent_data)
    plot_epsilon_return_to_prior(agent_data)
    compute_benchmark_table(
        model,
        tp,
        tf,
        tc,
        n_episodes=20,
        save_plot=False,
        overwrite_results=args.overwrite_results,
    )
    run_stress_tests(model)
    plot_training_curves()

    logger.info("=" * 65)
    logger.info("ALL DONE — figures saved to results/figures/")
    logger.info("=" * 65)

    for f in sorted(FIGURES_DIR.iterdir()):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
