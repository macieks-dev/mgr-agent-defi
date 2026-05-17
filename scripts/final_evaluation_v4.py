"""
Final V4 Model Evaluation — Chapter 5 Results.

PnL, Sharpe, Cohen's d, Welch's t-test, Bayesian decay visualization.
Outputs saved to results/ and results/figures/.
"""

import argparse
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import stats

matplotlib.use("Agg")

# ── Setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.config import config
from src.env.uniswap_v4_env import UniswapV4Env
from src.training.train_ppo import create_vec_env
from src.data.dataset_builder import DatasetBuilder
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STALE_FIGURES = [
    "v4_reward_distribution.png",
]

MODEL_PATH = str(config.model_save_dir / "ppo_v4_final_final.zip")
VECNORM_PATH = str(config.model_save_dir / "ppo_v4_final_vecnormalize.pkl")
N_EPISODES = 50
BASELINE_FEE_BPS = 30.0  # static 0.30%


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final V4 evaluation and generate appendix figures.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Overwrite results/v4_final_evaluation.json if it already exists.",
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


def cleanup_stale_figures() -> None:
    for name in STALE_FIGURES:
        path = FIGURES_DIR / name
        if path.exists():
            path.unlink()
            logger.info(f"Removed stale figure: {path.name}")


def load_test_data():
    """Load test split (Oct 2025 – Jan 2026)."""
    tf, tp, tc = DatasetBuilder.load_split("test", config.data_processed_dir)
    logger.info(f"Test data loaded: {len(tp):,} prices")

    max_eval = 500_000
    if len(tp) > max_eval:
        tp = tp[:max_eval]
        if tf is not None:
            tf = tf[:max_eval]
        if tc is not None:
            tc = tc[:max_eval]

    return tf, tp, tc


def run_evaluation(model, price_data, eval_features, eval_cex, n_episodes):
    """Run agent and baseline for n_episodes, return per-episode rewards."""
    agent_rewards = []
    baseline_rewards = []
    agent_fees_collected = []
    baseline_fees_collected = []

    for ep in range(n_episodes):
        if ep % 10 == 0:
            logger.info(f"  Episode {ep+1}/{n_episodes}...")

        # ── Agent episode ──
        agent_env = create_vec_env(
            n_envs=1,
            seed=config.seed + ep + 5000,
            price_data=price_data,
            feature_data=eval_features,
            cex_price_data=eval_cex,
            normalize=True,
        )
        eval_vec = agent_env.venv if isinstance(agent_env, VecNormalize) else agent_env
        if hasattr(eval_vec, "env_method"):
            eval_vec.env_method("set_curriculum_phase", 3)

        if Path(VECNORM_PATH).exists():
            saved_norm = VecNormalize.load(VECNORM_PATH, agent_env.venv)
            agent_env = saved_norm
            agent_env.training = False
            agent_env.norm_reward = False

        obs = agent_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = agent_env.step(action)
            total_reward += reward[0]
            done = dones[0]

        agent_rewards.append(total_reward)

        # Try to get fee info
        try:
            base_env = agent_env.venv.envs[0]
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, '_total_fees'):
                agent_fees_collected.append(base_env._total_fees)
        except Exception:
            pass

        agent_env.close()

        # ── Baseline episode ──
        env_baseline = UniswapV4Env(
            price_data=price_data,
            feature_data=eval_features,
            cex_price_data=eval_cex,
            seed=config.seed + ep + 5000,
            episode_length=config.episode_length,
            initial_liquidity_usd=config.initial_liquidity_usd,
            fee_min_bps=BASELINE_FEE_BPS,
            fee_max_bps=BASELINE_FEE_BPS,
            fee_step_bps=1.0,
            price_elasticity=config.price_elasticity,
            domain_randomize=False,
            action_delay_blocks=config.action_delay_blocks,
        )
        env_baseline.set_curriculum_phase(3)
        obs_b, _ = env_baseline.reset(seed=config.seed + ep + 5000)
        total_reward_b = 0.0
        terminated_b = truncated_b = False
        while not (terminated_b or truncated_b):
            action_b = np.array([0.0], dtype=np.float32)
            obs_b, reward_b, terminated_b, truncated_b, _ = env_baseline.step(action_b)
            total_reward_b += reward_b

        baseline_rewards.append(total_reward_b)
        if hasattr(env_baseline, '_total_fees'):
            baseline_fees_collected.append(env_baseline._total_fees)

    return (
        np.array(agent_rewards),
        np.array(baseline_rewards),
        agent_fees_collected,
        baseline_fees_collected,
    )


def compute_statistics(agent_rewards, baseline_rewards):
    """Compute all statistical metrics (Sharpe, Cohen's d, Welch's t-test)."""
    excess_returns = agent_rewards - baseline_rewards

    # Basic statistics
    agent_mean = np.mean(agent_rewards)
    agent_std = np.std(agent_rewards)
    baseline_mean = np.mean(baseline_rewards)
    baseline_std = np.std(baseline_rewards)
    improvement_pct = ((agent_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)) * 100

    # Sharpe Ratio (excess returns over baseline)
    excess_mean = np.mean(excess_returns)
    excess_std = np.std(excess_returns)
    sharpe_ratio = excess_mean / (excess_std + 1e-8)

    # Also compute agent-only Sharpe (reward per unit risk)
    agent_sharpe = agent_mean / (agent_std + 1e-8)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((len(agent_rewards) - 1) * agent_std**2 + (len(baseline_rewards) - 1) * baseline_std**2)
        / (len(agent_rewards) + len(baseline_rewards) - 2)
    )
    cohens_d = (agent_mean - baseline_mean) / (pooled_std + 1e-8)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(agent_rewards, baseline_rewards, equal_var=False)

    # Win rate
    wins = np.sum(agent_rewards > baseline_rewards)
    win_rate = wins / len(agent_rewards)

    # 95% CI for excess return
    ci_95 = stats.t.interval(
        0.95,
        df=len(excess_returns) - 1,
        loc=excess_mean,
        scale=stats.sem(excess_returns),
    )

    results = {
        "agent_mean": float(agent_mean),
        "agent_std": float(agent_std),
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "improvement_pct": float(improvement_pct),
        "sharpe_ratio_excess": float(sharpe_ratio),
        "sharpe_ratio_agent": float(agent_sharpe),
        "cohens_d": float(cohens_d),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "win_rate": float(win_rate),
        "n_episodes": int(len(agent_rewards)),
        "excess_return_mean": float(excess_mean),
        "excess_return_std": float(excess_std),
        "ci_95_lower": float(ci_95[0]),
        "ci_95_upper": float(ci_95[1]),
    }

    return results


def plot_reward_distribution(agent_rewards, baseline_rewards, stats_dict):
    """Plot per-episode reward distribution and cumulative PnL."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(agent_rewards, bins=20, alpha=0.7, label="Agent (V4)", color="#2196F3")
    ax.hist(baseline_rewards, bins=20, alpha=0.7, label="Baseline (0,30%)", color="#FF5722")
    ax.axvline(np.mean(agent_rewards), color="#1565C0", linestyle="--", linewidth=2, label=f"Agent μ={np.mean(agent_rewards):.1f}")
    ax.axvline(np.mean(baseline_rewards), color="#BF360C", linestyle="--", linewidth=2, label=f"Baseline μ={np.mean(baseline_rewards):.1f}")
    ax.set_xlabel("Nagroda za epizod (PnL)", fontsize=12)
    ax.set_ylabel("Liczba epizodów", fontsize=12)
    ax.set_title("Rozkład nagród: Agent vs Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Cumulative PnL over episodes
    ax2 = axes[1]
    episodes = np.arange(1, len(agent_rewards) + 1)
    ax2.plot(episodes, np.cumsum(agent_rewards), label="Agent (V4)", color="#2196F3", linewidth=2)
    ax2.plot(episodes, np.cumsum(baseline_rewards), label="Baseline (0,30%)", color="#FF5722", linewidth=2)
    ax2.fill_between(episodes, np.cumsum(agent_rewards), np.cumsum(baseline_rewards), alpha=0.15, color="#4CAF50")
    ax2.set_xlabel("Epizod", fontsize=12)
    ax2.set_ylabel("Skumulowana nagroda", fontsize=12)
    ax2.set_title("Skumulowany PnL w kolejnych epizodach", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = (
        f"Poprawa: {stats_dict['improvement_pct']:+.1f}%\n"
        f"Sharpe: {stats_dict['sharpe_ratio_excess']:.3f}\n"
        f"Cohen's d: {stats_dict['cohens_d']:.2f}\n"
        f"p-wartość: {stats_dict['p_value']:.2e}\n"
        f"Wygrane: {stats_dict['win_rate']:.1%}"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = FIGURES_DIR / "v4_reward_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_bayesian_decay_curve():
    """Visualize the Bayesian Prior Decay mechanism (ε̂ → prior during silence)."""
    EPSILON_PRIOR = -1.5
    EPSILON_DECAY_LAMBDA = 0.15
    elasticity_min = -3.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Decay from different starting points ──
    ax = axes[0]
    dt_range = np.linspace(0, 40, 200)
    start_epsilons = [-0.5, -1.0, -2.0, -2.5, -3.0]
    colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]

    for eps0, color in zip(start_epsilons, colors):
        confidence = np.exp(-EPSILON_DECAY_LAMBDA * dt_range)
        decayed = EPSILON_PRIOR + (eps0 - EPSILON_PRIOR) * confidence
        normalized = decayed / 5.0
        ax.plot(dt_range, normalized, color=color, linewidth=2,
                label=f"ε₀ = {eps0:.1f}")

    # Prior line
    ax.axhline(EPSILON_PRIOR / 5.0, color="gray", linestyle=":", linewidth=1.5,
               label=f"Prior = {EPSILON_PRIOR/5.0:.2f}")

    # Half-life marker
    half_life = np.log(2) / EPSILON_DECAY_LAMBDA
    ax.axvline(half_life, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(half_life + 0.5, -0.1, f"t½ ≈ {half_life:.1f}", fontsize=9, color="gray")

    ax.set_xlabel("Bloki od ostatniego pomiaru ε (Δt)", fontsize=11)
    ax.set_ylabel("Znormalizowane ε̂ (obserwacja agenta)", fontsize=11)
    ax.set_title("Bayesowski zanik do prioru dla elastyczności rynku", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.65, 0.02)

    # ── Panel 2: Confidence decay ──
    ax2 = axes[1]
    confidence = np.exp(-EPSILON_DECAY_LAMBDA * dt_range)
    ax2.plot(dt_range, confidence, color="#2196F3", linewidth=2.5)
    ax2.fill_between(dt_range, 0, confidence, alpha=0.15, color="#2196F3")

    # Confidence floor
    CONFIDENCE_FLOOR = 0.01
    floor_block = -np.log(CONFIDENCE_FLOOR) / EPSILON_DECAY_LAMBDA
    ax2.axhline(CONFIDENCE_FLOOR, color="red", linestyle="--", alpha=0.7,
                label=f"Próg = {CONFIDENCE_FLOOR} (Δt ≈ {floor_block:.0f})")
    ax2.axhline(0.5, color="orange", linestyle="--", alpha=0.5,
                label=f"50% pewności (t½ ≈ {half_life:.1f})")

    ax2.set_xlabel("Bloki od ostatniego pomiaru ε (Δt)", fontsize=11)
    ax2.set_ylabel("Pewność = exp(-λ·Δt)", fontsize=11)
    ax2.set_title("Pewność pomiaru w czasie", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    path = FIGURES_DIR / "bayesian_prior_decay.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_scenario_timeline():
    """Timeline scenario: active trading → silence → recovery."""
    fig, ax = plt.subplots(figsize=(12, 5))

    EPSILON_PRIOR = -1.5
    EPSILON_DECAY_LAMBDA = 0.15

    # Simulate a scenario
    blocks = np.arange(0, 80)
    epsilon_values = []
    raw_measurements = []

    # Phase 1 (0-20): Active trading, varied epsilon measurements
    np.random.seed(42)
    measurements_phase1 = [-2.0, -1.8, -2.2, -1.9, -2.1, -1.7, -2.3, -1.8, -2.0, -1.9,
                           -2.1, -1.6, -2.0, -1.8, -2.2, -1.7, -2.1, -1.9, -2.0, -1.8]

    # Phase 2 (20-50): Silence — no valid measurements
    # Phase 3 (50-80): Recovery — new measurements arrive

    measurements_phase3 = [-0.8, -0.7, -0.9, -0.6, -0.8, -0.7, -0.9, -0.5, -0.7, -0.8,
                           -0.6, -0.9, -0.7, -0.8, -0.5, -0.7, -0.9, -0.6, -0.8, -0.7,
                           -0.6, -0.8, -0.7, -0.5, -0.9, -0.6, -0.8, -0.7, -0.6, -0.5]

    last_valid = EPSILON_PRIOR
    blocks_since = 0

    for b in blocks:
        if b < 20:
            # Active phase — fresh measurement each block
            raw = measurements_phase1[b]
            last_valid = raw
            blocks_since = 0
            raw_measurements.append(raw / 5.0)
        elif b < 50:
            # Silence phase — decay
            blocks_since += 1
            confidence = np.exp(-EPSILON_DECAY_LAMBDA * blocks_since)
            if confidence < 0.01:
                confidence = 0.0
            decayed = EPSILON_PRIOR + (last_valid - EPSILON_PRIOR) * confidence
            raw_measurements.append(None)
        else:
            # Recovery phase — fresh measurements
            idx = b - 50
            raw = measurements_phase3[idx]
            last_valid = raw
            blocks_since = 0
            raw_measurements.append(raw / 5.0)

        # Compute current epsilon for plot
        if b < 20 or b >= 50:
            epsilon_values.append(last_valid / 5.0)
        else:
            confidence = np.exp(-EPSILON_DECAY_LAMBDA * blocks_since)
            if confidence < 0.01:
                confidence = 0.0
            decayed = EPSILON_PRIOR + (last_valid - EPSILON_PRIOR) * confidence
            epsilon_values.append(np.clip(decayed, -5.0, 0.0) / 5.0)

    epsilon_values = np.array(epsilon_values)

    # Plot
    ax.plot(blocks, epsilon_values, color="#2196F3", linewidth=2, label="ε̂ (obserwacja agenta)")

    # Mark phases
    ax.axvspan(0, 20, alpha=0.08, color="green", label="Aktywny handel")
    ax.axvspan(20, 50, alpha=0.08, color="red", label="Cisza (brak transakcji)")
    ax.axvspan(50, 80, alpha=0.08, color="green")

    # Prior line
    ax.axhline(EPSILON_PRIOR / 5.0, color="gray", linestyle=":", linewidth=1.5,
               label=f"Prior ε = {EPSILON_PRIOR/5.0:.2f}")

    # Zero line (dangerous - "perfectly inelastic")
    ax.axhline(0, color="red", linestyle="-", alpha=0.3, linewidth=1)
    ax.text(35, 0.015, "NIEBEZPIECZEŃSTWO: ε=0 (stary estymator)", fontsize=8, color="red", alpha=0.7)

    # Phase labels
    ax.text(10, -0.45, "Faza 1\nAktywny rynek", ha="center", fontsize=9, color="darkgreen", fontweight="bold")
    ax.text(35, -0.45, "Faza 2\n\"Cisza przed burzą\"", ha="center", fontsize=9, color="darkred", fontweight="bold")
    ax.text(65, -0.45, "Faza 3\nOdzyskiwanie", ha="center", fontsize=9, color="darkgreen", fontweight="bold")

    ax.set_xlabel("Numer bloku", fontsize=12)
    ax.set_ylabel("Znormalizowane ε̂ (obserwacja agenta)", fontsize=12)
    ax.set_title("Bayesowski zanik w działaniu: Aktywność → Cisza → Odzyskiwanie", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.55, 0.08)

    plt.tight_layout()
    path = FIGURES_DIR / "epsilon_decay_scenario.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def print_final_report(stats_dict):
    """Print formatted report for thesis Chapter 5."""
    print("\n" + "═" * 70)
    print("  FINAL V4 EVALUATION REPORT — Chapter 5")
    print("═" * 70)
    print(f"  Model:           PPO V4 (ε̂ estimator + Bayesian decay + slippage tax)")
    print(f"  Test Period:     Oct 2025 – Jan 2026 (out-of-sample)")
    print(f"  Episodes:        {stats_dict['n_episodes']}")
    print(f"  Baseline:        Static 0.30% fee")
    print("─" * 70)
    print(f"  Agent PnL:       {stats_dict['agent_mean']:+.2f} ± {stats_dict['agent_std']:.2f}")
    print(f"  Baseline PnL:    {stats_dict['baseline_mean']:+.2f} ± {stats_dict['baseline_std']:.2f}")
    print(f"  Improvement:     {stats_dict['improvement_pct']:+.1f}%")
    print("─" * 70)
    print(f"  Sharpe Ratio:    {stats_dict['sharpe_ratio_excess']:.4f}")
    print(f"  Agent Sharpe:    {stats_dict['sharpe_ratio_agent']:.4f}")
    print(f"  Cohen's d:       {stats_dict['cohens_d']:.4f}")
    print(f"  Win Rate:        {stats_dict['win_rate']:.1%}")
    print("─" * 70)
    print(f"  t-statistic:     {stats_dict['t_statistic']:.4f}")
    print(f"  p-value:         {stats_dict['p_value']:.2e}")
    sig = "***" if stats_dict['p_value'] < 0.001 else "**" if stats_dict['p_value'] < 0.01 else "*" if stats_dict['p_value'] < 0.05 else "n.s."
    print(f"  Significance:    {sig} (α=0.05)")
    print(f"  95% CI excess:   [{stats_dict['ci_95_lower']:+.2f}, {stats_dict['ci_95_upper']:+.2f}]")
    print("═" * 70)


def main():
    args = parse_args()

    logger.info("V4 Final Evaluation — Thesis Chapter 5")
    cleanup_stale_figures()

    logger.info("Loading trained V4 model...")
    model = PPO.load(MODEL_PATH)
    logger.info(f"Model loaded from: {MODEL_PATH}")

    logger.info("Loading test data (Oct 2025 – Jan 2026)...")
    eval_features, price_data, eval_cex = load_test_data()

    logger.info(f"Running {N_EPISODES}-episode evaluation...")
    agent_rewards, baseline_rewards, agent_fees, baseline_fees = run_evaluation(
        model, price_data, eval_features, eval_cex, N_EPISODES
    )

    logger.info("Computing statistics...")
    stats_dict = compute_statistics(agent_rewards, baseline_rewards)

    print_final_report(stats_dict)

    results_path = PROJECT_ROOT / "results" / "v4_final_evaluation.json"
    save_data = {
        **stats_dict,
        "agent_rewards": agent_rewards.tolist(),
        "baseline_rewards": baseline_rewards.tolist(),
        "model_version": "V4",
        "safety_mechanisms": [
            "Empirical ε estimator (no oracle bias)",
            "Bayesian Prior Decay (cisza przed burzą)",
            "Slippage Tax (exponential fee escalation)",
        ],
        "test_period": "2025-10-01 to 2026-01-31",
    }
    write_results_json(results_path, save_data, overwrite=args.overwrite_results)

    logger.info("Generating selected appendix figures...")
    plot_bayesian_decay_curve()
    plot_scenario_timeline()

    logger.info("All done! Files saved to results/ and results/figures/")


if __name__ == "__main__":
    main()
