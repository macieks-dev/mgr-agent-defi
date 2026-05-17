"""
PPO training pipeline: vectorized envs, TensorBoard callbacks,
curriculum learning, checkpointing, and backtesting vs static fee.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, sync_envs_normalization

from src.config import config
from src.env.uniswap_v4_env import UniswapV4Env
from src.data.tick_loader import generate_synthetic_data
from src.data.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#                     ENVIRONMENT FACTORY
# ══════════════════════════════════════════════════════════════════════════

def make_env(
    rank: int = 0,
    seed: int = 42,
    price_data: Optional[np.ndarray] = None,
    feature_data: Optional[np.ndarray] = None,
    cex_price_data: Optional[np.ndarray] = None,
    **env_kwargs,
) -> Callable:
    """Thunk returning a UniswapV4Env instance for vectorized env creation."""
    def _init() -> UniswapV4Env:
        defaults = dict(
            price_data=price_data,
            feature_data=feature_data,
            cex_price_data=cex_price_data,
            seed=seed + rank,
            episode_length=config.episode_length,
            initial_liquidity_usd=config.initial_liquidity_usd,
            fee_min_bps=config.fee_min_bps,
            fee_max_bps=config.fee_max_bps,
            fee_step_bps=config.fee_step_bps,
            price_elasticity=config.price_elasticity,
            elasticity_min=config.elasticity_min,
            elasticity_max=config.elasticity_max,
            domain_randomize=config.domain_randomize,
            ema_alpha=config.ema_alpha,
            gas_penalty_weight=config.gas_penalty_weight,
            action_delay_blocks=config.action_delay_blocks,
        )
        defaults.update(env_kwargs)
        env = UniswapV4Env(**defaults)
        env = Monitor(env)
        return env

    return _init


def create_vec_env(
    n_envs: int = 16,
    seed: int = 42,
    price_data: Optional[np.ndarray] = None,
    feature_data: Optional[np.ndarray] = None,
    cex_price_data: Optional[np.ndarray] = None,
    normalize: bool = True,
    **env_kwargs,
) -> VecNormalize | SubprocVecEnv:
    """Create SubprocVecEnv with optional VecNormalize wrapper."""
    env_fns = [
        make_env(
            rank=i, seed=seed,
            price_data=price_data,
            feature_data=feature_data,
            cex_price_data=cex_price_data,
            **env_kwargs,
        )
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)

    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config.gamma,
        )

    return vec_env


# ══════════════════════════════════════════════════════════════════════════
#                   CUSTOM TENSORBOARD CALLBACK
# ══════════════════════════════════════════════════════════════════════════

class UniswapMetricsCallback(BaseCallback):
    """TensorBoard callback for Uniswap-specific metrics (fees, LVR, arb fraction)."""

    def __init__(self, eval_freq: int = 5000, eval_env: Optional[VecNormalize] = None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self._last_eval_step = 0
        self._eval_env = eval_env

    def _on_step(self) -> bool:
        if self._eval_env is not None and isinstance(self.training_env, VecNormalize):
            sync_envs_normalization(self.training_env, self._eval_env)

        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.logger.record("rollout/ep_len", info["episode"]["l"])
                self.logger.record("rollout/ep_rew", info["episode"]["r"])

            if "fee_bps" in info:
                self.logger.record("uniswap/current_fee_bps", info["fee_bps"])

            if "lvr" in info:
                self.logger.record("uniswap/instantaneous_lvr", info["lvr"])

            if "fee_income" in info:
                self.logger.record("uniswap/fee_income", info["fee_income"])

            if "is_arbitrage" in info:
                self.logger.record("uniswap/is_arb_swap", float(info["is_arbitrage"]))

            if "cumulative_pnl" in info:
                self.logger.record("uniswap/cumulative_pnl", info["cumulative_pnl"])

            if "baseline_lp_value" in info and "lp_value" in info:
                improvement = info["lp_value"] - info["baseline_lp_value"]
                self.logger.record("uniswap/improvement_over_baseline", improvement)

            if "reward_components" in info:
                rc = info["reward_components"]
                self.logger.record("reward/fee_income_bps", rc.get("fee_income_bps", 0))
                self.logger.record("reward/lvr_bps", rc.get("lvr_bps", 0))
                self.logger.record("reward/volume_drop_penalty", rc.get("volume_drop_penalty", 0))
                self.logger.record("reward/reward_multiplicative", rc.get("reward_multiplicative", 0))
                self.logger.record("reward/churn_penalty", rc.get("churn_penalty", 0))
                self.logger.record("reward/episode_elasticity", rc.get("episode_elasticity", -1.5))

        return True

    def _on_rollout_end(self) -> None:
        pass


class CurriculumCallback(BaseCallback):
    """
    3-phase curriculum: (1) stability → (2) fee awareness → (3) full LVR objective.
    Transitions at configurable fractions of total_timesteps.
    """

    def __init__(
        self,
        phase2_frac: float = 0.30,
        phase3_frac: float = 0.70,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.phase2_frac = phase2_frac
        self.phase3_frac = phase3_frac
        self._current_phase = 1

    def _on_training_start(self) -> None:
        self._set_phase(1)

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        if progress >= self.phase3_frac and self._current_phase < 3:
            self._set_phase(3)
        elif progress >= self.phase2_frac and self._current_phase < 2:
            self._set_phase(2)
        return True

    def _set_phase(self, phase: int) -> None:
        if phase == self._current_phase and self.num_timesteps > 0:
            return
        self._current_phase = phase
        logger.info(
            f"[Curriculum] Phase {phase} activated at step {self.num_timesteps:,} "
            f"({self.num_timesteps / self.model._total_timesteps * 100:.1f}%)"
        )
        self.logger.record("curriculum/phase", phase)

        # Propagate to SubprocVecEnv workers
        env = self.training_env
        if isinstance(env, VecNormalize):
            env = env.venv
        if hasattr(env, "env_method"):
            try:
                env.env_method("set_curriculum_phase", phase)
            except Exception as e:
                logger.warning(f"Could not set curriculum phase on envs: {e}")


# ══════════════════════════════════════════════════════════════════════════
#                      LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear LR decay from initial_value to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def cosine_schedule(initial_value: float, min_value: float = 1e-6) -> Callable[[float], float]:
    """Cosine annealing LR schedule."""
    def schedule(progress_remaining: float) -> float:
        return min_value + 0.5 * (initial_value - min_value) * (
            1 + np.cos(np.pi * (1 - progress_remaining))
        )
    return schedule


# ══════════════════════════════════════════════════════════════════════════
#                        TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_agent(
    total_timesteps: Optional[int] = None,
    n_envs: Optional[int] = None,
    price_data: Optional[np.ndarray] = None,
    feature_data: Optional[np.ndarray] = None,
    cex_price_data: Optional[np.ndarray] = None,
    resume_from: Optional[str] = None,
    experiment_name: str = "ppo_uniswap_v4",
    use_lr_schedule: bool = True,
) -> PPO:
    """
    Train PPO on UniswapV4Env with curriculum learning.

    Data priority: explicit arrays > data/processed/ > synthetic GBM.
    """
    total_timesteps = total_timesteps or config.total_timesteps
    n_envs = n_envs or config.n_envs

    logger.info("=" * 60)
    logger.info(f"Training PPO Agent: {experiment_name}")
    logger.info(f"Timesteps: {total_timesteps:,}, Envs: {n_envs}")
    logger.info(f"Action delay: {config.action_delay_blocks} block(s)")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # ── Load training data ──
    if price_data is None:
        try:
            train_features, train_prices, train_cex = DatasetBuilder.load_split(
                "train", config.data_processed_dir
            )
            if len(train_prices) > config.episode_length * 2:
                price_data = train_prices
                feature_data = train_features
                cex_price_data = train_cex
                logger.info(
                    f"Loaded REAL training data: {len(price_data):,} prices, "
                    f"features {train_features.shape}"
                )
            else:
                raise ValueError("Training split too small")
        except Exception as e:
            logger.warning(f"Cannot load real data ({e}), falling back to synthetic")
            df = generate_synthetic_data(n_records=500_000, volatility=0.02, seed=config.seed)
            price_data = df["price"].values

    # SubprocVecEnv copies data to each worker — cap to avoid OOM
    max_train_rows = 4_000_000
    if price_data is not None and len(price_data) > max_train_rows:
        rng = np.random.default_rng(config.seed)
        start = rng.integers(0, len(price_data) - max_train_rows)
        end = start + max_train_rows
        logger.info(
            f"Subsampling training data: {len(price_data):,} → {max_train_rows:,} "
            f"(rows {start:,}–{end:,})"
        )
        price_data = price_data[start:end]
        if feature_data is not None:
            feature_data = feature_data[start:end]
        if cex_price_data is not None:
            cex_price_data = cex_price_data[start:end]

    # ── Create environments ──
    logger.info(f"Creating {n_envs} parallel environments")
    train_env = create_vec_env(
        n_envs=n_envs,
        seed=config.seed,
        price_data=price_data,
        feature_data=feature_data,
        cex_price_data=cex_price_data,
        normalize=True,
    )

    # Eval env (validation split if available)
    eval_prices = price_data
    eval_features = feature_data
    eval_cex = cex_price_data
    try:
        vf, vp, vc = DatasetBuilder.load_split("val", config.data_processed_dir)
        if len(vp) > config.episode_length * 2:
            eval_prices, eval_features, eval_cex = vp, vf, vc
            logger.info(f"Using validation split for eval: {len(vp):,} prices")
            max_eval_rows = 500_000
            if len(eval_prices) > max_eval_rows:
                eval_prices = eval_prices[:max_eval_rows]
                if eval_features is not None:
                    eval_features = eval_features[:max_eval_rows]
                if eval_cex is not None:
                    eval_cex = eval_cex[:max_eval_rows]
    except Exception:
        pass

    eval_env = create_vec_env(
        n_envs=1,
        seed=config.seed + 10000,
        price_data=eval_prices,
        feature_data=eval_features,
        cex_price_data=eval_cex,
        normalize=True,
    )

    # ── Learning rate ──
    lr = cosine_schedule(config.learning_rate) if use_lr_schedule else config.learning_rate

    # ── Create or load model ──
    if resume_from and Path(resume_from).exists():
        model = PPO.load(
            resume_from,
            env=train_env,
            tensorboard_log=str(config.tensorboard_log_dir),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=config.policy_kwargs,
            tensorboard_log=str(config.tensorboard_log_dir),
            verbose=1,
            seed=config.seed,
            device="auto",
        )

    logger.info(f"Model architecture: {model.policy}")

    # ── Callbacks ──
    checkpoint_dir = config.model_save_dir / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = CallbackList([
        CurriculumCallback(
            phase2_frac=config.curriculum_phase2_frac,
            phase3_frac=config.curriculum_phase3_frac,
        ),
        UniswapMetricsCallback(eval_freq=5000, eval_env=eval_env),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(checkpoint_dir),
            name_prefix=experiment_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best"),
            eval_freq=25_000,
            n_eval_episodes=10,
            deterministic=True,
        ),
    ])

    # ── Train ──
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed / 60:.1f} minutes")

    # ── Save final model ──
    final_path = config.model_save_dir / f"{experiment_name}_final"

    norm_path = config.model_save_dir / f"{experiment_name}_vecnormalize.pkl"
    train_env.save(str(norm_path))
    logger.info(f"VecNormalize: {norm_path}")

    # Cleanup
    train_env.close()
    eval_env.close()
    return model


# ══════════════════════════════════════════════════════════════════════════
#                       EVALUATION / BACKTESTING
# ══════════════════════════════════════════════════════════════════════════

def evaluate_agent(
    model: PPO,
    n_episodes: int = 100,
    price_data: Optional[np.ndarray] = None,
    baseline_fee_bps: float = 30.0,
    vecnormalize_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate trained agent vs static-fee baseline over n_episodes."""
    logger.info(f"Evaluating agent over {n_episodes} episodes")

    eval_features = None
    eval_cex = None

    if price_data is None:
        try:
            tf, tp, tc = DatasetBuilder.load_split("test", config.data_processed_dir)
            if len(tp) > config.episode_length * 2:
                price_data = tp
                eval_features = tf
                eval_cex = tc
                logger.info(f"Using TEST split: {len(tp):,} prices")
                # Subsample if too large
                max_eval = 500_000
                if len(price_data) > max_eval:
                    price_data = price_data[:max_eval]
                    if eval_features is not None:
                        eval_features = eval_features[:max_eval]
                    if eval_cex is not None:
                        eval_cex = eval_cex[:max_eval]
        except Exception:
            pass

    if price_data is None:
        df = generate_synthetic_data(n_records=200_000, volatility=0.02, seed=config.seed + 9999)
        price_data = df["price"].values

    # ── Load VecNormalize stats if available ──
    if vecnormalize_path is None:
        import glob
        candidates = sorted(glob.glob(str(config.model_save_dir / "*_vecnormalize.pkl")))
        if candidates:
            vecnormalize_path = candidates[-1]  # latest
            logger.info(f"Auto-discovered VecNormalize: {vecnormalize_path}")

    agent_rewards = []
    baseline_rewards = []
    agent_summaries = []

    for ep in range(n_episodes):
        agent_env = create_vec_env(
            n_envs=1,
            seed=config.seed + ep + 5000,
            price_data=price_data,
            feature_data=eval_features,
            cex_price_data=eval_cex,
            normalize=True,
        )

        # Phase 3 for eval (full objective)
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = agent_env.step(action)
            total_reward += reward[0]
            done = dones[0]

        agent_rewards.append(total_reward)

        # Try to get episode summary from underlying env
        try:
            base_env = agent_env.envs[0] if hasattr(agent_env, 'envs') else None
            if base_env is None and hasattr(agent_env, 'venv'):
                base_env = agent_env.venv.envs[0]
            if base_env and hasattr(base_env, 'get_episode_summary'):
                agent_summaries.append(base_env.get_episode_summary())
            elif base_env and hasattr(base_env, 'env') and hasattr(base_env.env, 'get_episode_summary'):
                agent_summaries.append(base_env.env.get_episode_summary())
        except Exception:
            pass

        agent_env.close()

        # Baseline episode (static fee)
        env_baseline = UniswapV4Env(
            price_data=price_data,
            feature_data=eval_features,
            cex_price_data=eval_cex,
            seed=config.seed + ep + 5000,
            episode_length=config.episode_length,
            initial_liquidity_usd=config.initial_liquidity_usd,
            fee_min_bps=baseline_fee_bps,
            fee_max_bps=baseline_fee_bps,
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

    # ── Results ──
    agent_mean = np.mean(agent_rewards)
    agent_std = np.std(agent_rewards)
    baseline_mean = np.mean(baseline_rewards)
    baseline_std = np.std(baseline_rewards)

    improvement = ((agent_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)) * 100

    results = {
        "agent_reward_mean": agent_mean,
        "agent_reward_std": agent_std,
        "baseline_reward_mean": baseline_mean,
        "baseline_reward_std": baseline_std,
        "improvement_pct": improvement,
        "n_episodes": n_episodes,
        "agent_summaries": agent_summaries,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Agent    — Mean Reward: {agent_mean:+.4f} ± {agent_std:.4f}")
        logger.info(f"Baseline — Mean Reward: {baseline_mean:+.4f} ± {baseline_std:.4f}")
        logger.info(f"Improvement: {improvement:+.2f}%")
        logger.info("=" * 60)

        if improvement >= 5.0:
            logger.info("✓ MILESTONE: Agent achieves ≥5% improvement over static 0.30% fee!")
        else:
            logger.info(f"✗ Milestone not reached: need ≥5%, got {improvement:.2f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════
#                           MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent for Uniswap V4 dynamic fees")
    parser.add_argument("--timesteps", "--total-timesteps", type=int, default=config.total_timesteps,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=config.n_envs)
    parser.add_argument("--experiment", type=str, default="ppo_uniswap_v4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", type=str, default=None, help="Path to model for eval only")
    parser.add_argument("--vecnormalize", type=str, default=None, help="Path to VecNormalize stats")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.seed is not None:
        config.seed = args.seed

    if args.eval_only:
        model = PPO.load(args.eval_only)
        evaluate_agent(model, n_episodes=args.eval_episodes, vecnormalize_path=args.vecnormalize)
    else:
        model = train_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            experiment_name=args.experiment,
            resume_from=args.resume,
        )
        # Auto-evaluate after training
        evaluate_agent(model, n_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
