
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import config
from src.env.lvr import LVRCalculator, ShadowPool

logger = logging.getLogger(__name__)


class UniswapV4Env(gym.Env):

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    MARKET_FEATURES_DIM = 3
    RUNTIME_FEATURES_DIM = 4
    OBS_DIM = MARKET_FEATURES_DIM + RUNTIME_FEATURES_DIM

    SLIPPAGE_TAX_K = 0.05

    EPSILON_PRIOR = -1.5
    EPSILON_DECAY_LAMBDA = 0.15
    EPSILON_CONFIDENCE_FLOOR = 0.01

    def __init__(
        self,
        price_data: Optional[np.ndarray] = None,
        feature_data: Optional[np.ndarray] = None,
        cex_price_data: Optional[np.ndarray] = None,
        episode_length: int = 1000,
        initial_price: float = 2000.0,
        initial_liquidity_usd: float = 1_000_000.0,
        fee_min_bps: float = 1.0,
        fee_max_bps: float = 200.0,
        fee_step_bps: float = 5.0,
        price_elasticity: float = -1.5,
        elasticity_min: float = -0.5,
        elasticity_max: float = -3.0,
        domain_randomize: bool = True,
        volatility_window: int = 100,
        ema_alpha: float = 0.1,
        arb_probability: float = 0.3,
        arb_threshold_bps: float = 5.0,
        gas_penalty_weight: float = 0.001,
        reward_scaling: float = 1.0,
        action_delay_blocks: int = 1,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        self.episode_length = episode_length
        self.initial_price = initial_price
        self.initial_liquidity_usd = initial_liquidity_usd
        self.fee_min_bps = fee_min_bps
        self.fee_max_bps = fee_max_bps
        self.fee_step_bps = fee_step_bps
        self.price_elasticity = price_elasticity
        self.elasticity_min = min(elasticity_min, elasticity_max)
        self.elasticity_max = max(elasticity_min, elasticity_max)
        self.domain_randomize = domain_randomize
        self._episode_elasticity = price_elasticity
        self.volatility_window = volatility_window
        self.ema_alpha = ema_alpha
        self.arb_probability = arb_probability
        self.arb_threshold_bps = arb_threshold_bps
        self.gas_penalty_weight = gas_penalty_weight
        self.reward_scaling = reward_scaling
        self.action_delay_blocks = max(0, action_delay_blocks)

        self.oracle_stale_threshold = 5
        self._cex_stale_count = 0
        self._last_cex_price = None
        self._slippage_tax_active = False
        self._slippage_tax_activations = 0

        self._last_valid_epsilon_raw = self.EPSILON_PRIOR
        self._blocks_since_valid_epsilon = 0

        self._reorg_depth = 0
        self._prev_observation_hash = 0

        self._curriculum_phase = 1
        self._total_episodes_seen = 0

        self._external_price_data = price_data
        self._external_feature_data = feature_data
        self._external_cex_data = cex_price_data
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self._rng: Optional[np.random.Generator] = None
        self._step_count = 0
        self._current_fee_bps = 30.0
        self._pending_fee_buffer: deque[float] = deque()
        self._prices: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None
        self._cex_prices: Optional[np.ndarray] = None
        self._lvr_calc: Optional[LVRCalculator] = None

        self._price_history: list[float] = []
        self._volume_history: list[float] = []
        self._arb_history: list[bool] = []
        self._fee_income_history: list[float] = []
        self._lvr_history: list[float] = []
        self._fee_bps_history: list[float] = []
        self._cumulative_pnl = 0.0
        self._ema_vol = 0.0

        self._baseline_lvr_calc: Optional[LVRCalculator] = None

        if seed is not None:
            self._rng = np.random.default_rng(seed)


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(config.seed)

        self._step_count = 0
        self._current_fee_bps = 30.0

        self._pending_fee_buffer = deque(
            [30.0] * self.action_delay_blocks,
        )

        if self._external_price_data is not None:
            max_start = len(self._external_price_data) - self.episode_length - 1
            if max_start <= 0:
                start = 0
            else:
                start = self._rng.integers(0, max_start)
            self._prices = self._external_price_data[start : start + self.episode_length + 1].copy()

            if self._external_feature_data is not None:
                self._features = self._external_feature_data[start : start + self.episode_length + 1].copy()
            else:
                self._features = None

            if self._external_cex_data is not None:
                self._cex_prices = self._external_cex_data[start : start + self.episode_length + 1].copy()
            else:
                self._cex_prices = None
        else:
            self._prices = self._generate_gbm_prices()
            self._features = None
            self._cex_prices = None

        self._lvr_calc = LVRCalculator()
        self._lvr_calc.initialize(
            price=self._prices[0],
            total_value_y=self.initial_liquidity_usd,
            fee_bps=self._current_fee_bps,
        )

        self._baseline_lvr_calc = LVRCalculator()
        self._baseline_lvr_calc.initialize(
            price=self._prices[0],
            total_value_y=self.initial_liquidity_usd,
            fee_bps=30.0,
        )

        self._price_history = [self._prices[0]]
        self._volume_history = []
        self._arb_history = []
        self._fee_income_history = []
        self._lvr_history = []
        self._fee_bps_history = []
        self._cumulative_pnl = 0.0
        self._ema_vol = 0.0

        self._cex_stale_count = 0
        self._last_cex_price = None
        self._slippage_tax_active = False
        self._slippage_tax_activations = 0
        self._last_valid_epsilon_raw = self.EPSILON_PRIOR
        self._blocks_since_valid_epsilon = 0
        self._reorg_depth = 0
        self._prev_observation_hash = 0

        if self.domain_randomize:
            self._episode_elasticity = float(
                self._rng.uniform(self.elasticity_min, self.elasticity_max)
            )
        else:
            self._episode_elasticity = self.price_elasticity

        obs = self._get_observation()
        info = {"initial_price": self._prices[0], "initial_fee_bps": self._current_fee_bps}

        return obs, info

    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        fee_delta = float(action[0]) * self.fee_step_bps
        old_fee = self._current_fee_bps

        if self.action_delay_blocks > 0:
            desired_fee = np.clip(
                self._current_fee_bps + fee_delta,
                self.fee_min_bps,
                self.fee_max_bps,
            )
            self._pending_fee_buffer.append(desired_fee)
            self._current_fee_bps = self._pending_fee_buffer.popleft()
        else:
            self._current_fee_bps = np.clip(
                self._current_fee_bps + fee_delta,
                self.fee_min_bps,
                self.fee_max_bps,
            )

        self._lvr_calc.update_fee(self._current_fee_bps)

        external_price = self._prices[self._step_count]
        prev_price = self._prices[self._step_count - 1]

        cex_price = None
        if self._cex_prices is not None and self._step_count < len(self._cex_prices):
            cex_price = self._cex_prices[self._step_count]

        if cex_price is not None:
            if self._last_cex_price is not None and abs(cex_price - self._last_cex_price) < 1e-10:
                self._cex_stale_count += 1
            else:
                self._cex_stale_count = 0
                if self._slippage_tax_active:
                    logger.debug("Oracle recovered — deactivating slippage tax")
                    self._slippage_tax_active = False
            self._last_cex_price = cex_price

            if self._cex_stale_count >= self.oracle_stale_threshold and not self._slippage_tax_active:
                self._slippage_tax_active = True
                self._slippage_tax_activations += 1
                logger.debug(
                    f"SLIPPAGE TAX ACTIVATED at step {self._step_count}: "
                    f"CEX price unchanged for {self._cex_stale_count} blocks."
                )

        if self._slippage_tax_active:
            stale_excess = self._cex_stale_count - self.oracle_stale_threshold
            tax_multiplier = np.exp(self.SLIPPAGE_TAX_K * stale_excess)
            taxed_fee = min(self._current_fee_bps * tax_multiplier, self.fee_max_bps)
            self._current_fee_bps = taxed_fee
            self._lvr_calc.update_fee(taxed_fee)

        obs_hash = hash(external_price) ^ hash(self._step_count)
        if obs_hash == self._prev_observation_hash and self._step_count > 1:
            self._reorg_depth = min(self._reorg_depth + 1, 3)
            logger.debug(f"Possible reorg detected at step {self._step_count}, depth={self._reorg_depth}")
        else:
            self._reorg_depth = max(0, self._reorg_depth - 1)
        self._prev_observation_hash = obs_hash

        price_divergence_bps = (
            abs(external_price - self._lvr_calc.shadow_pool.price)
            / self._lvr_calc.shadow_pool.price
            * 10_000
        )
        is_arbitrage = self._determine_arbitrage(price_divergence_bps)

        volume = self._simulate_volume(external_price, is_arbitrage)

        lvr_result = self._lvr_calc.step(
            external_price=external_price,
            is_arbitrage=is_arbitrage,
            block_number=self._step_count,
        )

        if not is_arbitrage and volume > 0:
            noise_fee_income = volume * (self._current_fee_bps / 10_000)
            lvr_result["instantaneous_fee"] += noise_fee_income

        baseline_volume_ref = self.initial_liquidity_usd * 0.01 * (1.0 + self._ema_vol * 100)
        baseline_noise_fee = baseline_volume_ref * (30.0 / 10_000) if not is_arbitrage else 0.0

        self._baseline_lvr_calc.step(
            external_price=external_price,
            is_arbitrage=is_arbitrage,
            block_number=self._step_count,
        )

        self._price_history.append(external_price)
        self._volume_history.append(volume)
        self._arb_history.append(is_arbitrage)
        self._fee_income_history.append(lvr_result["instantaneous_fee"])
        self._lvr_history.append(lvr_result["instantaneous_lvr"])
        self._fee_bps_history.append(self._current_fee_bps)

        log_return = np.log(external_price / prev_price) if prev_price > 0 else 0.0
        self._ema_vol = (
            self.ema_alpha * abs(log_return)
            + (1 - self.ema_alpha) * self._ema_vol
        )

        reward, reward_components = self._compute_reward(lvr_result, volume, old_fee)
        self._cumulative_pnl += reward

        terminated = False
        lp_value = lvr_result["lp_value"]
        if lp_value < self.initial_liquidity_usd * 0.5:
            terminated = True
            reward -= 100.0

        truncated = self._step_count >= self.episode_length

        obs = self._get_observation()
        info = {
            "step": self._step_count,
            "fee_bps": self._current_fee_bps,
            "fee_delta_bps": fee_delta,
            "pending_fee_bps": list(self._pending_fee_buffer),
            "action_delay_blocks": self.action_delay_blocks,
            "external_price": external_price,
            "amm_price": lvr_result["amm_price"],
            "is_arbitrage": is_arbitrage,
            "volume": volume,
            "fee_income": lvr_result["instantaneous_fee"],
            "lvr": lvr_result["instantaneous_lvr"],
            "net_lvr": lvr_result.get("net_lvr_instantaneous", 0.0),
            "lp_value": lp_value,
            "cumulative_pnl": self._cumulative_pnl,
            "reward_components": reward_components,
            "baseline_lp_value": self._baseline_lvr_calc.shadow_pool.total_value_y,
            "baseline_cumulative_fee": self._baseline_lvr_calc.shadow_pool.cumulative_fee_income,
            "slippage_tax_active": self._slippage_tax_active,
            "slippage_tax_activations": self._slippage_tax_activations,
            "oracle_stale_count": self._cex_stale_count,
            "reorg_depth": self._reorg_depth,
        }

        return obs, reward, terminated, truncated, info


    def _get_observation(self) -> np.ndarray:
        n = len(self._price_history)

        if self._features is not None and self._step_count < len(self._features):
            raw = self._features[self._step_count]
            if raw.shape[0] == self.MARKET_FEATURES_DIM:
                market_features = raw.copy()
            elif raw.shape[0] == 12:
                from src.data.feature_engineering import V3_INDICES_IN_REDUCED
                market_features = raw[V3_INDICES_IN_REDUCED].copy()
            else:
                from src.data.feature_engineering import REDUCED_INDICES_IN_FULL, V3_INDICES_IN_REDUCED
                reduced = raw[REDUCED_INDICES_IN_FULL]
                market_features = reduced[V3_INDICES_IN_REDUCED].copy()
        else:
            market_features = self._compute_market_features_from_history()

        fee_norm = (self._current_fee_bps - self.fee_min_bps) / (
            self.fee_max_bps - self.fee_min_bps + 1e-8
        )

        if len(self._lvr_history) >= 1:
            recent_lvr = np.sum(self._lvr_history[-10:])
            lvr_rate = (recent_lvr / (self.initial_liquidity_usd + 1e-8)) * 10_000
        else:
            lvr_rate = 0.0

        if len(self._arb_history) >= 1:
            recent_arb = self._arb_history[-20:]
            toxic_ratio = sum(recent_arb) / (len(recent_arb) + 1e-8)
        else:
            toxic_ratio = 0.0

        epsilon_hat = self._estimate_epsilon()

        runtime_features = np.array(
            [fee_norm, lvr_rate, toxic_ratio, epsilon_hat],
            dtype=np.float32,
        )

        obs = np.concatenate([market_features, runtime_features])
        return obs

    def _estimate_epsilon(self) -> float:
        n_vol = len(self._volume_history)
        n_fee = len(self._fee_bps_history)
        min_n = min(n_vol, n_fee)

        if min_n < 2:
            self._blocks_since_valid_epsilon += 1
            return self._apply_epsilon_decay()

        lookback = min(10, min_n - 1)

        elasticity_samples = []

        for i in range(1, lookback + 1):
            vol_cur = self._volume_history[-(lookback + 1 - i + 1)]
            vol_next = self._volume_history[-(lookback + 1 - i)]
            fee_cur = self._fee_bps_history[-(lookback + 1 - i + 1)]
            fee_next = self._fee_bps_history[-(lookback + 1 - i)]

            if vol_cur <= 0 or vol_next <= 0 or fee_cur <= 0 or fee_next <= 0:
                continue

            d_ln_fee = np.log(fee_next) - np.log(fee_cur)

            if abs(d_ln_fee) < 1e-6:
                continue

            d_ln_vol = np.log(vol_next) - np.log(vol_cur)
            epsilon_i = d_ln_vol / d_ln_fee
            elasticity_samples.append(epsilon_i)

        if len(elasticity_samples) < 1:
            self._blocks_since_valid_epsilon += 1
            return self._apply_epsilon_decay()

        epsilon_raw = float(np.mean(elasticity_samples))
        epsilon_clipped = np.clip(epsilon_raw, -5.0, 0.0)

        self._last_valid_epsilon_raw = epsilon_clipped
        self._blocks_since_valid_epsilon = 0

        return epsilon_clipped / 5.0

    def _apply_epsilon_decay(self) -> float:
        dt = self._blocks_since_valid_epsilon
        confidence = np.exp(-self.EPSILON_DECAY_LAMBDA * dt)

        if confidence < self.EPSILON_CONFIDENCE_FLOOR:
            confidence = 0.0

        epsilon_decayed = (
            self.EPSILON_PRIOR
            + (self._last_valid_epsilon_raw - self.EPSILON_PRIOR) * confidence
        )

        epsilon_clipped = np.clip(epsilon_decayed, -5.0, 0.0)
        return epsilon_clipped / 5.0

    def _compute_market_features_from_history(self) -> np.ndarray:
        n = len(self._price_history)
        features = np.zeros(self.MARKET_FEATURES_DIM, dtype=np.float32)

        if n >= 11:
            returns = np.diff(np.log(np.array(self._price_history[-11:])))
            features[0] = np.std(returns) if len(returns) > 0 else 0.0

        if n >= 101:
            returns_100 = np.diff(np.log(np.array(self._price_history[-101:])))
            sigma_100 = np.std(returns_100) if len(returns_100) > 0 else 0.0
            if sigma_100 > 1e-10:
                features[1] = features[0] / sigma_100
            else:
                features[1] = 1.0
        else:
            features[1] = 1.0

        if n >= 2:
            features[2] = np.log(self._price_history[-1] / self._price_history[-2])

        return features

    
    def set_curriculum_phase(self, phase: int):
        self._curriculum_phase = max(1, min(3, phase))

    def _compute_reward(
        self,
        lvr_result: dict,
        volume: float,
        old_fee_bps: float,
    ) -> tuple[float, dict]:
        norm = self.initial_liquidity_usd + 1e-8
        phase = self._curriculum_phase

        fee_income_bps = (lvr_result["instantaneous_fee"] / norm) * 10_000

        lvr_bps = (lvr_result["instantaneous_lvr"] / norm) * 10_000

        base_volume = self.initial_liquidity_usd * 0.01
        vol_multiplier = 1.0 + self._ema_vol * 100
        v_baseline = base_volume * vol_multiplier

        if v_baseline > 1e-8:
            volume_drop = max(0.0, 1.0 - volume / v_baseline)
        else:
            volume_drop = 0.0

        reward_mult = fee_income_bps * (1.0 - volume_drop)

        fee_change = abs(self._current_fee_bps - old_fee_bps)
        churn_penalty = 0.0
        if self.fee_step_bps > 0 and fee_change > self.fee_step_bps * 0.5:
            if phase == 1:
                churn_penalty = 0.05 * (fee_change / self.fee_step_bps) ** 2
            elif phase == 2:
                churn_penalty = 0.005 * (fee_change / self.fee_step_bps) ** 2

        if phase == 1:
            reward = reward_mult - churn_penalty
        elif phase == 2:
            reward = reward_mult - 0.3 * lvr_bps - churn_penalty
        else:
            reward = reward_mult - lvr_bps

        reward *= self.reward_scaling

        components = {
            "fee_income_bps": fee_income_bps,
            "lvr_bps": lvr_bps,
            "volume_drop_penalty": volume_drop,
            "reward_multiplicative": reward_mult,
            "churn_penalty": churn_penalty,
            "curriculum_phase": phase,
            "episode_elasticity": self._episode_elasticity,
            "raw_reward": reward,
        }

        return reward, components


    def _generate_gbm_prices(self) -> np.ndarray:
        n = self.episode_length + 1
        prices = np.zeros(n)
        prices[0] = self.initial_price

        base_vol = 0.01
        stress_vol = 0.04

        for i in range(1, n):
            if self._rng.random() < 0.05:
                vol = stress_vol
            else:
                vol = base_vol

            log_return = self._rng.normal(0, vol)
            prices[i] = prices[i - 1] * np.exp(log_return)

        return prices

    def _determine_arbitrage(self, price_divergence_bps: float) -> bool:
        if price_divergence_bps <= self.arb_threshold_bps:
            return False

        if price_divergence_bps <= self._current_fee_bps:
            return False

        margin = (price_divergence_bps - self._current_fee_bps) / price_divergence_bps
        prob = self.arb_probability * margin

        return self._rng.random() < prob

    def _simulate_volume(
        self, external_price: float, is_arbitrage: bool
    ) -> float:
        base_volume = self.initial_liquidity_usd * 0.01
        vol_multiplier = 1.0 + self._ema_vol * 100

        if is_arbitrage:
            price_div = abs(external_price - self._lvr_calc.shadow_pool.price)
            arb_volume = price_div * self._lvr_calc.shadow_pool.reserve_x
            return arb_volume * vol_multiplier

        fee_ref = 30.0
        fee_ratio = max(self._current_fee_bps, 0.1) / fee_ref

        elastic_volume = base_volume * (fee_ratio ** self._episode_elasticity)

        fee_excess = self._current_fee_bps - 30.0
        routing_leakage = 0.30 / (1.0 + np.exp(-(fee_excess - 20.0) / 20.0))
        elastic_volume *= (1.0 - routing_leakage)

        elastic_volume = min(elastic_volume, base_volume * 50.0)

        noise = self._rng.lognormal(0, 0.3)
        elastic_volume *= max(0.1, noise)

        return elastic_volume * vol_multiplier

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()

    def _render_ansi(self) -> str:
        n = self._step_count
        price = self._price_history[-1] if self._price_history else 0
        fee = self._current_fee_bps
        pnl = self._cumulative_pnl

        return (
            f"Step {n:4d}/{self.episode_length} | "
            f"Price: {price:8.2f} | "
            f"Fee: {fee:6.1f} bps | "
            f"PnL: {pnl:+8.2f}"
        )

    def _render_human(self):
        print(self._render_ansi())

    
    def get_episode_summary(self) -> dict:
        if self._lvr_calc is None:
            return {}

        agent_summary = self._lvr_calc.get_summary()
        baseline_summary = self._baseline_lvr_calc.get_summary() if self._baseline_lvr_calc else {}

        return {
            "agent": agent_summary,
            "baseline": baseline_summary,
            "cumulative_pnl": self._cumulative_pnl,
            "total_steps": self._step_count,
            "final_fee_bps": self._current_fee_bps,
            "action_delay_blocks": self.action_delay_blocks,
            "mean_fee_income": np.mean([h for h in self._fee_income_history]) if self._fee_income_history else 0,
            "mean_fee_bps": np.mean(self._fee_bps_history) if self._fee_bps_history else 30.0,
            "total_arb_swaps": sum(self._arb_history),
            "arb_fraction": sum(self._arb_history) / (len(self._arb_history) + 1e-8),
            "improvement_over_baseline_pct": (
                (agent_summary.get("lp_return_pct", 0) - baseline_summary.get("lp_return_pct", 0))
                if baseline_summary else 0
            ),
            "slippage_tax_activations": self._slippage_tax_activations,
            "slippage_tax_active_at_end": self._slippage_tax_active,
        }
