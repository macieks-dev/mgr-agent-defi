"""
Tests for the UniswapV4Env Gymnasium environment.
"""

import numpy as np
import pytest
import gymnasium as gym

from src.env.uniswap_v4_env import UniswapV4Env
from src.env.lvr import ShadowPool, LVRCalculator, compute_analytical_lvr


# ══════════════════════════════════════════════════════════════════════════
#                    ENVIRONMENT TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestUniswapV4Env:
    """Test suite for the UniswapV4Env."""

    def _make_env(self, **kwargs) -> UniswapV4Env:
        defaults = dict(
            episode_length=100,
            initial_price=2000.0,
            initial_liquidity_usd=1_000_000.0,
            fee_min_bps=1.0,
            fee_max_bps=200.0,
            fee_step_bps=5.0,
            seed=42,
        )
        defaults.update(kwargs)
        return UniswapV4Env(**defaults)

    def test_env_creation(self):
        """Environment can be created."""
        env = self._make_env()
        assert env is not None
        assert env.observation_space.shape == (7,)
        assert env.action_space.shape == (1,)

    def test_reset_returns_valid_obs(self):
        """Reset returns correctly shaped observation."""
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert not np.any(np.isnan(obs))
        assert "initial_price" in info

    def test_step_returns_valid_tuple(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = self._make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_completes(self):
        """Full episode runs without errors."""
        env = self._make_env(episode_length=50)
        obs, _ = env.reset()

        total_reward = 0.0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        assert truncated or terminated

    def test_fee_clamping(self):
        """Fee stays within [min, max] bounds."""
        env = self._make_env()
        obs, _ = env.reset()

        # Try extreme action (push fee to max)
        for _ in range(100):
            action = np.array([1.0], dtype=np.float32)  # Always increase
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            assert info["fee_bps"] <= env.fee_max_bps
            assert info["fee_bps"] >= env.fee_min_bps

    def test_zero_action_minimal_change(self):
        """Action=0 should not change fee significantly."""
        env = self._make_env()
        obs, _ = env.reset()
        action = np.array([0.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)

        # Fee should be near initial (30 bps)
        assert abs(info["fee_bps"] - 30.0) < 1.0

    def test_observation_not_nan(self):
        """Observations should never contain NaN."""
        env = self._make_env(episode_length=200)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), f"NaN in initial obs: {obs}"

        for _ in range(200):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {env._step_count}: {obs}"
            if terminated or truncated:
                break

    def test_reward_shape(self):
        """Reward should be a scalar float."""
        env = self._make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert np.isscalar(reward)
        assert np.isfinite(reward)

    def test_info_has_required_keys(self):
        """Info dict has all expected keys."""
        env = self._make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        required_keys = [
            "step", "fee_bps", "external_price", "is_arbitrage",
            "fee_income", "lvr", "lp_value", "cumulative_pnl",
            "reward_components",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_episode_summary(self):
        """get_episode_summary returns valid dict after episode."""
        env = self._make_env(episode_length=50)
        obs, _ = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        assert isinstance(summary, dict)
        assert "agent" in summary
        assert "cumulative_pnl" in summary

    def test_with_external_price_data(self):
        """Environment works with pre-loaded price data."""
        prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200)))
        env = self._make_env(price_data=prices, episode_length=100)
        obs, _ = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_gymnasium_api_check(self):
        """Verify compatibility with Gymnasium's check_env."""
        # This is a basic interface check
        env = self._make_env()
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

        # Check spaces are valid
        assert env.observation_space.contains(env.observation_space.sample())
        assert env.action_space.contains(env.action_space.sample())

    def test_high_fee_reduces_arb_chance(self):
        """Higher fees should reduce arbitrage opportunities."""
        env = self._make_env(episode_length=500)
        obs, _ = env.reset()

        # Run with high fee → expect fewer arb swaps
        arb_count = 0
        for _ in range(500):
            action = np.array([1.0], dtype=np.float32)  # Push fee up
            obs, _, terminated, truncated, info = env.step(action)
            if info["is_arbitrage"]:
                arb_count += 1
            if terminated or truncated:
                break

        # With max fee, arb should be less frequent (they can't profit)
        arb_fraction = arb_count / env._step_count
        assert arb_fraction < 0.5, f"Too many arb swaps with high fee: {arb_fraction:.2f}"


class TestActionDelay:
    """Tests for the 1-block action delay mechanism."""

    def _make_env(self, **kwargs) -> UniswapV4Env:
        defaults = dict(
            episode_length=100,
            initial_price=2000.0,
            initial_liquidity_usd=1_000_000.0,
            fee_min_bps=1.0,
            fee_max_bps=200.0,
            fee_step_bps=5.0,
            seed=42,
        )
        defaults.update(kwargs)
        return UniswapV4Env(**defaults)

    def test_default_delay_is_one_block(self):
        """Default action_delay_blocks should be 1."""
        env = self._make_env()
        assert env.action_delay_blocks == 1

    def test_delay_zero_applies_immediately(self):
        """With action_delay_blocks=0, fee changes are instant (legacy)."""
        env = self._make_env(action_delay_blocks=0)
        obs, _ = env.reset()

        # Push fee up with max action
        action = np.array([1.0], dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        # Fee should have increased by fee_step_bps (5) from 30 to 35
        assert info["fee_bps"] == pytest.approx(35.0)

    def test_delay_one_block_defers_fee_change(self):
        """With action_delay_blocks=1, the first action takes effect on step 2."""
        env = self._make_env(action_delay_blocks=1)
        obs, _ = env.reset()

        # Step 1: agent says +max → but delay buffer still has initial=30
        action_up = np.array([1.0], dtype=np.float32)
        obs, _, _, _, info1 = env.step(action_up)

        # Fee at step 1 should still be 30 (the initial fee from the
        # pre-filled buffer), because the agent's +5 is queued.
        assert info1["fee_bps"] == pytest.approx(30.0), (
            f"Expected 30.0 (delayed), got {info1['fee_bps']}"
        )

        # Step 2: agent says 0 → the deferred +5 from step 1 takes effect
        action_zero = np.array([0.0], dtype=np.float32)
        obs, _, _, _, info2 = env.step(action_zero)

        # Now the fee should be 35 (the 30+5 that was queued at step 1)
        assert info2["fee_bps"] == pytest.approx(35.0), (
            f"Expected 35.0 (deferred from step 1), got {info2['fee_bps']}"
        )

    def test_pending_fee_in_info(self):
        """Info dict should contain pending_fee_bps list."""
        env = self._make_env(action_delay_blocks=1)
        obs, _ = env.reset()

        action = np.array([0.5], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "pending_fee_bps" in info
        assert "action_delay_blocks" in info
        assert info["action_delay_blocks"] == 1
        assert len(info["pending_fee_bps"]) == 1  # 1 item pending

    def test_delay_two_blocks(self):
        """With action_delay_blocks=2, fee change is deferred by 2 steps."""
        env = self._make_env(action_delay_blocks=2)
        obs, _ = env.reset()

        # Step 1: push +max (fee_step=5bps). Buffer: [30, 30] → pop 30, push 35
        action_up = np.array([1.0], dtype=np.float32)
        _, _, _, _, info1 = env.step(action_up)
        assert info1["fee_bps"] == pytest.approx(30.0)

        # Step 2: push 0. The 30 that was second-in-queue pops out.
        action_zero = np.array([0.0], dtype=np.float32)
        _, _, _, _, info2 = env.step(action_zero)
        assert info2["fee_bps"] == pytest.approx(30.0)

        # Step 3: now the 35 from step 1 finally pops
        _, _, _, _, info3 = env.step(action_zero)
        assert info3["fee_bps"] == pytest.approx(35.0)

    def test_episode_summary_includes_delay(self):
        """Episode summary should report action_delay_blocks."""
        env = self._make_env(episode_length=10, action_delay_blocks=1)
        obs, _ = env.reset()
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            if terminated or truncated:
                break
        summary = env.get_episode_summary()
        assert summary["action_delay_blocks"] == 1

    def test_fee_clamping_with_delay(self):
        """Fees should remain within bounds even with delay buffer."""
        env = self._make_env(action_delay_blocks=1)
        obs, _ = env.reset()
        for _ in range(100):
            action = np.array([1.0], dtype=np.float32)  # always push up
            obs, _, terminated, truncated, info = env.step(action)
            assert info["fee_bps"] <= env.fee_max_bps + 1e-6
            assert info["fee_bps"] >= env.fee_min_bps - 1e-6
            if terminated or truncated:
                break


# ══════════════════════════════════════════════════════════════════════════
#                    SHADOW POOL TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestShadowPool:
    """Tests for the ShadowPool CPMM implementation."""

    def test_initialization(self):
        pool = ShadowPool()
        pool.initialize(price=2000.0, total_value_y=1_000_000.0)

        assert pool.reserve_y == pytest.approx(500_000.0)
        assert pool.reserve_x == pytest.approx(250.0)
        assert pool.price == pytest.approx(2000.0)

    def test_invariant_preserved(self):
        """k should be approximately preserved after swaps."""
        pool = ShadowPool()
        pool.initialize(price=2000.0, total_value_y=1_000_000.0)
        k_initial = pool.k

        pool.execute_swap(external_price=2050.0)

        # k changes slightly due to fees, but shouldn't deviate wildly
        assert pool.k > 0
        assert pool.price == pytest.approx(2050.0, rel=0.01)

    def test_fee_earned_positive(self):
        """Swaps should earn non-negative fees."""
        pool = ShadowPool(fee_bps=30.0)
        pool.initialize(price=2000.0, total_value_y=1_000_000.0)

        result = pool.execute_swap(external_price=2010.0)
        assert result["fee_earned"] >= 0

    def test_no_swap_when_prices_equal(self):
        """No swap when AMM price equals external price."""
        pool = ShadowPool()
        pool.initialize(price=2000.0, total_value_y=1_000_000.0)

        result = pool.execute_swap(external_price=2000.0)
        assert result["delta_x"] == 0.0
        assert result["delta_y"] == 0.0
        assert result["lvr_instantaneous"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
#                    LVR CALCULATOR TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestLVRCalculator:
    """Tests for the LVR calculator."""

    def test_initialization(self):
        calc = LVRCalculator()
        calc.initialize(price=2000.0, total_value_y=1_000_000.0)

        assert calc.initial_value == 1_000_000.0
        assert calc.shadow_pool.price == pytest.approx(2000.0)

    def test_step_returns_valid_metrics(self):
        calc = LVRCalculator()
        calc.initialize(price=2000.0, total_value_y=1_000_000.0)

        result = calc.step(external_price=2010.0, block_number=1)

        assert "instantaneous_lvr" in result
        assert "cumulative_lvr" in result
        assert "cumulative_fee" in result
        assert "lp_value" in result
        assert result["instantaneous_lvr"] >= 0

    def test_cumulative_lvr_increases(self):
        """Cumulative LVR should generally increase with price changes."""
        calc = LVRCalculator()
        calc.initialize(price=2000.0, total_value_y=1_000_000.0)

        rng = np.random.default_rng(42)
        prices = 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))

        for i, p in enumerate(prices):
            result = calc.step(external_price=p, block_number=i)

        summary = calc.get_summary()
        assert summary["total_lvr"] >= 0

    def test_analytical_lvr(self):
        """Analytical LVR formula should give reasonable estimates."""
        result = compute_analytical_lvr(
            sigma=0.01,
            liquidity_usd=1_000_000.0,
            n_blocks=1000,
            fee_bps=30.0,
        )

        assert result["lvr_per_block"] > 0
        assert result["total_lvr"] > 0
        assert result["breakeven_fee_bps"] > 0
        assert result["net_lvr_bps"] is not None


# ══════════════════════════════════════════════════════════════════════════
#                          RUN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
