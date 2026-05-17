"""
LVR (Loss-Versus-Rebalancing) engine — Milionis et al. (2022).

ShadowPool: virtual CPMM tracking LP value with fee income.
LVRCalculator: instantaneous + cumulative LVR = Rebalancing − LP value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#                       SHADOW POOL (CPMM)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ShadowPool:
    """Virtual CPMM (x·y = k) tracking LP reserves, fees, and impermanent loss."""

    reserve_x: float = 0.0
    reserve_y: float = 0.0
    k: float = 0.0
    fee_bps: float = 30.0
    cumulative_fee_income: float = 0.0
    cumulative_il: float = 0.0

    def initialize(self, price: float, total_value_y: float):
        """Set reserves for CPMM at given price: x = V/(2P), y = V/2."""
        self.reserve_y = total_value_y / 2.0
        self.reserve_x = total_value_y / (2.0 * price)
        self.k = self.reserve_x * self.reserve_y
        logger.debug(
            f"ShadowPool initialized: x={self.reserve_x:.4f}, "
            f"y={self.reserve_y:.2f}, k={self.k:.2f}, price={price:.2f}"
        )

    @property
    def price(self) -> float:
        """AMM price (Y per X) = reserve_y / reserve_x."""
        if self.reserve_x <= 0:
            return float("inf")
        return self.reserve_y / self.reserve_x

    @property
    def total_value_y(self) -> float:
        return self.reserve_x * self.price + self.reserve_y

    def compute_reserves_at_price(self, target_price: float) -> tuple[float, float]:
        """CPMM reserves at target_price: x = sqrt(k/P), y = sqrt(k*P)."""
        if target_price <= 0:
            raise ValueError(f"Price must be positive, got {target_price}")

        new_x = np.sqrt(self.k / target_price)
        new_y = np.sqrt(self.k * target_price)
        return new_x, new_y

    def execute_swap(
        self,
        external_price: float,
        is_arbitrage: bool = False,
    ) -> dict:
        """Move pool to external_price; compute fee income and instantaneous LVR."""
        amm_price_before = self.price

        if abs(amm_price_before - external_price) / external_price < 1e-8:
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "fee_earned": 0.0,
                "lvr_instantaneous": 0.0,
                "lvr_net": 0.0,
                "price_impact": 0.0,
                "amm_price_before": amm_price_before,
                "amm_price_after": amm_price_before,
            }

        # New reserves at external price
        new_x, new_y = self.compute_reserves_at_price(external_price)

        delta_x = new_x - self.reserve_x
        delta_y = new_y - self.reserve_y

        fee_rate = self.fee_bps / 10_000.0

        if delta_y > 0:
            fee_earned = abs(delta_y) * fee_rate
        else:
            fee_earned = abs(delta_x) * external_price * fee_rate

        # LVR = value arber extracts (Milionis et al.)
        if delta_x < 0:
            value_given_at_external = abs(delta_x) * external_price
            value_received_from_arber = abs(delta_y)
        else:
            value_given_at_external = abs(delta_y)
            value_received_from_arber = abs(delta_x) * external_price

        lvr_instantaneous = max(0.0, value_given_at_external - value_received_from_arber)

        # Net LVR after fees
        lvr_net = max(0.0, lvr_instantaneous - fee_earned)

        self.reserve_x = new_x
        self.reserve_y = new_y
        self.k = self.reserve_x * self.reserve_y

        self.cumulative_fee_income += fee_earned
        self.cumulative_il += lvr_net

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "fee_earned": fee_earned,
            "lvr_instantaneous": lvr_instantaneous,
            "lvr_net": lvr_net,
            "price_impact": external_price - amm_price_before,
            "amm_price_before": amm_price_before,
            "amm_price_after": self.price,
        }


# ══════════════════════════════════════════════════════════════════════════
#                       LVR CALCULATOR
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LVRCalculator:
    """Tracks LP portfolio vs rebalancing portfolio; computes instantaneous and cumulative LVR."""

    shadow_pool: ShadowPool = field(default_factory=ShadowPool)
    initial_value: float = 0.0
    initial_price: float = 0.0
    rebalancing_value: float = 0.0
    rebalancing_x: float = 0.0
    rebalancing_y: float = 0.0
    history: list = field(default_factory=list)

    def initialize(self, price: float, total_value_y: float, fee_bps: float = 30.0):
        """Init LP pool + rebalancing portfolio at given price and liquidity."""
        self.shadow_pool.fee_bps = fee_bps
        self.shadow_pool.initialize(price, total_value_y)

        self.initial_value = total_value_y
        self.initial_price = price
        self.rebalancing_value = total_value_y
        self.rebalancing_x = total_value_y / (2.0 * price)
        self.rebalancing_y = total_value_y / 2.0

        self.history = []

    def update_fee(self, new_fee_bps: float):
        self.shadow_pool.fee_bps = new_fee_bps

    def step(
        self,
        external_price: float,
        is_arbitrage: bool = False,
        block_number: int = 0,
    ) -> dict:
        """Process one step: swap in shadow pool, update rebalancing portfolio, compute LVR."""
        # 1. Execute swap in shadow pool
        swap_result = self.shadow_pool.execute_swap(
            external_price, is_arbitrage=is_arbitrage
        )

        # 2. Rebalancing portfolio value
        self.rebalancing_value = (
            self.rebalancing_x * external_price + self.rebalancing_y
        )

        lp_value = self.shadow_pool.total_value_y

        # LVR = rebalancing − LP (before fees); net = LVR − cumulative fees
        cumulative_lvr = self.rebalancing_value - lp_value
        cumulative_fee = self.shadow_pool.cumulative_fee_income
        net_lvr = cumulative_lvr - cumulative_fee

        lvr_bps = (swap_result["lvr_instantaneous"] / self.initial_value) * 10_000 if self.initial_value > 0 else 0.0
        net_lvr_bps = (net_lvr / self.initial_value) * 10_000 if self.initial_value > 0 else 0.0

        fee_bps_earned = (swap_result["fee_earned"] / self.initial_value) * 10_000 if self.initial_value > 0 else 0.0

        result = {
            "block_number": block_number,
            "external_price": external_price,
            "amm_price": self.shadow_pool.price,
            "price_divergence": abs(self.shadow_pool.price - external_price) / external_price,
            "lp_value": lp_value,
            "rebalancing_value": self.rebalancing_value,
            "instantaneous_lvr": swap_result["lvr_instantaneous"],
            "instantaneous_lvr_bps": lvr_bps,
            "instantaneous_fee": swap_result["fee_earned"],
            "instantaneous_fee_bps": fee_bps_earned,
            "net_lvr_instantaneous": swap_result.get("lvr_net", 0.0),
            "cumulative_lvr": cumulative_lvr,
            "cumulative_fee": cumulative_fee,
            "net_cumulative_lvr": net_lvr,
            "net_cumulative_lvr_bps": net_lvr_bps,
            "fee_bps": self.shadow_pool.fee_bps,
            "is_arbitrage": is_arbitrage,
        }

        self.history.append(result)
        return result

    def get_summary(self) -> dict:
        """Summary stats for the LVR tracking session."""
        if not self.history:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.history)

        return {
            "total_steps": len(df),
            "final_lp_value": df["lp_value"].iloc[-1],
            "final_rebalancing_value": df["rebalancing_value"].iloc[-1],
            "total_lvr": df["cumulative_lvr"].iloc[-1],
            "total_fees": df["cumulative_fee"].iloc[-1],
            "net_lvr": df["net_cumulative_lvr"].iloc[-1],
            "net_lvr_bps": df["net_cumulative_lvr_bps"].iloc[-1],
            "mean_instantaneous_lvr_bps": df["instantaneous_lvr_bps"].mean(),
            "mean_fee_bps_earned": df["instantaneous_fee_bps"].mean(),
            "arb_fraction": df["is_arbitrage"].mean(),
            "lp_return_pct": (
                (df["lp_value"].iloc[-1] - self.initial_value) / self.initial_value * 100
            ),
            "rebal_return_pct": (
                (df["rebalancing_value"].iloc[-1] - self.initial_value) / self.initial_value * 100
            ),
        }

    def reset(self):
        self.history = []
        self.shadow_pool.cumulative_fee_income = 0.0
        self.shadow_pool.cumulative_il = 0.0


# ══════════════════════════════════════════════════════════════════════════
#                    ANALYTICAL LVR FORMULA
# ══════════════════════════════════════════════════════════════════════════

def compute_analytical_lvr(
    sigma: float,
    liquidity_usd: float,
    n_blocks: int,
    fee_bps: float = 30.0,
) -> dict:
    """
    Analytical LVR via Milionis formula: E[LVR/block] ≈ (σ²/8) · L.
    Returns total LVR, estimated fee income, and break-even fee.
    """
    lvr_per_block = (sigma**2 / 8.0) * liquidity_usd
    total_lvr = lvr_per_block * n_blocks

    # Rough fee estimate: volume ≈ σ × L
    estimated_volume_per_block = sigma * liquidity_usd
    fee_rate = fee_bps / 10_000.0
    fee_per_block = estimated_volume_per_block * fee_rate
    total_fee = fee_per_block * n_blocks

    net_lvr = total_lvr - total_fee

    return {
        "sigma": sigma,
        "liquidity_usd": liquidity_usd,
        "n_blocks": n_blocks,
        "fee_bps": fee_bps,
        "lvr_per_block": lvr_per_block,
        "total_lvr": total_lvr,
        "fee_per_block": fee_per_block,
        "total_fee": total_fee,
        "net_lvr": net_lvr,
        "lvr_bps": (total_lvr / liquidity_usd) * 10_000,
        "fee_bps_earned": (total_fee / liquidity_usd) * 10_000,
        "net_lvr_bps": (net_lvr / liquidity_usd) * 10_000,
        "breakeven_fee_bps": (lvr_per_block / estimated_volume_per_block) * 10_000 if estimated_volume_per_block > 0 else float("inf"),
    }
