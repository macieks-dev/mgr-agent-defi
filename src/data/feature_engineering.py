"""
Feature Engineering — cechy pochodne dla agenta RL.

Implementuje wszystkie cechy wymagane przez promotora:
  1. Realized Volatility (okna przesuwne: 10, 50, 100, 500 bloków)
  2. Order Flow Imbalance (OFI)
  3. CEX Price Deviation (toxic flow signal)
  4. Liquidity features
  5. Gas price normalization
  6. Time features (godzina, dzień tygodnia)

Cechy są normalizowane z-score'em lub do [0, 1] — gotowe dla sieci neuronowej.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#              1. REALIZED VOLATILITY (Multi-Window)
# ══════════════════════════════════════════════════════════════════════════

def compute_realized_volatility(
    df: pd.DataFrame,
    windows: list[int] = (10, 50, 100, 500),
    column: str = "log_return",
) -> pd.DataFrame:
    """
    Zmienność realizowana w wielu oknach przesuwnych.
    Wyższa zmienność → wyższe fee (kompensacja za ryzyko LVR).
    """
    result = df.copy()

    for w in windows:
        col_name = f"realized_vol_{w}"
        result[col_name] = (
            result[column]
            .rolling(window=w, min_periods=1)
            .std()
            .fillna(0)
        )

    # EMA volatility (matching on-chain oracle)
    result["ema_vol"] = (
        result[column].abs()
        .ewm(alpha=0.1, adjust=False)
        .mean()
    )

    # Volatility ratio (short/long) — regime detection
    if 10 in windows and 100 in windows:
        result["vol_ratio_10_100"] = (
            result["realized_vol_10"] / (result["realized_vol_100"] + 1e-10)
        )

    return result


# ══════════════════════════════════════════════════════════════════════════
#              2. ORDER FLOW IMBALANCE (OFI)
# ══════════════════════════════════════════════════════════════════════════

def compute_order_flow_imbalance(
    df: pd.DataFrame,
    windows: list[int] = (10, 50),
) -> pd.DataFrame:
    """
    Order Flow Imbalance = buy_pressure - sell_pressure.
    OFI > 0 → presja kupna → może być toksyczny flow.
    Normalizowany jako z-score.
    """
    result = df.copy()

    if "amount0" in result.columns and "amount1" in result.columns:
        # signed flow: negatywny amount0 = kupno tokena 0
        result["signed_flow"] = -result["amount0"]
    elif "log_return" in result.columns:
        # Proxy: use log returns as directional signal
        result["signed_flow"] = result["log_return"]
    else:
        result["signed_flow"] = 0.0

    for w in windows:
        col_name = f"ofi_{w}"
        raw_ofi = result["signed_flow"].rolling(window=w, min_periods=1).sum()
        # Z-score normalization
        ofi_mean = raw_ofi.rolling(window=w * 5, min_periods=1).mean()
        ofi_std = raw_ofi.rolling(window=w * 5, min_periods=1).std().clip(lower=1e-10)
        result[col_name] = (raw_ofi - ofi_mean) / ofi_std
        result[col_name] = result[col_name].fillna(0).clip(-5, 5)

    return result


# ══════════════════════════════════════════════════════════════════════════
#     3. CEX PRICE DEVIATION (Toxic Flow Signal — klucz do LVR)
# ══════════════════════════════════════════════════════════════════════════

def compute_cex_deviation(
    df: pd.DataFrame,
    cex_price_col: str = "cex_price",
    amm_price_col: str = "price",
) -> pd.DataFrame:
    """
    Odchylenie AMM od CEX = główny wskaźnik Toxic Flow.
    Duże odchylenie → arbitrażyści zaraz wejdą → Hook podnosi fee.
    """
    result = df.copy()

    if cex_price_col not in result.columns:
        logger.warning(f"CEX price column '{cex_price_col}' not found. Skipping deviation.")
        result["cex_deviation_bps"] = 0.0
        result["cex_deviation_abs"] = 0.0
        result["cex_deviation_ema"] = 0.0
        result["cex_deviation_max_10"] = 0.0
        return result

    # Deviation in basis points
    result["cex_deviation_bps"] = (
        (result[amm_price_col] - result[cex_price_col])
        / (result[cex_price_col] + 1e-10)
        * 10_000
    )

    result["cex_deviation_abs"] = result["cex_deviation_bps"].abs()

    # EMA of absolute deviation (trend indicator)
    result["cex_deviation_ema"] = (
        result["cex_deviation_abs"]
        .ewm(alpha=0.1, adjust=False)
        .mean()
    )

    # Max deviation in last 10 blocks (spike detector)
    result["cex_deviation_max_10"] = (
        result["cex_deviation_abs"]
        .rolling(window=10, min_periods=1)
        .max()
    )

    return result


# ══════════════════════════════════════════════════════════════════════════
#              4. LIQUIDITY FEATURES
# ══════════════════════════════════════════════════════════════════════════

def compute_liquidity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Cechy płynności: z-score, zmiana %, współczynnik zmienności."""
    result = df.copy()

    if "liquidity" in result.columns:
        liq = result["liquidity"].astype(float)

        # Normalize to z-score
        liq_mean = liq.rolling(500, min_periods=1).mean()
        liq_std = liq.rolling(500, min_periods=1).std().clip(lower=1e-10)
        result["liquidity_norm"] = ((liq - liq_mean) / liq_std).clip(-5, 5).fillna(0)

        # Change rate
        result["liquidity_change"] = (
            liq.pct_change().fillna(0).clip(-1, 1)
        )

        # Rolling coefficient of variation
        result["liquidity_cv"] = (
            (liq_std / (liq_mean + 1e-10)).clip(0, 5).fillna(0)
        )
    else:
        result["liquidity_norm"] = 0.0
        result["liquidity_change"] = 0.0
        result["liquidity_cv"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════
#              5. PRICE FEATURES (Multi-Scale)
# ══════════════════════════════════════════════════════════════════════════

def compute_price_features(
    df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    """Multi-scale price features: log returns, tick delta, momentum."""
    result = df.copy()
    prices = result[price_col].values

    # Log returns at multiple horizons
    for lag in [1, 5, 20, 100]:
        col = f"log_return_{lag}"
        result[col] = np.log(prices / np.roll(prices, lag))
        result.iloc[:lag, result.columns.get_loc(col)] = 0.0

    # Tick delta (absolute, normalized)
    if "tick" in result.columns:
        ticks = result["tick"].values.astype(float)
        result["tick_delta"] = np.abs(np.diff(ticks, prepend=ticks[0])) / 100.0
    else:
        # Derive tick from price
        ticks = np.floor(np.log(prices) / np.log(1.0001))
        result["tick_delta"] = np.abs(np.diff(ticks, prepend=ticks[0])) / 100.0

    # Price momentum (SMA crossover)
    sma_short = pd.Series(prices).rolling(10, min_periods=1).mean()
    sma_long = pd.Series(prices).rolling(50, min_periods=1).mean()
    result["momentum"] = ((sma_short - sma_long) / (sma_long + 1e-10)).values
    result["momentum"] = result["momentum"].clip(-0.1, 0.1)

    return result


# ══════════════════════════════════════════════════════════════════════════
#              6. TIME FEATURES
# ══════════════════════════════════════════════════════════════════════════

def compute_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Cechy czasowe — kodowanie cykliczne sin/cos (23:59 ≈ 00:00).
    Rynek ETH ma wyraźne wzorce dzienne/tygodniowe.
    """
    result = df.copy()

    if timestamp_col in result.columns:
        ts = pd.to_datetime(result[timestamp_col], unit="s" if result[timestamp_col].dtype != "datetime64[ns, UTC]" else None, utc=True)

        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek

        # Cyclic encoding
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        result["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        result["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        result["hour_sin"] = 0.0
        result["hour_cos"] = 0.0
        result["dow_sin"] = 0.0
        result["dow_cos"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════
#              7. GAS PRICE FEATURES
# ══════════════════════════════════════════════════════════════════════════

def compute_gas_features(
    df: pd.DataFrame,
    gas_col: str = "gas_price_gwei",
) -> pd.DataFrame:
    """Cechy gazu — agent musi wiedzieć, kiedy zmiana fee jest opłacalna."""
    result = df.copy()

    if gas_col in result.columns:
        gas = result[gas_col].astype(float)
        gas_mean = gas.rolling(100, min_periods=1).mean()
        gas_std = gas.rolling(100, min_periods=1).std().clip(lower=1e-10)
        result["gas_norm"] = ((gas - gas_mean) / gas_std).clip(-5, 5).fillna(0)
        result["gas_surge"] = (gas / (gas_mean + 1e-10)).clip(0, 10).fillna(1)
    else:
        result["gas_norm"] = 0.0
        result["gas_surge"] = 1.0

    return result


# ══════════════════════════════════════════════════════════════════════════
#              MASTER PIPELINE — BUILD ALL FEATURES
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
#       FEATURE SETS — based on correlation analysis (Spearman/Pearson)
# ══════════════════════════════════════════════════════════════════════════
#
# Ablation study results (train split, 1M sample, Spearman):
#
# REDUNDANT CLUSTERS (|rho| > 0.7):
#   realized_vol_{10,50,100,500} + ema_vol — all ~0.96 correlated
#   momentum ~ ofi_50 ~ log_return_20 — 0.72-0.75 correlated
#
# CONSTANT/ZERO-VARIANCE (promoted CEX = AMM, synthetic liquidity/gas):
#   cex_deviation_{bps,abs,ema,max_10} — var=0.00 (CEX==AMM)
#   liquidity_{norm,change,cv}          — var=0.00 (synthetic constant)
#   gas_{norm,surge}                    — var=0.00 (no gas data)
#
# Decision: keep 1 representative from each redundancy cluster,
#           drop all constant features.
#
# Reference: Pachamanova & Fabozzi (2023), "Robust Portfolio Optimization"
#   — recommends removing features with |rho| > 0.7 to reduce noise.

# Full 27-dim feature set (for ablation comparison)
FEATURE_COLUMNS_FULL = [
    # Price features
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "log_return_100",
    "tick_delta",
    "momentum",
    # Volatility features
    "realized_vol_10",
    "realized_vol_50",
    "realized_vol_100",
    "realized_vol_500",
    "ema_vol",
    "vol_ratio_10_100",
    # Order flow
    "ofi_10",
    "ofi_50",
    # CEX deviation (toxic flow)
    "cex_deviation_bps",
    "cex_deviation_abs",
    "cex_deviation_ema",
    "cex_deviation_max_10",
    # Liquidity
    "liquidity_norm",
    "liquidity_change",
    "liquidity_cv",
    # Time
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Gas
    "gas_norm",
    "gas_surge",
]

# Reduced 12-dim feature set (post-ablation, non-redundant)
# Removes: 6 redundant (vol_50/100/500, ema_vol, ofi_50, momentum)
#          9 constant (cex_dev×4, liquidity×3, gas×2)
FEATURE_COLUMNS_REDUCED = [
    # Price features (4) — multi-scale returns
    "log_return_1",       # 1-block: immediate signal
    "log_return_5",       # 5-block: short-term trend
    "log_return_20",      # 20-block: medium trend (subsumes momentum)
    "log_return_100",     # 100-block: regime indicator
    # Price microstructure (1)
    "tick_delta",         # absolute tick change — proxy for swap size
    # Volatility (2) — one level + one ratio (sufficient per ablation)
    "realized_vol_10",    # short-term σ (best predictor of next-block LVR)
    "vol_ratio_10_100",   # volatility regime ratio (σ_short/σ_long)
    # Order flow (1)
    "ofi_10",             # 10-block OFI (directional pressure)
    # Time (4) — cyclic encoding of market microstructure patterns
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# ── V3 MINIMAL feature set (post-occlusion ablation) ──
# Advisor demand: "Wyrzucić te 15 cech, które mają 0% wpływu"
# Occlusion ablation (Zeiler & Fergus 2014) showed 15/17 features
# have ~0% contribution → agent learned a trivial fee-tracking policy.
#
# V3 keeps ONLY features with causal connection to LVR/Volume:
#   - realized_vol_10: LVR ∝ σ²L/√k (Milionis et al. 2023) — direct predictor
#   - vol_ratio_10_100: regime detection (rising vol = imminent LVR spike)
#   - log_return_1: immediate price change (next-step LVR proxy)
#
# Runtime features (computed by env, not in this list):
#   - current_fee_norm, lvr_rate_bps, toxic_flow_ratio, epsilon_hat
#
# Total obs_dim = 3 market + 4 runtime = 7
FEATURE_COLUMNS_V3 = [
    "realized_vol_10",    # σ_short: direct LVR predictor
    "vol_ratio_10_100",   # σ_short/σ_long: volatility regime
    "log_return_1",       # immediate price change
]

# V3 indices in the reduced 12-dim feature array
V3_INDICES_IN_REDUCED = [
    FEATURE_COLUMNS_REDUCED.index(f) for f in FEATURE_COLUMNS_V3
]

# Default: use V3 minimal set (backed by occlusion ablation + LVR theory)
FEATURE_COLUMNS = FEATURE_COLUMNS_V3

# Indices mapping: REDUCED feature index → FULL feature index
# Used to extract reduced features from full feature arrays
REDUCED_INDICES_IN_FULL = [
    FEATURE_COLUMNS_FULL.index(f) for f in FEATURE_COLUMNS_REDUCED
]

# Cechy, które agent NIE widzi, ale używamy do reward/LVR
META_COLUMNS = [
    "block_number",
    "timestamp",
    "price",
    "cex_price",
    "tick",
    "liquidity",
    "log_return",
]


def build_all_features(
    onchain_df: pd.DataFrame,
    cex_prices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Master pipeline: on-chain + CEX → pełny wektor cech dla agenta."""
    df = onchain_df.copy()

    # Ensure log_return exists
    if "log_return" not in df.columns and "price" in df.columns:
        df["log_return"] = np.log(df["price"] / df["price"].shift(1)).fillna(0)

    # Add CEX price if provided
    if cex_prices is not None:
        # Align lengths
        n = min(len(df), len(cex_prices))
        df = df.iloc[:n].copy()
        df["cex_price"] = cex_prices[:n]
    elif "cex_price" not in df.columns:
        df["cex_price"] = df.get("price", 0)  # fallback: CEX = AMM (no deviation)

    # Build all feature groups
    logger.info("Computing price features...")
    df = compute_price_features(df)

    logger.info("Computing realized volatility (multi-window)...")
    df = compute_realized_volatility(df)

    logger.info("Computing order flow imbalance...")
    df = compute_order_flow_imbalance(df)

    logger.info("Computing CEX deviation (toxic flow)...")
    df = compute_cex_deviation(df)

    logger.info("Computing liquidity features...")
    df = compute_liquidity_features(df)

    logger.info("Computing time features...")
    df = compute_time_features(df)

    logger.info("Computing gas features...")
    df = compute_gas_features(df)

    # Fill any remaining NaN/inf
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            logger.warning(f"Feature column '{col}' missing, filling with 0")
            df[col] = 0.0

    n_features = len(FEATURE_COLUMNS)
    logger.info(f"Feature engineering complete: {len(df):,} rows × {n_features} features")

    return df
