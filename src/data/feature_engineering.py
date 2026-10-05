
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_realized_volatility(
    df: pd.DataFrame,
    windows: list[int] = (10, 50, 100, 500),
    column: str = "log_return",
) -> pd.DataFrame:
    result = df.copy()

    for w in windows:
        col_name = f"realized_vol_{w}"
        result[col_name] = (
            result[column]
            .rolling(window=w, min_periods=1)
            .std()
            .fillna(0)
        )

    result["ema_vol"] = (
        result[column].abs()
        .ewm(alpha=0.1, adjust=False)
        .mean()
    )

    if 10 in windows and 100 in windows:
        result["vol_ratio_10_100"] = (
            result["realized_vol_10"] / (result["realized_vol_100"] + 1e-10)
        )

    return result


def compute_order_flow_imbalance(
    df: pd.DataFrame,
    windows: list[int] = (10, 50),
) -> pd.DataFrame:
    result = df.copy()

    if "amount0" in result.columns and "amount1" in result.columns:
        result["signed_flow"] = -result["amount0"]
    elif "log_return" in result.columns:
        result["signed_flow"] = result["log_return"]
    else:
        result["signed_flow"] = 0.0

    for w in windows:
        col_name = f"ofi_{w}"
        raw_ofi = result["signed_flow"].rolling(window=w, min_periods=1).sum()
        ofi_mean = raw_ofi.rolling(window=w * 5, min_periods=1).mean()
        ofi_std = raw_ofi.rolling(window=w * 5, min_periods=1).std().clip(lower=1e-10)
        result[col_name] = (raw_ofi - ofi_mean) / ofi_std
        result[col_name] = result[col_name].fillna(0).clip(-5, 5)

    return result


def compute_cex_deviation(
    df: pd.DataFrame,
    cex_price_col: str = "cex_price",
    amm_price_col: str = "price",
) -> pd.DataFrame:
    result = df.copy()

    if cex_price_col not in result.columns:
        logger.warning(f"CEX price column '{cex_price_col}' not found. Skipping deviation.")
        result["cex_deviation_bps"] = 0.0
        result["cex_deviation_abs"] = 0.0
        result["cex_deviation_ema"] = 0.0
        result["cex_deviation_max_10"] = 0.0
        return result

    result["cex_deviation_bps"] = (
        (result[amm_price_col] - result[cex_price_col])
        / (result[cex_price_col] + 1e-10)
        * 10_000
    )

    result["cex_deviation_abs"] = result["cex_deviation_bps"].abs()

    result["cex_deviation_ema"] = (
        result["cex_deviation_abs"]
        .ewm(alpha=0.1, adjust=False)
        .mean()
    )

    result["cex_deviation_max_10"] = (
        result["cex_deviation_abs"]
        .rolling(window=10, min_periods=1)
        .max()
    )

    return result


def compute_liquidity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    if "liquidity" in result.columns:
        liq = result["liquidity"].astype(float)

        liq_mean = liq.rolling(500, min_periods=1).mean()
        liq_std = liq.rolling(500, min_periods=1).std().clip(lower=1e-10)
        result["liquidity_norm"] = ((liq - liq_mean) / liq_std).clip(-5, 5).fillna(0)

        result["liquidity_change"] = (
            liq.pct_change().fillna(0).clip(-1, 1)
        )

        result["liquidity_cv"] = (
            (liq_std / (liq_mean + 1e-10)).clip(0, 5).fillna(0)
        )
    else:
        result["liquidity_norm"] = 0.0
        result["liquidity_change"] = 0.0
        result["liquidity_cv"] = 0.0

    return result


def compute_price_features(
    df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    result = df.copy()
    prices = result[price_col].values

    for lag in [1, 5, 20, 100]:
        col = f"log_return_{lag}"
        result[col] = np.log(prices / np.roll(prices, lag))
        result.iloc[:lag, result.columns.get_loc(col)] = 0.0

    if "tick" in result.columns:
        ticks = result["tick"].values.astype(float)
        result["tick_delta"] = np.abs(np.diff(ticks, prepend=ticks[0])) / 100.0
    else:
        ticks = np.floor(np.log(prices) / np.log(1.0001))
        result["tick_delta"] = np.abs(np.diff(ticks, prepend=ticks[0])) / 100.0

    sma_short = pd.Series(prices).rolling(10, min_periods=1).mean()
    sma_long = pd.Series(prices).rolling(50, min_periods=1).mean()
    result["momentum"] = ((sma_short - sma_long) / (sma_long + 1e-10)).values
    result["momentum"] = result["momentum"].clip(-0.1, 0.1)

    return result


def compute_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    result = df.copy()

    if timestamp_col in result.columns:
        ts = pd.to_datetime(result[timestamp_col], unit="s" if result[timestamp_col].dtype != "datetime64[ns, UTC]" else None, utc=True)

        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek

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


def compute_gas_features(
    df: pd.DataFrame,
    gas_col: str = "gas_price_gwei",
) -> pd.DataFrame:
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


FEATURE_COLUMNS_FULL = [
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "log_return_100",
    "tick_delta",
    "momentum",
    "realized_vol_10",
    "realized_vol_50",
    "realized_vol_100",
    "realized_vol_500",
    "ema_vol",
    "vol_ratio_10_100",
    "ofi_10",
    "ofi_50",
    "cex_deviation_bps",
    "cex_deviation_abs",
    "cex_deviation_ema",
    "cex_deviation_max_10",
    "liquidity_norm",
    "liquidity_change",
    "liquidity_cv",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "gas_norm",
    "gas_surge",
]

FEATURE_COLUMNS_REDUCED = [
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "log_return_100",
    "tick_delta",
    "realized_vol_10",
    "vol_ratio_10_100",
    "ofi_10",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

FEATURE_COLUMNS_V3 = [
    "realized_vol_10",
    "vol_ratio_10_100",
    "log_return_1",
]

V3_INDICES_IN_REDUCED = [
    FEATURE_COLUMNS_REDUCED.index(f) for f in FEATURE_COLUMNS_V3
]

FEATURE_COLUMNS = FEATURE_COLUMNS_V3

REDUCED_INDICES_IN_FULL = [
    FEATURE_COLUMNS_FULL.index(f) for f in FEATURE_COLUMNS_REDUCED
]

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
    df = onchain_df.copy()

    if "log_return" not in df.columns and "price" in df.columns:
        df["log_return"] = np.log(df["price"] / df["price"].shift(1)).fillna(0)

    if cex_prices is not None:
        n = min(len(df), len(cex_prices))
        df = df.iloc[:n].copy()
        df["cex_price"] = cex_prices[:n]
    elif "cex_price" not in df.columns:
        df["cex_price"] = df.get("price", 0)

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

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            logger.warning(f"Feature column '{col}' missing, filling with 0")
            df[col] = 0.0

    n_features = len(FEATURE_COLUMNS)
    logger.info(f"Feature engineering complete: {len(df):,} rows × {n_features} features")

    return df
