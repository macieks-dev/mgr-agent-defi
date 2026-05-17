"""
Tick-by-tick data loader for Uniswap V3/V4 swap events.

Supports:
  1. Loading from local CSV/Parquet files.
  2. Fetching from The Graph (Uniswap Subgraph).
  3. Fetching from Google BigQuery (Ethereum public dataset).
  4. Generating synthetic tick data for development/testing.

The output is always a standardised DataFrame with columns:
  block_number, timestamp, tick, sqrtPriceX96, liquidity,
  amount0, amount1, price, log_return
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import config

logger = logging.getLogger(__name__)



TICK_DATA_COLUMNS = [
    "block_number",
    "timestamp",
    "tick",
    "sqrtPriceX96",
    "liquidity",
    "amount0",
    "amount1",
    "price",
    "log_return",
]


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has the required schema."""
    for col in TICK_DATA_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df




def load_from_file(path: Path | str) -> pd.DataFrame:
    """Load tick data from a local CSV or Parquet file."""
    path = Path(path)
    logger.info(f"Loading tick data from {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df = _enrich_dataframe(df)
    return _validate_df(df)




UNISWAP_V3_SUBGRAPH = (
    "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
)

def fetch_from_subgraph(
    pool_address: str,
    n_records: int = 10_000,
    subgraph_url: str = UNISWAP_V3_SUBGRAPH,
) -> pd.DataFrame:
    """Fetch swap events from the Uniswap V3 Subgraph (paginated by 1000)."""
    logger.info(f"Fetching {n_records} swaps for pool {pool_address} from subgraph")

    all_swaps = []
    skip = 0
    page_size = 1000

    while len(all_swaps) < n_records:
        query = """
        {
          swaps(
            first: %d,
            skip: %d,
            orderBy: timestamp,
            orderDirection: asc,
            where: { pool: "%s" }
          ) {
            id
            timestamp
            tick
            sqrtPriceX96
            liquidity
            amount0
            amount1
            transaction {
              blockNumber
            }
          }
        }
        """ % (min(page_size, n_records - len(all_swaps)), skip, pool_address.lower())

        try:
            resp = requests.post(subgraph_url, json={"query": query}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Subgraph request failed: {e}")
            break

        swaps = data.get("data", {}).get("swaps", [])
        if not swaps:
            break

        all_swaps.extend(swaps)
        skip += page_size
        logger.debug(f"  Fetched {len(all_swaps)} / {n_records}")

    if not all_swaps:
        raise RuntimeError("No swap data returned from subgraph")

    df = pd.DataFrame(all_swaps)
    df["block_number"] = df["transaction"].apply(lambda x: int(x["blockNumber"]))
    df.drop(columns=["transaction", "id"], inplace=True, errors="ignore")

    # Type conversions
    for col in ["tick", "sqrtPriceX96", "liquidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["amount0", "amount1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_numeric(df["timestamp"])

    df = _enrich_dataframe(df)
    return _validate_df(df)



def fetch_from_bigquery(
    pool_address: str,
    n_records: int = 1_000_000,
    project_id: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch swap events from BigQuery Ethereum public dataset. Requires google-cloud-bigquery."""
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError(
            "Install google-cloud-bigquery: pip install google-cloud-bigquery"
        )

    project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")
    client = bigquery.Client(project=project_id)

    query = f"""
    SELECT
        block_number,
        block_timestamp AS timestamp,
        -- Decoded swap event fields (you may need to adjust based on your dataset)
        -- This is a template — actual field names depend on your decoded logs table
        *
    FROM `bigquery-public-data.crypto_ethereum.logs`
    WHERE
        LOWER(address) = LOWER('{pool_address}')
        AND topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
    ORDER BY block_number ASC
    LIMIT {n_records}
    """

    logger.info(f"Running BigQuery for pool {pool_address}, limit {n_records}")
    df = client.query(query).to_dataframe()

    df = _enrich_dataframe(df)
    return _validate_df(df)



def generate_synthetic_data(
    n_records: int = 100_000,
    initial_price: float = 2000.0,
    volatility: float = 0.02,
    mean_liquidity: float = 1e18,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic tick data via GBM. Used for dev/testing when real data is unavailable."""
    rng = np.random.default_rng(seed or config.seed)

    logger.info(f"Generating {n_records} synthetic tick-data records (σ={volatility})")

    # ── Price path via GBM ──
    dt = 1.0  # 1 block
    mu = 0.0  # no drift
    log_returns = rng.normal(mu * dt, volatility * np.sqrt(dt), size=n_records)
    log_returns[0] = 0.0
    prices = initial_price * np.exp(np.cumsum(log_returns))

    # ── Derive Uniswap tick from price ──
    # tick = floor(log(price) / log(1.0001))
    ticks = np.floor(np.log(prices) / np.log(1.0001)).astype(np.int64)

    # ── sqrtPriceX96 = sqrt(price) * 2^96 ──
    sqrt_prices = np.sqrt(prices)
    sqrtPriceX96 = (sqrt_prices * (2**96)).astype(np.float64)

    # ── Liquidity: mean with noise ──
    liquidity = rng.normal(mean_liquidity, mean_liquidity * 0.1, size=n_records)
    liquidity = np.clip(liquidity, mean_liquidity * 0.5, mean_liquidity * 2.0)

    # ── Swap amounts: proportional to price move ──
    price_delta = np.diff(prices, prepend=prices[0])
    amount0 = -price_delta * liquidity / (prices**2) * 1e18  # approximate
    amount1 = price_delta * liquidity / prices * 1e6  # approximate (USDC decimals)

    # ── Timestamps: 12s per block ──
    base_timestamp = 1_700_000_000
    timestamps = base_timestamp + np.arange(n_records) * 12

    # ── Block numbers ──
    base_block = 18_000_000
    block_numbers = base_block + np.arange(n_records)

    df = pd.DataFrame({
        "block_number": block_numbers,
        "timestamp": timestamps,
        "tick": ticks,
        "sqrtPriceX96": sqrtPriceX96,
        "liquidity": liquidity,
        "amount0": amount0,
        "amount1": amount1,
        "price": prices,
        "log_return": log_returns,
    })

    return _validate_df(df)



def _enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns (price, log_return) if not already present."""
    # Price from tick if not present
    if "price" not in df.columns and "tick" in df.columns:
        df["price"] = 1.0001 ** df["tick"].astype(float)

    # Price from sqrtPriceX96 if tick missing
    if "price" not in df.columns and "sqrtPriceX96" in df.columns:
        sqrt_price = df["sqrtPriceX96"].astype(float) / (2**96)
        df["price"] = sqrt_price**2

    # Log returns
    if "log_return" not in df.columns and "price" in df.columns:
        df["log_return"] = np.log(df["price"] / df["price"].shift(1))
        df.iloc[0, df.columns.get_loc("log_return")] = 0.0

    # Sort by block
    if "block_number" in df.columns:
        df = df.sort_values("block_number").reset_index(drop=True)

    return df



def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = 100,
    column: str = "log_return",
) -> pd.Series:
    """Rolling realized volatility (std of log returns) over a block window."""
    return df[column].rolling(window=window, min_periods=1).std()


def compute_ema_volatility(
    df: pd.DataFrame,
    alpha: float = 0.1,
    column: str = "log_return",
) -> pd.Series:
    """EMA-based volatility, matching on-chain EMA oracle logic."""
    abs_returns = df[column].abs()
    return abs_returns.ewm(alpha=alpha, adjust=False).mean()



def main():
    """Generate synthetic tick data and compute rolling volatility diagnostics."""
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Synthetic tick data pipeline")

    df = generate_synthetic_data(n_records=1_000_000, volatility=0.02, seed=42)

    logger.info(f"Loaded {len(df):,} records")
    logger.info(f"Price range: {df['price'].min():.2f} — {df['price'].max():.2f}")
    logger.info(f"Tick range:  {df['tick'].min()} — {df['tick'].max()}")
    logger.info(f"Block range: {df['block_number'].min()} — {df['block_number'].max()}")

    df["rolling_vol_100"] = compute_rolling_volatility(df, window=100)
    df["ema_vol"] = compute_ema_volatility(df, alpha=0.1)

    logger.info(f"Rolling Vol (100-block) — mean: {df['rolling_vol_100'].mean():.6f}")
    logger.info(f"EMA Vol — mean: {df['ema_vol'].mean():.6f}")

    output_path = config.data_processed_dir / "synthetic_tick_data.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    sample = df.iloc[::100]  # subsample for plotting

    axes[0].plot(sample["block_number"], sample["price"], linewidth=0.5)
    axes[0].set_ylabel("Price")
    axes[0].set_title("Synthetic Price Path (GBM)")

    axes[1].plot(sample["block_number"], sample["rolling_vol_100"], linewidth=0.5, color="orange")
    axes[1].set_ylabel("Rolling σ (100 blocks)")
    axes[1].set_title("Rolling Realized Volatility")

    axes[2].plot(sample["block_number"], sample["ema_vol"], linewidth=0.5, color="green")
    axes[2].set_ylabel("EMA Vol")
    axes[2].set_title("EMA Volatility (α=0.1)")
    axes[2].set_xlabel("Block Number")

    plt.tight_layout()
    plt.savefig(config.project_root / "data" / "volatility_analysis.png", dpi=150)
    plt.show()

    logger.info("Synthetic tick data pipeline complete.")


if __name__ == "__main__":
    main()
