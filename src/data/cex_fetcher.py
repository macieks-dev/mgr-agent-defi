
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


BINANCE_DATA_URL = "https://data.binance.vision"
BINANCE_API_URL = "https://api.binance.com"
BINANCE_BACKUP_API_URL = "https://data-api.binance.vision"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]

BLOCK_INTERVAL_SECONDS = 12


def _download_daily_klines_zip(
    symbol: str,
    date: str,
    interval: str = "1s",
    session: Optional[requests.Session] = None,
) -> Optional[pd.DataFrame]:
    s = session or requests.Session()
    filename = f"{symbol}-{interval}-{date}.zip"
    url = f"{BINANCE_DATA_URL}/data/spot/daily/klines/{symbol}/{interval}/{filename}"

    try:
        resp = s.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download {date}: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    except Exception as e:
        logger.warning(f"Failed to parse {date}: {e}")
        return None

    df.drop(columns=["_ignore"], errors="ignore", inplace=True)

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)

    return df


def _aggregate_to_block_interval(
    df_1s: pd.DataFrame,
    interval_seconds: int = BLOCK_INTERVAL_SECONDS,
) -> pd.DataFrame:
    df = df_1s.copy()

    ot = df["open_time"].values
    is_microseconds = ot > 1e13
    if is_microseconds.any():
        n_us = is_microseconds.sum()
        logger.info(f"Normalizing {n_us:,} microsecond timestamps to milliseconds")
        ot = ot.copy()
        ot[is_microseconds] = ot[is_microseconds] // 1000
        df["open_time"] = ot

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df["block_ts"] = df["timestamp"].dt.floor(f"{interval_seconds}s")

    agg = df.groupby("block_ts").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trades=("trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    ).reset_index()

    agg.rename(columns={"block_ts": "timestamp"}, inplace=True)
    return agg


def fetch_binance_block_klines(
    symbol: str = "ETHUSDT",
    start_date: str = "2023-07-01",
    end_date: str = "2026-01-31",
    save_path: Optional[Path] = None,
    max_workers: int = 4,
    checkpoint_every: int = 30,
    checkpoint_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    logger.info(
        f"Fetching {symbol} 1s klines: {start_date} → {end_date} "
        f"({len(all_dates)} days, {max_workers} workers)"
    )

    if checkpoint_dir is None:
        checkpoint_dir = Path("data/raw/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{symbol}_12s_checkpoint.parquet"

    requested_dates = set(all_dates)
    already_fetched_dates: set[str] = set()
    existing_chunks: list[pd.DataFrame] = []

    if checkpoint_file.exists():
        logger.info(f"Found checkpoint: {checkpoint_file}")
        ckpt = pd.read_parquet(checkpoint_file)
        if len(ckpt) > 0:
            ckpt_timestamps = pd.to_datetime(ckpt["timestamp"], utc=True)
            ckpt = ckpt.loc[
                ckpt_timestamps.dt.strftime("%Y-%m-%d").isin(requested_dates)
            ].copy()
            if len(ckpt) > 0:
                existing_chunks.append(ckpt)
                ckpt_dates = (
                    pd.to_datetime(ckpt["timestamp"], utc=True)
                    .dt.strftime("%Y-%m-%d")
                    .unique()
                )
                already_fetched_dates = set(ckpt_dates)
            logger.info(f"  Checkpoint covers {len(already_fetched_dates)} days, resuming...")

    dates_to_fetch = [d for d in all_dates if d not in already_fetched_dates]

    if not dates_to_fetch:
        logger.info("All dates already fetched from checkpoint!")
        df = pd.concat(existing_chunks, ignore_index=True) if existing_chunks else pd.DataFrame()
        if not df.empty:
            df = (
                df.drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        if save_path and len(df) > 0:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
        return df

    logger.info(f"Downloading {len(dates_to_fetch)} days of 1s data...")

    session = requests.Session()
    new_chunks: list[pd.DataFrame] = []
    fetched_count = 0
    failed_dates: list[str] = []

    batch_size = max_workers * 5
    next_checkpoint_count = checkpoint_every
    for batch_start in range(0, len(dates_to_fetch), batch_size):
        batch = dates_to_fetch[batch_start : batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _download_daily_klines_zip, symbol, date, "1s", session
                ): date
                for date in batch
            }

            for future in as_completed(futures):
                date = futures[future]
                try:
                    daily_df = future.result()
                    if daily_df is not None and len(daily_df) > 0:
                        block_df = _aggregate_to_block_interval(daily_df)
                        new_chunks.append(block_df)
                        fetched_count += 1
                    else:
                        failed_dates.append(date)
                except Exception as e:
                    logger.warning(f"Error processing {date}: {e}")
                    failed_dates.append(date)

        total_done = fetched_count + len(already_fetched_dates)
        logger.info(
            f"  Progress: {total_done}/{len(all_dates)} days "
            f"({total_done * 100 / len(all_dates):.1f}%)"
        )

        if fetched_count >= next_checkpoint_count:
            _save_checkpoint(existing_chunks + new_chunks, checkpoint_file)
            next_checkpoint_count = fetched_count + checkpoint_every

    if failed_dates:
        _save_checkpoint(existing_chunks + new_chunks, checkpoint_file)
        failed_sample = ", ".join(failed_dates[:10])
        remaining = "..." if len(failed_dates) > 10 else ""
        progress_note = (
            f"Successful days were saved to checkpoint {checkpoint_file}."
            if existing_chunks or new_chunks
            else "No successful days were available to checkpoint."
        )
        raise RuntimeError(
            f"Failed to fetch {len(failed_dates)} of {len(all_dates)} requested days "
            f"for {symbol}: {failed_sample}{remaining}. {progress_note}"
        )

    all_chunks = existing_chunks + new_chunks
    if not all_chunks:
        raise RuntimeError(f"No data fetched for {symbol}")

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Total: {len(df):,} block-level (12s) records for {symbol} "
        f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})"
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint cleaned up")

    return df


def _save_checkpoint(chunks: list[pd.DataFrame], path: Path):
    if not chunks:
        return
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)
    logger.info(f"  Checkpoint saved: {len(df):,} records → {path}")


VALID_INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                    "6h", "8h", "12h", "1d", "3d", "1w", "1M"]


def fetch_binance_klines(
    symbol: str = "ETHUSDT",
    interval: str = "1m",
    start_date: str = "2023-07-01",
    end_date: str = "2026-01-31",
    save_path: Optional[Path] = None,
    rate_limit_sleep: float = 0.1,
) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Must be one of {VALID_INTERVALS}")

    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d")
                  .replace(tzinfo=timezone.utc).timestamp() * 1000)

    logger.info(f"Fetching {symbol} {interval} klines: {start_date} → {end_date}")

    all_klines = []
    current_start = start_ms
    page = 0
    base_url = BINANCE_API_URL

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        try:
            resp = requests.get(
                f"{base_url}/api/v3/klines",
                params=params,
                timeout=30,
            )

            if resp.status_code == 451:
                logger.warning("Binance main API blocked, trying backup...")
                base_url = BINANCE_BACKUP_API_URL
                resp = requests.get(
                    f"{base_url}/api/v3/klines",
                    params=params,
                    timeout=30,
                )

            resp.raise_for_status()
            klines = resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error: {e}")
            if base_url == BINANCE_API_URL:
                logger.info("Retrying with backup URL...")
                base_url = BINANCE_BACKUP_API_URL
                continue
            break

        if not klines:
            break

        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        page += 1

        if page % 50 == 0:
            logger.info(f"  Fetched {len(all_klines):,} klines so far...")

        time.sleep(rate_limit_sleep)

    if not all_klines:
        raise RuntimeError(f"No klines returned for {symbol}")

    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)
    df.drop(columns=["_ignore"], inplace=True)

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df.rename(columns={"open_time": "timestamp"}, inplace=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Fetched {len(df):,} {interval} klines for {symbol} "
        f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})"
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    return df


def align_cex_to_blocks(
    cex_df: pd.DataFrame,
    block_timestamps: np.ndarray,
) -> np.ndarray:
    cex_unix = cex_df["timestamp"].astype(np.int64) // 10**9
    cex_prices = cex_df["close"].values

    aligned = np.interp(block_timestamps, cex_unix, cex_prices)
    return aligned


def compute_cex_features(cex_df: pd.DataFrame) -> pd.DataFrame:
    df = cex_df.copy()

    df["taker_buy_ratio"] = (
        df["taker_buy_quote"] / (df["quote_volume"] + 1e-8)
    )

    vol_mean = df["quote_volume"].rolling(300, min_periods=1).mean()
    vol_std = df["quote_volume"].rolling(300, min_periods=1).std().clip(lower=1e-8)
    df["volume_zscore"] = (df["quote_volume"] - vol_mean) / vol_std

    df["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    df["cex_log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["cex_log_return"] = df["cex_log_return"].fillna(0)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.config import config

    df = fetch_binance_block_klines(
        symbol="ETHUSDT",
        start_date=config.train_start,
        end_date=config.test_end,
        save_path=config.data_raw_dir / "binance_ethusdt_12s.parquet",
        checkpoint_dir=config.data_raw_dir / "checkpoints",
    )

    logger.info(f"Fetched {len(df):,} block-level (12s) records")
    logger.info(f"Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    logger.info(f"Price range: {df['close'].min():.2f} — {df['close'].max():.2f}")
    size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory usage: {size_mb:.1f} MB")
