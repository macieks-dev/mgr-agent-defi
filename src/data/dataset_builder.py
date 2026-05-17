"""
Dataset Builder — pełny pipeline: pobieranie → feature engineering → split.

Block-level granularity (12s) — dopasowane do czasu bloku Ethereum.
Dane od połowy 2023 — bardziej reprezentatywne dla obecnego ekosystemu MEV.

Rygorystyczny podział temporalny (NIGDY losowy!):
  Training:    Jul 2023 – Mar 2025 (boom, crash, recovery, MEV maturity)
  Validation:  Apr 2025 – Sep 2025 (tuning hiperparametrów)
  Test:        Oct 2025 – Jan 2026 (out-of-sample, NIGDY nie trenuj na tym)

Użycie:
    from src.data.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    builder.fetch_all()          # Pobierz dane
    builder.build_features()     # Feature engineering
    builder.split_and_save()     # Podziel i zapisz

Lub CLI:
    docker run kodv4-app python scripts/fetch_data.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import config
from src.data.tick_loader import (
    generate_synthetic_data,
    fetch_from_subgraph,
    load_from_file,
)
from src.data.cex_fetcher import (
    fetch_binance_block_klines,
    fetch_binance_klines,
    align_cex_to_blocks,
    compute_cex_features,
)
from src.data.feature_engineering import (
    build_all_features,
    FEATURE_COLUMNS,
    META_COLUMNS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#                    TEMPORAL SPLIT DEFINITION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DataSplit:
    """Definicja podziału temporalnego."""
    train_start: str = "2023-07-01"
    train_end: str = "2025-03-31"
    val_start: str = "2025-04-01"
    val_end: str = "2025-09-30"
    test_start: str = "2025-10-01"
    test_end: str = "2026-01-31"

    def to_dict(self) -> dict:
        return {
            "train": {"start": self.train_start, "end": self.train_end},
            "val": {"start": self.val_start, "end": self.val_end},
            "test": {"start": self.test_start, "end": self.test_end},
        }


# ══════════════════════════════════════════════════════════════════════════
#                    MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_market_regime(
    prices: np.ndarray,
    window: int = 200,
) -> np.ndarray:
    """
    Klasyfikuj reżim rynkowy: 0=konsolidacja, 1=hossa, 2=bessa, 3=wysoka zmienność.
    Agent musi widzieć ekstremalne warunki w treningu.
    """
    log_returns = np.log(prices[1:] / prices[:-1])
    log_returns = np.concatenate([[0], log_returns])

    # Rolling statistics
    df = pd.DataFrame({"ret": log_returns})
    roll_mean = df["ret"].rolling(window, min_periods=1).mean().values
    roll_vol = df["ret"].rolling(window, min_periods=1).std().fillna(0).values

    median_vol = np.median(roll_vol[roll_vol > 0]) if np.any(roll_vol > 0) else 0.01
    median_ret = np.median(np.abs(roll_mean[roll_mean != 0])) if np.any(roll_mean != 0) else 0.001

    regimes = np.zeros(len(prices), dtype=int)

    for i in range(len(prices)):
        if roll_vol[i] > 2 * median_vol:
            regimes[i] = 3  # high vol
        elif roll_mean[i] > median_ret:
            regimes[i] = 1  # bull
        elif roll_mean[i] < -median_ret:
            regimes[i] = 2  # bear
        else:
            regimes[i] = 0  # consolidation

    return regimes


# ══════════════════════════════════════════════════════════════════════════
#                       DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════

class DatasetBuilder:
    """
    Orkiestrator pipeline'u danych.

    Zbiera dane z wielu źródeł, buduje cechy, dzieli na zbiory.
    """

    # Uniswap V3 ETH/USDC 0.05% pool (najwyższa płynność)
    POOL_ADDRESS = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        split: Optional[DataSplit] = None,
    ):
        self.raw_dir = raw_dir or config.data_raw_dir
        self.processed_dir = processed_dir or config.data_processed_dir
        self.split = split or DataSplit()

        self.onchain_df: Optional[pd.DataFrame] = None
        self.cex_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.metadata: dict = {}

    # ─────────────────────────────────────────────────
    #               FETCH DATA
    # ─────────────────────────────────────────────────

    def fetch_all(
        self,
        use_subgraph: bool = True,
        synthetic_fallback: bool = True,
        n_synthetic: int = 1_000_000,
    ):
        """Pobierz on-chain + CEX. Promuje CEX jako primary jeśli on-chain jest syntetyczny."""
        # ── 1. CEX data ──
        self._fetch_cex()

        # ── 2. On-chain data ──
        self._fetch_onchain(use_subgraph, synthetic_fallback, n_synthetic)

        # ── 3. If on-chain is synthetic but CEX is real, promote CEX ──
        is_synthetic = self.metadata.get("onchain_source") == "synthetic"
        if not is_synthetic and self.onchain_df is not None:
            price_range = self.onchain_df["price"].max() - self.onchain_df["price"].min()
            min_price = self.onchain_df["price"].min()
            # Real ETH: ~$800–$5000, synthetic GBM: can go to 0 or 200k+
            if min_price < 100 or price_range > 50_000:
                is_synthetic = True
                logger.info(
                    f"Detected synthetic on-chain data "
                    f"(price {min_price:.2f}–{self.onchain_df['price'].max():.2f})"
                )

        if (
            is_synthetic
            and self.cex_df is not None
            and len(self.cex_df) > 0
        ):
            self._promote_cex_to_primary()

        logger.info("All data fetched successfully.")

    def _fetch_onchain(
        self,
        use_subgraph: bool,
        synthetic_fallback: bool,
        n_synthetic: int,
    ):
        """Pobierz lub wygeneruj dane on-chain."""
        # Check for existing raw file first
        onchain_path = self.raw_dir / "uniswap_v3_swaps.parquet"
        if onchain_path.exists():
            logger.info(f"Loading cached on-chain data from {onchain_path}")
            self.onchain_df = pd.read_parquet(onchain_path)
            logger.info(f"Loaded {len(self.onchain_df):,} on-chain records")
            self.metadata["onchain_source"] = "cached_file"
            return

        # Try The Graph
        if use_subgraph:
            try:
                logger.info("Fetching from Uniswap V3 subgraph...")
                self.onchain_df = fetch_from_subgraph(
                    pool_address=self.POOL_ADDRESS,
                    n_records=50_000,
                )
                self.onchain_df.to_parquet(onchain_path, index=False)
                self.metadata["onchain_source"] = "subgraph"
                logger.info(f"Fetched {len(self.onchain_df):,} on-chain records")
                return
            except Exception as e:
                logger.warning(f"Subgraph fetch failed: {e}")

        # Fallback: synthetic data
        if synthetic_fallback:
            logger.info(f"Generating {n_synthetic:,} synthetic on-chain records...")
            self.onchain_df = generate_synthetic_data(
                n_records=n_synthetic,
                initial_price=2000.0,
                volatility=0.015,  # Calibrated to real ETH vol
                seed=config.seed,
            )
            self.onchain_df.to_parquet(onchain_path, index=False)
            self.metadata["onchain_source"] = "synthetic"
            logger.info(f"Generated {len(self.onchain_df):,} synthetic records")
        else:
            raise RuntimeError("No on-chain data available and synthetic fallback disabled")

    def _fetch_cex(self):
        """Pobierz dane CEX z Binance (12s block-level)."""
        cex_path = self.raw_dir / "binance_ethusdt_12s.parquet"

        if cex_path.exists():
            logger.info(f"Loading cached CEX data from {cex_path}")
            self.cex_df = pd.read_parquet(cex_path)
            self.metadata["cex_source"] = "cached_file"
            self.metadata["cex_interval"] = "12s"
            logger.info(f"Loaded {len(self.cex_df):,} CEX records (12s)")
            return

        try:
            self.cex_df = fetch_binance_block_klines(
                symbol="ETHUSDT",
                start_date=self.split.train_start,
                end_date=self.split.test_end,
                save_path=cex_path,
                checkpoint_dir=self.raw_dir / "checkpoints",
            )
            self.metadata["cex_source"] = "binance_data_vision"
            self.metadata["cex_interval"] = "12s"
            logger.info(f"Fetched {len(self.cex_df):,} CEX records (12s)")

        except Exception as e:
            logger.warning(f"Block-level CEX fetch failed: {e}. Using synthetic prices as CEX proxy.")
            self.cex_df = None
            self.metadata["cex_source"] = "none"

    def _promote_cex_to_primary(self):
        """
        Użyj CEX jako głównego źródła cen (zamiast syntetycznych).

        Binance ETH/USDT jest najlepszym proxy dla AMM ETH/USDC —
        arbitrażyści utrzymują spread < 5bps.
        """
        logger.info(
            "PROMOTING CEX data as primary price source "
            f"({len(self.cex_df):,} records, real Binance data)"
        )

        cex = self.cex_df.copy()

        # ── Filter out corrupt/future timestamps ──
        ts_min = pd.Timestamp(self.split.train_start, tz="UTC")
        ts_max = pd.Timestamp(self.split.test_end, tz="UTC") + pd.Timedelta(days=1)
        mask = (cex["timestamp"] >= ts_min) & (cex["timestamp"] <= ts_max)
        n_before = len(cex)
        cex = cex[mask].reset_index(drop=True)
        if len(cex) < n_before:
            logger.warning(
                f"Filtered {n_before - len(cex):,} rows outside "
                f"{self.split.train_start} – {self.split.test_end}"
            )

        if len(cex) == 0:
            logger.error("No valid CEX records after filtering — keeping synthetic")
            return

        # Convert timestamp to unix seconds
        # datetime64[ms, UTC].astype(int64) → milliseconds since epoch
        ts_unix = cex["timestamp"].astype(np.int64) // 10**3  # ms → s

        onchain = pd.DataFrame({
            "price": cex["close"].values,
            "timestamp": ts_unix.values,
            "volume": cex["volume"].values,
            "quote_volume": cex["quote_volume"].values,
            # Synthetic block numbers (12s apart, starting from a recent block)
            "block_number": np.arange(len(cex)) + 18_000_000,
            # Additional CEX fields useful for features
            "open": cex["open"].values,
            "high": cex["high"].values,
            "low": cex["low"].values,
            "trades": cex["trades"].values if "trades" in cex.columns else 0,
        })

        # ── Realistic liquidity: volume-derived, time-varying ──
        # Uni V3 ETH/USDC typically has TVL ~ 10-20x daily volume
        # Use rolling 100-block EMA of quote_volume as proxy
        qv = cex["quote_volume"].values.astype(np.float64)
        qv_ema = pd.Series(qv).ewm(span=100, adjust=False).mean().values
        # Scale: TVL ≈ 15x 100-block average volume, clamped to [100M, 2B]
        liquidity = np.clip(qv_ema * 15 * (12 / 86400) * 86400, 100_000_000, 2_000_000_000)
        onchain["liquidity"] = liquidity.astype(np.float32)

        # ── Synthetic gas prices with realistic variance ──
        # Historical ETH gas: mean ~30 gwei, std ~20, correlated with volume
        rng = np.random.default_rng(42)
        base_gas = 30.0 + 15.0 * (qv / (np.mean(qv) + 1e-8) - 1.0)  # vol-correlated
        gas_noise = rng.lognormal(0, 0.3, len(cex))
        gas_prices = np.clip(base_gas * gas_noise, 5.0, 500.0)
        onchain["gas_price_gwei"] = gas_prices.astype(np.float32)

        # Replace synthetic data
        self.onchain_df = onchain
        self.metadata["onchain_source"] = "cex_promoted"
        self.metadata["price_source"] = "binance_ethusdt_12s"
        self.metadata["price_range"] = f"{cex['close'].min():.2f} – {cex['close'].max():.2f}"
        self.metadata["n_cex_promoted_records"] = len(onchain)

        logger.info(
            f"CEX promoted: {len(onchain):,} records, "
            f"price {cex['close'].min():.2f} – {cex['close'].max():.2f}"
        )

    # ─────────────────────────────────────────────────
    #           BUILD FEATURES
    # ─────────────────────────────────────────────────

    def build_features(self):
        """Uruchom pełny feature engineering pipeline (on-chain + CEX → wektor cech)."""
        if self.onchain_df is None:
            raise RuntimeError("No on-chain data loaded. Call fetch_all() first.")

        # Align CEX prices to block timestamps if available
        cex_prices = None
        if self.metadata.get("onchain_source") == "cex_promoted":
            # AMM-CEX microstructure noise to simulate the 1-5bps arb spread
            rng = np.random.default_rng(123)
            amm_prices = self.onchain_df["price"].values.astype(np.float64)
            # Spread: mean ~2bps, std ~3bps, heavier tails via Laplace
            spread_bps = rng.laplace(0, 2.0, len(amm_prices))
            # During high volatility, spread widens
            if "high" in self.onchain_df.columns and "low" in self.onchain_df.columns:
                hl_ratio = (self.onchain_df["high"].values / (self.onchain_df["low"].values + 1e-8))
                vol_multiplier = np.clip(hl_ratio - 1, 0, 0.05) * 200  # 1% H/L → +2bps extra
                spread_bps += vol_multiplier * rng.standard_normal(len(amm_prices))
            cex_prices = amm_prices * (1 + spread_bps / 10_000)
            logger.info(
                f"Using promoted CEX prices with microstructure noise "
                f"(mean spread: {np.mean(np.abs(spread_bps)):.2f} bps)"
            )
        elif self.cex_df is not None and "timestamp" in self.onchain_df.columns:
            try:
                block_timestamps = self.onchain_df["timestamp"].values.astype(float)
                cex_prices = align_cex_to_blocks(self.cex_df, block_timestamps)
                logger.info("CEX prices aligned to block timestamps")
            except Exception as e:
                logger.warning(f"CEX alignment failed: {e}")

        # Build all features
        self.features_df = build_all_features(
            onchain_df=self.onchain_df,
            cex_prices=cex_prices,
        )

        # Add market regime labels
        if "price" in self.features_df.columns:
            self.features_df["market_regime"] = detect_market_regime(
                self.features_df["price"].values
            )

        # Statistics
        self.metadata["n_features"] = len(FEATURE_COLUMNS)
        self.metadata["n_records"] = len(self.features_df)
        self.metadata["feature_columns"] = FEATURE_COLUMNS

        if "market_regime" in self.features_df.columns:
            regime_counts = self.features_df["market_regime"].value_counts().to_dict()
            regime_names = {0: "consolidation", 1: "bull", 2: "bear", 3: "high_vol"}
            self.metadata["regime_distribution"] = {
                regime_names.get(k, str(k)): v for k, v in regime_counts.items()
            }

        logger.info(f"Features built: {len(self.features_df):,} rows × {len(FEATURE_COLUMNS)} features")

    # ─────────────────────────────────────────────────
    #           SPLIT & SAVE
    # ─────────────────────────────────────────────────

    def split_and_save(self):
        """Podziel dane temporalnie i zapisz. NIGDY losowy split — no leakage."""
        if self.features_df is None:
            raise RuntimeError("No features built. Call build_features() first.")

        df = self.features_df.copy()

        # If we have timestamp-based splitting
        if "timestamp" in df.columns and self.metadata.get("onchain_source") not in ("synthetic",):
            splits = self._temporal_split(df)
        else:
            # For synthetic data: use ratio-based split (70/15/15)
            splits = self._ratio_split(df, ratios=(0.70, 0.15, 0.15))

        # Save splits
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                logger.warning(f"Split '{split_name}' is empty!")
                continue

            path = self.processed_dir / f"features_{split_name}.parquet"
            split_df.to_parquet(path, index=False)
            logger.info(f"Saved {split_name}: {len(split_df):,} rows → {path}")

            self.metadata[f"{split_name}_size"] = len(split_df)

            # Per-split regime stats
            if "market_regime" in split_df.columns:
                regime_names = {0: "consolidation", 1: "bull", 2: "bear", 3: "high_vol"}
                dist = split_df["market_regime"].value_counts().to_dict()
                self.metadata[f"{split_name}_regimes"] = {
                    regime_names.get(k, str(k)): v for k, v in dist.items()
                }

        # Also save the full feature array for environment (price + features)
        self._save_env_arrays(splits)

        # Save metadata
        meta_path = self.processed_dir / "dataset_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {meta_path}")

    def _temporal_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Podział na podstawie timestamp."""
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        train_mask = (ts >= self.split.train_start) & (ts < self.split.train_end)
        val_mask = (ts >= self.split.val_start) & (ts < self.split.val_end)
        test_mask = (ts >= self.split.test_start) & (ts <= self.split.test_end)

        return {
            "train": df[train_mask].reset_index(drop=True),
            "val": df[val_mask].reset_index(drop=True),
            "test": df[test_mask].reset_index(drop=True),
        }

    def _ratio_split(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> dict[str, pd.DataFrame]:
        """
        Podział proporcjonalny (dla danych syntetycznych).
        Wciąż TEMPORALNY — nigdy nie mieszamy!
        """
        n = len(df)
        train_end = int(n * ratios[0])
        val_end = int(n * (ratios[0] + ratios[1]))

        return {
            "train": df.iloc[:train_end].reset_index(drop=True),
            "val": df.iloc[train_end:val_end].reset_index(drop=True),
            "test": df.iloc[val_end:].reset_index(drop=True),
        }

    def _save_env_arrays(self, splits: dict[str, pd.DataFrame]):
        """Zapisz tablice numpy (features, prices, cex_prices) per split."""
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                continue

            # Feature matrix
            feature_cols = [c for c in FEATURE_COLUMNS if c in split_df.columns]
            features = split_df[feature_cols].values.astype(np.float32)
            np.save(
                self.processed_dir / f"features_{split_name}.npy",
                features,
            )

            # Prices
            if "price" in split_df.columns:
                np.save(
                    self.processed_dir / f"prices_{split_name}.npy",
                    split_df["price"].values.astype(np.float64),
                )

            # CEX prices
            if "cex_price" in split_df.columns:
                np.save(
                    self.processed_dir / f"cex_prices_{split_name}.npy",
                    split_df["cex_price"].values.astype(np.float64),
                )

            logger.info(
                f"Saved {split_name} arrays: features={features.shape}, "
                f"prices={(len(split_df),)}"
            )

    # ─────────────────────────────────────────────────
    #           LOAD PROCESSED DATA
    # ─────────────────────────────────────────────────

    @staticmethod
    def load_split(
        split: str = "train",
        processed_dir: Optional[Path] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Wczytaj gotowy split (features, prices, cex_prices) jako numpy arrays."""
        processed_dir = processed_dir or config.data_processed_dir

        features = np.load(processed_dir / f"features_{split}.npy")
        prices = np.load(processed_dir / f"prices_{split}.npy")

        cex_path = processed_dir / f"cex_prices_{split}.npy"
        cex_prices = np.load(cex_path) if cex_path.exists() else None

        return features, prices, cex_prices

    @staticmethod
    def load_metadata(processed_dir: Optional[Path] = None) -> dict:
        """Wczytaj metadane datasetu."""
        processed_dir = processed_dir or config.data_processed_dir
        meta_path = processed_dir / "dataset_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    # ─────────────────────────────────────────────────
    #           PRINT SUMMARY
    # ─────────────────────────────────────────────────

    def print_summary(self):
        """Wydrukuj podsumowanie datasetu."""
        print("=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        if self.onchain_df is not None:
            print(f"\nOn-chain data: {len(self.onchain_df):,} records")
            print(f"  Source: {self.metadata.get('onchain_source', 'unknown')}")
            if "price" in self.onchain_df.columns:
                print(f"  Price range: {self.onchain_df['price'].min():.2f} — {self.onchain_df['price'].max():.2f}")
            if "block_number" in self.onchain_df.columns:
                print(f"  Block range: {self.onchain_df['block_number'].min()} — {self.onchain_df['block_number'].max()}")

        if self.cex_df is not None:
            print(f"\nCEX data: {len(self.cex_df):,} records")
            print(f"  Source: {self.metadata.get('cex_source', 'unknown')}")
            print(f"  Price range: {self.cex_df['close'].min():.2f} — {self.cex_df['close'].max():.2f}")

        if self.features_df is not None:
            print(f"\nFeatures: {len(self.features_df):,} rows × {len(FEATURE_COLUMNS)} features")
            print(f"  Feature columns: {FEATURE_COLUMNS}")

        for split in ["train", "val", "test"]:
            size = self.metadata.get(f"{split}_size", "?")
            regimes = self.metadata.get(f"{split}_regimes", {})
            print(f"\n  {split.upper()}: {size} rows")
            if regimes:
                for regime, count in regimes.items():
                    print(f"    {regime}: {count:,}")

        print("\n" + "=" * 70)
