"""
Tests for the tick data pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.tick_loader import (
    generate_synthetic_data,
    compute_rolling_volatility,
    compute_ema_volatility,
    TICK_DATA_COLUMNS,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic tick data generation."""

    def test_generates_correct_size(self):
        df = generate_synthetic_data(n_records=1000, seed=42)
        assert len(df) == 1000

    def test_has_required_columns(self):
        df = generate_synthetic_data(n_records=100, seed=42)
        for col in TICK_DATA_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_prices_positive(self):
        df = generate_synthetic_data(n_records=10_000, seed=42)
        assert (df["price"] > 0).all()

    def test_deterministic_with_seed(self):
        df1 = generate_synthetic_data(n_records=100, seed=42)
        df2 = generate_synthetic_data(n_records=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        df1 = generate_synthetic_data(n_records=100, seed=42)
        df2 = generate_synthetic_data(n_records=100, seed=123)
        assert not df1["price"].equals(df2["price"])

    def test_volatility_affects_price_range(self):
        df_low = generate_synthetic_data(n_records=10_000, volatility=0.001, seed=42)
        df_high = generate_synthetic_data(n_records=10_000, volatility=0.05, seed=42)

        range_low = df_low["price"].max() - df_low["price"].min()
        range_high = df_high["price"].max() - df_high["price"].min()

        assert range_high > range_low

    def test_block_numbers_monotonic(self):
        df = generate_synthetic_data(n_records=100, seed=42)
        assert (df["block_number"].diff().dropna() > 0).all()

    def test_timestamps_monotonic(self):
        df = generate_synthetic_data(n_records=100, seed=42)
        assert (df["timestamp"].diff().dropna() > 0).all()

    def test_large_dataset_performance(self):
        """Can generate 1M records without errors."""
        df = generate_synthetic_data(n_records=1_000_000, seed=42)
        assert len(df) == 1_000_000


class TestVolatilityComputation:
    """Tests for volatility calculations."""

    def test_rolling_volatility_shape(self):
        df = generate_synthetic_data(n_records=1000, seed=42)
        vol = compute_rolling_volatility(df, window=100)
        assert len(vol) == 1000

    def test_rolling_volatility_positive(self):
        df = generate_synthetic_data(n_records=1000, seed=42)
        vol = compute_rolling_volatility(df, window=100)
        # After warm-up, should be positive
        assert (vol.iloc[100:] > 0).all()

    def test_ema_volatility_shape(self):
        df = generate_synthetic_data(n_records=1000, seed=42)
        ema = compute_ema_volatility(df, alpha=0.1)
        assert len(ema) == 1000

    def test_ema_volatility_non_negative(self):
        df = generate_synthetic_data(n_records=1000, seed=42)
        ema = compute_ema_volatility(df, alpha=0.1)
        assert (ema >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
