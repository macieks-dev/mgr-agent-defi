from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data import cex_fetcher


def _one_second_kline(day: str) -> pd.DataFrame:
    open_time = int(
        datetime.strptime(day, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    return pd.DataFrame(
        {
            "open_time": [open_time],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [2.0],
            "quote_volume": [201.0],
            "trades": [1],
            "taker_buy_base": [1.0],
            "taker_buy_quote": [100.5],
        }
    )


class TestFetchBinanceBlockKlines:
    def test_checkpoint_interval_is_not_skipped_by_batch_sizes(self, monkeypatch, tmp_path):
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=39)
        saved_record_counts = []

        monkeypatch.setattr(
            cex_fetcher,
            "_download_daily_klines_zip",
            lambda symbol, day, interval, session: _one_second_kline(day),
        )
        monkeypatch.setattr(
            cex_fetcher,
            "_save_checkpoint",
            lambda chunks, path: saved_record_counts.append(sum(len(chunk) for chunk in chunks)),
        )

        result = cex_fetcher.fetch_binance_block_klines(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            max_workers=4,
            checkpoint_every=30,
            checkpoint_dir=tmp_path,
        )

        assert len(result) == 40
        assert saved_record_counts == [40]

    def test_failed_days_raise_and_keep_successful_checkpoint(
        self, monkeypatch, tmp_path
    ):
        start = datetime(2024, 1, 1)
        end = start + timedelta(days=4)
        checkpoint_file = tmp_path / "ETHUSDT_12s_checkpoint.parquet"
        output_file = tmp_path / "prices.parquet"

        def download_day(symbol, day, interval, session):
            if day == "2024-01-03":
                return None
            return _one_second_kline(day)

        def save_checkpoint(chunks, path):
            path.write_text("successful partial download", encoding="utf-8")

        monkeypatch.setattr(cex_fetcher, "_download_daily_klines_zip", download_day)
        monkeypatch.setattr(cex_fetcher, "_save_checkpoint", save_checkpoint)

        with pytest.raises(RuntimeError, match="Failed to fetch 1 of 5 requested days"):
            cex_fetcher.fetch_binance_block_klines(
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                save_path=output_file,
                max_workers=1,
                checkpoint_dir=tmp_path,
            )

        assert checkpoint_file.exists()
        assert not output_file.exists()

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"max_workers": 0}, "max_workers must be at least 1"),
            ({"checkpoint_every": 0}, "checkpoint_every must be at least 1"),
            (
                {"start_date": "2024-01-02", "end_date": "2024-01-01"},
                "end_date must be on or after start_date",
            ),
        ],
    )
    def test_rejects_invalid_download_ranges_and_options(self, options, message, tmp_path):
        arguments = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "checkpoint_dir": tmp_path,
        }
        arguments.update(options)

        with pytest.raises(ValueError, match=message):
            cex_fetcher.fetch_binance_block_klines(**arguments)
