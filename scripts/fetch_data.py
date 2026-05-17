#!/usr/bin/env python3
"""
CLI do uruchamiania pełnego pipeline'u danych.

Użycie:
    # Wewnątrz Dockera:
    docker compose run --rm app python scripts/fetch_data.py

    # Lub lokalnie:
    python scripts/fetch_data.py --synthetic-only

    # Z pełnym pobieraniem (CEX + subgraph):
    python scripts/fetch_data.py --fetch-cex --fetch-onchain

    # Tylko feature engineering (dane już pobrane):
    python scripts/fetch_data.py --features-only

Opcje:
    --synthetic-only    Używaj tylko danych syntetycznych (bez API)
    --fetch-cex         Pobierz dane CEX z Binance
    --fetch-onchain     Spróbuj pobrać on-chain z The Graph
    --features-only     Tylko feature engineering (dane w data/raw/)
    --n-records N       Liczba rekordów syntetycznych (default: 1_000_000)
    --verbose           Szczegółowe logowanie
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_builder import DatasetBuilder, DataSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline danych dla Uniswap V4 RL Hook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Nie pobieraj z API — używaj danych syntetycznych",
    )
    parser.add_argument(
        "--fetch-cex",
        action="store_true",
        default=True,
        help="Pobierz dane CEX z Binance (domyślnie włączone)",
    )
    parser.add_argument(
        "--no-cex",
        action="store_true",
        help="Nie pobieraj danych CEX",
    )
    parser.add_argument(
        "--fetch-onchain",
        action="store_true",
        help="Spróbuj pobrać on-chain dane z The Graph subgraph",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Pomiń pobieranie, buduj cechy z istniejących plików",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=1_000_000,
        help="Liczba rekordów syntetycznych (default: 1_000_000)",
    )
    parser.add_argument(
        "--train-start", default="2023-07-01",
        help="Początek zbioru treningowego (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end", default="2025-03-31",
        help="Koniec zbioru treningowego",
    )
    parser.add_argument(
        "--val-start", default="2025-04-01",
        help="Początek zbioru walidacyjnego",
    )
    parser.add_argument(
        "--val-end", default="2025-09-30",
        help="Koniec zbioru walidacyjnego",
    )
    parser.add_argument(
        "--test-start", default="2025-10-01",
        help="Początek zbioru testowego",
    )
    parser.add_argument(
        "--test-end", default="2026-01-31",
        help="Koniec zbioru testowego",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Szczegółowe logowanie (DEBUG)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("fetch_data")

    # Print banner
    print("=" * 70)
    print("  UNISWAP V4 RL HOOK — DATA PIPELINE")
    print("=" * 70)

    # Create split definition
    split = DataSplit(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    builder = DatasetBuilder(split=split)

    t0 = time.time()

    # ── Step 1: Fetch data ──
    if args.features_only:
        logger.info("FEATURES ONLY mode — loading existing raw data...")
        raw_path = builder.raw_dir / "uniswap_v3_swaps.parquet"
        if raw_path.exists():
            import pandas as pd
            builder.onchain_df = pd.read_parquet(raw_path)
            builder.metadata["onchain_source"] = "cached_file"
        else:
            logger.error(f"No raw data found at {raw_path}. Run without --features-only first.")
            sys.exit(1)

        cex_path = builder.raw_dir / "binance_ethusdt_1m.parquet"
        if cex_path.exists():
            import pandas as pd
            builder.cex_df = pd.read_parquet(cex_path)
            builder.metadata["cex_source"] = "cached_file"
    else:
        use_subgraph = args.fetch_onchain and not args.synthetic_only
        synthetic_fallback = True

        if args.synthetic_only:
            logger.info("SYNTHETIC ONLY mode — generating synthetic data...")
            use_subgraph = False
        elif args.fetch_onchain:
            logger.info("FULL FETCH mode — trying subgraph + Binance...")

        builder.fetch_all(
            use_subgraph=use_subgraph,
            synthetic_fallback=synthetic_fallback,
            n_synthetic=args.n_records,
        )

    # ── Step 2: Build features ──
    logger.info("Building features...")
    builder.build_features()

    # ── Step 3: Split & save ──
    logger.info("Splitting and saving...")
    builder.split_and_save()

    elapsed = time.time() - t0

    # ── Print summary ──
    builder.print_summary()

    print(f"\nPipeline completed in {elapsed:.1f}s")
    print(f"Output directory: {builder.processed_dir}")
    print()

    # Verify output files
    print("Generated files:")
    for f in sorted(builder.processed_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
