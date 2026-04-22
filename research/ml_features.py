"""
research/ml_features.py

Builds the labelled dataset for the Phase 5 ML signal layer.

For every A_6M signal in the 15-year backtest, computes features available
strictly at signal date (no lookahead) and assigns a binary label:

    label = 1 if (return_pct - benchmark_return) > 10 percentage points
    label = 0 otherwise

Restructured to iterate by unique signal date (772) rather than by signal
(3,882) — calls price_features() and fundamental_features_edgar() once per
date across all tickers, then filters to the signals that fired that date.
This is ~5x faster than the original per-signal loop.

Output: research/data/ml_labels.parquet

Usage:
    python research/ml_features.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import polars as pl
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.engine import get_prices_as_of, get_funds_as_of
from screener.features import price_features, fundamental_features_edgar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_THRESHOLD = 10.0
OUTPUT_DIR      = ROOT / "research" / "data"
OUTPUT_FILE     = OUTPUT_DIR / "ml_labels.parquet"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    log.info("Loading trades, prices, fundamentals...")
    trades = pl.read_parquet(ROOT / "backtest" / "trades_A_6M.parquet")
    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    funds  = pl.read_parquet(ROOT / "data" / "cache" / "fundamentals_edgar.parquet")
    log.info(f"Trades: {trades.shape}  Prices: {prices.shape}  Funds: {funds.shape}")
    return trades, prices, funds

# ── Label construction ────────────────────────────────────────────────────────

def build_labels(trades: pl.DataFrame) -> pl.DataFrame:
    return trades.with_columns([
        (pl.col("return_pct") - pl.col("benchmark_return")).alias("alpha"),
        ((pl.col("return_pct") - pl.col("benchmark_return")) > ALPHA_THRESHOLD)
            .cast(pl.Int8)
            .alias("label")
    ])

# ── Volatility features ───────────────────────────────────────────────────────

def compute_volatility_features(px_ticker: pl.DataFrame) -> dict:
    """
    Compute stock-level volatility from single-ticker price series.
    px_ticker: prices for one ticker strictly before signal date, sorted by date.
    """
    if len(px_ticker) < 21:
        return {"stock_vol_20d": None, "stock_vol_63d": None, "vol_ratio": None}

    close   = px_ticker["close"].to_numpy()
    returns = np.diff(close) / close[:-1]

    vol_20d   = float(np.std(returns[-20:])) if len(returns) >= 20 else None
    vol_63d   = float(np.std(returns[-63:])) if len(returns) >= 63 else None
    vol_ratio = float(vol_20d / vol_63d) if (vol_20d and vol_63d and vol_63d > 0) else None

    return {
        "stock_vol_20d": vol_20d,
        "stock_vol_63d": vol_63d,
        "vol_ratio":     vol_ratio,
    }

# ── Regime features ───────────────────────────────────────────────────────────

def compute_regime_features(px_spy: pl.DataFrame) -> dict:
    """
    Compute market regime features from SPY prices strictly before signal date.
    """
    empty = {
        "market_vol_20d":  None,
        "spy_momentum_3m": None,
        "spy_momentum_1m": None,
        "market_drawdown": None,
    }

    if len(px_spy) < 200:
        return empty

    close   = px_spy["close"].to_numpy()
    returns = np.diff(close) / close[:-1]

    market_vol_20d  = float(np.std(returns[-20:])) if len(returns) >= 20 else None
    spy_momentum_3m = float((close[-1] / close[-63]  - 1) * 100) if len(close) >= 63  else None
    spy_momentum_1m = float((close[-1] / close[-21]  - 1) * 100) if len(close) >= 21  else None
    high_252        = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    market_drawdown = float((close[-1] / high_252 - 1) * 100)

    return {
        "market_vol_20d":  market_vol_20d,
        "spy_momentum_3m": spy_momentum_3m,
        "spy_momentum_1m": spy_momentum_1m,
        "market_drawdown": market_drawdown,
    }

# ── Time/cycle features ───────────────────────────────────────────────────────

def compute_time_features(signal_date: datetime) -> dict:
    """
    Calendar and cycle features from signal date.
    month: seasonality effects
    year_normalized: position in sample 0=2011 to 1=2026
    """
    return {
        "month":           signal_date.month,
        "year_normalized": round((signal_date.year - 2011) / (2026 - 2011), 4),
    }

# ── Main build loop ───────────────────────────────────────────────────────────

def build_dataset(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
    funds:  pl.DataFrame,
) -> pl.DataFrame:
    """
    Iterate by unique signal date (772 iterations instead of 3,882).
    For each date: compute features once for all tickers, join onto signals.
    """
    signal_dates = trades["signal_date"].unique().sort().to_list()
    total        = len(signal_dates)
    rows         = []

    # Pre-filter SPY for regime features
    spy_all = prices.filter(pl.col("ticker") == "SPY").sort("date")

    for i, signal_date in enumerate(signal_dates):
        if i % 100 == 0:
            log.info(
                f"Processing date {i+1}/{total} — "
                f"{signal_date.date()} ({i/total*100:.0f}%)"
            )

        # Signals on this date
        signals_today = trades.filter(pl.col("signal_date") == signal_date)

        # Point-in-time price data — all tickers
        px_pit = get_prices_as_of(prices, signal_date)

        # Price features — one call for all tickers
        try:
            pf = price_features(px_pit)
        except Exception as e:
            log.warning(f"price_features failed on {signal_date}: {e}")
            pf = pl.DataFrame()

        # Fundamental features — one call for all tickers
        try:
            ff = fundamental_features_edgar(funds, signal_date)
        except Exception as e:
            log.warning(f"fundamental_features_edgar failed on {signal_date}: {e}")
            ff = pl.DataFrame()

        # Regime features — SPY only
        spy_pit = spy_all.filter(pl.col("date") < signal_date)
        regime  = compute_regime_features(spy_pit)

        # Time features
        time_feats = compute_time_features(signal_date)

        # Per-ticker assembly
        for row in signals_today.iter_rows(named=True):
            ticker = row["ticker"]

            # Price features for this ticker
            pf_ticker = {}
            if len(pf) > 0 and "ticker" in pf.columns:
                pf_row = pf.filter(pl.col("ticker") == ticker)
                if len(pf_row) > 0:
                    pf_ticker = pf_row.to_dicts()[0]
                    pf_ticker.pop("ticker", None)

            # Fundamental features for this ticker
            ff_ticker = {}
            if len(ff) > 0 and "ticker" in ff.columns:
                ff_row = ff.filter(pl.col("ticker") == ticker)
                if len(ff_row) > 0:
                    ff_ticker = ff_row.to_dicts()[0]
                    ff_ticker.pop("ticker", None)

            # Volatility — single ticker price series
            px_ticker = px_pit.filter(pl.col("ticker") == ticker).sort("date")
            vol_feats = compute_volatility_features(px_ticker)

            record = {
                # Identifiers
                "ticker":      ticker,
                "signal_date": signal_date,
                "label":       row["label"],
                "alpha":       row["alpha"],
                "score":       row["score"],
                # Price features
                **pf_ticker,
                # Fundamental features
                **ff_ticker,
                # Volatility
                **vol_feats,
                # Regime
                **regime,
                # Time/cycle
                **time_feats,
            }

            rows.append(record)

    return pl.DataFrame(rows)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    trades, prices, funds = load_data()
    trades = build_labels(trades)

    total    = len(trades)
    positive = trades["label"].sum()
    log.info(f"Label distribution: {positive}/{total} positive ({positive/total*100:.1f}%)")

    log.info("Building feature dataset — this will take several minutes...")
    dataset = build_dataset(trades, prices, funds)

    log.info(f"Raw dataset shape: {dataset.shape}")

    # Cap revenue growth outliers
    if "revenue_growth_yoy" in dataset.columns:
        dataset = dataset.with_columns(
            pl.col("revenue_growth_yoy").clip(-100, 500)
        )

    # Drop rows missing core price features
    core = [c for c in ["momentum_6m", "above_200ma", "rsi_14"] if c in dataset.columns]
    if core:
        dataset = dataset.drop_nulls(subset=core)

    log.info(f"Final dataset shape: {dataset.shape}")
    log.info(f"Final label balance: {dataset['label'].sum()}/{len(dataset)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(OUTPUT_FILE)
    log.info(f"Saved to {OUTPUT_FILE}")

    # Summary
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())

    print(f"\nAll features ({len(dataset.columns)} total):")
    for col in sorted(dataset.columns):
        null_pct = dataset[col].null_count() / len(dataset) * 100
        print(f"  {col:<40} nulls: {null_pct:.1f}%")

    print("\nSample rows:")
    print(dataset.select([
        "ticker", "signal_date", "label", "alpha", "score",
        "momentum_6m", "stock_vol_20d", "market_vol_20d", "month"
    ]).head(5))


if __name__ == "__main__":
    main()