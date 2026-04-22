"""
research/ml_features.py

Builds the labelled dataset for the Phase 5 ML signal layer.

For every A_6M signal in the 15-year backtest, computes features available
strictly at signal date (no lookahead) and assigns a binary label:

    label = 1 if (return_pct - benchmark_return) > 10 percentage points
    label = 0 otherwise

This dataset is training data for the XGBoost false-positive filter.
It is NOT the model. The model is built after paper trading gate passes.

Output: research/data/ml_labels.parquet

Usage:
    python research/ml_features.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import the lookahead firewall directly from the backtest engine.
# Do not reimplement these functions — the same logic that protects
# the backtest protects the feature engineering.
from backtest.engine import get_prices_as_of, get_funds_as_of

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_THRESHOLD = 10.0   # Label = 1 if alpha > this many percentage points
OUTPUT_DIR      = ROOT / "research" / "data"
OUTPUT_FILE     = OUTPUT_DIR / "ml_labels.parquet"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    log.info("Loading trades, prices, fundamentals...")

    trades = pl.read_parquet(ROOT / "backtest" / "trades_A_6M.parquet")
    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    funds  = pl.read_parquet(ROOT / "data" / "cache" / "fundamentals_edgar.parquet")

    log.info(f"Trades:         {trades.shape}")
    log.info(f"Prices:         {prices.shape}")
    log.info(f"Fundamentals:   {funds.shape}")

    return trades, prices, funds


# ── Label construction ────────────────────────────────────────────────────────

def build_labels(trades: pl.DataFrame) -> pl.DataFrame:
    """
    Compute alpha and assign binary label.
    Alpha = stock return - benchmark return over the hold period.
    Label = 1 if alpha > ALPHA_THRESHOLD, 0 otherwise.
    """
    return trades.with_columns([
        (pl.col("return_pct") - pl.col("benchmark_return")).alias("alpha"),
        ((pl.col("return_pct") - pl.col("benchmark_return")) > ALPHA_THRESHOLD)
            .cast(pl.Int8)
            .alias("label")
    ])


# ── Price features ────────────────────────────────────────────────────────────

def compute_price_features(
    ticker: str,
    signal_date: datetime,
    prices: pl.DataFrame
) -> dict:
    """
    Compute price-based features for one ticker at one signal date.
    Uses get_prices_as_of to enforce the lookahead firewall.
    All features use only data strictly before signal_date.
    """
    # Filter to this ticker, strictly before signal date
    px = get_prices_as_of(prices, signal_date).filter(
        pl.col("ticker") == ticker
    ).sort("date")

    # Need at least 200 rows for 200MA — return nulls if insufficient
    if len(px) < 200:
        return {
            "momentum_6m":        None,
            "momentum_3m":        None,
            "momentum_1m":        None,
            "above_200ma":        None,
            "pct_below_52w_high": None,
            "rsi_14":             None,
            "volume_trend_20d":   None,
            "spy_above_200ma":    None,
        }

    close  = px["close"].to_numpy()
    volume = px["volume"].to_numpy()

    # Momentum — returns over lookback windows
    # 126 trading days ≈ 6 months, 63 ≈ 3 months, 21 ≈ 1 month
    def momentum(n: int) -> float | None:
        if len(close) < n + 1:
            return None
        return float((close[-1] / close[-n] - 1) * 100)

    momentum_6m = momentum(126)
    momentum_3m = momentum(63)
    momentum_1m = momentum(21)

    # 200-day moving average
    ma_200      = float(np.mean(close[-200:]))
    above_200ma = int(close[-1] > ma_200)

    # % below 52-week high (252 trading days)
    high_252 = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    pct_below_52w_high = float((close[-1] / high_252 - 1) * 100)

    # RSI(14) — relative strength index
    # Measures momentum quality: overbought (>70) vs oversold (<30)
    if len(close) >= 15:
        deltas = np.diff(close[-15:])
        gains  = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs     = avg_gain / avg_loss
            rsi_14 = float(100 - (100 / (1 + rs)))
    else:
        rsi_14 = None

    # Volume trend — slope of 20-day volume (positive = expanding)
    if len(volume) >= 20:
        vol_20      = volume[-20:].astype(float)
        x           = np.arange(len(vol_20))
        volume_trend_20d = float(np.polyfit(x, vol_20, 1)[0])
    else:
        volume_trend_20d = None

    return {
        "momentum_6m":        momentum_6m,
        "momentum_3m":        momentum_3m,
        "momentum_1m":        momentum_1m,
        "above_200ma":        above_200ma,
        "pct_below_52w_high": pct_below_52w_high,
        "rsi_14":             rsi_14,
        "volume_trend_20d":   volume_trend_20d,
    }


# ── Market regime features ────────────────────────────────────────────────────

def compute_regime_features(
    signal_date: datetime,
    prices: pl.DataFrame
) -> dict:
    """
    Compute market-wide regime features at signal date.
    Uses SPY as the market proxy.
    """
    spy_px = get_prices_as_of(prices, signal_date).filter(
        pl.col("ticker") == "SPY"
    ).sort("date")

    if len(spy_px) < 200:
        return {"spy_above_200ma": None, "spy_momentum_6m": None}

    spy_close    = spy_px["close"].to_numpy()
    spy_ma200    = float(np.mean(spy_close[-200:]))
    spy_above_200ma   = int(spy_close[-1] > spy_ma200)
    spy_momentum_6m   = float((spy_close[-1] / spy_close[-126] - 1) * 100) \
                        if len(spy_close) >= 126 else None

    return {
        "spy_above_200ma":  spy_above_200ma,
        "spy_momentum_6m":  spy_momentum_6m,
    }


# ── Fundamental features ──────────────────────────────────────────────────────

def compute_fundamental_features(
    ticker: str,
    signal_date: datetime,
    funds: pl.DataFrame
) -> dict:
    """
    Compute fundamental features for one ticker at one signal date.
    Uses get_funds_as_of to enforce point-in-time filing dates.
    Only EDGAR data filed strictly before signal_date is used.
    """
    empty = {
        "gross_margin":        None,
        "net_margin":          None,
        "revenue_growth":      None,
        "net_income_trend":    None,
        "asset_liability_ratio": None,
        "operating_cf_trend":  None,
    }

    f = get_funds_as_of(funds, signal_date).filter(
        pl.col("ticker") == ticker
    ).sort("period_end")

    # Need at least 2 quarters to compute trends
    if len(f) < 2:
        return empty

    # Most recent quarter
    latest = f[-1]

    gross_margin = float(latest["gross_margin"][0]) \
                   if latest["gross_margin"][0] is not None else None
    net_margin   = float(latest["net_margin"][0]) \
                   if latest["net_margin"][0] is not None else None

    # Revenue growth — newest vs oldest available quarter
    rev_new = latest["revenue"][0]
    rev_old = f[0]["revenue"][0]
    revenue_growth = float((rev_new / rev_old - 1) * 100) \
                     if (rev_new and rev_old and rev_old != 0) else None

    # Net income trend — positive = improving
    ni_new = latest["net_income"][0]
    ni_old = f[0]["net_income"][0]
    net_income_trend = int(ni_new > ni_old) \
                       if (ni_new is not None and ni_old is not None) else None

    # Asset/liability ratio — latest
    assets = latest["total_assets"][0]
    liabs  = latest["total_liabilities"][0]
    asset_liability_ratio = float(assets / liabs) \
                            if (assets and liabs and liabs != 0) else None

    # Operating cashflow trend
    cf_new = latest["operating_cashflow"][0]
    cf_old = f[0]["operating_cashflow"][0]
    operating_cf_trend = int(cf_new > cf_old) \
                         if (cf_new is not None and cf_old is not None) else None

    return {
        "gross_margin":          gross_margin,
        "net_margin":            net_margin,
        "revenue_growth":        revenue_growth,
        "net_income_trend":      net_income_trend,
        "asset_liability_ratio": asset_liability_ratio,
        "operating_cf_trend":    operating_cf_trend,
    }


# ── Main build loop ───────────────────────────────────────────────────────────

def build_dataset(trades: pl.DataFrame, prices: pl.DataFrame, funds: pl.DataFrame) -> pl.DataFrame:
    """
    For every signal in the trades file, compute all features at signal_date.
    Returns a DataFrame with one row per signal, all features, and the label.
    """
    rows = []
    total = len(trades)

    for i, row in enumerate(trades.iter_rows(named=True)):
        if i % 200 == 0:
            log.info(f"Processing signal {i+1}/{total} ({i/total*100:.0f}%)")

        ticker      = row["ticker"]
        signal_date = row["signal_date"]

        # Price features — point-in-time via get_prices_as_of
        price_feats  = compute_price_features(ticker, signal_date, prices)

        # Regime features — SPY state at signal date
        regime_feats = compute_regime_features(signal_date, prices)

        # Fundamental features — point-in-time via get_funds_as_of
        fund_feats   = compute_fundamental_features(ticker, signal_date, funds)

        # Combine everything into one row
        record = {
            # Identifiers
            "ticker":      ticker,
            "signal_date": signal_date,
            "label":       row["label"],
            "alpha":       row["alpha"],
            # From trades — already available at signal time
            "score":       row["score"],
            # Price features
            **price_feats,
            # Regime features
            **regime_feats,
            # Fundamental features
            **fund_feats,
        }

        rows.append(record)

    return pl.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    trades, prices, funds = load_data()

    # Build label column
    trades = build_labels(trades)

    # Log class balance
    total    = len(trades)
    positive = trades["label"].sum()
    log.info(f"Label distribution: {positive}/{total} positive ({positive/total*100:.1f}%)")

    # Build feature dataset
    log.info("Building feature dataset — this will take a few minutes...")
    dataset = build_dataset(trades, prices, funds)

    # Drop rows with too many nulls — need at least price features
    dataset = dataset.drop_nulls(subset=["momentum_6m", "above_200ma", "rsi_14"])

    log.info(f"Dataset shape after dropping nulls: {dataset.shape}")
    log.info(f"Final label balance: {dataset['label'].sum()}/{len(dataset)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(OUTPUT_FILE)
    log.info(f"Saved to {OUTPUT_FILE}")

    # Print feature summary
    print("\nFeature summary:")
    print(dataset.describe())

    print("\nLabel distribution:")
    print(dataset["label"].value_counts())

    print("\nTop 10 rows:")
    print(dataset.select([
        "ticker", "signal_date", "label", "alpha",
        "score", "momentum_6m", "above_200ma", "gross_margin"
    ]).head(10))


if __name__ == "__main__":
    main()