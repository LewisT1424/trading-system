"""
screener/features.py
Computes all features needed for the screener and ICD criteria scoring.

Two outputs:
    price_features(prices)       — one row per ticker, price-based features
    fundamental_features(funds)  — one row per ticker, fundamental-based features

Both are joined in screener/run.py to produce the full feature set.

ICD criteria mapping:
    C1 — larger than priced        → asset_to_mcap_ratio
    C2 — consistent 3 of 4 qtrs   → revenue_consistency
    C3 — recovery trajectory       → revenue_trajectory, margin_trajectory
    C4 — assets up, liabilities ↔  → asset_growth, liability_growth
    C5 — dip from prior high       → pct_below_52w_high
    C6 — sector tailwind           → sector_rs (relative strength vs SPY)
    C7 — moat / pricing power      → gross_margin_level
"""

import logging
import polars as pl
import polars.selectors as cs
from pathlib import Path

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


# ── Price features ────────────────────────────────────────────────────────────

def price_features(prices: pl.DataFrame) -> pl.DataFrame:
    """
    Compute price-based features for every ticker.
    Input:  long-format prices (ticker, date, open, high, low, close, volume)
    Output: one row per ticker with all price features

    Features:
        momentum_6m         — 6-month price return (126 trading days)
        momentum_1m         — 1-month price return (21 trading days)
        above_200ma         — bool: close > 200-day moving average
        rsi_14              — RSI(14)
        volume_trend_50d    — slope of 50-day volume (positive = increasing)
        pct_below_52w_high  — % below 52-week high (C5)
        sector_rs           — placeholder, filled in join with sector data
    """
    log.info("Computing price features...")

    prices = prices.sort(["ticker", "date"])

    # Compute per-ticker rolling features
    prices = prices.with_columns([
        # 200-day MA
        pl.col("close")
          .rolling_mean(window_size=200, min_periods=150)
          .over("ticker")
          .alias("ma_200"),

        # 52-week high (252 trading days)
        pl.col("close")
          .rolling_max(window_size=252, min_periods=200)
          .over("ticker")
          .alias("high_52w"),

        # 50-day volume mean for trend
        pl.col("volume")
          .cast(pl.Float64)
          .rolling_mean(window_size=50, min_periods=40)
          .over("ticker")
          .alias("vol_ma_50"),
    ])

    # RSI(14) — compute gain/loss series then rolling means
    prices = _add_rsi(prices, period=14)

    # Get the most recent row per ticker — use sort-aware approach
    prices = prices.with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn_pf"))
    latest = (
        prices.group_by("ticker")
        .agg(pl.col("_rn_pf").max().alias("_max_rn"), pl.all().last())
        .drop("_rn_pf", "_max_rn")
        .sort("ticker")
    )
    prices = prices.drop("_rn_pf")

    # 6M momentum: close today vs close 126 days ago
    momentum_6m = _momentum(prices, lookback=126)
    momentum_1m = _momentum(prices, lookback=21)

    # Volume trend: positive slope over last 50 days (simple: recent vs earlier)
    vol_trend = _volume_trend(prices, window=50)

    # Join everything onto latest
    result = (
        latest
        .join(momentum_6m, on="ticker", how="left")
        .join(momentum_1m, on="ticker", how="left", suffix="_1m")
        .join(vol_trend,   on="ticker", how="left")
        .with_columns([
            # C5 — % below 52-week high
            pl.when(pl.col("high_52w") > 0)
              .then((pl.col("close") - pl.col("high_52w")) / pl.col("high_52w") * 100)
              .otherwise(None)
              .alias("pct_below_52w_high"),

            # C5 boolean — in a dip (>5% below 52w high)
            pl.when(pl.col("high_52w") > 0)
              .then((pl.col("close") / pl.col("high_52w")) < 0.95)
              .otherwise(False)
              .alias("in_dip"),

            # 200MA crossover boolean
            (pl.col("close") > pl.col("ma_200")).alias("above_200ma"),
        ])
        .select([
            "ticker",
            "close",
            "ma_200",
            "above_200ma",
            "high_52w",
            "pct_below_52w_high",
            "in_dip",
            "rsi_14",
            "momentum_6m",
            "momentum_1m",
            "volume_trend_50d",
        ])
    )

    log.info(f"Price features computed: {len(result)} tickers")
    return result


def _add_rsi(prices: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add RSI column computed over the given period."""
    prices = prices.with_columns([
        pl.col("close").diff().over("ticker").alias("_delta")
    ])
    prices = prices.with_columns([
        pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0).alias("_gain"),
        pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0).alias("_loss"),
    ])
    prices = prices.with_columns([
        pl.col("_gain").rolling_mean(window_size=period, min_periods=period).over("ticker").alias("_avg_gain"),
        pl.col("_loss").rolling_mean(window_size=period, min_periods=period).over("ticker").alias("_avg_loss"),
    ])
    prices = prices.with_columns([
        pl.when(pl.col("_avg_loss") == 0)
          .then(100.0)
          .otherwise(100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
          .alias("rsi_14")
    ])
    return prices.drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def _momentum(prices: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """
    Return a DataFrame with ticker and momentum_Nd column.
    Momentum = (current_close - close_N_days_ago) / close_N_days_ago * 100
    """
    col_name = f"momentum_{lookback // 21}m" if lookback >= 21 else f"momentum_{lookback}d"

    prices_sorted = prices.sort(["ticker", "date"])

    latest = (
        prices_sorted
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .group_by("ticker").agg(pl.col("_rn").max().alias("_max_rn"), pl.all().last())
        .drop("_rn", "_max_rn")
        .select(["ticker", pl.col("close").alias("close_now")])
    )

    past = (
        prices_sorted
        .group_by("ticker")
        .agg(
            pl.col("close")
              .get(pl.len() - lookback - 1)
              .alias("close_past")
        )
    )

    return (
        latest
        .join(past, on="ticker", how="left")
        .with_columns([
            pl.when(pl.col("close_past").is_not_null() & (pl.col("close_past") > 0))
              .then((pl.col("close_now") - pl.col("close_past")) / pl.col("close_past") * 100)
              .otherwise(None)
              .alias(col_name)
        ])
        .select(["ticker", col_name])
    )


def _volume_trend(prices: pl.DataFrame, window: int = 50) -> pl.DataFrame:
    """
    Simple volume trend: mean volume in last 25 days vs prior 25 days.
    Positive = volume increasing. Returned as a ratio (recent / prior).
    """
    half = window // 2

    result = (
        prices
        .sort(["ticker", "date"])
        .group_by("ticker")
        .agg([
            pl.col("volume").tail(half).mean().alias("vol_recent"),
            pl.col("volume").slice(-window, half).mean().alias("vol_prior"),
        ])
        .with_columns([
            pl.when(
                pl.col("vol_prior").is_not_null() & (pl.col("vol_prior") > 0)
            )
            .then(pl.col("vol_recent") / pl.col("vol_prior"))
            .otherwise(None)
            .alias("volume_trend_50d")
        ])
        .select(["ticker", "volume_trend_50d"])
    )
    return result


# ── Fundamental features ──────────────────────────────────────────────────────

def fundamental_features(funds: pl.DataFrame) -> pl.DataFrame:
    """
    Compute fundamental-based features for every ticker.
    Input:  long-format fundamentals (ticker, period, revenue, ...)
            periods are newest-first strings like '2025-12-31'
    Output: one row per ticker with all fundamental features

    Features:
        revenue_growth_yoy      — YoY revenue growth (newest vs 4 quarters ago)
        gross_margin_latest     — gross margin in most recent quarter (C7)
        gross_margin_avg        — average gross margin across 4 quarters
        fcf_margin              — FCF / revenue
        asset_to_mcap_ratio     — total assets / market cap (C1)
        asset_growth            — QoQ asset growth rate (C4)
        liability_growth        — QoQ liability growth rate (C4)
        revenue_consistency     — fraction of quarters with positive revenue (C2)
        revenue_trajectory      — slope direction: +1 improving, -1 declining, 0 flat (C3)
        margin_trajectory       — slope direction of gross margin (C3)
        market_cap_bucket       — 0=micro, 1=small, 2=mid, 3=large, 4=mega
        trailing_pe             — from .info
        revenue_growth_ttm      — from .info
        fcf_yield               — free_cashflow / market_cap
    """
    log.info("Computing fundamental features...")

    # Sort newest first within each ticker — sort is the source of truth,
    # never use group_by().first/last without an explicit sort
    funds = (
        funds
        .with_columns(pl.col("period").str.to_date("%Y-%m-%d", strict=False).alias("period_dt"))
        .sort(["ticker", "period_dt"], descending=[False, True])
    )

    # Take the 4 most recent quarters per ticker — preserve sort order
    funds_4q = (
        funds
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_row_num"))
        .filter(pl.col("_row_num") < 4)
        .drop("_row_num")
    )

    # Latest quarter only — first row after sort = most recent
    latest_q = (
        funds
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_row_num"))
        .filter(pl.col("_row_num") == 0)
        .drop("_row_num")
    )

    # ── Per-ticker aggregations ────────────────────────────────────────────

    # Revenue growth YoY: newest quarter vs 4th quarter back
    rev_yoy = _revenue_growth_yoy(funds_4q)

    # Consistency: fraction of 4 quarters with positive revenue (C2)
    consistency = (
        funds_4q
        .group_by("ticker")
        .agg([
            (pl.col("revenue") > 0).sum().cast(pl.Float64).alias("_pos_rev_count"),
            pl.len().cast(pl.Float64).alias("_q_count"),
        ])
        .with_columns(
            (pl.col("_pos_rev_count") / pl.col("_q_count")).alias("revenue_consistency")
        )
        .select(["ticker", "revenue_consistency"])
    )

    # Trajectory: is the most recent quarter better than the oldest of the 4?
    trajectory = _compute_trajectory(funds_4q)

    # Asset and liability growth (C4): newest vs oldest of 4 quarters
    balance_growth = _balance_growth(funds_4q)

    # Gross margin average across 4 quarters
    margin_avg = (
        funds_4q
        .group_by("ticker")
        .agg(pl.col("gross_margin").mean().alias("gross_margin_avg"))
    )

    # FCF margin: FCF / revenue on latest quarter
    fcf_margin = (
        latest_q
        .with_columns([
            pl.when(
                pl.col("revenue").is_not_null() & (pl.col("revenue") > 0) &
                pl.col("free_cashflow").is_not_null()
            )
            .then(pl.col("free_cashflow") / pl.col("revenue"))
            .otherwise(None)
            .alias("fcf_margin")
        ])
        .select(["ticker", "fcf_margin"])
    )

    # Market cap bucket
    mcap_bucket = (
        latest_q
        .with_columns([
            pl.col("market_cap").cast(pl.Float64).alias("market_cap_f")
        ])
        .with_columns([
            pl.when(pl.col("market_cap_f") >= 200e9).then(4)
             .when(pl.col("market_cap_f") >= 10e9).then(3)
             .when(pl.col("market_cap_f") >= 2e9).then(2)
             .when(pl.col("market_cap_f") >= 300e6).then(1)
             .otherwise(0)
             .alias("market_cap_bucket")
        ])
        .select(["ticker", "market_cap_f", "market_cap_bucket"])
    )

    # Asset to market cap ratio (C1)
    asset_mcap = (
        latest_q
        .join(mcap_bucket, on="ticker", how="left")
        .with_columns([
            pl.when(
                pl.col("market_cap_f").is_not_null() & (pl.col("market_cap_f") > 0) &
                pl.col("total_assets").is_not_null()
            )
            .then(pl.col("total_assets") / pl.col("market_cap_f"))
            .otherwise(None)
            .alias("asset_to_mcap_ratio")
        ])
        .select(["ticker", "asset_to_mcap_ratio"])
    )

    # FCF yield
    fcf_yield = (
        latest_q
        .join(mcap_bucket, on="ticker", how="left")
        .with_columns([
            pl.when(
                pl.col("market_cap_f").is_not_null() & (pl.col("market_cap_f") > 0) &
                pl.col("free_cashflow").is_not_null()
            )
            .then(pl.col("free_cashflow") / pl.col("market_cap_f"))
            .otherwise(None)
            .alias("fcf_yield")
        ])
        .select(["ticker", "fcf_yield"])
    )

    # Base: latest quarter fundamentals + .info fields
    base = latest_q.select([
        "ticker",
        "gross_margin",
        "net_margin",
        "total_assets",
        "total_liabilities",
        "total_debt",
        "cash",
        "revenue",
        "operating_income",
        "net_income",
        "trailing_pe",
        "revenue_growth",
    ]).rename({
        "gross_margin": "gross_margin_latest",
        "revenue": "revenue_latest",
    })

    # Join everything
    result = (
        base
        .join(rev_yoy,       on="ticker", how="left")
        .join(consistency,   on="ticker", how="left")
        .join(trajectory,    on="ticker", how="left")
        .join(balance_growth, on="ticker", how="left")
        .join(margin_avg,    on="ticker", how="left")
        .join(fcf_margin,    on="ticker", how="left")
        .join(mcap_bucket.select(["ticker", "market_cap_f", "market_cap_bucket"]),
              on="ticker", how="left")
        .join(asset_mcap,    on="ticker", how="left")
        .join(fcf_yield,     on="ticker", how="left")
        .rename({"market_cap_f": "market_cap"})
    )

    log.info(f"Fundamental features computed: {len(result)} tickers")
    return result


def _revenue_growth_yoy(funds_4q: pl.DataFrame) -> pl.DataFrame:
    """YoY revenue: newest quarter vs same quarter last year (4th row back)."""
    newest = (
        funds_4q
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .filter(pl.col("_rn") == 0).drop("_rn")
        .select(["ticker", pl.col("revenue").alias("rev_new")])
    )
    oldest = (
        funds_4q
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .group_by("ticker").agg(pl.col("_rn").max().alias("_max_rn"), pl.all().last())
        .drop("_rn", "_max_rn")
        .select(["ticker", pl.col("revenue").alias("rev_old")])
    )
    return (
        newest.join(oldest, on="ticker", how="left")
        .with_columns([
            pl.when(
                pl.col("rev_old").is_not_null() & (pl.col("rev_old") > 0) &
                pl.col("rev_new").is_not_null()
            )
            .then((pl.col("rev_new") - pl.col("rev_old")) / pl.col("rev_old"))
            .otherwise(None)
            .alias("revenue_growth_yoy")
        ])
        .select(["ticker", "revenue_growth_yoy"])
    )


def _compute_trajectory(funds_4q: pl.DataFrame) -> pl.DataFrame:
    """
    Trajectory: compare newest quarter vs oldest of the 4.
    +1 = improving, -1 = declining, 0 = flat (within 5% tolerance)
    Returns revenue_trajectory and margin_trajectory.
    """
    newest = (
        funds_4q
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .filter(pl.col("_rn") == 0).drop("_rn")
        .select(["ticker", pl.col("revenue").alias("rev_new"), pl.col("gross_margin").alias("margin_new")])
    )
    oldest = (
        funds_4q.sort(["ticker", "period_dt"], descending=[False, False])
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .filter(pl.col("_rn") == 0).drop("_rn")
        .select(["ticker", pl.col("revenue").alias("rev_old"), pl.col("gross_margin").alias("margin_old")])
    )
    return (
        newest.join(oldest, on="ticker", how="left")
        .with_columns([
            pl.when(pl.col("rev_old").is_not_null() & (pl.col("rev_old") != 0))
              .then(
                  pl.when((pl.col("rev_new") - pl.col("rev_old")) / pl.col("rev_old").abs() > 0.05)
                    .then(1)
                    .when((pl.col("rev_new") - pl.col("rev_old")) / pl.col("rev_old").abs() < -0.05)
                    .then(-1)
                    .otherwise(0)
              )
              .otherwise(None)
              .alias("revenue_trajectory"),

            pl.when(pl.col("margin_old").is_not_null() & pl.col("margin_new").is_not_null())
              .then(
                  pl.when((pl.col("margin_new") - pl.col("margin_old")) > 0.02)
                    .then(1)
                    .when((pl.col("margin_new") - pl.col("margin_old")) < -0.02)
                    .then(-1)
                    .otherwise(0)
              )
              .otherwise(None)
              .alias("margin_trajectory"),
        ])
        .select(["ticker", "revenue_trajectory", "margin_trajectory"])
    )


def _balance_growth(funds_4q: pl.DataFrame) -> pl.DataFrame:
    """
    Asset and liability growth: newest vs oldest of 4 quarters.
    Positive = growing, negative = shrinking.
    """
    newest = (
        funds_4q
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .filter(pl.col("_rn") == 0).drop("_rn")
        .select(["ticker", pl.col("total_assets").alias("assets_new"), pl.col("total_liabilities").alias("liab_new")])
    )
    oldest = (
        funds_4q.sort(["ticker", "period_dt"], descending=[False, False])
        .with_columns(pl.int_range(pl.len()).over("ticker").alias("_rn"))
        .filter(pl.col("_rn") == 0).drop("_rn")
        .select(["ticker", pl.col("total_assets").alias("assets_old"), pl.col("total_liabilities").alias("liab_old")])
    )
    return (
        newest.join(oldest, on="ticker", how="left")
        .with_columns([
            pl.when(pl.col("assets_old").is_not_null() & (pl.col("assets_old") > 0))
              .then((pl.col("assets_new") - pl.col("assets_old")) / pl.col("assets_old"))
              .otherwise(None)
              .alias("asset_growth"),

            pl.when(pl.col("liab_old").is_not_null() & (pl.col("liab_old") > 0))
              .then((pl.col("liab_new") - pl.col("liab_old")) / pl.col("liab_old"))
              .otherwise(None)
              .alias("liability_growth"),
        ])
        .select(["ticker", "asset_growth", "liability_growth"])
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")

    prices = pl.read_parquet(CACHE_DIR / "prices.parquet")
    funds  = pl.read_parquet(CACHE_DIR / "fundamentals.parquet")

    pf = price_features(prices)
    ff = fundamental_features(funds)

    print("\n=== PRICE FEATURES (sample) ===")
    print(pf.head(5))
    print("\nColumns:", pf.columns)

    print("\n=== FUNDAMENTAL FEATURES (sample) ===")
    print(ff.head(5))
    print("\nColumns:", ff.columns)

    # Quick sanity check on known tickers
    known = ["PLTR", "AMD", "ASML"]
    print("\n=== SANITY CHECK — known winners ===")
    for t in known:
        prow = pf.filter(pl.col("ticker") == t)
        frow = ff.filter(pl.col("ticker") == t)
        if not prow.is_empty() and not frow.is_empty():
            print(f"\n{t}:")
            print(f"  momentum_6m:        {prow['momentum_6m'][0]:.1f}%")
            print(f"  pct_below_52w_high: {prow['pct_below_52w_high'][0]:.1f}%")
            print(f"  rsi_14:             {prow['rsi_14'][0]:.1f}")
            print(f"  above_200ma:        {prow['above_200ma'][0]}")
            print(f"  gross_margin_latest:{frow['gross_margin_latest'][0]:.2f}")
            print(f"  asset_growth:       {frow['asset_growth'][0]:.3f}")
            print(f"  revenue_trajectory: {frow['revenue_trajectory'][0]}")
            print(f"  revenue_consistency:{frow['revenue_consistency'][0]:.2f}")
        else:
            print(f"{t}: not found in features")