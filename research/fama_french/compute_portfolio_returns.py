"""
research/fama_french/compute_portfolio_returns.py
==================================================
Converts A_6M trade-level backtest results into monthly portfolio returns
using mark-to-market pricing for the Fama-French spanning test.

The problem with the previous approach
---------------------------------------
The previous version distributed each trade's total return evenly across
its 6-month hold period (net_return_pct / 6). This produced ~148 contributions
per month which were then averaged. Averaging 148 numbers cancels almost all
idiosyncratic noise, producing artificially smooth monthly returns with an
inflated Sharpe ratio (~3.3) and inflated t-statistic (~12.5).

The correct approach — mark-to-market
--------------------------------------
For each calendar month:
    1. Find all trades active during that month
    2. For each active trade, find the actual close price at:
           - month_start_price: last close on or before the first trading day
           - month_end_price:   last close on or before the last trading day
    3. Compute the actual price return for that month:
           monthly_return = (month_end_price / month_start_price - 1) * 100
    4. Equal-weight average across all active positions

This produces genuine monthly portfolio returns because:
    - Each position moves independently based on real price changes
    - Bad months for some positions are not smoothed away by 140 others
    - The resulting Sharpe and t-statistic reflect actual portfolio volatility

Position count constraint
--------------------------
The backtest holds up to TOP_N=10 positions simultaneously by design.
However, 149 trades are "active" in January 2015 because the backtest
generates signals weekly — many overlapping 6-month windows.

For the spanning test we apply the same TOP_N=10 constraint the live
system uses: on each rebalance date, only the top 10 signals by score
are held. We reconstruct this by tracking which signals were actually
selected, using signal_date and score to determine priority.

Output
------
    research/fama_french/a6m_monthly_returns.parquet

    Columns: date | portfolio_return | n_positions | portfolio_excess

Usage
-----
    python research/fama_french/compute_portfolio_returns.py

    Requires:
        backtest/trades_A_6M.parquet
        data/cache/prices.parquet
        research/fama_french/ff3_factors.parquet
"""

import logging
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent.parent
TRADES_FILE  = ROOT / "backtest" / "trades_A_6M.parquet"
PRICES_FILE  = ROOT / "data" / "cache" / "prices.parquet"
FACTORS_FILE = ROOT / "research" / "fama_french" / "ff3_factors.parquet"
OUTPUT_FILE  = ROOT / "research" / "fama_french" / "a6m_monthly_returns.parquet"

# Maximum positions held simultaneously — matches backtest TOP_N
MAX_POSITIONS = 10


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load trades and prices."""
    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Trade file not found: {TRADES_FILE}")
    if not PRICES_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICES_FILE}")

    trades = pl.read_parquet(TRADES_FILE).with_columns([
        pl.col("entry_date").cast(pl.Date),
        pl.col("exit_date").cast(pl.Date),
        pl.col("signal_date").cast(pl.Date),
    ])

    # Load prices — only close price needed
    prices = pl.read_parquet(PRICES_FILE).select([
        pl.col("ticker"),
        pl.col("date").cast(pl.Date),
        pl.col("close"),
    ]).sort(["ticker", "date"])

    log.info(f"Loaded {len(trades)} trades")
    log.info(f"Loaded {prices['ticker'].n_unique()} tickers, "
             f"{len(prices):,} price rows")

    return trades, prices


def build_price_index(prices: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Build a per-ticker price dictionary for fast lookups.
    Avoids filtering the full price DataFrame inside the main loop.
    """
    log.info("Building per-ticker price index...")
    index = {}
    for ticker in prices["ticker"].unique().to_list():
        index[ticker] = prices.filter(pl.col("ticker") == ticker)
    log.info(f"  Index built for {len(index)} tickers")
    return index


def get_price_on_or_before(
    px: pl.DataFrame,
    target_date: date,
) -> float | None:
    """
    Get the most recent close price on or before target_date.
    Returns None if no price available.
    """
    subset = px.filter(pl.col("date") <= target_date)
    if subset.is_empty():
        return None
    return float(subset["close"][-1])


def reconstruct_active_portfolio(
    trades: pl.DataFrame,
    month_start: date,
    month_end: date,
) -> pl.DataFrame:
    """
    Reconstruct which trades were actually held during a given month.

    The backtest selects TOP_N=10 signals per rebalance date by score.
    Many more signals overlap in any given month due to weekly rebalancing
    with 6-month hold periods.

    We approximate the actual portfolio by:
        1. Taking top 10 signals per signal_date (by score)
        2. Deduplicating by ticker (keep most recent signal)
        3. Capping at MAX_POSITIONS total
    """
    # Trades overlapping with this month
    candidates = trades.filter(
        (pl.col("entry_date") <= month_end) &
        (pl.col("exit_date") >= month_start)
    )

    if candidates.is_empty():
        return pl.DataFrame()

    # Top 10 per signal date, then deduplicate by ticker
    selected = (
        candidates
        .sort(["signal_date", "score"], descending=[False, True])
        .group_by("signal_date")
        .head(MAX_POSITIONS)
        .sort("signal_date", descending=True)
        .unique(subset=["ticker"], keep="first")
        .sort("score", descending=True)
        .head(MAX_POSITIONS)
    )

    return selected


def compute_monthly_return(
    active_trades: pl.DataFrame,
    price_index: dict[str, pl.DataFrame],
    month_start: date,
    month_end: date,
) -> tuple[float | None, int]:
    """
    Compute equal-weighted portfolio return for one month using
    actual mark-to-market prices.

    For each active position:
        - Use entry_price if trade entered this month, else last price <= month_start
        - Use exit_price if trade exited this month, else last price <= month_end
        - Monthly return = (end / start - 1) * 100

    Returns (portfolio_return_pct, n_valid_positions).
    """
    monthly_returns = []

    for row in active_trades.iter_rows(named=True):
        ticker = row["ticker"]
        px     = price_index.get(ticker)

        if px is None or px.is_empty():
            continue

        # Start price — entry price if trade just entered this month
        if row["entry_date"] >= month_start:
            start_price = float(row["entry_price"])
        else:
            start_price = get_price_on_or_before(px, month_start)

        # End price — exit price if trade exited during this month
        if row["exit_date"] <= month_end:
            end_price = float(row["exit_price"])
        else:
            end_price = get_price_on_or_before(px, month_end)

        if start_price is None or end_price is None:
            continue
        if start_price <= 0:
            continue

        monthly_ret = (end_price / start_price - 1) * 100
        monthly_returns.append(monthly_ret)

    if not monthly_returns:
        return None, 0

    return float(np.mean(monthly_returns)), len(monthly_returns)


def build_month_grid(trades: pl.DataFrame) -> list[tuple[date, date]]:
    """Build (month_start, month_end) tuples covering the full backtest period."""
    first_entry = trades["entry_date"].min()
    last_exit   = trades["exit_date"].max()

    month_starts = pl.date_range(
        start=pl.date(first_entry.year, first_entry.month, 1),
        end=pl.date(last_exit.year, last_exit.month, 1),
        interval="1mo",
        eager=True,
    )

    month_ends = month_starts.dt.month_end()

    return list(zip(month_starts.to_list(), month_ends.to_list()))


def compute_all_monthly_returns(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """Compute mark-to-market monthly portfolio returns for the full period."""
    month_grid  = build_month_grid(trades)
    price_index = build_price_index(prices)

    log.info(f"Computing mark-to-market returns for {len(month_grid)} months")

    records = []
    skipped = 0

    for i, (month_start, month_end) in enumerate(month_grid):
        if i % 24 == 0:
            log.info(f"  {month_start} ({i}/{len(month_grid)})")

        # Reconstruct portfolio held this month
        active = reconstruct_active_portfolio(trades, month_start, month_end)

        if active.is_empty():
            records.append({
                "date":             month_start,
                "portfolio_return": 0.0,
                "n_positions":      0,
            })
            continue

        # Mark-to-market return
        portfolio_ret, n_valid = compute_monthly_return(
            active, price_index, month_start, month_end
        )

        if portfolio_ret is None:
            skipped += 1
            records.append({
                "date":             month_start,
                "portfolio_return": 0.0,
                "n_positions":      0,
            })
            continue

        records.append({
            "date":             month_start,
            "portfolio_return": portfolio_ret,
            "n_positions":      n_valid,
        })

    if skipped > 0:
        log.warning(f"Skipped {skipped} months with no valid price data")

    result = pl.DataFrame(records)

    log.info(f"Complete — {len(result)} months")
    log.info(f"  Mean monthly return: {result['portfolio_return'].mean():.3f}%")
    log.info(f"  Std monthly return:  {result['portfolio_return'].std():.3f}%")
    log.info(f"  Avg positions/month: {result['n_positions'].mean():.1f}")

    return result


def add_excess_return(
    monthly: pl.DataFrame,
    factors: pl.DataFrame,
) -> pl.DataFrame:
    """Add excess return: portfolio_return - RF."""
    rf     = factors.select(["date", "rf"])
    merged = monthly.join(rf, on="date", how="inner")
    merged = merged.with_columns(
        (pl.col("portfolio_return") - pl.col("rf")).alias("portfolio_excess")
    )
    dropped = len(monthly) - len(merged)
    if dropped > 0:
        log.warning(f"Dropped {dropped} months not in factor data")
    log.info(f"Mean excess return: {merged['portfolio_excess'].mean():.3f}%")
    return merged.drop("rf")


def print_summary(monthly: pl.DataFrame) -> None:
    """Print summary statistics."""
    ann_return = monthly["portfolio_return"].mean() * 12
    ann_excess = monthly["portfolio_excess"].mean() * 12
    sharpe = (
        monthly["portfolio_excess"].mean() /
        monthly["portfolio_excess"].std()
    ) * np.sqrt(12)

    print()
    print("=" * 58)
    print("A_6M MONTHLY PORTFOLIO RETURNS — MARK TO MARKET")
    print("=" * 58)
    print(f"  Months:              {len(monthly)}")
    print(f"  Date range:          {monthly['date'].min()} → {monthly['date'].max()}")
    print(f"  Avg positions/month: {monthly['n_positions'].mean():.1f}")
    print(f"  Max positions:       {MAX_POSITIONS}")
    print()
    print(f"  Mean monthly return: {monthly['portfolio_return'].mean():.3f}%")
    print(f"  Annualised return:   {ann_return:.1f}%")
    print(f"  Mean excess return:  {monthly['portfolio_excess'].mean():.3f}%")
    print(f"  Annualised excess:   {ann_excess:.1f}%")
    print(f"  Std dev monthly:     {monthly['portfolio_return'].std():.3f}%")
    print(f"  Sharpe (annualised): {sharpe:.3f}")
    print()
    print("  Position count distribution:")
    for n in range(0, MAX_POSITIONS + 1):
        count = (monthly["n_positions"] == n).sum()
        if count > 0:
            pct = count / len(monthly) * 100
            bar = "█" * int(pct / 2)
            print(f"    {n:2d} positions: {count:3d} months ({pct:4.1f}%) {bar}")
    print()


def main() -> pl.DataFrame:
    log.info("Computing A_6M monthly portfolio returns — mark-to-market method")

    trades, prices = load_data()
    monthly        = compute_all_monthly_returns(trades, prices)

    if not FACTORS_FILE.exists():
        raise FileNotFoundError(
            f"Factor file not found: {FACTORS_FILE}\n"
            "Run: python research/fama_french/fetch_factors.py first"
        )
    factors = pl.read_parquet(FACTORS_FILE)
    monthly = add_excess_return(monthly, factors)

    assert monthly["portfolio_return"].null_count() == 0
    assert monthly["portfolio_excess"].null_count() == 0
    log.info("Validation passed")

    monthly.write_parquet(OUTPUT_FILE)
    log.info(f"Saved to {OUTPUT_FILE}")

    print_summary(monthly)
    return monthly


if __name__ == "__main__":
    main()