"""
backtest/engine.py
Core backtesting engine.

Two modes:
    Backtest A — price signals only (C5, C6, C9), Apr 2019 – Apr 2026
    Backtest B — full 9 criteria, Aug 2024 – Apr 2026

Design principles:
    - Every data access goes through get_prices_as_of(date) or get_funds_as_of(date)
    - These functions REJECT any data with timestamp >= simulation date
    - No exceptions. If Sharpe > 1.5 assume lookahead first.
    - All features recomputed fresh at each rebalance date on the filtered subset

Usage:
    python backtest/engine.py --mode A --hold 3
    python backtest/engine.py --mode B --hold 3
    python backtest/engine.py --mode A --hold 1 3 6   # test multiple hold periods
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screener"))

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRANSACTION_COST  = 0.0015   # 0.15% round trip (Trading 212 FX fee)
PORTFOLIO_VALUE   = 10_000.0 # Starting portfolio in £
POSITION_SIZE     = 1_000.0  # Fixed £ per position (strict equal weight, 10%)
TOP_N             = 10       # Signals to take each month
BENCHMARK         = "SPY"
WARMUP_DAYS       = 252      # Trading days before first trade — ensures all signals are valid
                              # 200MA needs 200 days, 6M momentum needs 126 days

# Backtest windows
WINDOW_A_START     = datetime(2019, 4, 22)  # first available price date
WINDOW_A_END       = datetime(2026, 4, 17)  # latest available price date
FIRST_TRADE_DATE   = datetime(2020, 5, 1)   # after 252-day warmup from Apr 2019
WINDOW_B_START     = datetime(2024, 8, 1)   # first fundamental data available
WINDOW_B_END       = datetime(2026, 4, 17)

# Walk-forward splits (for Backtest A — all after warmup)
SPLITS = {
    "train":    (datetime(2020, 5, 1),   datetime(2021, 4, 30)),
    "validate": (datetime(2021, 5, 1),   datetime(2022, 4, 30)),
    "test":     (datetime(2022, 5, 1),   datetime(2024, 4, 30)),
    "live":     (datetime(2024, 5, 1),   datetime(2026, 4, 17)),
}


# ── Point-in-time data access — the lookahead firewall ────────────────────────

def get_prices_as_of(prices: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """
    Return all price rows with date STRICTLY BEFORE as_of.
    This is the lookahead firewall — never use >= or ==.
    Any feature computed from this subset is clean.
    """
    return prices.filter(pl.col("date") < as_of)


def get_funds_as_of(funds: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """
    Return all fundamental rows with period STRICTLY BEFORE as_of.
    Quarterly earnings are only available after the filing date,
    not the period end date. yfinance uses period end date as the key,
    which slightly understates the delay — we accept this limitation
    and document it. In a production system this would use fillingDate.
    """
    as_of_str = as_of.strftime("%Y-%m-%d")
    return funds.filter(pl.col("period") < as_of_str)


def get_benchmark_return(
    prices: pl.DataFrame,
    entry_date: datetime,
    exit_date: datetime,
    ticker: str = BENCHMARK,
) -> float | None:
    """Return the % return of the benchmark between two dates."""
    df = prices.filter(pl.col("ticker") == ticker).sort("date")
    entry_row = df.filter(pl.col("date") >= entry_date).head(1)
    exit_row  = df.filter(pl.col("date") >= exit_date).head(1)
    if entry_row.is_empty() or exit_row.is_empty():
        return None
    p0 = entry_row["close"][0]
    p1 = exit_row["close"][0]
    return (p1 - p0) / p0 * 100


# ── Monthly rebalance dates ────────────────────────────────────────────────────

def get_rebalance_dates(
    prices: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """
    Return the first trading day of each month between start and end.
    Uses SPY as the reference calendar.
    """
    spy = prices.filter(pl.col("ticker") == BENCHMARK).sort("date")
    monthly = (
        spy
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ])
        .group_by(["year", "month"])
        .agg(pl.col("date").min().alias("rebalance_date"))
        .sort("rebalance_date")
        .filter(
            (pl.col("rebalance_date") >= start) &
            (pl.col("rebalance_date") <= end)
        )
    )
    return monthly["rebalance_date"].to_list()


def get_exit_date(
    prices: pl.DataFrame,
    entry_date: datetime,
    hold_months: int,
) -> datetime | None:
    """
    Return the first trading day approximately hold_months after entry_date.
    Uses SPY calendar. Returns None if not enough data.
    """
    target = entry_date + timedelta(days=hold_months * 30)
    spy = prices.filter(pl.col("ticker") == BENCHMARK).sort("date")
    future = spy.filter(pl.col("date") >= target)
    if future.is_empty():
        return None
    return future["date"][0]


# ── Signal scoring ─────────────────────────────────────────────────────────────

def score_price_signals(prices_pit: pl.DataFrame) -> pl.DataFrame:
    """
    Backtest A — score tickers on price signals only.
    Uses only data in prices_pit (already filtered to simulation date).

    Signals:
        C5 — below 52w high AND 3M momentum < -10%
        C6 — above 200-day moving average
        C9 — 6M positive momentum

    Returns DataFrame: ticker | c5 | c6 | c9 | score | momentum_1w
    """
    from features import price_features

    # Re-compute features on point-in-time subset
    pf = price_features(prices_pit)

    # Exclude benchmark tickers from signal universe
    pf = pf.filter(~pl.col("ticker").is_in(["SPY", "QQQ"]))

    return pf.with_columns([
        # C5
        pl.when(
            pl.col("pct_below_52w_high").is_not_null() &
            pl.col("momentum_3m").is_not_null() &
            (pl.col("pct_below_52w_high") < 0) &
            (pl.col("momentum_3m") < -10.0)
        ).then(True).otherwise(False).alias("c5"),

        # C6
        pl.when(pl.col("above_200ma").is_not_null())
          .then(pl.col("above_200ma"))
          .otherwise(False)
          .alias("c6"),

        # C9
        pl.when(
            pl.col("momentum_6m").is_not_null() &
            (pl.col("momentum_6m") > 0)
        ).then(True).otherwise(False).alias("c9"),
    ]).with_columns([
        (
            pl.col("c5").cast(pl.Int32) +
            pl.col("c6").cast(pl.Int32) +
            pl.col("c9").cast(pl.Int32)
        ).alias("score")
    ]).select([
        "ticker", "c5", "c6", "c9", "score",
        "momentum_1w", "momentum_6m", "momentum_3m",
        "close", "pct_below_52w_high",
    ])


def score_full_signals(
    prices_pit: pl.DataFrame,
    funds_pit: pl.DataFrame,
) -> pl.DataFrame:
    """
    Backtest B — score tickers on all 9 criteria.
    Uses only point-in-time data subsets.

    Returns DataFrame with all criteria columns + score.
    """
    from features import price_features, fundamental_features
    from run import score_tickers

    if funds_pit.is_empty():
        return pl.DataFrame()

    pf = price_features(prices_pit)
    ff = fundamental_features(funds_pit)

    pf = pf.filter(~pl.col("ticker").is_in(["SPY", "QQQ"]))

    scored = score_tickers(pf, ff)
    return scored


def select_top_n(
    signals: pl.DataFrame,
    n: int = TOP_N,
) -> pl.DataFrame:
    """
    Select top N tickers by score, with 1W momentum as tiebreaker.
    Returns only tickers with score > 0.
    """
    return (
        signals
        .filter(pl.col("score") > 0)
        .sort(["score", "momentum_1w"], descending=[True, True], nulls_last=True)
        .head(n)
    )


# ── Trade execution ────────────────────────────────────────────────────────────

def get_next_open(
    prices: pl.DataFrame,
    ticker: str,
    after_date: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Get the open price on the NEXT trading day after after_date.
    Signals are computed on after_date close — we execute next day open.
    Returns (open_price, actual_date) or (None, None) if not available.
    """
    df = (
        prices
        .filter(pl.col("ticker") == ticker)
        .filter(pl.col("date") > after_date)   # strictly after — next day
        .sort("date")
    )
    if df.is_empty():
        return None, None
    return df["open"][0], df["date"][0]


def get_entry_price(
    prices: pl.DataFrame,
    ticker: str,
    entry_date: datetime,
) -> tuple[float | None, datetime | None]:
    """Entry at next trading day open after signal date."""
    return get_next_open(prices, ticker, entry_date)


def get_exit_price(
    prices: pl.DataFrame,
    ticker: str,
    exit_date: datetime,
) -> tuple[float | None, datetime | None]:
    """Exit at next trading day open after planned exit date."""
    return get_next_open(prices, ticker, exit_date)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_backtest(
    prices: pl.DataFrame,
    funds: pl.DataFrame,
    mode: str,
    hold_months: int,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """
    Run the full backtest loop.

    Returns a trade log DataFrame with columns:
        entry_date | exit_date | ticker | entry_price | exit_price |
        return_pct | net_return_pct | score | hold_months |
        benchmark_return | mode
    """
    rebalance_dates = get_rebalance_dates(prices, start, end)

    # Apply warmup — skip rebalance dates before FIRST_TRADE_DATE
    first_trade = FIRST_TRADE_DATE if mode == "A" else WINDOW_B_START
    tradeable_dates = [d for d in rebalance_dates if d >= first_trade]

    log.info(f"Mode {mode} | hold={hold_months}M | {len(rebalance_dates)} rebalance dates | "
             f"{start.date()} → {end.date()}")
    log.info(f"  Warmup period: {start.date()} → {first_trade.date()} "
             f"({len(rebalance_dates) - len(tradeable_dates)} months skipped)")
    log.info(f"  Trading from: {first_trade.date()} | {len(tradeable_dates)} tradeable months")

    trades = []
    open_positions: dict[str, dict] = {}  # ticker -> position info

    for rebal_dt in rebalance_dates:
        # ── Point-in-time data ────────────────────────────────────────────
        prices_pit = get_prices_as_of(prices, rebal_dt)
        funds_pit  = get_funds_as_of(funds, rebal_dt)

        if prices_pit.is_empty():
            continue

        # ── Score signals ─────────────────────────────────────────────────
        if mode == "A":
            signals = score_price_signals(prices_pit)
        else:
            signals = score_full_signals(prices_pit, funds_pit)
            if signals.is_empty():
                log.warning(f"  {rebal_dt.date()}: no signals generated (mode B, insufficient funds data)")
                continue

        top = select_top_n(signals, TOP_N)

        # ── Exit positions due for exit ───────────────────────────────────
        for ticker, pos in list(open_positions.items()):
            if rebal_dt >= pos["planned_exit"]:
                exit_px, actual_exit_dt = get_exit_price(prices, ticker, rebal_dt)
                if exit_px is None:
                    del open_positions[ticker]
                    continue

                raw_return = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                net_return = raw_return - TRANSACTION_COST * 100 * 2  # entry + exit

                bench_ret = get_benchmark_return(
                    prices, pos["actual_entry_date"], actual_exit_dt or rebal_dt
                )

                trades.append({
                    "entry_date":       pos["actual_entry_date"],
                    "signal_date":      pos["signal_date"],
                    "exit_date":        actual_exit_dt or rebal_dt,
                    "ticker":           ticker,
                    "entry_price":      pos["entry_price"],
                    "exit_price":       exit_px,
                    "return_pct":       raw_return,
                    "net_return_pct":   net_return,
                    "score":            pos["score"],
                    "hold_months":      hold_months,
                    "benchmark_return": bench_ret,
                    "mode":             mode,
                    "outperformed":     net_return > (bench_ret or 0),
                })
                del open_positions[ticker]

        # ── Enter new positions (only after warmup) ───────────────────────
        if rebal_dt < first_trade:
            continue   # still in warmup — compute signals but don't trade

        for row in top.iter_rows(named=True):
            ticker = row["ticker"]
            if ticker in open_positions:
                continue  # already holding

            # Enter at NEXT DAY OPEN — not today's close
            entry_px, actual_entry_dt = get_entry_price(prices, ticker, rebal_dt)
            if entry_px is None:
                continue

            planned_exit = get_exit_date(prices, rebal_dt, hold_months)
            if planned_exit is None:
                continue

            open_positions[ticker] = {
                "signal_date":       rebal_dt,
                "actual_entry_date": actual_entry_dt or rebal_dt,
                "entry_price":       entry_px,
                "planned_exit":      planned_exit,
                "score":             row["score"],
            }

        n_open = len(open_positions)
        n_signals = len(top)
        log.debug(f"  {rebal_dt.date()} | signals={n_signals} | open={n_open}")

    if not trades:
        log.warning("No trades completed — check date range and signal criteria")
        return pl.DataFrame()

    log.info(f"Backtest complete — {len(trades)} trades closed")
    return pl.DataFrame(trades)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "both"], default="A")
    parser.add_argument("--hold", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--split", choices=["full", "train", "validate", "test", "live"],
                        default="full")
    args = parser.parse_args()

    log.info("Loading cached data...")
    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    funds  = pl.read_parquet(ROOT / "data" / "cache" / "fundamentals.parquet")

    modes = ["A", "B"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        if mode == "A":
            start, end = WINDOW_A_START, WINDOW_A_END
        else:
            start, end = WINDOW_B_START, WINDOW_B_END

        if args.split != "full" and mode == "A":
            start, end = SPLITS[args.split]

        for hold in args.hold:
            key = f"{mode}_{hold}M"
            log.info(f"\n{'='*60}")
            log.info(f"Running Backtest {key}")
            trades = run_backtest(prices, funds, mode, hold, start, end)
            if not trades.is_empty():
                results[key] = trades
                # Save trade log
                out = ROOT / "backtest" / f"trades_{key}.parquet"
                out.parent.mkdir(exist_ok=True)
                trades.write_parquet(out)
                log.info(f"Trade log saved to {out}")

    return results


if __name__ == "__main__":
    main()