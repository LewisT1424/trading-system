"""
backtest/engine.py
Core backtesting engine.

Three modes:
    Backtest A — price signals only (C5, C6, C9), May 2020 – Apr 2026
    Backtest B — full 9 criteria, yfinance fundamentals (legacy, short window)
    Backtest C — full 9 criteria, EDGAR fundamentals + constituent filter (primary)

Design principles:
    - Every data access goes through get_prices_as_of(date) or get_funds_as_of(date)
    - These functions REJECT any data with timestamp >= simulation date
    - No exceptions. If Sharpe > 1.5 assume lookahead first.
    - All features recomputed fresh at each rebalance date on the filtered subset

Usage:
    python backtest/engine.py --mode A --hold 3
    python backtest/engine.py --mode C --hold 3
    python backtest/engine.py --mode A --hold 1 3 6   # test multiple hold periods
    python backtest/engine.py --mode C --hold 1 3 6   # full 9-criteria, all hold periods
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
WINDOW_A_START       = datetime(2010, 1, 4)   # earliest price data available
WINDOW_A_END         = datetime(2026, 4, 17)  # latest available price date
FIRST_TRADE_DATE     = datetime(2011, 1, 3)   # after 252-day warmup from Jan 2010
WINDOW_B_START       = datetime(2024, 8, 1)   # first yfinance fundamental data
WINDOW_B_END         = datetime(2026, 4, 17)
WINDOW_C_START       = datetime(2010, 1, 4)   # EDGAR from 2009, price from 2010
WINDOW_C_END         = datetime(2026, 4, 17)
WINDOW_C_FIRST_TRADE = datetime(2011, 1, 3)   # same warmup as A

# Rebalance frequency — switch between monthly and weekly
REBALANCE_FREQ = "weekly"   # "monthly" or "weekly"

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
    Return all fundamental rows available STRICTLY BEFORE as_of.
    For EDGAR data: filters on `filed` date — the actual SEC filing date.
    For yfinance data: filters on `period` date (period end, slight lookahead).
    True point-in-time: a Q3 report filed Nov 14 is not available until Nov 14.
    """
    as_of_str = as_of.strftime("%Y-%m-%d")
    # EDGAR data has `filed` column — use it for true point-in-time
    if "filed" in funds.columns:
        return funds.filter(
            pl.col("filed").is_not_null() &
            (pl.col("filed") != "") &
            (pl.col("filed") < as_of_str)
        )
    # yfinance fallback — uses period end date
    return funds.filter(pl.col("period") < as_of_str)


def get_constituents_on_date(
    as_of: datetime,
    compositions: pl.DataFrame,
) -> set[str]:
    """
    Return the set of S&P 500 tickers that were members on as_of date.
    Uses the most recent monthly composition on or before as_of.
    """
    as_of_str = as_of.strftime("%Y-%m-%d")
    valid = compositions.filter(pl.col("date") <= as_of_str)
    if valid.is_empty():
        valid = compositions.head(1)
    row = valid.tail(1)
    return set(row["tickers"][0])


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
    freq: str = REBALANCE_FREQ,
) -> list[datetime]:
    """
    Return rebalance dates between start and end.
    freq: "monthly" = first trading day of each month
          "weekly"  = first trading day of each ISO week
    Uses SPY as the reference calendar.
    """
    spy = prices.filter(pl.col("ticker") == BENCHMARK).sort("date")

    if freq == "weekly":
        weekly = (
            spy
            .with_columns([
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.iso_year().alias("iso_year"),
                pl.col("date").dt.week().alias("week"),
            ])
            .group_by(["iso_year", "week"])
            .agg(pl.col("date").min().alias("rebalance_date"))
            .sort("rebalance_date")
            .filter(
                (pl.col("rebalance_date") >= start) &
                (pl.col("rebalance_date") <= end)
            )
        )
        return weekly["rebalance_date"].to_list()
    else:
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


def score_full_signals_edgar(
    prices_pit: pl.DataFrame,
    edgar_pit: pl.DataFrame,
    as_of: datetime,
    constituent_tickers: set[str],
) -> pl.DataFrame:
    """
    Backtest C — score tickers on all 9 criteria using EDGAR fundamentals.

    Key differences from score_full_signals():
        1. Uses fundamental_features_edgar() — EDGAR data with filed date filter
        2. Filters universe to S&P 500 constituents on as_of date
        3. Intersects with available price data

    Returns DataFrame with all criteria columns + score.
    """
    from features import price_features, fundamental_features_edgar
    from run import score_tickers

    if edgar_pit.is_empty():
        return pl.DataFrame()

    # Price features — full universe
    pf = price_features(prices_pit)
    pf = pf.filter(~pl.col("ticker").is_in(["SPY", "QQQ"]))

    # Constituent filter — only score tickers in the index on this date
    if constituent_tickers:
        pf = pf.filter(pl.col("ticker").is_in(constituent_tickers))

    if pf.is_empty():
        return pl.DataFrame()

    # EDGAR fundamental features — point-in-time
    ff = fundamental_features_edgar(edgar_pit, as_of)

    if ff.is_empty():
        return pl.DataFrame()

    scored = score_tickers(pf, ff)
    return scored


def select_top_n(
    signals: pl.DataFrame,
    n: int = TOP_N,
    min_score: int = 1,
) -> pl.DataFrame:
    """
    Select top N tickers by score, with 1W momentum as tiebreaker.
    min_score: minimum score threshold (use 7 for mode C, 1 for mode A).
    """
    return (
        signals
        .filter(pl.col("score") >= min_score)
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
    compositions: pl.DataFrame | None = None,
    freq: str = REBALANCE_FREQ,
) -> pl.DataFrame:
    """
    Run the full backtest loop.

    Modes:
        A — price signals only (C5, C6, C9)
        B — full 9 criteria, yfinance fundamentals (legacy)
        C — full 9 criteria, EDGAR fundamentals + S&P 500 constituent filter

    Returns a trade log DataFrame with columns:
        entry_date | exit_date | ticker | entry_price | exit_price |
        return_pct | net_return_pct | score | hold_months |
        benchmark_return | mode
    """
    rebalance_dates = get_rebalance_dates(prices, start, end, freq=freq)

    # Warmup
    if mode == "C":
        first_trade = WINDOW_C_FIRST_TRADE
    elif mode == "A":
        first_trade = FIRST_TRADE_DATE
    else:
        first_trade = WINDOW_B_START

    tradeable_dates = [d for d in rebalance_dates if d >= first_trade]

    log.info(f"Mode {mode} | hold={hold_months}M | {len(rebalance_dates)} rebalance dates | "
             f"{start.date()} → {end.date()}")
    log.info(f"  Warmup period: {start.date()} → {first_trade.date()} "
             f"({len(rebalance_dates) - len(tradeable_dates)} months skipped)")
    log.info(f"  Trading from: {first_trade.date()} | {len(tradeable_dates)} tradeable months")

    trades = []
    open_positions: dict[str, dict] = {}

    for rebal_dt in rebalance_dates:
        # ── Point-in-time data ────────────────────────────────────────────
        prices_pit = get_prices_as_of(prices, rebal_dt)
        funds_pit  = get_funds_as_of(funds, rebal_dt)

        if prices_pit.is_empty():
            continue

        # ── Score signals ─────────────────────────────────────────────────
        if mode == "A":
            signals = score_price_signals(prices_pit)
        elif mode == "C":
            # EDGAR fundamentals + constituent filter
            constituent_tickers: set[str] = set()
            if compositions is not None:
                constituent_tickers = get_constituents_on_date(rebal_dt, compositions)
            signals = score_full_signals_edgar(
                prices_pit, funds_pit, rebal_dt, constituent_tickers
            )
            if signals.is_empty():
                log.warning(f"  {rebal_dt.date()}: no signals (mode C)")
                continue
        else:
            signals = score_full_signals(prices_pit, funds_pit)
            if signals.is_empty():
                log.warning(f"  {rebal_dt.date()}: no signals generated (mode B, insufficient funds data)")
                continue

        min_score = 7 if mode == "C" else 1
        top = select_top_n(signals, TOP_N, min_score=min_score)

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
    parser.add_argument("--mode", choices=["A", "B", "C", "both"], default="A")
    parser.add_argument("--hold", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--split", choices=["full", "train", "validate", "test", "live"],
                        default="full")
    parser.add_argument("--freq", choices=["weekly", "monthly"], default=REBALANCE_FREQ,
                        help="Rebalance frequency (default: weekly)")
    args = parser.parse_args()

    log.info("Loading cached data...")
    prices       = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    compositions = pl.read_parquet(ROOT / "data" / "cache" / "constituents.parquet")

    # Load fundamentals — EDGAR for mode C, yfinance for mode B
    edgar_path = ROOT / "data" / "cache" / "fundamentals_edgar.parquet"
    yf_path    = ROOT / "data" / "cache" / "fundamentals.parquet"
    edgar = pl.read_parquet(edgar_path) if edgar_path.exists() else pl.DataFrame()
    funds = pl.read_parquet(yf_path)    if yf_path.exists()    else pl.DataFrame()

    log.info(f"Prices:       {prices['ticker'].n_unique()} tickers, {len(prices):,} rows")
    log.info(f"EDGAR funds:  {edgar['ticker'].n_unique() if not edgar.is_empty() else 0} tickers")
    log.info(f"Constituents: {len(compositions)} monthly compositions")

    modes = ["A", "B", "C"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        if mode == "A":
            start, end = WINDOW_A_START, WINDOW_A_END
            fund_data  = funds
        elif mode == "C":
            start, end = WINDOW_C_START, WINDOW_C_END
            fund_data  = edgar
        else:
            start, end = WINDOW_B_START, WINDOW_B_END
            fund_data  = funds

        if args.split != "full" and mode in ("A", "C"):
            start, end = SPLITS[args.split]

        for hold in args.hold:
            key = f"{mode}_{hold}M"
            log.info(f"\n{'='*60}")
            log.info(f"Running Backtest {key}")
            trades = run_backtest(
                prices, fund_data, mode, hold, start, end,
                compositions=compositions if mode == "C" else None,
                freq=args.freq,
            )
            if not trades.is_empty():
                results[key] = trades
                out = ROOT / "backtest" / f"trades_{key}.parquet"
                out.parent.mkdir(exist_ok=True)
                trades.write_parquet(out)
                log.info(f"Trade log saved to {out}")

    return results


if __name__ == "__main__":
    main()