"""
backtest/portfolio.py
Daily portfolio simulation from a trade log.

Takes the trade log output from engine.py and builds a proper daily
portfolio value series. Each position is tracked daily using actual
price data. Portfolio value = sum of open positions + cash.

Design:
    - Fixed £1,000 per position (strict equal weight)
    - Cash earns nothing (conservative)
    - Max 10 positions open simultaneously
    - Transaction costs already deducted in trade log

Usage:
    from backtest.portfolio import build_portfolio
    daily = build_portfolio(trades, prices)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

POSITION_SIZE = 1_000.0   # £ per position — strict equal weight
STARTING_CASH = 10_000.0  # starting portfolio value
BENCHMARK     = "SPY"
RISK_FREE_RATE = 0.04


def build_portfolio(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build a daily portfolio value series from a trade log.

    For each calendar day in the backtest:
        1. Identify all open positions (entered but not yet exited)
        2. Look up each position's current close price
        3. Compute current market value of each position
        4. Sum positions + remaining cash = portfolio value

    Returns DataFrame: date | portfolio_value | n_positions | cash | daily_return_pct
    """
    if trades.is_empty():
        log.warning("Empty trade log — cannot build portfolio")
        return pl.DataFrame()

    # Get all trading days from SPY
    spy_dates = (
        prices
        .filter(pl.col("ticker") == BENCHMARK)
        .sort("date")
        .select("date")
    )

    # Backtest date range — from first entry to last exit
    bt_start = trades["entry_date"].min()
    bt_end   = trades["exit_date"].max()

    trading_days = (
        spy_dates
        .filter((pl.col("date") >= bt_start) & (pl.col("date") <= bt_end))
        ["date"].to_list()
    )

    log.info(f"Building daily portfolio: {len(trading_days)} trading days, "
             f"{len(trades)} trades")

    # Pre-index prices for fast lookup: {ticker: {date: close}}
    price_index: dict[str, dict] = {}
    for row in prices.iter_rows(named=True):
        t = row["ticker"]
        if t not in price_index:
            price_index[t] = {}
        price_index[t][row["date"]] = row["close"]

    # Build trade lookup structures
    # open_on[date] = list of trades entering on that date
    # close_on[date] = list of trades exiting on that date
    open_on: dict  = {}
    close_on: dict = {}

    for row in trades.iter_rows(named=True):
        ed = row["entry_date"]
        xd = row["exit_date"]
        if ed not in open_on:  open_on[ed]  = []
        if xd not in close_on: close_on[xd] = []
        open_on[ed].append(row)
        close_on[xd].append(row)

    # Simulate day by day
    cash = STARTING_CASH
    open_positions: dict[str, dict] = {}  # ticker -> {shares, cost_basis}
    daily_records = []

    for day in trading_days:
        # Exit positions closing today
        for trade in close_on.get(day, []):
            ticker = trade["ticker"]
            if ticker in open_positions:
                pos = open_positions.pop(ticker)
                # Return cash: exit value = shares * exit_price
                exit_val = pos["shares"] * trade["exit_price"]
                cash += exit_val

        # Enter positions opening today
        for trade in open_on.get(day, []):
            ticker = trade["ticker"]
            if ticker in open_positions:
                continue  # already holding
            if cash < POSITION_SIZE:
                log.debug(f"  {day.date()}: insufficient cash for {ticker}")
                continue
            # Deduct transaction cost from position size
            cost = POSITION_SIZE * (1 - 0.0015)  # entry cost
            shares = cost / trade["entry_price"]
            open_positions[ticker] = {
                "shares":     shares,
                "cost_basis": POSITION_SIZE,
                "entry_date": day,
            }
            cash -= POSITION_SIZE

        # Mark to market — value all open positions at today's close
        position_value = 0.0
        for ticker, pos in open_positions.items():
            # Find today's close for this ticker
            ticker_prices = price_index.get(ticker, {})
            today_close = ticker_prices.get(day)
            if today_close is None:
                # Use most recent available price
                past_prices = {d: p for d, p in ticker_prices.items() if d <= day}
                today_close = past_prices[max(past_prices)] if past_prices else pos["cost_basis"] / pos["shares"]
            position_value += pos["shares"] * today_close

        portfolio_value = cash + position_value

        daily_records.append({
            "date":            day,
            "portfolio_value": portfolio_value,
            "n_positions":     len(open_positions),
            "position_value":  position_value,
            "cash":            cash,
        })

    if not daily_records:
        return pl.DataFrame()

    daily = pl.DataFrame(daily_records)

    # Daily return
    daily = daily.with_columns([
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1)
        .alias("daily_return")
    ])

    log.info(f"Portfolio built: {len(daily)} days | "
             f"start={daily['portfolio_value'][0]:.0f} | "
             f"end={daily['portfolio_value'][-1]:.0f}")

    return daily


def compute_portfolio_metrics(
    daily: pl.DataFrame,
    prices: pl.DataFrame,
    label: str = "",
) -> dict:
    """
    Compute performance metrics from a daily portfolio series.
    This is the correct way to compute Sharpe and max drawdown —
    from daily portfolio returns, not from individual trade returns.
    """
    if daily.is_empty():
        return {"error": "Empty portfolio"}

    portfolio_values = daily["portfolio_value"].to_numpy()
    daily_returns    = daily["daily_return"].drop_nulls().to_numpy()
    dates            = daily["date"].to_list()

    start = dates[0]
    end   = dates[-1]
    n_days = len(daily_returns)
    n_years = n_days / 252

    # ── Annualised return ─────────────────────────────────────────────────
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    ann_return   = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # ── Sharpe — computed on daily returns ───────────────────────────────
    rf_daily = RISK_FREE_RATE / 252
    excess   = daily_returns - rf_daily
    sharpe   = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # ── Max drawdown — from daily portfolio curve ─────────────────────────
    peak     = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    max_dd   = float(drawdown.min())

    # ── Calmar ───────────────────────────────────────────────────────────
    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0

    # ── Benchmark ────────────────────────────────────────────────────────
    spy = (
        prices
        .filter(pl.col("ticker") == BENCHMARK)
        .filter((pl.col("date") >= start) & (pl.col("date") <= end))
        .sort("date")
    )
    if not spy.is_empty():
        spy_vals   = spy["close"].to_numpy()
        spy_total  = (spy_vals[-1] / spy_vals[0] - 1) * 100
        spy_ann    = ((1 + spy_total / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        spy_peak   = np.maximum.accumulate(spy_vals)
        spy_dd     = (spy_vals - spy_peak) / spy_peak * 100
        spy_max_dd = float(spy_dd.min())

        spy_daily_ret = np.diff(spy_vals) / spy_vals[:-1]
        spy_excess    = spy_daily_ret - rf_daily
        spy_sharpe    = float(spy_excess.mean() / spy_excess.std() * np.sqrt(252)) \
                        if spy_excess.std() > 0 else 0
    else:
        spy_ann = spy_max_dd = spy_sharpe = 0.0

    # ── Gates ─────────────────────────────────────────────────────────────
    gate_sharpe   = sharpe > 0.5
    gate_beats    = ann_return > spy_ann
    gate_drawdown = abs(max_dd) <= abs(spy_max_dd) * 1.5

    return {
        "label":           label,
        "start":           start,
        "end":             end,
        "n_days":          n_days,
        "starting_value":  float(portfolio_values[0]),
        "ending_value":    float(portfolio_values[-1]),
        "total_return":    total_return,
        "ann_return":      ann_return,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "calmar":          calmar,
        "spy_ann_return":  spy_ann,
        "spy_max_drawdown": spy_max_dd,
        "spy_sharpe":      spy_sharpe,
        "alpha":           ann_return - spy_ann,
        "gate_sharpe":     gate_sharpe,
        "gate_beats_spy":  gate_beats,
        "gate_drawdown":   gate_drawdown,
        "all_gates_pass":  gate_sharpe and gate_beats and gate_drawdown,
        # Survivorship bias disclaimer
        "survivorship_note": (
            "Returns likely overstated by ~3-7% due to survivorship bias "
            "(universe = current S&P500/Nasdaq100 constituents only)."
        ),
    }


def print_portfolio_metrics(m: dict) -> None:
    """Print portfolio-level metrics."""
    if "error" in m:
        print(f"  ERROR: {m['error']}")
        return

    label = f" [{m['label']}]" if m.get("label") else ""
    start = m["start"].strftime("%b %Y") if hasattr(m["start"], "strftime") else str(m["start"])
    end   = m["end"].strftime("%b %Y")   if hasattr(m["end"],   "strftime") else str(m["end"])

    print(f"\n{'='*65}")
    print(f"  PORTFOLIO RESULTS{label} | {start} → {end}")
    print(f"  (Daily simulation — strict equal weight £{POSITION_SIZE:.0f}/position)")
    print(f"{'='*65}")

    print(f"\n  Portfolio")
    print(f"  {'Starting value':<32} £{m['starting_value']:>8,.0f}")
    print(f"  {'Ending value':<32} £{m['ending_value']:>8,.0f}")
    print(f"  {'Total return':<32} {m['total_return']:>+8.2f}%")

    print(f"\n  Performance")
    print(f"  {'Annualised return':<32} {m['ann_return']:>+8.2f}%")
    print(f"  {'SPY annualised return':<32} {m['spy_ann_return']:>+8.2f}%")
    print(f"  {'Alpha':<32} {m['alpha']:>+8.2f}%")

    print(f"\n  Risk (daily portfolio curve)")
    print(f"  {'Sharpe ratio (annualised)':<32} {m['sharpe']:>8.3f}")
    print(f"  {'SPY Sharpe':<32} {m['spy_sharpe']:>8.3f}")
    print(f"  {'Max drawdown':<32} {m['max_drawdown']:>8.2f}%")
    print(f"  {'SPY max drawdown':<32} {m['spy_max_drawdown']:>8.2f}%")
    print(f"  {'Calmar ratio':<32} {m['calmar']:>8.3f}")

    print(f"\n  Phase 3 gates")
    def g(p): return "✓ PASS" if p else "✗ FAIL"
    print(f"  {g(m['gate_sharpe'])}  Sharpe > 0.5         ({m['sharpe']:.3f})")
    print(f"  {g(m['gate_beats_spy'])}  Beats SPY annualised "
          f"({m['ann_return']:+.2f}% vs {m['spy_ann_return']:+.2f}%)")
    print(f"  {g(m['gate_drawdown'])}  Drawdown ≤ 1.5× SPY  "
          f"({m['max_drawdown']:.2f}% vs {m['spy_max_drawdown'] * 1.5:.2f}% limit)")

    result = "✓ ALL GATES PASSED" if m["all_gates_pass"] else "✗ GATES FAILED"
    print(f"\n  {result}")
    print(f"\n  ⚠ {m['survivorship_note']}")
    print(f"{'='*65}\n")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ROOT   = Path(__file__).parent.parent
    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")

    for trade_file in sorted((ROOT / "backtest").glob("trades_*.parquet")):
        label  = trade_file.stem.replace("trades_", "")
        trades = pl.read_parquet(trade_file)

        log.info(f"\nBuilding portfolio for {label}...")
        daily = build_portfolio(trades, prices)

        if daily.is_empty():
            continue

        # Save daily portfolio
        out = ROOT / "backtest" / f"portfolio_{label}.parquet"
        daily.write_parquet(out)
        log.info(f"Saved to {out}")

        m = compute_portfolio_metrics(daily, prices, label=label)
        print_portfolio_metrics(m)