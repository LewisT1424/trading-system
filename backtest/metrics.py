"""
backtest/metrics.py
Computes all performance metrics from a trade log.

Metrics:
    - Annualised return
    - Sharpe ratio (risk-free = 4%)
    - Max drawdown
    - Hit rate
    - Average win / average loss
    - Win/loss ratio
    - Outperformance vs benchmark
    - Calmar ratio (annualised return / max drawdown)

Usage:
    from backtest.metrics import compute_metrics, print_metrics
    metrics = compute_metrics(trades, prices, start, end)
    print_metrics(metrics)
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
import numpy as np

log = logging.getLogger(__name__)

RISK_FREE_RATE  = 0.04   # 4% annual
TRANSACTION_COST = 0.0015 # 0.15% round trip
BENCHMARK        = "SPY"


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
    start: datetime,
    end: datetime,
    hold_months: int = 3,
) -> dict:
    """
    Compute full performance metrics from a completed trade log.

    Args:
        trades:      Trade log from engine.run_backtest()
        prices:      Full price DataFrame (for benchmark and equity curve)
        start:       Backtest start date
        end:         Backtest end date
        hold_months: Hold period (for annualisation)

    Returns:
        Dict of all metrics
    """
    if trades.is_empty():
        return {"error": "No trades to analyse"}

    returns = trades["net_return_pct"].to_numpy()
    bench   = trades["benchmark_return"].drop_nulls().to_numpy()

    # ── Basic stats ───────────────────────────────────────────────────────
    n_trades   = len(trades)
    n_winners  = int((returns > 0).sum())
    n_losers   = int((returns <= 0).sum())
    hit_rate   = n_winners / n_trades if n_trades > 0 else 0

    wins  = returns[returns > 0]
    losses = returns[returns <= 0]

    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # ── Annualised return ─────────────────────────────────────────────────
    # Each trade represents a hold_months position
    # Annualise by scaling to 12 months
    trades_per_year  = 12 / hold_months
    mean_trade_return = float(returns.mean())
    # Compound annualised return
    ann_return = ((1 + mean_trade_return / 100) ** trades_per_year - 1) * 100

    # ── Sharpe ratio ──────────────────────────────────────────────────────
    # Risk-free rate scaled to hold period
    rf_per_period  = ((1 + RISK_FREE_RATE) ** (hold_months / 12) - 1) * 100
    excess_returns = returns - rf_per_period
    if len(excess_returns) > 1 and excess_returns.std() > 0:
        sharpe_per_period = excess_returns.mean() / excess_returns.std()
        sharpe_annualised = sharpe_per_period * np.sqrt(trades_per_year)
    else:
        sharpe_annualised = 0.0

    # ── Max drawdown ──────────────────────────────────────────────────────
    # Build equity curve: start at 100, compound each trade return
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_drawdown = float(drawdown.min())

    # ── Benchmark metrics ─────────────────────────────────────────────────
    spy_period_return = _get_spy_period_return(prices, start, end)
    spy_annual_return = ((1 + spy_period_return / 100) ** (12 / _months_between(start, end)) - 1) * 100 \
                        if _months_between(start, end) > 0 else 0.0

    avg_outperformance = float((returns - bench[:len(returns)]).mean()) \
                         if len(bench) >= len(returns) else None
    outperform_rate = float(trades["outperformed"].mean()) * 100

    # ── Calmar ratio ──────────────────────────────────────────────────────
    calmar = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0.0

    # ── Benchmark drawdown for comparison ────────────────────────────────
    spy_max_dd = _get_spy_max_drawdown(prices, start, end)

    # ── Gate checks ───────────────────────────────────────────────────────
    gate_sharpe  = sharpe_annualised > 0.5
    gate_beats_spy = ann_return > spy_annual_return
    gate_drawdown  = abs(max_drawdown) <= abs(spy_max_dd) * 1.5

    return {
        # Trade stats
        "n_trades":          n_trades,
        "n_winners":         n_winners,
        "n_losers":          n_losers,
        "hit_rate":          hit_rate,
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "win_loss_ratio":    win_loss_ratio,
        # Returns
        "mean_trade_return": mean_trade_return,
        "ann_return":        ann_return,
        "sharpe":            sharpe_annualised,
        "max_drawdown":      max_drawdown,
        "calmar":            calmar,
        # Benchmark
        "spy_annual_return": spy_annual_return,
        "spy_max_drawdown":  spy_max_dd,
        "outperform_rate":   outperform_rate,
        "avg_outperformance": avg_outperformance,
        # Gate checks
        "gate_sharpe":       gate_sharpe,
        "gate_beats_spy":    gate_beats_spy,
        "gate_drawdown":     gate_drawdown,
        "all_gates_pass":    gate_sharpe and gate_beats_spy and gate_drawdown,
        # Meta
        "hold_months":       hold_months,
        "start":             start,
        "end":               end,
    }


def compute_period_metrics(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
    hold_months: int,
    splits: dict,
) -> dict[str, dict]:
    """
    Compute metrics for each walk-forward split separately.
    Returns dict keyed by split name.
    """
    results = {}
    for split_name, (split_start, split_end) in splits.items():
        split_trades = trades.filter(
            (pl.col("entry_date") >= split_start) &
            (pl.col("entry_date") <= split_end)
        )
        if split_trades.is_empty():
            log.warning(f"  No trades in {split_name} period")
            continue
        results[split_name] = compute_metrics(
            split_trades, prices, split_start, split_end, hold_months
        )
    return results


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_spy_period_return(
    prices: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> float:
    """SPY total return over the backtest period."""
    spy = prices.filter(pl.col("ticker") == BENCHMARK).sort("date")
    p0_row = spy.filter(pl.col("date") >= start).head(1)
    p1_row = spy.filter(pl.col("date") <= end).tail(1)
    if p0_row.is_empty() or p1_row.is_empty():
        return 0.0
    p0 = p0_row["close"][0]
    p1 = p1_row["close"][0]
    return (p1 - p0) / p0 * 100


def _get_spy_max_drawdown(
    prices: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> float:
    """SPY max drawdown over the backtest period."""
    spy = (
        prices
        .filter(pl.col("ticker") == BENCHMARK)
        .filter((pl.col("date") >= start) & (pl.col("date") <= end))
        .sort("date")
    )
    if spy.is_empty():
        return 0.0
    closes = spy["close"].to_numpy()
    peak = np.maximum.accumulate(closes)
    dd = (closes - peak) / peak * 100
    return float(dd.min())


def _months_between(start: datetime, end: datetime) -> float:
    delta = end - start
    return delta.days / 30.44


# ── Output formatting ─────────────────────────────────────────────────────────

def print_metrics(
    metrics: dict,
    label: str = "",
    verbose: bool = True,
) -> None:
    """Pretty-print metrics to terminal."""
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    tag = f" [{label}]" if label else ""
    hold = metrics["hold_months"]
    start = metrics["start"].strftime("%b %Y")
    end   = metrics["end"].strftime("%b %Y")

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS{tag} | {hold}M hold | {start} → {end}")
    print(f"{'='*65}")

    print(f"\n  Performance")
    print(f"  {'Annualised return':<30} {metrics['ann_return']:>+8.2f}%")
    print(f"  {'SPY annualised return':<30} {metrics['spy_annual_return']:>+8.2f}%")
    print(f"  {'Outperformance':<30} {metrics['ann_return'] - metrics['spy_annual_return']:>+8.2f}%")
    print(f"  {'Mean trade return':<30} {metrics['mean_trade_return']:>+8.2f}%")

    print(f"\n  Risk")
    print(f"  {'Sharpe ratio (annualised)':<30} {metrics['sharpe']:>8.3f}")
    print(f"  {'Max drawdown':<30} {metrics['max_drawdown']:>8.2f}%")
    print(f"  {'SPY max drawdown':<30} {metrics['spy_max_drawdown']:>8.2f}%")
    print(f"  {'Calmar ratio':<30} {metrics['calmar']:>8.3f}")

    print(f"\n  Trade statistics")
    print(f"  {'Total trades':<30} {metrics['n_trades']:>8}")
    print(f"  {'Hit rate':<30} {metrics['hit_rate']:>8.1%}")
    print(f"  {'Average win':<30} {metrics['avg_win']:>+8.2f}%")
    print(f"  {'Average loss':<30} {metrics['avg_loss']:>+8.2f}%")
    print(f"  {'Win/loss ratio':<30} {metrics['win_loss_ratio']:>8.2f}x")
    print(f"  {'Outperformed SPY':<30} {metrics['outperform_rate']:>8.1f}%")

    print(f"\n  Phase 3 gates")
    def gate(passed): return "✓ PASS" if passed else "✗ FAIL"
    print(f"  {gate(metrics['gate_sharpe'])}  Sharpe > 0.5         ({metrics['sharpe']:.3f})")
    print(f"  {gate(metrics['gate_beats_spy'])}  Beats SPY            "
          f"({metrics['ann_return']:+.2f}% vs {metrics['spy_annual_return']:+.2f}%)")
    print(f"  {gate(metrics['gate_drawdown'])}  Drawdown ≤ 1.5× SPY  "
          f"({metrics['max_drawdown']:.2f}% vs {metrics['spy_max_drawdown'] * 1.5:.2f}% limit)")

    overall = "✓ ALL GATES PASSED" if metrics["all_gates_pass"] else "✗ GATES FAILED"
    print(f"\n  {overall}")
    print(f"{'='*65}\n")


def print_split_metrics(split_results: dict[str, dict]) -> None:
    """Print metrics for each walk-forward split."""
    print(f"\n{'='*65}")
    print("  WALK-FORWARD SPLIT SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Split':<12} {'Ann Ret':>8} {'SPY':>8} {'Alpha':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'HitRate':>8}")
    print(f"  {'-'*64}")

    for name, m in split_results.items():
        if "error" in m:
            continue
        alpha = m["ann_return"] - m["spy_annual_return"]
        print(
            f"  {name:<12} {m['ann_return']:>+8.2f}% {m['spy_annual_return']:>+8.2f}%"
            f" {alpha:>+8.2f}% {m['sharpe']:>8.3f} {m['max_drawdown']:>8.2f}%"
            f" {m['hit_rate']:>8.1%}"
        )
    print()


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    ROOT = Path(__file__).parent.parent

    from backtest.engine import SPLITS, WINDOW_A_START, WINDOW_A_END

    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")

    # Load all available trade logs
    for trade_file in sorted((ROOT / "backtest").glob("trades_*.parquet")):
        trades = pl.read_parquet(trade_file)
        key    = trade_file.stem.replace("trades_", "")
        hold   = int(key.split("_")[1].replace("M", ""))

        m = compute_metrics(trades, prices, WINDOW_A_START, WINDOW_A_END, hold)
        print_metrics(m, label=key)

        # Walk-forward splits for mode A
        if key.startswith("A"):
            split_m = compute_period_metrics(trades, prices, hold, SPLITS)
            print_split_metrics(split_m)