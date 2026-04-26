"""
honest_results.py
=================
Trade-level performance analysis for the systematic equity screener.

This script deliberately ignores the compounded portfolio curve and analyses
raw per-trade statistics instead. The goal is to answer one question honestly:
does the screener have a genuine edge, or is the backtest curve misleading?

Metrics computed
----------------
- Average trade return vs SPY same-period benchmark
- Alpha per trade (outperformance vs benchmark)
- Hit rate — % of trades that beat SPY over the hold period
- Win/loss ratio
- % of calendar months where the strategy beat SPY
- Stress test — 2022 bear market performance

Usage
-----
    # Run backtest first to generate trade parquet files
    python backtest/engine.py --mode <strategy> --hold <months>

    # Then analyse
    python honest_results.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent


def analyse_strategy(trades_path: Path, label: str) -> None:
    """Print honest per-trade statistics for a single strategy/hold combination."""
    if not trades_path.exists():
        print(f"[SKIP] {label} — no trades file found at {trades_path}")
        print("       Run the backtest engine first to generate trade data.\n")
        return

    t     = pl.read_parquet(trades_path)
    rets  = t["net_return_pct"].to_list()
    bench = t["benchmark_return"].to_list()

    alpha_per_trade = [r - b for r, b in zip(rets, bench) if b is not None]

    hit_rate  = t["outperformed"].mean() * 100
    avg_ret   = np.mean(rets)
    avg_bench = np.mean([b for b in bench if b is not None])
    avg_alpha = np.mean(alpha_per_trade)

    wins   = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    avg_win  = np.mean(wins)   if wins   else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    # 2022 bear market stress test
    t22    = t.filter(
        (pl.col("entry_date") >= date(2022, 1, 1)) &
        (pl.col("entry_date") <= date(2022, 12, 31))
    )
    ret22 = t22["net_return_pct"].mean()   if not t22.is_empty() else None
    spy22 = t22["benchmark_return"].mean() if not t22.is_empty() else None

    # % of calendar months beating SPY
    t_monthly = (
        t.with_columns(pl.col("entry_date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg([
            pl.col("net_return_pct").mean().alias("avg_ret"),
            pl.col("benchmark_return").mean().alias("avg_spy"),
        ])
        .with_columns((pl.col("avg_ret") - pl.col("avg_spy")).alias("monthly_alpha"))
    )
    pct_months_alpha = (t_monthly["monthly_alpha"] > 0).mean() * 100

    print(f"{'='*55}")
    print(f"{label}  ({len(t)} trades, {t['ticker'].n_unique()} unique tickers)")
    print(f"{'='*55}")
    print(f"Avg trade return     : {avg_ret:+.2f}%")
    print(f"Avg SPY same period  : {avg_bench:+.2f}%")
    print(f"Avg alpha per trade  : {avg_alpha:+.2f}%")
    print(f"Hit rate vs SPY      : {hit_rate:.1f}%")
    print(f"Avg win              : {avg_win:+.2f}%")
    print(f"Avg loss             : {avg_loss:+.2f}%")
    print(f"Win / loss ratio     : {abs(avg_win / avg_loss):.2f}x" if avg_loss != 0 else "Win / loss ratio     : ∞")
    print(f"% months beating SPY : {pct_months_alpha:.1f}%")
    if ret22 is not None:
        print(f"2022 avg return      : {ret22:+.2f}%  (SPY avg: {spy22:+.2f}%)")
    print()


def main() -> None:
    print("=" * 55)
    print("HONEST TRADE-LEVEL ANALYSIS")
    print("Per-trade statistics — compounded curve excluded")
    print("=" * 55)
    print()

    # ── Discover available trade files ───────────────────────────────────────
    # Trade files are written by backtest/engine.py as:
    #   backtest/trades_<strategy>_<hold>M.parquet
    # They are gitignored — regenerate by running the backtest engine.

    trade_files = sorted((ROOT / "backtest").glob("trades_*.parquet"))

    if not trade_files:
        print("No trade files found in backtest/")
        print("Run the backtest engine first:")
        print("  python backtest/engine.py --mode <strategy> --hold <months>")
        return

    for path in trade_files:
        # Parse label from filename — e.g. trades_A_6M.parquet → "Strategy A | 6M hold"
        stem  = path.stem                      # trades_A_6M
        parts = stem.split("_")               # ['trades', 'A', '6M']
        if len(parts) >= 3:
            strategy = parts[1]
            hold     = parts[2]
            label    = f"Strategy {strategy} | {hold} hold"
        else:
            label = path.stem

        analyse_strategy(path, label)


if __name__ == "__main__":
    main()