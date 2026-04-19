import polars as pl
import numpy as np
from pathlib import Path
from datetime import date

ROOT = Path.cwd()

print("=== HONEST TRADE-LEVEL ANALYSIS ===\n")
print("(Ignoring compounded portfolio curve — simulation bug)")
print("(Real per-trade statistics below)\n")

for mode in ["A", "C"]:
    for hold in [1, 3, 6]:
        t = pl.read_parquet(ROOT / "backtest" / f"trades_{mode}_{hold}M.parquet")
        rets  = t["net_return_pct"].to_list()
        bench = t["benchmark_return"].to_list()
        alpha_per_trade = [r - b for r, b in zip(rets, bench) if b is not None]

        hit_rate  = t["outperformed"].mean() * 100
        avg_ret   = np.mean(rets)
        avg_bench = np.mean([b for b in bench if b is not None])
        avg_alpha = np.mean(alpha_per_trade)
        wins      = [r for r in rets if r > 0]
        losses    = [r for r in rets if r <= 0]
        avg_win   = np.mean(wins)   if wins   else 0
        avg_loss  = np.mean(losses) if losses else 0

        t22 = t.filter(
            (pl.col("entry_date") >= date(2022, 1, 1)) &
            (pl.col("entry_date") <= date(2022, 12, 31))
        )
        ret22 = t22["net_return_pct"].mean()   if not t22.is_empty() else None
        spy22 = t22["benchmark_return"].mean() if not t22.is_empty() else None

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
        print(f"{mode}_{hold}M  ({len(t)} trades, {t['ticker'].n_unique()} tickers)")
        print(f"{'='*55}")
        print(f"Avg trade return:     {avg_ret:+.2f}%")
        print(f"Avg SPY same period:  {avg_bench:+.2f}%")
        print(f"Avg alpha per trade:  {avg_alpha:+.2f}%")
        print(f"Hit rate vs SPY:      {hit_rate:.1f}%")
        print(f"Avg win:              {avg_win:+.2f}%")
        print(f"Avg loss:             {avg_loss:+.2f}%")
        print(f"Win/loss ratio:       {abs(avg_win/avg_loss):.2f}x")
        print(f"% months beating SPY: {pct_months_alpha:.1f}%")
        if ret22 is not None:
            print(f"2022 avg return:      {ret22:+.2f}%  (SPY: {spy22:+.2f}%)")
        print()
