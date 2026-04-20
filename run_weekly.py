#!/usr/bin/env python3
"""
run_weekly.py — Sunday pipeline for A_6M + Mode C paper trading
================================================================
Run every Sunday before market open (or Friday close).

Usage
-----
    python run_weekly.py                          # new week, no open positions
    python run_weekly.py --portfolio 2600         # set portfolio value
    python run_weekly.py --positions AAPL NVDA    # review open positions
    python run_weekly.py --portfolio 2800 --positions AAPL NVDA --position-values 224 168
    python run_weekly.py --no-fetch               # skip price refresh (data already fresh)
    python run_weekly.py --top 15                 # show top 15 signals per strategy (default 10)

What this does
--------------
1. Refresh prices via data/fetch.py
2. Run screener via screener/run.py
3. Filter signals for both strategies:
     A_6M   — C5 ∧ C6 ∧ C9 must all pass, score ≥ 2
     Mode C — score ≥ 7, no specific criteria combination required
4. Size every signal via sizer rules
5. Print a clean decision-journal-ready summary to stdout
6. Append a dated entry to paper_trading_log.csv

Strategies are tracked separately in the log (row_type: A6M_SIGNAL / C_SIGNAL).
"""

import argparse
import csv
import subprocess
import sys
import textwrap
from datetime import date, datetime
from pathlib import Path

# ── project root (this file lives at project root) ──────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_FILE     = PROJECT_ROOT / "paper_trading_log.csv"

# ── strategy parameters ──────────────────────────────────────────────────────
A6M_MIN_SCORE   = 2   # score ≥ 2, must pass C5 ∧ C6 ∧ C9
MODE_C_MIN_SCORE = 7  # score ≥ 7, no criteria combination required


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], step_label: str) -> subprocess.CompletedProcess:
    """Run a subprocess, stream output, exit on failure."""
    print(f"\n{'─'*60}")
    print(f"  STEP: {step_label}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] {step_label} exited with code {result.returncode}.")
        print("Fix the issue above before proceeding.")
        sys.exit(result.returncode)
    return result


def _size_label(score: int, portfolio: float, mode_c: bool = False) -> tuple[float, str]:
    """
    Position sizing rules.

    A_6M:   score ≥ 3 → 8%, score 2 → 5%
    Mode C: score 9   → 8% (high conviction), score 7-8 → 5% (medium)
    Hard max £1,000  |  Hard min £50
    """
    if mode_c:
        pct  = 0.08 if score >= 8 else 0.05
        tier = "HIGH CONVICTION" if score >= 8 else "MEDIUM"
    else:
        pct  = 0.08 if score >= 3 else 0.05
        tier = "HIGH CONVICTION" if score >= 3 else "MEDIUM"

    raw  = portfolio * pct
    size = max(50.0, min(raw, 1000.0))
    return size, tier


def _load_screener_results() -> list[dict]:
    """
    Load the most recent screener output CSV.
    Prefers screener/output/screener_YYYYMMDD.csv matching today,
    then falls back to the most recently modified file in that directory.
    """
    out_dir    = PROJECT_ROOT / "screener" / "output"
    today_file = out_dir / f"screener_{date.today().strftime('%Y%m%d')}.csv"

    if today_file.exists():
        csv_path = today_file
    else:
        candidates = sorted(out_dir.glob("screener_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"\n[WARN] No screener output found in {out_dir}")
            print("       Run: python screener/run.py --out screener/output")
            return []
        csv_path = candidates[-1]
        print(f"[INFO] Using most recent screener file: {csv_path.name}")

    COLUMN_MAP = {
        "ticker": ["ticker", "Ticker", "symbol", "Symbol"],
        "score":  ["score",  "Score",  "total_score"],
        "C5":     ["C5", "c5", "dip", "c5_dip_entry"],
        "C6":     ["C6", "c6", "above_200ma", "c6_above_200ma"],
        "C9":     ["C9", "c9", "momentum_6m", "c9_momentum_6m"],
    }

    def find_col(header: list[str], aliases: list[str]) -> str | None:
        for a in aliases:
            if a in header:
                return a
        return None

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        col    = {k: find_col(header, v) for k, v in COLUMN_MAP.items()}
        missing = [k for k, v in col.items() if v is None]
        if missing:
            print(f"[WARN] Screener CSV missing columns: {missing}. Check screener/run.py.")
            return []
        for row in reader:
            rows.append({
                "ticker": row[col["ticker"]],
                "score":  int(row[col["score"]]),
                "C5":     str(row[col["C5"]]).strip().lower() in ("1", "true", "yes", "pass"),
                "C6":     str(row[col["C6"]]).strip().lower() in ("1", "true", "yes", "pass"),
                "C9":     str(row[col["C9"]]).strip().lower() in ("1", "true", "yes", "pass"),
            })
    return rows


def _filter_a6m(rows: list[dict]) -> list[dict]:
    """A_6M: C5 ∧ C6 ∧ C9 must all pass, score ≥ 2."""
    return sorted(
        [r for r in rows if r["score"] >= A6M_MIN_SCORE and r["C5"] and r["C6"] and r["C9"]],
        key=lambda r: r["score"], reverse=True,
    )


def _filter_mode_c(rows: list[dict]) -> list[dict]:
    """Mode C: score ≥ 7, no specific criteria combination required."""
    return sorted(
        [r for r in rows if r["score"] >= MODE_C_MIN_SCORE],
        key=lambda r: r["score"], reverse=True,
    )


def _cash_line(portfolio: float) -> str:
    return (f"10% cash reserve = £{portfolio*0.10:,.0f}  |  "
            f"Max deployable = £{portfolio*0.90:,.0f}")


def _print_signals(signals: list[dict], portfolio: float, top: int, mode_c: bool = False) -> None:
    """Print a formatted signal table."""
    label    = "Mode C (score ≥ 7)" if mode_c else "A_6M (C5 ∧ C6 ∧ C9, score ≥ 2)"
    top_sigs = signals[:top]

    print(f"\n  ── {label} ──\n")

    if not top_sigs:
        print("  No signals this week.")
        return

    print(f"  {'#':<4} {'TICKER':<8} {'SCORE':<7} {'TIER':<18} {'SIZE':>8}  {'C5':^4} {'C6':^4} {'C9':^4}")
    print(f"  {'─'*4} {'─'*8} {'─'*7} {'─'*18} {'─'*8}  {'─'*4} {'─'*4} {'─'*4}")

    for i, sig in enumerate(top_sigs, 1):
        size, tier = _size_label(sig["score"], portfolio, mode_c=mode_c)
        raw  = portfolio * (0.08 if (mode_c and sig["score"] == 9) or
                                    (not mode_c and sig["score"] >= 3) else 0.05)
        note = "⚡cap" if raw > 1000 else ("⚡flr" if raw < 50 else "")
        c5   = "✓" if sig["C5"] else "✗"
        c6   = "✓" if sig["C6"] else "✗"
        c9   = "✓" if sig["C9"] else "✗"
        print(
            f"  {i:<4} {sig['ticker']:<8} {sig['score']:<7} {tier:<18} "
            f"£{size:>6,.0f} {note:<5}  {c5:^4} {c6:^4} {c9:^4}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A_6M + Mode C Sunday paper-trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python run_weekly.py --portfolio 2600
              python run_weekly.py --portfolio 2750 --positions AAPL NVDA --position-values 224 168
              python run_weekly.py --no-fetch --top 15
        """),
    )
    parser.add_argument("--portfolio",       type=float, default=2600.0,
                        help="Current ISA portfolio value in £ (default: 2600)")
    parser.add_argument("--positions",       nargs="*",  default=[],   metavar="TICKER",
                        help="Open paper positions to review")
    parser.add_argument("--position-values", nargs="*",  type=float, default=[], metavar="£",
                        help="Current market value of each open position (same order as --positions)")
    parser.add_argument("--entry-prices",    nargs="*",  type=float, default=[], metavar="£",
                        help="Entry price per share for each open position")
    parser.add_argument("--entry-dates",     nargs="*",  default=[],   metavar="YYYY-MM-DD",
                        help="Entry date for each open position")
    parser.add_argument("--no-fetch",        action="store_true",
                        help="Skip price refresh (data already current)")
    parser.add_argument("--top",             type=int,   default=10,
                        help="Top N signals to show per strategy (default: 10)")
    args = parser.parse_args()

    today     = date.today()
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    portfolio = args.portfolio

    # ── validate position args ───────────────────────────────────────────────
    positions    = args.positions or []
    pos_values   = args.position_values or []
    entry_prices = args.entry_prices or []
    entry_dates  = args.entry_dates  or []

    for flag, lst in [("--position-values", pos_values),
                      ("--entry-prices",    entry_prices),
                      ("--entry-dates",     entry_dates)]:
        if lst and len(lst) != len(positions):
            print(f"[ERROR] {flag} must have same count as --positions")
            sys.exit(1)

    pos_values   = pos_values   + [None] * (len(positions) - len(pos_values))
    entry_prices = entry_prices + [None] * (len(positions) - len(entry_prices))
    entry_dates  = entry_dates  + [None] * (len(positions) - len(entry_dates))

    # ── STEP 1: Refresh prices ───────────────────────────────────────────────
    if not args.no_fetch:
        run([sys.executable, str(PROJECT_ROOT / "data" / "fetch.py")],
            "Refresh prices (data/fetch.py)")
    else:
        print("\n[SKIP] Price refresh skipped (--no-fetch)")

    # ── STEP 2: Run screener ─────────────────────────────────────────────────
    run([sys.executable, str(PROJECT_ROOT / "screener" / "run.py"),
         "--out", str(PROJECT_ROOT / "screener" / "output")],
        "Run screener (screener/run.py)")

    # ── STEP 3: Load and filter ──────────────────────────────────────────────
    all_rows = _load_screener_results()
    a6m_sigs = _filter_a6m(all_rows)
    c_sigs   = _filter_mode_c(all_rows)

    # ── STEP 4: Summary header ───────────────────────────────────────────────
    WIDTH = 72
    SEP   = "═" * WIDTH

    print(f"\n\n{SEP}")
    print(f"  WEEKLY SIGNAL SUMMARY  —  {today}  ({now})")
    print(SEP)
    print(f"  Portfolio value  : £{portfolio:,.2f}")
    print(f"  {_cash_line(portfolio)}")
    print(f"  A_6M signals     : {len(a6m_sigs)}  "
          f"(showing top {min(len(a6m_sigs), args.top)})")
    print(f"  Mode C signals   : {len(c_sigs)}  "
          f"(showing top {min(len(c_sigs), args.top)})")
    print(f"  Open positions   : {len(positions)}")
    print(SEP)

    # ── STEP 5: Signal tables ────────────────────────────────────────────────
    _print_signals(a6m_sigs, portfolio, args.top, mode_c=False)
    _print_signals(c_sigs,   portfolio, args.top, mode_c=True)

    # ── STEP 6: Open position review ─────────────────────────────────────────
    if positions:
        print(f"\n  ── OPEN POSITION REVIEW ──\n")
        print(f"  {'TICKER':<8} {'STRATEGY':<10} {'DATE':<12} {'ENTRY':>8} "
              f"{'CUR':>8} {'RETURN':>8} {'STOP':>8}  STATUS")
        print(f"  {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*28}")

        for ticker, cur_val, ep, ed in zip(positions, pos_values, entry_prices, entry_dates):
            ep_str  = f"£{ep:,.2f}"      if ep      is not None else "—"
            cur_str = f"£{cur_val:,.2f}" if cur_val is not None else "—"
            ed_str  = ed                 if ed      is not None else "—"

            if cur_val is not None and ep is not None:
                ret_pct  = (cur_val - ep) / ep * 100
                ret_str  = f"{ret_pct:+.1f}%"
                stop_str = f"£{ep*0.70:,.2f}"
                if ret_pct <= -30:
                    status = "⛔ STOP — mandatory written review"
                elif ret_pct >= 30:
                    status = "✅ +30% — review thesis, consider trim"
                else:
                    status = "HOLD — confirm thesis still valid"
            else:
                ret_str  = "—"
                stop_str = "—"
                status   = "supply --entry-prices and --position-values"

            in_a6m   = any(s["ticker"] == ticker for s in a6m_sigs)
            in_c     = any(s["ticker"] == ticker for s in c_sigs)
            strategy = ("A6M+C" if in_a6m and in_c
                        else "A6M" if in_a6m
                        else "C"   if in_c
                        else "—")

            print(f"  {ticker:<8} {strategy:<10} {ed_str:<12} {ep_str:>8} "
                  f"{cur_str:>8} {ret_str:>8} {stop_str:>8}  {status}")

    # ── Decision journal checklist ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DECISION JOURNAL CHECKLIST")
    print(SEP)
    for check in [
        "Open Decision Journal — new entry for today",
        "For EACH A_6M signal: record ticker, score, C5/C6/C9, decision, reasoning",
        "For EACH Mode C signal: record ticker, score, decision, reasoning",
        "Note any tickers in BOTH strategies — higher conviction, journal separately",
        "For EACH open position: confirm thesis still valid (or log invalidation)",
        "Log ANY overrides with written reasoning",
        "Note market conditions / anything unusual this week",
        "Record open position returns in monthly summary (last Sunday of month)",
    ]:
        print(f"  [ ] {check}")

    print(f"\n  Override rate reminder: >30% over any 3-month window = fix the system.")
    print(f"\n{SEP}\n")

    # ── STEP 7: Append to log ────────────────────────────────────────────────
    _append_log(today, portfolio, a6m_sigs, c_sigs, positions, pos_values, args.top)

    print(f"  Log appended → {LOG_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  Done. Good luck this week.\n")


def _append_log(
    today:      date,
    portfolio:  float,
    a6m_sigs:   list[dict],
    c_sigs:     list[dict],
    positions:  list[str],
    pos_values: list,
    top:        int,
) -> None:
    """Append week summary + per-signal rows to paper_trading_log.csv."""
    write_header = not LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "date", "portfolio_value",
                "total_a6m_signals", "total_c_signals",
                "ticker", "score", "tier", "recommended_size_gbp",
                "open_positions", "row_type",
            ])

        open_str = "|".join(positions)

        writer.writerow([
            today, f"{portfolio:.2f}",
            len(a6m_sigs), len(c_sigs),
            "", "", "", "",
            open_str, "WEEK_SUMMARY",
        ])

        for sig in a6m_sigs[:top]:
            size, tier = _size_label(sig["score"], portfolio, mode_c=False)
            writer.writerow([
                today, f"{portfolio:.2f}",
                len(a6m_sigs), len(c_sigs),
                sig["ticker"], sig["score"], tier, f"{size:.0f}",
                "", "A6M_SIGNAL",
            ])

        for sig in c_sigs[:top]:
            size, tier = _size_label(sig["score"], portfolio, mode_c=True)
            writer.writerow([
                today, f"{portfolio:.2f}",
                len(a6m_sigs), len(c_sigs),
                sig["ticker"], sig["score"], tier, f"{size:.0f}",
                "", "C_SIGNAL",
            ])


if __name__ == "__main__":
    main()