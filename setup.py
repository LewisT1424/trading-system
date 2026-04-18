"""
Night 7 — setup.py
Run once to create folder structure and fetch ticker universe.
Usage: python setup.py
"""

import os
import sys
import pandas as pd
import polars as pl
from pathlib import Path

ROOT = Path(__file__).parent

# ── Folder structure ──────────────────────────────────────────────────────────

DIRS = [
    "data",
    "screener",
    "backtest",
    "signals",
    "journal",
    "risk",
]

print("Creating folder structure...")
for d in DIRS:
    path = ROOT / d
    path.mkdir(exist_ok=True)
    (path / ".gitkeep").touch()
    print(f"  ✓ {d}/")


# ── Ticker universe ───────────────────────────────────────────────────────────

print("\nFetching ticker universe from Wikipedia...")

def fetch_sp500() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        # Clean: some entries have dots (BRK.B) — yfinance uses dashes (BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"  ✓ S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  ✗ S&P 500 fetch failed: {e}")
        return []

def fetch_nasdaq100() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # Find the table with a Ticker or Symbol column
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "ticker" in cols:
                tickers = t["Ticker"].dropna().tolist()
                print(f"  ✓ Nasdaq 100: {len(tickers)} tickers")
                return tickers
            elif "symbol" in cols:
                tickers = t["Symbol"].dropna().tolist()
                print(f"  ✓ Nasdaq 100: {len(tickers)} tickers")
                return tickers
        print("  ✗ Nasdaq 100: could not find ticker column")
        return []
    except Exception as e:
        print(f"  ✗ Nasdaq 100 fetch failed: {e}")
        return []

sp500  = fetch_sp500()
ndx100 = fetch_nasdaq100()

# Combine and deduplicate
universe = sorted(set(sp500 + ndx100))
print(f"\n  Total unique tickers: {len(universe)}")

# Save to data/
if universe:
    pl.DataFrame({"ticker": universe}).write_csv(ROOT / "data" / "universe.csv")
    print(f"  ✓ Saved to data/universe.csv")
else:
    print("  ✗ No tickers fetched — check internet connection and try again")
    sys.exit(1)

print("\nNight 7 setup complete. Next: python data/fetch.py")