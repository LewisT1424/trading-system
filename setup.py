"""
setup.py
Run once to fetch the ticker universe and save to data/universe.csv.
Usage: python setup.py
"""

import io
import sys
import urllib.request
import pandas as pd
import polars as pl
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_html(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def fetch_sp500() -> list[str]:
    try:
        html  = fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df    = pd.read_html(io.BytesIO(html))[0]
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
        print(f"  ✓ S&P 500   — {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  ✗ S&P 500 failed: {e}")
        return []


def fetch_nasdaq100() -> list[str]:
    try:
        html    = fetch_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        tables  = pd.read_html(io.BytesIO(html))
        tickers = tables[4]["Ticker"].dropna().tolist()  # table 4 confirmed
        print(f"  ✓ Nasdaq 100 — {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  ✗ Nasdaq 100 failed: {e}")
        return []


print("Fetching ticker universe...")
sp500  = fetch_sp500()
ndx100 = fetch_nasdaq100()

universe = sorted(set(sp500 + ndx100))
print(f"\n  Total unique tickers: {len(universe)}")

if not universe:
    print("No tickers fetched — check connection and try again.")
    sys.exit(1)

out = DATA_DIR / "universe.csv"
pl.DataFrame({"ticker": universe}).write_csv(out)
print(f"  ✓ Saved to {out}")
print("\nSetup complete. Next: python data/fetch.py")