"""
inject_ciks.py
Adds the 15 manually looked-up CIKs to the cached CIK map.
Run from project root: python inject_ciks.py
"""
import polars as pl
from pathlib import Path

ROOT     = Path.cwd()
CIK_PATH = ROOT / "data" / "cache" / "cik_map.parquet"

MANUAL_CIKS = {
    "SIVB": "0000719739",
    "FRC":  "0001132979",
    "SBNY": "0001288784",
    "CMA":  "0000028412",
    "FL":   "0000850209",
    "GPS":  "0000039911",
    "HBI":  "0001359841",
    "IPG":  "0000051644",
    "TWTR": "0001418091",
    "ATVI": "0000718877",
    "XLNX": "0000743988",
    "CELG": "0000816284",
    "ALXN": "0000899866",
    "AGN":  "0001578845",
    "CERN": "0000804753",
}

# Load existing map
existing = pl.read_parquet(CIK_PATH)
print(f"Existing CIK map: {len(existing)} entries")

existing_tickers = set(existing["ticker"].to_list())

# Add new entries
new_rows = [
    {"ticker": t, "cik": c}
    for t, c in MANUAL_CIKS.items()
    if t not in existing_tickers
]

already_had = [t for t in MANUAL_CIKS if t in existing_tickers]
if already_had:
    print(f"Already in map: {already_had}")

if new_rows:
    additions = pl.DataFrame(new_rows)
    updated = pl.concat([existing, additions])
    updated.write_parquet(CIK_PATH)
    print(f"Added {len(new_rows)} new CIKs: {[r['ticker'] for r in new_rows]}")
    print(f"Updated CIK map: {len(updated)} entries")
else:
    print("Nothing to add — all tickers already in map")