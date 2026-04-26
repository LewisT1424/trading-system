"""
research/fama_french/fetch_factors.py
======================================
Downloads Fama-French 3-factor monthly data from Ken French's public library
at Dartmouth and saves it as a clean parquet file for the spanning test.

What this downloads
-------------------
The standard Fama-French three-factor monthly file contains:
    - Mkt-RF : market excess return (market return minus risk-free rate)
    - SMB    : Small Minus Big (size factor)
    - HML    : High Minus Low (value factor)
    - RF     : risk-free rate (1-month T-bill)

All values are in percentage points (e.g. 1.23 means 1.23% that month).

Output
------
    research/fama_french/ff3_factors.parquet

Usage
-----
    python research/fama_french/fetch_factors.py
"""

import io
import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT / "research" / "fama_french"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ff3_factors.parquet"

# ── Ken French data library URL ───────────────────────────────────────────────

FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)

# ── Sample period — match our backtest window ─────────────────────────────────

SAMPLE_START = 201101   # January 2011 — YYYYMM integer format
SAMPLE_END   = 202604   # April 2026


def download_ff3() -> str:
    """
    Download the Ken French zip file and return the CSV content as a string.

    The zip contains a single CSV file. We extract it in memory — no temp
    files written to disk.
    """
    log.info(f"Downloading Fama-French 3-factor data from Dartmouth...")
    log.info(f"  URL: {FF3_URL}")

    response = requests.get(FF3_URL, timeout=30)
    response.raise_for_status()

    log.info(f"  Downloaded {len(response.content):,} bytes")

    # Extract CSV from zip in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        names = z.namelist()
        log.info(f"  Files in zip: {names}")
        csv_name = [n for n in names if n.endswith(".CSV") or n.endswith(".csv")][0]
        csv_content = z.read(csv_name).decode("utf-8", errors="replace")
        log.info(f"  Extracted: {csv_name} ({len(csv_content):,} chars)")

    return csv_content


def parse_ff3(csv_content: str) -> pl.DataFrame:
    """
    Parse the Ken French CSV into a clean Polars DataFrame.

    The CSV has a specific structure:
        - Several lines of header text before the data begins
        - Monthly data section: YYYYMM, Mkt-RF, SMB, HML, RF
        - Annual data section follows after a blank line (we ignore this)
        - Footer text at the end (we ignore this)

    We detect the start of monthly data by finding the first row where
    the first column is a 6-digit integer (YYYYMM format).
    """
    lines = csv_content.split("\n")

    # ── Find where monthly data starts ────────────────────────────────────────
    # Monthly data rows look like: "196307,  0.39, -0.44, -0.87,  0.27"
    # We skip any line where the first token isn't a 6-digit integer

    data_rows = []
    in_monthly_section = True

    for line in lines:
        line = line.strip()
        if not line:
            # Blank line signals end of monthly section — annual section follows
            if data_rows:
                in_monthly_section = False
            continue

        if not in_monthly_section:
            continue

        parts = [p.strip() for p in line.split(",")]

        # Skip header lines — first token must be a 6-digit integer
        if len(parts) < 5:
            continue
        try:
            date_int = int(parts[0])
        except ValueError:
            continue

        # Valid YYYYMM: year between 1900-2100, month between 01-12
        year  = date_int // 100
        month = date_int % 100
        if not (1900 <= year <= 2100 and 1 <= month <= 12):
            continue

        # Parse the four factor values
        try:
            mkt_rf = float(parts[1])
            smb    = float(parts[2])
            hml    = float(parts[3])
            rf     = float(parts[4])
        except (ValueError, IndexError):
            continue

        data_rows.append({
            "date_int": date_int,
            "mkt_rf":   mkt_rf,
            "smb":      smb,
            "hml":      hml,
            "rf":       rf,
        })

    log.info(f"Parsed {len(data_rows)} monthly rows from CSV")

    if not data_rows:
        raise ValueError("No valid monthly data found in CSV — check URL or file format")

    # ── Build DataFrame ────────────────────────────────────────────────────────
    df = pl.DataFrame(data_rows)

    # Convert YYYYMM integer to proper date (first day of each month)
    df = df.with_columns([
        pl.col("date_int").cast(pl.Utf8).str.strptime(
            pl.Date, "%Y%m"
        ).alias("date")
    ]).drop("date_int")

    # Reorder columns — date first
    df = df.select(["date", "mkt_rf", "smb", "hml", "rf"])

    return df


def filter_sample_period(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter to our backtest sample period: January 2011 to April 2026.

    Ken French data goes back to 1926 — we only need the period that
    matches our A_6M backtest for the spanning test.
    """
    start = pl.date(2011, 1, 1)
    end   = pl.date(2026, 4, 30)

    filtered = df.filter(
        (pl.col("date") >= start) &
        (pl.col("date") <= end)
    )

    log.info(f"Sample period filter: {len(df)} → {len(filtered)} months")
    log.info(f"  Date range: {filtered['date'].min()} → {filtered['date'].max()}")

    return filtered


def validate(df: pl.DataFrame) -> None:
    """
    Sanity checks on the downloaded factor data.
    Raises ValueError if anything looks wrong.
    """
    # Should have exactly 4 columns plus date
    assert set(df.columns) == {"date", "mkt_rf", "smb", "hml", "rf"}, \
        f"Unexpected columns: {df.columns}"

    # No nulls anywhere
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    assert total_nulls == 0, f"Unexpected nulls: {null_counts}"

    # Should have roughly 184 months (Jan 2011 to Apr 2026)
    assert 180 <= len(df) <= 190, \
        f"Unexpected row count: {len(df)} — expected ~184"

    # Market factor should have positive mean over long periods
    assert df["mkt_rf"].mean() > 0, \
        "Market factor mean is negative — data may be corrupted"

    # Risk free rate should always be non-negative
    assert (df["rf"] >= 0).all(), \
        "Negative risk-free rate found — data may be corrupted"

    # Values should be in percentage points — sanity check bounds
    assert df["mkt_rf"].abs().max() < 50, \
        "Market factor has extreme values — check units (should be % points)"

    log.info("All validation checks passed")


def print_summary(df: pl.DataFrame) -> None:
    """Print a clean summary of the downloaded factor data."""
    print()
    print("=" * 55)
    print("FAMA-FRENCH 3-FACTOR DATA — SUMMARY")
    print("=" * 55)
    print(f"  Months:      {len(df)}")
    print(f"  Date range:  {df['date'].min()} → {df['date'].max()}")
    print()
    print(f"  {'Factor':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Ann.':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for col in ["mkt_rf", "smb", "hml", "rf"]:
        series = df[col]
        mean   = series.mean()
        std    = series.std()
        mn     = series.min()
        mx     = series.max()
        ann    = mean * 12  # annualised

        print(f"  {col:<12} {mean:>8.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f} {ann:>8.2f}%")

    print()
    print("  Note: values are in percentage points per month")
    print("        Ann. = mean × 12 (simple annualisation)")
    print()

    # Regime check — RF before and after 2022
    pre_2022  = df.filter(pl.col("date") < pl.date(2022, 1, 1))["rf"].mean()
    post_2022 = df.filter(pl.col("date") >= pl.date(2022, 1, 1))["rf"].mean()
    print(f"  RF regime:   pre-2022 avg {pre_2022:.3f}%/mo  |  "
          f"post-2022 avg {post_2022:.3f}%/mo")
    print()


def main() -> pl.DataFrame:
    log.info("Starting Fama-French factor data download")

    # Step 1 — download
    csv_content = download_ff3()

    # Step 2 — parse
    df_full = parse_ff3(csv_content)
    log.info(f"Full dataset: {len(df_full)} months "
             f"({df_full['date'].min()} → {df_full['date'].max()})")

    # Step 3 — filter to sample period
    df = filter_sample_period(df_full)

    # Step 4 — validate
    validate(df)

    # Step 5 — save
    df.write_parquet(OUTPUT_FILE)
    log.info(f"Saved to {OUTPUT_FILE}")

    # Step 6 — print summary
    print_summary(df)

    return df


if __name__ == "__main__":
    main()