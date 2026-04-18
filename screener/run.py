"""
screener/run.py
Scores every ticker against 9 ICD criteria, ranks by score,
outputs top N to terminal and optionally saves CSV.

ICD Criteria (/9):
    C1 — Larger than priced         sector-aware: asset/mcap OR margin+growth
    C2 — Consistent revenue         positive in 3 of 4 quarters
    C3 — Net income improving       newest quarter > oldest quarter
    C4 — Ratio improving            asset/liability ratio newest > oldest
    C5 — Dip entry                  below 52w high AND momentum_3m < -10%
    C6 — Above 200MA                close > 200-day moving average
    C7 — Margin quality             stable/improving AND recent 2Q avg > 0.40
    C8 — Cash flow improving        operating CF newest > oldest
    C9 — 6M positive momentum       momentum_6m > 0

Conviction tiers:
    8–9 = High     → 4–5% of portfolio
    6–7 = Medium   → 2–3% of portfolio
    <6  = Speculative → max 1%

Usage:
    python screener/run.py
    python screener/run.py --top 30
    python screener/run.py --out results/
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Criteria definitions ──────────────────────────────────────────────────────

CRITERIA = {
    "C1": "larger than priced — asset-heavy: asset/mcap, hardware: margin>0.45+growth>15%, light: margin>0.55+growth>12%",
    "C2": "revenue positive in 3 of 4 quarters",
    "C3": "net income improving newest vs oldest quarter",
    "C4": "asset/liability ratio improving newest vs oldest quarter",
    "C5": "dip — below 52w high AND momentum_3m < -10%",
    "C6": "above 200-day moving average",
    "C7": "gross margin stable or improving AND recent 2Q avg > 0.40",
    "C8": "operating cash flow improving newest vs oldest quarter",
    "C9": "6M positive momentum",
}

# ── Thresholds ────────────────────────────────────────────────────────────────

C1_ASSET_MCAP_MIN           = 0.5
C1_LIGHT_GROSS_MARGIN_MIN   = 0.55
C1_LIGHT_REVENUE_GROWTH_MIN = 0.12
C2_CONSISTENCY_MIN          = 0.75
C5_MOMENTUM_3M_MAX          = -10.0
C7_GROSS_MARGIN_FLOOR       = 0.40
C1_HARDWARE_GROSS_MARGIN_MIN   = 0.45
C1_HARDWARE_REVENUE_GROWTH_MIN = 0.15

# ── Sector classification ─────────────────────────────────────────────────────

ASSET_HEAVY_SECTORS = {
    "Energy", "Utilities", "Industrials",
    "Basic Materials", "Real Estate", "Financial Services",
}
ASSET_LIGHT_SECTORS = {
    "Technology", "Healthcare", "Communication Services",
    "Consumer Cyclical", "Consumer Defensive",
}
# Hardware/semiconductor companies — lower margins structural, compensate with growth
HARDWARE_INDUSTRIES = {
    "Semiconductors", "Semiconductor Equipment & Materials",
    "Electronic Components", "Electronics & Computer Distribution",
    "Computer Hardware",
}

SECTOR_ENCODING = {
    "Basic Materials":        0,
    "Communication Services": 1,
    "Consumer Cyclical":      2,
    "Consumer Defensive":     3,
    "Energy":                 4,
    "Financial Services":     5,
    "Healthcare":             6,
    "Industrials":            7,
    "Real Estate":            8,
    "Technology":             9,
    "Utilities":              10,
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_tickers(pf: pl.DataFrame, ff: pl.DataFrame) -> pl.DataFrame:
    """
    Join price and fundamental features, score each ticker 0–9.
    Returns full DataFrame with per-criterion boolean columns, total score,
    conviction tier, sector encoding, and 1W momentum tiebreaker.
    """
    df = pf.join(ff, on="ticker", how="inner")

    df = df.with_columns([

        # C1 — sector-aware: asset-heavy uses balance sheet ratio,
        #      asset-light uses margin quality + revenue growth
        pl.when(pl.col("sector").is_in(list(ASSET_HEAVY_SECTORS)))
          .then(
              pl.col("asset_to_mcap_ratio").is_not_null() &
              (pl.col("asset_to_mcap_ratio") > C1_ASSET_MCAP_MIN)
          )
          .when(pl.col("sector").is_in(list(ASSET_LIGHT_SECTORS)) &
               ~pl.col("industry").is_in(list(HARDWARE_INDUSTRIES)))
          .then(
              pl.col("gross_margin_latest").is_not_null() &
              (pl.col("gross_margin_latest") > C1_LIGHT_GROSS_MARGIN_MIN) &
              pl.col("revenue_growth_yoy").is_not_null() &
              (pl.col("revenue_growth_yoy") > C1_LIGHT_REVENUE_GROWTH_MIN)
          )
          .when(pl.col("industry").is_in(list(HARDWARE_INDUSTRIES)))
          .then(
              pl.col("gross_margin_latest").is_not_null() &
              (pl.col("gross_margin_latest") > C1_HARDWARE_GROSS_MARGIN_MIN) &
              pl.col("revenue_growth_yoy").is_not_null() &
              (pl.col("revenue_growth_yoy") > C1_HARDWARE_REVENUE_GROWTH_MIN)
          )
          .otherwise(
              pl.col("asset_to_mcap_ratio").is_not_null() &
              (pl.col("asset_to_mcap_ratio") > C1_ASSET_MCAP_MIN)
          )
          .alias("c1_larger_than_priced"),

        # C2 — revenue positive in 3 of 4 quarters
        pl.when(
            pl.col("revenue_consistency").is_not_null() &
            (pl.col("revenue_consistency") >= C2_CONSISTENCY_MIN)
        ).then(True).otherwise(False).alias("c2_consistent_revenue"),

        # C3 — net income improving newest vs oldest quarter
        pl.when(pl.col("net_income_improving").is_not_null())
          .then(pl.col("net_income_improving"))
          .otherwise(False)
          .alias("c3_net_income_improving"),

        # C4 — asset/liability ratio improving newest vs oldest
        pl.when(pl.col("asset_liability_ratio_improving").is_not_null())
          .then(pl.col("asset_liability_ratio_improving"))
          .otherwise(False)
          .alias("c4_ratio_improving"),

        # C5 — swing trade dip: below 52w high AND 3M momentum < -10%
        pl.when(
            pl.col("pct_below_52w_high").is_not_null() &
            pl.col("momentum_3m").is_not_null() &
            (pl.col("pct_below_52w_high") < 0) &
            (pl.col("momentum_3m") < C5_MOMENTUM_3M_MAX)
        ).then(True).otherwise(False).alias("c5_dip_entry"),

        # C6 — above 200-day moving average
        pl.when(pl.col("above_200ma").is_not_null())
          .then(pl.col("above_200ma"))
          .otherwise(False)
          .alias("c6_above_200ma"),

        # C7 — gross margin stable or improving AND recent 2Q avg above floor
        pl.when(
            pl.col("margin_stable_or_improving").is_not_null() &
            pl.col("margin_recent_avg").is_not_null() &
            pl.col("margin_stable_or_improving") &
            (pl.col("margin_recent_avg") > C7_GROSS_MARGIN_FLOOR)
        ).then(True).otherwise(False).alias("c7_margin_quality"),

        # C8 — operating cash flow improving newest vs oldest quarter
        pl.when(pl.col("cashflow_improving").is_not_null())
          .then(pl.col("cashflow_improving"))
          .otherwise(False)
          .alias("c8_cashflow_improving"),

        # C9 — 6M positive momentum
        pl.when(
            pl.col("momentum_6m").is_not_null() &
            (pl.col("momentum_6m") > 0)
        ).then(True).otherwise(False).alias("c9_momentum_6m"),
    ])

    # Total score /9
    df = df.with_columns([
        (
            pl.col("c1_larger_than_priced").cast(pl.Int32) +
            pl.col("c2_consistent_revenue").cast(pl.Int32) +
            pl.col("c3_net_income_improving").cast(pl.Int32) +
            pl.col("c4_ratio_improving").cast(pl.Int32) +
            pl.col("c5_dip_entry").cast(pl.Int32) +
            pl.col("c6_above_200ma").cast(pl.Int32) +
            pl.col("c7_margin_quality").cast(pl.Int32) +
            pl.col("c8_cashflow_improving").cast(pl.Int32) +
            pl.col("c9_momentum_6m").cast(pl.Int32)
        ).alias("score")
    ])

    # Conviction tier
    df = df.with_columns([
        pl.when(pl.col("score") >= 8).then(pl.lit("High"))
         .when(pl.col("score") >= 6).then(pl.lit("Medium"))
         .otherwise(pl.lit("Speculative"))
         .alias("tier")
    ])

    # Sector encoding for ML
    df = df.with_columns([
        pl.col("sector")
          .replace(SECTOR_ENCODING, default=-1)
          .alias("sector_encoded")
    ])

    # Sort: score descending, then 1W momentum as tiebreaker
    return df.sort(["score", "momentum_1w"], descending=[True, True], nulls_last=True)


def format_results(df: pl.DataFrame, top_n: int = 20) -> pl.DataFrame:
    return df.head(top_n).select([
        "ticker", "score", "tier",
        "c1_larger_than_priced", "c2_consistent_revenue",
        "c3_net_income_improving", "c4_ratio_improving",
        "c5_dip_entry", "c6_above_200ma",
        "c7_margin_quality", "c8_cashflow_improving", "c9_momentum_6m",
        "close", "momentum_6m", "momentum_3m", "momentum_1w",
        "pct_below_52w_high", "rsi_14",
        "gross_margin_latest", "margin_recent_avg",
        "net_income_improving", "cashflow_improving",
        "asset_liability_ratio_improving",
        "revenue_consistency", "revenue_growth_yoy",
        "market_cap", "sector", "sector_encoded",
    ])


def print_results(results: pl.DataFrame) -> None:
    print("\n" + "=" * 108)
    print(f"  SCREENER RESULTS — {datetime.today().strftime('%d %b %Y')}")
    print("=" * 108)
    print(
        f"  {'Rank':<5} {'Ticker':<8} {'Score':<7} {'Tier':<12}"
        f" {'C1':^3} {'C2':^3} {'C3':^3} {'C4':^3} {'C5':^3}"
        f" {'C6':^3} {'C7':^3} {'C8':^3} {'C9':^3}"
        f" {'6M%':>7} {'3M%':>7} {'1W%':>5} {'52wHi':>7} {'GM':>5}"
    )
    print("-" * 108)

    for i, row in enumerate(results.iter_rows(named=True), 1):
        def f(v): return " ✓" if v else " ✗"

        def fmt(v, fmt_str):
            return fmt_str.format(v) if v is not None else "—"

        print(
            f"  {i:<5} {row['ticker']:<8} {row['score']:<7} {row['tier']:<12}"
            f"{f(row['c1_larger_than_priced'])}"
            f"{f(row['c2_consistent_revenue'])}"
            f"{f(row['c3_net_income_improving'])}"
            f"{f(row['c4_ratio_improving'])}"
            f"{f(row['c5_dip_entry'])}"
            f"{f(row['c6_above_200ma'])}"
            f"{f(row['c7_margin_quality'])}"
            f"{f(row['c8_cashflow_improving'])}"
            f"{f(row['c9_momentum_6m'])}"
            f"  {fmt(row['momentum_6m'], '{:+.1f}%'):>6}"
            f"  {fmt(row['momentum_3m'], '{:+.1f}%'):>6}"
            f"  {fmt(row['momentum_1w'], '{:+.1f}%'):>4}"
            f"  {fmt(row['pct_below_52w_high'], '{:.1f}%'):>6}"
            f"  {fmt(row['gross_margin_latest'], '{:.2f}'):>4}"
        )

    print("=" * 108)
    print(
        "\n  C1=larger than priced  C2=consistent rev  C3=net income↑  C4=ratio↑"
        "  C5=dip  C6=200MA  C7=margin  C8=cashflow↑  C9=6M mom"
        "\n  Tiebreaker within same score: 1W momentum\n"
    )


def sanity_check(df: pl.DataFrame) -> None:
    known = ["PLTR", "AMD", "ASML"]
    top30 = df.head(30)["ticker"].to_list()
    print("SANITY CHECK — known winners in top 30:")
    all_pass = True
    for t in known:
        rank  = next((i + 1 for i, r in enumerate(df.iter_rows(named=True)) if r["ticker"] == t), None)
        row   = df.filter(pl.col("ticker") == t)
        score = row["score"][0] if not row.is_empty() else "N/A"
        in30  = t in top30
        status = "✓ PASS" if in30 else "✗ FAIL"
        if not in30:
            all_pass = False
        # Show which criteria failed
        if not row.is_empty():
            r = row.to_dicts()[0]
            fails = [k for k in CRITERIA if not r.get(f"c{k[1:]}_" + {
                "C1":"larger_than_priced","C2":"consistent_revenue",
                "C3":"net_income_improving","C4":"ratio_improving",
                "C5":"dip_entry","C6":"above_200ma",
                "C7":"margin_quality","C8":"cashflow_improving","C9":"momentum_6m",
            }.get(k,""), False)]
            print(f"  {status}  {t:<6}  rank={rank}  score={score}/9  failing={fails or 'none'}")
        else:
            print(f"  {status}  {t:<6}  not found")

    print(f"\n  Gate {'PASSED' if all_pass else 'FAILED'}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(top_n: int = 20, out_dir: str | None = None) -> pl.DataFrame:
    t0 = time.time()

    log.info("Loading cached data...")
    prices = pl.read_parquet(CACHE_DIR / "prices.parquet")
    funds  = pl.read_parquet(CACHE_DIR / "fundamentals.parquet")

    log.info("Computing features...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import price_features, fundamental_features
    pf = price_features(prices)
    ff = fundamental_features(funds)

    log.info("Scoring tickers...")
    scored = score_tickers(pf, ff)

    elapsed = time.time() - t0
    log.info(f"Scored {len(scored)} tickers in {elapsed:.1f}s")

    results = format_results(scored, top_n)
    print_results(results)
    sanity_check(scored)

    # Score distribution
    dist = scored.group_by("score").len().sort("score", descending=True)
    print("Score distribution:")
    for row in dist.iter_rows(named=True):
        bar = "█" * max(1, row["len"] // 5)
        print(f"  {row['score']}/9  {row['len']:>4}  {bar}")
    print()

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True)
        fname = out_path / f"screener_{datetime.today().strftime('%Y%m%d')}.csv"
        results.write_csv(fname)
        log.info(f"Results saved to {fname}")

    if elapsed > 180:
        log.warning(f"Runtime {elapsed:.1f}s exceeded 3 minute target")

    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int, default=20,   help="Number of top tickers to display")
    parser.add_argument("--out",  type=str, default=None, help="Directory to save CSV output")
    args = parser.parse_args()
    main(top_n=args.top, out_dir=args.out)