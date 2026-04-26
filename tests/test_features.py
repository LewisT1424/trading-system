"""
tests/test_features.py
Night 10 — feature validation and lookahead bias checks.

Run:
    pytest tests/test_features.py -v
    pytest tests/test_features.py -v -m "not slow"
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screener"))

from features import (
    price_features,
    fundamental_features,
    _add_rsi,
    _momentum,
    _volume_trend,
    _revenue_growth_yoy,
    _compute_trajectory,
    _balance_growth,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(
    ticker: str = "TEST",
    n: int = 300,
    start_price: float = 100.0,
    trend: float = 0.001,
) -> pl.DataFrame:
    """Generate synthetic price data with a gentle uptrend."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = [start_price * (1 + trend) ** i for i in range(n)]
    return pl.DataFrame({
        "ticker": [ticker] * n,
        "date":   dates.to_list(),
        "open":   [p * 0.99 for p in prices],
        "high":   [p * 1.01 for p in prices],
        "low":    [p * 0.98 for p in prices],
        "close":  prices,
        "volume": [1_000_000] * n,
    })


def make_funds(
    ticker: str = "TEST",
    n_quarters: int = 4,
    revenue: float = 1e9,
    gross_margin: float = 0.50,
    growing: bool = True,
) -> pl.DataFrame:
    """Generate synthetic quarterly fundamentals, newest first."""
    periods = []
    base = datetime(2025, 12, 31)
    for i in range(n_quarters):
        d = base - timedelta(days=90 * i)
        periods.append(d.strftime("%Y-%m-%d"))

    multiplier = 1.05 if growing else 0.95
    revenues = [revenue * (multiplier ** (n_quarters - 1 - i)) for i in range(n_quarters)]

    return pl.DataFrame({
        "ticker":            [ticker] * n_quarters,
        "period":            periods,
        "revenue":           revenues,
        "gross_profit":      [r * gross_margin for r in revenues],
        "operating_income":  [r * 0.20 for r in revenues],
        "net_income":        [r * 0.15 for r in revenues],
        "cost_of_revenue":   [r * (1 - gross_margin) for r in revenues],
        "total_assets":      [5e9 * (1.02 ** (n_quarters - 1 - i)) for i in range(n_quarters)],
        "total_liabilities": [2e9] * n_quarters,
        "total_debt":        [1e9] * n_quarters,
        "cash":              [500e6] * n_quarters,
        "operating_cashflow":[r * 0.18 for r in revenues],
        "free_cashflow":     [r * 0.15 for r in revenues],
        "capex":             [r * 0.03 for r in revenues],
        "gross_margin":      [gross_margin] * n_quarters,
        "net_margin":        [0.15] * n_quarters,
        "market_cap":        [10_000_000_000] * n_quarters,
        "trailing_pe":       [25.0] * n_quarters,
        "revenue_growth":    [0.10] * n_quarters,
        "free_cashflow_yield": [None] * n_quarters,
    })


# ── Price feature tests ───────────────────────────────────────────────────────

class TestPriceFeatures:

    def test_returns_one_row_per_ticker(self):
        prices = pl.concat([make_prices("AAA"), make_prices("BBB")])
        result = price_features(prices)
        assert len(result) == 2
        assert set(result["ticker"].to_list()) == {"AAA", "BBB"}

    def test_all_expected_columns_present(self):
        result = price_features(make_prices())
        expected = [
            "ticker", "close", "ma_200", "above_200ma", "high_52w",
            "pct_below_52w_high", "in_dip", "rsi_14",
            "momentum_6m", "momentum_1m", "volume_trend_50d",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_close_is_most_recent_price(self):
        prices = make_prices(n=300, start_price=100.0, trend=0.001)
        result = price_features(prices)
        last_price = prices.sort("date")["close"][-1]
        assert result["close"][0] == pytest.approx(last_price, rel=1e-6)

    def test_above_200ma_bool_type(self):
        result = price_features(make_prices())
        assert result["above_200ma"].dtype == pl.Boolean

    def test_above_200ma_true_for_uptrend(self):
        prices = make_prices(trend=0.005, n=300)
        result = price_features(prices)
        assert result["above_200ma"][0] is True

    def test_above_200ma_false_for_downtrend(self):
        prices = make_prices(trend=-0.003, n=300)
        result = price_features(prices)
        assert result["above_200ma"][0] is False

    def test_pct_below_52w_high_is_zero_or_negative(self):
        result = price_features(make_prices())
        assert (result["pct_below_52w_high"] <= 0).all()

    def test_pct_below_52w_high_is_zero_at_new_high(self):
        """Stock making new highs every day — should be 0%."""
        prices = make_prices(trend=0.01, n=300)
        result = price_features(prices)
        assert result["pct_below_52w_high"][0] == pytest.approx(0.0, abs=1.0)

    def test_pct_below_52w_high_negative_after_crash(self):
        """Stock crashed 30% — should be roughly -30%."""
        prices = make_prices(trend=0.002, n=280)
        last_close = prices["close"][-1]
        crash_dates = pd.date_range(
            prices["date"][-1] + timedelta(days=1), periods=20, freq="B"
        )
        crash_prices = [last_close * (0.985 ** i) for i in range(1, 21)]
        crash = pl.DataFrame({
            "ticker": ["TEST"] * 20,
            "date":   crash_dates.to_list(),
            "open":   crash_prices,
            "high":   crash_prices,
            "low":    [p * 0.99 for p in crash_prices],
            "close":  crash_prices,
            "volume": [1_000_000] * 20,
        })
        full = pl.concat([prices, crash])
        result = price_features(full)
        assert result["pct_below_52w_high"][0] < -10.0

    def test_rsi_bounded_0_to_100(self):
        result = price_features(make_prices())
        rsi = result["rsi_14"].drop_nulls()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_high_for_strong_uptrend(self):
        prices = make_prices(trend=0.015, n=300)
        result = price_features(prices)
        assert result["rsi_14"][0] > 70

    def test_rsi_low_for_strong_downtrend(self):
        prices = make_prices(trend=-0.01, n=300)
        result = price_features(prices)
        assert result["rsi_14"][0] < 40

    def test_momentum_6m_positive_for_uptrend(self):
        result = price_features(make_prices(trend=0.003, n=300))
        assert result["momentum_6m"][0] > 0

    def test_momentum_6m_negative_for_downtrend(self):
        result = price_features(make_prices(trend=-0.003, n=300))
        assert result["momentum_6m"][0] < 0

    def test_volume_trend_ratio_type(self):
        result = price_features(make_prices())
        assert result["volume_trend_50d"].dtype == pl.Float64

    def test_no_nulls_in_price_features_for_full_history(self):
        """With 300 rows of data all price features should be non-null."""
        result = price_features(make_prices(n=300))
        for col in result.columns:
            if col != "ticker":
                assert result[col].is_null().sum() == 0, f"{col} has unexpected nulls"

    def test_in_dip_false_at_new_high(self):
        prices = make_prices(trend=0.01, n=300)
        result = price_features(prices)
        assert result["in_dip"][0] is False

    def test_in_dip_true_after_significant_decline(self):
        prices = make_prices(trend=-0.005, n=300)
        result = price_features(prices)
        assert result["in_dip"][0] is True


# ── RSI tests ─────────────────────────────────────────────────────────────────

class TestRSI:

    def test_rsi_column_added(self):
        prices = make_prices()
        result = _add_rsi(prices, period=14)
        assert "rsi_14" in result.columns

    def test_rsi_not_null_after_warmup(self):
        prices = make_prices(n=100)
        result = _add_rsi(prices, period=14)
        non_null = result["rsi_14"].drop_nulls()
        assert len(non_null) > 0

    def test_rsi_all_gains_approaches_100(self):
        """Constant daily gains should push RSI toward 100."""
        n = 100
        prices_list = [100.0 + i * 2 for i in range(n)]
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pl.DataFrame({
            "ticker": ["TEST"] * n,
            "date": dates.to_list(),
            "close": prices_list,
        })
        result = _add_rsi(df, period=14)
        last_rsi = result["rsi_14"].drop_nulls()[-1]
        assert last_rsi > 90


# ── Momentum tests ────────────────────────────────────────────────────────────

class TestMomentum:

    def test_positive_momentum_for_uptrend(self):
        result = _momentum(make_prices(trend=0.003, n=300), lookback=126)
        assert result["momentum_6m"][0] > 0

    def test_negative_momentum_for_downtrend(self):
        result = _momentum(make_prices(trend=-0.003, n=300), lookback=126)
        assert result["momentum_6m"][0] < 0

    def test_no_lookahead_momentum(self):
        """
        Lookahead check: momentum computed on data up to day N
        should not use any prices from after day N.
        """
        full = make_prices(trend=0.002, n=300)
        subset_200 = full.head(200)

        m200 = _momentum(subset_200, lookback=126)["momentum_6m"][0]

        prices_sorted = full.sort("date")
        close_now  = prices_sorted["close"][199]
        close_past = prices_sorted["close"][199 - 126]
        expected = (close_now - close_past) / close_past * 100
        assert m200 == pytest.approx(expected, rel=0.01)


# ── Volume trend tests ────────────────────────────────────────────────────────

class TestVolumeTrend:

    def test_ratio_above_1_for_increasing_volume(self):
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        volumes = list(range(500_000, 500_000 + n * 10_000, 10_000))
        df = pl.DataFrame({
            "ticker": ["TEST"] * n,
            "date":   dates.to_list(),
            "open":   [100.0] * n,
            "high":   [101.0] * n,
            "low":    [99.0] * n,
            "close":  [100.0] * n,
            "volume": volumes,
        })
        result = _volume_trend(df, window=50)
        assert result["volume_trend_50d"][0] > 1.0

    def test_ratio_below_1_for_decreasing_volume(self):
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        volumes = list(range(1_500_000, 500_000, -10_000))[:n]
        df = pl.DataFrame({
            "ticker": ["TEST"] * n,
            "date":   dates.to_list(),
            "open":   [100.0] * n,
            "high":   [101.0] * n,
            "low":    [99.0] * n,
            "close":  [100.0] * n,
            "volume": volumes,
        })
        result = _volume_trend(df, window=50)
        assert result["volume_trend_50d"][0] < 1.0


# ── Fundamental feature tests ─────────────────────────────────────────────────

class TestFundamentalFeatures:

    def test_returns_one_row_per_ticker(self):
        funds = pl.concat([make_funds("AAA"), make_funds("BBB")])
        result = fundamental_features(funds)
        assert len(result) == 2

    def test_all_expected_columns_present(self):
        result = fundamental_features(make_funds())
        expected = [
            "ticker", "gross_margin_latest", "gross_margin_avg",
            "asset_growth", "liability_growth",
            "revenue_consistency", "revenue_trajectory", "margin_trajectory",
            "asset_to_mcap_ratio", "market_cap_bucket",
            "revenue_growth_yoy", "fcf_margin", "fcf_yield",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_revenue_consistency_1_for_all_positive(self):
        result = fundamental_features(make_funds(revenue=1e9))
        assert result["revenue_consistency"][0] == pytest.approx(1.0)

    def test_revenue_trajectory_1_for_growing(self):
        result = fundamental_features(make_funds(growing=True))
        assert result["revenue_trajectory"][0] == 1

    def test_revenue_trajectory_minus1_for_declining(self):
        result = fundamental_features(make_funds(growing=False))
        assert result["revenue_trajectory"][0] == -1

    def test_asset_growth_positive_for_growing_assets(self):
        result = fundamental_features(make_funds())
        assert result["asset_growth"][0] > 0

    def test_gross_margin_latest_matches_input(self):
        result = fundamental_features(make_funds(gross_margin=0.75))
        assert result["gross_margin_latest"][0] == pytest.approx(0.75, abs=0.01)

    def test_asset_to_mcap_ratio_computed(self):
        result = fundamental_features(make_funds())
        expected = 5e9 * (1.02 ** 3) / 10e9
        assert result["asset_to_mcap_ratio"][0] == pytest.approx(expected, rel=0.01)

    def test_market_cap_bucket_large(self):
        result = fundamental_features(make_funds())
        assert result["market_cap_bucket"][0] == 3

    def test_market_cap_bucket_mega(self):
        funds = make_funds()
        funds = funds.with_columns(pl.lit(500_000_000_000).cast(pl.Int64).alias("market_cap"))
        result = fundamental_features(funds)
        assert result["market_cap_bucket"][0] == 4

    def test_fcf_margin_computed(self):
        result = fundamental_features(make_funds())
        assert result["fcf_margin"][0] == pytest.approx(0.15, rel=0.05)

    def test_negative_gross_margin_handled(self):
        funds = make_funds(gross_margin=-0.10)
        result = fundamental_features(funds)
        assert result["gross_margin_latest"][0] < 0

    def test_revenue_growth_yoy_positive_for_growing(self):
        result = fundamental_features(make_funds(growing=True))
        assert result["revenue_growth_yoy"][0] > 0

    def test_revenue_growth_yoy_negative_for_declining(self):
        result = fundamental_features(make_funds(growing=False))
        assert result["revenue_growth_yoy"][0] < 0


# ── Lookahead bias checks ─────────────────────────────────────────────────────

class TestLookahead:

    def test_price_features_use_only_past_data(self):
        """
        Compute features on N rows and N+10 rows.
        The close price in the N-row result must equal close[N-1].
        """
        prices = make_prices(trend=0.002, n=310)
        subset_n   = prices.head(300)
        subset_n10 = prices.head(310)

        result_n   = price_features(subset_n)
        result_n10 = price_features(subset_n10)

        close_n   = result_n["close"][0]
        close_n10 = result_n10["close"][0]

        assert close_n != close_n10

        expected_close = prices.sort("date")["close"][299]
        assert close_n == pytest.approx(expected_close, rel=1e-9)

    def test_52w_high_uses_only_past_data(self):
        prices = make_prices(trend=0.002, n=300)
        result = price_features(prices)
        actual_max = prices.sort("date").tail(252)["close"].max()
        assert result["high_52w"][0] == pytest.approx(actual_max, rel=1e-6)

    def test_fundamental_features_use_only_available_quarters(self):
        funds_4q = make_funds(n_quarters=4)
        funds_6q = make_funds(n_quarters=6)

        result_4q = fundamental_features(funds_4q)
        result_6q = fundamental_features(funds_6q)

        assert result_4q["gross_margin_latest"][0] == pytest.approx(
            result_6q["gross_margin_latest"][0], rel=1e-6
        )

    def test_ma_200_does_not_use_future_prices(self):
        prices = make_prices(trend=0.001, n=300)
        result = price_features(prices)
        expected_ma = prices.sort("date").tail(200)["close"].mean()
        assert result["ma_200"][0] == pytest.approx(expected_ma, rel=1e-6)


# ── Integration — full pipeline on cached data ────────────────────────────────

@pytest.mark.slow
class TestFullPipeline:

    @pytest.fixture
    def cached_data(self):
        cache = ROOT / "data" / "cache"
        if not (cache / "prices.parquet").exists():
            pytest.skip("Cached data not available — run data/fetch.py first")
        prices = pl.read_parquet(cache / "prices.parquet")
        funds  = pl.read_parquet(cache / "fundamentals.parquet")
        return prices, funds

    def test_price_features_on_full_universe(self, cached_data):
        prices, _ = cached_data
        result = price_features(prices)
        assert len(result) >= 500
        assert result["rsi_14"].is_null().sum() == 0

        # Allow small null count for momentum_6m — tickers with < 126 days
        # of history (e.g. recent IPOs or newly added tickers) will be null.
        # Threshold: fewer than 1% of tickers may have null momentum_6m.
        null_count = result["momentum_6m"].is_null().sum()
        max_allowed = int(len(result) * 0.01)
        assert null_count <= max_allowed, (
            f"momentum_6m has {null_count} nulls — expected fewer than {max_allowed}. "
            "Check for tickers with insufficient price history."
        )

        assert (result["pct_below_52w_high"] <= 0).all()

    def test_fundamental_features_on_full_universe(self, cached_data):
        _, funds = cached_data
        result = fundamental_features(funds)
        assert len(result) >= 500
        assert result["revenue_consistency"].is_null().sum() == 0
        assert (result["revenue_consistency"] >= 0).all()
        assert (result["revenue_consistency"] <= 1).all()

    def test_known_winners_pass_quality_criteria(self, cached_data):
        """
        PLTR, AMD, ASML should have strong fundamentals.
        Skip individual tickers gracefully if missing from cache —
        some tickers may not have EDGAR coverage or sufficient history.
        Fail only if ALL tickers are missing, which indicates a data problem.
        """
        prices, funds = cached_data
        pf = price_features(prices)
        ff = fundamental_features(funds)

        tickers_to_check = ["PLTR", "AMD", "ASML"]
        checked = 0

        for ticker in tickers_to_check:
            p = pf.filter(pl.col("ticker") == ticker)
            f = ff.filter(pl.col("ticker") == ticker)

            if p.is_empty() or f.is_empty():
                # Log which ticker is missing but don't fail — EDGAR coverage
                # is patchy and some tickers may have been recently added
                import warnings
                warnings.warn(
                    f"{ticker} missing from {'price' if p.is_empty() else 'fundamental'} "
                    f"features — skipping. Check EDGAR coverage for this ticker.",
                    UserWarning
                )
                continue

            checked += 1

            # Revenue consistency — all quarters positive
            assert f["revenue_consistency"][0] == pytest.approx(1.0), \
                f"{ticker} revenue_consistency should be 1.0"

            # Positive gross margin above 30%
            gm = f["gross_margin_latest"][0]
            assert gm is not None and gm > 0.3, \
                f"{ticker} gross margin too low: {gm}"

            # Improving revenue trajectory
            assert f["revenue_trajectory"][0] == 1, \
                f"{ticker} revenue_trajectory should be 1"

        assert checked >= 1, (
            "All known winner tickers (PLTR, AMD, ASML) are missing from features. "
            "This indicates a data pipeline problem — run data/fetch.py and data/edgar_fetch.py."
        )

    def test_no_future_dates_in_features(self, cached_data):
        """Price features must not reference dates beyond today."""
        prices, _ = cached_data
        result = price_features(prices)
        today = datetime.today()
        max_date = prices["date"].max()
        assert max_date.replace(tzinfo=None) <= today + timedelta(days=2)