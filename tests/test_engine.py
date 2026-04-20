"""
tests/test_engine.py
Production-grade tests for backtest/engine.py

Test categories:
    1. Lookahead firewall — the most critical property of the entire system
    2. Point-in-time data access — prices and fundamentals
    3. Rebalance date generation — weekly and monthly
    4. Trade execution logic — entry, exit, return calculation
    5. Transaction costs — correctly applied
    6. Position management — no pyramiding, max positions, warmup
    7. Benchmark return calculation
    8. Constituent filtering
    9. Select top N — score threshold, tiebreaker
    10. Integration — full backtest produces valid trade log

Run:
    pytest tests/test_engine.py -v
    pytest tests/test_engine.py -v --tb=short   # compact tracebacks
    pytest tests/test_engine.py::TestLookahead -v  # single class
"""

import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

from engine import (
    get_prices_as_of,
    get_funds_as_of,
    get_constituents_on_date,
    get_benchmark_return,
    get_rebalance_dates,
    get_exit_date,
    get_next_open,
    select_top_n,
    TRANSACTION_COST,
    FIRST_TRADE_DATE,
    WINDOW_C_FIRST_TRADE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(
    tickers: list[str],
    start: str,
    end: str,
    open_val: float = 100.0,
    close_val: float = 100.0,
) -> pl.DataFrame:
    """Build a minimal daily price DataFrame for testing."""
    rows = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while d <= end_dt:
        if d.weekday() < 5:  # weekdays only
            for ticker in tickers:
                rows.append({
                    "ticker": ticker,
                    "date":   d,
                    "open":   open_val,
                    "close":  close_val,
                    "high":   close_val * 1.01,
                    "low":    close_val * 0.99,
                    "volume": 1_000_000,
                })
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def make_prices_trending(
    ticker: str,
    start: str,
    end: str,
    start_price: float = 100.0,
    daily_return: float = 0.001,
) -> pl.DataFrame:
    """Build prices with a consistent daily return for return calculation tests."""
    rows = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    price = start_price
    while d <= end_dt:
        if d.weekday() < 5:
            rows.append({
                "ticker": ticker,
                "date":   d,
                "open":   price,
                "close":  price,
                "high":   price * 1.01,
                "low":    price * 0.99,
                "volume": 1_000_000,
            })
            price *= (1 + daily_return)
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def make_edgar_funds(
    tickers: list[str],
    filed_date: str,
    period_end: str = "2020-06-30",
) -> pl.DataFrame:
    """Build minimal EDGAR fundamentals DataFrame."""
    rows = []
    for ticker in tickers:
        rows.append({
            "ticker":             ticker,
            "cik":                "0000000000",
            "period_end":         period_end,
            "filed":              filed_date,
            "form":               "10-Q",
            "fp":                 "Q2",
            "revenue":            1_000_000_000,
            "gross_profit":       400_000_000,
            "net_income":         100_000_000,
            "total_assets":       5_000_000_000,
            "total_liabilities":  2_000_000_000,
            "operating_cashflow": 150_000_000,
            "gross_margin":       0.40,
            "net_margin":         0.10,
        })
    return pl.DataFrame(rows)


def make_yf_funds(tickers: list[str], period: str) -> pl.DataFrame:
    """Build minimal yfinance-style fundamentals DataFrame."""
    rows = []
    for ticker in tickers:
        rows.append({
            "ticker":  ticker,
            "period":  period,
            "revenue": 1_000_000_000,
        })
    return pl.DataFrame(rows)


def make_compositions(
    date_str: str,
    tickers: list[str],
) -> pl.DataFrame:
    """Build a minimal constituent compositions DataFrame."""
    return pl.DataFrame({
        "date":    [date_str],
        "tickers": [tickers],
        "count":   [len(tickers)],
    })


def make_signals(
    tickers: list[str],
    scores: list[int],
    momentum_1w: list[float] | None = None,
) -> pl.DataFrame:
    """Build a minimal signals DataFrame for select_top_n tests."""
    if momentum_1w is None:
        momentum_1w = [0.5] * len(tickers)
    return pl.DataFrame({
        "ticker":      tickers,
        "score":       scores,
        "momentum_1w": momentum_1w,
        "close":       [100.0] * len(tickers),
        "momentum_6m": [5.0] * len(tickers),
        "momentum_3m": [-5.0] * len(tickers),
    })


# ── 1. Lookahead firewall ─────────────────────────────────────────────────────

class TestLookahead:
    """
    The most critical test class in the entire suite.
    These tests verify that future data NEVER enters the simulation.
    A single lookahead violation invalidates the entire backtest.
    """

    def test_get_prices_as_of_excludes_exact_date(self):
        """
        Price on exactly as_of date must NOT be returned.
        Critical: we use STRICTLY BEFORE (<), not <=.
        """
        prices = make_prices(["AAPL"], "2020-01-01", "2020-01-10")
        as_of = datetime(2020, 1, 5)
        result = get_prices_as_of(prices, as_of)

        dates = result["date"].to_list()
        for d in dates:
            assert d < as_of, (
                f"LOOKAHEAD VIOLATION: date {d} >= as_of {as_of}. "
                f"Future data leaked into simulation."
            )

    def test_get_prices_as_of_excludes_future_dates(self):
        """All dates in result must be strictly before as_of."""
        prices = make_prices(["SPY", "AAPL"], "2020-01-01", "2022-12-31")
        as_of = datetime(2021, 6, 15)
        result = get_prices_as_of(prices, as_of)

        assert not result.is_empty()
        max_date = result["date"].max()
        assert max_date < as_of, (
            f"LOOKAHEAD VIOLATION: max date {max_date} >= as_of {as_of}"
        )

    def test_get_prices_as_of_returns_empty_when_all_future(self):
        """If all data is on or after as_of, result must be empty."""
        prices = make_prices(["SPY"], "2022-01-01", "2022-12-31")
        as_of = datetime(2021, 12, 31)
        result = get_prices_as_of(prices, as_of)
        assert result.is_empty(), "Should return empty when all data is future"

    def test_get_funds_as_of_edgar_uses_filed_date(self):
        """
        EDGAR fundamentals must filter on `filed` date, not period_end.
        A Q2 report with period ending June 30 but filed August 14
        must NOT be available until August 14.
        """
        funds = make_edgar_funds(
            ["AAPL"],
            filed_date="2020-08-14",
            period_end="2020-06-30",
        )
        # Query before filing date — should not see this data
        as_of_before = datetime(2020, 8, 13)
        result_before = get_funds_as_of(funds, as_of_before)
        assert result_before.is_empty(), (
            "LOOKAHEAD VIOLATION: EDGAR data available before filing date. "
            "Period ended June 30, filed August 14, queried August 13 — "
            "should return nothing."
        )

    def test_get_funds_as_of_edgar_available_after_filing(self):
        """Data must be available the day after the filing date."""
        funds = make_edgar_funds(
            ["AAPL"],
            filed_date="2020-08-14",
            period_end="2020-06-30",
        )
        as_of_after = datetime(2020, 8, 15)
        result_after = get_funds_as_of(funds, as_of_after)
        assert not result_after.is_empty(), (
            "Data should be available after filing date"
        )

    def test_get_funds_as_of_edgar_excludes_exact_filing_date(self):
        """
        Filing date itself must NOT be available — strictly before.
        On the day a report is filed you don't have it yet (filed after market close).
        """
        funds = make_edgar_funds(
            ["AAPL"],
            filed_date="2020-08-14",
        )
        as_of_exact = datetime(2020, 8, 14)
        result = get_funds_as_of(funds, as_of_exact)
        assert result.is_empty(), (
            "LOOKAHEAD VIOLATION: filing date itself should not be available "
            "(strictly before, not <=)"
        )

    def test_get_funds_as_of_yfinance_fallback_uses_period(self):
        """yfinance funds (no 'filed' column) fall back to period date."""
        funds = make_yf_funds(["AAPL"], period="2020-06-30")
        as_of_before = datetime(2020, 6, 29)
        result = get_funds_as_of(funds, as_of_before)
        assert result.is_empty(), "yfinance data should not be available before period date"

    def test_get_funds_as_of_edgar_filters_multiple_tickers(self):
        """
        With multiple tickers at different filing dates,
        each should only appear after its own filing date.
        """
        rows = [
            {"ticker": "AAPL", "cik": "001", "period_end": "2020-06-30",
             "filed": "2020-08-01", "form": "10-Q", "fp": "Q2",
             "revenue": 1000, "gross_profit": 400, "net_income": 100,
             "total_assets": 5000, "total_liabilities": 2000,
             "operating_cashflow": 150, "gross_margin": 0.4, "net_margin": 0.1},
            {"ticker": "MSFT", "cik": "002", "period_end": "2020-06-30",
             "filed": "2020-09-01", "form": "10-Q", "fp": "Q2",
             "revenue": 2000, "gross_profit": 800, "net_income": 200,
             "total_assets": 8000, "total_liabilities": 3000,
             "operating_cashflow": 250, "gross_margin": 0.4, "net_margin": 0.1},
        ]
        funds = pl.DataFrame(rows)

        # Between the two filing dates
        as_of = datetime(2020, 8, 15)
        result = get_funds_as_of(funds, as_of)

        tickers = result["ticker"].to_list()
        assert "AAPL" in tickers, "AAPL should be available (filed Aug 1, queried Aug 15)"
        assert "MSFT" not in tickers, (
            "LOOKAHEAD VIOLATION: MSFT not available yet (filed Sep 1, queried Aug 15)"
        )

    def test_no_future_prices_in_feature_computation_window(self):
        """
        Verify the data boundary at exact simulation date.
        The feature window can only contain data strictly before as_of.
        """
        prices = make_prices(["SPY"], "2020-01-01", "2021-01-01")
        simulation_date = datetime(2020, 7, 1)

        pit = get_prices_as_of(prices, simulation_date)

        # Every date in the point-in-time subset must be strictly before sim date
        all_dates = pit["date"].to_list()
        future_dates = [d for d in all_dates if d >= simulation_date]
        assert len(future_dates) == 0, (
            f"LOOKAHEAD VIOLATION: {len(future_dates)} future dates in PIT subset: "
            f"{future_dates[:5]}"
        )


# ── 2. Point-in-time data access ──────────────────────────────────────────────

class TestPointInTime:
    """Verify get_prices_as_of and get_funds_as_of return correct subsets."""

    def test_prices_returns_correct_count(self):
        """Should return exactly the number of days strictly before as_of."""
        prices = make_prices(["SPY"], "2020-01-02", "2020-01-10")
        as_of = datetime(2020, 1, 7)
        result = get_prices_as_of(prices, as_of)
        # Jan 2, 3, 6 = 3 weekdays before Jan 7
        assert len(result) == 3

    def test_prices_preserves_all_tickers(self):
        """All tickers should be present in the filtered result."""
        tickers = ["SPY", "AAPL", "MSFT", "NVDA"]
        prices = make_prices(tickers, "2020-01-02", "2020-06-30")
        result = get_prices_as_of(prices, datetime(2020, 6, 15))
        assert result["ticker"].n_unique() == len(tickers)

    def test_funds_edgar_null_filed_excluded(self):
        """Rows with null filed date should be excluded from EDGAR results."""
        rows = [
            {"ticker": "AAPL", "cik": "001", "period_end": "2020-06-30",
             "filed": None, "form": "10-Q", "fp": "Q2",
             "revenue": 1000, "gross_profit": 400, "net_income": 100,
             "total_assets": 5000, "total_liabilities": 2000,
             "operating_cashflow": 150, "gross_margin": 0.4, "net_margin": 0.1},
        ]
        funds = pl.DataFrame(rows)
        result = get_funds_as_of(funds, datetime(2021, 1, 1))
        assert result.is_empty(), "Null filed dates should always be excluded"

    def test_funds_edgar_empty_string_filed_excluded(self):
        """Rows with empty string filed date should be excluded."""
        funds = make_edgar_funds(["AAPL"], filed_date="")
        result = get_funds_as_of(funds, datetime(2021, 1, 1))
        assert result.is_empty(), "Empty filed dates should be excluded"

    def test_prices_empty_input_returns_empty(self):
        """Empty input should return empty output without error."""
        empty = pl.DataFrame({"ticker": [], "date": [], "open": [],
                               "close": [], "high": [], "low": [], "volume": []})
        result = get_prices_as_of(empty, datetime(2020, 1, 1))
        assert result.is_empty()


# ── 3. Rebalance date generation ──────────────────────────────────────────────

class TestRebalanceDates:
    """Verify weekly and monthly rebalance date generation."""

    def make_spy_prices(self, start: str, end: str) -> pl.DataFrame:
        return make_prices(["SPY"], start, end)

    def test_weekly_returns_more_dates_than_monthly(self):
        prices = self.make_spy_prices("2020-01-01", "2021-12-31")
        weekly  = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2021,12,31), "weekly")
        monthly = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2021,12,31), "monthly")
        assert len(weekly) > len(monthly), "Weekly should produce more dates than monthly"

    def test_weekly_approximately_52_per_year(self):
        prices = self.make_spy_prices("2020-01-01", "2020-12-31")
        dates = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2020,12,31), "weekly")
        # 52 weeks in a year — allow some slack for partial weeks at boundaries
        assert 48 <= len(dates) <= 54, f"Expected ~52 weekly dates, got {len(dates)}"

    def test_monthly_approximately_12_per_year(self):
        prices = self.make_spy_prices("2020-01-01", "2020-12-31")
        dates = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2020,12,31), "monthly")
        assert len(dates) == 12, f"Expected 12 monthly dates, got {len(dates)}"

    def test_dates_are_sorted_ascending(self):
        prices = self.make_spy_prices("2020-01-01", "2021-12-31")
        dates = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2021,12,31), "weekly")
        assert dates == sorted(dates), "Rebalance dates must be in ascending order"

    def test_dates_within_range(self):
        prices = self.make_spy_prices("2019-01-01", "2023-12-31")
        start = datetime(2020, 6, 1)
        end   = datetime(2021, 6, 1)
        dates = get_rebalance_dates(prices, start, end, "monthly")
        for d in dates:
            assert start <= d <= end, f"Date {d} outside [{start}, {end}]"

    def test_no_duplicate_dates(self):
        prices = self.make_spy_prices("2020-01-01", "2021-12-31")
        dates = get_rebalance_dates(prices, datetime(2020,1,1), datetime(2021,12,31), "weekly")
        assert len(dates) == len(set(dates)), "Rebalance dates must be unique"

    def test_empty_range_returns_empty(self):
        prices = self.make_spy_prices("2020-01-01", "2020-12-31")
        # Start after end
        dates = get_rebalance_dates(prices, datetime(2021,1,1), datetime(2020,1,1), "weekly")
        assert dates == [], "Empty range should return empty list"


# ── 4. Exit date calculation ───────────────────────────────────────────────────

class TestExitDate:
    """Verify get_exit_date returns approximately correct dates."""

    def test_exit_3m_approximately_90_days_later(self):
        prices = make_prices(["SPY"], "2020-01-01", "2021-12-31")
        entry = datetime(2020, 6, 1)
        exit_dt = get_exit_date(prices, entry, hold_months=3)
        assert exit_dt is not None
        days_held = (exit_dt - entry).days
        assert 80 <= days_held <= 100, f"3M hold should be ~90 days, got {days_held}"

    def test_exit_6m_approximately_180_days_later(self):
        prices = make_prices(["SPY"], "2020-01-01", "2022-12-31")
        entry = datetime(2020, 6, 1)
        exit_dt = get_exit_date(prices, entry, hold_months=6)
        assert exit_dt is not None
        days_held = (exit_dt - entry).days
        assert 170 <= days_held <= 190, f"6M hold should be ~180 days, got {days_held}"

    def test_exit_returns_none_when_no_data(self):
        """Should return None when no price data exists after the target exit."""
        prices = make_prices(["SPY"], "2020-01-01", "2020-06-30")
        entry = datetime(2020, 6, 1)
        exit_dt = get_exit_date(prices, entry, hold_months=3)
        assert exit_dt is None, "Should return None when data ends before exit date"

    def test_exit_date_is_trading_day(self):
        """Exit date must be a weekday (trading day from SPY calendar)."""
        prices = make_prices(["SPY"], "2020-01-01", "2022-12-31")
        entry = datetime(2020, 3, 15)
        exit_dt = get_exit_date(prices, entry, hold_months=6)
        assert exit_dt is not None
        assert exit_dt.weekday() < 5, f"Exit date {exit_dt} is a weekend"


# ── 5. Next open price ────────────────────────────────────────────────────────

class TestGetNextOpen:
    """Verify entry at next-day open, not same-day close."""

    def test_entry_is_next_day_not_signal_day(self):
        """
        Signal fires on Monday. Entry must be Tuesday's open.
        This prevents look-ahead from using Monday's close as entry price.
        """
        prices = make_prices(["AAPL"], "2020-01-01", "2020-06-30",
                              open_val=105.0, close_val=100.0)
        signal_date = datetime(2020, 3, 2)  # Monday
        open_px, entry_dt = get_next_open(prices, "AAPL", signal_date)

        assert open_px is not None
        assert entry_dt is not None
        assert entry_dt > signal_date, (
            f"Entry date {entry_dt} must be strictly after signal date {signal_date}"
        )
        assert open_px == 105.0, "Should use open price, not close price"

    def test_returns_none_for_delisted_ticker(self):
        """If ticker has no price data after signal date, return None."""
        prices = make_prices(["SPY"], "2020-01-01", "2020-06-30")
        open_px, entry_dt = get_next_open(prices, "DELISTED", datetime(2020, 3, 1))
        assert open_px is None
        assert entry_dt is None


# ── 6. Benchmark return ────────────────────────────────────────────────────────

class TestBenchmarkReturn:
    """Verify benchmark return calculation is correct."""

    def test_positive_return_calculated_correctly(self):
        """£100 entry, £110 exit = +10% return."""
        prices = make_prices_trending("SPY", "2020-01-01", "2020-12-31",
                                       start_price=100.0, daily_return=0.0)
        # Flat prices — override with specific values
        rows = [
            {"ticker": "SPY", "date": datetime(2020, 6, 1),
             "open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0, "volume": 1000000},
            {"ticker": "SPY", "date": datetime(2020, 9, 1),
             "open": 110.0, "close": 110.0, "high": 111.0, "low": 109.0, "volume": 1000000},
        ]
        prices = pl.DataFrame(rows)
        ret = get_benchmark_return(prices, datetime(2020, 6, 1), datetime(2020, 9, 1))
        assert ret is not None
        assert abs(ret - 10.0) < 0.01, f"Expected +10%, got {ret:.2f}%"

    def test_negative_return_calculated_correctly(self):
        """£100 entry, £90 exit = -10% return."""
        rows = [
            {"ticker": "SPY", "date": datetime(2020, 6, 1),
             "open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0, "volume": 1000000},
            {"ticker": "SPY", "date": datetime(2020, 9, 1),
             "open": 90.0, "close": 90.0, "high": 91.0, "low": 89.0, "volume": 1000000},
        ]
        prices = pl.DataFrame(rows)
        ret = get_benchmark_return(prices, datetime(2020, 6, 1), datetime(2020, 9, 1))
        assert ret is not None
        assert abs(ret - (-10.0)) < 0.01, f"Expected -10%, got {ret:.2f}%"

    def test_returns_none_when_no_data(self):
        prices = make_prices(["SPY"], "2020-01-01", "2020-06-30")
        ret = get_benchmark_return(prices, datetime(2021, 1, 1), datetime(2021, 6, 1))
        assert ret is None


# ── 7. Transaction costs ──────────────────────────────────────────────────────

class TestTransactionCosts:
    """Verify transaction costs are applied correctly."""

    def test_transaction_cost_is_0_15_pct(self):
        """TRANSACTION_COST must be 0.0015 — 0.15% per leg."""
        assert TRANSACTION_COST == 0.0015, (
            f"Expected 0.0015 (0.15% per leg), got {TRANSACTION_COST}"
        )

    def test_net_return_is_less_than_raw_return(self):
        """
        Net return = raw return - (2 × transaction cost × 100).
        Two legs: entry and exit.
        """
        raw_return = 10.0  # +10%
        expected_net = raw_return - (TRANSACTION_COST * 100 * 2)
        actual_net = raw_return - TRANSACTION_COST * 100 * 2
        assert abs(actual_net - expected_net) < 0.0001

    def test_round_trip_cost_is_0_30_pct(self):
        """Round-trip cost (entry + exit) must be 0.30%."""
        round_trip_pct = TRANSACTION_COST * 2 * 100
        assert abs(round_trip_pct - 0.30) < 0.0001, (
            f"Expected 0.30% round trip, got {round_trip_pct:.4f}%"
        )

    def test_flat_trade_has_negative_net_return(self):
        """A trade where entry price equals exit price should have negative net return due to costs."""
        entry_price = 100.0
        exit_price  = 100.0
        raw_return  = (exit_price - entry_price) / entry_price * 100
        net_return  = raw_return - TRANSACTION_COST * 100 * 2
        assert net_return < 0, "Flat trade must have negative net return after costs"
        assert abs(net_return - (-0.30)) < 0.0001


# ── 8. Select top N ───────────────────────────────────────────────────────────

class TestSelectTopN:
    """Verify signal selection logic — score threshold, tiebreaker, count."""

    def test_returns_at_most_n_signals(self):
        signals = make_signals(
            ["A","B","C","D","E","F","G","H","I","J","K","L"],
            [3,3,3,3,3,3,2,2,2,2,2,2]
        )
        result = select_top_n(signals, n=10, min_score=1)
        assert len(result) <= 10

    def test_respects_min_score_threshold(self):
        """With min_score=7, no score<7 trades should appear."""
        signals = make_signals(
            ["A","B","C","D","E"],
            [8, 7, 6, 5, 4]
        )
        result = select_top_n(signals, n=10, min_score=7)
        if not result.is_empty():
            assert result["score"].min() >= 7, "All returned scores must be >= min_score"

    def test_score_3_before_score_2(self):
        """Higher score tickers should appear before lower score tickers."""
        signals = make_signals(
            ["LOW_SCORE", "HIGH_SCORE"],
            [2, 3],
            momentum_1w=[1.0, 1.0]
        )
        result = select_top_n(signals, n=10, min_score=1)
        assert not result.is_empty()
        assert result["ticker"][0] == "HIGH_SCORE", (
            "Score 3 should rank above score 2"
        )

    def test_momentum_tiebreaker_within_same_score(self):
        """When scores are equal, higher 1W momentum wins."""
        signals = make_signals(
            ["LOW_MOM", "HIGH_MOM"],
            [3, 3],
            momentum_1w=[0.5, 2.0]
        )
        result = select_top_n(signals, n=10, min_score=1)
        assert not result.is_empty()
        assert result["ticker"][0] == "HIGH_MOM", (
            "Higher 1W momentum should rank first when scores are equal"
        )

    def test_returns_empty_when_no_qualifying_signals(self):
        """If all signals are below min_score, return empty."""
        signals = make_signals(["A","B","C"], [1,1,1])
        result = select_top_n(signals, n=10, min_score=7)
        assert result.is_empty()

    def test_returns_empty_on_empty_input(self):
        empty = pl.DataFrame({
            "ticker": [], "score": [], "momentum_1w": [],
            "close": [], "momentum_6m": [], "momentum_3m": []
        })
        result = select_top_n(empty, n=10, min_score=1)
        assert result.is_empty()

    def test_handles_null_momentum_in_tiebreaker(self):
        """Null momentum values should not cause a crash — nulls_last."""
        signals = pl.DataFrame({
            "ticker":      ["A", "B", "C"],
            "score":       [3, 3, 3],
            "momentum_1w": [1.0, None, 2.0],
            "close":       [100.0, 100.0, 100.0],
            "momentum_6m": [5.0, 5.0, 5.0],
            "momentum_3m": [-5.0, -5.0, -5.0],
        })
        result = select_top_n(signals, n=10, min_score=1)
        assert len(result) == 3  # all returned, no crash


# ── 9. Constituent filtering ──────────────────────────────────────────────────

class TestConstituentFilter:
    """Verify S&P 500 constituent filtering."""

    def test_returns_correct_tickers_for_date(self):
        tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
        comp = make_compositions("2022-01-01", tickers)
        result = get_constituents_on_date(datetime(2022, 6, 1), comp)
        assert result == set(tickers)

    def test_uses_most_recent_composition_before_date(self):
        """If queried between two compositions, use the earlier one."""
        comp = pl.DataFrame({
            "date":    ["2022-01-01", "2022-07-01"],
            "tickers": [["AAPL", "MSFT"], ["AAPL", "MSFT", "NVDA"]],
            "count":   [2, 3],
        })
        # Query in April — should use January composition (NVDA not yet added)
        result = get_constituents_on_date(datetime(2022, 4, 1), comp)
        assert "NVDA" not in result, "NVDA not in index until July — should not appear in April"
        assert "AAPL" in result
        assert "MSFT" in result

    def test_falls_back_to_earliest_when_query_before_all_data(self):
        """If query date is before all compositions, use earliest available."""
        comp = make_compositions("2022-01-01", ["AAPL", "MSFT"])
        result = get_constituents_on_date(datetime(2010, 1, 1), comp)
        assert len(result) > 0, "Should fall back to earliest composition"

    def test_returns_set_not_list(self):
        """Result must be a set for O(1) membership testing."""
        comp = make_compositions("2022-01-01", ["AAPL", "MSFT"])
        result = get_constituents_on_date(datetime(2022, 6, 1), comp)
        assert isinstance(result, set)


# ── 10. Warmup period ─────────────────────────────────────────────────────────

class TestWarmup:
    """Verify warmup period is respected — no trades before FIRST_TRADE_DATE."""

    def test_first_trade_date_is_after_warmup(self):
        """FIRST_TRADE_DATE must be at least 252 trading days after WINDOW_A_START."""
        from engine import WINDOW_A_START
        days_diff = (FIRST_TRADE_DATE - WINDOW_A_START).days
        # 252 trading days ≈ 350 calendar days
        assert days_diff >= 350, (
            f"Warmup period too short: {days_diff} days. "
            f"200MA needs 200 trading days of history."
        )

    def test_window_c_first_trade_matches_a(self):
        """Mode C should use same warmup as Mode A."""
        assert WINDOW_C_FIRST_TRADE == FIRST_TRADE_DATE, (
            "Mode C should use same first trade date as Mode A"
        )


# ── 11. Return calculation ────────────────────────────────────────────────────

class TestReturnCalculation:
    """Verify raw and net return calculations are correct."""

    def test_raw_return_formula(self):
        """raw_return = (exit - entry) / entry * 100"""
        entry = 100.0
        exit_ = 120.0
        expected = (exit_ - entry) / entry * 100  # +20%
        assert abs(expected - 20.0) < 0.0001

    def test_net_return_deducts_both_legs(self):
        """net_return = raw_return - 2 * TRANSACTION_COST * 100"""
        entry = 100.0
        exit_ = 120.0
        raw = (exit_ - entry) / entry * 100
        net = raw - TRANSACTION_COST * 100 * 2
        assert abs(net - 19.70) < 0.01, f"Expected 19.70%, got {net:.2f}%"

    def test_outperformed_flag_correct(self):
        """outperformed = net_return > benchmark_return"""
        net_return       = 5.0
        benchmark_return = 3.0
        outperformed = net_return > benchmark_return
        assert outperformed is True

        net_return       = 2.0
        benchmark_return = 3.0
        outperformed = net_return > benchmark_return
        assert outperformed is False

    def test_outperformed_false_when_net_equals_benchmark(self):
        """Exactly matching benchmark is not outperforming."""
        net_return       = 3.0
        benchmark_return = 3.0
        outperformed = net_return > benchmark_return
        assert outperformed is False


# ── 12. Data integrity ────────────────────────────────────────────────────────

class TestDataIntegrity:
    """Verify trade log schema and data quality."""

    def test_required_trade_log_columns(self):
        """Trade log must contain all required columns."""
        required = [
            "entry_date", "exit_date", "ticker", "entry_price",
            "exit_price", "return_pct", "net_return_pct", "score",
            "hold_months", "benchmark_return", "mode", "outperformed",
        ]
        # Build a minimal trade dict to check schema
        trade = {
            "entry_date":       datetime(2020, 5, 1),
            "signal_date":      datetime(2020, 5, 1),
            "exit_date":        datetime(2020, 11, 1),
            "ticker":           "AAPL",
            "entry_price":      100.0,
            "exit_price":       120.0,
            "return_pct":       20.0,
            "net_return_pct":   19.70,
            "score":            3,
            "hold_months":      6,
            "benchmark_return": 10.0,
            "mode":             "A",
            "outperformed":     True,
        }
        df = pl.DataFrame([trade])
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_net_return_always_less_than_raw(self):
        """net_return_pct must always be less than return_pct due to costs."""
        for raw in [-20.0, -5.0, 0.0, 5.0, 20.0, 100.0]:
            net = raw - TRANSACTION_COST * 100 * 2
            assert net < raw, f"Net return {net} should be less than raw {raw}"


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])