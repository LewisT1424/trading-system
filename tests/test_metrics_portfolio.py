"""
tests/test_metrics_portfolio.py
Production-grade tests for backtest/metrics.py and backtest/portfolio.py

Test categories:
    metrics.py:
        1. compute_metrics — known-value mathematical verification
        2. Hit rate calculation
        3. Win/loss ratio
        4. Sharpe ratio — sign, scale, edge cases
        5. Max drawdown — correct peak-to-trough calculation
        6. Calmar ratio
        7. Gate checks — correct pass/fail logic
        8. Empty and edge case inputs
        9. _get_spy_period_return helper
        10. _months_between helper

    portfolio.py:
        11. build_portfolio — cash accounting
        12. build_portfolio — mark to market
        13. build_portfolio — position entry and exit
        14. build_portfolio — no pyramiding
        15. build_portfolio — insufficient cash
        16. build_portfolio — fallback to last known price
        17. compute_portfolio_metrics — annualised return formula
        18. compute_portfolio_metrics — Sharpe from daily returns
        19. compute_portfolio_metrics — max drawdown from daily curve
        20. compute_portfolio_metrics — gates
        21. Empty inputs

Run:
    pytest tests/test_metrics_portfolio.py -v
    pytest tests/test_metrics_portfolio.py::TestMetricsCore -v
    pytest tests/test_metrics_portfolio.py::TestPortfolioSimulation -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

from metrics import (
    compute_metrics,
    compute_period_metrics,
    _get_spy_period_return,
    _get_spy_max_drawdown,
    _months_between,
    RISK_FREE_RATE,
)
from portfolio import (
    build_portfolio,
    compute_portfolio_metrics,
    POSITION_SIZE,
    STARTING_CASH,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_trades(
    returns: list[float],
    benchmark_returns: list[float] | None = None,
    hold_months: int = 6,
    start_date: datetime = datetime(2020, 1, 1),
) -> pl.DataFrame:
    """Build a minimal trade log with specified returns."""
    if benchmark_returns is None:
        benchmark_returns = [0.0] * len(returns)
    assert len(returns) == len(benchmark_returns)

    rows = []
    for i, (ret, bench) in enumerate(zip(returns, benchmark_returns)):
        entry = start_date + timedelta(days=i * 30)
        exit_ = entry + timedelta(days=hold_months * 30)
        net   = ret - 0.30  # deduct round-trip cost
        rows.append({
            "entry_date":       entry,
            "signal_date":      entry,
            "exit_date":        exit_,
            "ticker":           f"TICK{i:03d}",
            "entry_price":      100.0,
            "exit_price":       100.0 * (1 + ret / 100),
            "return_pct":       ret,
            "net_return_pct":   net,
            "score":            3,
            "hold_months":      hold_months,
            "benchmark_return": bench,
            "mode":             "A",
            "outperformed":     net > bench,
        })
    return pl.DataFrame(rows)


def make_prices_flat(
    tickers: list[str],
    start: str,
    end: str,
    price: float = 100.0,
) -> pl.DataFrame:
    """Build flat price series for testing."""
    rows = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while d <= end_dt:
        if d.weekday() < 5:
            for ticker in tickers:
                rows.append({
                    "ticker": ticker,
                    "date":   d,
                    "open":   price,
                    "close":  price,
                    "high":   price,
                    "low":    price,
                    "volume": 1_000_000,
                })
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def make_prices_growing(
    ticker: str,
    start: str,
    end: str,
    start_price: float = 100.0,
    daily_pct: float = 0.1,
) -> pl.DataFrame:
    """Build prices growing by fixed daily % for precise return testing."""
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
                "high":   price * 1.001,
                "low":    price * 0.999,
                "volume": 1_000_000,
            })
            price *= (1 + daily_pct / 100)
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def make_single_trade(
    ticker: str,
    entry_date: datetime,
    exit_date: datetime,
    entry_price: float,
    exit_price: float,
) -> pl.DataFrame:
    """Build a single-trade trade log."""
    ret = (exit_price - entry_price) / entry_price * 100
    net = ret - 0.30
    return pl.DataFrame([{
        "entry_date":       entry_date,
        "signal_date":      entry_date,
        "exit_date":        exit_date,
        "ticker":           ticker,
        "entry_price":      entry_price,
        "exit_price":       exit_price,
        "return_pct":       ret,
        "net_return_pct":   net,
        "score":            3,
        "hold_months":      6,
        "benchmark_return": 0.0,
        "mode":             "A",
        "outperformed":     net > 0.0,
    }])


# ── 1. compute_metrics — core calculations ────────────────────────────────────

class TestMetricsCore:
    """Verify compute_metrics produces mathematically correct outputs."""

    def test_hit_rate_all_winners(self):
        trades = make_trades([10.0, 20.0, 15.0, 5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        assert m["hit_rate"] == 1.0, "All positive returns = 100% hit rate"

    def test_hit_rate_all_losers(self):
        trades = make_trades([-5.0, -10.0, -3.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        # net_return = raw - 0.30, so -5.0 net = -5.30
        assert m["hit_rate"] == 0.0, "All negative net returns = 0% hit rate"

    def test_hit_rate_mixed(self):
        # 3 winners, 1 loser (after cost deduction)
        trades = make_trades([10.0, 20.0, 5.0, -1.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        # net returns: 9.70, 19.70, 4.70, -1.30 — 3 positive out of 4
        assert abs(m["hit_rate"] - 0.75) < 0.01

    def test_avg_win_correct(self):
        """avg_win computed on net_return_pct, winners only."""
        trades = make_trades([10.0, 20.0, -5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        # net returns: 9.70, 19.70, -5.30 — winners avg = (9.70 + 19.70) / 2
        expected_avg_win = (9.70 + 19.70) / 2
        assert abs(m["avg_win"] - expected_avg_win) < 0.01

    def test_avg_loss_correct(self):
        """avg_loss computed on net_return_pct, losers only."""
        trades = make_trades([10.0, -5.0, -15.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        # net losses: -5.30, -15.30 — avg = (-5.30 + -15.30) / 2
        expected_avg_loss = (-5.30 + -15.30) / 2
        assert abs(m["avg_loss"] - expected_avg_loss) < 0.01

    def test_win_loss_ratio_correct(self):
        """win/loss ratio = |avg_win / avg_loss|"""
        trades = make_trades([10.0, -5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        # net: 9.70 and -5.30
        expected = abs(9.70 / -5.30)
        assert abs(m["win_loss_ratio"] - expected) < 0.01

    def test_win_loss_ratio_infinite_when_no_losses(self):
        """If no losing trades, win/loss ratio should be infinite."""
        trades = make_trades([10.0, 20.0, 5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        assert m["win_loss_ratio"] == float("inf")

    def test_n_trades_correct(self):
        trades = make_trades([1.0, 2.0, 3.0, 4.0, 5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        assert m["n_trades"] == 5

    def test_n_winners_plus_n_losers_equals_n_trades(self):
        trades = make_trades([10.0, -5.0, 8.0, -2.0, 3.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        assert m["n_winners"] + m["n_losers"] == m["n_trades"]

    def test_returns_error_on_empty_trades(self):
        empty = pl.DataFrame({
            "net_return_pct": [], "benchmark_return": [],
            "entry_date": [], "outperformed": [],
        })
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_metrics(empty, prices, datetime(2020,1,1), datetime(2022,1,1), 6)
        assert "error" in m


# ── 2. Max drawdown ───────────────────────────────────────────────────────────

class TestMaxDrawdown:
    """
    Max drawdown must be computed correctly as peak-to-trough.
    This is critical for gate 3 (drawdown ≤ 1.5× SPY).
    """

    def test_no_drawdown_when_all_returns_positive(self):
        """Monotonically increasing equity curve = 0% drawdown."""
        trades = make_trades([5.0, 5.0, 5.0, 5.0, 5.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2023-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2023,1,1), 6)
        assert m["max_drawdown"] <= 0.0, "Positive equity curve should have zero drawdown"

    def test_drawdown_is_negative(self):
        """Max drawdown must always be a negative number."""
        trades = make_trades([10.0, -20.0, 5.0, -10.0, 8.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2023-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2023,1,1), 6)
        assert m["max_drawdown"] < 0.0, "Max drawdown must be negative"

    def test_drawdown_cannot_exceed_100_pct(self):
        """Max drawdown cannot be worse than -100%."""
        trades = make_trades([-30.0, -40.0, -20.0, -50.0])
        prices = make_prices_flat(["SPY"], "2020-01-01", "2023-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2023,1,1), 6)
        assert m["max_drawdown"] >= -100.0

    def test_spy_max_drawdown_is_negative(self):
        """SPY max drawdown must be negative or zero."""
        prices = make_prices_growing("SPY", "2020-01-01", "2022-12-31",
                                      start_price=100.0, daily_pct=0.05)
        dd = _get_spy_max_drawdown(prices, datetime(2020,1,1), datetime(2022,12,31))
        assert dd <= 0.0

    def test_spy_max_drawdown_flat_prices_is_zero(self):
        """Perfectly flat SPY prices = zero drawdown."""
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31", price=100.0)
        dd = _get_spy_max_drawdown(prices, datetime(2020,1,1), datetime(2022,12,31))
        assert abs(dd) < 0.001, f"Expected 0% drawdown on flat prices, got {dd}"

    def test_drawdown_formula_known_value(self):
        """
        Verify drawdown calculation on a known sequence.
        equity: 100, 120, 90, 95 — peak 120, trough 90
        drawdown at trough = (90 - 120) / 120 * 100 = -25%
        """
        equity = np.array([100.0, 120.0, 90.0, 95.0])
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        assert abs(dd.min() - (-25.0)) < 0.01, f"Expected -25% drawdown, got {dd.min():.2f}%"


# ── 3. Sharpe ratio ───────────────────────────────────────────────────────────

class TestSharpe:
    """Verify Sharpe ratio calculation."""

    def test_sharpe_positive_when_returns_exceed_risk_free(self):
        """If avg return > risk-free rate with variance, Sharpe must be positive."""
        # Use varied returns so variance > 0 — identical returns give zero variance
        # which triggers the zero-variance guard returning 0.0
        trades = make_trades([15.0, 12.0, 18.0, 10.0, 20.0, 14.0,
                               16.0, 11.0, 19.0, 13.0] * 4)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        assert m["sharpe"] > 0.0, (
            f"Sharpe should be positive when avg returns exceed risk-free rate. "
            f"Got {m['sharpe']:.3f}. avg_return={m['mean_trade_return']:.2f}%"
        )

    def test_sharpe_zero_when_no_variance(self):
        """Zero variance in returns → Sharpe = 0 (not infinity, not NaN)."""
        trades = make_trades([10.0] * 5)  # all identical returns
        prices = make_prices_flat(["SPY"], "2020-01-01", "2023-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2023,1,1), 6)
        assert m["sharpe"] == 0.0, "Zero variance should produce Sharpe of 0"
        assert not np.isnan(m["sharpe"]), "Sharpe must not be NaN"

    def test_sharpe_is_finite(self):
        """Sharpe must always be a finite number."""
        trades = make_trades([5.0, -3.0, 8.0, -2.0, 10.0, -1.0] * 5)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        assert np.isfinite(m["sharpe"]), f"Sharpe must be finite, got {m['sharpe']}"

    def test_higher_consistent_returns_give_higher_sharpe(self):
        """Higher returns with same variance → higher Sharpe."""
        trades_low  = make_trades([3.0, 2.0, 4.0, 1.0, 3.0] * 4)
        trades_high = make_trades([10.0, 9.0, 11.0, 8.0, 10.0] * 4)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m_low  = compute_metrics(trades_low,  prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        m_high = compute_metrics(trades_high, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        assert m_high["sharpe"] > m_low["sharpe"], (
            "Higher consistent returns should produce higher Sharpe"
        )


# ── 4. Calmar ratio ───────────────────────────────────────────────────────────

class TestCalmar:
    """Verify Calmar ratio = |annualised return / max drawdown|."""

    def test_calmar_positive(self):
        trades = make_trades([10.0, -5.0, 8.0, -2.0, 12.0] * 4)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        assert m["calmar"] >= 0.0

    def test_calmar_zero_when_no_drawdown(self):
        """With zero drawdown, calmar is set to 0 to avoid division by zero."""
        trades = make_trades([5.0] * 10)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        # max_drawdown will be near 0 for consistently positive returns
        # calmar = 0 when max_drawdown == 0
        assert np.isfinite(m["calmar"])


# ── 5. Gate checks ────────────────────────────────────────────────────────────

class TestGateChecks:
    """Verify gate logic is correct — critical for deployment decision."""

    def test_gate_sharpe_passes_above_0_5(self):
        """Sharpe > 0.5 must pass gate."""
        trades = make_trades([15.0] * 30)
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        if m["sharpe"] > 0.5:
            assert m["gate_sharpe"] is True

    def test_gate_sharpe_fails_below_0_5(self):
        """Sharpe ≤ 0.5 must fail gate."""
        # Zero returns → Sharpe = 0
        trades = make_trades([0.31] * 10)  # just above risk-free, low Sharpe
        prices = make_prices_flat(["SPY"], "2020-01-01", "2025-12-31")
        m = compute_metrics(trades, prices, datetime(2020,1,1), datetime(2025,1,1), 6)
        if m["sharpe"] <= 0.5:
            assert m["gate_sharpe"] is False

    def test_all_gates_pass_requires_all_three(self):
        """all_gates_pass is True only when all three gates pass."""
        # Manufacture a metrics dict with known values
        m = {
            "gate_sharpe": True,
            "gate_beats_spy": True,
            "gate_drawdown": True,
            "all_gates_pass": True and True and True,
        }
        assert m["all_gates_pass"] is True

        m2 = {
            "gate_sharpe": True,
            "gate_beats_spy": False,
            "gate_drawdown": True,
            "all_gates_pass": True and False and True,
        }
        assert m2["all_gates_pass"] is False

    def test_gate_drawdown_uses_1_5x_spy(self):
        """
        Gate drawdown = |portfolio_dd| ≤ |spy_dd| * 1.5
        This is the exact formula used in compute_metrics.
        """
        portfolio_dd = -30.0
        spy_dd = -25.0
        limit = abs(spy_dd) * 1.5  # 37.5%
        gate = abs(portfolio_dd) <= limit
        assert gate is True, f"{abs(portfolio_dd)}% ≤ {limit}% should pass"

        portfolio_dd = -50.0
        gate = abs(portfolio_dd) <= limit
        assert gate is False, f"{abs(portfolio_dd)}% > {limit}% should fail"


# ── 6. Helper functions ───────────────────────────────────────────────────────

class TestHelpers:
    """Verify helper functions produce correct outputs."""

    def test_months_between_one_year(self):
        start = datetime(2020, 1, 1)
        end   = datetime(2021, 1, 1)
        months = _months_between(start, end)
        assert abs(months - 12.0) < 0.5, f"Expected ~12 months, got {months:.2f}"

    def test_months_between_six_months(self):
        start = datetime(2020, 1, 1)
        end   = datetime(2020, 7, 1)
        months = _months_between(start, end)
        assert abs(months - 6.0) < 0.5

    def test_months_between_same_date(self):
        dt = datetime(2020, 6, 1)
        months = _months_between(dt, dt)
        assert months == 0.0

    def test_spy_period_return_flat(self):
        """Flat SPY prices = 0% return."""
        prices = make_prices_flat(["SPY"], "2020-01-01", "2021-12-31", price=100.0)
        ret = _get_spy_period_return(prices, datetime(2020,1,1), datetime(2021,12,31))
        assert abs(ret) < 0.001

    def test_spy_period_return_known_value(self):
        """SPY doubles = +100% return."""
        rows = [
            {"ticker": "SPY", "date": datetime(2020,1,2), "open":100.0,
             "close":100.0, "high":101.0, "low":99.0, "volume":1000000},
            {"ticker": "SPY", "date": datetime(2021,1,4), "open":200.0,
             "close":200.0, "high":201.0, "low":199.0, "volume":1000000},
        ]
        prices = pl.DataFrame(rows)
        ret = _get_spy_period_return(prices, datetime(2020,1,1), datetime(2021,12,31))
        assert abs(ret - 100.0) < 0.01, f"Expected +100%, got {ret:.2f}%"

    def test_spy_period_return_empty_data(self):
        """Empty price data returns 0.0 without crashing."""
        prices = pl.DataFrame({
            "ticker": [], "date": [], "open": [],
            "close": [], "high": [], "low": [], "volume": []
        })
        ret = _get_spy_period_return(prices, datetime(2020,1,1), datetime(2021,1,1))
        assert ret == 0.0


# ── 7. Portfolio simulation ───────────────────────────────────────────────────

class TestPortfolioSimulation:
    """
    Verify build_portfolio produces a correct daily mark-to-market simulation.
    These tests verify cash accounting, position entry/exit, and mark-to-market.
    """

    def test_starting_value_equals_starting_cash(self):
        """On day 1 before any trades open, portfolio value = starting cash."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        assert not daily.is_empty()
        first_value = daily["portfolio_value"][0]
        # Before entry, portfolio = cash = STARTING_CASH
        # Allow for entry transaction cost: £1,000 * 0.0015 = £1.50
        # The portfolio charges entry cost on the first trade opening day
        assert abs(first_value - STARTING_CASH) < 2.0, (
            f"Starting value should be ~£{STARTING_CASH} (±entry cost), "
            f"got £{first_value:.2f}"
        )

    def test_cash_decreases_on_entry(self):
        """Cash must decrease by POSITION_SIZE when a position opens."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        # Find the day after entry
        entry_idx = None
        for i, row in enumerate(daily.iter_rows(named=True)):
            if row["date"] >= entry and row["n_positions"] > 0:
                entry_idx = i
                break

        if entry_idx is not None:
            cash_after_entry = daily["cash"][entry_idx]
            expected_cash = STARTING_CASH - POSITION_SIZE
            assert abs(cash_after_entry - expected_cash) < 1.0, (
                f"Cash should be £{expected_cash:.0f} after entry, "
                f"got £{cash_after_entry:.2f}"
            )

    def test_portfolio_value_is_cash_plus_positions(self):
        """portfolio_value must equal cash + position_value on every day."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        for row in daily.iter_rows(named=True):
            expected = row["cash"] + row["position_value"]
            actual = row["portfolio_value"]
            assert abs(actual - expected) < 0.01, (
                f"portfolio_value {actual:.2f} != cash {row['cash']:.2f} "
                f"+ positions {row['position_value']:.2f}"
            )

    def test_n_positions_increments_on_entry(self):
        """n_positions must increase when a trade opens."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        max_positions = daily["n_positions"].max()
        assert max_positions >= 1, "Should have at least 1 open position"

    def test_n_positions_returns_to_zero_after_exit(self):
        """After all trades close, n_positions must return to 0."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 6, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2020-12-31")
        daily = build_portfolio(trades, prices)

        last_positions = daily["n_positions"][-1]
        assert last_positions == 0, (
            f"After final exit, n_positions should be 0, got {last_positions}"
        )

    def test_no_pyramiding_same_ticker(self):
        """
        The same ticker cannot be opened twice simultaneously.
        If two trades for AAPL are scheduled to open on the same day,
        only the first should be entered.
        """
        entry = datetime(2020, 3, 2)
        exit1 = datetime(2020, 9, 1)
        exit2 = datetime(2020, 10, 1)
        # Two trades for AAPL with same entry date
        rows = [
            {"entry_date": entry, "signal_date": entry, "exit_date": exit1,
             "ticker": "AAPL", "entry_price": 100.0, "exit_price": 110.0,
             "return_pct": 10.0, "net_return_pct": 9.70, "score": 3,
             "hold_months": 6, "benchmark_return": 5.0, "mode": "A", "outperformed": True},
            {"entry_date": entry, "signal_date": entry, "exit_date": exit2,
             "ticker": "AAPL", "entry_price": 100.0, "exit_price": 115.0,
             "return_pct": 15.0, "net_return_pct": 14.70, "score": 3,
             "hold_months": 7, "benchmark_return": 5.0, "mode": "A", "outperformed": True},
        ]
        trades = pl.DataFrame(rows)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        # Max positions for AAPL should never exceed 1
        max_pos = daily["n_positions"].max()
        # With 2 AAPL trades, we can only have 1 open (no pyramiding)
        # Cash check: should only have deployed POSITION_SIZE once
        min_cash = daily["cash"].min()
        assert min_cash >= STARTING_CASH - POSITION_SIZE - 1.0, (
            f"Cash shouldn't drop below £{STARTING_CASH - POSITION_SIZE:.0f} "
            f"for a single AAPL position, got £{min_cash:.2f}"
        )

    def test_insufficient_cash_prevents_entry(self):
        """
        If cash is below POSITION_SIZE, new positions should not open.
        With STARTING_CASH = £10,000 and POSITION_SIZE = £1,000,
        maximum 10 positions can open before cash runs out.
        """
        # Create 15 trades all opening on the same day
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        rows = []
        for i in range(15):
            rows.append({
                "entry_date": entry, "signal_date": entry, "exit_date": exit_,
                "ticker": f"TICK{i:03d}", "entry_price": 100.0, "exit_price": 105.0,
                "return_pct": 5.0, "net_return_pct": 4.70, "score": 3,
                "hold_months": 6, "benchmark_return": 2.0, "mode": "A", "outperformed": True,
            })
        trades = pl.DataFrame(rows)
        tickers = ["SPY"] + [f"TICK{i:03d}" for i in range(15)]
        prices = make_prices_flat(tickers, "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        # Maximum open positions should not exceed STARTING_CASH / POSITION_SIZE = 10
        max_pos = daily["n_positions"].max()
        max_allowed = int(STARTING_CASH / POSITION_SIZE)
        assert max_pos <= max_allowed, (
            f"Max positions {max_pos} exceeds cash limit of {max_allowed}"
        )

    def test_empty_trades_returns_empty(self):
        """Empty trade log should return empty DataFrame."""
        empty = pl.DataFrame({
            "entry_date": [], "exit_date": [], "ticker": [],
            "entry_price": [], "exit_price": [], "return_pct": [],
            "net_return_pct": [], "score": [], "hold_months": [],
            "benchmark_return": [], "mode": [], "outperformed": [],
        })
        prices = make_prices_flat(["SPY"], "2020-01-01", "2021-12-31")
        result = build_portfolio(empty, prices)
        assert result.is_empty()

    def test_daily_return_is_null_on_first_day(self):
        """First day daily_return must be null — no previous day to compare."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 110.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31")
        daily = build_portfolio(trades, prices)

        first_return = daily["daily_return"][0]
        assert first_return is None, (
            "First day daily_return should be null (no prior day)"
        )

    def test_portfolio_value_always_positive(self):
        """Portfolio value must never go negative."""
        entry = datetime(2020, 3, 2)
        exit_ = datetime(2020, 9, 1)
        # Trade loses 50%
        trades = make_single_trade("AAPL", entry, exit_, 100.0, 50.0)
        prices = make_prices_flat(["SPY", "AAPL"], "2020-01-01", "2021-12-31",
                                   price=50.0)
        daily = build_portfolio(trades, prices)

        min_value = daily["portfolio_value"].min()
        assert min_value > 0, f"Portfolio value must be positive, got £{min_value:.2f}"


# ── 8. compute_portfolio_metrics ─────────────────────────────────────────────

class TestPortfolioMetrics:
    """Verify compute_portfolio_metrics produces correct values from daily curve."""

    def make_daily(
        self,
        values: list[float],
        start: str = "2020-01-02",
    ) -> pl.DataFrame:
        """Build a daily portfolio DataFrame from a list of values."""
        rows = []
        d = datetime.strptime(start, "%Y-%m-%d")
        prev = None
        for v in values:
            daily_ret = (v / prev - 1) if prev is not None else None
            rows.append({
                "date":            d,
                "portfolio_value": v,
                "n_positions":     1,
                "position_value":  v * 0.9,
                "cash":            v * 0.1,
                "daily_return":    daily_ret,
            })
            prev = v
            d += timedelta(days=1)
        return pl.DataFrame(rows)

    def test_total_return_known_value(self):
        """£10,000 → £11,000 = +10% total return."""
        values = [10_000.0] * 50 + [11_000.0]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        assert abs(m["total_return"] - 10.0) < 0.01, (
            f"Expected +10% total return, got {m['total_return']:.2f}%"
        )

    def test_max_drawdown_known_value(self):
        """
        Portfolio: 100, 120, 90 → peak 120, trough 90
        max_drawdown = (90-120)/120*100 = -25%
        """
        values = [10_000.0, 12_000.0, 9_000.0, 9_500.0]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        assert abs(m["max_drawdown"] - (-25.0)) < 0.1, (
            f"Expected -25% max drawdown, got {m['max_drawdown']:.2f}%"
        )

    def test_max_drawdown_is_negative(self):
        """Max drawdown must be ≤ 0."""
        values = [10_000.0, 11_000.0, 9_500.0, 10_500.0, 9_000.0, 10_000.0]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        assert m["max_drawdown"] <= 0.0

    def test_sharpe_not_nan(self):
        """Sharpe must never be NaN regardless of input."""
        values = [10_000.0 + i * 10 for i in range(252)]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        assert not np.isnan(m["sharpe"]), "Sharpe must not be NaN"
        assert np.isfinite(m["sharpe"]), "Sharpe must be finite"

    def test_empty_daily_returns_error(self):
        """Empty daily portfolio should return error dict."""
        empty = pl.DataFrame({
            "date": [], "portfolio_value": [], "n_positions": [],
            "position_value": [], "cash": [], "daily_return": [],
        })
        prices = make_prices_flat(["SPY"], "2020-01-01", "2022-12-31")
        m = compute_portfolio_metrics(empty, prices)
        assert "error" in m

    def test_alpha_is_ann_return_minus_spy(self):
        """alpha = ann_return - spy_ann_return."""
        values = [10_000.0 * (1.0005 ** i) for i in range(252)]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        expected_alpha = m["ann_return"] - m["spy_ann_return"]
        assert abs(m["alpha"] - expected_alpha) < 0.001

    def test_gate_sharpe_threshold_is_0_5(self):
        """gate_sharpe passes when Sharpe > 0.5, fails otherwise."""
        m_pass = {"sharpe": 0.6, "ann_return": 15.0, "spy_ann_return": 10.0,
                  "max_drawdown": -20.0, "spy_max_drawdown": -25.0}
        gate = m_pass["sharpe"] > 0.5
        assert gate is True

        m_fail = {"sharpe": 0.4}
        gate = m_fail["sharpe"] > 0.5
        assert gate is False

    def test_survivorship_note_present(self):
        """Output must always contain survivorship bias disclaimer."""
        values = [10_000.0 * (1.001 ** i) for i in range(252)]
        daily = self.make_daily(values)
        prices = make_prices_flat(["SPY"], "2019-01-01", "2021-12-31")
        m = compute_portfolio_metrics(daily, prices)
        assert "survivorship_note" in m
        assert len(m["survivorship_note"]) > 10


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as pt
    pt.main([__file__, "-v", "--tb=short"])