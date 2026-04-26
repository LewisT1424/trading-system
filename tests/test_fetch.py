"""
tests/test_fetch.py
pytest test suite for setup.py and data/fetch.py

Run:
    pytest tests/ -v
    pytest tests/ -v -m "not slow"   # skip tests that hit the network
"""

import io
import pytest
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

# ── Import modules under test ─────────────────────────────────────────────────

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import fetch as fetch_module
from fetch import (
    _passes_quality,
    _normalise_types,
    _safe_get,
    load_universe,
    MISSING_THRESH,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_price_frame():
    """A clean price DataFrame that should pass all quality checks."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    return pl.DataFrame({
        "ticker": ["AAPL"] * 100,
        "date":   dates.to_list(),
        "open":   [150.0] * 100,
        "high":   [155.0] * 100,
        "low":    [148.0] * 100,
        "close":  [152.0] * 100,
        "volume": [1_000_000] * 100,
    })


@pytest.fixture
def tmp_universe(tmp_path):
    """Write a small universe CSV and return its directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Use 5 tickers — isolated from real universe.csv which may grow over time
    pl.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA", "PLTR", "AMD"]}).write_csv(
        data_dir / "universe.csv"
    )
    return tmp_path


# ── _normalise_types ──────────────────────────────────────────────────────────

class TestNormaliseTypes:

    def test_volume_cast_to_int64(self):
        df = pl.DataFrame({"volume": [1.0, 2.0, 3.0]})
        result = _normalise_types(df)
        assert result["volume"].dtype == pl.Int64

    def test_volume_already_int64_unchanged(self):
        df = pl.DataFrame({"volume": [1, 2, 3]})
        result = _normalise_types(df)
        assert result["volume"].dtype == pl.Int64

    def test_ohlc_cast_to_float64(self):
        df = pl.DataFrame({
            "open":  [100, 101, 102],
            "high":  [105, 106, 107],
            "low":   [98, 99, 100],
            "close": [103, 104, 105],
        })
        result = _normalise_types(df)
        for col in ["open", "high", "low", "close"]:
            assert result[col].dtype == pl.Float64, f"{col} should be Float64"

    def test_missing_columns_handled_gracefully(self):
        df = pl.DataFrame({"ticker": ["AAPL"]})
        result = _normalise_types(df)
        assert "ticker" in result.columns

    def test_values_preserved_after_cast(self):
        df = pl.DataFrame({"volume": [1_000_000.0], "close": [152]})
        result = _normalise_types(df)
        assert result["volume"][0] == 1_000_000
        assert result["close"][0] == pytest.approx(152.0)


# ── _passes_quality ───────────────────────────────────────────────────────────

class TestPassesQuality:

    def test_valid_frame_passes(self, valid_price_frame):
        assert _passes_quality(valid_price_frame, "AAPL") is True

    def test_empty_frame_fails(self):
        assert _passes_quality(pl.DataFrame(), "AAPL") is False

    def test_missing_close_column_fails(self):
        df = pl.DataFrame({"ticker": ["AAPL"], "date": [datetime.now()]})
        assert _passes_quality(df, "AAPL") is False

    def test_too_many_null_closes_fails(self):
        n = 100
        null_count = int(n * (MISSING_THRESH + 0.05))
        closes = [None] * null_count + [152.0] * (n - null_count)
        dates = pd.date_range("2023-01-01", periods=n, freq="B").to_list()
        df = pl.DataFrame({
            "ticker": ["AAPL"] * n,
            "date":   dates,
            "close":  closes,
        })
        assert _passes_quality(df, "AAPL") is False

    def test_just_under_missing_threshold_passes(self, valid_price_frame):
        n = len(valid_price_frame)
        null_count = int(n * (MISSING_THRESH - 0.01))
        closes = [None] * null_count + [152.0] * (n - null_count)
        df = valid_price_frame.with_columns(pl.Series("close", closes))
        assert _passes_quality(df, "AAPL") is True

    def test_duplicate_dates_fails(self):
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 4,
            "date":   [datetime(2023, 1, 1)] * 4,
            "close":  [152.0] * 4,
        })
        assert _passes_quality(df, "AAPL") is False

    def test_negative_close_fails(self):
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 5,
            "date":   pd.date_range("2023-01-01", periods=5, freq="B").to_list(),
            "close":  [-1.0, 152.0, 153.0, 154.0, 155.0],
        })
        assert _passes_quality(df, "AAPL") is False

    def test_zero_close_does_not_fail(self):
        """Zero is unusual but not a negative — should not be excluded on its own."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B").to_list()
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 5,
            "date":   dates,
            "close":  [0.0, 152.0, 153.0, 154.0, 155.0],
        })
        assert _passes_quality(df, "AAPL") is True


# ── _safe_get ─────────────────────────────────────────────────────────────────

class TestSafeGet:

    def _make_stmt(self, field, period, value):
        """Build a minimal yfinance-style DataFrame."""
        return pd.DataFrame({period: {field: value}})

    def test_returns_float_for_valid_field(self):
        df = self._make_stmt("Total Revenue", "2024-03-31", 5_000_000_000)
        result = _safe_get(df, "Total Revenue", "2024-03-31")
        assert result == pytest.approx(5_000_000_000.0)

    def test_returns_none_for_missing_field(self):
        df = self._make_stmt("Total Revenue", "2024-03-31", 5_000_000_000)
        result = _safe_get(df, "Gross Profit", "2024-03-31")
        assert result is None

    def test_returns_none_for_missing_period(self):
        df = self._make_stmt("Total Revenue", "2024-03-31", 5_000_000_000)
        result = _safe_get(df, "Total Revenue", "2023-12-31")
        assert result is None

    def test_returns_none_for_nan_value(self):
        df = self._make_stmt("Total Revenue", "2024-03-31", float("nan"))
        result = _safe_get(df, "Total Revenue", "2024-03-31")
        assert result is None

    def test_returns_none_on_empty_dataframe(self):
        result = _safe_get(pd.DataFrame(), "Total Revenue", "2024-03-31")
        assert result is None

    def test_handles_negative_values(self):
        """Net income can be negative — must not be discarded."""
        df = self._make_stmt("Net Income", "2024-03-31", -50_000_000)
        result = _safe_get(df, "Net Income", "2024-03-31")
        assert result == pytest.approx(-50_000_000.0)


# ── load_universe ─────────────────────────────────────────────────────────────

class TestLoadUniverse:

    def test_loads_tickers_from_csv(self, tmp_universe, monkeypatch):
        """
        Test load_universe() in isolation.
        Writes a known 5-ticker CSV to tmp dir and patches DATA_DIR to point there.
        Isolated from real universe.csv which grows over time.
        """
        import polars as pl
        data_dir = tmp_universe / "data"
        data_dir.mkdir(exist_ok=True)
        pl.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA", "PLTR", "AMD"]}).write_csv(
            data_dir / "universe.csv"
        )
        monkeypatch.setattr(fetch_module, "DATA_DIR", data_dir)
        tickers = fetch_module.load_universe()
        assert isinstance(tickers, list)
        assert len(tickers) == 7
        assert "AAPL" in tickers

    def test_raises_on_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_module, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_universe()

    def test_returns_strings(self, tmp_universe, monkeypatch):
        monkeypatch.setattr(fetch_module, "DATA_DIR", tmp_universe / "data")
        tickers = load_universe()
        assert all(isinstance(t, str) for t in tickers)

    def test_real_universe_has_minimum_tickers(self):
        """Real universe.csv should have at least 500 tickers — grows over time."""
        universe_path = ROOT / "data" / "universe.csv"
        if not universe_path.exists():
            pytest.skip("universe.csv not present")
        tickers = load_universe()
        assert len(tickers) >= 500, (
            f"Real universe has {len(tickers)} tickers — expected at least 500. "
            "If universe.csv was intentionally reduced, update this threshold."
        )


# ── Cache round-trip ──────────────────────────────────────────────────────────

class TestCacheRoundTrip:

    def test_price_parquet_roundtrip(self, tmp_path, valid_price_frame):
        path = tmp_path / "prices.parquet"
        valid_price_frame.write_parquet(path)
        loaded = pl.read_parquet(path)
        assert loaded.shape == valid_price_frame.shape
        assert loaded.columns == valid_price_frame.columns
        assert loaded["close"].to_list() == valid_price_frame["close"].to_list()

    def test_fundamentals_parquet_roundtrip(self, tmp_path):
        df = pl.DataFrame({
            "ticker":         ["AAPL", "AAPL"],
            "period":         ["2024-03-31", "2023-12-31"],
            "revenue":        [90_000_000_000.0, 89_000_000_000.0],
            "gross_margin":   [0.45, 0.44],
            "net_income":     [20_000_000_000.0, 19_000_000_000.0],
            "total_assets":   [300_000_000_000.0, 290_000_000_000.0],
            "total_liabilities": [200_000_000_000.0, 195_000_000_000.0],
            "free_cashflow":  [25_000_000_000.0, 24_000_000_000.0],
        })
        path = tmp_path / "fundamentals.parquet"
        df.write_parquet(path)
        loaded = pl.read_parquet(path)
        assert loaded["revenue"][0] == pytest.approx(90_000_000_000.0)
        assert loaded["gross_margin"][1] == pytest.approx(0.44)


# ── Integration — network tests (marked slow) ─────────────────────────────────

@pytest.mark.slow
class TestNetworkIntegration:

    def test_sp500_wikipedia_fetch(self):
        import urllib.request
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read()
        tables = pd.read_html(io.BytesIO(html))
        tickers = tables[0]["Symbol"].tolist()
        assert len(tickers) >= 500
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_nasdaq100_wikipedia_fetch(self):
        """
        Fetch Nasdaq-100 tickers from Wikipedia.
        Wikipedia occasionally changes table structure and column names —
        this test finds the ticker column defensively rather than hardcoding
        a table index or column name.
        """
        import urllib.request
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read()
        tables = pd.read_html(io.BytesIO(html))

        # Find the table that contains AAPL — more robust than hardcoding index
        ticker_col_names = ["Ticker", "Symbol", "Ticker symbol"]
        nasdaq_tickers = None

        for table in tables:
            for col_name in ticker_col_names:
                if col_name in table.columns:
                    candidates = table[col_name].dropna().tolist()
                    if "AAPL" in candidates:
                        nasdaq_tickers = candidates
                        break
            if nasdaq_tickers is not None:
                break

        assert nasdaq_tickers is not None, (
            "Could not find Nasdaq-100 ticker table on Wikipedia. "
            "Page structure may have changed — check the URL manually."
        )
        assert len(nasdaq_tickers) >= 90
        assert "AAPL" in nasdaq_tickers

    def test_yfinance_price_fetch(self):
        import yfinance as yf
        hist = yf.Ticker("AAPL").history(period="5d")
        assert not hist.empty
        assert "Close" in hist.columns
        assert len(hist) >= 3

    def test_yfinance_fundamentals_fetch(self):
        import yfinance as yf
        t = yf.Ticker("AAPL")
        assert not t.quarterly_income_stmt.empty
        assert "Total Revenue" in t.quarterly_income_stmt.index
        assert not t.quarterly_balance_sheet.empty
        assert "Total Assets" in t.quarterly_balance_sheet.index

    def test_fetch_prices_sample(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_module, "PRICE_CACHE", tmp_path / "prices.parquet")
        monkeypatch.setattr(fetch_module, "FUND_CACHE",  tmp_path / "funds.parquet")
        result = fetch_module.fetch_prices(["AAPL", "MSFT"])
        assert not result.is_empty()
        assert "ticker" in result.columns
        assert "date"   in result.columns
        assert "close"  in result.columns
        assert "volume" in result.columns
        assert result["volume"].dtype == pl.Int64
        assert result["close"].dtype  == pl.Float64
        assert result["close"].is_null().sum() == 0

    def test_fetch_fundamentals_sample(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_module, "FUND_CACHE", tmp_path / "funds.parquet")
        result = fetch_module.fetch_fundamentals(["AAPL"])
        assert not result.is_empty()
        expected_cols = [
            "ticker", "period", "revenue", "gross_profit",
            "total_assets", "total_liabilities", "free_cashflow",
            "gross_margin", "net_margin",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert result["ticker"][0] == "AAPL"