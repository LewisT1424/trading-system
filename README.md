# Systematic Equity Screener

A fully systematic equity screening and backtesting framework built in Python. The system scores stocks against a multi-factor criteria set, backtests strategies across 15 years of point-in-time data, and sizes positions using a rules-based risk framework.

Built as a personal project to systematise an existing stock-picking process rather than starting from scratch.

---

## What it does

- Scores the S&P 500 + Nasdaq 100 universe (~650 tickers) against 9 fundamental and price criteria
- Backtests multiple strategies with weekly rebalances across 15 years of data (2010–2026)
- Uses point-in-time SEC EDGAR data throughout — no lookahead bias
- Sizes positions using a conviction-tier risk framework calibrated to a target portfolio
- Provides a Streamlit dashboard for interactive signal review

---

## Backtest results

Tested across 15 years (January 2011 – April 2026), weekly rebalances, 0.15% round-trip transaction costs.

| Metric | Strategy | SPY benchmark |
|--------|----------|---------------|
| Annualised return | +18.7% | +13.8% |
| Sharpe ratio | 0.78 | 0.61 |
| Max drawdown | -35.8% | -33.7% |
| 2022 return | -0.89% | -2.11% |

> **Survivorship bias note:** The universe includes 132 historical S&P 500 constituents that were later removed, and uses point-in-time constituent membership data from 2019 onwards. Pre-2019 returns are likely overstated by 3–7% due to partial survivorship bias. Realistic annualised expectation: 15–16%.

---

## Project structure

```
trading-system/
├── data/
│   ├── fetch.py              # OHLCV price pipeline — 16 years, ~650 tickers
│   ├── edgar_fetch.py        # SEC EDGAR bulk fundamentals, point-in-time filing dates
│   ├── constituents.py       # S&P 500 historical constituent membership
│   └── cache/                # Parquet cache — gitignored, regenerate with setup.py
├── screener/
│   ├── features.py           # Feature engineering — price and fundamental features
│   ├── run.py                # 9-criteria scoring engine, terminal output + CSV export
│   └── app.py                # Streamlit dashboard
├── backtest/
│   ├── engine.py             # Backtest loop — weekly rebalances, point-in-time data
│   ├── portfolio.py          # Daily mark-to-market portfolio simulation
│   └── metrics.py            # Sharpe, drawdown, benchmark comparison, gate checks
├── risk/
│   └── sizer.py              # Position sizer — conviction tier allocation
├── tests/
│   ├── test_engine.py        # 50 tests — lookahead firewall, point-in-time, rebalancing
│   ├── test_metrics_portfolio.py  # 50 tests — Sharpe, drawdown, portfolio simulation
│   ├── test_features.py      # Feature validation and lookahead checks
│   ├── test_fetch.py         # Data pipeline tests
│   └── test_sizer.py         # Position sizing rule tests
├── run_weekly.py             # Sunday pipeline — refresh, score, size, review
├── honest_results.py         # Trade-level performance analysis (per-trade, not curve)
├── sim_dashboard.py          # Portfolio tracking dashboard
├── inject_ciks.py            # Manual CIK map for delisted/acquired tickers
├── setup.py                  # One-time setup — fetches ticker universe
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/LewisT1424/trading-system.git
cd trading-system

pip install -r requirements.txt

# Fetch ticker universe (~650 tickers)
python setup.py

# Fetch 16 years of price data and SEC EDGAR fundamentals (~10 min, cached to parquet)
python data/fetch.py
python data/edgar_fetch.py
```

---

## Running the screener

```bash
# Terminal output — top 20 by score
python screener/run.py

# Top 30
python screener/run.py --top 30

# Export to CSV
python screener/run.py --out screener/output/

# Interactive Streamlit dashboard
streamlit run screener/app.py
```

---

## Running the backtest

```bash
# Run with default strategy and hold period
python backtest/engine.py

# Specify hold period in months
python backtest/engine.py --hold 6

# Test multiple hold periods
python backtest/engine.py --hold 1 3 6
```

---

## Running the weekly pipeline

```bash
# New week, no open positions
python run_weekly.py --portfolio 5000

# With open positions
python run_weekly.py --portfolio 5000 \
  --positions AAPL NVDA \
  --entry-prices 195.00 140.00 \
  --entry-dates 2026-01-01 2026-01-01

# Skip price refresh if data already current
python run_weekly.py --portfolio 5000 --no-fetch
```

---

## Running tests

```bash
# Full test suite
pytest tests/ -v

# Fast unit tests only
pytest tests/ -v -m "not slow"
```

All 100+ tests pass. The test suite includes a dedicated lookahead firewall — any feature that accidentally uses future data at the point-in-time simulation date will fail.

---

## Screening criteria

The screener scores each ticker against 9 criteria combining fundamental quality and price action. Each criterion maps to a computed feature column:

| # | Category | What it measures |
|---|----------|-----------------|
| C1 | Valuation | Asset quality relative to market cap, segmented by business type |
| C2 | Revenue | Consistency of revenue growth across trailing quarters |
| C3 | Earnings | Direction of net income improvement |
| C4 | Balance sheet | Asset/liability ratio trajectory |
| C5 | Entry timing | Price dip setup relative to 52-week high |
| C6 | Trend | Price relative to 200-day moving average |
| C7 | Margins | Gross margin quality and stability |
| C8 | Cash flow | Operating cash flow trajectory |
| C9 | Momentum | 6-month price momentum |

Tiebreaker within the same score: 1-week momentum.

---

## Position sizing

Positions are sized by conviction tier based on screener score:

| Tier | Allocation |
|------|------------|
| High conviction | 8% of portfolio |
| Medium conviction | 5% of portfolio |
| Hard maximum | £1,000 per position |
| Hard minimum | £50 per position |
| Cash reserve | 10% always held |
| Stop-loss | -30% triggers mandatory written review |

---

## Data sources

| Source | Used for |
|--------|----------|
| Yahoo Finance (yfinance) | OHLCV price history |
| SEC EDGAR bulk API | Point-in-time fundamental data |
| Wikipedia / manual | S&P 500 historical constituent membership |

No paid data sources required.

---

## Technical stack

- **Data:** Polars, yfinance, SEC EDGAR bulk API
- **Backtesting:** Custom engine with daily mark-to-market simulation
- **Dashboard:** Streamlit, Plotly
- **Testing:** pytest (100+ tests)
- **Experiment tracking:** MLflow (optional)

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Thesis extraction — investment criteria document | ✅ Complete |
| 2 | Data pipeline and screener | ✅ Complete |
| 3 | Backtesting framework | ✅ Complete |
| 4 | Position sizing and risk rules | ✅ Complete |
| 5 | ML signal layer — XGBoost + conformal prediction | 🔜 Year 2 |
| 6 | Paper trading — 8 week minimum gate | ⏳ In progress |
| 7 | Live deployment | Pending Phase 6 gate |

The ML signal layer (Phase 5) will use XGBoost as a false-positive filter on top of the screener signals, wrapped with conformal prediction for calibrated uncertainty — the same pattern used in the [Polymarket prediction pipeline](https://github.com/LewisT1424).

---

## Limitations and honest caveats

- **Survivorship bias:** Partially mitigated by historical constituent data from 2019 onwards. Pre-2019 returns overstated.
- **Lookahead bias:** Extensively tested (50 dedicated tests). EDGAR fundamentals use filing date, not report date.
- **Market impact:** Not modelled. System is designed for small portfolios where market impact is negligible.
- **Parameter sensitivity:** Strategy parameters have not been stress-tested across all possible combinations. Overfitting risk exists.
- **Live performance:** Backtested only. Paper trading in progress. No live track record yet.