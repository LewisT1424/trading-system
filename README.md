# equity-screener

Systematic equity screener with ML signal layer. Built from personal trading edge — systematising what already works rather than starting from scratch.

## Structure

```
equity-screener/
├── data/
│   ├── fetch.py          # Price + fundamentals pipeline (511 tickers, cached to parquet)
│   └── cache/            # Parquet cache — not tracked in git
├── screener/
│   ├── features.py       # Feature engineering — price and fundamental features
│   ├── run.py            # Scoring engine — 9 criteria, terminal output
│   └── app.py            # Streamlit dashboard
├── backtest/             # Phase 3 — not built yet
├── signals/              # Phase 5 — not built yet
├── journal/              # Decision journal templates
├── risk/                 # Phase 4 — sizer.py not built yet
├── tests/
│   ├── test_fetch.py     # 15 tests — data pipeline
│   └── test_features.py  # 48 tests — feature validation + lookahead checks
├── setup.py              # Run once — fetches ticker universe
├── dashboard.py          # Portfolio dashboard (separate from screener)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python setup.py                  # fetch S&P 500 + Nasdaq 100 universe (~517 tickers)
python data/fetch.py             # fetch prices + fundamentals (~10 min, cached)
```

## Running the screener

```bash
# Terminal output — top 20
python screener/run.py

# Top 30
python screener/run.py --top 30

# Save CSV
python screener/run.py --out results/

# Streamlit dashboard
streamlit run screener/app.py
```

## Running tests

```bash
pytest tests/ -v -m "not slow"   # fast unit tests only
pytest tests/ -v                 # full suite including network tests
```

## Build status

| Phase | Nights | Status |
|-------|--------|--------|
| 1 — Thesis extraction | 1–6 | ✅ Complete |
| 2 — Data pipeline + screener | 7–12 | ✅ Complete |
| 3 — Backtesting | 13–20 | 🔜 Next |
| 4 — Position sizing | 21 | Pending |
| 5 — ML signal layer | 22+ | Pending |
| 6 — Paper trading | 8 weeks | Pending |
| 7 — Live deploy | Month 5–6 | Pending |

## Phase 2 gate

- ✅ Screener runs end-to-end in 0.1s (target: under 3 minutes)
- ✅ AMD rank 1 (8/9), PLTR rank 13 (7/9)
- ⚠️ ASML rank 180 (6/9) — failing C5 and C8 on current market conditions (near ATH, not in dip). Would have passed at September 2024 entry. Conditionally accepted — backtest will validate on historical data.
- ✅ Dashboard loads and displays correctly

## Investment Criteria Document (/9)

| # | Criterion | Measurement |
|---|-----------|-------------|
| C1 | Larger than priced | Asset-heavy: asset/mcap > 0.5 · Hardware: GM > 0.45 + growth > 15% · Asset-light: GM > 0.55 + growth > 12% |
| C2 | Consistent revenue | Positive in 3 of 4 quarters (consistency >= 0.75) |
| C3 | Net income improving | Newest quarter > oldest of 4 quarters |
| C4 | Ratio improving | Asset/liability ratio newest > oldest |
| C5 | Dip entry | Below 52w high AND 3M momentum < -10% |
| C6 | Above 200MA | Close > 200-day moving average |
| C7 | Margin quality | Stable or improving AND recent 2Q avg > 0.40 |
| C8 | Cash flow improving | Operating CF newest quarter > oldest |
| C9 | 6M positive momentum | 6M momentum > 0% |

**Tiebreaker within same score:** 1W momentum (higher ranked first)

**Conviction tiers:**
- 8–9 = High → 4–5% of portfolio
- 6–7 = Medium → 2–3% of portfolio
- <6 = Speculative → max 1%

## Exclusion rules (any one = no trade)

1. Entry into a parabolic spike
2. Pre-revenue or negligible revenue
3. Negative cash flow with no recovery trajectory
4. Financial statements not reviewed before buying
5. Social media as primary source of conviction
6. Adding to a losing position without new information
7. Position size exceeds conviction tier limit

## Phase 3 — Backtesting (next)

⚠️ **Lookahead bias is the primary risk.** Every feature must use only data available at the simulation date. Quarterly earnings only available after filing date, not report date.

Gates (all three must pass before paper trading):
- Out-of-sample Sharpe > 0.5 on 2022–2023 data
- Screener beats SPY buy-and-hold after costs over 2019–2023
- 2022 max drawdown ≤ 1.5× SPY 2022 drawdown