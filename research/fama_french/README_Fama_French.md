# Fama & French (1992) — Cross-Section of Expected Stock Returns
## Implementation and Spanning Test for the A_6M Strategy

**Paper:** Fama, E.F. & French, K.R. (1992). "The Cross-Section of Expected Stock Returns." *The Journal of Finance*, 47(2), 427–465.  
**Started:** April 2026  
**Status:** ✅ Complete — paper read, implementation built, spanning test run, findings documented

---

## Why this paper

Fama & French (1992) is the most cited paper in empirical finance. It introduces the three-factor model — market, size (SMB), and value (HML) — that the entire industry uses to benchmark returns and decompose portfolio performance.

For the A_6M trading system, the question was specific: **does A_6M's +2.81% mean alpha per trade reflect genuine momentum edge, or is it just exposure to known risk factors that could be replicated more cheaply with index funds?**

Testing this is called a spanning test — regressing portfolio excess returns against the three Fama-French factors and examining what's left over. What remains after stripping out factor exposure is genuine alpha.

---

## What the paper argues

### The failure of CAPM

The Capital Asset Pricing Model (CAPM) says a stock's expected return is fully explained by its beta — its sensitivity to the overall market. High beta stocks are riskier and should earn higher returns. This was the dominant theory in finance for decades.

Fama and French show it is empirically wrong. Sorting stocks into portfolios by beta and examining average returns produces a flat relationship — high beta stocks do not reliably earn more than low beta stocks. Beta fails to explain the cross-section of stock returns.

### What actually explains returns

Two variables do the heavy lifting:

**Size (market capitalisation):** Small companies earn higher average returns than large companies. This is the SMB factor — Small Minus Big. Historically +0.2% per month in excess of what beta predicts.

**Value (book-to-market ratio):** Companies with high book value relative to market price ("value stocks") earn higher returns than companies with low book-to-market ("growth stocks"). This is the HML factor — High Minus Low. Historically +0.3% per month.

Together with the market factor, size and value explain most of the variation in stock returns across the cross-section. Beta adds almost nothing once these two variables are controlled for.

### The three-factor model

Any portfolio's excess return can be decomposed as:

```
R_portfolio - RF = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML) + ε
```

Where:
- **α (alpha)** — return not explained by the three factors. Genuine edge if positive and statistically significant
- **β_mkt** — market exposure. How much the portfolio moves with the market
- **β_smb** — size exposure. Positive means tilts toward smaller companies
- **β_hml** — value exposure. Positive means tilts toward value stocks, negative means growth tilt

### Why they exclude financial firms

Banks and financial companies are structurally highly leveraged — leverage is their business model, not a signal of distress. Including them would contaminate the leverage variable, which is supposed to signal financial distress for other companies. Fama and French exclude financials to preserve the integrity of that signal.

### The look-ahead gap

Accounting data from December of year t-1 is matched with returns starting from July of year t. This prevents look-ahead bias — using information no real investor could have acted on yet. By July, even the most delayed filers have published their accounts. The same principle is applied in the A_6M backtest through `get_prices_as_of()` and `get_funds_as_of()`.

### The two-pass beta estimation

Estimating beta for individual stocks is unreliable because idiosyncratic noise — company-specific events — drowns out the market signal. Fama and French group stocks into portfolios by pre-ranking beta first. Idiosyncratic noise cancels across the portfolio, leaving a cleaner market signal. Those portfolio betas are then assigned back to individual stocks.

This pre-empts the most obvious attack on their findings: if the betas were badly estimated, the failure of beta to explain returns could be a measurement problem rather than a theoretical failure. By using portfolio betas, that criticism is closed before it can be made.

---

## The Fama-MacBeth regression results

The paper runs cross-sectional regressions of stock returns on candidate variables, testing each combination. Key findings:

| Variables tested | Beta significant? | Size significant? | BE/ME significant? |
|---|---|---|---|
| Beta alone | No (t=0.46) | — | — |
| Size alone | — | Yes (t=-2.58) | — |
| BE/ME alone | — | — | Yes (t=5.71) |
| Size + BE/ME | — | Marginal | Yes (t=4.44) |
| Beta + Size | No (t=-1.21) | Yes (t=-3.41) | — |

Beta fails every test. Once size and book-to-market are included, beta contributes nothing. Size and book-to-market together absorb the explanatory power of leverage and earnings-to-price as well — those variables lose significance once the two main factors are in the model.

The "parsimonious model" — two variables doing the work of many — is the paper's central contribution. Not five variables, not ten. Size and book-to-market. Everything else is a proxy for one or both of these.

---

## Implementation

### Phase 1 — Factor data

Downloaded Ken French's public three-factor monthly data from Dartmouth. Filtered to the A_6M backtest sample period: January 2011 to April 2026.

**Script:** `research/fama_french/fetch_factors.py`  
**Output:** `research/fama_french/ff3_factors.parquet`

**Real factor returns in the sample period:**

| Factor | Mean/month | Annualised | Notes |
|---|---|---|---|
| Mkt-RF | +1.045% | +12.5% | Strong bull market period |
| SMB | -0.154% | -1.85% | Size premium reversed in this period |
| HML | -0.046% | -0.55% | Value premium weak/absent |
| RF | +0.119% | +1.42% | Two regimes: near-zero pre-2022, elevated post-2022 |

The SMB and HML findings are important context for interpreting the spanning test. The size and value premiums that Fama and French documented on 1963-1990 data were largely absent or reversed during 2011-2026. Large-cap growth stocks — particularly technology — dominated returns throughout this period.

### Phase 2 — Monthly portfolio returns

Converted A_6M trade-level backtest data into monthly portfolio returns using mark-to-market pricing.

**Method:** For each calendar month, reconstruct which 10 positions were held (applying the same TOP_N=10 priority logic as the backtest). For each active position, find the actual close price at month start and month end. Compute the real price return for that month. Equal-weight average across all positions.

**Why mark-to-market matters:** An earlier approach distributed each trade's total return evenly across its 6-month hold period and averaged all 149 active signals per month. Averaging 149 numbers cancels almost all idiosyncratic noise, producing an artificially smooth return series with Sharpe 3.3 and t-statistic 12.5 — statistical artefacts, not real results. Mark-to-market with 10 positions produces genuine monthly volatility and realistic statistics.

**Script:** `research/fama_french/compute_portfolio_returns.py`  
**Output:** `research/fama_french/a6m_monthly_returns.parquet`

**Portfolio statistics:**

| Metric | Value |
|---|---|
| Months | 182 |
| Avg positions/month | 10.0 |
| Mean monthly return | +1.43% |
| Annualised return | +17.1% |
| Std dev monthly | 5.09% |
| Sharpe (annualised) | 0.890 |

### Phase 3 — Spanning test

Ran OLS regression of A_6M monthly excess returns on the three Fama-French factors.

**Script:** `research/fama_french/spanning_test.py`  
**Output:** `research/fama_french/spanning_test_results.parquet`

---

## Results

```
R_excess = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML) + ε

Sample: January 2011 — February 2026 (182 months)
```

| Metric | Value |
|---|---|
| **Monthly alpha** | **+0.341%** |
| **Annualised alpha** | **+4.10%** |
| **T-statistic** | **1.571** |
| **P-value** | **0.118** |
| **Significant?** | **No (threshold: t > 2.0)** |
| β_mkt | +0.959 (t = 18.16) ✓ |
| β_smb | +0.220 (t = 2.52) ✓ |
| β_hml | +0.059 (t = 0.93) ✗ |
| R-squared | 0.699 |
| Information ratio | 0.425 |

### What the results mean

**Alpha is positive but not yet statistically significant.** A_6M generates +4.1% annualised alpha above what the three factors predict. This is economically meaningful — but with 182 months of data and 5% monthly portfolio volatility, the t-statistic of 1.571 falls short of the 2.0 threshold. There is approximately a 12% probability this result is noise rather than genuine edge.

This does not mean the alpha is fake. It means the sample is not yet large enough to prove it is real. More live data is required.

**Market beta dominates.** β_mkt of 0.959 means A_6M moves nearly one-for-one with the market. Holding 10 concentrated S&P 500 positions, this is expected. It also means A_6M is not a market-neutral strategy — in sustained bear markets it will fall with the index. The R-squared of 0.699 confirms the market factor alone explains most of A_6M's return variation.

**Small-cap tilt within large-cap universe.** β_smb of +0.220 is statistically significant. Despite operating exclusively in S&P 500 stocks, A_6M systematically selects the smaller, more volatile members of the index. The dip-entry and momentum criteria favour stocks that have moved significantly — these tend to be smaller names within the large-cap universe rather than mega-cap stalwarts.

**No value tilt.** β_hml of +0.059 is not significant. A_6M is style-agnostic on the value-growth dimension. This is consistent with a momentum strategy — momentum selects recent winners regardless of their book-to-market ratio.

---

## Comparison with synthetic verification

Before running on real data, the spanning test was verified on synthetic data with known planted parameters. This confirmed the regression correctly recovers planted alpha and factor loadings, and established intuition for what affects alpha detection:

- More noise → harder to detect alpha (t-statistic falls)
- Less data → harder to detect alpha (t-statistic falls)
- Changing factor loadings does not change alpha
- Noise drift in finite samples is absorbed into the alpha estimate

The synthetic verification is in `research/fama_french/synthetic_verification.ipynb`.

---

## Limitations

**Sample period.** 2011-2026 is predominantly a bull market. The size and value premiums that Fama and French documented were weak or absent in this period. HML was -0.55% annualised — growth massively outperformed value. Results should not be generalised beyond this sample.

**Universe.** S&P 500 large-caps only. The SMB loading near zero-to-slightly-positive is partially constrained by the universe — all stocks are large by absolute standards. Full spanning tests typically use broader universes.

**Survivorship bias.** Post-2019 constituent filter is applied — backtest only includes tickers that were in the S&P 500 at each rebalance date. Pre-2019 is uncorrected due to data availability. This likely overstates returns by 3-7%, which carries through to the alpha estimate.

**Monthly return construction.** Mark-to-market with 10 positions is the correct approach but still an approximation. The backtest uses weekly rebalancing — some positions entered mid-month, some exited mid-month. Month-start and month-end prices are used as anchors, which slightly smooths within-month volatility.

**Statistical power.** 182 months is at the lower bound for detecting alpha of this magnitude with this level of portfolio volatility. The synthetic stress tests showed that with 3% noise and 0.30% monthly alpha, 184 months was insufficient to clear significance. A_6M has higher monthly volatility (5%) and the alpha is +0.34% monthly — more data is needed.

---

## Connection to the trading system

| Finding | Implication |
|---|---|
| Alpha +4.1% ann., t-stat 1.571 | Edge exists but unproven — live track record is the priority |
| β_mkt = 0.959 | Regime detector is critical — bear market protection needed |
| β_smb = 0.220 | Small-cap tilt within S&P 500 — monitor when universe expands |
| β_hml = 0.059 | No value bias — momentum is genuinely style-agnostic |
| R² = 0.699 | 30% of return variation is unexplained — where alpha and noise live |
| Sharpe = 0.890 | Realistic — consistent with genuine but moderate edge |

### What changes in the trading system

**Nothing changes immediately.** The spanning test confirms the system is working as designed. No signals to stop, no signals to modify.

**What this validates:**
- The paper trading gate approach is correct — more live data is the next priority
- The regime detector build (after Phase 6 gate) is important given β_mkt of 0.959
- The ML signal layer is the highest-leverage improvement available once the gate passes

**What this does not validate:**
- The exact alpha magnitude — 3-7% survivorship bias adjustment applies
- Statistical significance — 182 months is insufficient, live data required

---

## What comes next

| Milestone | Target | Purpose |
|---|---|---|
| Phase 6 gate (paper trading) | June 2026 | 20+ signals, accuracy >52% vs SPY |
| Regime detector | After gate | SPY 200MA + VIX — must come before vol scaling |
| ML model training | After gate | XGBoost on existing 3,879-signal dataset |
| Re-run spanning test | After 12mo live | Test whether live alpha clears t > 2.0 |
| Universe expansion | After Phase 7 stable | Mid-cap addition — will increase SMB loading |

---

## Files

```
research/fama_french/
    fetch_factors.py                — Downloads Ken French 3-factor data
    compute_portfolio_returns.py    — Mark-to-market monthly returns
    spanning_test.py                — OLS regression and diagnostics
    synthetic_verification.ipynb    — Synthetic data verification (Stages 1-4)
    ff3_factors.parquet             — Real Ken French factor data (2011-2026)
    a6m_monthly_returns.parquet     — A_6M monthly returns (mark-to-market)
    spanning_test_results.parquet   — Regression output
    spanning_test_plot.png          — Diagnostic charts
```

---

## Key vocabulary for fund conversations

Being able to discuss these results precisely matters in the fund role context. The right framing:

*"I tested A_6M against the Fama-French three-factor model on 182 months of backtest data. Alpha is +4.1% annualised with a t-statistic of 1.571 — positive and economically meaningful but not yet statistically significant at this sample size. Market beta is 0.96, consistent with a concentrated large-cap momentum strategy. There's a small but significant SMB loading of 0.22, which suggests the screener naturally gravitates toward the smaller end of the S&P 500. The value loading is near zero, consistent with a style-agnostic momentum approach. R-squared is 0.70 — the factors explain 70% of return variation, with the remaining 30% being alpha and idiosyncratic noise. The next step is accumulating live paper trading data to re-run the test on clean out-of-sample returns."*

That answer distinguishes someone who understands what they built from someone who just ran a backtest.