# Momentum Crashes — Replication & Analysis

**Paper:** Daniel, K. & Moskowitz, T.J. (2016). "Momentum Crashes."  
*Journal of Financial Economics*, 122(2), 221–247.  
**NBER PDF:** https://www.nber.org/system/files/working_papers/w20439/w20439.pdf  
**Completed:** April 2026  
**Author:** Personal research project — implementation from scratch, no vibe coding

---

## What this is

A from-scratch replication of Daniel & Moskowitz (2016), a landmark paper showing that momentum strategy crashes are forecastable and that a dynamic strategy scaling exposure by forecast return and variance approximately doubles the Sharpe ratio of a static momentum portfolio.

This project was built in two stages:

- **Notebook 1** (`dm_synthetic.ipynb`) — full implementation verified on synthetic data
- **Notebook 2** (`dm_real.ipynb`) — ported to 15 years of real S&P 500 + Nasdaq 100 data

The real data results diverge meaningfully from the paper. That divergence is the most interesting finding.

---

## What the paper argues

Momentum strategies — long past winners, short past losers — earn strong average returns but suffer infrequent, severe crashes. These crashes are not random. They occur in "panic states": after prolonged market declines (bear state) when the market then rebounds sharply.

The mechanism: in bear markets, past loser stocks accumulate extreme market beta through financial distress and leverage. When the market recovers, these high-beta losers rocket upward, destroying the short leg of the momentum portfolio.

D&M build a dynamic strategy that scales momentum exposure each month using:
- A **mean forecast** — expected WML return based on bear state and market variance
- A **variance forecast** — GJR-GARCH combined with realised volatility

The weight formula:

$$w^*_{t-1} = \frac{1}{2\lambda} \cdot \frac{\mu_{t-1}}{\sigma^2_{t-1}}$$

Result: Sharpe ratio approximately doubles vs static momentum. The strategy is not explained by market, size, value, or variance risk factors.

---

## Implementation

### Notebook 1 — Synthetic data

Built and verified every component against a synthetic universe of 100 stocks (5 years, monthly returns):

| Component | What was built | Gate result |
|---|---|---|
| WML construction | 12-1 formation, decile ranking, holding returns | ✓ Passed |
| Bear state indicator | 24-month rolling market return, binary flag | ✓ Passed |
| Crash characterisation | Conditional WML returns, Table 3 regression | ✓ Passed (signs correct) |
| Realised volatility | 6-month rolling std | ✓ Passed |
| GJR-GARCH | arch library, p=1, o=1, q=1 | ⚠ γ insignificant — sample too small, deferred |
| Mean forecast | Table 5 regression | ✓ Passed (signs correct) |
| Dynamic weighting | μ/σ² formula, λ calibration, floor at zero | ✓ Passed |
| Ordering check | Dynamic > Cvol > Static Sharpe | ✓ Passed |

All mechanics verified. Sample size limitations on synthetic data documented explicitly throughout.

### Notebook 2 — Real data

Applied to 2,027,088 daily OHLCV observations across 505 tickers (S&P 500 + Nasdaq 100 + historical removals), 2010–2026.

**Universe:** Large-cap US equities. Survivorship bias partially mitigated by including 132 historical index removals. Pre-2019 period uses full universe without constituent filtering — returns likely overstated by 1–3% annualised.

**Market proxy:** SPY monthly returns.

**Sample period:** January 2011 – April 2026 (181 months of WML returns after formation period).

---

## Results

### WML portfolio statistics

| Metric | This replication | D&M paper |
|---|---|---|
| Mean monthly return | +1.11% | +1.85% |
| Annualised Sharpe | 0.344 | 0.600 |
| Skewness | +0.53 | −4.70 |
| Worst month | −27.3% (2020-11) | ~−80% (1932) |

The momentum premium is confirmed. The Sharpe difference is expected — 15 years vs 87 years, modern large-cap vs full universe. The skewness divergence is the notable finding — see below.

### Bear state analysis

Only **one bear state month** was identified in the full sample (October 2023, rolling 24-month SPY return = −6.1%). The D&M crash detection framework requires multiple bear market cycles to estimate the forecasting relationship statistically. This sample — predominantly a bull market — provides insufficient variation.

The 24-month rolling return threshold is too coarse for modern markets. A shorter-horizon regime indicator (SPY 200-day MA + VIX) would fire more frequently and be more appropriate for this era.

### Variance forecasting

GJR-GARCH fitted on daily WML returns (3,763 observations):

| Parameter | Estimate | Significant? |
|---|---|---|
| omega | 0.049 | No |
| alpha | 0.007 | Marginal |
| gamma (asymmetry) | −0.006 | No |
| beta (persistence) | 0.996 | Yes (t=403) |

Gamma is negative and insignificant — the large-cap long-short WML portfolio does not exhibit asymmetric volatility response. This is consistent with the universe: large liquid stocks don't accumulate extreme beta in the same way distressed small caps do.

Standard GARCH(1,1) used instead. Volatility persistence (beta = 0.985) is the dominant effect and is strongly significant.

Combined variance forecast: GARCH contributed negligibly (coefficient = −0.004). Realised volatility alone used as variance forecast (coefficient = 0.844, dominant).

### Mean forecast

SPY variance coefficient: **+2.47** — wrong sign. High market volatility predicted better WML returns in this sample, opposite to the paper's finding.

Root cause: in 2011–2026, volatility spikes (2020 COVID, 2022 rate shock) were followed by sharp recoveries where momentum performed well. The paper's negative relationship between variance and WML returns requires bear market cycles where high volatility precedes crashes, not recoveries.

### Dynamic strategy results

| Strategy | Sharpe | Notes |
|---|---|---|
| Static WML | 0.344 | Baseline |
| Constant volatility | 0.220 | Underperforms — see below |

The constant volatility strategy **underperformed** static WML. The worst cvol months were cases where trailing volatility was low (weight high) and WML then crashed — 2023-01, 2017-04, 2014-04 all followed this pattern.

This is the key finding: WML crashes on large-cap US stocks in 2011–2026 occur predominantly from low-volatility environments. Scaling up exposure in low-vol periods amplifies losses rather than protecting against them. Volatility provides no advance warning of crashes on this sample.

---

## Key findings

**Finding 1 — The crash mechanism is real but rare in this sample**

2020-11 (COVID recovery, WML = −27.3%) matches the D&M panic state exactly — prolonged bear market followed by sharp rally, losers ripping upward. The mechanism is real. The sample just doesn't provide enough episodes to build a statistical model.

**Finding 2 — Volatility scaling requires a regime filter**

Naive inverse-vol scaling actively hurts on modern large-cap data (Sharpe 0.344 → 0.220). Volatility scaling should only be applied conditionally when a regime indicator confirms dangerous conditions — not unconditionally.

**Finding 3 — Large-cap universe behaves differently**

The paper's extreme negative skewness (−4.70) is driven by distressed small-cap stocks accumulating extreme beta in bear markets. Large-cap S&P 500 stocks don't distress the same way. This universe is less crash-prone but also captures less of the crash recovery alpha.

**Finding 4 — The D&M framework is conditions-dependent**

The full dynamic strategy requires: multiple bear market cycles, crashes that follow identifiable high-volatility panic states, and a long enough sample for the statistical relationships to be stable. Modern large-cap bull market data doesn't provide these conditions. The logic is sound — the conditions haven't been present.

---

## Implications for A_6M trading system

This replication directly informs three decisions in the A_6M research roadmap:

**1 — Do not apply unconditional volatility scaling**
Real data shows this hurts Sharpe. Volatility scaling is only safe when combined with a regime signal that confirms the bear state.

**2 — Regime detector design**
The D&M 24-month rolling return threshold is too coarse for modern markets (fires once in 15 years). The planned SPY 200MA + VIX flag is better calibrated to modern market cycles and should be implemented first.

**3 — Revisit after a bear market**
If markets enter a prolonged bear cycle — 24-month SPY return negative, VIX elevated, momentum underperforming — the D&M framework becomes applicable. The infrastructure is built and ready. The conditions just haven't arrived.

---

## What was deliberately not implemented

- International equity markets (Section 5) — data not available in existing pipeline
- Other asset classes (bonds, currencies, commodities) — out of scope
- Full spanning tests — insufficient bear state variation to run meaningfully
- Short positions — ISA account constraints, long-only universe only
- Pre-2010 data — existing price pipeline starts 2010

---

## Limitations

- **Sample period:** 2011–2026 is a predominantly bull market era. The D&M findings are strongest over full market cycles including prolonged bear markets.
- **Universe:** Large-cap S&P 500 + Nasdaq 100 only. Academic momentum is stronger in smaller stocks.
- **Survivorship bias:** Partially mitigated by 132 historical removals. Pre-2019 returns overstated by estimated 1–3% annualised.
- **No shorting:** Real implementation is long-only. WML as constructed here is theoretical — the short leg is not executable in an ISA account.
- **Monthly data for GARCH:** Paper uses daily data throughout. Monthly GARCH is less reliable — daily WML returns used for GARCH fitting as a partial fix.

---

## Files

```
research/momentum_crashes/
├── notebooks/
│   ├── dm_synthetic.ipynb     # Notebook 1 — synthetic verification
│   └── dm_real.ipynb          # Notebook 2 — real data results
├── utils/
│   ├── wml.py                 # WML construction functions
│   ├── signals.py             # Bear state, variance forecast, mean forecast
│   └── dynamic.py             # Weight computation, strategy comparison
└── README_Momentum_Crashes.md # This file
```

---

## Citation

Daniel, K. & Moskowitz, T.J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221–247. https://doi.org/10.1016/j.jfineco.2015.12.002

All findings in this replication are the author's own work on independent data. The original paper's methodology is reproduced for educational and research purposes only.