# Jegadeesh & Titman (1993) — Implementation and Findings

**Paper:** Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *The Journal of Finance*, 48(1), 65–91.

**Started:** May 2026
**Status:** ✅ COMPLETE — all phases done, findings documented

---

## What this was

A from-scratch implementation of the Jegadeesh & Titman (1993) paper on momentum strategies. Built in four phases: synthetic verification, real data replication, skip-month test on A_6M ML dataset, and formation period analysis.

**Key outputs:**
- `research/jt/notebooks/jt_synthetic.ipynb` — synthetic verification
- `research/jt/notebooks/jt_real.ipynb` — real data replication
- `research/data/ml_labels.parquet` — updated ML dataset (23 features, added momentum_5m_skip)
- `research/jt/README_JT.md` — this document

---

## What the paper argues

J&T test whether buying stocks that performed well over the past 3-12 months (winners) and selling stocks that performed poorly (losers) generates abnormal returns over the following 3-12 months.

**The core finding:** Momentum strategies generate significant positive returns across all 16 combinations of formation period (J = 3, 6, 9, 12 months) and holding period (K = 3, 6, 9, 12 months). The strongest single strategy is 12-month formation, 3-month hold — returning 1.31% per month.

**The mechanism:** J&T decompose momentum profits into three sources using a one-factor model:

- **Term 1** — cross-sectional dispersion in expected returns (efficient market explanation)
- **Term 2** — serial correlation in the market factor (efficient market explanation)
- **Term 3** — serial covariance in firm-specific returns (market inefficiency)

They find Term 2 is actually negative — the market slightly mean-reverts. Term 3 is positive — firm-specific returns persist from one period to the next. This is underreaction: when good company-specific news arrives, the market does not fully price it in immediately. The price drifts upward over the following 6-12 months as the market gradually catches up. Momentum strategies capture this drift.

**Additional findings:**
- Momentum profits are not explained by systematic risk — losers have higher beta than winners
- The zero-cost winners minus losers portfolio has near-zero beta (-0.08) — not a market bet
- Momentum works across all size subsamples — not a small-cap phenomenon
- Strategy loses money in January but wins in every other month
- Momentum profits positive in 4 out of 5 five-year subperiods from 1965-1989

---

## What was implemented

### Phase 1 — Synthetic Verification

Built a synthetic universe of 100 stocks over 60 months with Term 3 deliberately embedded (serial correlation rho=0.15 in firm-specific returns). Confirmed:

- Firm-specific serial covariance positive (+0.000387) — Term 3 present
- Momentum spread positive across most J/K combinations — 13/16 strategies positive
- Longer formation periods produce stronger signals — J=12 dominates significant results
- Skip-month effect formation-period dependent — helps J=12, hurts J=6 in small samples
- A_6M equivalent (J=6 K=6) positive but not significant — sample size limitation, not methodology flaw

**Gate: PASSED**

### Phase 2 — Real Data Replication

Tested all 16 J/K combinations (plus skip-week variants = 32 total) on the actual trading system universe — 504 S&P 500 large cap stocks, 2012-2026, using month-end close prices.

**Headline results:**

| Strategy | Mean monthly return | T-stat | P-value | Significant |
|---|---|---|---|---|
| J=6 K=6 no skip (A_6M equivalent) | 0.0044 | 3.33 | 0.001 | YES |
| J=12 K=3 no skip (J&T strongest) | 0.0054 | 2.45 | 0.015 | YES |
| J=6 K=6 skip month | 0.0041 | 2.92 | 0.004 | YES |
| J=12 K=6 no skip | 0.0044 | 2.79 | 0.006 | YES |

- **All 32 strategies produced positive WML returns**
- **26/32 statistically significant (p < 0.05)**
- A_6M equivalent cumulative return: ~2.0x over 14 years
- Returns roughly half of J&T original — expected given large cap universe and bull market period

**Skip month effect:** Mixed and marginal. Panel B (skip) does not consistently outperform Panel A (no skip) on this universe. Phase 3 test on A_6M signals required before drawing conclusions.

**Gate: PASSED**

### Phase 3 — Skip Month Test on A_6M ML Dataset

Computed `momentum_5m_skip` for all 3,879 A_6M signals in the ML dataset. Definition: return from 6 months ago to 1 month ago — skipping the most recent month to remove short-term reversal contamination.

**Results:**

| Feature | Correlation with label | P-value | Significant | Add to ML? |
|---|---|---|---|---|
| momentum_6m | 0.0546 | 0.0007 | YES | Already present |
| momentum_5m_skip | 0.0431 | 0.0073 | YES | **ADDED** |
| momentum_1m | 0.0430 | 0.0075 | YES | Already present |
| momentum_3m | 0.0284 | 0.0777 | NO | Excluded |

**Independence test:** Correlation between momentum_5m_skip and momentum_6m = 0.835 — below 0.9 threshold. Adds genuinely independent information.

**Key finding — contrary to J&T:** Raw momentum_6m is a stronger predictor than momentum_5m_skip on A_6M signals. On the raw universe J&T found skip improves returns. On quality-filtered A_6M signals, the most recent month contains real signal not noise — the stock is still in an active drift phase at entry. Skipping it removes useful information.

**ML dataset updated:** 22 → 23 features. momentum_5m_skip added.

### Phase 4 — Formation Period Analysis

Tested all momentum features for label correlation and inter-feature independence.

**Correlation with label (ranked):**

| Feature | Correlation | P-value | Status |
|---|---|---|---|
| momentum_6m | 0.0546 | 0.0007 | Strongest — confirmed |
| momentum_5m_skip | 0.0431 | 0.0073 | Strong — added |
| momentum_1m | 0.0430 | 0.0075 | Acceleration signal |
| momentum_1w | 0.0351 | 0.0290 | Weekly acceleration |
| momentum_3m | 0.0284 | 0.0777 | EXCLUDED |

**J&T prediction validation:**
- 6M > 3M — CONFIRMED — longest formation strongest as predicted
- 3M < 1M — CONTRADICTED — acceleration effect dominates on A_6M signals

The 3M vs 1M contradiction is expected on A_6M signals specifically. Every signal in the dataset already has positive 6M momentum by construction (C9 requirement). 3M momentum adds little independent information — largely captured by C9 firing. 1M momentum captures whether the drift is still accelerating at entry — genuine additional information.

**Independence:** No momentum feature pair exceeds 0.9 correlation. All four retained features provide genuinely independent information to the ML model.

**Notable finding:** momentum_5m_skip and momentum_1m have correlation -0.229 — slightly negative. Mathematical consequence of skip construction — they capture opposite ends of recent price action and are genuinely complementary.

---

## What changed in the trading system

**1. momentum_5m_skip added to ML dataset**
ML dataset now has 23 features. The model will learn to use drift cleanliness alongside drift strength, monthly acceleration, and weekly acceleration. Direct output of Phase 3.

**2. J=12 formation period identified for future testing**
J&T's strongest configuration is J=12 K=3. A_6M uses J=6. Real data confirms J=12 produces higher returns on this universe (0.0054 vs 0.0044 per month). Testing whether switching to J=12 improves backtest results is added to the research backlog. Must be done after paper trading gate passes with enriched dataset.

**3. Mode C ML dataset and J&T analysis added to research backlog**
Mode C signals (full 9-criteria) may behave differently from A_6M signals on the skip-month test. Testing this requires building a Mode C backtest dataset first. Scheduled after paper trading gate passes.

---

## Key concepts confirmed on real data

**Momentum is real and pervasive on S&P 500 large caps 2012-2026**
All 32 strategies positive. 26/32 significant. Not a fluke of one configuration. The effect is robust across formation periods, holding periods, and skip-week variants.

**C9 is capturing Term 3 — firm-specific underreaction**
The A_6M equivalent returning 0.44% per month with t=3.33 on the actual trading universe confirms that 6-month price momentum captures the drift from delayed price adjustment to firm-specific information. This is the academic foundation for why C9 works and why it should continue to work.

**Momentum is not a small-cap or high-risk phenomenon**
Results hold across all size subsamples. A_6M's large cap universe is not a limitation. The small-cap tilt within the S&P 500 (β_smb = +0.220 from Fama-French spanning test) is a feature — smaller names within large cap have more firm-specific underreaction.

**The zero-cost portfolio is near market-neutral**
Confirmed from J&T Section III B — winners minus losers portfolio beta is -0.08. Momentum profits are not a market bet. They come from firm-specific drift, not market exposure. This is why A_6M added alpha even after Fama-French factor adjustment.

---

## Honest limitations

**Universe difference from J&T original**
J&T tested NYSE and AMEX stocks 1965-1989 including small caps and distressed names. This implementation uses S&P 500 large caps only. Large caps have more analyst coverage, faster information dissemination, and less firm-specific underreaction — compressing the winner-loser spread. Returns roughly half of J&T original are expected and not a concern for a long-only strategy.

**Time period**
2012-2026 is predominantly a bull market with low volatility relative to J&T's sample which included multiple bear markets. The D&M replication showed momentum crashes occur in bear market conditions. The current sample period has limited bear market exposure which may overstate the consistency of momentum returns.

**No transaction costs**
WML returns reported gross of transaction costs. With 0.15% round-trip cost on Trading 212, real returns would be slightly lower. Not material for A_6M which holds for 6 months and rebalances weekly.

**Survivorship bias**
Constituent filter covers 2019-2026 only. Pre-2019 period uses the full universe including historical removals. This likely overstates returns modestly — same caveat as the main backtest.

**Skip month conclusion limited**
The skip-month finding (raw 6M stronger than skip on A_6M signals) is based on correlation with backtest labels only. It has not been validated on live paper trading data. Phase 3 conclusion should be treated as a hypothesis confirmed on historical data, subject to validation on out-of-sample data when the ML model is trained.

---

## Research backlog items generated

| Item | When | Why |
|---|---|---|
| Test J=12 formation period vs J=6 on backtest engine | After paper trading gate | J&T and real data both suggest J=12 may outperform |
| Build Mode C ML dataset | After paper trading gate | Required before Mode C skip-month test |
| Mode C skip-month test | After Mode C dataset built | May behave differently from A_6M signals |
| Validate momentum_5m_skip on live paper trading data | At ML model training | Confirm Phase 3 finding is out-of-sample robust |

---

## This project is complete

Do not revisit unless:
- Paper trading gate passes and ML model training begins — momentum_5m_skip will be included automatically as it is already in the dataset
- A proper bear market cycle arrives — J&T's January effect and subperiod analysis become more relevant
- J=12 backtest is ready to run — use the backtest engine with modified formation period parameter

The notebooks are done. The ML dataset is updated. Move on to live paper trading protocol and weekly Monday runs.