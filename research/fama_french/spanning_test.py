"""
research/fama_french/spanning_test.py
======================================
Fama-French three-factor spanning test for the A_6M strategy.

The question
------------
Does A_6M's alpha survive after controlling for market, size, and value factors?

    R_portfolio - RF = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML) + ε

If α is positive and statistically significant (t-stat > 2.0), A_6M generates
genuine momentum alpha not explained by the three factors.

If α collapses to zero, A_6M's returns were just exposure to known risk factors
that could be replicated more cheaply with index funds.

Inputs
------
    research/fama_french/ff3_factors.parquet       — Ken French factor returns
    research/fama_french/a6m_monthly_returns.parquet — A_6M monthly returns

Output
------
    Printed results + research/fama_french/spanning_test_results.parquet

Usage
-----
    python research/fama_french/spanning_test.py

    Run fetch_factors.py and compute_portfolio_returns.py first.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent.parent
FACTORS_FILE  = ROOT / "research" / "fama_french" / "ff3_factors.parquet"
RETURNS_FILE  = ROOT / "research" / "fama_french" / "a6m_monthly_returns.parquet"
RESULTS_FILE  = ROOT / "research" / "fama_french" / "spanning_test_results.parquet"
PLOT_FILE     = ROOT / "research" / "fama_french" / "spanning_test_plot.png"


def load_and_merge() -> pl.DataFrame:
    """
    Load factor data and portfolio returns, merge on date.

    Both files use the first day of each month as the date key.
    Inner join ensures we only use months present in both datasets.
    """
    if not FACTORS_FILE.exists():
        raise FileNotFoundError(
            f"Factor file not found: {FACTORS_FILE}\n"
            "Run: python research/fama_french/fetch_factors.py"
        )
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}\n"
            "Run: python research/fama_french/compute_portfolio_returns.py"
        )

    factors = pl.read_parquet(FACTORS_FILE)
    returns = pl.read_parquet(RETURNS_FILE)

    log.info(f"Factor data:   {len(factors)} months")
    log.info(f"Return data:   {len(returns)} months")

    # Inner join on date — both already filtered to 2011-2026
    merged = factors.join(returns, on="date", how="inner")

    log.info(f"Merged:        {len(merged)} months")
    log.info(f"Date range:    {merged['date'].min()} → {merged['date'].max()}")

    return merged


def run_ols(data: pl.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression of A_6M excess returns on the three Fama-French factors.

    Regression equation:
        portfolio_excess = α + β_mkt × mkt_rf + β_smb × smb + β_hml × hml + ε

    X matrix: [const, mkt_rf, smb, hml] — shape (n_months, 4)
    y vector: portfolio_excess           — shape (n_months,)

    The constant (intercept) is alpha — the return not explained by factors.
    """
    X = data.select(["mkt_rf", "smb", "hml"]).to_numpy()
    y = data["portfolio_excess"].to_numpy()

    # sm.add_constant prepends a column of ones to X
    # This allows the regression to estimate the intercept (alpha)
    X_with_const = sm.add_constant(X)

    model  = sm.OLS(y, X_with_const)
    result = model.fit()

    log.info(f"OLS regression complete — {len(y)} observations")

    return result


def interpret_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    data: pl.DataFrame,
) -> dict:
    """
    Extract and interpret key outputs from the OLS result.

    Returns a dictionary of all key metrics for printing and saving.
    """
    alpha_monthly    = result.params[0]
    alpha_annualised = alpha_monthly * 12
    alpha_tstat      = result.tvalues[0]
    alpha_pval       = result.pvalues[0]

    beta_mkt = result.params[1]
    beta_smb = result.params[2]
    beta_hml = result.params[3]

    tstat_mkt = result.tvalues[1]
    tstat_smb = result.tvalues[2]
    tstat_hml = result.tvalues[3]

    r_squared    = result.rsquared
    adj_r_squared = result.rsquared_adj
    n_months     = len(data)

    # Information ratio — alpha divided by residual standard deviation
    # Annualised version: multiply monthly alpha by 12, monthly std by sqrt(12)
    residuals     = result.resid
    residual_std  = float(np.std(residuals))
    info_ratio    = (alpha_monthly * 12) / (residual_std * np.sqrt(12))

    # Raw portfolio stats for context
    ann_raw_return  = data["portfolio_return"].mean() * 12
    ann_excess      = data["portfolio_excess"].mean() * 12
    monthly_std     = data["portfolio_excess"].std()
    sharpe          = (data["portfolio_excess"].mean() / monthly_std) * np.sqrt(12)

    return {
        # Alpha
        "alpha_monthly":    alpha_monthly,
        "alpha_annualised": alpha_annualised,
        "alpha_tstat":      alpha_tstat,
        "alpha_pval":       alpha_pval,
        "alpha_significant": abs(alpha_tstat) > 2.0,

        # Factor loadings
        "beta_mkt":   beta_mkt,
        "beta_smb":   beta_smb,
        "beta_hml":   beta_hml,
        "tstat_mkt":  tstat_mkt,
        "tstat_smb":  tstat_smb,
        "tstat_hml":  tstat_hml,

        # Model fit
        "r_squared":      r_squared,
        "adj_r_squared":  adj_r_squared,
        "n_months":       n_months,
        "info_ratio":     info_ratio,

        # Raw portfolio stats
        "ann_raw_return": ann_raw_return,
        "ann_excess":     ann_excess,
        "sharpe":         sharpe,
    }


def print_results(metrics: dict) -> None:
    """Print a clean, well-structured results summary."""

    sig = "✓ SIGNIFICANT" if metrics["alpha_significant"] else "✗ NOT SIGNIFICANT"

    print()
    print("=" * 60)
    print("FAMA-FRENCH SPANNING TEST — A_6M STRATEGY")
    print("=" * 60)
    print(f"  Sample:  Jan 2011 — Apr 2026  ({metrics['n_months']} months)")
    print(f"  Model:   R_excess = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML)")
    print()

    print("── Portfolio (raw, before factor adjustment) ────────")
    print(f"  Annualised return:    {metrics['ann_raw_return']:+.2f}%")
    print(f"  Annualised excess:    {metrics['ann_excess']:+.2f}%")
    print(f"  Sharpe ratio:         {metrics['sharpe']:.3f}")
    print()

    print("── Alpha (genuine edge after factor adjustment) ─────")
    print(f"  Monthly alpha:        {metrics['alpha_monthly']:+.4f}%")
    print(f"  Annualised alpha:     {metrics['alpha_annualised']:+.2f}%")
    print(f"  T-statistic:          {metrics['alpha_tstat']:.3f}  {sig}")
    print(f"  P-value:              {metrics['alpha_pval']:.4f}")
    print(f"  Information ratio:    {metrics['info_ratio']:.3f}")
    print()

    print("── Factor loadings (what the model says drives returns)")
    print(f"  β_mkt:  {metrics['beta_mkt']:+.4f}  (t = {metrics['tstat_mkt']:+.2f})")
    print(f"  β_smb:  {metrics['beta_smb']:+.4f}  (t = {metrics['tstat_smb']:+.2f})")
    print(f"  β_hml:  {metrics['beta_hml']:+.4f}  (t = {metrics['tstat_hml']:+.2f})")
    print()

    print("── Model fit ────────────────────────────────────────")
    print(f"  R-squared:            {metrics['r_squared']:.4f}")
    print(f"  Adj. R-squared:       {metrics['adj_r_squared']:.4f}")
    print()

    print("── Interpretation ───────────────────────────────────")

    # Alpha interpretation
    if metrics["alpha_significant"] and metrics["alpha_monthly"] > 0:
        print(f"  ALPHA: Genuine momentum alpha confirmed.")
        print(f"         A_6M earns {metrics['alpha_annualised']:+.1f}% annualised")
        print(f"         above what the three factors predict.")
    elif not metrics["alpha_significant"] and metrics["alpha_monthly"] > 0:
        print(f"  ALPHA: Positive but not statistically significant.")
        print(f"         Cannot distinguish from noise with this sample.")
        print(f"         More data needed (longer live track record).")
    else:
        print(f"  ALPHA: Alpha does not survive factor adjustment.")
        print(f"         Returns may be explained by factor exposures.")

    # SMB interpretation
    if metrics["beta_smb"] > 0.1:
        print(f"  SMB:   Positive loading — tilts toward small caps.")
    elif metrics["beta_smb"] < -0.1:
        print(f"  SMB:   Negative loading — large-cap tilt (expected for S&P 500).")
    else:
        print(f"  SMB:   Near-zero — no meaningful size tilt (expected).")

    # HML interpretation
    if metrics["beta_hml"] < -0.1:
        print(f"  HML:   Negative loading — growth tilt (expected for momentum).")
    elif metrics["beta_hml"] > 0.1:
        print(f"  HML:   Positive loading — value tilt (unexpected for momentum).")
    else:
        print(f"  HML:   Near-zero — no meaningful value tilt.")

    print()
    print("── Limitations ──────────────────────────────────────")
    print("  - Sample: 2011-2026 is predominantly a bull market.")
    print("  - Universe: S&P 500 large-caps only — SMB near zero by construction.")
    print("  - Survivorship bias: partially controlled post-2019, uncontrolled pre-2019.")
    print("  - Value premium (HML) was weak/negative in this period.")
    print("  - Results specific to this universe and period — not universal claims.")
    print()


def plot_results(
    data: pl.DataFrame,
    result: sm.regression.linear_model.RegressionResultsWrapper,
    metrics: dict,
) -> None:
    """
    Generate diagnostic plots for the spanning test.

    Four panels:
        1. Cumulative portfolio returns vs fitted (factor-explained) returns
        2. Alpha contribution over time (cumulative residuals)
        3. Factor loadings bar chart
        4. Residuals distribution
    """
    data_pd    = data.to_pandas()
    fitted     = result.fittedvalues
    residuals  = result.resid
    cum_actual = (1 + data_pd["portfolio_excess"] / 100).cumprod()
    cum_fitted = (1 + fitted / 100).cumprod()
    cum_alpha  = (1 + residuals / 100).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fama-French Spanning Test — A_6M Strategy\n"
        f"Alpha: {metrics['alpha_annualised']:+.1f}% ann.  "
        f"t-stat: {metrics['alpha_tstat']:.2f}  "
        f"R²: {metrics['r_squared']:.3f}",
        fontsize=13, fontweight="bold"
    )

    dates = data_pd["date"]

    # Panel 1 — actual vs fitted
    axes[0, 0].plot(dates, cum_actual, label="Actual excess return", color="steelblue", lw=1.5)
    axes[0, 0].plot(dates, cum_fitted, label="Factor-explained return", color="orange",
                    lw=1.5, linestyle="--")
    axes[0, 0].axhline(y=1, color="black", linestyle=":", lw=0.8)
    axes[0, 0].set_title("Actual vs Factor-Explained Returns")
    axes[0, 0].set_ylabel("Growth of £1")
    axes[0, 0].legend(fontsize=9)

    # Panel 2 — cumulative alpha (residuals)
    sig_label = "✓ Significant" if metrics["alpha_significant"] else "✗ Not Significant"
    axes[0, 1].plot(dates, cum_alpha, color="darkgreen", lw=1.5)
    axes[0, 1].axhline(y=1, color="black", linestyle="--", lw=0.8)
    axes[0, 1].set_title(f"Cumulative Alpha (Residuals)\n{sig_label}")
    axes[0, 1].set_ylabel("Cumulative residual return")

    # Panel 3 — factor loadings
    factors_labels = ["Mkt-RF", "SMB", "HML"]
    betas          = [metrics["beta_mkt"], metrics["beta_smb"], metrics["beta_hml"]]
    tstats         = [metrics["tstat_mkt"], metrics["tstat_smb"], metrics["tstat_hml"]]
    colors         = ["steelblue" if abs(t) > 2.0 else "lightgrey" for t in tstats]

    bars = axes[1, 0].bar(factors_labels, betas, color=colors, edgecolor="black", linewidth=0.8)
    axes[1, 0].axhline(y=0, color="black", lw=0.8)
    axes[1, 0].set_title("Factor Loadings\n(blue = significant, grey = not)")
    axes[1, 0].set_ylabel("Beta coefficient")

    for bar, tstat in zip(bars, tstats):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 if bar.get_height() >= 0 else bar.get_height() - 0.02,
            f"t={tstat:.2f}",
            ha="center", va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=9
        )

    # Panel 4 — residuals distribution
    axes[1, 1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1, 1].axvline(x=0, color="red", linestyle="--", lw=1.5)
    axes[1, 1].set_title("Residuals Distribution\n(should be roughly normal)")
    axes[1, 1].set_xlabel("Monthly residual (%)")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved to {PLOT_FILE}")
    plt.close()


def save_results(metrics: dict) -> None:
    """Save key metrics to parquet for future reference."""
    results_df = pl.DataFrame([metrics])
    results_df.write_parquet(RESULTS_FILE)
    log.info(f"Results saved to {RESULTS_FILE}")


def main() -> dict:
    log.info("Running Fama-French spanning test for A_6M")

    # Step 1 — load and merge
    data = load_and_merge()

    # Step 2 — run OLS regression
    result = run_ols(data)

    # Step 3 — interpret results
    metrics = interpret_results(result, data)

    # Step 4 — print results
    print_results(metrics)

    # Step 5 — generate plots
    try:
        plot_results(data, result, metrics)
    except Exception as e:
        log.warning(f"Plot generation failed: {e} — results still valid")

    # Step 6 — save results
    save_results(metrics)

    return metrics


if __name__ == "__main__":
    main()