"""
screener/app.py
Night 12 — Streamlit screener dashboard.

Features:
  - Ranked table with all 9 criteria scores
  - Per-stock criteria pass/fail breakdown
  - 6M price sparkline
  - Filters: tier, sector, min score

Usage:
    streamlit run screener/app.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
import yfinance as yf
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screener"))

from features import price_features, fundamental_features
from run import score_tickers, CRITERIA

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Screener",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #080808; }
section[data-testid="stSidebar"] { background-color: #0f0f0f; border-right: 1px solid #1e1e1e; }

.metric-card {
    background: #0f0f0f; border: 1px solid #1e1e1e;
    border-radius: 4px; padding: 14px 18px; margin-bottom: 8px;
}
.metric-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #444; font-family: 'IBM Plex Mono', monospace; margin-bottom: 4px;
}
.metric-value {
    font-size: 20px; font-weight: 500;
    font-family: 'IBM Plex Mono', monospace; color: #e0e0e0;
}
.metric-value.pos { color: #4ade80; }
.metric-value.neg { color: #f87171; }

.criteria-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 12px 0;
}
.crit-pill {
    display: flex; align-items: center; gap: 6px;
    background: #0f0f0f; border: 1px solid #1e1e1e;
    border-radius: 3px; padding: 6px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
}
.crit-pill.pass { border-color: #166534; background: #052e16; color: #4ade80; }
.crit-pill.fail { border-color: #7f1d1d; background: #2d0a0a; color: #f87171; }

.ticker-header {
    font-size: 28px; font-weight: 500; color: #f0f0f0;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em;
}
.ticker-sub { font-size: 13px; color: #555; }

.score-badge {
    display: inline-block; padding: 4px 12px; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    font-weight: 500; letter-spacing: 0.05em;
}
.score-high { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.score-med  { background: #1c1a08; color: #fbbf24; border: 1px solid #713f12; }
.score-spec { background: #1a0a0a; color: #f87171; border: 1px solid #7f1d1d; }

.section-label {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #333; font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #1a1a1a; padding-bottom: 6px; margin: 20px 0 12px;
}

h1, h2, h3 { color: #e0e0e0 !important; }
.stSelectbox label, .stRadio label, .stSlider label, .stMultiSelect label {
    color: #666 !important; font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_scored() -> pl.DataFrame:
    prices = pl.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    funds  = pl.read_parquet(ROOT / "data" / "cache" / "fundamentals.parquet")
    pf = price_features(prices)
    ff = fundamental_features(funds)
    return score_tickers(pf, ff)


@st.cache_data(ttl=1800)
def fetch_sparkline(ticker: str) -> pl.DataFrame:
    yf_ticker = "INRG.L" if ticker == "INRG" else ticker
    try:
        hist = yf.download(yf_ticker, period="6mo", progress=False, auto_adjust=True)
        if hist.empty:
            return pl.DataFrame()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.reset_index()
        hist.columns = [str(c) for c in hist.columns]
        return pl.from_pandas(hist)
    except Exception:
        return pl.DataFrame()


def metric(label, value, css=""):
    return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value {css}'>{value}</div></div>")


def score_badge(score, tier):
    css = {"High": "score-high", "Medium": "score-med"}.get(tier, "score-spec")
    return f"<span class='score-badge {css}'>{score}/9 — {tier}</span>"


def tick(v):
    if v is True:  return "<span style='color:#4ade80'>✓</span>"
    if v is False: return "<span style='color:#2a2a2a'>✗</span>"
    return "<span style='color:#2a2a2a'>—</span>"


def pct(v, d=1):
    if v is None: return "—"
    col = "#4ade80" if v > 0 else "#f87171" if v < 0 else "#555"
    return f"<span style='color:{col}'>{v:+.{d}f}%</span>"


# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Running screener..."):
    scored = load_scored()

all_sectors = sorted(scored["sector"].drop_nulls().unique().to_list())


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ▲ Screener")
    st.markdown("---")

    min_score = st.slider("Min score", 0, 9, 6)
    tier_filter = st.multiselect("Tier", ["High", "Medium", "Speculative"],
                                  default=["High", "Medium"])
    sector_filter = st.multiselect("Sector", all_sectors, default=[])

    st.markdown("---")
    for k, v in CRITERIA.items():
        st.markdown(
            f"<div style='font-size:10px;color:#333;font-family:IBM Plex Mono;margin-bottom:3px'>"
            f"<span style='color:#555'>{k}</span> {v[:50]}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:10px;color:#2a2a2a;font-family:IBM Plex Mono'>"
        f"Universe: {len(scored)} tickers</div>",
        unsafe_allow_html=True,
    )


# ── Filter ────────────────────────────────────────────────────────────────────

filtered = scored.filter(pl.col("score") >= min_score)
if tier_filter:
    filtered = filtered.filter(pl.col("tier").is_in(tier_filter))
if sector_filter:
    filtered = filtered.filter(pl.col("sector").is_in(sector_filter))
filtered = filtered.sort(["score", "momentum_1w"], descending=[True, True], nulls_last=True)


# ── Summary cards ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric("Showing", f"{len(filtered)} tickers"), unsafe_allow_html=True)
with c2:
    n = len(filtered.filter(pl.col("tier") == "High"))
    st.markdown(metric("High conviction", str(n), "pos"), unsafe_allow_html=True)
with c3:
    n = len(filtered.filter(pl.col("tier") == "Medium"))
    st.markdown(metric("Medium conviction", str(n)), unsafe_allow_html=True)
with c4:
    n = len(filtered.filter(pl.col("c5_dip_entry") == True))
    st.markdown(metric("In dip setup", str(n), "pos"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Results table ─────────────────────────────────────────────────────────────

st.markdown("<div class='section-label'>Ranked results</div>", unsafe_allow_html=True)

if filtered.is_empty():
    st.info("No tickers match the current filters.")
else:
    tier_col = {"High": "#4ade80", "Medium": "#fbbf24", "Speculative": "#f87171"}

    rows_html = ""
    for i, row in enumerate(filtered.iter_rows(named=True), 1):
        tc = tier_col.get(row["tier"], "#555")
        mc = row.get("market_cap")
        mc_str = f"${mc/1e9:.1f}bn" if mc and mc >= 1e9 else f"${mc/1e6:.0f}m" if mc else "—"
        gm = row.get("gross_margin_latest")
        gm_str = f"{gm:.2f}" if gm is not None else "—"

        rows_html += f"""
        <tr style='border-bottom:1px solid #0f0f0f'>
          <td style='color:#2a2a2a;font-size:11px;padding:7px 5px'>{i}</td>
          <td style='font-family:IBM Plex Mono;font-weight:500;color:#e0e0e0;padding:7px 5px'>{row['ticker']}</td>
          <td style='font-family:IBM Plex Mono;font-size:12px;color:{tc};padding:7px 5px'>{row['score']}/9</td>
          <td style='font-size:11px;color:{tc};padding:7px 5px'>{row['tier']}</td>
          <td style='font-size:10px;color:#333;padding:7px 5px'>{(row.get('sector') or '—')[:18]}</td>
          <td style='text-align:center'>{tick(row['c1_larger_than_priced'])}</td>
          <td style='text-align:center'>{tick(row['c2_consistent_revenue'])}</td>
          <td style='text-align:center'>{tick(row['c3_net_income_improving'])}</td>
          <td style='text-align:center'>{tick(row['c4_ratio_improving'])}</td>
          <td style='text-align:center'>{tick(row['c5_dip_entry'])}</td>
          <td style='text-align:center'>{tick(row['c6_above_200ma'])}</td>
          <td style='text-align:center'>{tick(row['c7_margin_quality'])}</td>
          <td style='text-align:center'>{tick(row['c8_cashflow_improving'])}</td>
          <td style='text-align:center'>{tick(row['c9_momentum_6m'])}</td>
          <td style='font-family:IBM Plex Mono;font-size:11px;padding:7px 5px'>{pct(row.get('momentum_6m'))}</td>
          <td style='font-family:IBM Plex Mono;font-size:11px;padding:7px 5px'>{pct(row.get('momentum_3m'))}</td>
          <td style='font-family:IBM Plex Mono;font-size:11px;padding:7px 5px'>{pct(row.get('momentum_1w'))}</td>
          <td style='font-family:IBM Plex Mono;font-size:11px;color:#444;padding:7px 5px'>{gm_str}</td>
          <td style='font-family:IBM Plex Mono;font-size:11px;color:#333;padding:7px 5px'>{mc_str}</td>
        </tr>"""

    th = ("color:#2a2a2a;font-size:10px;letter-spacing:0.07em;"
          "text-transform:uppercase;padding:6px 5px;font-weight:500;text-align:left")
    thc = th + ";text-align:center"

    st.markdown(f"""
    <div style='overflow-x:auto'>
    <table style='width:100%;border-collapse:collapse;font-size:12px'>
      <thead><tr style='border-bottom:1px solid #1a1a1a'>
        <th style='{th}'>#</th>
        <th style='{th}'>Ticker</th>
        <th style='{th}'>Score</th>
        <th style='{th}'>Tier</th>
        <th style='{th}'>Sector</th>
        <th style='{thc}'>C1</th><th style='{thc}'>C2</th>
        <th style='{thc}'>C3</th><th style='{thc}'>C4</th>
        <th style='{thc}'>C5</th><th style='{thc}'>C6</th>
        <th style='{thc}'>C7</th><th style='{thc}'>C8</th>
        <th style='{thc}'>C9</th>
        <th style='{th}'>6M</th><th style='{th}'>3M</th>
        <th style='{th}'>1W</th><th style='{th}'>GM</th>
        <th style='{th}'>Mkt cap</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)


# ── Stock drill-down ──────────────────────────────────────────────────────────

st.markdown("<br><div class='section-label'>Stock detail</div>", unsafe_allow_html=True)

ticker_list = (filtered["ticker"].to_list()
               if not filtered.is_empty()
               else scored["ticker"].to_list())

selected = st.selectbox("Select ticker", ticker_list, label_visibility="collapsed")

if selected:
    row_df = scored.filter(pl.col("ticker") == selected)
    if not row_df.is_empty():
        row = row_df.to_dicts()[0]

        col_t, col_b = st.columns([3, 1])
        with col_t:
            st.markdown(
                f"<div class='ticker-header'>{selected}</div>"
                f"<div class='ticker-sub'>{row.get('sector','—')} · {row.get('industry','—')}</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"<div style='text-align:right;margin-top:10px'>"
                f"{score_badge(row['score'], row.get('tier',''))}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        close = row.get("close")
        mom6  = row.get("momentum_6m")
        mom3  = row.get("momentum_3m")
        b52   = row.get("pct_below_52w_high")
        gm    = row.get("gross_margin_latest")

        with m1:
            st.markdown(metric("Price", f"${close:.2f}" if close else "—"), unsafe_allow_html=True)
        with m2:
            v = f"{mom6:+.1f}%" if mom6 is not None else "—"
            st.markdown(metric("6M momentum", v, "pos" if mom6 and mom6>0 else "neg"), unsafe_allow_html=True)
        with m3:
            v = f"{mom3:+.1f}%" if mom3 is not None else "—"
            st.markdown(metric("3M momentum", v, "pos" if mom3 and mom3>0 else "neg"), unsafe_allow_html=True)
        with m4:
            v = f"{b52:.1f}%" if b52 is not None else "—"
            st.markdown(metric("Below 52w high", v, "neg" if b52 and b52<0 else ""), unsafe_allow_html=True)
        with m5:
            st.markdown(metric("Gross margin", f"{gm:.2f}" if gm is not None else "—"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Criteria pills
        st.markdown("<div class='section-label'>Criteria breakdown</div>", unsafe_allow_html=True)
        crit_map = {
            "c1_larger_than_priced":   "C1 — Larger than priced",
            "c2_consistent_revenue":   "C2 — Consistent revenue",
            "c3_net_income_improving": "C3 — Net income ↑",
            "c4_ratio_improving":      "C4 — Asset/liability ratio ↑",
            "c5_dip_entry":            "C5 — Dip entry",
            "c6_above_200ma":          "C6 — Above 200MA",
            "c7_margin_quality":       "C7 — Margin quality",
            "c8_cashflow_improving":   "C8 — Cash flow ↑",
            "c9_momentum_6m":          "C9 — 6M momentum",
        }
        pills = "".join(
            f"<div class='crit-pill {'pass' if row.get(k) else 'fail'}'>"
            f"<span>{'✓' if row.get(k) else '✗'}</span>{label}</div>"
            for k, label in crit_map.items()
        )
        st.markdown(f"<div class='criteria-grid'>{pills}</div>", unsafe_allow_html=True)

        # Key metrics
        st.markdown("<div class='section-label'>Key metrics</div>", unsafe_allow_html=True)
        km1, km2, km3, km4 = st.columns(4)
        mc  = row.get("market_cap")
        rc  = row.get("revenue_consistency")
        rgy = row.get("revenue_growth_yoy")
        amr = row.get("asset_to_mcap_ratio")

        with km1:
            v = f"${mc/1e9:.1f}bn" if mc and mc >= 1e9 else f"${mc/1e6:.0f}m" if mc else "—"
            st.markdown(metric("Market cap", v), unsafe_allow_html=True)
        with km2:
            st.markdown(metric("Revenue consistency", f"{rc:.0%}" if rc is not None else "—"), unsafe_allow_html=True)
        with km3:
            v = f"{rgy:+.1%}" if rgy is not None else "—"
            st.markdown(metric("Revenue growth YoY", v, "pos" if rgy and rgy>0 else "neg"), unsafe_allow_html=True)
        with km4:
            st.markdown(metric("Asset/mcap ratio", f"{amr:.2f}x" if amr is not None else "—"), unsafe_allow_html=True)

        # Sparkline
        st.markdown("<div class='section-label'>6-month price</div>", unsafe_allow_html=True)
        with st.spinner(f"Loading {selected}..."):
            hist = fetch_sparkline(selected)

        if not hist.is_empty() and "Close" in hist.columns:
            closes = hist["Close"].to_list()
            line_col = "#4ade80" if closes[-1] >= closes[0] else "#f87171"
            fill_rgb = "74,222,128" if closes[-1] >= closes[0] else "248,113,113"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["Date"].to_list(), y=closes,
                mode="lines",
                line=dict(color=line_col, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({fill_rgb},0.06)",
                hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>",
                showlegend=False,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
                font=dict(family="IBM Plex Mono", color="#444", size=10),
                xaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(color="#2a2a2a", size=9)),
                yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False,
                           tickfont=dict(color="#2a2a2a", size=9)),
                margin=dict(l=0, r=0, t=8, b=0),
                height=260, hovermode="x unified",
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Price history unavailable.")