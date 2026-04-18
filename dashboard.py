import streamlit as st
import polars as pl
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────


YF_TICKER_MAP = {"INRG": "INRG.L"}


st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0a0a; }
section[data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }

.metric-card {
    background: #111; border: 1px solid #222;
    border-radius: 4px; padding: 16px 20px; margin-bottom: 8px;
}
.metric-label {
    font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #555; font-family: 'IBM Plex Mono', monospace; margin-bottom: 6px;
}
.metric-value {
    font-size: 22px; font-weight: 500;
    font-family: 'IBM Plex Mono', monospace; color: #f0f0f0;
}
.metric-value.positive { color: #4ade80; }
.metric-value.negative { color: #f87171; }
.metric-sub { font-size: 11px; color: #444; font-family: 'IBM Plex Mono', monospace; margin-top: 4px; }

.ticker-pill {
    display: inline-block; background: #1a1a1a; border: 1px solid #333;
    border-radius: 3px; padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #aaa; margin: 2px;
}
.trade-row {
    background: #111; border: 1px solid #1e1e1e; border-radius: 3px;
    padding: 10px 14px; margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
}
.buy-label  { color: #4ade80; }
.sell-label { color: #f87171; }



.mode-banner {
    border-radius: 4px; padding: 10px 16px; margin-bottom: 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
}
.sim-banner  { background: #1a1200; border: 1px solid #3d2e00; color: #f59e0b; }
.live-banner { background: #0a1a0a; border: 1px solid #1a3a1a; color: #4ade80; }

.section-header {
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #444; font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #1e1e1e; padding-bottom: 8px;
    margin-bottom: 16px; margin-top: 24px;
}
h1, h2, h3 { color: #f0f0f0 !important; }
.stSelectbox label, .stRadio label, .stDateInput label, .stToggle label {
    color: #888 !important; font-size: 12px !important;
}
div[data-testid="stMarkdownContainer"] p { color: #888; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_m(val) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
    except Exception:
        return "—"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}bn"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}m"
    return f"${v:,.0f}"

    try:
        return f"{float(val)*100:.1f}%"
    except Exception:
        return "—"

def pct_css(v) -> str:
    try:
        return "positive" if float(v) > 0 else "negative"
    except Exception:
        return ""

def metric_card(label: str, value: str, css: str = "", sub: str = "") -> str:
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value {css}'>{value}</div>"
        f"{sub_html}</div>"
    )


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_trades() -> pl.DataFrame:
    csv_files = [
        "data/y1.csv",
        "data/y2.csv",
    ]
    frames = []
    for path in csv_files:
        if os.path.exists(path):
            frames.append(pl.read_csv(path, infer_schema_length=500))
    if not frames:
        st.error("CSV files not found. Place both trading CSVs in the same directory as this script.")
        st.stop()

    return (
        pl.concat(frames, how="diagonal")
        .with_columns(pl.col("Time").str.to_datetime("%Y-%m-%d %H:%M:%S"))
        .filter(pl.col("Action").is_in(["Market buy", "Market sell", "Limit buy", "Limit sell"]))
        .with_columns(pl.col("Action").str.to_lowercase().str.contains("buy").alias("is_buy"))
        .sort("Time")
    )


@st.cache_data(ttl=3600)
def fetch_price_history(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Fetch OHLCV. end is exclusive — pass the day after the last day you want."""
    yf_ticker = YF_TICKER_MAP.get(ticker, ticker)
    try:
        hist_pd = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist_pd.empty and yf_ticker != ticker:
            hist_pd = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist_pd.empty:
            return pl.DataFrame()
        if isinstance(hist_pd.columns, pd.MultiIndex):
            hist_pd.columns = hist_pd.columns.get_level_values(0)
        hist_pd = hist_pd.reset_index()
        hist_pd.columns = [str(c) for c in hist_pd.columns]
        return pl.from_pandas(hist_pd)
    except Exception as e:
        st.warning(f"Could not fetch price data for {ticker}: {e}")
        return pl.DataFrame()




def get_open_positions(df: pl.DataFrame) -> dict[str, float]:
    net = (
        df
        .with_columns(
            pl.when(pl.col("is_buy"))
              .then(pl.col("No. of shares"))
              .otherwise(-pl.col("No. of shares"))
              .alias("signed_shares")
        )
        .group_by("Ticker")
        .agg(pl.col("signed_shares").sum().alias("net_shares"))
        .filter(pl.col("net_shares") > 0.001)
    )
    return dict(zip(net["Ticker"].to_list(), net["net_shares"].to_list()))


def get_ticker_trades(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    return df.filter(pl.col("Ticker") == ticker).sort("Time")


def price_on(hist: pl.DataFrame, target: datetime) -> float | None:
    """Closest available closing price on or before target date."""
    if hist.is_empty() or "Close" not in hist.columns:
        return None
    ts = pl.datetime(target.year, target.month, target.day)
    subset = hist.filter(pl.col("Date") <= ts)
    if subset.is_empty():
        return None
    return float(subset["Close"].tail(1).item())


# ── Bootstrap ─────────────────────────────────────────────────────────────────

trades_df    = load_trades()
open_pos     = get_open_positions(trades_df)
all_tickers  = sorted(trades_df["Ticker"].unique().to_list())
open_tickers = sorted(open_pos.keys())


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Portfolio")
    st.markdown("---")

    view_mode   = st.radio("Show positions", ["Open only", "All traded"], index=0)
    ticker_list = open_tickers if view_mode == "Open only" else all_tickers
    selected_ticker = st.selectbox("Select ticker", ticker_list)

    st.markdown("---")

    # Mode toggle
    sim_mode = st.toggle("Simulation mode", value=False)

    # In sim mode, the "anchor date" is the buy date by default but user can scrub it
    ticker_trades_tmp = get_ticker_trades(trades_df, selected_ticker)
    first_buy_tmp     = ticker_trades_tmp.filter(pl.col("is_buy"))["Time"].min()
    today             = datetime.today()

    if sim_mode:
        st.caption("Select a buy date — chart shows market conditions up to that moment.")
        buys_tmp = ticker_trades_tmp.filter(pl.col("is_buy")).sort("Time")
        buy_date_options = {}
        for row in buys_tmp.iter_rows(named=True):
            lbl = f"{row['Time'].strftime('%d %b %Y')}  ·  {row['Price / share']:.2f} {row['Currency (Price / share)']}  ·  £{row['Total']:.0f}"
            buy_date_options[lbl] = row["Time"]
        selected_label = st.selectbox("Buy date", list(buy_date_options.keys()))
        anchor_dt = datetime.combine(buy_date_options[selected_label].date(), datetime.max.time())
    else:
        anchor_dt = today

    st.markdown("---")
    st.markdown("**Chart lookback**")
    lookback_options = {
        "1 month":  30,
        "3 months": 90,
        "6 months": 180,
        "1 year":   365,
        "2 years":  730,
        "Max":      1825,
    }
    lookback_days = lookback_options[st.selectbox("Lookback", list(lookback_options.keys()), index=2)]

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:#444;font-family:IBM Plex Mono'>"
        f"Open: {len(open_tickers)} &nbsp;·&nbsp; All: {len(all_tickers)}</div>",
        unsafe_allow_html=True,
    )


# ── Per-ticker setup ──────────────────────────────────────────────────────────

ticker_trades = get_ticker_trades(trades_df, selected_ticker)
ticker_name   = ticker_trades["Name"].head(1).item()

buys  = ticker_trades.filter(pl.col("is_buy"))
sells = ticker_trades.filter(~pl.col("is_buy"))

first_buy_dt: datetime = buys["Time"].min()
is_open        = selected_ticker in open_pos
net_shares     = open_pos.get(selected_ticker, 0.0)
total_invested = float(buys["Total"].sum())
total_proceeds = float(sells["Total"].sum()) if sells.height > 0 else 0.0
entry_price_local = float(buys["Price / share"].head(1).item())
entry_currency    = buys["Currency (Price / share)"].head(1).item()

# Chart window: from (anchor - lookback) up to anchor
chart_end   = anchor_dt
chart_start = anchor_dt - timedelta(days=lookback_days)

# Fetch history — always end at anchor, look back by selected amount
hist = fetch_price_history(
    selected_ticker,
    chart_start.strftime("%Y-%m-%d"),
    (chart_end + timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
)

# In sim mode only show trades that had occurred by the anchor date
visible_trades = ticker_trades.filter(pl.col("Time") <= anchor_dt)
visible_buys   = visible_trades.filter(pl.col("is_buy"))
visible_sells  = visible_trades.filter(~pl.col("is_buy"))

# Returns relative to anchor date
price_at_anchor = price_on(hist, anchor_dt)
price_at_start  = price_on(hist, chart_start) if not hist.is_empty() else None

period_return = None
if price_at_anchor and price_at_start and price_at_start > 0:
    period_return = (price_at_anchor - price_at_start) / price_at_start * 100

# YTD return (Jan 1 of anchor year → anchor date)
ytd_start  = datetime(anchor_dt.year, 1, 1)
ytd_hist   = fetch_price_history(
    selected_ticker,
    ytd_start.strftime("%Y-%m-%d"),
    (anchor_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
)
ytd_return = None
if not ytd_hist.is_empty() and "Close" in ytd_hist.columns:
    ytd_p0 = float(ytd_hist["Close"].head(1).item())
    ytd_p1 = price_on(ytd_hist, anchor_dt)
    if ytd_p0 and ytd_p1 and ytd_p0 > 0:
        ytd_return = (ytd_p1 - ytd_p0) / ytd_p0 * 100


# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(f"# {ticker_name}")
    st.markdown(f"<span class='ticker-pill'>{selected_ticker}</span>", unsafe_allow_html=True)
with col_status:
    color = "#4ade80" if is_open else "#888"
    label = "OPEN" if is_open else "CLOSED"
    st.markdown(
        f"<div style='text-align:right;margin-top:16px;font-family:IBM Plex Mono;"
        f"font-size:12px;color:{color};letter-spacing:0.1em'>{label}</div>",
        unsafe_allow_html=True,
    )

if sim_mode:
    st.markdown(
        f"<div class='mode-banner sim-banner'>"
        f"⏱ Simulation mode &nbsp;·&nbsp; Anchor: <strong>{anchor_dt.strftime('%d %b %Y')}</strong>"
        f" &nbsp;·&nbsp; Showing {lookback_days}d of history leading up to this date"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='mode-banner live-banner'>"
        f"● Live mode &nbsp;·&nbsp; Showing last {lookback_days}d up to today"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Metric cards ──────────────────────────────────────────────────────────────

price_label = f"Price on {anchor_dt.strftime('%d %b %Y')}" if sim_mode else "Current price"
price_str   = f"{price_at_anchor:.2f} {entry_currency}" if price_at_anchor else f"{entry_price_local:.2f} {entry_currency}"

def interval_return(days: int) -> float | None:
    """Return % change from (anchor - days) to anchor using the already-fetched full history."""
    if hist.is_empty() or "Close" not in hist.columns:
        return None
    p_end   = price_on(hist, anchor_dt)
    p_start = price_on(hist, anchor_dt - timedelta(days=days))
    if p_end and p_start and p_start > 0:
        return (p_end - p_start) / p_start * 100
    return None

r1d  = interval_return(1)
r1w  = interval_return(7)
r1m  = interval_return(30)
r3m  = interval_return(90)
r1y  = interval_return(365)

# Row 1 — interval returns
st.markdown("<div class='section-header'>Returns from anchor date</div>", unsafe_allow_html=True)
i1, i2, i3, i4, i5, i6, i7 = st.columns(7)
with i1:
    v = f"{r1d:+.2f}%" if r1d is not None else "—"
    st.markdown(metric_card("1 day", v, pct_css(r1d or 0), "prior day → anchor"), unsafe_allow_html=True)
with i2:
    v = f"{r1w:+.2f}%" if r1w is not None else "—"
    st.markdown(metric_card("1 week", v, pct_css(r1w or 0), "7d → anchor"), unsafe_allow_html=True)
with i3:
    v = f"{r1m:+.2f}%" if r1m is not None else "—"
    st.markdown(metric_card("1 month", v, pct_css(r1m or 0), "30d → anchor"), unsafe_allow_html=True)
with i4:
    v = f"{r3m:+.2f}%" if r3m is not None else "—"
    st.markdown(metric_card("3 months", v, pct_css(r3m or 0), "90d → anchor"), unsafe_allow_html=True)
with i5:
    v = f"{r1y:+.2f}%" if r1y is not None else "—"
    st.markdown(metric_card("1 year", v, pct_css(r1y or 0), "365d → anchor"), unsafe_allow_html=True)
with i6:
    v = f"{ytd_return:+.2f}%" if ytd_return is not None else "—"
    sub = f"Jan 1 → {anchor_dt.strftime('%d %b')}"
    st.markdown(metric_card("YTD", v, pct_css(ytd_return or 0), sub), unsafe_allow_html=True)
with i7:
    st.markdown(metric_card(price_label, price_str), unsafe_allow_html=True)

# Row 2 — position info
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    v = f"{period_return:+.1f}%" if period_return is not None else "—"
    st.markdown(metric_card(f"{lookback_days}d lookback return", v, pct_css(period_return or 0), f"chart window change"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Total invested", f"£{total_invested:,.2f}", sub=f"{buys.height} buy(s)"), unsafe_allow_html=True)
with m3:
    net_cash_display = total_proceeds - total_invested
    st.markdown(metric_card("Net cash flow", f"£{net_cash_display:+,.2f}", pct_css(net_cash_display)), unsafe_allow_html=True)
with m4:
    shares_display = float(visible_buys["No. of shares"].sum()) - float(visible_sells["No. of shares"].sum() if visible_sells.height > 0 else 0)
    st.markdown(metric_card("Shares at anchor", f"{shares_display:.4f}"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Price chart ───────────────────────────────────────────────────────────────

st.markdown("<div class='section-header'>Price history</div>", unsafe_allow_html=True)

if not hist.is_empty() and "Close" in hist.columns:
    dates  = hist["Date"].to_list()
    closes = hist["Close"].to_list()

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode="lines", name="Price",
        line=dict(color="#e0e0e0", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
    ))
    # Fill under line
    fig.add_trace(go.Scatter(
        x=dates, y=closes, fill="tozeroy",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    def add_markers(rows_df: pl.DataFrame, is_buy_flag: bool):
        symbol  = "triangle-up"   if is_buy_flag else "triangle-down"
        color   = "#4ade80"       if is_buy_flag else "#f87171"
        label   = "BUY"           if is_buy_flag else "SELL"
        textpos = "top center"    if is_buy_flag else "bottom center"

        for row in rows_df.iter_rows(named=True):
            trade_dt = row["Time"]
            # Only plot if trade is within the chart window
            if trade_dt < chart_start or trade_dt > chart_end:
                continue
            p = price_on(hist, trade_dt)
            if p is None:
                continue
            ts = pl.datetime(trade_dt.year, trade_dt.month, trade_dt.day)
            closest = hist.filter(pl.col("Date") <= ts)
            if closest.is_empty():
                continue
            plot_date = closest["Date"].tail(1).item()
            fig.add_trace(go.Scatter(
                x=[plot_date], y=[p],
                mode="markers+text",
                marker=dict(color=color, size=10, symbol=symbol),
                text=[f"{label} £{row['Total']:.0f}"],
                textposition=textpos,
                textfont=dict(size=10, color=color),
                name=f"{label} {trade_dt.strftime('%d %b %y')}",
                hovertemplate=(
                    f"{label} — {trade_dt.strftime('%d %b %Y')}<br>"
                    f"£{row['Total']:.2f} @ {row['Price / share']:.2f} "
                    f"{row['Currency (Price / share)']}<extra></extra>"
                ),
            ))

    add_markers(visible_buys, True)
    add_markers(visible_sells, False)

    # Vertical line at anchor date in sim mode
    if sim_mode and price_at_anchor:
        anchor_ms = int(anchor_dt.timestamp() * 1000)
        fig.add_vline(
            x=anchor_ms,
            line=dict(color="#f59e0b", width=1, dash="dot"),
            annotation_text="anchor",
            annotation_font=dict(color="#f59e0b", size=10),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d0d",
        font=dict(family="IBM Plex Mono", color="#666", size=11),
        xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False),
                   tickfont=dict(color="#444", size=10), color="#333"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=False,
                   tickfont=dict(color="#444", size=10), color="#333"),
        legend=dict(font=dict(size=10, color="#555"), bgcolor="rgba(0,0,0,0)",
                    bordercolor="#222", borderwidth=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    if "Volume" in hist.columns and hist["Volume"].sum() > 0:
        st.markdown("<div class='section-header'>Volume</div>", unsafe_allow_html=True)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=dates, y=hist["Volume"].to_list(),
            marker_color="#1e1e1e", marker_line_width=0,
            hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:,.0f}<extra></extra>",
        ))
        fig_vol.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d0d",
            font=dict(family="IBM Plex Mono", color="#444", size=10),
            xaxis=dict(showgrid=False, tickfont=dict(color="#333", size=9)),
            yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#333", size=9)),
            margin=dict(l=0, r=0, t=0, b=0), height=140, showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.info("No price data available for this ticker in the selected range.")


# ── Trade log ─────────────────────────────────────────────────────────────────

st.markdown("<div class='section-header'>Trade history</div>", unsafe_allow_html=True)

for row in visible_trades.iter_rows(named=True):
    label     = "BUY" if row["is_buy"] else "SELL"
    label_cls = "buy-label" if row["is_buy"] else "sell-label"
    st.markdown(f"""
    <div class='trade-row'>
        <span class='{label_cls}'>{label}</span>
        &nbsp;&nbsp;{row['Time'].strftime('%d %b %Y')}
        &nbsp;&nbsp;·&nbsp;&nbsp;{row['No. of shares']:.4f} shares
        @ {row['Price / share']:.2f} {row['Currency (Price / share)']}
        &nbsp;&nbsp;·&nbsp;&nbsp;£{row['Total']:.2f}
    </div>""", unsafe_allow_html=True)


# ── Position summary (live mode only) ────────────────────────────────────────

if is_open and not sim_mode:
    st.markdown("<div class='section-header'>Position summary</div>", unsafe_allow_html=True)
    net_cash = total_proceeds - total_invested
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("Shares held", f"{net_shares:.4f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Buys / sells", f"{buys.height} / {sells.height}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Net cash flow", f"£{net_cash:+,.2f}", pct_css(net_cash)), unsafe_allow_html=True)