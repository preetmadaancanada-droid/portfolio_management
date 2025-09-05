# app.py — Beautiful, mobile-friendly MPT Optimizer (percent ER & Risk)
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import minimize

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="MPT Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# Theme & CSS (finance-style, glass UI, responsive)
# =========================================================
THEME_CSS = """
<style>
:root{
  --bg:#0a0f14; --panel:#0e151b; --panel-2:#101921; --border:#1f2a33;
  --text:#e9f1f0; --muted:#9ab0b3; --brand:#23c382; --brand-2:#18a56c; --accent:#4ea1ff;
  --shadow: 0 10px 28px rgba(0,0,0,0.35), 0 2px 8px rgba(0,0,0,0.25);
}
html, body, .stApp { background: var(--bg) !important; }

.block-container{ padding-top:0.75rem; }

/* HERO */
.hero{
  border-radius:18px; padding:20px 18px;
  background: radial-gradient(1100px 600px at -10% -20%, rgba(45,182,118,0.20), transparent 60%),
              radial-gradient(800px 500px at 110% -10%, rgba(78,161,255,0.18), transparent 55%),
              linear-gradient(145deg, #0b1218 0%, #0e151b 100%);
  border:1px solid var(--border);
  box-shadow: var(--shadow);
}
.hero h1{ color:var(--text); margin:0; letter-spacing:.3px; }
.hero p{ color:var(--muted); margin:.25rem 0 0; }

/* SIDEBAR */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0e151b 0%, #0b1016 100%);
  border-right:1px solid var(--border);
}
.sidebar-card{
  background: rgba(16,25,33,0.85);
  border:1px solid var(--border);
  border-radius:14px; padding:12px;
}

/* KPI GRID (glassy cards) */
.kpi-grid{
  display:grid; gap:14px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top:10px;
}
.kpi{
  background: linear-gradient(180deg, rgba(16,25,33,0.75), rgba(16,25,33,0.55));
  backdrop-filter: blur(5px);
  border:1px solid var(--border); border-radius:16px;
  padding:14px 16px; box-shadow: var(--shadow);
  transition: transform .18s ease, border-color .18s ease;
}
.kpi:hover{ transform: translateY(-2px); border-color:#2a3a46; }
.kpi-title{ color:#a8bcc0; font-size:.95rem; margin-bottom:6px; }
.kpi-value{ color:var(--text); font-size:1.15rem; }
.kpi-sub{ color:var(--muted); font-size:.90rem; }

/* TICKER CHIP */
.ticker-chip{
  display:inline-block; padding:6px 10px; margin:4px 6px 0 0;
  background: rgba(35,195,130,0.12);
  border:1px solid rgba(35,195,130,0.35);
  color:#b7f2d7; border-radius:999px; font-size:.9rem;
}

/* TABLE WRAP */
.table-wrap{
  border:1px solid var(--border); border-radius:16px; padding:10px;
  background: linear-gradient(180deg, rgba(16,25,33,0.8), rgba(16,25,33,0.6));
  box-shadow: var(--shadow);
}

/* BUTTONS */
.stButton > button, .stDownloadButton > button{
  background: var(--brand); color:#0b120f; font-weight:600;
  border:0; border-radius:12px; padding:.55rem 1.0rem;
}
.stButton > button:hover, .stDownloadButton > button:hover{ background: var(--brand-2); }

/* PLOT SPACING */
.stPlotlyChart, .stMarkdown{ margin-bottom:.85rem; }

/* HEADINGS & TEXT */
h1,h2,h3,h4,h5,h6{ color:var(--text)!important; }
p, .stMarkdown, .stDataFrame, .stCaption{ color:var(--text)!important; }

/* Tooltips (Plotly) font size tweak */
.legendtext{ font-size: 12px !important; }

/* Breakpoints */
@media (max-width:1200px){ .kpi-grid{ grid-template-columns: repeat(3, minmax(0,1fr)); } }
@media (max-width:820px){
  .block-container{ padding-left:.6rem; padding-right:.6rem; }
  .kpi-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width:440px){ .kpi-grid{ grid-template-columns: 1fr; } }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# =========================================================
# Sidebar (controls)
# =========================================================
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown('<div class="sidebar-card">Choose up to 5 symbols. Use exchange suffixes (e.g., <b>.TO</b> for TSX).</div>', unsafe_allow_html=True)

tickers_text = st.sidebar.text_input("Tickers (comma-separated)", value="RY.TO, ENB.TO, FTS.TO, CNQ.TO, SHOP.TO")
lookback_years = st.sidebar.slider("Lookback (years)", 3, 10, 5)
risk_free = st.sidebar.number_input("Risk-free (annual)", value=0.03, step=0.005, format="%.3f")

with st.sidebar.expander("Advanced options", expanded=False):
    long_only = st.checkbox("Long-only (no shorting)", value=True)
    max_weight = st.slider("Per-stock max weight", 0.10, 1.00, 1.00, 0.05)
    debug = st.checkbox("Show debug info", value=False)

# =========================================================
# Parse tickers
# =========================================================
raw_tickers = [t.strip() for t in tickers_text.replace(";", ",").split(",") if t.strip()]
if len(raw_tickers) == 0:
    st.stop()
if len(raw_tickers) > 5:
    st.sidebar.error("Please enter at most 5 tickers.")
    st.stop()

# =========================================================
# Data load & stats
# =========================================================
@st.cache_data(show_spinner=True)
def load_prices(symbols, years):
    end = datetime.today()
    start = end - timedelta(days=int(365 * years))
    data = yf.download(symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    # Adjust handling for 1 vs many tickers
    if "Adj Close" in data.columns:
        px = data["Adj Close"]
    else:
        if isinstance(data, pd.Series):
            px = data.to_frame(name=symbols[0])
        else:
            px = data["Close"] if "Close" in data.columns else data
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(axis=1, how="all")
    valid = [t for t in symbols if t in px.columns]
    return px[valid], valid

prices, tickers = load_prices(raw_tickers, lookback_years)
if len(tickers) == 0:
    st.error("No usable price data. Check tickers/suffixes (e.g., .TO for TSX).")
    st.stop()
if len(tickers) < len(raw_tickers):
    st.warning(f"Missing data for: {sorted(set(raw_tickers) - set(tickers))}")
if len(tickers) == 1:
    st.error("Need at least 2 tickers to build a portfolio.")
    st.stop()

rets = np.log(prices[tickers].pct_change().add(1)).dropna()
if rets.empty:
    st.error("Not enough historical data for the selected window. Try a longer lookback or different tickers.")
    st.stop()

mu = rets.mean() * 252.0          # annual expected returns (vector)
cov = rets.cov() * 252.0          # annual covariance matrix
mu_vec = mu.values
cov_mat = cov.values
n = len(tickers)

# =========================================================
# MPT helpers
# =========================================================
def perf(w):
    w = np.array(w, dtype=float)
    er = float(w @ mu_vec)
    vol = float(np.sqrt(w @ cov_mat @ w))
    return er, vol

def neg_sharpe(w):
    er, vol = perf(w)
    return 1e6 if vol == 0 else -(er - risk_free) / vol

# Bounds/constraints
if long_only:
    bounds = tuple((0.0, float(max_weight)) for _ in range(n))
else:
    bounds = tuple((-float(max_weight), float(max_weight)) for _ in range(n))
base_cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
w0 = np.full(n, 1.0 / n)

def safe_minimize(fun, w0, bounds, cons, label):
    try:
        res = minimize(fun, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-9})
        if not res.success:
            raise RuntimeError(res.message)
        return res.x
    except Exception as e:
        if debug:
            st.warning(f"{label} optimizer fallback: {e}")
        return w0

# Key portfolios
w_eq = w0.copy()
eq_er, eq_vol = perf(w_eq)

w_gmv = safe_minimize(lambda w: perf(w)[1]**2, w0, bounds, base_cons, "GMV")
gmv_er, gmv_vol = perf(w_gmv)

w_ms = safe_minimize(neg_sharpe, w0, bounds, base_cons, "Max Sharpe")
ms_er, ms_vol = perf(w_ms)
ms_sharpe = (ms_er - risk_free) / ms_vol if ms_vol > 0 else np.nan

# Efficient Frontier
t_min = float(min(gmv_er, ms_er, np.min(mu_vec)))
t_max = float(np.percentile(mu_vec, 95))
targets = np.linspace(t_min, t_max, 56)
ef_vols, ef_ers = [], []
for tr in targets:
    cons = base_cons + [{"type": "eq", "fun": lambda w, tr=tr: w @ mu_vec - tr}]
    w_tr = safe_minimize(lambda w: perf(w)[1]**2, w0, bounds, cons, "Frontier")
    er, vol = perf(w_tr)
    ef_ers.append(er); ef_vols.append(vol)

# =========================================================
# HERO header
# =========================================================
st.markdown(
    f"""
    <div class="hero">
      <h1>Modern Portfolio Theory — Optimizer</h1>
      <p>Select up to five Yahoo Finance tickers, explore the Efficient Frontier, and view optimal weights (GMV & Max-Sharpe).</p>
      <div style="margin-top:.5rem">
        {"".join([f'<span class="ticker-chip">{t}</span>' for t in tickers])}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Per-stock KPIs (percent formatted)
# =========================================================
st.markdown("#### Per-Stock Annualized KPIs")
kpi_html = ['<div class="kpi-grid">']
for t in tickers:
    er = float(mu[t])
    vol = float(np.sqrt(cov.loc[t, t]))
    kpi_html.append(
        f"""<div class="kpi">
              <div class="kpi-title">{t}</div>
              <div class="kpi-value">ER: {er:.2%}</div>
              <div class="kpi-sub">Risk (Vol): {vol:.2%}</div>
            </div>"""
    )
kpi_html.append("</div>")
st.markdown("\n".join(kpi_html), unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# Plotly chart (styled)
# =========================================================
fig = go.Figure()

# Frontier line
fig.add_trace(go.Scatter(
    x=ef_vols, y=ef_ers, mode="lines", name="Efficient Frontier",
    line=dict(width=3, color="#23c382"),
    hovertemplate="Vol: %{x:.2%}<br>ER: %{y:.2%}<extra></extra>"
))

# Add key points
def add_point(x, y, name, color, symbol):
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[name], textposition="top center",
        name=name, marker=dict(size=14, symbol=symbol, color=color,
                               line=dict(color="#0b1218", width=1)),
        hovertemplate=f"{name}<br>Vol: {x:.2%}<br>ER: {y:.2%}<extra></extra>"
    ))

add_point(eq_vol,  eq_er,  "Equal-Weight", "#4C78A8", "circle")
add_point(gmv_vol, gmv_er, "GMV (Min Risk)", "#72B7B2", "x")
add_point(ms_vol,  ms_er,  "Max-Sharpe", "#F58518", "diamond")

fig.update_layout(
    template="plotly_dark",
    title=f"Efficient Frontier — {', '.join(tickers)}",
    xaxis_title="Risk (Annualized Volatility)",
    yaxis_title="Expected Return (Annualized)",
    margin=dict(l=8, r=8, t=35, b=8),
    legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    height=440,
)
fig.update_xaxes(ticks="outside", tickformat=".1%", gridcolor="#253341", zerolinecolor="#2b3b46")
fig.update_yaxes(ticks="outside", tickformat=".1%", gridcolor="#253341", zerolinecolor="#2b3b46")

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# =========================================================
# Weights tables (percent display; numeric CSV)
# =========================================================
def weights_tables(name, w):
    er, vol = perf(w)
    df_num = pd.DataFrame({"Ticker": tickers, "Weight": w})
    df_ui = df_num.copy()
    df_ui["Weight"] = (df_ui["Weight"] * 100).map(lambda x: f"{x:.2f}%")
    summary_ui = pd.DataFrame({
        "Portfolio":[name],
        "Expected Return (annual)":[f"{er:.2%}"],
        "Risk (annual stdev)":[f"{vol:.2%}"],
        "Sharpe":[f"{((er - risk_free)/vol):.2f}" if vol>0 else "—"]
    })
    return df_ui.sort_values("Ticker").reset_index(drop=True), df_num, summary_ui

st.markdown("### Optimal Weights")
t1, t2, t3 = st.tabs(["Equal-Weight", "GMV", "Max-Sharpe"])

with t1:
    df_ui, df_num, sm_ui = weights_tables("Equal-Weight", w_eq)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_ui, use_container_width=True, height=360)
    st.write(sm_ui)
    st.download_button("Download Equal-Weight (CSV)", df_num.to_csv(index=False).encode(), "equal_weight.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with t2:
    df_ui, df_num, sm_ui = weights_tables("GMV", w_gmv)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_ui, use_container_width=True, height=360)
    st.write(sm_ui)
    st.download_button("Download GMV (CSV)", df_num.to_csv(index=False).encode(), "gmv_weights.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with t3:
    df_ui, df_num, sm_ui = weights_tables("Max-Sharpe", w_ms)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_ui, use_container_width=True, height=360)
    st.write(sm_ui)
    st.download_button("Download Max-Sharpe (CSV)", df_num.to_csv(index=False).encode(), "max_sharpe_weights.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Debug (optional)
# =========================================================
if debug:
    st.write("Tickers:", tickers)
    st.write("mu (annual ER):", mu)
    st.write("cov (annual):", cov)

st.caption("Notes: ER and Risk are annualized from historical daily log returns. "
           "Percentages are for display; CSV exports are numeric. Educational use only.")

