# app.py — Canadian/Global Stocks MPT Website (Yahoo Finance)
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from datetime import datetime, timedelta

# -----------------------------
# Page Setup (Finance Theme)
# -----------------------------
st.set_page_config(page_title="MPT Portfolio Optimizer", page_icon="📈", layout="wide")

FINANCE_CSS = """
<style>
:root {
  --brand:#0f9d58;           /* finance green */
  --brand-2:#2bb673;         /* accent green */
}
header, .stApp { background-color:#0b0f12; }
section.main > div { padding-top: 0 !important; }
h1,h2,h3,h4,h5,h6, .stMetric { color:#e8f0ea !important; }
p, .stMarkdown, .stCaption, .stDataFrame { color:#d9e2dc !important; }
div[data-testid="stSidebar"] { background:#0e1317; border-right:1px solid #1a232a; }
.block-container { padding-top: 1rem; }
.st-emotion-cache-1wmy9hl, .st-emotion-cache-1v0mbdj { color:#d9e2dc !important; }
.css-zt5igj { color:#d9e2dc !important; }
.stButton > button, .stDownloadButton > button { background:var(--brand); color:white; border:0; border-radius:10px; padding:0.5rem 1rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--brand-2); }
hr { border-top: 1px solid #27323a; }
.kpi-card { border:1px solid #223039; border-radius:14px; padding:12px 14px; background:#0e1519; }
.kpi-title { font-size:0.9rem; color:#a8b6af; margin-bottom:4px; }
.kpi-val { font-size:1.1rem; color:#e8f0ea; }
.kpi-sub { font-size:0.85rem; color:#9fb0a7; }
.table-wrap { border:1px solid #223039; border-radius:14px; padding:8px; background:#0d1418; }
</style>
"""
st.markdown(FINANCE_CSS, unsafe_allow_html=True)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("Enter up to **5 tickers** (Yahoo Finance symbols), separated by commas:")
user_tickers_text = st.sidebar.text_input(
    "Tickers (e.g., RY.TO, TD.TO, ENB.TO, SHOP.TO, AAPL)",
    value="RY.TO, ENB.TO, FTS.TO, CNQ.TO, SHOP.TO"
)

lookback_years = st.sidebar.slider("Lookback (years)", 3, 10, 5)
risk_free = st.sidebar.number_input("Risk-free rate (annual)", value=0.03, step=0.005, format="%.3f")
long_only = st.sidebar.checkbox("Long-only (no shorting)", value=True)
max_weight = st.sidebar.slider("Per-stock max weight", 0.10, 1.00, 1.00, 0.05)

st.sidebar.caption("Tip: Use exchange suffixes (e.g., *.TO* for TSX).")

# Parse tickers
tickers = [t.strip() for t in user_tickers_text.replace(";", ",").split(",") if t.strip()]
if len(tickers) == 0:
    st.stop()
if len(tickers) > 5:
    st.sidebar.error("Please enter **at most 5** tickers.")
    st.stop()

# -----------------------------
# Data Load & Return Stats
# -----------------------------
@st.cache_data(show_spinner=True)
def load_prices(symbols, years):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))["Adj Close"]
    # Standardize to DataFrame even for single symbol
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(axis=1, how="all")
    valid = [t for t in symbols if t in df.columns]
    return df[valid], valid

prices, valid = load_prices(tickers, lookback_years)
if len(valid) == 0:
    st.error("No price data found. Check tickers/suffixes.")
    st.stop()
if len(valid) < len(tickers):
    st.warning(f"Missing data for: {sorted(set(tickers) - set(valid))}")

tickers = valid  # use only those with data
rets = np.log(prices[tickers].pct_change().add(1)).dropna()
mu = rets.mean() * 252.0         # annualized expected returns
cov = rets.cov() * 252.0         # annualized covariance
mu_vec = mu.values
cov_mat = cov.values
n = len(tickers)

# -----------------------------
# MPT Helpers
# -----------------------------
def perf(w):
    w = np.array(w)
    er = float(w @ mu_vec)
    vol = float(np.sqrt(w @ cov_mat @ w))
    return er, vol

def neg_sharpe(w):
    er, vol = perf(w)
    return 1e6 if vol == 0 else -( er - risk_free ) / vol

def min_var_for_target(target_er, bounds, base_cons):
    w0 = np.ones(n) / n
    cons = base_cons + [{"type": "eq", "fun": lambda w, tr=target_er: w @ mu_vec - tr}]
    res = minimize(lambda w: perf(w)[1]**2, w0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res

def optimize_gmv(bounds, base_cons):
    w0 = np.ones(n) / n
    res = minimize(lambda w: perf(w)[1]**2, w0, method="SLSQP",
                   bounds=bounds, constraints=base_cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res

def optimize_max_sharpe(bounds, base_cons):
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=base_cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    return res

# Bounds/constraints
if long_only:
    bounds = tuple((0.0, max_weight) for _ in range(n))
else:
    bounds = tuple((-max_weight, max_weight) for _ in range(n))
base_cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

# -----------------------------
# Key Portfolios & Frontier
# -----------------------------
w_eq = np.ones(n) / n
eq_er, eq_vol = perf(w_eq)

gmv_res = optimize_gmv(bounds, base_cons)
w_gmv = gmv_res.x; gmv_er, gmv_vol = perf(w_gmv)

ms_res = optimize_max_sharpe(bounds, base_cons)
w_ms = ms_res.x; ms_er, ms_vol = perf(w_ms)
ms_sharpe = (ms_er - risk_free) / ms_vol if ms_vol > 0 else np.nan

# Efficient Frontier
t_min = float(min(gmv_er, ms_er, np.min(mu_vec)))
t_max = float(np.percentile(mu_vec, 95))
targets = np.linspace(t_min, t_max, 50)

ef_vols, ef_ers = [], []
for tr in targets:
    r = min_var_for_target(tr, bounds, base_cons)
    if r.success:
        er, vol = perf(r.x)
        ef_ers.append(er); ef_vols.append(vol)

# -----------------------------
# Header & Per-Stock KPIs
# -----------------------------
st.title("📈 Modern Portfolio Theory — Interactive Optimizer")
st.caption("Pick up to 5 Yahoo Finance tickers. We’ll compute per-stock risk/return, "
           "the Efficient Frontier, and optimal weights (GMV & Max-Sharpe).")

k1, k2, k3 = st.columns([0.5, 0.25, 0.25])
with k1:
    st.subheader("Chosen Tickers")
    st.write(", ".join(tickers))
with k2:
    st.metric("Lookback (years)", lookback_years)
with k3:
    st.metric("Risk-free (annual)", f"{risk_free:.2%}")

st.markdown("#### Per-Stock Annualized KPIs")
row = st.columns(min(5, len(tickers)))
for i, t in enumerate(tickers):
    er = float(mu[t]); vol = float(np.sqrt(cov.loc[t, t]))
    with row[i % len(row)]:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">{t}</div>
                <div class="kpi-val">ER: {er:.2%}</div>
                <div class="kpi-sub">Risk (Vol): {vol:.2%}</div>
            </div>""",
            unsafe_allow_html=True
        )

st.markdown("---")

# -----------------------------
# Plotly Efficient Frontier (Interactive)
# -----------------------------
fig = go.Figure()
# Frontier line
fig.add_trace(go.Scatter(
    x=ef_vols, y=ef_ers, mode="lines", name="Efficient Frontier",
    line=dict(width=3, color="#2bb673"),
    hovertemplate="Vol: %{x:.2%}<br>ER: %{y:.2%}<extra></extra>"
))
# Key points
def add_point(x, y, name, color, symbol):
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[name], textposition="top center",
        name=name, marker=dict(size=14, symbol=symbol, color=color, line=dict(color="#0b0f12", width=1)),
        hovertemplate=f"{name}<br>Vol: {x:.2%}<br>ER: {y:.2%}"
    ))
add_point(eq_vol,  eq_er,  "Equal-Weight", "#4C78A8", "circle")
add_point(gmv_vol, gmv_er, "GMV (Min Risk)", "#72B7B2", "x")
add_point(ms_vol,  ms_er,  "Max-Sharpe", "#F58518", "diamond")

fig.update_layout(
    template="plotly_dark",
    title=f"Efficient Frontier — {', '.join(tickers)}",
    xaxis_title="Risk (Annualized Volatility)",
    yaxis_title="Expected Return (Annualized)",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(ticks="outside", tickformat=".1%")
fig.update_yaxes(ticks="outside", tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Optimal Weights Tables
# -----------------------------
def weights_table(name, w):
    df = pd.DataFrame({"Ticker": tickers, "Weight": w})
    er, vol = perf(w)
    summary = pd.DataFrame({
        "Portfolio":[name],
        "Expected Return (annual)":[er],
        "Risk (annual stdev)":[vol],
        "Sharpe":[(er - risk_free)/vol if vol>0 else np.nan]
    })
    return df.sort_values("Weight", ascending=False).reset_index(drop=True), summary

st.markdown("### Optimal Weights")
tab1, tab2, tab3 = st.tabs(["Equal-Weight", "GMV", "Max-Sharpe"])

with tab1:
    df_eq, sm_eq = weights_table("Equal-Weight", w_eq)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_eq, use_container_width=True)
    st.write(sm_eq)
    st.download_button("Download Equal-Weight (CSV)", df_eq.to_csv(index=False).encode(), "equal_weight.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    df_gmv, sm_gmv = weights_table("GMV", w_gmv)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_gmv, use_container_width=True)
    st.write(sm_gmv)
    st.download_button("Download GMV (CSV)", df_gmv.to_csv(index=False).encode(), "gmv_weights.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    df_ms, sm_ms = weights_table("Max-Sharpe", w_ms)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df_ms, use_container_width=True)
    st.write(sm_ms)
    st.download_button("Download Max-Sharpe (CSV)", df_ms.to_csv(index=False).encode(), "max_sharpe_weights.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Notes: Expected returns and risk are annualized from historical daily log returns. "
           "This is an educational tool and not investment advice.")
