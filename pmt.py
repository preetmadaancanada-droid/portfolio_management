# app.py — Minimal, hardened MPT app for Streamlit Cloud
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import minimize

st.set_page_config(page_title="MPT Portfolio Optimizer", page_icon="📈", layout="wide")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Controls")
tickers_text = st.sidebar.text_input(
    "Enter up to 5 Yahoo tickers (comma-separated)",
    value="RY.TO, ENB.TO, FTS.TO, CNQ.TO, SHOP.TO",
)
lookback_years = st.sidebar.slider("Lookback (years)", 3, 10, 5)
risk_free = st.sidebar.number_input("Risk-free (annual)", value=0.03, step=0.005, format="%.3f")
long_only = st.sidebar.checkbox("Long-only", value=True)
max_weight = st.sidebar.slider("Max weight per stock", 0.10, 1.00, 1.00, 0.05)
debug = st.sidebar.checkbox("Show debug info", value=False)

# Parse tickers
raw = [t.strip() for t in tickers_text.replace(";", ",").split(",") if t.strip()]
if len(raw) == 0:
    st.stop()
if len(raw) > 5:
    st.sidebar.error("Please enter at most 5 tickers.")
    st.stop()

# -----------------------------
# Data load (robust)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_prices(symbols, years):
    end = datetime.today()
    start = end - timedelta(days=int(365 * years))
    data = yf.download(symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    # Handle both 1 and many tickers
    if "Adj Close" in data.columns:
        px = data["Adj Close"]
    else:
        # Some yfinance versions return a Series for single tickers
        if isinstance(data, pd.Series):
            px = data.to_frame(name=symbols[0])
        else:
            # Last resort: try Close
            px = data["Close"] if "Close" in data.columns else data
    # Ensure DataFrame
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(axis=1, how="all")
    valid = [t for t in symbols if t in px.columns]
    return px[valid], valid

prices, tickers = load_prices(raw, lookback_years)
if len(tickers) == 0:
    st.error("No usable price data. Check tickers/suffixes (e.g., .TO for TSX).")
    st.stop()
if len(tickers) < len(raw):
    st.warning(f"Missing data for: {sorted(set(raw) - set(tickers))}")

# Need at least 2 tickers to form a frontier
if len(tickers) == 1:
    st.error("Need at least 2 tickers. Add one or more symbols.")
    st.stop()

# Returns & stats (annualized)
rets = np.log(prices[tickers].pct_change().add(1)).dropna()
if rets.empty:
    st.error("Not enough historical data for selected window. Try a longer lookback or different tickers.")
    st.stop()

mu = rets.mean() * 252.0               # expected returns
cov = rets.cov() * 252.0               # covariance
mu_vec = mu.values
cov_mat = cov.values
n = len(tickers)

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

# Optimizers with safety nets
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
        # Fallback to equal weights if optimizer fails
        return w0

# Global Min Variance
w_gmv = safe_minimize(lambda w: perf(w)[1]**2, w0, bounds, base_cons, "GMV")
gmv_er, gmv_vol = perf(w_gmv)

# Max Sharpe
w_ms = safe_minimize(neg_sharpe, w0, bounds, base_cons, "Max Sharpe")
ms_er, ms_vol = perf(w_ms)

# Equal-weight (baseline)
w_eq = w0.copy()
eq_er, eq_vol = perf(w_eq)

# Efficient frontier (target-return sweep)
t_min = float(min(gmv_er, ms_er, np.min(mu_vec)))
t_max = float(np.percentile(mu_vec, 95))
targets = np.linspace(t_min, t_max, 40)

ef_vols, ef_ers = [], []
for tr in targets:
    cons = base_cons + [{"type": "eq", "fun": lambda w, tr=tr: w @ mu_vec - tr}]
    w_tr = safe_minimize(lambda w: perf(w)[1]**2, w0, bounds, cons, "Frontier")
    er, vol = perf(w_tr)
    ef_ers.append(er); ef_vols.append(vol)

# -----------------------------
# Header & stock KPIs
# -----------------------------
st.title("📈 Modern Portfolio Theory — Optimizer")
sub_l, sub_r1, sub_r2 = st.columns([0.6, 0.2, 0.2])
with sub_l:
    st.caption("Pick up to 5 tickers. We’ll compute per-stock ER/Risk, the Efficient Frontier, and optimal weights.")
with sub_r1:
    st.metric("Lookback (yrs)", lookback_years)
with sub_r2:
    st.metric("Risk-free (ann.)", f"{risk_free:.2%}")

st.markdown("#### Per-Stock Annualized KPIs")
cols = st.columns(min(5, len(tickers)))
for i, t in enumerate(tickers):
    er = float(mu[t]); vol = float(np.sqrt(cov.loc[t, t]))
    with cols[i % len(cols)]:
        st.markdown(
            f"<div style='border:1px solid #223039;border-radius:12px;padding:10px;background:#0e1519'>"
            f"<div style='color:#a8b6af;font-size:0.9rem'>{t}</div>"
            f"<div style='color:#e8f0ea'>ER: {er:.2%}</div>"
            f"<div style='color:#9fb0a7'>Risk: {vol:.2%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# -----------------------------
# Plotly chart
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ef_vols, y=ef_ers, mode="lines", name="Efficient Frontier",
    line=dict(width=3, color="#2bb673"),
    hovertemplate="Vol: %{x:.2%}<br>ER: %{y:.2%}<extra></extra>"
))
def add_point(x, y, name, color, symbol):
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[name], textposition="top center",
        name=name, marker=dict(size=14, symbol=symbol, color=color, line=dict(color='#0b0f12', width=1)),
        hovertemplate=f"{name}<br>Vol: {x:.2%}<br>ER: {y:.2%}<extra></extra>"
    ))
add_point(eq_vol,  eq_er,  "Equal-Weight", "#4C78A8", "circle")
add_point(gmv_vol, gmv_er, "GMV", "#72B7B2", "x")
add_point(ms_vol,  ms_er,  "Max Sharpe", "#F58518", "diamond")

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
# Weights tables
# -----------------------------
def weights_table(name, w):
    df = pd.DataFrame({"Ticker": tickers, "Weight": w}).sort_values("Weight", ascending=False).reset_index(drop=True)
    er, vol = perf(w)
    summary = pd.DataFrame({
        "Portfolio":[name],
        "Expected Return (annual)":[er],
        "Risk (annual stdev)":[vol],
        "Sharpe":[(er - risk_free)/vol if vol>0 else np.nan]
    })
    return df, summary

st.markdown("### Optimal Weights")
t1, t2, t3 = st.tabs(["Equal-Weight", "GMV", "Max-Sharpe"])
for (tab, name, w) in [(t1,"Equal-Weight",w_eq),(t2,"GMV",w_gmv),(t3,"Max-Sharpe",w_ms)]:
    with tab:
        df, sm = weights_table(name, w)
        st.dataframe(df, use_container_width=True)
        st.write(sm)
        st.download_button(f"Download {name} (CSV)", df.to_csv(index=False).encode(), f"{name.lower().replace(' ','_')}.csv", "text/csv")

if debug:
    st.write("Tickers:", tickers)
    st.write("mu (annual ER):", mu)
    st.write("cov (annual):", cov)
