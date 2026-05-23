"""
Portfolio Optimization Dashboard — Bloomberg-style
Streamlit + Plotly  |  Run: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from portfolio_optimizer import (
    NIFTY50_TICKERS, SECTOR_MAP, ESG_SCORES,
    fetch_data, compute_returns,
    MarkowitzOptimizer, BlackLittermanModel,
    build_constrained_optimizer, compare_to_benchmark
)

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer | Goldman Sachs Style",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0A0E1A; color: #E2E8F0; }
  [data-testid="stSidebar"] { background-color: #0D1221 !important; border-right: 1px solid #1E2A3A; }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="metric-container"] {
    background: #111827; border: 1px solid #1E2A3A;
    border-radius: 8px; padding: 12px 16px;
  }
  [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.6rem !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
  [data-testid="stMetricDelta"] svg { display: none; }
  h1, h2, h3 { color: #F1F5F9 !important; }
  hr { border-color: #1E2A3A !important; }
  [data-baseweb="tab-list"] { background: #111827 !important; border-radius: 8px; }
  [data-baseweb="tab"] { color: #64748B !important; }
  [aria-selected="true"] { color: #3B82F6 !important; border-bottom: 2px solid #3B82F6 !important; }
  .section-tag {
    display: inline-block; font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #3B82F6; margin-bottom: 12px;
  }
  .ticker-row { display: flex; gap: 24px; padding: 8px 0; border-bottom: 1px solid #1E2A3A; margin-bottom: 20px; flex-wrap: wrap; }
  .ticker-item { font-size: 12px; color: #94A3B8; }
  .ticker-item b { color: #F1F5F9; margin-right: 4px; }
  .ticker-up { color: #10B981 !important; }
  .ticker-dn { color: #EF4444 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ────────────────────────────────────────────────
def apply_dark_theme(fig, title_text, xaxis_title="", yaxis_title="", height=480, extra=None):
    """Apply Bloomberg dark theme to any Plotly figure cleanly."""
    fig.update_layout(
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0D1221",
        font=dict(color="#94A3B8", size=12),
        legend=dict(bgcolor="#111827", bordercolor="#1E2A3A", borderwidth=1),
        title=dict(text=title_text, font=dict(size=15, color="#F1F5F9")),
        height=height,
        **(extra or {})
    )
    fig.update_xaxes(
        gridcolor="#1E2A3A", zerolinecolor="#1E2A3A",
        title_text=xaxis_title, title_font=dict(color="#94A3B8")
    )
    fig.update_yaxes(
        gridcolor="#1E2A3A", zerolinecolor="#1E2A3A",
        title_text=yaxis_title, title_font=dict(color="#94A3B8")
    )
    return fig

def chart(fig):
    """Render plotly chart — compatible with all Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")
    risk_free     = st.slider("Risk-Free Rate (India 10Y)", 0.04, 0.10, 0.065, 0.005, format="%.3f")
    risk_aversion = st.slider("Market Risk Aversion (δ)", 1.0, 5.0, 2.5, 0.5)
    tau           = st.slider("BL Prior Uncertainty (τ)", 0.01, 0.20, 0.05, 0.01)
    max_sector    = st.slider("Max Sector Weight", 0.20, 0.60, 0.40, 0.05, format="%.0%%")
    min_esg       = st.slider("Min ESG Score", 50, 90, 65, 5)
    st.markdown("---")
    st.markdown("### 📐 BL Views")
    view1_ret    = st.slider("TCS Target Return", 0.05, 0.30, 0.12, 0.01, format="%.0%%")
    view2_spread = st.slider("INFY vs SBIN Spread", 0.01, 0.15, 0.05, 0.01, format="%.0%%")
    st.markdown("---")
    run = st.button("▶  Run Optimization", use_container_width=True, type="primary")

# ─── HEADER ─────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:16px; margin-bottom:8px;'>
  <div style='background:#1D4ED8; border-radius:8px; padding:8px 14px; font-size:20px; font-weight:700; letter-spacing:0.05em;'>GS</div>
  <div>
    <div style='font-size:22px; font-weight:600; color:#F1F5F9;'>Portfolio Optimization Engine</div>
    <div style='font-size:12px; color:#475569;'>Markowitz MVO · Black-Litterman · ESG Constraints · NIFTY 50 Benchmark</div>
  </div>
</div><hr>
""", unsafe_allow_html=True)

# ─── DATA ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching market data from NSE...")
def load_data():
    prices  = fetch_data(NIFTY50_TICKERS)
    returns = compute_returns(prices)
    return prices, returns

prices, returns = load_data()

# Ticker strip
last_ret  = returns.iloc[-1]
strip_html = "<div class='ticker-row'>"
for t in NIFTY50_TICKERS[:10]:
    if t in last_ret.index:
        r   = float(last_ret[t]) * 100
        cls = "ticker-up" if r >= 0 else "ticker-dn"
        arrow = "▲" if r >= 0 else "▼"
        strip_html += f"<div class='ticker-item'><b>{t.replace('.NS','')}</b><span class='{cls}'>{arrow} {abs(r):.2f}%</span></div>"
strip_html += "</div>"
st.markdown(strip_html, unsafe_allow_html=True)

# ─── OPTIMIZATION RUN ───────────────────────────────────────
if run or "results" not in st.session_state:
    with st.spinner("Running optimization models..."):
        mvo  = MarkowitzOptimizer(returns)
        ms   = mvo.max_sharpe(risk_free_rate=risk_free)
        mv   = mvo.min_variance()
        ef   = mvo.efficient_frontier(n_points=60)

        bl_model = BlackLittermanModel(returns, risk_aversion=risk_aversion, tau=tau)
        tl = returns.columns.tolist()
        n  = len(tl)
        P  = np.zeros((2, n))
        P[0, tl.index("TCS.NS")]  = 1
        P[1, tl.index("INFY.NS")] = 1
        P[1, tl.index("SBIN.NS")] = -1
        bl_model.add_views(P, np.array([view1_ret, view2_spread]))
        bl_result  = bl_model.optimize(risk_free_rate=risk_free)
        constrained = build_constrained_optimizer(returns, max_sector, min_esg, risk_free)
        bench       = compare_to_benchmark(ms["weights"], returns)

        st.session_state["results"] = dict(
            ms=ms, mv=mv, ef=ef, bl=bl_result,
            constrained=constrained, bench=bench
        )

res = st.session_state["results"]
ms, mv, ef, bl_r, con, bench = (
    res["ms"], res["mv"], res["ef"], res["bl"], res["constrained"], res["bench"]
)

# ─── KPI STRIP ──────────────────────────────────────────────
st.markdown("<div class='section-tag'>Portfolio Summary</div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Max Sharpe Return",  f"{ms['annual_return']*100:.1f}%",
          f"vs NIFTY {(ms['annual_return']-bench['benchmark_annual_return'])*100:+.1f}%")
k2.metric("Sharpe Ratio",       f"{ms['sharpe_ratio']:.3f}",    "Target > 1.0")
k3.metric("Volatility",         f"{ms['annual_volatility']*100:.1f}%")
k4.metric("Portfolio Alpha",    f"{bench['alpha_annual']*100:.1f}%", "Jensen's α")
k5.metric("Beta",               f"{bench['beta']:.3f}",         "vs NIFTY 50")
k6.metric("Information Ratio",  f"{bench['information_ratio']:.3f}", "Target > 0.5")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Efficient Frontier",
    "⚖️  Weight Comparison",
    "🧠  Black-Litterman",
    "🌿  ESG & Constraints",
    "📊  Benchmark"
])

# ── TAB 1: Efficient Frontier ───────────────────────────────
with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ef["volatility"]*100, y=ef["return"]*100,
        mode="lines", line=dict(color="#3B82F6", width=2),
        name="Efficient Frontier"
    ))

    mu_a  = returns.mean() * 252
    vol_a = returns.std()  * np.sqrt(252)
    for t in returns.columns:
        fig.add_trace(go.Scatter(
            x=[float(vol_a[t])*100], y=[float(mu_a[t])*100],
            mode="markers+text", marker=dict(size=7, color="#475569"),
            text=[t.replace(".NS","")], textposition="top right",
            textfont=dict(size=9, color="#64748B"), showlegend=False
        ))

    for vol, ret, col, name in [
        (ms["annual_volatility"],  ms["annual_return"],  "#10B981", "Max Sharpe"),
        (mv["annual_volatility"],  mv["annual_return"],  "#F59E0B", "Min Variance"),
        (bl_r["annual_volatility"],bl_r["annual_return"],"#A855F7", "Black-Litterman"),
        (con["annual_volatility"], con["annual_return"],  "#EC4899", "ESG Constrained"),
    ]:
        fig.add_trace(go.Scatter(
            x=[float(vol)*100], y=[float(ret)*100], mode="markers",
            marker=dict(size=14, color=col, symbol="diamond",
                        line=dict(width=1.5, color="#0A0E1A")), name=name
        ))

    vols_cml = np.linspace(0, float(ms["annual_volatility"])*100*1.5, 50)
    rets_cml = risk_free*100 + (ms["annual_return"]-risk_free) / ms["annual_volatility"] * (vols_cml/100)*100
    fig.add_trace(go.Scatter(
        x=vols_cml, y=rets_cml, mode="lines",
        line=dict(color="#F59E0B", width=1, dash="dot"), name="Capital Market Line"
    ))

    apply_dark_theme(fig, "Efficient Frontier — NIFTY 50 Constituents",
                     "Annualised Volatility (%)", "Annualised Return (%)", 520)
    chart(fig)
    st.markdown("**Capital Market Line (CML):** The dotted line from the risk-free rate tangent to the frontier. The tangency point is the Max Sharpe portfolio — every rational investor holds a mix of this and the risk-free asset.")

# ── TAB 2: Weight Comparison ────────────────────────────────
with tab2:
    all_tickers = sorted(set(list(ms["weights"]) + list(bl_r["weights"]) + list(con["weights"])))
    labels = [t.replace(".NS","") for t in all_tickers]
    w_ms   = [float(ms["weights"].get(t, 0))   for t in all_tickers]
    w_bl   = [float(bl_r["weights"].get(t, 0)) for t in all_tickers]
    w_con  = [float(con["weights"].get(t, 0))  for t in all_tickers]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Max Sharpe (MVO)", x=labels, y=[w*100 for w in w_ms],  marker_color="#3B82F6"))
    fig2.add_trace(go.Bar(name="Black-Litterman",  x=labels, y=[w*100 for w in w_bl],  marker_color="#A855F7"))
    fig2.add_trace(go.Bar(name="ESG Constrained",  x=labels, y=[w*100 for w in w_con], marker_color="#10B981"))

    apply_dark_theme(fig2, "Portfolio Weight Comparison Across Models",
                     "", "Weight (%)", 480, extra=dict(barmode="group"))
    # Custom x-axis tick font after layout applied
    fig2.update_xaxes(tickfont=dict(size=11))
    chart(fig2)

    df_w = pd.DataFrame({
        "Ticker":          labels,
        "MVO Max Sharpe":  [f"{w*100:.1f}%" for w in w_ms],
        "Black-Litterman": [f"{w*100:.1f}%" for w in w_bl],
        "ESG Constrained": [f"{w*100:.1f}%" for w in w_con],
    })
    df_w = df_w[df_w[["MVO Max Sharpe","Black-Litterman","ESG Constrained"]].apply(
        lambda r: any(float(v.strip("%")) > 0 for v in r), axis=1)]
    st.dataframe(df_w.set_index("Ticker"), use_container_width=True)

# ── TAB 3: Black-Litterman ──────────────────────────────────
with tab3:
    tl        = returns.columns.tolist()
    labels3   = [t.replace(".NS","") for t in tl]
    eq_rets   = [float(bl_r["equilibrium_returns"].get(t, 0))*100 for t in tl]
    post_rets = [float(bl_r["posterior_returns"].get(t, 0))*100   for t in tl]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=labels3, y=eq_rets, mode="lines+markers",
                              name="Equilibrium Π (CAPM prior)",
                              line=dict(color="#F59E0B", width=2), marker=dict(size=8)))
    fig3.add_trace(go.Scatter(x=labels3, y=post_rets, mode="lines+markers",
                              name="Posterior μ_BL (after views)",
                              line=dict(color="#3B82F6", width=2),
                              marker=dict(size=8, symbol="diamond")))

    for asset, label in [("TCS.NS","View 1: TCS"), ("INFY.NS","View 2: INFY>SBIN")]:
        if asset in tl:
            idx = tl.index(asset)
            fig3.add_annotation(x=labels3[idx], y=post_rets[idx]+0.5, text=label,
                                showarrow=True, arrowhead=2,
                                font=dict(color="#EC4899", size=10), arrowcolor="#EC4899")

    apply_dark_theme(fig3, "Black-Litterman: Equilibrium vs Posterior Expected Returns",
                     "", "Annualised Return (%)", 460)
    chart(fig3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**View 1 — Absolute:** TCS will return {view1_ret*100:.0f}% annually")
        st.markdown(f"**View 2 — Relative:** INFY outperforms SBIN by {view2_spread*100:.0f}%")
        st.markdown(f"**τ (prior uncertainty):** {tau:.2f} — lower = trust equilibrium more")
    with c2:
        st.metric("BL Return",     f"{bl_r['annual_return']*100:.2f}%")
        st.metric("BL Sharpe",     f"{bl_r['sharpe_ratio']:.3f}")
        st.metric("BL Volatility", f"{bl_r['annual_volatility']*100:.2f}%")

# ── TAB 4: ESG & Constraints ────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        esg_df = pd.DataFrame({
            "Ticker":    [t.replace(".NS","") for t in NIFTY50_TICKERS],
            "ESG Score": [ESG_SCORES.get(t, 0) for t in NIFTY50_TICKERS],
            "Eligible":  ["✅ Pass" if ESG_SCORES.get(t,0) >= min_esg else "❌ Filtered"
                          for t in NIFTY50_TICKERS]
        }).sort_values("ESG Score", ascending=True)

        fig4 = px.bar(esg_df, x="ESG Score", y="Ticker", orientation="h",
                      color="Eligible",
                      color_discrete_map={"✅ Pass":"#10B981","❌ Filtered":"#EF4444"})
        fig4.add_vline(x=min_esg, line_dash="dot", line_color="#F59E0B",
                       annotation_text=f"Min: {min_esg}", annotation_font_color="#F59E0B")
        apply_dark_theme(fig4, "ESG Scores — Asset Filter", height=420)
        chart(fig4)

    with c2:
        sector_data = {k: float(v) for k, v in con["sector_exposure"].items() if float(v) > 0.001}
        fig5 = go.Figure(go.Pie(
            labels=list(sector_data.keys()),
            values=[v*100 for v in sector_data.values()],
            hole=0.55,
            marker=dict(colors=["#3B82F6","#10B981","#F59E0B","#A855F7","#EC4899","#06B6D4"],
                        line=dict(color="#0A0E1A", width=2)),
            textfont=dict(color="#F1F5F9")
        ))
        fig5.add_annotation(text=f"ESG<br>{con['esg_score']:.0f}", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=18, color="#F1F5F9"))
        apply_dark_theme(fig5, f"Sector Allocation (Max {max_sector*100:.0f}% cap)", height=420,
                         extra=dict(showlegend=True))
        chart(fig5)

    st.markdown(f"""
| Constraint | Setting | Impact |
|---|---|---|
| ESG Filter | Score ≥ {min_esg} | {15 - len(con['eligible_assets'])} asset(s) removed |
| Sector Cap | ≤ {max_sector*100:.0f}% per sector | Forces diversification |
| No Short-Selling | w_i ≥ 0 | Long-only |
| Budget | Σw_i = 1 | Fully invested |
    """)

# ── TAB 5: Benchmark ────────────────────────────────────────
with tab5:
    tickers_w  = [t for t in ms["weights"] if float(ms["weights"][t]) > 0.001]
    weights_w  = np.array([float(ms["weights"][t]) for t in tickers_w])
    port_ret_s = (returns[tickers_w] * weights_w).sum(axis=1)
    port_cum   = (1 + port_ret_s).cumprod() - 1

    bench_raw = yf.download("^NSEI",
                             start=returns.index[0].strftime("%Y-%m-%d"),
                             end=returns.index[-1].strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)["Close"]
    if isinstance(bench_raw, pd.DataFrame):
        bench_raw = bench_raw.squeeze()
    bench_ret = bench_raw.pct_change().dropna().squeeze()
    bench_ret.index = pd.to_datetime(bench_ret.index)
    common    = port_cum.index.intersection(bench_ret.index)
    bench_cum = (1 + bench_ret.loc[common]).cumprod() - 1

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values*100,
                              mode="lines", name="Max Sharpe Portfolio",
                              line=dict(color="#3B82F6", width=2.5)))
    fig6.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values*100,
                              mode="lines", name="NIFTY 50 Benchmark",
                              line=dict(color="#F59E0B", width=2, dash="dot")))
    apply_dark_theme(fig6, "Cumulative Return: Portfolio vs NIFTY 50",
                     "", "Cumulative Return (%)", 420)
    chart(fig6)

    st.markdown("### Performance Attribution")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Portfolio α",   f"{bench['alpha_annual']*100:.2f}%")
    b2.metric("Beta (β)",      f"{bench['beta']:.3f}")
    b3.metric("Info Ratio",    f"{bench['information_ratio']:.3f}")
    b4.metric("Tracking Err",  f"{bench['tracking_error']*100:.2f}%")
    b5.metric("Active Return", f"{(bench['portfolio_annual_return']-bench['benchmark_annual_return'])*100:.2f}%")

    st.markdown("""
| Metric | Formula | Interpretation |
|---|---|---|
| Alpha (α) | R_p − R_f − β(R_b − R_f) | Excess return above CAPM prediction |
| Beta (β) | Cov(R_p, R_b) / Var(R_b) | < 1 = less volatile than market |
| Information Ratio | Active Return / Tracking Error | > 0.5 is good; > 1.0 is exceptional |
| Tracking Error | σ(R_p − R_b) × √252 | How closely portfolio tracks benchmark |
    """)

# ─── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:12px;'>"
    "Portfolio Optimization Engine · Markowitz MVO + Black-Litterman · "
    "Built with Python, SciPy, Plotly, Streamlit · Data: Yahoo Finance / NSE"
    "</div>", unsafe_allow_html=True
)
