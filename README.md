# Portfolio Optimization Engine

**Markowitz Mean-Variance Optimization · Black-Litterman Model · ESG Constraints · NIFTY 50 Benchmark**

Implements institutional-grade portfolio construction techniques in Python using NIFTY 50 equity data.

---

## Table of Contents

- [Theory Overview](#theory-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Theory Overview

### 1. Mean-Variance Optimization (Markowitz, 1952)

The core idea: combining assets with low correlations reduces portfolio risk without sacrificing proportional return — the only true "free lunch" in finance.

For a portfolio with weight vector **w**, the expected return and variance are:

```
μ_p = wᵀ μ        (weighted average of asset returns)
σ²_p = wᵀ Σ w     (accounts for all pairwise correlations)
```

We solve a quadratic programme for two objectives:

- **Max Sharpe** — maximise `(μ_p − R_f) / σ_p` (return per unit of risk)
- **Min Variance** — find the global minimum variance portfolio (GMVP)

The set of optimal portfolios traces the **efficient frontier** — no rational investor should hold a portfolio below it.

### 2. Black-Litterman Model (Black & Litterman, Goldman Sachs, 1990)

Raw Markowitz is fragile: a 1% change in expected return estimates can cause 30% weight swings. Black-Litterman fixes this in two steps:

**Step 1 — Reverse Optimisation.** Instead of estimating expected returns from noisy history, imply them from what the market already prices in (via CAPM):

```
Π = δ · Σ · w_market
```

`Π` represents the returns the market *must* be expecting if current market-cap weights are rational. `δ ≈ 2.5` is the market risk-aversion coefficient.

**Step 2 — Bayesian Update with Views.** Express analyst views as `P μ = Q + ε`, where:
- `P` is the pick matrix (which assets each view is about)
- `Q` is the vector of view returns
- `Ω` is a diagonal matrix of view uncertainties

The posterior expected returns are:

```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
```

Result: diversified, stable portfolios that shift weights only proportionally to analyst conviction.

### 3. ESG & Sector Constraints

Assets below an ESG score threshold are removed pre-optimisation. Sector concentration is capped via inequality constraints:

```
Σ_{i ∈ sector_g} w_i ≤ κ    for each sector g
```

Both constraints are added to the SciPy SLSQP solver alongside the standard budget and no-short-selling constraints.

### 4. Benchmark Attribution

Performance is measured against the NIFTY 50 index using:

| Metric | Formula | What it measures |
|---|---|---|
| Jensen's α | `(R_p − R_f) − β(R_b − R_f)` | Excess return above CAPM prediction |
| Beta β | `Cov(R_p, R_b) / Var(R_b)` | Market sensitivity |
| Tracking Error | `σ(R_p − R_b) × √252` | Deviation from benchmark |
| Information Ratio | `Active Return / Tracking Error` | Consistency of outperformance |

---

## Project Structure

```
portfolio_optimizer/
├── portfolio_optimizer.py   # Core engine — MVO, Black-Litterman, constraints, attribution
├── dashboard.py             # Bloomberg-style Streamlit dashboard (5 interactive tabs)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Requirements

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

**Dependencies:**

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
yfinance>=0.2.36
streamlit>=1.32.0
plotly>=5.19.0
```

> Internet access is required — the script fetches live price data from Yahoo Finance (NSE).

---

## How to Run

### Option A — Command-line script

Runs all four optimization models and prints results to terminal.

```bash
python3 portfolio_optimizer.py
```

**Expected output:**

```
============================================================
  PORTFOLIO OPTIMIZER: Markowitz + Black-Litterman
============================================================
[DATA] Fetching 15 tickers from Yahoo Finance...
[DATA] Got 986 days x 15 assets

[1] MARKOWITZ MAX SHARPE
  Return: 29.95% | Vol: 16.28% | Sharpe: 1.840
  Top holdings: BHARTIARTL 35.1% | ITC 31.3% | LT 18.1% | SBIN 9.9% | ICICIBANK 5.6%

[2] MARKOWITZ MINIMUM VARIANCE
  Return: 14.68% | Vol: 12.47% | Sharpe: 1.177

[3] BLACK-LITTERMAN MODEL
  Return: 8.60% | Vol: 20.59% | Sharpe: 0.102

[4] CONSTRAINED (ESG + SECTOR LIMITS)
  Return: 30.91% | Vol: 18.31% | Sharpe: 1.333
  Portfolio ESG Score: 68.75
  Sector Exposure: Telecom: 40.0% | Finance: 31.7% | Industrials: 28.3%

[5] BENCHMARK COMPARISON vs NIFTY 50
  Portfolio Return: 30.09%
  NIFTY 50 Return:  14.43%
  Alpha: 16.53% | Beta: 0.890 | Information Ratio: 1.557
```

### Option B — Interactive Dashboard

Launches a Bloomberg-style web dashboard in your browser.

```bash
streamlit run dashboard.py
```

Then open `http://localhost:8501` in your browser.

**Dashboard tabs:**

| Tab | What you see |
|---|---|
| Efficient Frontier | All assets plotted, 4 optimal portfolios, Capital Market Line |
| Weight Comparison | Grouped bar chart comparing MVO vs Black-Litterman vs ESG weights |
| Black-Litterman | Equilibrium vs posterior returns with view annotations |
| ESG & Constraints | ESG score filter chart + sector allocation donut |
| Benchmark | Cumulative return vs NIFTY 50, full attribution table |

Use the **sidebar sliders** to adjust risk-free rate, BL views, sector caps, and ESG threshold live.

---

## Results

| Portfolio | Annual Return | Volatility | Sharpe Ratio |
|---|---|---|---|
| Max Sharpe (MVO) | 29.95% | 16.28% | **1.840** |
| Min Variance (MVO) | 14.68% | 12.47% | 1.177 |
| Black-Litterman | 8.60% | 20.59% | 0.102 |
| ESG Constrained | 30.91% | 18.31% | 1.333 |
| NIFTY 50 (benchmark) | 14.43% | — | — |

**Max Sharpe portfolio vs NIFTY 50:** Alpha +16.53% · Beta 0.890 · Information Ratio 1.557

> Note: Results are in-sample (2021–2024). The high Information Ratio reflects backtest optimism; out-of-sample validation is recommended before live deployment.

---

## References

- Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance, 7(1), 77–91.
- Black, F. & Litterman, R. (1990). *Asset Allocation: Combining Investor Views with Market Equilibrium.* Goldman Sachs Fixed Income Research.
- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal, 48(5), 28–43.
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios.* Goldman Sachs Asset Management.
- Idzorek, T. (2005). *A Step-by-Step Guide to the Black-Litterman Model.* Zephyr Associates.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.

---

*Built with Python · SciPy · Plotly · Streamlit · Data: Yahoo Finance / NSE*
