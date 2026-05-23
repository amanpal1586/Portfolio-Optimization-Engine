"""
Portfolio Optimization Tool: Markowitz + Black-Litterman
Author: [Your Name] | Goldman Sachs Summer Analyst Project
Language: Python 3.10+
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATA LAYER
# ─────────────────────────────────────────────

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS",
    "BHARTIARTL.NS", "ITC.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS"
]

BENCHMARK_TICKER = "^NSEI"  # NIFTY 50 Index


def fetch_data(tickers: list, start: str = "2021-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download adjusted close prices for given tickers."""
    print(f"[DATA] Fetching {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(axis=1, how="all").dropna()
    print(f"[DATA] Got {prices.shape[0]} days x {prices.shape[1]} assets")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────
# 2. MARKOWITZ MEAN-VARIANCE OPTIMIZATION
# ─────────────────────────────────────────────

class MarkowitzOptimizer:
    """
    Classic Mean-Variance Optimizer.
    Solves: min w'Σw  subject to w'μ = target_return, Σw_i = 1, w_i >= 0
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mu = returns.mean() * 252          # Annualised mean returns
        self.cov = returns.cov() * 252          # Annualised covariance matrix
        self.n = len(self.mu)
        self.tickers = returns.columns.tolist()

    def _portfolio_stats(self, weights: np.ndarray):
        port_return = weights @ self.mu.values
        port_vol = np.sqrt(weights @ self.cov.values @ weights)
        sharpe = port_return / port_vol  # Assuming Rf = 0 for simplicity
        return port_return, port_vol, sharpe

    def max_sharpe(self, risk_free_rate: float = 0.065) -> dict:
        """Find portfolio with maximum Sharpe Ratio."""
        def neg_sharpe(w):
            r, v, _ = self._portfolio_stats(w)
            return -(r - risk_free_rate) / v

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * self.n
        w0 = np.ones(self.n) / self.n

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000})

        r, v, s = self._portfolio_stats(result.x)
        return {
            "weights": dict(zip(self.tickers, np.round(result.x, 4))),
            "annual_return": round(r, 4),
            "annual_volatility": round(v, 4),
            "sharpe_ratio": round(s, 4)
        }

    def min_variance(self) -> dict:
        """Find Global Minimum Variance Portfolio."""
        def port_var(w):
            return w @ self.cov.values @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * self.n
        w0 = np.ones(self.n) / self.n

        result = minimize(port_var, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        r, v, s = self._portfolio_stats(result.x)
        return {
            "weights": dict(zip(self.tickers, np.round(result.x, 4))),
            "annual_return": round(r, 4),
            "annual_volatility": round(v, 4),
            "sharpe_ratio": round(s, 4)
        }

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Generate the full efficient frontier."""
        target_returns = np.linspace(self.mu.min(), self.mu.max(), n_points)
        frontier = []

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ self.mu.values - t}
            ]
            bounds = [(0, 1)] * self.n
            w0 = np.ones(self.n) / self.n
            result = minimize(lambda w: w @ self.cov.values @ w, w0,
                              method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                r, v, s = self._portfolio_stats(result.x)
                frontier.append({"return": r, "volatility": v, "sharpe": s})

        return pd.DataFrame(frontier)


# ─────────────────────────────────────────────
# 3. BLACK-LITTERMAN MODEL
# ─────────────────────────────────────────────

class BlackLittermanModel:
    """
    Black-Litterman Model (1990, Goldman Sachs).

    Combines:
    - Equilibrium returns (from CAPM / reverse optimization)
    - Investor views (your subjective opinions)

    Output: Posterior expected returns → feed into Markowitz.

    Key formula:
    μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}Π + P'Ω^{-1}Q]
    """

    def __init__(self, returns: pd.DataFrame, market_caps: dict = None,
                 risk_aversion: float = 2.5, tau: float = 0.05):
        self.returns = returns
        self.cov = returns.cov().values * 252       # Σ
        self.tickers = returns.columns.tolist()
        self.n = len(self.tickers)
        self.delta = risk_aversion                  # λ (market risk aversion)
        self.tau = tau                              # Uncertainty in prior (typically 0.01–0.10)

        # Market cap weights (equal if not provided)
        if market_caps:
            total = sum(market_caps.values())
            self.w_mkt = np.array([market_caps.get(t, 0) / total for t in self.tickers])
        else:
            self.w_mkt = np.ones(self.n) / self.n

        # Π = δΣw_mkt  (Implied Equilibrium Returns via reverse optimization)
        self.Pi = self.delta * self.cov @ self.w_mkt

    def add_views(self, P: np.ndarray, Q: np.ndarray, omega: np.ndarray = None):
        """
        P: (k x n) Pick matrix — which assets each view is about
        Q: (k,) View returns vector
        Ω: (k x k) Uncertainty of views (diagonal). Auto-computed if None.
        """
        self.P = P
        self.Q = Q

        if omega is None:
            # Idzorek's method: Ω = diag(τ * P Σ P')
            self.Omega = np.diag(np.diag(self.tau * P @ self.cov @ P.T))
        else:
            self.Omega = omega

    def compute_posterior(self) -> np.ndarray:
        """Compute posterior expected returns μ_BL."""
        tau_sigma_inv = inv(self.tau * self.cov)   # (τΣ)^{-1}
        omega_inv = inv(self.Omega)                # Ω^{-1}

        # Posterior precision (inverse covariance)
        M = tau_sigma_inv + self.P.T @ omega_inv @ self.P

        # Posterior mean
        mu_bl = inv(M) @ (tau_sigma_inv @ self.Pi + self.P.T @ omega_inv @ self.Q)
        return mu_bl

    def optimize(self, risk_free_rate: float = 0.065) -> dict:
        """Run full BL pipeline → Markowitz on posterior returns."""
        mu_bl = self.compute_posterior()

        # Use BL posterior returns in place of historical means
        mu_series = pd.Series(mu_bl, index=self.tickers)
        cov_df = pd.DataFrame(self.cov, index=self.tickers, columns=self.tickers)

        n = self.n

        def neg_sharpe(w):
            r = w @ mu_series.values
            v = np.sqrt(w @ cov_df.values @ w)
            return -(r - risk_free_rate) / v

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        r = result.x @ mu_series.values
        v = np.sqrt(result.x @ cov_df.values @ result.x)
        s = (r - risk_free_rate) / v

        return {
            "equilibrium_returns": dict(zip(self.tickers, np.round(self.Pi, 4))),
            "posterior_returns": dict(zip(self.tickers, np.round(mu_bl, 4))),
            "weights": dict(zip(self.tickers, np.round(result.x, 4))),
            "annual_return": round(r, 4),
            "annual_volatility": round(v, 4),
            "sharpe_ratio": round(s, 4)
        }


# ─────────────────────────────────────────────
# 4. CONSTRAINTS ENGINE
# ─────────────────────────────────────────────

SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Finance",
    "INFY.NS": "IT",
    "HINDUNILVR.NS": "FMCG",
    "ICICIBANK.NS": "Finance",
    "KOTAKBANK.NS": "Finance",
    "LT.NS": "Industrials",
    "SBIN.NS": "Finance",
    "AXISBANK.NS": "Finance",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "WIPRO.NS": "IT",
    "ULTRACEMCO.NS": "Materials",
    "NESTLEIND.NS": "FMCG"
}

# Dummy ESG scores (0-100, higher = better)
ESG_SCORES = {
    "RELIANCE.NS": 62, "TCS.NS": 85, "HDFCBANK.NS": 78, "INFY.NS": 90,
    "HINDUNILVR.NS": 88, "ICICIBANK.NS": 74, "KOTAKBANK.NS": 76, "LT.NS": 70,
    "SBIN.NS": 65, "AXISBANK.NS": 71, "BHARTIARTL.NS": 68, "ITC.NS": 55,
    "WIPRO.NS": 87, "ULTRACEMCO.NS": 60, "NESTLEIND.NS": 82
}


def build_constrained_optimizer(returns: pd.DataFrame,
                                 max_sector_weight: float = 0.40,
                                 min_esg_score: float = 65.0,
                                 risk_free_rate: float = 0.065) -> dict:
    """
    Max Sharpe with sector concentration + ESG constraints.
    Filters out assets below ESG threshold, then caps sector exposure.
    """
    tickers = returns.columns.tolist()

    # ESG filter: remove assets below threshold
    eligible = [t for t in tickers if ESG_SCORES.get(t, 100) >= min_esg_score]
    print(f"[CONSTRAINTS] ESG filter: {len(tickers)} → {len(eligible)} assets")
    filtered_returns = returns[eligible]

    mu = filtered_returns.mean() * 252
    cov = filtered_returns.cov() * 252
    n = len(eligible)

    # Sector groupings for eligible assets
    sectors = list(set(SECTOR_MAP.get(t, "Other") for t in eligible))
    sector_indices = {s: [i for i, t in enumerate(eligible) if SECTOR_MAP.get(t) == s] for s in sectors}

    def neg_sharpe(w):
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        return -(r - risk_free_rate) / v

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Sector cap constraints
    for sector, idxs in sector_indices.items():
        if idxs:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idxs: max_sector_weight - np.sum(w[idx])
            })

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000})

    r = result.x @ mu.values
    v = np.sqrt(result.x @ cov.values @ result.x)
    s = (r - risk_free_rate) / v

    return {
        "eligible_assets": eligible,
        "weights": dict(zip(eligible, np.round(result.x, 4))),
        "annual_return": round(r, 4),
        "annual_volatility": round(v, 4),
        "sharpe_ratio": round(s, 4),
        "esg_score": round(sum(ESG_SCORES.get(t, 0) * w for t, w in zip(eligible, result.x)), 2),
        "sector_exposure": {s: round(sum(result.x[i] for i in idx), 4)
                            for s, idx in sector_indices.items()}
    }


# ─────────────────────────────────────────────
# 5. BENCHMARK COMPARISON
# ─────────────────────────────────────────────

def compare_to_benchmark(portfolio_weights: dict, returns: pd.DataFrame,
                           benchmark_ticker: str = "^NSEI") -> dict:
    """Compute Information Ratio vs NIFTY 50."""
    tickers = list(portfolio_weights.keys())
    weights = np.array([portfolio_weights[t] for t in tickers])

    port_returns = (returns[tickers] * weights).sum(axis=1)

    bench_raw = yf.download(benchmark_ticker, start=returns.index[0].strftime("%Y-%m-%d"),
                             end=returns.index[-1].strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)["Close"]
    # yf sometimes returns DataFrame with MultiIndex columns — squeeze to Series
    if isinstance(bench_raw, pd.DataFrame):
        bench_raw = bench_raw.squeeze()
    bench_returns = bench_raw.pct_change().dropna()
    bench_returns.index = pd.to_datetime(bench_returns.index)
    bench_returns = bench_returns.squeeze()  # guarantee 1-D Series

    # Align on common trading dates
    common_idx = port_returns.index.intersection(bench_returns.index)
    port_r = port_returns.loc[common_idx]
    bench_r = bench_returns.loc[common_idx]

    active_returns = port_r - bench_r
    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)

    # Beta calculation
    cov_matrix = np.cov(port_r, bench_r)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Alpha (Jensen's Alpha)
    rf_daily = 0.065 / 252
    alpha = (port_r.mean() - rf_daily) - beta * (bench_r.mean() - rf_daily)
    alpha_annual = alpha * 252

    return {
        "portfolio_annual_return": round(port_r.mean() * 252, 4),
        "benchmark_annual_return": round(bench_r.mean() * 252, 4),
        "beta": round(beta, 4),
        "alpha_annual": round(alpha_annual, 4),
        "information_ratio": round(information_ratio, 4),
        "tracking_error": round(active_returns.std() * np.sqrt(252), 4)
    }


# ─────────────────────────────────────────────
# 6. MAIN RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PORTFOLIO OPTIMIZER: Markowitz + Black-Litterman")
    print("=" * 60)

    # Step 1: Fetch data
    prices = fetch_data(NIFTY50_TICKERS)
    returns = compute_returns(prices)

    # Step 2: Markowitz — Max Sharpe
    print("\n[1] MARKOWITZ MAX SHARPE")
    mvo = MarkowitzOptimizer(returns)
    ms = mvo.max_sharpe()
    print(f"  Return: {ms['annual_return']*100:.2f}% | Vol: {ms['annual_volatility']*100:.2f}% | Sharpe: {ms['sharpe_ratio']:.3f}")
    top5 = sorted(ms['weights'].items(), key=lambda x: -float(x[1]))[:5]
    top5_str = ", ".join(f"{t.replace('.NS','')} {w*100:.1f}%" for t, w in top5)
    print(f"  Top holdings: {top5_str}")

    # Step 3: Markowitz — Min Variance
    print("\n[2] MARKOWITZ MINIMUM VARIANCE")
    mv = mvo.min_variance()
    print(f"  Return: {mv['annual_return']*100:.2f}% | Vol: {mv['annual_volatility']*100:.2f}% | Sharpe: {mv['sharpe_ratio']:.3f}")

    # Step 4: Black-Litterman
    print("\n[3] BLACK-LITTERMAN MODEL")
    bl = BlackLittermanModel(returns)

    # Views: We believe TCS outperforms by 5%, SBIN underperforms by 3%
    tickers_list = returns.columns.tolist()
    tcs_idx = tickers_list.index("TCS.NS")
    sbin_idx = tickers_list.index("SBIN.NS")
    infy_idx = tickers_list.index("INFY.NS")

    n = len(tickers_list)
    P = np.zeros((2, n))
    P[0, tcs_idx] = 1               # View 1: TCS absolute return
    P[1, infy_idx] = 1; P[1, sbin_idx] = -1  # View 2: INFY outperforms SBIN

    Q = np.array([0.12, 0.05])      # View 1: 12% return; View 2: INFY beats SBIN by 5%

    bl.add_views(P, Q)
    bl_result = bl.optimize()
    print(f"  Return: {bl_result['annual_return']*100:.2f}% | Vol: {bl_result['annual_volatility']*100:.2f}% | Sharpe: {bl_result['sharpe_ratio']:.3f}")

    # Step 5: Constrained Optimization
    print("\n[4] CONSTRAINED (ESG + SECTOR LIMITS)")
    constrained = build_constrained_optimizer(returns, max_sector_weight=0.40, min_esg_score=65)
    print(f"  Return: {constrained['annual_return']*100:.2f}% | Vol: {constrained['annual_volatility']*100:.2f}% | Sharpe: {constrained['sharpe_ratio']:.3f}")
    print(f"  Portfolio ESG Score: {constrained['esg_score']}")
    sector_str = " | ".join(f"{s}: {v*100:.1f}%" for s, v in constrained['sector_exposure'].items() if float(v) > 0)
    print(f"  Sector Exposure: {sector_str}")

    # Step 6: Benchmark comparison
    print("\n[5] BENCHMARK COMPARISON vs NIFTY 50")
    bench_compare = compare_to_benchmark(ms['weights'], returns)
    print(f"  Portfolio Return: {bench_compare['portfolio_annual_return']*100:.2f}%")
    print(f"  NIFTY 50 Return:  {bench_compare['benchmark_annual_return']*100:.2f}%")
    print(f"  Alpha: {bench_compare['alpha_annual']*100:.2f}% | Beta: {bench_compare['beta']:.3f}")
    print(f"  Information Ratio: {bench_compare['information_ratio']:.3f}")

    print("\n" + "=" * 60)
    print("                   Run complete. ")
    print("=" * 60)
