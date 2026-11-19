"""Scoring, portfolio optimization (MILP) and Monte Carlo for cash-secured puts."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, fields

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, brentq, milp
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Risk profiles
# ---------------------------------------------------------------------------


@dataclass
class RiskProfile:
    name: str
    max_delta: float            # max |delta| of a single contract
    max_avg_delta: float        # collateral-weighted average |delta| cap
    min_dte: int
    max_dte: int
    max_weight_ticker: float    # max share of capital in one ticker
    max_weight_sector: float    # max share of capital in one sector
    cash_buffer: float          # share of capital left unencumbered
    stress_shock: float         # market drop in the systemic stress test
    max_stress_loss: float      # max portfolio loss in that stress (share of capital)
    gap_shock: float            # single-name gap down
    max_gap_loss: float         # max loss if any one ticker gaps (share of capital)
    min_cushion: float          # min % drop from spot to breakeven
    min_tickers: int
    max_spread_pct: float       # max (ask-bid)/mid
    min_open_interest: int
    avoid_earnings: bool        # exclude contracts whose life spans an earnings date


PROFILES = {
    "very_low": RiskProfile("very_low", 0.12, 0.10, 21, 45, 0.15, 0.30, 0.20,
                            0.20, 0.05, 0.25, 0.015, 0.07, 6, 0.10, 500, True),
    "low":      RiskProfile("low", 0.20, 0.16, 14, 45, 0.20, 0.35, 0.10,
                            0.20, 0.08, 0.25, 0.025, 0.05, 5, 0.12, 300, True),
    "medium":   RiskProfile("medium", 0.30, 0.24, 7, 60, 0.30, 0.45, 0.05,
                            0.25, 0.12, 0.30, 0.04, 0.03, 4, 0.15, 100, False),
    "high":     RiskProfile("high", 0.40, 0.32, 7, 60, 0.40, 0.60, 0.00,
                            0.25, 0.18, 0.30, 0.06, 0.01, 3, 0.20, 50, False),
}


def make_profile(name: str, overrides: dict | None = None) -> RiskProfile:
    p = PROFILES[name]
    if overrides:
        valid = {f.name for f in fields(RiskProfile)}
        bad = set(overrides) - valid
        if bad:
            raise ValueError(f"unknown risk parameters: {sorted(bad)}")
        p = replace(p, **overrides)
        if "max_delta" in overrides and "max_avg_delta" not in overrides:
            p = replace(p, max_avg_delta=min(p.max_avg_delta, p.max_delta))
    return p

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def bs_put(S, K, T, r, sigma):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, -norm.cdf(-d1)


def implied_vol_put(price, S, K, T, r):
    intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    if not (price > intrinsic + 1e-4) or T <= 0:
        return np.nan
    f = lambda s: float(bs_put(S, K, T, r, s)[0]) - price
    try:
        return brentq(f, 1e-3, 5.0, maxiter=100)
    except ValueError:
        return np.nan


def prob_above(S, X, T, mu, sigma):
    d = (np.log(S / X) + (mu - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def clean_chain(chain: pd.DataFrame, r: float) -> pd.DataFrame:
    """Normalize provider output: mid, IV (solve if missing), delta (compute if missing)."""
    df = chain.copy()
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"]) & (df["expiry_dte"] > 0)]
    df = df[df["strike"] < df["spot"] * 1.02]                       # puts at/below the money
    df["T"] = df["expiry_dte"] / 365
    df["mid"] = (df["bid"] + df["ask"]) / 2
    iv = df["iv"] if "iv" in df else pd.Series(np.nan, index=df.index)
    iv = pd.to_numeric(iv, errors="coerce")
    iv = iv.where(iv < 3, iv / 100)                                 # percent -> decimal
    bad = iv.isna() | (iv <= 0.01)
    if bad.any():
        iv.loc[bad] = [implied_vol_put(m, s, k, t, r) for m, s, k, t in
                       df.loc[bad, ["mid", "spot", "strike", "T"]].itertuples(index=False)]
    df["iv"] = iv
    df = df[df["iv"].notna()]
    _, d = bs_put(df["spot"], df["strike"], df["T"], r, df["iv"])
    if "delta" in df:
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce").fillna(pd.Series(d, index=df.index))
    else:
        df["delta"] = d
    return df.reset_index(drop=True)


def score_chain(chain: pd.DataFrame, p: RiskProfile, r=0.04, fee=0.65) -> pd.DataFrame:
    df = clean_chain(chain, r)
    df["abs_delta"] = df["delta"].abs()
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df["premium"] = df["mid"] * 100 - fee                       # $ per contract, net of fee
    df["collateral"] = df["strike"] * 100
    df["roc"] = df["premium"] / df["collateral"]
    df["ann_yield"] = df["roc"] * 365 / df["expiry_dte"]
    df["ann_premium"] = df["premium"] * 365 / df["expiry_dte"]
    df["breakeven"] = df["strike"] - df["premium"] / 100
    df["cushion"] = 1 - df["breakeven"] / df["spot"]
    df["p_otm"] = prob_above(df["spot"], df["strike"], df["T"], r, df["iv"])
    df["p_profit"] = prob_above(df["spot"], df["breakeven"], df["T"], r, df["iv"])

    beta = df["beta"].fillna(1.0) if "beta" in df else 1.0
    drop = np.minimum(p.stress_shock * beta, 0.95)
    df["stress_loss"] = np.maximum(0, (df["strike"] - df["spot"] * (1 - drop)) * 100 - df["premium"])
    df["stress_loss_pct"] = df["stress_loss"] / df["collateral"]
    df["gap_loss"] = np.maximum(0, (df["strike"] - df["spot"] * (1 - p.gap_shock)) * 100 - df["premium"])
    df["gap_loss_pct"] = df["gap_loss"] / df["collateral"]
    tail = np.maximum(df["stress_loss_pct"], df["gap_loss_pct"])
    df["ret_risk"] = df["ann_yield"] / np.maximum(tail, 0.005)

    hv = df["hv"] if "hv" in df else pd.Series(np.nan, index=df.index)
    df["vrp"] = df["iv"] - hv
    vol_real = hv.fillna(df["iv"])
    payout, _ = bs_put(df["spot"], df["strike"], df["T"], r, vol_real)
    df["edge"] = df["premium"] - payout * 100
    df["edge_ann"] = df["edge"] / df["collateral"] * 365 / df["expiry_dte"]

    ok = ((df["abs_delta"] <= p.max_delta)
          & df["expiry_dte"].between(p.min_dte, p.max_dte)
          & (df["spread_pct"] <= p.max_spread_pct)
          & (df["cushion"] >= p.min_cushion)
          & (df["premium"] > 0))
    if "open_interest" in df:
        ok &= df["open_interest"].fillna(0) >= p.min_open_interest
    if p.avoid_earnings and "earnings_in_window" in df:
        ok &= ~df["earnings_in_window"].fillna(False).astype(bool)
    df["eligible"] = ok
    return df


def prune(c: pd.DataFrame, per_ticker=40) -> pd.DataFrame:
    """Keep the best candidates per ticker (by yield and by return/risk) to keep the MILP small."""
    keep = []
    for _, g in c.groupby("ticker"):
        top = set(g.nlargest(per_ticker // 2, "ann_yield").index) | set(g.nlargest(per_ticker // 2, "ret_risk").index)
        keep.extend(top)
    return c.loc[sorted(keep)].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


@dataclass
class Result:
    status: str
    objective: str
    portfolio: pd.DataFrame
    summary: dict


def _constraints(c, capital, p):
    n = len(c)
    tickers = list(c["ticker"].unique())
    m = len(tickers)
    nv = n + m
    coll = c["collateral"].to_numpy()
    A, lo, hi = [], [], []

    def add(a, l, h):
        A.append(a); lo.append(l); hi.append(h)

    a = np.zeros(nv); a[:n] = coll
    add(a, -np.inf, capital * (1 - p.cash_buffer))
    for j, t in enumerate(tickers):
        idx = (c["ticker"] == t).to_numpy()
        cap_t = p.max_weight_ticker * capital
        a = np.zeros(nv); a[:n][idx] = coll[idx]; a[n + j] = -cap_t
        add(a, -np.inf, 0)                                     # weight cap, only if used
        a = np.zeros(nv); a[:n][idx] = 1; a[n + j] = -1
        add(a, 0, np.inf)                                      # used -> >= 1 contract
        a = np.zeros(nv); a[:n][idx] = c["gap_loss"].to_numpy()[idx]
        add(a, -np.inf, p.max_gap_loss * capital)
    for s in c["sector"].unique():
        idx = (c["sector"] == s).to_numpy()
        a = np.zeros(nv); a[:n][idx] = coll[idx]
        add(a, -np.inf, p.max_weight_sector * capital)
    a = np.zeros(nv); a[:n] = coll * (c["abs_delta"].to_numpy() - p.max_avg_delta)
    add(a, -np.inf, 0)
    a = np.zeros(nv); a[:n] = c["stress_loss"].to_numpy()
    add(a, -np.inf, p.max_stress_loss * capital)
    a = np.zeros(nv); a[n:] = 1
    add(a, min(p.min_tickers, m), np.inf)
    ub = np.concatenate([np.maximum(np.floor(p.max_weight_ticker * capital / coll), 0), np.ones(m)])
    return np.array(A), np.array(lo), np.array(hi), ub


def optimize(scored: pd.DataFrame, capital: float, p: RiskProfile,
             objective="max_premium", target=None, time_limit=60) -> Result:
    """objective = 'max_premium': highest annualized premium inside the risk limits.
       objective = 'target'     : lowest stress loss that still reaches `target`
                                  annualized yield on capital (falls back to max_premium)."""
    c = scored[scored["eligible"]]
    if c.empty:
        return Result("no_eligible_contracts", objective, c, {})
    c = prune(c)
    n = len(c)
    A, lo, hi, ub = _constraints(c, capital, p)
    nv = A.shape[1]
    cons = [LinearConstraint(A, lo, hi)]
    ann = np.zeros(nv); ann[:n] = c["ann_premium"].to_numpy()
    stress = np.zeros(nv); stress[:n] = c["stress_loss"].to_numpy()
    kw = dict(integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub),
              options=dict(time_limit=time_limit, mip_rel_gap=1e-3))

    status = None
    res = None
    if objective == "target" and target is not None:
        res = milp(stress + 1e-6, constraints=cons + [LinearConstraint(ann, target * capital, np.inf)], **kw)
        status = "target_met" if res.success else None
    if status is None:
        res = milp(-ann + 1e-3 * stress, constraints=cons, **kw)
        if not res.success:
            return Result("infeasible", objective, c.iloc[0:0], {})
        status = "max_premium" if objective == "max_premium" else "target_unreachable_max_premium_shown"
    x = np.round(res.x[:n]).astype(int)
    port = c.loc[x > 0].copy()
    port["contracts"] = x[x > 0]
    return Result(status, objective, port, summarize(port, capital, p))


def summarize(port: pd.DataFrame, capital: float, p: RiskProfile) -> dict:
    k = port["contracts"]
    port["total_collateral"] = k * port["collateral"]
    port["total_premium"] = k * port["premium"]
    port["total_stress_loss"] = k * port["stress_loss"]
    port["total_gap_loss"] = k * port["gap_loss"]
    port["weight"] = port["total_collateral"] / capital
    coll = port["total_collateral"].sum()
    ann = (k * port["ann_premium"]).sum()
    w = port["total_collateral"] / coll if coll else 0
    return dict(
        profile=p.name, capital=capital, contracts=int(k.sum()), lines=len(port),
        tickers=int(port["ticker"].nunique()),
        capital_required=coll, cash_free=capital - coll, utilization=coll / capital,
        premium_collected=port["total_premium"].sum(),
        ann_premium=ann, ann_yield_on_capital=ann / capital,
        ann_yield_on_collateral=ann / coll if coll else 0,
        avg_delta=float((port["abs_delta"] * w).sum()),
        avg_dte=float((port["expiry_dte"] * w).sum()),
        avg_cushion=float((port["cushion"] * w).sum()),
        stress_loss=port["total_stress_loss"].sum(),
        stress_loss_pct=port["total_stress_loss"].sum() / capital,
        worst_gap_loss_pct=port.groupby("ticker")["total_gap_loss"].sum().max() / capital,
        max_sector_weight=port.groupby("sector")["total_collateral"].sum().max() / capital,
        expected_edge_ann=float((k * port["edge_ann"] * port["collateral"]).sum() / capital),
    )

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def monte_carlo(port, capital, sims=20000, mkt_vol=0.18, mu=0.04, df_t=4, seed=1) -> dict:
    """One-factor, fat-tailed (Student-t) simulation to expiry. Uses historical vol
    (hv) as the real-world vol when available, else IV."""
    if port.empty:
        return {}
    rng = np.random.default_rng(seed)
    sc = math.sqrt((df_t - 2) / df_t)
    zm = rng.standard_t(df_t, sims) * sc
    pnl = np.zeros(sims); n_ass = np.zeros(sims); zi = {}
    for _, row in port.iterrows():
        T, beta = row["T"], (row.get("beta") if pd.notna(row.get("beta")) else 1.0)
        vol = row["hv"] if pd.notna(row.get("hv", np.nan)) else row["iv"]
        idio = math.sqrt(max(vol ** 2 - (beta * mkt_vol) ** 2, (0.25 * vol) ** 2))
        if row["ticker"] not in zi:
            zi[row["ticker"]] = rng.standard_t(df_t, sims) * sc
        ret = (mu - 0.5 * vol ** 2) * T + beta * mkt_vol * math.sqrt(T) * zm + idio * math.sqrt(T) * zi[row["ticker"]]
        ST = row["spot"] * np.exp(ret)
        pnl += row["contracts"] * (row["premium"] - np.maximum(row["strike"] - ST, 0) * 100)
        n_ass += row["contracts"] * (ST < row["strike"])
    q05, q01 = np.quantile(pnl, [0.05, 0.01])
    return dict(sims=sims, expected_pnl=pnl.mean(), p_loss=(pnl < 0).mean(),
                p_any_assignment=(n_ass > 0).mean(), expected_assigned=n_ass.mean(),
                VaR95=-q05, CVaR95=-pnl[pnl <= q05].mean(), VaR99=-q01,
                CVaR99=-pnl[pnl <= q01].mean(), worst=-pnl.min())
