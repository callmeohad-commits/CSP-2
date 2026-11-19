#!/usr/bin/env python3
"""CSP optimizer - live data -> max-premium mix of large caps under a risk level.

Examples
    python run.py --capital 250000 --risk low
    python run.py --capital 500000 --risk low --overrides '{"max_delta": 0.15}'
    python run.py --capital 250000 --risk low --objective target --target 0.10
    python run.py --chain-csv saved_chain.csv --capital 250000 --risk medium   # offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from csp import engine, report
from csp.providers import fetch_universe

ROOT = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--risk", choices=list(engine.PROFILES), default="low")
    ap.add_argument("--objective", choices=["max_premium", "target"], default="max_premium")
    ap.add_argument("--target", type=float, help="annualized yield on capital, for --objective target")
    ap.add_argument("--overrides", default="", help='JSON of risk params, e.g. {"max_delta":0.15,"max_dte":30}')
    ap.add_argument("--provider", default="auto", help="auto | barchart | tradier | marketdata | yfinance | cboe | comma list")
    ap.add_argument("--universe", default=str(ROOT / "config" / "universe.csv"))
    ap.add_argument("--tickers", default="", help="comma list to restrict the universe")
    ap.add_argument("--chain-csv", help="use a saved chain instead of fetching")
    ap.add_argument("--r", type=float, default=0.04, help="risk-free rate")
    ap.add_argument("--fee", type=float, default=0.65, help="commission per contract")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison across risk levels")
    ap.add_argument("--out", default=str(ROOT / "output"))
    a = ap.parse_args(argv)

    overrides = json.loads(a.overrides) if a.overrides.strip() else None
    profile = engine.make_profile(a.risk, overrides)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    uni = pd.read_csv(a.universe)
    if a.tickers:
        want = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
        known = uni[uni["ticker"].isin(want)]
        extra = pd.DataFrame({"ticker": [t for t in want if t not in set(uni["ticker"])], "sector": "Other",
                              "beta_fallback": 1.0})
        uni = pd.concat([known, extra], ignore_index=True)

    warnings = []
    if a.chain_csv:
        chain = pd.read_csv(a.chain_csv)
    else:
        # fetch the widest DTE window any profile needs, so the comparison table is meaningful
        lo = min(p.min_dte for p in engine.PROFILES.values()) if not a.no_compare else profile.min_dte
        hi = max(max(p.max_dte for p in engine.PROFILES.values()), profile.max_dte) if not a.no_compare else profile.max_dte
        chain, warnings = fetch_universe(uni, a.provider, lo, hi, log=lambda m: print(m, file=sys.stderr))
        chain.to_csv(out / "chain_raw.csv", index=False)

    scored = engine.score_chain(chain, profile, r=a.r, fee=a.fee)
    res = engine.optimize(scored, a.capital, profile, a.objective, a.target)
    mc = engine.monte_carlo(res.portfolio, a.capital, sims=a.sims, mu=a.r) if not res.portfolio.empty else {}

    comp = None
    if not a.no_compare:
        rows = []
        for name in engine.PROFILES:
            p = engine.make_profile(name)
            r_ = engine.optimize(engine.score_chain(chain, p, r=a.r, fee=a.fee), a.capital, p, "max_premium")
            rows.append({"profile": name, "status": r_.status, **{k: v for k, v in r_.summary.items() if k != "profile"}})
        comp = pd.DataFrame(rows)
        comp.to_csv(out / "risk_comparison.csv", index=False)

    meta = dict(sources=", ".join(sorted(chain["source"].dropna().unique())) if "source" in chain else "csv",
                fetched_at=str(chain["fetched_at"].iloc[0]) if "fetched_at" in chain else "-",
                n_contracts=len(scored), n_eligible=int(scored["eligible"].sum()))
    params = dict(capital=a.capital, objective=a.objective, target=a.target)
    md = report.build(res, mc, params, profile, comp, warnings, meta)

    (out / "report.md").write_text(md, encoding="utf-8")
    scored.to_csv(out / "chain_scored.csv", index=False)
    res.portfolio.to_csv(out / "portfolio.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        dict(status=res.status, params=params, profile=profile.__dict__, summary=res.summary, monte_carlo=mc,
             warnings=warnings, meta=meta), default=float, indent=2, ensure_ascii=False), encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)
    return 0 if not res.portfolio.empty else 2


if __name__ == "__main__":
    sys.exit(main())
