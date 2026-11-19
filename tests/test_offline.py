"""Offline tests: mock each provider's HTTP payload (in its documented format) and
run the full pipeline. Run with:  python -m pytest -q   (or python tests/test_offline.py)"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csp import engine, providers as P  # noqa: E402

UNIVERSE = [("AAPL", "Technology", 230, 0.26, 1.2), ("MSFT", "Technology", 420, 0.23, 0.95),
            ("JPM", "Financials", 240, 0.22, 1.05), ("JNJ", "Health Care", 155, 0.17, 0.5),
            ("XOM", "Energy", 115, 0.24, 0.85), ("PG", "Consumer Staples", 165, 0.16, 0.45),
            ("AMZN", "Consumer Discretionary", 200, 0.31, 1.2), ("META", "Communication", 580, 0.34, 1.25)]
TODAY = P.today_ny()
EXPS = [(TODAY + dt.timedelta(days=d)).isoformat() for d in (10, 17, 24, 31, 38, 45, 52)]


def synth(ticker):
    _, _, S, iv0, _ = next(u for u in UNIVERSE if u[0] == ticker)
    step = 2.5 if S < 300 else 5
    rows = []
    for e in EXPS:
        T = P._dte(e) / 365
        for m in np.arange(0.78, 1.0, 0.0125):
            K = round(S * m / step) * step
            iv = iv0 * (1 + 0.9 * np.log(S / K))
            px, d = engine.bs_put(S, K, T, 0.04, iv)
            px = float(px)
            if px < 0.05:
                continue
            h = max(0.01, px * 0.03)
            rows.append(dict(e=e, K=K, bid=round(px - h, 2), ask=round(px + h, 2), iv=float(iv), d=float(d)))
    return S, pd.DataFrame(rows).drop_duplicates(["e", "K"])


def history(ticker, n=300, seed=0):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    m = np.random.default_rng(seed).normal(0, 0.011, n)
    beta = 1.0 if ticker == "SPY" else next(u[4] for u in UNIVERSE if u[0] == ticker)
    r = m if ticker == "SPY" else beta * m + rng.normal(0, 0.009, n)
    days = pd.bdate_range(end=pd.Timestamp(TODAY), periods=n).strftime("%Y-%m-%d")
    return list(days), list(100 * np.exp(np.cumsum(r)))


def fake_http(url, params=None, headers=None, **_):
    params = params or {}
    # ---- Tradier
    if "tradier.com" in url:
        sym = params.get("symbol") or params.get("symbols")
        if url.endswith("/markets/quotes"):
            return {"quotes": {"quote": {"symbol": sym, "last": synth(sym)[0]}}}
        if url.endswith("/options/expirations"):
            return {"expirations": {"date": EXPS}}
        if url.endswith("/options/chains"):
            S, df = synth(sym)
            df = df[df.e == params["expiration"]]
            opts = [{"symbol": "x", "option_type": "put", "strike": r.K, "bid": r.bid, "ask": r.ask,
                     "open_interest": 5000, "volume": 100, "expiration_date": r.e,
                     "greeks": {"delta": r.d, "mid_iv": r.iv}} for r in df.itertuples()]
            opts.append({**opts[0], "option_type": "call"})
            return {"options": {"option": opts}}
        if url.endswith("/markets/history"):
            d, c = history(sym)
            return {"history": {"day": [{"date": a, "close": b} for a, b in zip(d, c)]}}
    # ---- MarketData.app (columnar)
    if "marketdata.app" in url:
        sym = url.rstrip("/").split("/")[-1]
        if "/options/chain/" in url:
            S, df = synth(sym)
            ts = [int(dt.datetime.combine(dt.date.fromisoformat(e), dt.time(16), P.NY).timestamp()) for e in df.e]
            return {"s": "ok", "optionSymbol": ["x"] * len(df), "expiration": ts, "strike": df.K.tolist(),
                    "bid": df.bid.tolist(), "ask": df.ask.tolist(), "iv": df.iv.tolist(), "delta": df.d.tolist(),
                    "openInterest": [4000] * len(df), "volume": [50] * len(df), "underlyingPrice": [S] * len(df)}
        if "/stocks/candles/" in url:
            d, c = history(sym)
            return {"s": "ok", "t": [int(pd.Timestamp(x, tz=P.NY).timestamp()) for x in d], "c": c}
    # ---- Barchart OnDemand
    if "barchart.com" in url:
        sym = params.get("symbols") or params.get("underlying_symbols") or params.get("symbol")
        if "getQuote" in url:
            return {"status": {"code": 200}, "results": [{"symbol": sym, "lastPrice": synth(sym)[0]}]}
        if "getEquityOptions" in url:
            S, df = synth(sym)
            return {"status": {"code": 200}, "results": [
                {"underlying_symbol": sym, "type": "Put", "strike": r.K, "expirationDate": r.e, "bid": r.bid,
                 "ask": r.ask, "volatility": r.iv * 100, "delta": r.d, "openInterest": 3000} for r in df.itertuples()]}
        if "getHistory" in url:
            d, c = history(sym)
            return {"results": [{"tradingDay": a, "close": b} for a, b in zip(d, c)]}
    # ---- Cboe
    if "cboe.com" in url:
        sym = url.split("/")[-1].replace(".json", "")
        S, df = synth(sym)
        opts = [{"option": f"{sym}{r.e[2:4]}{r.e[5:7]}{r.e[8:10]}P{int(round(r.K * 1000)):08d}", "bid": r.bid,
                 "ask": r.ask, "iv": r.iv, "delta": r.d, "open_interest": 2000, "volume": 10} for r in df.itertuples()]
        return {"data": {"symbol": sym, "current_price": S, "options": opts}}
    raise AssertionError(f"unexpected URL {url}")


def _setup(monkey_env):
    P.http_get = fake_http
    for k in ("TRADIER_TOKEN", "MARKETDATA_TOKEN", "BARCHART_API_KEY"):
        os.environ[k] = "test"
    P.Tradier.chain.__globals__["time"].sleep = lambda *_: None


def universe():
    return pd.DataFrame([(u[0], u[1], u[4]) for u in UNIVERSE], columns=["ticker", "sector", "beta_fallback"])


def test_each_provider_parses():
    _setup(None)
    ref = None
    for name in ["tradier", "marketdata", "barchart", "cboe"]:
        df = P.REGISTRY[name]().chain("AAPL", 7, 60)
        assert len(df) > 50, name
        assert df["spot"].iloc[0] == 230
        assert set(df["expiry"]) <= set(EXPS)
        df = df.sort_values(["expiry", "strike"]).reset_index(drop=True)
        if ref is None:
            ref = df
        else:
            pd.testing.assert_series_equal(ref["strike"], df["strike"], check_names=False)
            np.testing.assert_allclose(ref["bid"], df["bid"])
        print(f"  {name}: {len(df)} puts OK")


def test_occ_parser():
    assert P.parse_occ("AAPL261218P00200000") == ("2026-12-18", "P", 200.0)
    assert P.parse_occ("BRKB261218P00450500") == ("2026-12-18", "P", 450.5)


def test_pipeline_all_providers_and_fallback(tmp="output_test"):
    _setup(None)
    # fallback: first provider explodes, second works
    class Broken(P.Provider):
        name = "broken"
        def chain(self, *a):
            raise RuntimeError("403 blocked")
    P.REGISTRY["broken"] = Broken
    chain, warns = P.fetch_universe(universe(), "broken,tradier", 7, 60, with_earnings=False, log=lambda m: None)
    assert set(chain["source"]) == {"tradier"}
    assert chain["hv"].notna().all() and chain["beta"].notna().all()
    b = chain.groupby("ticker")["beta"].first()
    assert abs(b["PG"] - 0.45) < 0.2 and abs(b["AMZN"] - 1.2) < 0.25, b
    for name in ["very_low", "low", "medium", "high"]:
        p = engine.make_profile(name)
        res = engine.optimize(engine.score_chain(chain, p), 250_000, p, "max_premium")
        s = res.summary
        assert res.status == "max_premium", (name, res.status)
        # constraints hold
        assert s["capital_required"] <= 250_000 * (1 - p.cash_buffer) + 1e-6
        assert s["stress_loss"] <= p.max_stress_loss * 250_000 + 1e-6
        assert s["avg_delta"] <= p.max_avg_delta + 1e-9
        assert s["tickers"] >= min(p.min_tickers, chain["ticker"].nunique())
        assert (res.portfolio["abs_delta"] <= p.max_delta).all()
        w = res.portfolio.groupby("ticker")["total_collateral"].sum() / 250_000
        assert (w <= p.max_weight_ticker + 1e-9).all()
        print(f"  {name:9s} premium ${s['premium_collected']:,.0f}  ann {s['ann_yield_on_capital']:.2%}  "
              f"capital ${s['capital_required']:,.0f}  stress {s['stress_loss_pct']:.2%}")


def test_target_mode():
    _setup(None)
    chain, _ = P.fetch_universe(universe(), "cboe", 7, 60, with_earnings=False, log=lambda m: None)
    p = engine.make_profile("low", {"max_delta": 0.15})
    sc = engine.score_chain(chain, p)
    mx = engine.optimize(sc, 300_000, p, "max_premium").summary["ann_yield_on_capital"]
    r = engine.optimize(sc, 300_000, p, "target", target=mx * 0.6)
    assert r.status == "target_met" and r.summary["ann_yield_on_capital"] >= mx * 0.6 - 1e-9
    assert r.summary["stress_loss"] <= engine.optimize(sc, 300_000, p, "max_premium").summary["stress_loss"] + 1e-6
    r2 = engine.optimize(sc, 300_000, p, "target", target=mx * 1.5)
    assert r2.status == "target_unreachable_max_premium_shown"


if __name__ == "__main__":
    for f in [test_occ_parser, test_each_provider_parses, test_pipeline_all_providers_and_fallback, test_target_mode]:
        print(f.__name__); f()
    print("ALL PASSED")
