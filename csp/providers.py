"""Live option-chain providers. All return the same normalized put chain:

    ticker, expiry (YYYY-MM-DD), expiry_dte, strike, bid, ask, iv, delta,
    open_interest, volume, spot, source

Provider         | key / secret        | data                          | notes
-----------------|---------------------|-------------------------------|---------------------------------
barchart         | BARCHART_API_KEY    | Barchart OnDemand getEquityOptions | paid, official Barchart API
tradier          | TRADIER_TOKEN       | real-time chains + greeks     | free w/ Tradier brokerage account
                 | (TRADIER_SANDBOX=1) | sandbox: 15-min delayed, no greeks
marketdata       | MARKETDATA_TOKEN    | chains + IV + greeks          | 1 credit per contract returned
yfinance         | none                | Yahoo, IV only                | unofficial; often throttled on CI
cboe             | none                | Cboe delayed-quotes JSON      | unofficial public endpoint, 15-min delay

Scraping barchart.com web pages is deliberately NOT implemented: use OnDemand.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

NY = ZoneInfo("America/New_York")
COLS = ["ticker", "expiry", "expiry_dte", "strike", "bid", "ask", "iv", "delta",
        "open_interest", "volume", "spot", "source"]
UA = {"User-Agent": "csp-optimizer/1.0 (+github actions)"}


def today_ny() -> dt.date:
    return dt.datetime.now(NY).date()


def http_get(url, params=None, headers=None, timeout=20, retries=3):
    """Single HTTP entry point (patched in tests). Retries on 429/5xx."""
    for i in range(retries):
        r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
            time.sleep(2 ** i * 2)
            continue
        r.raise_for_status()
        return r.json()


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _dte(expiry: str) -> int:
    return (dt.date.fromisoformat(expiry[:10]) - today_ny()).days


def _frame(rows, ticker, spot, source):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["ticker"], df["source"] = ticker, source
    if "spot" not in df or df["spot"].isna().all():
        df["spot"] = spot
    df["expiry_dte"] = df["expiry"].map(_dte)
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    for c in ["strike", "bid", "ask", "iv", "delta", "open_interest", "volume", "spot"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLS]


OCC = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ(sym: str):
    m = OCC.match(sym.replace(" ", ""))
    if not m:
        return None
    _, ymd, cp, k = m.groups()
    return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}", cp, int(k) / 1000

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class Provider:
    name = "base"
    needs = None                      # env var required

    def available(self) -> bool:
        return self.needs is None or bool(os.getenv(self.needs))

    def chain(self, ticker, min_dte, max_dte) -> pd.DataFrame:
        raise NotImplementedError

    def history(self, ticker, days=400) -> pd.Series | None:
        return None


class Tradier(Provider):
    name, needs = "tradier", "TRADIER_TOKEN"

    @property
    def base(self):
        return "https://sandbox.tradier.com/v1" if os.getenv("TRADIER_SANDBOX") == "1" else "https://api.tradier.com/v1"

    def _get(self, path, **params):
        return http_get(f"{self.base}{path}", params,
                        {"Authorization": f"Bearer {os.environ['TRADIER_TOKEN']}", "Accept": "application/json"})

    def chain(self, ticker, min_dte, max_dte):
        q = _as_list(self._get("/markets/quotes", symbols=ticker)["quotes"]["quote"])[0]
        spot = q.get("last") or (q.get("bid", 0) + q.get("ask", 0)) / 2
        exps = _as_list((self._get("/markets/options/expirations", symbol=ticker,
                                   includeAllRoots="true").get("expirations") or {}).get("date"))
        rows = []
        for e in exps:
            if not (min_dte <= _dte(e) <= max_dte):
                continue
            data = self._get("/markets/options/chains", symbol=ticker, expiration=e, greeks="true")
            for o in _as_list((data.get("options") or {}).get("option")):
                if o.get("option_type") != "put":
                    continue
                g = o.get("greeks") or {}
                rows.append(dict(expiry=o.get("expiration_date", e), strike=o["strike"], bid=o.get("bid"),
                                 ask=o.get("ask"), iv=g.get("mid_iv") or g.get("smv_vol"), delta=g.get("delta"),
                                 open_interest=o.get("open_interest"), volume=o.get("volume")))
            time.sleep(0.5)                                   # stay under 120 req/min
        return _frame(rows, ticker, spot, self.name)

    def history(self, ticker, days=400):
        start = (today_ny() - dt.timedelta(days=days)).isoformat()
        d = _as_list((self._get("/markets/history", symbol=ticker, interval="daily", start=start)
                      .get("history") or {}).get("day"))
        if not d:
            return None
        return pd.Series({x["date"]: x["close"] for x in d}, dtype=float)


class MarketData(Provider):
    name, needs = "marketdata", "MARKETDATA_TOKEN"
    base = "https://api.marketdata.app/v1"

    def _get(self, path, **params):
        return http_get(f"{self.base}{path}", params, {"Authorization": f"Bearer {os.environ['MARKETDATA_TOKEN']}"})

    def chain(self, ticker, min_dte, max_dte):
        t = today_ny()
        d = self._get(f"/options/chain/{ticker}/", side="put", range="otm",
                      **{"from": (t + dt.timedelta(days=min_dte)).isoformat(),
                         "to": (t + dt.timedelta(days=max_dte)).isoformat()})
        if d.get("s") != "ok":
            return _frame([], ticker, np.nan, self.name)
        n = len(d["optionSymbol"])
        get = lambda k: d.get(k) or [None] * n
        rows = [dict(expiry=dt.datetime.fromtimestamp(get("expiration")[i], NY).date().isoformat(),
                     strike=get("strike")[i], bid=get("bid")[i], ask=get("ask")[i], iv=get("iv")[i],
                     delta=get("delta")[i], open_interest=get("openInterest")[i], volume=get("volume")[i],
                     spot=get("underlyingPrice")[i]) for i in range(n)]
        return _frame(rows, ticker, np.nan, self.name)

    def history(self, ticker, days=400):
        t = today_ny()
        d = self._get(f"/stocks/candles/D/{ticker}/", **{"from": (t - dt.timedelta(days=days)).isoformat(),
                                                         "to": t.isoformat()})
        if d.get("s") != "ok":
            return None
        idx = [dt.datetime.fromtimestamp(x, NY).date().isoformat() for x in d["t"]]
        return pd.Series(d["c"], index=idx, dtype=float)


class Barchart(Provider):
    """Barchart OnDemand (official, paid). Field names follow the OnDemand docs;
    several aliases are accepted because field sets depend on your entitlement."""
    name, needs = "barchart", "BARCHART_API_KEY"
    base = "https://ondemand.websol.barchart.com"

    def _get(self, path, **params):
        return http_get(f"{self.base}/{path}", {"apikey": os.environ["BARCHART_API_KEY"], **params})

    @staticmethod
    def _pick(o, *keys):
        for k in keys:
            if o.get(k) not in (None, ""):
                return o[k]
        return None

    def chain(self, ticker, min_dte, max_dte):
        q = (self._get("getQuote.json", symbols=ticker).get("results") or [{}])[0]
        spot = self._pick(q, "lastPrice", "close", "previousClose")
        res = self._get("getEquityOptions.json", underlying_symbols=ticker, type="Put",
                        fields="bid,ask,volatility,delta,openInterest,volume,expirationDate,strike").get("results") or []
        rows = []
        for o in res:
            exp = self._pick(o, "expirationDate", "expiration_date", "expiration")
            if not exp:
                continue
            exp = str(exp)[:10]
            if not (min_dte <= _dte(exp) <= max_dte):
                continue
            rows.append(dict(expiry=exp, strike=self._pick(o, "strike", "strikePrice"),
                             bid=self._pick(o, "bid", "bidPrice"), ask=self._pick(o, "ask", "askPrice"),
                             iv=self._pick(o, "volatility", "impliedVolatility"), delta=o.get("delta"),
                             open_interest=self._pick(o, "openInterest", "open_interest"), volume=o.get("volume")))
        return _frame(rows, ticker, spot, self.name)

    def history(self, ticker, days=400):
        start = (today_ny() - dt.timedelta(days=days)).strftime("%Y%m%d")
        res = self._get("getHistory.json", symbol=ticker, type="daily", startDate=start).get("results") or []
        if not res:
            return None
        return pd.Series({x["tradingDay"]: x["close"] for x in res}, dtype=float)


class Cboe(Provider):
    """Public delayed-quotes JSON behind cboe.com quote pages. Unofficial: no SLA,
    format may change; check Cboe's terms for your use."""
    name = "cboe"
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"

    def chain(self, ticker, min_dte, max_dte):
        d = http_get(self.url.format(ticker.replace(".", "")))["data"]
        spot = d.get("current_price") or d.get("close")
        rows = []
        for o in d.get("options", []):
            p = parse_occ(o.get("option", ""))
            if not p or p[1] != "P" or not (min_dte <= _dte(p[0]) <= max_dte):
                continue
            rows.append(dict(expiry=p[0], strike=p[2], bid=o.get("bid"), ask=o.get("ask"), iv=o.get("iv"),
                             delta=o.get("delta"), open_interest=o.get("open_interest"), volume=o.get("volume")))
        return _frame(rows, ticker, spot, self.name)


class YFinance(Provider):
    """Yahoo Finance via yfinance (unofficial). IV only, no delta (computed locally)."""
    name = "yfinance"

    def available(self):
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def chain(self, ticker, min_dte, max_dte):
        import yfinance as yf
        tk = yf.Ticker(ticker.replace(".", "-"))
        spot = float(tk.history(period="5d")["Close"].iloc[-1])
        rows = []
        for e in tk.options:
            if not (min_dte <= _dte(e) <= max_dte):
                continue
            puts = tk.option_chain(e).puts
            for o in puts.itertuples():
                rows.append(dict(expiry=e, strike=o.strike, bid=o.bid, ask=o.ask, iv=o.impliedVolatility,
                                 open_interest=o.openInterest, volume=o.volume))
        return _frame(rows, ticker, spot, self.name)

    def history(self, ticker, days=400):
        import yfinance as yf
        h = yf.Ticker(ticker.replace(".", "-")).history(period=f"{days}d")["Close"]
        if h.empty:
            return None
        h.index = h.index.strftime("%Y-%m-%d")
        return h

    def next_earnings(self, ticker):
        import yfinance as yf
        cal = yf.Ticker(ticker.replace(".", "-")).calendar
        ed = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
        ed = _as_list(ed)
        return ed[0] if ed else None


REGISTRY = {c.name: c for c in (Barchart, Tradier, MarketData, YFinance, Cboe)}
AUTO_ORDER = ["barchart", "tradier", "marketdata", "yfinance", "cboe"]


def resolve(name: str) -> list[Provider]:
    names = AUTO_ORDER if name == "auto" else [n.strip() for n in name.split(",")]
    provs = [REGISTRY[n]() for n in names]
    return [p for p in provs if p.available()]

# ---------------------------------------------------------------------------
# Enrichment: HV, beta, earnings
# ---------------------------------------------------------------------------


def hv_beta(px: pd.Series | None, spy: pd.Series | None, hv_days=60):
    if px is None or len(px) < hv_days + 2:
        return np.nan, np.nan
    r = np.log(px.astype(float)).diff().dropna()
    hv = float(r.tail(hv_days).std() * np.sqrt(252))
    beta = np.nan
    if spy is not None and len(spy) > 60:
        m = np.log(spy.astype(float)).diff().dropna()
        j = pd.concat([r, m], axis=1, join="inner").dropna().tail(252)
        if len(j) > 60:
            beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
    return hv, beta


def fetch_universe(universe: pd.DataFrame, provider="auto", min_dte=7, max_dte=60,
                   with_earnings=True, log=print):
    """Fetch chains for every ticker with provider fallback, then add hv/beta/sector/earnings."""
    provs = resolve(provider)
    if not provs:
        raise RuntimeError(f"No data provider available for '{provider}'. Set an API key secret "
                           "(BARCHART_API_KEY / TRADIER_TOKEN / MARKETDATA_TOKEN) or install yfinance.")
    log(f"providers in order: {[p.name for p in provs]}")
    hist_provs = provs + [p for p in (YFinance(),) if p.available() and p.name not in {q.name for q in provs}]

    def get_hist(t):
        for p in hist_provs:
            try:
                h = p.history(t)
                if h is not None and len(h) > 60:
                    return h
            except Exception as e:  # noqa: BLE001
                log(f"  history {t} via {p.name} failed: {e}")
        return None

    spy = get_hist("SPY")
    frames, warnings = [], []
    yf_ = YFinance() if YFinance().available() else None
    for row in universe.itertuples():
        t = row.ticker
        df = None
        for p in provs:
            try:
                df = p.chain(t, min_dte, max_dte)
                if df is not None and len(df):
                    log(f"  {t}: {len(df)} puts via {p.name}")
                    break
            except Exception as e:  # noqa: BLE001
                log(f"  {t} via {p.name} failed: {e}")
                df = None
        if df is None or df.empty:
            warnings.append(f"{t}: no option data from any provider")
            continue
        hv, beta = hv_beta(get_hist(t), spy)
        if np.isnan(beta):
            beta = getattr(row, "beta_fallback", 1.0)
            warnings.append(f"{t}: beta from static fallback ({beta})")
        if np.isnan(hv):
            warnings.append(f"{t}: no price history -> HV unknown, edge uses IV")
        df["sector"], df["hv"], df["beta"] = row.sector, hv, beta
        df["earnings_in_window"] = np.nan
        if with_earnings and yf_:
            try:
                ed = yf_.next_earnings(t)
                if ed is not None:
                    ed = pd.Timestamp(ed).date()
                    df["next_earnings"] = ed.isoformat()
                    df["earnings_in_window"] = pd.to_datetime(df["expiry"]).dt.date >= ed
            except Exception as e:  # noqa: BLE001
                log(f"  earnings {t} failed: {e}")
        if df["earnings_in_window"].isna().all():
            warnings.append(f"{t}: earnings date unknown - check manually")
        frames.append(df)
    if not frames:
        raise RuntimeError("No option data fetched for any ticker")
    out = pd.concat(frames, ignore_index=True)
    out["fetched_at"] = dt.datetime.now(NY).isoformat(timespec="minutes")
    return out, warnings
