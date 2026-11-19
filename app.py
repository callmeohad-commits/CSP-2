"""Streamlit UI for the CSP optimizer.   streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from csp import engine
from csp.providers import fetch_universe, resolve

ROOT = Path(__file__).parent
st.set_page_config(page_title="CSP Optimizer", page_icon="📈", layout="wide")
st.markdown("""<style>
.stMarkdown, .stMetric, .stAlert, h1, h2, h3, label, .stRadio, .stCaption {direction: rtl; text-align: right;}
</style>""", unsafe_allow_html=True)

# Streamlit Cloud secrets -> env vars used by the providers
try:
    for k in ("BARCHART_API_KEY", "TRADIER_TOKEN", "TRADIER_SANDBOX", "MARKETDATA_TOKEN"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:  # noqa: BLE001  (no secrets file locally)
    pass

UNIVERSE = pd.read_csv(ROOT / "config" / "universe.csv")
FETCH_MIN_DTE = min(p.min_dte for p in engine.PROFILES.values())
FETCH_MAX_DTE = max(p.max_dte for p in engine.PROFILES.values())
RISK_HE = {"very_low": "נמוך מאוד", "low": "נמוך", "medium": "בינוני", "high": "גבוה"}


@st.cache_data(ttl=900, show_spinner=False)
def load_chain(tickers: tuple, provider: str):
    uni = UNIVERSE[UNIVERSE["ticker"].isin(tickers)]
    extra = [t for t in tickers if t not in set(UNIVERSE["ticker"])]
    if extra:
        uni = pd.concat([uni, pd.DataFrame({"ticker": extra, "sector": "Other", "beta_fallback": 1.0})])
    log = []
    chain, warnings = fetch_universe(uni, provider, FETCH_MIN_DTE, FETCH_MAX_DTE, log=log.append)
    return chain, warnings, log


def money(x):
    return f"${x:,.0f}"


# --------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("פרמטרים")
capital = sb.number_input("הון (USD)", min_value=10_000, max_value=50_000_000, value=250_000, step=10_000)
risk = sb.selectbox("רמת סיכון", list(engine.PROFILES), index=1, format_func=lambda k: f"{RISK_HE[k]} ({k})")
objective = sb.radio("מטרה", ["max_premium", "target"],
                     format_func=lambda k: "פרמיה מקסימלית" if k == "max_premium" else "יעד תשואה בסיכון מינימלי")
target = None
if objective == "target":
    target = sb.number_input("יעד תשואה שנתית על ההון (%)", 1.0, 100.0, 8.0, 0.5) / 100

base = engine.PROFILES[risk]
with sb.expander("כיוון פרמטרי סיכון"):
    ov = dict(
        max_delta=st.slider("דלתא מקס' לחוזה", 0.05, 0.50, base.max_delta, 0.01),
        max_avg_delta=st.slider("דלתא ממוצעת מקס'", 0.05, 0.50, base.max_avg_delta, 0.01),
        min_dte=st.slider("ימים לפקיעה - מינימום", 1, 90, base.min_dte),
        max_dte=st.slider("ימים לפקיעה - מקסימום", 7, 120, base.max_dte),
        max_weight_ticker=st.slider("משקל מקס' למניה", 0.05, 1.0, base.max_weight_ticker, 0.05),
        max_weight_sector=st.slider("משקל מקס' לסקטור", 0.10, 1.0, base.max_weight_sector, 0.05),
        cash_buffer=st.slider("כרית מזומן", 0.0, 0.5, base.cash_buffer, 0.05),
        max_stress_loss=st.slider(f"הפסד מקס' בירידת שוק {base.stress_shock:.0%} × בטא (מההון)",
                                  0.01, 0.40, base.max_stress_loss, 0.01),
        max_gap_loss=st.slider(f"הפסד מקס' מצניחת מניה {base.gap_shock:.0%} (מההון)",
                               0.005, 0.15, base.max_gap_loss, 0.005),
        min_cushion=st.slider("כרית מינ' עד נקודת איזון", 0.0, 0.20, base.min_cushion, 0.01),
        min_tickers=st.slider("מינ' מניות שונות", 1, 15, base.min_tickers),
        avoid_earnings=st.checkbox("להימנע מחוזים שחוצים דוחות", base.avoid_earnings),
    )
    if ov["max_dte"] > FETCH_MAX_DTE:
        st.caption(f"נתונים נשלפים עד {FETCH_MAX_DTE} יום.")
profile = engine.make_profile(risk, ov)

sb.header("נתונים")
providers_ok = [p.name for p in resolve("auto")]
provider = sb.selectbox("מקור", ["auto", "barchart", "tradier", "marketdata", "yfinance", "cboe"],
                        help="auto = הראשון הזמין: " + " → ".join(providers_ok))
tickers = sb.multiselect("מניות", UNIVERSE["ticker"].tolist(), default=UNIVERSE["ticker"].tolist()[:15],
                         help="יותר מניות = שליפה ארוכה יותר (בערך 5-10 שניות למניה)")
custom = sb.text_input("מניות נוספות (מופרדות בפסיק)", "")
tickers = tuple(dict.fromkeys(tickers + [t.strip().upper() for t in custom.split(",") if t.strip()]))
upload = sb.file_uploader("או: שרשרת שמורה (chain_raw.csv)", type="csv")
fee = sb.number_input("עמלה לחוזה ($)", 0.0, 10.0, 0.65, 0.05)
rate = sb.number_input("ריבית חסרת סיכון (%)", 0.0, 10.0, 4.0, 0.25) / 100
go = sb.button("הרץ", type="primary", use_container_width=True)
if sb.button("רענן נתונים (נקה מטמון)"):
    load_chain.clear()

# --------------------------------------------------------------------------- main
st.title("CSP Optimizer - תמהיל מניות גדולות לפרמיה מקסימלית")
if not providers_ok:
    st.warning("לא הוגדר מקור נתונים. הוסיפו TRADIER_TOKEN / BARCHART_API_KEY / MARKETDATA_TOKEN ב-Secrets.")

if go:
    try:
        if upload is not None:
            chain, warnings, log = pd.read_csv(upload), [], ["loaded from upload"]
        else:
            if not tickers:
                st.error("בחרו לפחות מניה אחת.")
                st.stop()
            with st.spinner(f"שולף שרשראות אופציות ל-{len(tickers)} מניות..."):
                chain, warnings, log = load_chain(tickers, provider)
        st.session_state["data"] = (chain, warnings, log)
    except Exception as e:  # noqa: BLE001
        st.error(f"שליפת הנתונים נכשלה: {e}")
        st.stop()

if "data" not in st.session_state:
    st.info("בחרו פרמטרים בסרגל הצד ולחצו **הרץ**.")
    st.stop()

chain, warnings, log = st.session_state["data"]
scored = engine.score_chain(chain, profile, r=rate, fee=fee)
res = engine.optimize(scored, capital, profile, objective, target)
src = ", ".join(sorted(chain["source"].dropna().unique())) if "source" in chain else "csv"
when = chain["fetched_at"].iloc[0] if "fetched_at" in chain else "-"
st.caption(f"מקור: {src} · נשלף {when} (ניו יורק) · {len(scored)} חוזי Put, {int(scored['eligible'].sum())} עוברים את סינון הסיכון")

STATUS = {"max_premium": ("success", "התמהיל עם הפרמיה המקסימלית בתוך מגבלות הסיכון"),
          "target_met": ("success", "היעד הושג בסיכון המינימלי"),
          "target_unreachable_max_premium_shown": ("warning", "היעד לא בר-השגה ברמת הסיכון הזו - מוצג המקסימום האפשרי"),
          "infeasible": ("error", "אין תמהיל שעומד בכל המגבלות - הקלו את מגבלות הסיכון"),
          "no_eligible_contracts": ("error", "אין חוזים שעוברים את הסינון - הקלו דלתא/ימים/נזילות")}
kind, msg = STATUS.get(res.status, ("info", res.status))
getattr(st, kind)(msg)

if not res.portfolio.empty:
    s = res.summary
    mc = engine.monte_carlo(res.portfolio, capital, sims=20000, mu=rate)
    c = st.columns(5)
    c[0].metric("פרמיה שתיגבה עכשיו", money(s["premium_collected"]))
    c[1].metric("תשואה שנתית על ההון", f"{s['ann_yield_on_capital']:.1%}",
                f"{s['ann_yield_on_collateral']:.1%} על הביטחונות", delta_color="off")
    c[2].metric("הון נדרש", money(s["capital_required"]), f"{s['utilization']:.0%} מההון", delta_color="off")
    c[3].metric("הפסד בירידת שוק", money(s["stress_loss"]), f"{s['stress_loss_pct']:.1%} מההון", delta_color="off")
    c[4].metric("הסתברות להפסד (MC)", f"{mc['p_loss']:.0%}",
                f"הקצאה: {mc['p_any_assignment']:.0%}", delta_color="off")

tab_mix, tab_cmp, tab_risk, tab_all, tab_log = st.tabs(["התמהיל", "השוואת רמות סיכון", "סיכון ו-Monte Carlo",
                                                        "כל החוזים", "אזהרות ולוג"])
with tab_mix:
    if res.portfolio.empty:
        st.write("אין תמהיל.")
    else:
        p = res.portfolio.sort_values("total_premium", ascending=False)
        view = pd.DataFrame({
            "מניה": p["ticker"], "סקטור": p["sector"], "פקיעה": p["expiry"], "ימים": p["expiry_dte"],
            "Strike": p["strike"], "Spot": p["spot"].round(2), "חוזים": p["contracts"],
            "פרמיה לחוזה": p["premium"].round(0), "פרמיה כוללת": p["total_premium"].round(0),
            "הון נדרש": p["total_collateral"], "משקל": (p["weight"] * 100).round(1),
            "דלתא": p["abs_delta"].round(3), "כרית %": (p["cushion"] * 100).round(1),
            "תשואה שנתית %": (p["ann_yield"] * 100).round(1), "P(OTM) %": (p["p_otm"] * 100).round(0),
            "IV %": (p["iv"] * 100).round(1)})
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption("האופטימיזציה ממקסמת פרמיה בקצב שנתי, כך שחוזים באורכים שונים ניתנים להשוואה.")
        st.bar_chart(p.groupby("ticker")["total_collateral"].sum(), horizontal=True)
        st.download_button("הורדת התמהיל (CSV)", p.to_csv(index=False).encode(), "portfolio.csv", "text/csv")

with tab_cmp:
    rows = []
    for name in engine.PROFILES:
        pr = engine.make_profile(name)
        r_ = engine.optimize(engine.score_chain(chain, pr, r=rate, fee=fee), capital, pr, "max_premium")
        sm = r_.summary
        rows.append({"רמת סיכון": RISK_HE[name], "פרמיה עכשיו": round(sm.get("premium_collected", 0)),
                     "תשואה שנתית %": round(sm.get("ann_yield_on_capital", 0) * 100, 1),
                     "הון נדרש": round(sm.get("capital_required", 0)),
                     "הפסד בירידת שוק %": round(sm.get("stress_loss_pct", 0) * 100, 1),
                     "דלתא ממוצעת": round(sm.get("avg_delta", 0), 3), "מניות": sm.get("tickers", 0)})
    cmp = pd.DataFrame(rows)
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.caption("כל רמה עם הפרמטרים המקוריים שלה (בלי הכיוונים מסרגל הצד), באותו הון.")
    st.bar_chart(cmp.set_index("רמת סיכון")[["תשואה שנתית %", "הפסד בירידת שוק %"]])

with tab_risk:
    if not res.portfolio.empty:
        a, b = st.columns(2)
        a.markdown(f"""
| מדד | ערך |
|---|---|
| דלתא ממוצעת | {s['avg_delta']:.3f} |
| ימים לפקיעה ממוצע | {s['avg_dte']:.0f} |
| כרית ממוצעת | {s['avg_cushion']:.1%} |
| הפסד מקס' מצניחת מניה אחת | {s['worst_gap_loss_pct']:.1%} מההון |
| משקל סקטור מקס' | {s['max_sector_weight']:.0%} |
| יתרון צפוי שנתי (IV מול HV) | {s['expected_edge_ann']:.2%} |
""")
        b.markdown(f"""
| Monte Carlo (עד פקיעה) | ערך |
|---|---|
| רווח צפוי למחזור | {money(mc['expected_pnl'])} |
| הסתברות להפסד | {mc['p_loss']:.1%} |
| הסתברות להקצאה (חוזה אחד לפחות) | {mc['p_any_assignment']:.1%} |
| VaR / CVaR 95% | {money(mc['VaR95'])} / {money(mc['CVaR95'])} |
| VaR / CVaR 99% | {money(mc['VaR99'])} / {money(mc['CVaR99'])} |
""")

with tab_all:
    only = st.checkbox("רק חוזים שעוברים את הסינון", True)
    df = scored[scored["eligible"]] if only else scored
    cols = ["ticker", "expiry", "expiry_dte", "strike", "spot", "bid", "ask", "iv", "abs_delta", "ann_yield",
            "cushion", "p_otm", "stress_loss_pct", "gap_loss_pct", "edge_ann", "ret_risk", "open_interest", "eligible"]
    st.dataframe(df[[c for c in cols if c in df]].sort_values("ret_risk", ascending=False),
                 hide_index=True, use_container_width=True)
    st.download_button("הורדת השרשרת הגולמית (לשימוש חוזר)", chain.to_csv(index=False).encode(),
                       "chain_raw.csv", "text/csv")

with tab_log:
    for w in warnings:
        st.write("• " + w)
    with st.expander("לוג שליפה"):
        st.code("\n".join(log) or "-")

st.caption("כלי ניתוח בלבד, לא ייעוץ השקעות. מחירים לפי mid; החישוב עד פקיעה, ללא הקצאה מוקדמת, גלגול או מס.")
