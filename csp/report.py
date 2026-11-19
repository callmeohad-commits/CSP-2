"""Markdown report (used for GitHub job summary and output/report.md)."""
from __future__ import annotations

import pandas as pd

STATUS_HE = {
    "max_premium": "נבחר התמהיל עם הפרמיה המקסימלית בתוך מגבלות הסיכון",
    "target_met": "היעד הושג בסיכון המינימלי",
    "target_unreachable_max_premium_shown": "היעד לא בר-השגה ברמת הסיכון הזו - מוצג המקסימום האפשרי",
    "infeasible": "אין תמהיל שעומד בכל המגבלות",
    "no_eligible_contracts": "אין חוזים שעוברים את סינון הסיכון",
}


def _money(x):
    return f"${x:,.0f}"


def _pct(x, d=1):
    return f"{x:.{d}%}"


def build(result, mc, params, profile, comparison: pd.DataFrame | None, warnings, meta) -> str:
    s = result.summary
    L = [f"# CSP - תמהיל מניות גדולות לפרמיה מקסימלית",
         "",
         f"**נתונים:** {meta['sources']} · נשלף {meta['fetched_at']} (שעון ניו יורק) · "
         f"{meta['n_contracts']} חוזי Put, {meta['n_eligible']} עוברים את סינון הסיכון",
         "",
         "## פרמטרים",
         "",
         "| פרמטר | ערך |", "|---|---|",
         f"| הון | {_money(params['capital'])} |",
         f"| רמת סיכון | `{profile.name}` |",
         f"| מטרה | `{params['objective']}`" + (f" (יעד {_pct(params['target'])} שנתי)" if params.get("target") else "") + " |",
         f"| דלתא מקסימלית לחוזה / ממוצעת | {profile.max_delta:.2f} / {profile.max_avg_delta:.2f} |",
         f"| ימים לפקיעה | {profile.min_dte}-{profile.max_dte} |",
         f"| משקל מקס' למניה / לסקטור | {_pct(profile.max_weight_ticker,0)} / {_pct(profile.max_weight_sector,0)} |",
         f"| כרית מזומן | {_pct(profile.cash_buffer,0)} |",
         f"| הפסד מקס' בירידת שוק {_pct(profile.stress_shock,0)} × בטא | {_pct(profile.max_stress_loss)} מההון |",
         f"| הפסד מקס' מצניחת מניה בודדת {_pct(profile.gap_shock,0)} | {_pct(profile.max_gap_loss)} מההון |",
         f"| הימנעות מדוחות | {'כן' if profile.avoid_earnings else 'לא'} |",
         "",
         f"## תוצאה: {STATUS_HE.get(result.status, result.status)}", ""]
    if result.portfolio.empty:
        L.append("לא נמצא תמהיל. נסו להקל את מגבלות הסיכון או להגדיל את ההון.")
    else:
        L += ["| מדד | ערך |", "|---|---|",
              f"| **פרמיה שתיגבה עכשיו** | **{_money(s['premium_collected'])}** |",
              f"| פרמיה בקצב שנתי | {_money(s['ann_premium'])} ({_pct(s['ann_yield_on_capital'])} על ההון, "
              f"{_pct(s['ann_yield_on_collateral'])} על הביטחונות) |",
              f"| **הון נדרש (ביטחונות)** | **{_money(s['capital_required'])}** ({_pct(s['utilization'])} מההון) |",
              f"| מזומן פנוי | {_money(s['cash_free'])} |",
              f"| חוזים / מניות | {s['contracts']} / {s['tickers']} |",
              f"| דלתא ממוצעת · ימים ממוצע · כרית ממוצעת | {s['avg_delta']:.3f} · {s['avg_dte']:.0f} · {_pct(s['avg_cushion'])} |",
              f"| הפסד בתרחיש ירידת שוק | {_money(s['stress_loss'])} ({_pct(s['stress_loss_pct'])} מההון) |",
              f"| הפסד מקס' מצניחת מניה אחת | {_pct(s['worst_gap_loss_pct'])} מההון |",
              f"| יתרון צפוי (פרמיה מול תנודתיות היסטורית), שנתי | {_pct(s['expected_edge_ann'], 2)} |", ""]
        if mc:
            L += ["### Monte Carlo (עד פקיעה, זנבות עבים)", "",
                  "| מדד | ערך |", "|---|---|",
                  f"| רווח צפוי למחזור | {_money(mc['expected_pnl'])} |",
                  f"| הסתברות להפסד | {_pct(mc['p_loss'])} |",
                  f"| הסתברות להקצאה של חוזה אחד לפחות | {_pct(mc['p_any_assignment'])} |",
                  f"| VaR 95% / CVaR 95% | {_money(mc['VaR95'])} / {_money(mc['CVaR95'])} |",
                  f"| VaR 99% / CVaR 99% | {_money(mc['VaR99'])} / {_money(mc['CVaR99'])} |", ""]
        p = result.portfolio.sort_values("total_premium", ascending=False)
        L += ["_האופטימיזציה ממקסמת פרמיה **בקצב שנתי** (כך חוזה של 14 יום ושל 45 יום ניתנים להשוואה). "
              "\"פרמיה שתיגבה עכשיו\" היא הסכום בפועל מהחוזים שנבחרו._", "",
              "## התמהיל", "",
              "| מניה | סקטור | פקיעה | ימים | Strike | Spot | חוזים | פרמיה לחוזה | פרמיה כוללת | הון נדרש | דלתא | כרית | תשואה שנתית | P(OTM) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in p.itertuples():
            L.append(f"| {r.ticker} | {r.sector} | {r.expiry} | {r.expiry_dte} | {r.strike:g} | {r.spot:,.2f} | "
                     f"{r.contracts} | {_money(r.premium)} | {_money(r.total_premium)} | {_money(r.total_collateral)} | "
                     f"{r.abs_delta:.2f} | {_pct(r.cushion)} | {_pct(r.ann_yield)} | {_pct(r.p_otm,0)} |")
        L.append("")
    if comparison is not None and len(comparison):
        L += ["## השוואה בין רמות סיכון (אותו הון, פרמיה מקסימלית)", "",
              "| רמת סיכון | פרמיה עכשיו | תשואה שנתית על ההון | הון נדרש | הפסד בתרחיש שוק | דלתא ממוצעת | מניות |",
              "|---|---|---|---|---|---|---|"]
        for r in comparison.itertuples():
            if r.status in ("infeasible", "no_eligible_contracts"):
                L.append(f"| {r.profile} | - | - | - | - | - | - |")
                continue
            L.append(f"| {r.profile} | {_money(r.premium_collected)} | {_pct(r.ann_yield_on_capital)} | "
                     f"{_money(r.capital_required)} | {_pct(r.stress_loss_pct)} | {r.avg_delta:.3f} | {r.tickers} |")
        L.append("")
    if warnings:
        L += ["<details><summary>אזהרות נתונים (" + str(len(warnings)) + ")</summary>", ""]
        L += [f"- {w}" for w in warnings] + ["", "</details>", ""]
    L += ["---", "_כלי ניתוח בלבד, לא ייעוץ השקעות. מחירים לפי mid; ביצוע בפועל עלול להיות גרוע יותר. "
          "החישוב עד פקיעה ואינו כולל הקצאה מוקדמת, גלגול או מס._"]
    return "\n".join(L)
