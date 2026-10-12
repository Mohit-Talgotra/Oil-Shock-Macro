"""Evidence for the plan's hypotheses H1-H6 and the robustness comparison, read from artifacts.

Nothing here is typed in: every statement is formatted from the stored IRFs and stress scores. Each
function takes ``art(name) -> DataFrame | None`` (the app's artifact loader) so it never touches disk itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.models import scenario

Loader = Callable[[str], "pd.DataFrame | None"]

CLAIMS = {
    "H1": "Pass-through: a 10% Brent rise raises CPI inflation with a peak several months later "
    "(and the effect on WPI is larger than on CPI).",
    "H2": "Currency: the rupee depreciates on impact, and the depreciation adds to the inflation effect.",
    "H3": "Policy: the repo rate responds with a lag, and more systematically after inflation targeting began in 2016.",
    "H4": "Growth: GDP growth falls two to four quarters after the shock.",
    "H5": "Regime change: CPI pass-through is weaker in 2022 and 2026 than in 2008.",
    "H6": "Forecast failure: models trained before an up-shock under-predict inflation and depreciation, "
    "and their 80% bands cover the outcome less than 80% of the time.",
}


def _row(hid: str, status: str, evidence: str) -> dict[str, str]:
    return {"id": hid, "hypothesis": CLAIMS[hid], "status": status, "evidence": evidence}


def _na(hid: str, why: str) -> dict[str, str]:
    return _row(hid, "not available", why)


def _bps(art: Loader, window: str) -> float | None:
    irf = art(f"var/irf_A_{window}.parquet")
    return None if irf is None else scenario.first_year_cpi_bps(irf, 10.0)


def _h1(art: Loader) -> dict[str, str]:
    irf = art("var/irf_A_full.parquet")
    if irf is None:
        return _na("H1", "Model A (full sample) not built.")
    s = scenario.summarise_response(scenario.scale_irf(irf, 10.0), "cpi", True)
    ok = s["peak"] > 0 and s["peak_month"] >= 3
    return _row(
        "H1",
        "consistent" if ok else "not consistent",
        f"Per +10% oil the price level peaks at {s['peak'] * 100:+.0f} bps in month {s['peak_month']} "
        f"(90% band {s['lo90_at_peak'] * 100:+.0f} to {s['hi90_at_peak'] * 100:+.0f} bps). "
        "The WPI half is not estimated in the VAR: compare CPI and WPI fuel on the Policy Lab page.",
    )


def _h2(art: Loader) -> dict[str, str]:
    irf = art("var/irf_A_full.parquet")
    if irf is None:
        return _na("H2", "Model A (full sample) not built.")
    d = scenario.scale_irf(irf, 10.0)
    r = d[(d["response"] == "inr") & (~d["cumulative"]) & (d["horizon"] == 0)].iloc[0]
    ok = r["point"] > 0
    return _row(
        "H2",
        "consistent" if ok else "not consistent",
        f"Rupee impact response to +10% oil: {r['point']:+.2f}% (90% band {r['lo90']:+.2f} to {r['hi90']:+.2f}). "
        "Whether depreciation adds to the inflation effect is not separately identified here.",
    )


def _h3(art: Loader) -> dict[str, str]:
    pre, post = art("var/irf_A_pre_it.parquet"), art("var/irf_A_post_it.parquet")
    if pre is None or post is None:
        return _na("H3", "Sub-sample VARs (pre/post inflation targeting) not built.")
    out = []
    for name, irf in (("2004-2015", pre), ("2016-latest", post)):
        out.append((name, scenario.summarise_response(scenario.scale_irf(irf, 10.0), "repo", True)))
    stronger = abs(out[1][1]["peak"]) > abs(out[0][1]["peak"])
    txt = "; ".join(
        f"{n}: repo peaks at {s['peak']:+.2f} pp in month {s['peak_month']}"
        f"{'' if s['significant_at_peak'] else ' (not significant)'}"
        for n, s in out
    )
    return _row("H3", "consistent" if stronger else "not consistent", f"Per +10% oil, {txt}.")


def _h4(art: Loader) -> dict[str, str]:
    irf = art("var/irf_B_full.parquet")
    if irf is None:
        return _na("H4", "Model B (full sample) not built.")
    d = scenario.scale_irf(irf, 10.0)
    g = d[(d["response"] == "gdp_growth") & (~d["cumulative"])].set_index("horizon")
    window = g.loc[[h for h in (2, 3, 4) if h in g.index]]
    neg = bool((window["point"] < 0).any())
    sig = bool((window["hi90"] < 0).any())
    status = "consistent" if sig else ("weakly consistent" if neg else "not consistent")
    band = "excludes zero" if sig else "includes zero"
    return _row(
        "H4",
        status,
        "GDP-growth response per +10% oil at quarters 2-4: "
        + ", ".join(f"q{h} {r['point']:+.2f} pp" for h, r in window.iterrows())
        + f"; the 90% band {band} at the relevant quarters.",
    )


def _h5(art: Loader) -> dict[str, str]:
    vals = {w: _bps(art, w) for w in ("pre2008", "pre2022", "pre2026")}
    later = [v for k, v in vals.items() if k != "pre2008" and v is not None]
    if vals["pre2008"] is None or not later:
        return _na("H5", "Pre-episode windows not built.")
    weaker = all(v < vals["pre2008"] for v in later)
    txt = ", ".join(f"{k}: {v:.0f} bps" for k, v in vals.items() if v is not None)
    return _row(
        "H5",
        "consistent" if weaker else "not consistent",
        f"First-year CPI effect per +10% oil by estimation window: {txt}. The pre-2008 window is short, so "
        "its estimate is imprecise; see the rolling-window chart for the full picture.",
    )


def _h6(art: Loader) -> dict[str, str]:
    s = art("stress/summary.parquet")
    if s is None:
        return _na("H6", "Stress-test results not built.")
    d = s[
        s["episode"].isin(["ukraine_2022", "westasia_2026"])
        & (s["info_set"] == "blind")
        & s["target"].isin(["cpi", "inr"])
        & (s["model"] != "naive")
    ]
    if d.empty:
        return _na("H6", "No blind up-shock scores available.")
    under = float((d["bias"] > 0).mean())
    cov = float(d["coverage80"].mean())
    ok = under > 0.5 and cov < 0.80
    return _row(
        "H6",
        "consistent" if ok else "not consistent",
        f"Blind forecasts of CPI and USD/INR in the two up-shocks: {under:.0%} of model-target scores "
        f"under-predict (positive bias); mean 80% coverage is {cov:.0%} (nominal 80%).",
    )


def evaluate(art: Loader) -> list[dict[str, str]]:
    """One row per hypothesis: claim, status, evidence."""
    return [f(art) for f in (_h1, _h2, _h3, _h4, _h5, _h6)]


def robustness(art: Loader) -> tuple[pd.DataFrame, str]:
    """First-year CPI effect per +10% oil under each identification, plus a one-line verdict."""
    rows = []
    for key, label in (
        ("A_full", "Baseline (oil first)"),
        ("A_blockexog_full", "Oil equation: own lags only (block exogeneity)"),
        ("A_inr2_full", "Rupee ordered second"),
    ):
        d = art(f"var/irf_{key}.parquet")
        if d is not None:
            rows.append({"specification": label, "bps": scenario.first_year_cpi_bps(d, 10.0)})
    df = pd.DataFrame(rows)
    if len(df) < 2:
        return df, "Robustness estimates have not been built."
    base = df["bps"].iloc[0]
    tol = float(config.models_cfg()["scenario"]["block_exog_similar_pct"]) / 100.0
    close = bool(np.all(np.abs(df["bps"].iloc[1:] - base) <= tol * max(abs(base), 1e-9)))
    if close:
        verdict = (
            f"The alternative specifications stay within {tol:.0%} of the baseline, so the oil-first "
            "ordering is not driving the inflation result."
        )
    else:
        verdict = (
            f"At least one alternative differs from the baseline by more than {tol:.0%}: the "
            "identification matters for the inflation result and should be read with caution."
        )
    return df, verdict
