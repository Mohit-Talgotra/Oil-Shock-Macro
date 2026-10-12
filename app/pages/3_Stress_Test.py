"""Stress test: where forecasts broke."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import components as C
from oilshock import config
from oilshock.forecast.base import get_target, target_names
from oilshock.stress.episodes import get_episode
from oilshock.viz import charts

C.page_setup("Stress test: where forecasts broke", "🧪")
summary = C.artifact("stress/summary.parquet")
C.require(summary, "The stress-test results")
pan = C.require(C.panels(), "The data panels")
monthly, quarterly = pan

c1, c2, c3 = st.columns(3)
with c1:
    ep_name = C.episode_picker()
ep = get_episode(ep_name)
info = c2.radio(
    "Information set",
    ["blind", "oracle"],
    horizontal=True,
    format_func=lambda s: {
        "blind": "Blind (oil stays flat)",
        "oracle": "Oracle oil (actual oil path given)",
    }[s],
    help="Blind: could anyone have seen this coming? Oracle: given the oil shock, did the model know how India would respond?",
)
tname = c3.selectbox("Target", target_names(), format_func=lambda n: get_target(n).label)
spec = get_target(tname)

if ep.caution:
    st.warning(ep.caution)
if ep.live:
    tag = config.models_cfg()["stress"]["freeze_tag"]
    st.info(
        f"Live episode: forecasts were frozen at end-February 2026 (git tag `{tag}`) and are re-scored, never regenerated, as new data arrive. Oracle runs use only the oil months observed so far."
    )

res = C.artifact(f"stress/{ep_name}.parquet")
if res is None:
    st.info("No results for this episode yet.")
    st.stop()
res = res[(res["target"] == tname) & (res["info_set"] == info)].copy()
if res.empty:
    st.info("No forecasts for this selection (naive and AR models have no oracle version).")
    st.stop()
res["date"] = pd.to_datetime(res["date"])

panel = monthly if spec.freq == "M" else quarterly
origin = panel.index[panel.index <= ep.train_end].max()
hist = panel[spec.z] if spec.kind == "level" else None
st.plotly_chart(
    charts.episode_chart(res, hist, monthly["brent_usd"], spec.label, spec.units, origin),
    use_container_width=True,
)
C.source_caption(
    [{"cpi": "cpi", "inr": "inr_usd", "gdp": "gdp_growth"}[tname], "brent_usd"],
    f"Each model's frozen median (dashed) and 80% band from the single origin {origin:%b %Y}, against what happened (black). "
    "Trained only on data up to the origin, with today's revised data (pseudo real-time)."
    + (" USD/INR is the % change from the origin." if spec.kind == "cum" else ""),
)

# ---- scores -------------------------------------------------------------------------------
sm = summary[
    (summary["episode"] == ep_name) & (summary["target"] == tname) & (summary["info_set"] == info)
].copy()
if sm.empty:
    st.info("No scores yet: the actual values for this window are not available.")
    st.stop()
st.subheader("Scores")
tbl = sm.copy()
tbl["model"] = tbl["model"].map(lambda x: charts.MODEL_LABELS.get(x, x))
tbl["months_to_surprise"] = tbl["months_to_surprise"].map(
    lambda v: "never" if pd.isna(v) else f"{int(v)}"
)
show = tbl[
    ["model", "rmse", "bias", "shock_penalty", "coverage80", "months_to_surprise", "policy_gap_bps"]
].set_index("model")
show.columns = [
    "RMSE",
    "Bias (actual − forecast)",
    "Shock penalty (×)",
    "80% coverage",
    "Months to surprise",
    "Policy gap (bp)",
]
st.dataframe(
    show.style.format(
        {
            "RMSE": "{:.2f}",
            "Bias (actual − forecast)": "{:+.2f}",
            "Shock penalty (×)": "{:.2f}",
            "80% coverage": "{:.0%}",
            "Policy gap (bp)": "{:.0f}",
        },
        na_rep="–",
    ),
    use_container_width=True,
)
st.caption(
    "Bias > 0: the model under-predicted. Shock penalty = episode RMSE ÷ the model's calm-period RMSE at the same horizons (above 1 = it degrades in shocks). "
    "Months to surprise = first month the outcome left the 95% band (never = a good result). Policy gap = average |Taylor-rule rate from forecast vs actual inflation|, CPI target only."
)

# ---- who got burned -----------------------------------------------------------------------
worst = sm.loc[sm["rmse"].idxmax()]
best = sm.loc[sm["rmse"].idxmin()]
lab = lambda m: charts.MODEL_LABELS.get(m, m)  # noqa: E731
msg = (
    f"**Which model got burned?** On {spec.label.split(' (')[0]} in {ep.short} ({info}), **{lab(worst['model'])}** missed by the most "
    f"(RMSE {worst['rmse']:.2f}{spec.units}, 80% band covered {worst['coverage80']:.0%} of outcomes). The best was **{lab(best['model'])}** (RMSE {best['rmse']:.2f}{spec.units})."
)
if tname == "cpi" and sm["policy_gap_bps"].notna().any():
    g = sm.loc[sm["policy_gap_bps"].idxmax()]
    msg += f" A rate-setter leaning on {lab(g['model'])} would have had the Taylor-rule rate off by {g['policy_gap_bps']:.0f} bp on average."
st.success(msg)

st.markdown(
    "**The Zillow parallel.** Zillow's pricing model kept predicting rising house prices, so the company overpaid for homes it then could not sell. "
    "A central bank that leans on a forecast which misses an oil shock sets the policy rate in the wrong place, and the policy gap above, in basis points, is the cost."
)

st.subheader("Do the bands warn?")
st.plotly_chart(charts.coverage_bar(sm), use_container_width=True)
C.source_caption(
    [{"cpi": "cpi", "inr": "inr_usd", "gdp": "gdp_growth"}[tname]],
    "Share of outcomes inside each model's 80% band, against the nominal 80%.",
)
