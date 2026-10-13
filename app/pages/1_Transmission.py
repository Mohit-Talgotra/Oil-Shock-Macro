"""Transmission: how an oil shock travels through the economy (the centrepiece)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
from oilshock import config
from oilshock.models import hypotheses, scenario
from oilshock.viz import charts

manifest = C.page_setup("Transmission: how an oil shock moves India", "📈")
shock = C.shock_slider()
window = st.session_state["window"]
band = int(st.session_state["band"])
wlabel = config.models_cfg()["windows"][window]["label"]
mcfg = config.models_cfg()

c1, c2 = st.columns([2, 2])
model_choice = c1.radio(
    "Model", ["Monthly core (Model A)", "Quarterly extended (Model B)"], horizontal=True
)
show_lp = c2.checkbox(
    "Show local projections (robustness)",
    value=False,
    disabled=model_choice.startswith("Quarterly"),
)
model = "A" if model_choice.startswith("Monthly") else "B"
vcfg = mcfg["var_monthly" if model == "A" else "var_quarterly"]
unit = "months" if model == "A" else "quarters"

st.info(
    "The model is **linear**: a +30% and a −30% shock give (nearly) mirror-image answers, and a bigger "
    "shock is just a proportional rescaling. Real oil shocks may not behave so symmetrically "
    "(see the asymmetry check on the About page).",
    icon="ℹ️",
)

irf = C.artifact(f"var/irf_{model}_{window}.parquet")
if irf is None:
    st.warning(
        f"No Model {model} estimate for the window '{wlabel}' (too few observations or not built yet)."
    )
    C.require(C.artifact("manifest.json"), "The VAR artifacts")
    st.stop()
irf = irf.copy()
scaled = scenario.scale_irf(irf, shock)
sd = float(irf["shock_size_logpts"].iloc[0])
st.caption(
    f"Estimation window: **{wlabel}**. A {shock:+.0f}% oil price change is {scenario.log_points(shock):+.1f} log points, "
    f"or {scenario.scale_factor(shock, sd):+.2f} times the estimated one-standard-deviation oil shock ({sd:.1f} log points)."
)

# ---- transmission chain -------------------------------------------------------------------
irf_a = C.artifact(f"var/irf_A_{window}.parquet")
irf_b = C.artifact(f"var/irf_B_{window}.parquet")
sa = scenario.scale_irf(irf_a, shock) if irf_a is not None else None
sb = scenario.scale_irf(irf_b, shock) if irf_b is not None else None


def node(title, df, resp, cumulative, unit_txt, num, color, tmax=None, tunit="months", digits=2):
    if df is None:
        return {
            "num": num,
            "title": title,
            "value": "n/a",
            "detail": "not estimated for this window",
            "color": color,
        }
    d = df if tmax is None else df[df["horizon"] <= tmax]
    try:
        s = scenario.summarise_response(d, resp, cumulative)
    except KeyError:
        return {"num": num, "title": title, "value": "n/a", "detail": "", "color": color}
    sig = "" if s["significant_at_peak"] else " (not significant)"
    return {
        "num": num,
        "title": title,
        "value": f"{s['peak']:+.{digits}f} {unit_txt}",
        "detail": f"peaks in {tunit[:-1]} {s['peak_month']}{sig}",
        "color": color,
    }


V = charts.VARIABLE_COLORS
oil_peak = scenario.log_points(shock)
nodes = [
    {
        "title": "Oil price",
        "value": f"{shock:+.0f}% ({oil_peak:+.1f} log pts)",
        "detail": "the shock",
        "color": V["oil"],
    },
    node("Trade balance", sb, "cab_gdp", False, "pp of GDP", "1", V["cab_gdp"], tunit="quarters"),
    node("Exchange rate", sa, "inr", True, "% (rupee weaker)", "2", V["inr"]),
    node("Direct prices", sa, "cpi", True, "pp on price level", "3", V["cpi"], tmax=6),
    node("Second-round prices", sa, "cpi", True, "pp on price level", "4", V["cpi"]),
    node("Monetary policy", sa, "repo", True, "pp on repo", "5", V["repo"]),
    {
        "num": "6",
        "title": "Fiscal",
        "value": "discretionary",
        "detail": "excise and subsidy choices: see Policy Lab",
        "color": "#888888",
    },
    node(
        "Real activity",
        sb,
        "gdp_growth",
        False,
        "pp GDP growth",
        "7",
        V["gdp_growth"],
        tunit="quarters",
    ),
]
st.subheader("The seven channels")
st.plotly_chart(charts.chain_diagram(nodes), use_container_width=True)
C.source_caption(
    ["brent_usd", "inr_usd", "cpi", "repo", "gdp_growth", "cab"],
    "Each box: scaled peak effect and when it peaks (0 = impact month). Boxes 1 and 7 come from the quarterly model.",
)

# ---- headline sentence from the data -----------------------------------------------------
if sa is not None:
    cs = scenario.summarise_response(sa, "cpi", True)
    bps10 = scenario.first_year_cpi_bps(irf_a, 10.0)
    scn = mcfg["scenario"]
    ys = scenario.yoy_inflation_path(sa)
    ypk = ys.loc[ys["point"].abs().idxmax()]
    st.markdown(
        f"**Reading the result:** after a {shock:+.0f}% oil shock the price level is {cs['peak']:+.2f} pp higher at its peak "
        f"(month {cs['peak_month']}; 90% band for the peak: {cs['lo90_at_peak']:+.2f} to {cs['hi90_at_peak']:+.2f} pp). Year-on-year inflation is {ypk['point']:+.2f} pp higher at its peak. "
        f"Per +10% oil, the first-year CPI effect is **{bps10:.0f} bps**; the {scn['rbi_benchmark_label']} benchmark is about "
        f"{scn['rbi_benchmark_bps_10pct'][0]}–{scn['rbi_benchmark_bps_10pct'][1]} bps "
        f"({'inside' if scn['red_flag_bps_range_10pct'][0] <= bps10 <= scn['red_flag_bps_range_10pct'][1] else 'OUTSIDE'} "
        f"the {scn['red_flag_bps_range_10pct'][0]}–{scn['red_flag_bps_range_10pct'][1]} bps sanity range)."
    )

# ---- IRF grid -------------------------------------------------------------------------------
st.subheader(f"Impulse responses to a {shock:+.0f}% oil shock")
extra = {"cpi_yoy": scenario.yoy_inflation_path(scaled)} if model == "A" else None
lp = None
if show_lp and model == "A":
    lp_raw = C.artifact("var/irf_A_lp_full.parquet")
    if lp_raw is not None:
        lp = scenario.scale_irf(lp_raw, shock)
    else:
        st.caption("Local projections have not been built.")
fig = charts.irf_grid(scaled, band, vcfg["differenced"], list(vcfg["variables"]), lp, extra, unit)
if extra:  # label the y-o-y panel
    fig.layout.annotations[min(len(vcfg["variables"]), 5)].text = "CPI inflation, y-o-y (pp)"
st.plotly_chart(fig, use_container_width=True)
C.source_caption(
    ["brent_usd", "inr_usd", "cpi", "repo", "iip"]
    if model == "A"
    else ["brent_usd", "inr_usd", "cpi", "repo", "gdp_growth", "cab"],
    f"Cumulative responses (levels) for differenced variables, {band}% bootstrap bands, in {unit} since the shock. "
    "Units: % for oil, rupee and output; percentage points for prices, repo and GDP growth.",
)

# ---- FEVD + rolling pass-through -----------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("How much of each variable does oil explain?")
    fevd = C.artifact(f"var/fevd_{model}_{window}.parquet")
    if fevd is not None:
        hs = (6, 12, 24) if model == "A" else (2, 4, 8)
        st.plotly_chart(
            charts.fevd_bar(fevd, hs, list(vcfg["variables"])), use_container_width=True
        )
        C.source_caption(
            ["brent_usd", "cpi"],
            f"Forecast error variance decomposition at {hs[0]}, {hs[1]} and {hs[2]} {unit}.",
        )
with right:
    st.subheader("Has pass-through changed over time?")
    roll = C.artifact("var/rolling_A_full.parquet")
    if roll is not None and len(roll):
        st.plotly_chart(charts.rolling_passthrough(roll, band), use_container_width=True)
        C.source_caption(
            ["brent_usd", "cpi"],
            "12-month CPI response to a +10% oil rise, 96-month rolling windows stepping 6 months. Wide bands in short windows are expected.",
        )
    else:
        st.caption("Rolling-window estimates have not been built.")

# ---- robustness + diagnostics ---------------------------------------------------------------
with st.expander("Robustness: identification and diagnostics"):
    rob, verdict = hypotheses.robustness(C.artifact)
    if len(rob):
        st.dataframe(
            rob.rename(
                columns={
                    "specification": "specification",
                    "bps": "CPI effect, first year, per +10% oil (bps)",
                }
            ).round(1),
            hide_index=True,
        )
    st.caption(verdict)
    diag = C.artifact("var/diagnostics.parquet")
    if diag is not None:
        d = diag[(diag["model"] == model) & (diag["window"] == window)]
        if len(d):
            show = d.T.reset_index()
            show.columns = ["check", "value"]
            st.dataframe(show, hide_index=True)
            st.caption(
                "Stability, whiteness (Portmanteau), normality (Jarque-Bera), Granger causality from oil and variance shares at 6/12/24."
            )
