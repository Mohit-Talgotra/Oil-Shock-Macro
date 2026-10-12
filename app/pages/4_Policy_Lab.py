"""Policy Lab: translate the shock into policy trade-offs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

import components as C
from oilshock import config
from oilshock.models import policy, scenario
from oilshock.viz import charts

C.page_setup("Policy Lab: what would the RBI and the government face?", "🏦")
pcfg = config.models_cfg()["policy"]
pan = C.require(C.panels(), "The data panels")
monthly, quarterly = pan
shock = C.shock_slider()
window = st.session_state["window"]

# ---- Taylor rule ---------------------------------------------------------------------------
st.subheader("Taylor rule against the VAR's estimated policy response")
t1, t2, t3 = st.columns(3)
r_star = t1.slider(
    "r* (neutral real rate, %)",
    0.0,
    4.0,
    float(pcfg["r_star"]),
    0.25,
    help="An assumption, not an estimate.",
)
a_pi = t2.slider(
    "Weight on inflation gap (aπ)",
    0.0,
    1.5,
    float(pcfg["a_pi"]),
    0.1,
    help="Taylor's original weight is 0.5.",
)
a_y = t3.slider(
    "Weight on output gap (ay)",
    0.0,
    1.5,
    float(pcfg["a_y"]),
    0.1,
    help="Taylor's original weight is 0.5.",
)
st.caption(
    f"Inflation target π* = {pcfg['pi_target']:.0f}% (RBI). r* is an **assumption**; the output gap is an HP-filter estimate on GDP."
)

cur_pi = float(monthly["cpi_yoy"].dropna().iloc[-1])
cur_repo = float(monthly["repo"].dropna().iloc[-1])
gap_series = policy.output_gap(quarterly["gdp_level"], is_log=True)
cur_gap = float(gap_series.iloc[-1]) if len(gap_series) else 0.0
rule_now = policy.taylor_rate(cur_pi, pcfg["pi_target"], cur_gap, r_star, a_pi, a_y)
m1, m2, m3 = st.columns(3)
m1.metric("Latest CPI inflation", f"{cur_pi:.2f}%")
m2.metric("Taylor-rule rate today", f"{rule_now:.2f}%", f"{rule_now - cur_repo:+.2f} pp vs repo")
m3.metric("Actual repo rate", f"{cur_repo:.2f}%")

irf = C.artifact(f"var/irf_A_{window}.parquet")
if irf is None:
    st.info("The VAR for this window has not been built; choose another window in the sidebar.")
else:
    sc = scenario.scale_irf(irf, shock)
    ypath = scenario.yoy_inflation_path(sc)
    rule_delta = (1.0 + a_pi) * ypath["point"].to_numpy()
    repo = sc[(sc["response"] == "repo") & (sc["cumulative"])].sort_values("horizon")
    st.plotly_chart(
        charts.rule_vs_var(
            repo["horizon"].to_numpy(),
            rule_delta,
            repo["point"].to_numpy(),
            repo["lo90"].to_numpy(),
            repo["hi90"].to_numpy(),
        ),
        use_container_width=True,
    )
    C.source_caption(
        ["repo", "cpi", "brent_usd"],
        f"Change in the policy rate after a {shock:+.0f}% oil shock: rule-implied ((1+aπ) × change in y-o-y inflation, output gap held constant) vs the VAR's repo response, 90% band.",
    )
    peak_rule = float(rule_delta[np.abs(rule_delta).argmax()])
    peak_var = float(repo["point"].iloc[repo["point"].abs().argmax()])
    st.markdown(
        f"At the peak the rule would move the rate by **{peak_rule:+.2f} pp** while the VAR says the RBI has historically moved it by **{peak_var:+.2f} pp**: "
        f"{'less' if abs(peak_var) < abs(peak_rule) else 'more'} than a simple rule suggests."
    )

st.divider()

# ---- Import bill ------------------------------------------------------------------------------
st.subheader("Oil import bill")
latest_brent = float(monthly["brent_usd"].dropna().iloc[-1])
latest_inr = float(monthly["inr_usd"].dropna().iloc[-1])
i1, i2, i3 = st.columns(3)
dprice = i1.number_input(
    "Change in Brent (US$/bbl)",
    -100.0,
    150.0,
    round(latest_brent * shock / 100.0, 1),
    1.0,
    help=f"Defaults to the sidebar shock ({shock:+.0f}%) applied to the latest Brent price (${latest_brent:.1f}).",
)
vol_mt = i2.number_input(
    "Annual crude imports (million tonnes)",
    50.0,
    400.0,
    float(pcfg["crude_imports_mt_fy25"]),
    1.0,
    help=f"FY25: {pcfg['crude_imports_mt_fy25']} Mt (PPAC).",
)
fx = i3.number_input("INR per US$", 40.0, 150.0, round(latest_inr, 2), 0.5)
qgdp = quarterly["gdp_nominal_usd_bn"].dropna()
gdp_usd = float(qgdp.iloc[-4:].sum()) if len(qgdp) >= 4 else None
bill = policy.import_bill_change(
    dprice, policy.tonnes_to_barrels(vol_mt, pcfg["barrels_per_tonne"]), fx, gdp_usd
)
b1, b2, b3 = st.columns(3)
b1.metric("Change in annual import bill", f"US$ {bill['usd_bn']:+.1f} bn")
b2.metric("In rupees", f"₹ {bill['inr_crore']:+,.0f} crore")
b3.metric("As % of GDP", f"{bill['pct_gdp']:+.2f}%" if "pct_gdp" in bill else "n/a")
st.caption(
    f"Volume converted at {pcfg['barrels_per_tonne']} barrels per tonne. % of GDP uses the last four quarters of nominal GDP in US$ where available. Assumes volumes are unchanged."
)

# ---- Excise ---------------------------------------------------------------------------------
st.subheader("Excise cost of shielding consumers")
e1, e2, e3 = st.columns(3)
cut = e1.number_input("Excise cut (₹ per litre)", 0.0, 30.0, 5.0, 0.5)
petrol = e2.number_input(
    "Petrol volume (billion litres/year)",
    1.0,
    200.0,
    pcfg["petrol_litres_per_year"] / 1e9,
    1.0,
    help="Editable assumption: replace with PPAC consumption data.",
)
diesel = e3.number_input(
    "Diesel volume (billion litres/year)",
    1.0,
    400.0,
    pcfg["diesel_litres_per_year"] / 1e9,
    1.0,
    help="Editable assumption: replace with PPAC consumption data.",
)
cost = policy.excise_cost(cut, (petrol + diesel) * 1e9)
st.metric("Annual revenue cost", f"₹ {policy.rupees_to_crore(cost):,.0f} crore")
st.caption(
    "Volumes are placeholder **assumptions**, not sourced data. Revenue cost = cut per litre × volume of petrol and diesel."
)

st.divider()

# ---- CPI vs WPI -----------------------------------------------------------------------------
st.subheader("Wholesale vs retail pass-through")
st.plotly_chart(charts.cpi_vs_wpi(monthly), use_container_width=True)
C.source_caption(
    ["wpi_fuel", "cpi_fuel", "cpi"],
    "Producer fuel prices (WPI) move far more than retail (CPI) because retail fuel prices were administered; the gap is the government and oil companies absorbing the shock.",
)
