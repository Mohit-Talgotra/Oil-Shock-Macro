"""About: method, data and limitations in plain language."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import components as C
from oilshock import config
from oilshock.models import hypotheses

manifest = C.page_setup("About: method, data and limits", "ℹ️")
mcfg = config.models_cfg()

st.markdown(
    """
**Working title:** *When Forecasts Meet Shocks: How an Oil Price Shock Moves India's Economy, and Where Models Break*

### Three questions
1. When crude oil prices jump, how large and how persistent are the effects on India's current account, exchange rate, inflation, policy rate, fiscal balance and GDP growth?
2. Did that transmission differ across the 2008, 2014–16 and 2022 oil episodes?
3. How accurately do forecasting models trained only on pre-shock data predict these variables during a shock, and do their uncertainty bands warn of the miss?

### Method in plain language
- **Transmission.** A vector autoregression (VAR) with oil ordered *first*: India cannot move the world oil price, so nothing Indian can move Brent within the month. A monthly model (oil, industrial output, CPI inflation, repo rate, rupee) gives the precise inflation and rupee dynamics; a small quarterly model adds GDP growth and the current account. Impulse-response bands come from 1,000 bootstrap replications with a fixed seed. Responses are scaled to any shock size because the model is linear.
- **Forecasting.** Naive and seasonal-naive baselines, an AR model, ARIMAX (with oil as an input), the VAR, and XGBoost with quantile objectives, all scored against "no change" on the same rolling origins from 2012. LSTMs are left out deliberately: ~260 monthly observations would overfit them.
- **Stress test.** Every model is frozen the month before each episode and forecasts through it twice: *blind* (oil stays flat) and *oracle* (the actual oil path is supplied). Bad blind but good oracle means oil could not be foreseen; bad in both means the transmission itself changed.
"""
)

st.subheader("Data sources")
rows = []
for s in config.series_cfg()["series"]:
    for src in s["sources"]:
        rows.append(
            {
                "variable": s["name"],
                "publisher": src.get("publisher", ""),
                "id / file": src.get("id") or src.get("file") or src.get("formula", ""),
                "link": src.get("url", ""),
            }
        )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={"link": st.column_config.LinkColumn("link")},
)

st.subheader("Splicing new and old base years")
st.markdown(
    "MOSPI switched to a 2024-base CPI from the January 2026 release and a 2022-23-base GDP series on 27 February 2026 (full back series expected by December 2026). "
    "The OECD copies on FRED stopped updating (CPI in March 2025, real GDP in Q3 2023), so they are used for history only. "
    "Index levels are **never** spliced across base years: each series is converted to growth rates within its own base and the growth rates are joined at a stated switch month "
    "(CPI: January 2014; GDP: the first quarter of each new base; IIP: May 2012, OECD history to MOSPI). A level index, where needed, is rebuilt by cumulating the spliced growth rates."
)
sp = config.paths().reports / "splice_check.csv"
if sp.exists():
    chk = pd.read_csv(sp)
    st.dataframe(chk, hide_index=True)
    thr = config.series_cfg()["checks"]["splice_mean_gap_threshold_pp"]
    over = chk[chk["mean_abs_gap"] > thr] if "mean_abs_gap" in chk else chk.iloc[0:0]
    if len(over):
        st.warning(
            "These splices have a mean overlap gap above 0.5 percentage points, so results around the switch dates deserve extra caution: "
            + ", ".join(over["series"])
        )
else:
    st.caption(
        "The splice overlap report (`reports/splice_check.csv`) is produced by the data stage and is not part of this deployment."
    )

st.subheader("What the checks said")
plaus = (manifest or {}).get("plausibility")
if plaus:
    st.dataframe(pd.DataFrame(plaus)[["name", "ok", "level", "detail"]], hide_index=True)
    st.caption(
        "A failed check is a prompt to investigate (sample, ordering, lags, splicing), not something to hide. If a sign is wrong after investigation, it is reported here."
    )
else:
    st.caption("No plausibility results recorded in the manifest.")

st.subheader("Hypotheses, tested against the estimates")
hyp = pd.DataFrame(hypotheses.evaluate(C.artifact)).rename(
    columns={"id": "", "hypothesis": "Hypothesis", "status": "Status", "evidence": "Evidence"}
)
st.dataframe(hyp, hide_index=True, width="stretch")
st.caption(
    "Computed from the stored estimates, not typed in. 'Consistent' is not proof: bands are wide and samples short."
)

st.subheader("Does the oil-first ordering matter?")
rob, verdict = hypotheses.robustness(C.artifact)
if len(rob):
    st.dataframe(
        rob.rename(
            columns={
                "specification": "Specification",
                "bps": "CPI effect, first year, per +10% oil (bps)",
            }
        ).round(1),
        hide_index=True,
    )
st.markdown(verdict)

st.subheader("Known issues in this build")
st.markdown(
    """
- **The rupee appreciates on impact.** In the oil-first VAR the rupee moves *stronger* on impact after an oil shock (about −0.19 s.d.), which fails the plan's plausibility check. The sign is the same in most sub-samples and under other orderings, and the 12-month cumulative rupee response is not significant in any window. Only the post-inflation-targeting window has the expected sign, and it is not significant. A possible explanation, **not tested**, is that Brent in dollars co-moves with a weak dollar. We report the result rather than change the check.
- **Inflation response is at the low end.** About 18 bps per +10% oil in the first year, just under the 20–60 bps RBI benchmark range. Oil→CPI and oil→rupee Granger tests are not significant in any monthly window, so expect wide bands.
- **IIP ends in March 2026**, which caps the monthly VAR sample (and the live 2026 episode inside it). Industrial output before April 2012 comes from OECD data on FRED, joined to MOSPI on month-on-month growth (overlap gap about 0.005 pp).
- **Quarterly model (Model B) is thin.** Residual whiteness fails in several windows and the pre-2008 window has very few observations. Treat it as indicative.
- **Short pre-2008 window.** About 47 monthly observations and one lag.
- **The 2026 stress test is pseudo out-of-sample.** The model is frozen using data up to February 2026, but it was generated after the fact, so it is not a pre-registered forecast.
- **Data gaps.** CPI fuel and core sub-indices, long excise history, WPI fuel before April 2012, WACR before July 2020 and the 2022-base GDP back series are unavailable. Hand-checked spot values against official releases are still to be added.
"""
)

st.subheader("Limitations, stated openly")
st.markdown(
    """
- **Real-time data.** The stress test uses *today's revised data*, not the numbers forecasters had at the time (a "pseudo real-time" exercise). Seasonal adjustment also uses the full sample.
- **Linearity.** The VAR treats rises and falls symmetrically and scales responses proportionally: a +30% and a −30% shock give mirror-image answers. The asymmetry extension (local projections on oil increases vs decreases) is the place to test this.
- **Identification.** The recursive ordering is an assumption. Block-exogeneity and re-ordering checks are on the Transmission page; results could change under a structural model.
- **Sample size.** About 260 monthly and 85 quarterly observations, so confidence bands are wide. The 2008 episode has the shortest training history (from 2004) and the widest uncertainty.
- **Policy changes.** Fuel-tax and price-freeze decisions are discretionary, so past pass-through may not predict future pass-through. Retail fuel prices were held flat for four years until May 2026.
- **Conditional bands.** ARIMAX and VAR bands are conditional on the assumed oil path; they exclude oil-price uncertainty. This is part of why blind bands can be too narrow.
- **Not a forecast service.** The dashboard illustrates relationships and model risk; it is not investment or policy advice.
"""
)

st.subheader("Reproducibility")
if manifest:
    st.json(
        {
            k: manifest[k]
            for k in (
                "data_cutoff",
                "build_time_utc",
                "git_commit",
                "package_versions",
                "var_seed",
                "bootstrap_reps",
                "backtest_stride",
            )
            if k in manifest
        }
    )
repo_url = mcfg.get("links", {}).get("repo_url")
if repo_url:
    st.markdown(f"Source code and artifacts: [{repo_url}]({repo_url})")
else:
    st.caption(
        "Repository link: set `links.repo_url` in `config/models.yaml` once the public repo exists."
    )
st.caption(
    "Everything is rebuilt from raw data with `make all` (see the repository README). Seeds live in `config/models.yaml`."
)
