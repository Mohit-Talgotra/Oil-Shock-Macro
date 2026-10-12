"""Forecasts: where is the economy heading, and how sure can we be?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

import components as C
from oilshock import config
from oilshock.forecast.base import get_target, target_names
from oilshock.models import scenario
from oilshock.viz import charts

C.page_setup("Forecasts: where is it heading?", "🔮")
fcfg = config.models_cfg()["forecast"]

metrics = C.artifact("forecasts/metrics.parquet")
C.require(metrics, "The forecast backtests")
pan = C.require(C.panels(), "The data panels")
monthly, quarterly = pan

c1, c2, c3 = st.columns([1.2, 2, 1.2])
tname = c1.selectbox("Target", target_names(), format_func=lambda n: get_target(n).label)
spec = get_target(tname)
panel = monthly if spec.freq == "M" else quarterly
available = [
    m for m in charts.MODEL_COLORS if C.artifact(f"forecasts/{m}_{tname}.parquet") is not None
]
default = [m for m in ("naive", "arimax", "var", "xgb") if m in available]
models = c2.multiselect(
    "Models", available, default=default, format_func=lambda m: charts.MODEL_LABELS.get(m, m)
)
horizon = c3.slider(
    "Horizon", 1, spec.horizons, spec.horizons, help="Months (monthly targets) or quarters (GDP)"
)

# ---- oil path ------------------------------------------------------------------------------
st.subheader("Oil path for the forecast period")
o1, o2 = st.columns([1.4, 3])
choice = o1.radio("Next 12 months", ["Flat", "+20%", "+50%", "−20%", "Custom"], horizontal=False)
presets = {"Flat": 0.0, "+20%": 20.0, "+50%": 50.0, "−20%": -20.0}
steps_total = 12 if spec.freq == "M" else 4
if choice == "Custom":
    qcols = o2.columns(4)
    q_pct = [
        qcols[i].number_input(f"Q{i + 1} change (%)", -60.0, 150.0, 0.0, 5.0, key=f"oilq{i}")
        for i in range(4)
    ]
    path = scenario.oil_path_from_quarters(q_pct, 3 if spec.freq == "M" else 1)
    total_pct = float((np.prod([1 + x / 100 for x in q_pct]) - 1) * 100)
else:
    total_pct = presets[choice]
    path = np.full(steps_total, scenario.log_points(total_pct) / steps_total)
o2.caption(
    f"Total change over the next year: **{total_pct:+.0f}%**. ARIMAX and VAR recompute live; XGBoost snaps to the nearest pre-computed grid value; baselines ignore oil."
)


# ---- forecasts per model -----------------------------------------------------------------
def latest_rows(model: str) -> pd.DataFrame:
    d = C.artifact(f"forecasts/{model}_{tname}.parquet").copy()
    d["origin"] = pd.to_datetime(d["origin"])
    return d[d["origin"] == d["origin"].max()].sort_values("horizon")


def scenario_rows(model: str) -> tuple[pd.DataFrame, str]:
    base = latest_rows(model)
    if (
        total_pct == 0
        and choice == "Flat"
        or model in {"naive", "snaive", "ar"}
        or model.startswith("timesfm")
    ):
        return base, "flat oil" if model not in {"naive", "snaive", "ar"} else "oil not used"
    if model in {"arimax", "var"}:
        obj = C.pickled(f"forecasts/fitted_{model}_{tname}.pkl")
        if obj is None:
            return base, "fitted model not available; showing flat-oil forecast"
        fc = obj.predict(spec.horizons, pd.DataFrame({"oil": path}))
        out = base.copy()
        for q in config.QCOLS:
            out[q] = fc[q].to_numpy()
        return out, "recomputed live"
    if model == "xgb":
        grid = C.artifact(f"forecasts/xgb_grid_{tname}.parquet")
        if grid is None:
            return base, "grid not available; showing flat-oil forecast"
        pcts = np.sort(grid["oil_pct"].unique())
        snap = float(pcts[np.abs(pcts - np.clip(total_pct, pcts.min(), pcts.max())).argmin()])
        g = grid[grid["oil_pct"] == snap].sort_values("horizon")
        out = base.copy()
        for q in config.QCOLS:
            out[q] = g[q].to_numpy()
        return out, f"pre-computed grid, snapped to {snap:+.0f}%"
    return base, ""


st.subheader("Fan charts")
if not models:
    st.info("Pick at least one model.")
else:
    tabs = st.tabs([charts.MODEL_LABELS.get(m, m) for m in models])
    z = panel[spec.z]
    for tab, m in zip(tabs, models, strict=True):
        with tab:
            rows, how = scenario_rows(m)
            rows = rows[rows["horizon"] <= horizon]
            hist = z if spec.kind == "level" else None
            st.plotly_chart(
                charts.fan_chart(
                    rows,
                    hist,
                    f"{charts.MODEL_LABELS.get(m, m)}: {spec.label}",
                    charts.MODEL_COLORS.get(m, "#1f77b4"),
                    spec.units,
                    name="Median",
                ),
                use_container_width=True,
            )
            origin = pd.to_datetime(rows["origin"]).max()
            last = rows.iloc[-1]
            C.source_caption(
                [{"cpi": "cpi", "inr": "inr_usd", "gdp": "gdp_growth"}[tname], "brent_usd"],
                f"Median with 50/80/95% bands from {origin:%b %Y} ({how}); at horizon {int(last['horizon'])} the median is {last['q50']:+.2f}{spec.units} "
                f"(80% band {last['q10']:+.2f} to {last['q90']:+.2f}). The 50% band is interpolated from the stored quantiles."
                + (
                    " USD/INR is shown as the % change from the origin, so there is no history line."
                    if spec.kind == "cum"
                    else ""
                ),
            )

# ---- comparison table -------------------------------------------------------------------------
st.subheader("Model comparison")
regime = st.radio(
    "Backtest sample",
    ["all", "calm", "episode"],
    horizontal=True,
    format_func=lambda r: {
        "all": "All target dates",
        "calm": "Calm periods",
        "episode": "Inside oil episodes",
    }[r],
)
m = metrics[
    (metrics["target"] == tname) & (metrics["regime"] == regime) & (metrics["horizon"] == horizon)
].copy()
if m.empty:
    st.caption("No metrics for this selection.")
else:
    m["model"] = m["model"].map(lambda x: charts.MODEL_LABELS.get(x, x))
    show = m[
        [
            "model",
            "rmse",
            "mae",
            "mase",
            "coverage80",
            "coverage95",
            "pinball",
            "dm_pvalue_vs_naive",
            "mean_width80",
            "n",
        ]
    ].set_index("model")
    show.columns = [
        "RMSE",
        "MAE",
        "MASE",
        "80% coverage",
        "95% coverage",
        "Pinball",
        "DM p vs naive",
        "80% band width",
        "N",
    ]

    def highlight(col: pd.Series):
        if col.name in {"RMSE", "MAE", "MASE", "Pinball", "DM p vs naive"}:
            best = col.min()
        elif col.name == "80% coverage":
            best = (col - 0.80).abs().idxmin()
            return [
                "font-weight: bold; background-color: rgba(0,158,115,0.25)" if i == best else ""
                for i in col.index
            ]
        elif col.name == "95% coverage":
            best = (col - 0.95).abs().idxmin()
            return [
                "font-weight: bold; background-color: rgba(0,158,115,0.25)" if i == best else ""
                for i in col.index
            ]
        else:
            return [""] * len(col)
        return [
            "font-weight: bold; background-color: rgba(0,158,115,0.25)" if v == best else ""
            for v in col
        ]

    styled = show.style.apply(highlight, axis=0).format(
        {
            "RMSE": "{:.2f}",
            "MAE": "{:.2f}",
            "MASE": "{:.2f}",
            "80% coverage": "{:.0%}",
            "95% coverage": "{:.0%}",
            "Pinball": "{:.3f}",
            "DM p vs naive": "{:.3f}",
            "80% band width": "{:.2f}",
            "N": "{:.0f}",
        },
        na_rep="–",
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(
        f"Rolling-origin backtest from 2012, horizon {horizon}. MASE below 1 beats 'no change'; coverage should be near 80% / 95%; "
        "DM p < 0.05 means the accuracy difference from naive is statistically meaningful. Best value per column in green. "
        "Uses today's revised data (pseudo real-time)."
    )

# ---- SHAP -----------------------------------------------------------------------------------
shap = C.artifact(f"forecasts/shap_xgb_{tname}.parquet")
if shap is not None:
    with st.expander("What drives the XGBoost forecast? (SHAP)"):
        st.plotly_chart(charts.shap_bar(shap), use_container_width=True)
        st.caption(
            "Mean absolute SHAP value of each input for the median forecast at a 6-period horizon. `oil_fut_cum` is the assumed future oil path; `_lK` is the value K−1 periods before the origin."
        )
