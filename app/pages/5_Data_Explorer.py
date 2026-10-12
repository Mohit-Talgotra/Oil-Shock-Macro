"""Data Explorer: let the reader check the inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

import components as C
from oilshock import config
from oilshock.viz import charts

C.page_setup("Data Explorer", "🗂️")
pan = C.require(C.panels(), "The data panels")
monthly, quarterly = pan
sbk = config.series_by_key()

freq = st.radio("Frequency", ["Monthly", "Quarterly"], horizontal=True)
df = monthly if freq == "Monthly" else quarterly
num_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in {"month"}]
default = (
    [c for c in ("brent_usd", "inr_usd", "cpi_yoy", "repo") if c in num_cols][:3]
    if freq == "Monthly"
    else [c for c in ("gdp_growth", "cab_gdp") if c in num_cols]
)
cols = st.multiselect("Series", num_cols, default=default)
transform = st.radio("Transform", ["Level", "Year-on-year %", "Δ log ×100"], horizontal=True)
lo, hi = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), format="YYYY-MM")

if not cols:
    st.info("Pick at least one series.")
    st.stop()
x = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)), cols]
periods = 12 if freq == "Monthly" else 4
if transform == "Year-on-year %":
    x = 100 * (df[cols] / df[cols].shift(periods) - 1).loc[x.index]
elif transform == "Δ log ×100":
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (100 * np.log(df[cols]).diff()).loc[x.index]
st.plotly_chart(charts.series_lines(x, cols), use_container_width=True)
keys = [k for k in cols if k in sbk] or ["brent_usd"]
C.source_caption(keys, f"{transform}; shaded bands are the oil episodes.")
st.download_button(
    "Download this selection (CSV)",
    x.to_csv().encode("utf-8"),
    file_name=f"oilshock_{freq.lower()}_{transform.split()[0].lower()}.csv",
    mime="text/csv",
)

st.subheader("Series dictionary")
rows = []
for s in config.series_cfg()["series"]:
    src = s["sources"][0]
    rows.append(
        {
            "key": s["key"],
            "variable": s["name"],
            "frequency": s["freq"],
            "units": s["units"],
            "used from": s.get("use_from", ""),
            "transformation": s.get("transform", ""),
            "source": src.get("publisher", ""),
            "id / file": src.get("id") or src.get("file") or src.get("formula", ""),
            "link": src.get("url", ""),
            "notes": s.get("notes", ""),
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
    column_config={"link": st.column_config.LinkColumn("link")},
)
