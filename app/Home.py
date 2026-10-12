"""Landing page: orient the user in ten seconds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import components as C

manifest = C.page_setup("When Forecasts Meet Shocks", "🛢️")
st.subheader("How an oil price shock moves India's economy, and where models break")

pan = C.panels()
if pan is None:
    C.require(None, "The data panels")
monthly, _ = pan


def tile(col, label, series, fmt, delta_fmt, unit=""):
    s = series.dropna()
    if s.empty:
        col.metric(label, "n/a")
        return
    last, prev = float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else float("nan")
    col.metric(
        f"{label} · {s.index[-1]:%b %Y}",
        fmt.format(last) + unit,
        delta_fmt.format(last - prev) if prev == prev else None,
    )


c1, c2, c3, c4 = st.columns(4)
tile(c1, "Brent crude (monthly avg)", monthly["brent_usd"], "${:,.1f}", "{:+,.1f} m/m")
tile(c2, "₹ per US$ (monthly avg)", monthly["inr_usd"], "₹{:,.2f}", "{:+,.2f} m/m")
tile(c3, "CPI inflation (y-o-y)", monthly["cpi_yoy"], "{:.2f}", "{:+.2f} pp m/m", "%")
tile(c4, "Repo rate", monthly["repo"], "{:.2f}", "{:+.2f} pp m/m", "%")
st.caption(
    "Latest monthly values and the change on the previous month. "
    + C.config.source_line(["brent_usd", "inr_usd", "cpi", "repo"])
)

st.markdown(
    """
India buys almost 90% of its crude oil abroad, so a jump in the oil price reaches the economy through
the trade balance, the rupee, fuel and second-round prices, the RBI's policy rate, the government
budget and, eventually, growth. This dashboard **estimates** how large and how persistent each of those
effects is, **forecasts** inflation, the rupee and GDP growth with uncertainty bands, and then **stress
tests** those forecasts against past oil episodes: would the models have warned us, or been caught out
like Zillow's pricing model?
"""
)

b1, b2, b3 = st.columns(3)
b1.page_link("pages/1_Transmission.py", label="Transmission: set an oil shock", icon="📈")
b2.page_link("pages/2_Forecasts.py", label="Forecasts: fan charts and models", icon="🔮")
b3.page_link("pages/3_Stress_Test.py", label="Stress test: where forecasts broke", icon="🧪")
st.page_link("pages/4_Policy_Lab.py", label="Policy Lab", icon="🏦")
st.page_link("pages/5_Data_Explorer.py", label="Data Explorer", icon="🗂️")
st.page_link("pages/6_About.py", label="About: method, data and limits", icon="ℹ️")
