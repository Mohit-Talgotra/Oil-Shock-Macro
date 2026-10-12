"""Every Plotly figure the app uses (plan Section 6, ``viz/charts.py``).

Pure figure builders. Conventions (plan Section 10):
one colour per variable and per model on every page; episodes shaded the same way on every time-series
chart; no hard-coded backgrounds, so figures follow Streamlit's light and dark themes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from oilshock import config

# Okabe-Ito based, colour-blind safe
VARIABLE_COLORS = {
    "oil": "#D55E00",
    "inr": "#0072B2",
    "cpi": "#CC79A7",
    "repo": "#009E73",
    "gdp_growth": "#E69F00",
    "cab_gdp": "#56B4E9",
    "ip": "#7f7f7f",
}
VARIABLE_LABELS = {
    "oil": "Brent oil price",
    "inr": "Rupee (INR per USD)",
    "cpi": "CPI (price level)",
    "repo": "Repo rate",
    "gdp_growth": "GDP growth",
    "cab_gdp": "Current account (% GDP)",
    "ip": "Industrial output",
}
VARIABLE_UNITS = {
    "oil": "%",
    "inr": "% (up = weaker rupee)",
    "cpi": "percentage points",
    "repo": "percentage points",
    "gdp_growth": "percentage points",
    "cab_gdp": "percentage points of GDP",
    "ip": "%",
}
MODEL_COLORS = {
    "naive": "#7f7f7f",
    "snaive": "#bcbd22",
    "ar": "#8c564b",
    "arimax": "#1f77b4",
    "var": "#2ca02c",
    "xgb": "#d62728",
    "timesfm_zs": "#9467bd",
    "timesfm_cov": "#e377c2",
}
MODEL_LABELS = {
    "naive": "Naive",
    "snaive": "Seasonal naive",
    "ar": "AR(p)",
    "arimax": "ARIMAX",
    "var": "VAR",
    "xgb": "XGBoost",
    "timesfm_zs": "TimesFM (zero-shot)",
    "timesfm_cov": "TimesFM (+oil)",
}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------------ shared pieces
def add_episode_shading(
    fig: go.Figure, x_min=None, x_max=None, row=None, col=None, labels: bool = True
) -> go.Figure:
    """Shade every episode the same way on every time-series chart (labels from ``episodes.yaml``)."""
    for name, ep in config.episode_defs().items():
        lo = pd.Timestamp(ep["test_start"])
        hi = (
            pd.Timestamp(ep["test_end"])
            if ep.get("test_end")
            else pd.Timestamp(x_max or lo + pd.DateOffset(months=18))
        )
        if x_min is not None and hi < pd.Timestamp(x_min):
            continue
        if x_max is not None and lo > pd.Timestamp(x_max):
            continue
        kw = {"row": row, "col": col} if row else {}
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor=rgba(ep.get("color", "#888888"), 0.10),
            line_width=0,
            layer="below",
            annotation_text=ep.get("short", name) if labels else None,
            annotation_position="top left",
            annotation_font_size=10,
            **kw,
        )
    return fig


def _layout(
    fig: go.Figure, title: str | None = None, height: int = 420, ytitle: str | None = None
) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    return fig


def _band(fig, x, lo, hi, color, alpha, name, row=None, col=None, showlegend=False):
    kw = {"row": row, "col": col} if row else {}
    fig.add_trace(
        go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
        **kw,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lo,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(color, alpha),
            hoverinfo="skip",
            name=name,
            showlegend=showlegend,
        ),
        **kw,
    )


# ------------------------------------------------------------------ Layer 1
def irf_grid(
    irfs: pd.DataFrame,
    band: int = 90,
    differenced: list[str] | None = None,
    order: list[str] | None = None,
    lp: pd.DataFrame | None = None,
    extra: dict[str, pd.DataFrame] | None = None,
    time_unit: str = "months",
    title: str | None = None,
) -> go.Figure:
    """2 x 3 grid of (cumulative for differenced variables) impulse responses with bands.

    ``irfs`` is a scaled long-format IRF table; ``extra`` adds panels such as the y-o-y inflation path
    (name -> frame with horizon + point/lo/hi columns); ``lp`` overlays local-projection estimates.
    """
    vars_ = order or list(dict.fromkeys(irfs["response"]))
    panels = [(v, None) for v in vars_] + [(k, df) for k, df in (extra or {}).items()]
    panels = panels[:6]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[VARIABLE_LABELS.get(p[0], p[0]) for p in panels],
        vertical_spacing=0.16,
    )
    lo_c, hi_c = (f"lo{band}", f"hi{band}")
    for i, (name, frame) in enumerate(panels):
        r, c = divmod(i, 3)
        if frame is None:
            use_cum = (differenced is None) or (name in differenced)
            d = irfs[(irfs["response"] == name) & (irfs["cumulative"] == use_cum)].sort_values(
                "horizon"
            )
        else:
            d = frame.sort_values("horizon")
        color = VARIABLE_COLORS.get(name.split("_yoy")[0], VARIABLE_COLORS["cpi"])
        _band(fig, d["horizon"], d[lo_c], d[hi_c], color, 0.22, f"{band}% band", r + 1, c + 1)
        fig.add_trace(
            go.Scatter(
                x=d["horizon"],
                y=d["point"],
                mode="lines",
                line=dict(color=color, width=2.4),
                name=VARIABLE_LABELS.get(name, name),
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        if lp is not None and frame is None:
            ld = lp[lp["response"] == name].sort_values("horizon")
            if len(ld):
                fig.add_trace(
                    go.Scatter(
                        x=ld["horizon"],
                        y=ld["point"],
                        mode="lines",
                        line=dict(color=color, dash="dot", width=1.6),
                        name="Local projection",
                        showlegend=(i == 0),
                    ),
                    row=r + 1,
                    col=c + 1,
                )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=r + 1, col=c + 1)
        fig.update_xaxes(title_text=time_unit if r == 1 else None, row=r + 1, col=c + 1)
    return _layout(fig, title, height=560)


def chain_diagram(nodes: list[dict]) -> go.Figure:
    """Transmission chain: ``nodes`` is a list of 8 dicts with ``title``, ``value`` and ``detail`` strings.

    Boxes are laid out left to right in two rows (oil to GDP) with arrows between them. An optional
    ``num`` key numbers the channels 1-7 (the oil node has none).
    """
    n = len(nodes)
    per_row = 4
    fig = go.Figure()
    w, h, gx, gy = 0.21, 0.34, 0.04, 0.12
    pos = []
    for i in range(n):
        r, c = divmod(i, per_row)
        x0 = 0.01 + c * (w + gx)
        y0 = 0.98 - (r + 1) * h - r * gy
        pos.append((x0, y0))
    for i, (nd, (x0, y0)) in enumerate(zip(nodes, pos, strict=True)):
        color = nd.get("color", "#888888")
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x0 + w,
            y1=y0 + h,
            line=dict(color=color, width=2),
            fillcolor=rgba(color, 0.12),
            xref="paper",
            yref="paper",
        )
        fig.add_annotation(
            x=x0 + w / 2,
            y=y0 + h / 2,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="center",
            text=f"<b>{nd['num'] + '. ' if nd.get('num') else ''}{nd['title']}</b><br>{nd.get('value', '')}<br><span style='font-size:11px'>{nd.get('detail', '')}</span>",
            font=dict(size=13),
        )
        if i < n - 1:
            if (i + 1) % per_row:
                x_from, y_from, x_to, y_to = (
                    x0 + w,
                    y0 + h / 2,
                    pos[i + 1][0],
                    pos[i + 1][1] + h / 2,
                )
            else:  # wrap to the second row
                x_from, y_from, x_to, y_to = (
                    x0 + w / 2,
                    y0,
                    pos[i + 1][0] + w / 2,
                    pos[i + 1][1] + h,
                )
            fig.add_annotation(
                x=x_to,
                y=y_to,
                ax=x_from,
                ay=y_from,
                xref="paper",
                yref="paper",
                axref="paper",
                ayref="paper",
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor="gray",
            )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def fevd_bar(
    fevd: pd.DataFrame, horizons: tuple[int, ...] = (6, 12, 24), order: list[str] | None = None
) -> go.Figure:
    """Share of each variable's forecast-error variance explained by oil, at several horizons."""
    d = fevd[fevd["horizon"].isin(horizons)]
    vars_ = order or list(dict.fromkeys(d["variable"]))
    vars_ = [v for v in vars_ if v != "oil"]
    fig = go.Figure()
    shades = [0.45, 0.7, 1.0]
    for h, a in zip(horizons, shades, strict=False):
        s = d[d["horizon"] == h].set_index("variable")["share_oil"]
        fig.add_bar(
            x=[VARIABLE_LABELS.get(v, v) for v in vars_],
            y=[100 * s.get(v, np.nan) for v in vars_],
            name=f"{h} periods",
            marker_color=[rgba(VARIABLE_COLORS.get(v, "#888888"), a) for v in vars_],
        )
    fig.update_layout(barmode="group")
    return _layout(fig, None, 360, "% of forecast-error variance due to oil")


def rolling_passthrough(df: pd.DataFrame, band: int = 90) -> go.Figure:
    """12-month cumulative CPI response to a 10% oil rise, from rolling windows (wide bands are shown)."""
    fig = go.Figure()
    x = pd.to_datetime(df["end_date"])
    _band(
        fig,
        x,
        100 * df[f"lo{band}"],
        100 * df[f"hi{band}"],
        VARIABLE_COLORS["cpi"],
        0.22,
        f"{band}% band",
        showlegend=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=100 * df["point"],
            mode="lines+markers",
            line=dict(color=VARIABLE_COLORS["cpi"], width=2.4),
            name="Point estimate",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    it = pd.Timestamp(config.events_cfg()["dummies"]["inflation_targeting"]["start"])
    fig.add_vline(
        x=it,
        line_dash="dot",
        line_color="gray",
        annotation_text="Inflation targeting",
        annotation_position="top left",
    )
    add_episode_shading(fig, x.min(), x.max(), labels=False)
    return _layout(
        fig, None, 400, "bps added to CPI over 12 months per +10% oil (96-month windows)"
    )


# ------------------------------------------------------------------ Layer 2
def _q25_q75(fc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """50% band from the stored quantiles, interpolating as if normal between q10, q50 and q90."""
    k = 0.6744897501960817 / 1.2815515655446004
    return fc["q50"] + (fc["q10"] - fc["q50"]) * k, fc["q50"] + (fc["q90"] - fc["q50"]) * k


def fan_chart(
    fc: pd.DataFrame,
    actual: pd.Series | None,
    title: str | None = None,
    color: str = "#1f77b4",
    units: str = "",
    history: int = 36,
    name: str = "Median forecast",
) -> go.Figure:
    """Fan chart: 50%, 80% and 95% bands around the median with the last ``history`` actual periods.

    ``fc`` has ``target_date`` and q05..q95 for one origin. The 50% band interpolates the stored
    quantiles (normal shape between q10, q50 and q90); 80% and 95% bands are exact.
    """
    fc = fc.sort_values("target_date")
    x = pd.to_datetime(fc["target_date"])
    q25, q75 = _q25_q75(fc)
    fig = go.Figure()
    if actual is not None and len(actual):
        a = actual.dropna().iloc[-history:]
        fig.add_trace(
            go.Scatter(
                x=a.index,
                y=a.to_numpy(),
                mode="lines",
                line=dict(color="gray", width=2),
                name="Actual",
            )
        )
    _band(fig, x, fc["q05"], fc["q95"], color, 0.14, "95%", showlegend=True)
    _band(fig, x, fc["q10"], fc["q90"], color, 0.22, "80%", showlegend=True)
    _band(fig, x, q25, q75, color, 0.34, "50%", showlegend=True)
    fig.add_trace(
        go.Scatter(x=x, y=fc["q50"], mode="lines", line=dict(color=color, width=2.6), name=name)
    )
    return _layout(fig, title, 380, units)


def coverage_bar(summary: pd.DataFrame, nominal: float = 0.80) -> go.Figure:
    """80% band coverage by model (bars) against the nominal level (dashed line)."""
    d = summary.groupby("model", as_index=False)["coverage80"].mean()
    fig = go.Figure(
        go.Bar(
            x=[MODEL_LABELS.get(m, m) for m in d["model"]],
            y=100 * d["coverage80"],
            marker_color=[MODEL_COLORS.get(m, "#888888") for m in d["model"]],
        )
    )
    fig.add_hline(
        y=100 * nominal, line_dash="dash", line_color="gray", annotation_text="Nominal 80%"
    )
    fig.update_yaxes(range=[0, 105])
    return _layout(fig, None, 340, "% of outcomes inside the 80% band")


# ------------------------------------------------------------------ Layer 3
def episode_chart(
    res: pd.DataFrame,
    history: pd.Series | None,
    oil: pd.Series | None,
    target_label: str,
    units: str,
    origin: pd.Timestamp | None = None,
    show_models: list[str] | None = None,
) -> go.Figure:
    """Actual path against each model's frozen median and 80% band; oil path on a secondary panel."""
    rows = 2 if oil is not None else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28] if rows == 2 else [1.0],
        vertical_spacing=0.06,
    )
    models = show_models or list(dict.fromkeys(res["model"]))
    dates = pd.to_datetime(res["date"])
    for m in models:
        d = res[res["model"] == m].sort_values("date")
        if d.empty:
            continue
        col = MODEL_COLORS.get(m, "#888888")
        x = pd.to_datetime(d["date"])
        _band(fig, x, d["q10"], d["q90"], col, 0.12, f"{MODEL_LABELS.get(m, m)} 80%", 1, 1)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d["q50"],
                mode="lines",
                line=dict(color=col, width=2, dash="dash"),
                name=MODEL_LABELS.get(m, m),
            ),
            row=1,
            col=1,
        )
    actual = res.dropna(subset=["actual"]).drop_duplicates("date").sort_values("date")
    if history is not None and len(history):
        h = history.dropna()
        lo = (origin - pd.DateOffset(months=24)) if origin is not None else h.index.min()
        h = h[(h.index >= lo) & (h.index <= (origin if origin is not None else h.index.max()))]
        fig.add_trace(
            go.Scatter(
                x=h.index,
                y=h.to_numpy(),
                mode="lines",
                line=dict(color="gray", width=2.4),
                name="Actual (before)",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(actual["date"]),
            y=actual["actual"],
            mode="lines+markers",
            line=dict(color="black", width=2.6),
            name="Actual",
        ),
        row=1,
        col=1,
    )
    if origin is not None:
        fig.add_vline(
            x=origin,
            line_dash="dot",
            line_color="gray",
            annotation_text="Frozen origin",
            annotation_position="top left",
            row=1,
            col=1,
        )
    if oil is not None:
        o = oil.dropna()
        if origin is not None:
            o = o[(o.index >= origin - pd.DateOffset(months=24)) & (o.index <= dates.max())]
        fig.add_trace(
            go.Scatter(
                x=o.index,
                y=o.to_numpy(),
                mode="lines",
                line=dict(color=VARIABLE_COLORS["oil"], width=2),
                name="Brent (US$/bbl)",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="US$/bbl", row=2, col=1)
    fig.update_yaxes(
        title_text=f"{target_label} ({units})" if units else target_label, row=1, col=1
    )
    return _layout(fig, None, 560 if rows == 2 else 420)


def shap_bar(shap_df: pd.DataFrame, top: int = 15) -> go.Figure:
    d = shap_df.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=d["mean_abs_shap"], y=d["feature"], orientation="h", marker_color=MODEL_COLORS["xgb"]
        )
    )
    fig.update_layout(hovermode="closest")
    return _layout(fig, None, 420, None).update_xaxes(
        title_text="mean |SHAP| (effect on the median forecast)"
    )


def cpi_vs_wpi(panel: pd.DataFrame) -> go.Figure:
    """Wholesale vs retail pass-through: WPI fuel, CPI fuel and headline CPI (y-o-y %)."""
    d = panel.loc["2012":]
    fig = go.Figure()
    for col, label, colr, dash in (
        ("wpi_fuel_yoy", "WPI fuel & power (producer)", VARIABLE_COLORS["oil"], "solid"),
        ("cpi_fuel_yoy", "CPI fuel & light (retail)", VARIABLE_COLORS["cpi"], "dash"),
        ("cpi_yoy", "CPI headline", "gray", "solid"),
    ):
        if col in d and d[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=d[col],
                    mode="lines",
                    name=label,
                    line=dict(color=colr, dash=dash, width=2),
                )
            )
    hike = pd.Timestamp(config.models_cfg()["policy"]["retail_hike_annotation"])
    if hike <= d.index.max():
        fig.add_vline(
            x=hike,
            line_dash="dot",
            line_color="gray",
            annotation_text="May 2026: first retail fuel hike in four years",
            annotation_position="top left",
        )
    add_episode_shading(fig, d.index.min(), d.index.max(), labels=False)
    return _layout(fig, None, 400, "y-o-y %")


def rule_vs_var(
    h: np.ndarray,
    rule_path: np.ndarray,
    var_point: np.ndarray,
    var_lo: np.ndarray | None = None,
    var_hi: np.ndarray | None = None,
) -> go.Figure:
    """Rule-implied change in the policy rate against the VAR's estimated repo response."""
    fig = go.Figure()
    if var_lo is not None and var_hi is not None:
        _band(fig, h, var_lo, var_hi, VARIABLE_COLORS["repo"], 0.2, "VAR 90% band", showlegend=True)
    fig.add_trace(
        go.Scatter(
            x=h,
            y=var_point,
            mode="lines",
            line=dict(color=VARIABLE_COLORS["repo"], width=2.6),
            name="VAR: estimated repo response",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h,
            y=rule_path,
            mode="lines",
            line=dict(color="gray", width=2.4, dash="dash"),
            name="Taylor rule implied",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_xaxes(title_text="months after the shock")
    return _layout(fig, None, 380, "change in repo rate (percentage points)")


def series_lines(
    df: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None, shade: bool = True
) -> go.Figure:
    """Explorer chart: any set of series with episode shading."""
    fig = go.Figure()
    for c in columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[c],
                mode="lines",
                name=(labels or {}).get(c, c),
                line=dict(
                    color=VARIABLE_COLORS.get(c.split("_")[0]) if c in VARIABLE_COLORS else None
                ),
            )
        )
    if shade and len(df):
        add_episode_shading(fig, df.index.min(), df.index.max())
    return _layout(fig, None, 420)
