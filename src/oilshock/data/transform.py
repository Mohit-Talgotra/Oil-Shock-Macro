"""Pure transformation functions (no I/O). Plan Section 6, ``data/transform.py``."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

MONTH_END = "ME"
QUARTER_END = "QE"


def to_period_end(index: pd.Index | pd.Series, freq: str = "M") -> pd.DatetimeIndex:
    """Map any dates to the end of their month ("M") or quarter ("Q")."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    per = idx.to_period("M" if freq.upper().startswith("M") else "Q")
    return per.to_timestamp(how="end").normalize()


def log_diff(s: pd.Series, scale: float = 100.0) -> pd.Series:
    """``scale`` x first difference of the natural log (100 x dlog = approx. % change)."""
    if (s.dropna() <= 0).any():
        raise ValueError("log_diff needs strictly positive values")
    return scale * np.log(s).diff()


def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage change, in percent."""
    return 100.0 * (s / s.shift(periods) - 1.0)


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    """Year-on-year growth in percent: 12 periods for monthly data, 4 for quarterly."""
    return pct_change(s, periods)


def seasonal_adjust(s: pd.Series, method: str = "stl", period: int = 12) -> pd.Series:
    """Seasonally adjust with STL. Multiplicative (via logs) for positive series, additive otherwise."""
    if method != "stl":
        raise ValueError(f"unknown method {method!r}; only 'stl' is implemented")
    from statsmodels.tsa.seasonal import STL

    x = s.dropna()
    if len(x) < 2 * period + 1:
        raise ValueError("not enough observations for STL")
    multiplicative = bool((x > 0).all())
    work = np.log(x) if multiplicative else x
    fit = STL(work, period=period, robust=True).fit()
    adj = work - fit.seasonal
    adj = np.exp(adj) if multiplicative else adj
    return adj.reindex(s.index).rename(s.name)


def to_quarterly(s: pd.Series, how: str = "mean") -> pd.Series:
    """Monthly -> quarterly (period-end index). ``how``: mean | last | sum.

    A quarter with fewer than three monthly observations is NaN rather than a partial average.
    """
    if how not in {"mean", "last", "sum"}:
        raise ValueError("how must be mean, last or sum")
    grp = s.resample(QUARTER_END)
    out = getattr(grp, how)()
    cnt = grp.count()
    out[cnt < 3] = np.nan
    return out.rename(s.name)


def step_series(events: pd.Series | pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Expand an event table (date -> value at change) into a step series on ``index``.

    The value in force at each date is the last event on or before it.
    """
    ev = events.iloc[:, 0] if isinstance(events, pd.DataFrame) else events
    ev = ev.sort_index()
    ev = ev[~ev.index.duplicated(keep="last")]
    # a change dated inside a month is in force for that month-end
    union = ev.index.union(index)
    return ev.reindex(union).ffill().reindex(index).rename(ev.name)


def window_dummy(index: pd.DatetimeIndex, start: str, end: str | None) -> pd.Series:
    """0/1 dummy equal to 1 for dates between ``start`` and ``end`` (inclusive); ``end=None`` = open."""
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else pd.Timestamp.max
    return pd.Series(((index >= lo) & (index <= hi)).astype(float), index=index)


def quarter_dummy(index: pd.DatetimeIndex, quarters: list[str]) -> pd.Series:
    """0/1 dummy for named quarters such as ``2020Q2`` on a quarter-end index."""
    labels = {pd.Period(q, "Q") for q in quarters}
    return pd.Series([1.0 if p in labels else 0.0 for p in index.to_period("Q")], index=index)


def gdp_level_from_yoy(g: pd.Series) -> pd.Series:
    """Log-level proxy of real GDP rebuilt from y-o-y growth: ln Y_t = ln Y_{t-4} + ln(1+g_t/100).

    The first four quarters are set to 0 (an index of 100). Used only as input to the HP-filter
    output gap, where a constant offset is irrelevant.
    """
    g = g.dropna()
    lg = np.log1p(g / 100.0)
    out = pd.Series(0.0, index=g.index)
    for i in range(4, len(g)):
        out.iloc[i] = out.iloc[i - 4] + lg.iloc[i]
    return out.rename("gdp_log_level")


def cumulate_growth(growth_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Rebuild a level index by cumulating period growth rates (in %) from ``base`` (splicing rule 4)."""
    g = growth_pct.fillna(0.0) / 100.0
    return (base * (1.0 + g).cumprod()).rename(growth_pct.name)


def month_end_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=MONTH_END)


def apply_lags(df: pd.DataFrame, lags: Mapping[str, int]) -> pd.DataFrame:
    """Columns ``{col}_l{k}`` for k = 1..n where lag 1 is the current (latest known) value.

    This is the "latest value known at the origin" convention used by the forecasting features.
    """
    cols: dict[str, pd.Series] = {}
    for col, n in lags.items():
        for k in range(1, n + 1):
            cols[f"{col}_l{k}"] = df[col].shift(k - 1)
    return pd.DataFrame(cols, index=df.index)
