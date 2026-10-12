"""Metrics, coverage and Diebold-Mariano tests (plan Section 8).

Input frames follow the forecast schema (``origin, target_date, horizon, q05..q95, actual``).
``build_metrics`` returns the ``metrics.parquet`` schema plus two additive columns, ``regime``
(all | calm | episode) and ``mean_width80``/``n``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from oilshock import config

METRIC_COLUMNS = [
    "model",
    "target",
    "horizon",
    "rmse",
    "mae",
    "mase",
    "coverage80",
    "coverage95",
    "pinball",
    "dm_pvalue_vs_naive",
    "regime",
    "mean_width80",
    "n",
]


def pinball_loss(df: pd.DataFrame, q: float | str) -> float:
    """Mean pinball loss of one quantile column (``"q10"`` or ``0.10``)."""
    col = q if isinstance(q, str) else config.QCOLS[list(config.QUANTILES).index(q)]
    tau = config.QMAP[col]
    d = df.dropna(subset=["actual", col])
    if d.empty:
        return float("nan")
    e = d["actual"] - d[col]
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def point_metrics(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["actual", "q50"])
    if d.empty:
        return {"rmse": np.nan, "mae": np.nan, "n": 0}
    e = d["actual"] - d["q50"]
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "n": int(len(d)),
    }


def interval_metrics(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["actual", "q05", "q95"])
    if d.empty:
        return {"coverage80": np.nan, "coverage95": np.nan, "mean_width80": np.nan}
    return {
        "coverage80": float(((d["actual"] >= d["q10"]) & (d["actual"] <= d["q90"])).mean()),
        "coverage95": float(((d["actual"] >= d["q05"]) & (d["actual"] <= d["q95"])).mean()),
        "mean_width80": float((d["q90"] - d["q10"]).mean()),
    }


def mean_pinball(df: pd.DataFrame) -> float:
    return float(np.nanmean([pinball_loss(df, c) for c in config.QCOLS]))


def diebold_mariano(
    e1: np.ndarray | pd.Series, e2: np.ndarray | pd.Series, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss with Newey-West variance and the Harvey et al. correction.

    Positive statistic: model 1 has the larger loss. Returns ``(statistic, two-sided p-value)``.
    """
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dbar = d.mean()
    dc = d - dbar
    lags = max(h - 1, 0)
    gamma0 = float(dc @ dc) / n
    var = gamma0
    for lg in range(1, lags + 1):
        var += 2.0 * (1.0 - lg / (lags + 1.0)) * float(dc[lg:] @ dc[:-lg]) / n
    var = max(var, 1e-12)
    stat = dbar / np.sqrt(var / n)
    hl = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hl
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def episode_mask(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """True where a date falls inside any episode test window (open end for the live episode)."""
    idx = pd.DatetimeIndex(dates)
    mask = np.zeros(len(idx), dtype=bool)
    for ep in config.episode_defs().values():
        lo = pd.Timestamp(ep["test_start"])
        hi = pd.Timestamp(ep["test_end"]) if ep.get("test_end") else pd.Timestamp.max
        mask |= (idx >= lo) & (idx <= hi)
    return mask


def _errors(df: pd.DataFrame) -> pd.Series:
    d = df.dropna(subset=["actual", "q50"])
    return (d["actual"] - d["q50"]).set_axis(pd.DatetimeIndex(d["origin"]))


def build_metrics(forecasts: dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
    """Metrics for every model at every horizon, for all / calm / episode target dates.

    ``forecasts`` maps model name to its backtest frame; ``"naive"`` must be present (MASE and DM
    are relative to it, matched on origin and horizon).
    """
    if "naive" not in forecasts:
        raise KeyError("the naive forecast is required as the benchmark")
    rows = []
    prepared: dict[str, pd.DataFrame] = {}
    for model, df in forecasts.items():
        d = df.copy()
        d["origin"] = pd.to_datetime(d["origin"])
        d["target_date"] = pd.to_datetime(d["target_date"])
        d = d[d["actual"].notna()].copy()
        d["_epi"] = episode_mask(d["target_date"])
        prepared[model] = d
    naive = prepared["naive"]
    for model, d in prepared.items():
        for regime in ("all", "calm", "episode"):
            sub = d if regime == "all" else d[d["_epi"] == (regime == "episode")]
            nsub = naive if regime == "all" else naive[naive["_epi"] == (regime == "episode")]
            for h, g in sub.groupby("horizon"):
                ng = nsub[nsub["horizon"] == h]
                pm, im = point_metrics(g), interval_metrics(g)
                merged = g.merge(ng[["origin", "actual", "q50"]], on="origin", suffixes=("", "_n"))
                if len(merged):
                    mae_n = float(np.mean(np.abs(merged["actual"] - merged["q50_n"])))
                    mae_m = float(np.mean(np.abs(merged["actual"] - merged["q50"])))
                    mase = mae_m / mae_n if mae_n > 0 else np.nan
                else:
                    mase = np.nan
                if model == "naive" or merged.empty:
                    dm_p = np.nan if model != "naive" else np.nan
                else:
                    e1 = (merged["actual"] - merged["q50"]).to_numpy()
                    e2 = (merged["actual"] - merged["q50_n"]).to_numpy()
                    dm_p = diebold_mariano(e1, e2, int(h))[1]
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "horizon": int(h),
                        "rmse": pm["rmse"],
                        "mae": pm["mae"],
                        "mase": 1.0 if model == "naive" and pm["n"] else mase,
                        "coverage80": im["coverage80"],
                        "coverage95": im["coverage95"],
                        "pinball": mean_pinball(g),
                        "dm_pvalue_vs_naive": dm_p,
                        "regime": regime,
                        "mean_width80": im["mean_width80"],
                        "n": pm["n"],
                    }
                )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
