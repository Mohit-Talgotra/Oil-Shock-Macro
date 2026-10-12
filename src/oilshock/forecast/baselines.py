"""Naive, seasonal naive and AR(p) baselines (plan Section 8).

Naive and seasonal naive take their uncertainty from the empirical quantiles of their own past
h-step errors; AR(p) uses a residual bootstrap. None of them uses oil.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.forecast.base import (
    QUANTILES,
    Forecaster,
    TargetSpec,
    normal_quantiles,
    quantile_frame,
    target_values,
)


def _point_paths(y: pd.Series, h: int, kind: str, seasonal: int | None) -> pd.Series:
    """Naive / seasonal-naive point forecast of the h-step target for *every* origin in ``y``."""
    if seasonal is None:  # no change in the level of z (level) or zero cumulative change (cum)
        return y if kind == "level" else pd.Series(0.0, index=y.index)
    if kind == "level":
        m = math.ceil(h / seasonal)
        return y.shift(seasonal * m - h)
    total = pd.Series(0.0, index=y.index)
    for j in range(1, h + 1):
        m = math.ceil(j / seasonal)
        total = total + y.shift(seasonal * m - j)
    return total


class NaiveForecaster(Forecaster):
    name = "naive"
    seasonal_period: int | None = None

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> NaiveForecaster:
        self.y = y.astype(float)
        self.min_err = int(config.models_cfg()["forecast"]["baselines"]["empirical_min_errors"])
        return self

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        kind = self.spec.kind
        y = self.y
        out = np.zeros((h, len(quantiles)))
        for j in range(1, h + 1):
            point_all = _point_paths(y, j, kind, self.seasonal_period)
            point = float(point_all.iloc[-1])
            real = target_values(y, j, kind)
            err = (real - point_all).dropna()
            if len(err) >= self.min_err:
                eq = np.quantile(err.to_numpy(), quantiles)
            else:  # too little history: normal with the s.d. of one-step changes scaled by sqrt(h)
                sd = float(y.diff().std(ddof=1)) if len(y) > 2 else 1.0
                eq = normal_quantiles(np.zeros(1), np.array([sd * math.sqrt(j)]), quantiles)[0]
            out[j - 1] = point + eq
        return quantile_frame(out, h)


class SeasonalNaiveForecaster(NaiveForecaster):
    name = "snaive"

    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        self.seasonal_period = config.models_cfg()["forecast"]["baselines"]["seasonal_period"][
            spec.freq
        ]


class ARForecaster(Forecaster):
    """AR(p) by BIC on the training window (statsmodels ``AutoReg``), residual-bootstrap intervals."""

    name = "ar"

    def __init__(self, spec: TargetSpec, seed: int | None = None):
        super().__init__(spec)
        cfg = config.models_cfg()
        self.maxlags = int(cfg["forecast"]["baselines"]["ar_maxlags"][spec.freq])
        self.reps = int(cfg["forecast"]["baselines"]["bootstrap_reps"])
        self.seed = int(cfg["seed"] if seed is None else seed)

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> ARForecaster:
        from statsmodels.tsa.ar_model import AutoReg

        yy = y.to_numpy(dtype=float)
        top = max(1, min(self.maxlags, len(yy) // 4))
        best_bic, best = np.inf, None
        for p in range(1, top + 1):
            res = AutoReg(yy, lags=p, trend="c", hold_back=top, old_names=False).fit()
            if res.bic < best_bic:
                best_bic, best = res.bic, p
        res = AutoReg(yy, lags=best, trend="c", old_names=False).fit()
        self.p = int(best)
        self.params = np.asarray(res.params, dtype=float)  # const, phi_1..phi_p
        self.resid = np.asarray(res.resid, dtype=float)
        self.tail = yy[-self.p :][::-1].copy()  # most recent first
        return self

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.reps
        e = rng.choice(self.resid - self.resid.mean(), size=(n, h), replace=True)
        hist = np.tile(self.tail, (n, 1))  # (n, p), most recent first
        c, phi = self.params[0], self.params[1:]
        paths = np.zeros((n, h))
        for s in range(h):
            nxt = c + hist @ phi + e[:, s]
            paths[:, s] = nxt
            hist = np.column_stack([nxt, hist[:, :-1]]) if self.p > 1 else nxt[:, None]
        tgt = paths if self.spec.kind == "level" else np.cumsum(paths, axis=1)
        out = np.quantile(tgt, quantiles, axis=0).T  # (h, nq)
        return quantile_frame(out, h)
