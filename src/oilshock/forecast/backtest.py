"""Rolling-origin backtest engine (plan Section 8).

At every origin each model is refit on data dated at or before the origin only; the engine asserts it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.forecast.base import (
    Forecaster,
    TargetSpec,
    future_dates,
    target_values,
)

log = logging.getLogger(__name__)

MODEL_ORDER = ["naive", "snaive", "ar", "arimax", "var", "xgb"]


def make_forecaster(name: str, spec: TargetSpec) -> Forecaster:
    """Registry of model families (imports are lazy so the app never needs xgboost for baselines)."""
    if name == "naive":
        from oilshock.forecast.baselines import NaiveForecaster

        return NaiveForecaster(spec)
    if name == "snaive":
        from oilshock.forecast.baselines import SeasonalNaiveForecaster

        return SeasonalNaiveForecaster(spec)
    if name == "ar":
        from oilshock.forecast.baselines import ARForecaster

        return ARForecaster(spec)
    if name == "arimax":
        from oilshock.forecast.arima import ARIMAXForecaster

        return ARIMAXForecaster(spec)
    if name == "var":
        from oilshock.forecast.arima import VARForecaster

        return VARForecaster(spec)
    if name == "xgb":
        from oilshock.forecast.xgb import XGBQuantileForecaster

        return XGBQuantileForecaster(spec)
    if name in {"timesfm_zs", "timesfm_cov"}:
        from oilshock.forecast.timesfm_runner import TimesFMForecaster

        return TimesFMForecaster(spec, covariates=(name == "timesfm_cov"))
    raise KeyError(name)


def make_origins(
    panel: pd.DataFrame, spec: TargetSpec, stride: int | None = None
) -> list[pd.Timestamp]:
    """Origins from 2012 (Jan monthly, Q1 quarterly) to the latest observation minus one period."""
    cfg = config.models_cfg()["forecast"]
    start = pd.Timestamp(cfg["origins"]["monthly_start" if spec.freq == "M" else "quarterly_start"])
    stride = int(cfg["stride"] if stride is None else stride)
    z = panel[spec.z].dropna()
    last = z.index.max()
    idx = z.index[(z.index >= start) & (z.index < last)]
    min_train = int(cfg["min_train_months" if spec.freq == "M" else "min_train_quarters"])
    idx = [t for t in idx if (z.index <= t).sum() >= min_train]
    return idx[::stride]


def _forecast_rows(
    fc: pd.DataFrame,
    origin: pd.Timestamp,
    spec: TargetSpec,
    actual_by_h: dict[int, float],
    oil_assumption: str,
) -> pd.DataFrame:
    h = len(fc)
    dates = future_dates(origin, h, spec.alias)
    out = fc.reset_index()
    out.insert(0, "origin", origin)
    out.insert(1, "target_date", dates)
    out["actual"] = [actual_by_h.get(int(k), np.nan) for k in out["horizon"]]
    out["oil_assumption"] = oil_assumption
    return out[config.FORECAST_COLUMNS]


def rolling_origin(
    forecaster: Forecaster,
    panel: pd.DataFrame,
    target: TargetSpec,
    origins: Iterable[pd.Timestamp],
    horizons: int | None = None,
    oil_assumption: str = "flat",
) -> pd.DataFrame:
    """Backtest one model on one target. Returns rows in the ``forecasts/{model}_{target}`` schema.

    Training data is ``panel.loc[:origin]``; the engine asserts that no training row is dated after the
    origin. ``oil_assumption="flat"`` keeps oil at its origin value over the horizon.
    """
    h_max = int(horizons or target.horizons)
    z_full = panel[target.z]
    realised = {h: target_values(z_full, h, target.kind) for h in range(1, h_max + 1)}
    frames = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        hist = panel.loc[:origin]
        assert hist.index.max() <= origin, "training data extends past the origin"
        y = hist[target.z].dropna()
        if y.index.max() != origin:
            continue  # the model series is not observed at this origin
        forecaster.fit(y, hist)
        assert forecaster.train_end <= origin, "forecaster trained past the origin"
        fc = forecaster.predict(h_max, None)
        actual = {h: realised[h].get(origin, np.nan) for h in range(1, h_max + 1)}
        frames.append(_forecast_rows(fc, origin, target, actual, oil_assumption))
    if not frames:
        return pd.DataFrame(columns=config.FORECAST_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def latest_forecast(
    forecaster: Forecaster,
    panel: pd.DataFrame,
    target: TargetSpec,
    oil_path: np.ndarray | None = None,
    oil_assumption: str = "flat",
) -> pd.DataFrame:
    """Fit on all data and forecast from the final observation (actual is NaN: not yet observed)."""
    y = panel[target.z].dropna()
    origin = y.index.max()
    forecaster.fit(y, panel.loc[:origin])
    fc = forecaster.predict(
        target.horizons,
        None if oil_path is None else pd.DataFrame({"oil": oil_path}),
    )
    return _forecast_rows(fc, origin, target, {}, oil_assumption)
