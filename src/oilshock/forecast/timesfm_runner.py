"""STRETCH: TimesFM zero-shot and covariate runs (plan Section 8, "TimesFM protocol").

* Run offline only (laptop or Google Colab) and never add ``timesfm`` to ``requirements.txt``.
* ``timesfm`` is imported lazily so the app and the rest of the pipeline never need it.
* Writes the same forecast schema as every other model, so metrics are comparable.
* Version 2.5 (200M parameters, Apache-2.0 weights) is implemented. Version 3.0 (native multivariate
  covariates, different licence for the weights) is NOT implemented: check the licence first.

The TimesFM API has changed between releases and this module could not be exercised in the build
environment (no network). Treat it as a scaffold: if a call fails, the error message says which one,
and ``manifest.json`` records the exact package version and checkpoint actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.forecast.base import (
    QUANTILES,
    Forecaster,
    TargetSpec,
    quantile_frame,
)

_MODEL_CACHE: dict[str, object] = {}
Z90 = 1.2815515655446004
Z95 = 1.6448536269514722


def _load_model(version: str, horizon: int, context: int):
    try:
        import timesfm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'timesfm is not installed. Install it in the dev environment only: pip install "timesfm[torch]"'
        ) from exc
    if version != "2.5":
        raise NotImplementedError(
            "Only TimesFM 2.5 is implemented. For 3.0, confirm the weights' licence and add a loader here."
        )
    key = f"{version}:{horizon}:{context}"
    if key not in _MODEL_CACHE:
        cfg = config.models_cfg()["timesfm"]
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg["checkpoint"])
        model.compile(
            timesfm.ForecastConfig(
                max_context=context,
                max_horizon=max(horizon, 16),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def package_info() -> dict[str, str]:
    """Exact package version and checkpoint, for ``manifest.json``."""
    try:
        from importlib.metadata import version

        v = version("timesfm")
    except Exception:
        v = "not installed"
    cfg = config.models_cfg()["timesfm"]
    return {"timesfm": v, "checkpoint": cfg["checkpoint"], "model_version": cfg["version"]}


def run_timesfm(
    series: pd.Series,
    horizon: int,
    covariates: pd.DataFrame | None = None,
    version: str = "2.5",
) -> pd.DataFrame:
    """Forecast ``series`` ``horizon`` steps ahead. Returns columns mean, q05, q10, q50, q90, q95.

    TimesFM returns deciles; q05 and q95 are extrapolated from q10/q50/q90 assuming a normal tail
    (a documented approximation). ``covariates`` must contain past values aligned with ``series`` plus
    ``horizon`` future rows (e.g. the assumed oil path), one column per covariate.
    """
    cfg = config.models_cfg()["timesfm"]
    context = int(cfg["context"])
    model = _load_model(version, horizon, context)
    ctx = series.dropna().to_numpy(dtype=float)[-context:]
    if covariates is None:
        _, quant = model.forecast(horizon=horizon, inputs=[ctx])
    else:
        n = len(series.dropna())
        cov = {
            c: [covariates[c].to_numpy(dtype=float)[-(context + horizon) :]]
            for c in covariates.columns
        }
        if not hasattr(model, "forecast_with_covariates"):
            raise RuntimeError(
                "this timesfm release has no forecast_with_covariates; use zero-shot runs"
            )
        _, quant = model.forecast_with_covariates(
            inputs=[ctx],
            dynamic_numerical_covariates=cov,
            xreg_mode="xreg + timesfm",
            ridge=0.0,
            normalize_xreg_target_per_input=True,
        )
        del n
    q = np.asarray(quant)[0]  # (horizon, 10): mean, q10..q90
    q10, q50, q90 = q[:, 1], q[:, 5], q[:, 9]
    lo_sd = (q50 - q10) / Z90
    hi_sd = (q90 - q50) / Z90
    out = pd.DataFrame(
        {
            "mean": q[:, 0],
            "q05": q50 - Z95 * lo_sd,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "q95": q50 + Z95 * hi_sd,
        },
        index=pd.RangeIndex(1, horizon + 1, name="horizon"),
    )
    return out


class TimesFMForecaster(Forecaster):
    """Wraps ``run_timesfm`` in the common interface (``timesfm_zs`` or ``timesfm_cov``)."""

    supports_covariates = True

    def __init__(self, spec: TargetSpec, covariates: bool = False):
        super().__init__(spec)
        self.use_cov = covariates
        self.name = "timesfm_cov" if covariates else "timesfm_zs"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> TimesFMForecaster:
        self.y = y
        self.X = None if X is None else X.loc[y.index]
        return self

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        cov = None
        if self.use_cov and self.X is not None:
            oil_f = self.oil_from(X_future, h)
            hist = self.X[["oil"]].astype(float).ffill().fillna(0.0)
            cov = pd.concat([hist, pd.DataFrame({"oil": oil_f})], ignore_index=True)
        fc = run_timesfm(self.y, h, cov, config.models_cfg()["timesfm"]["version"])
        if self.spec.kind == "cum":
            sd = ((fc["q90"] - fc["q10"]) / (2 * Z90)).to_numpy()
            mean = fc["q50"].cumsum().to_numpy()
            cum_sd = np.sqrt(np.cumsum(sd**2))  # approximation: independent step errors
            from oilshock.forecast.base import normal_quantiles

            return quantile_frame(normal_quantiles(mean, cum_sd, quantiles), h)
        return quantile_frame(fc[config.QCOLS].to_numpy(), h)
