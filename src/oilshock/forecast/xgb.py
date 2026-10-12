"""Direct multi-horizon quantile XGBoost with SHAP summaries (plan Section 8).

One model per horizon and quantile (direct strategy). Features use only data available at the origin;
"lag 1" is the latest value known at the origin. The future oil path enters as one extra feature,
``oil_fut_cum`` = cumulative oil change over (origin, origin + h]: in training it is the realised value,
at prediction time it is the assumed path (flat = 0, realised path in oracle runs, a grid value in the
app). So the same fitted models serve blind and oracle information sets and the oil-path grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.data import transform
from oilshock.forecast.base import (
    QUANTILES,
    Forecaster,
    TargetSpec,
    get_target,
    quantile_frame,
    target_values,
)

OIL_FUT = "oil_fut_cum"


def feature_lags(spec: TargetSpec, cfg: dict | None = None) -> dict[str, int]:
    cfg = cfg or config.models_cfg()
    freq = "monthly" if spec.freq == "M" else "quarterly"
    return dict(cfg["forecast"]["xgb"]["features"][spec.name][freq])


def make_features(
    panel: pd.DataFrame, target: TargetSpec | str, lags: dict[str, int] | None = None
) -> pd.DataFrame:
    """Feature matrix for every origin date in ``panel`` (uses only values dated at or before it)."""
    spec = target if isinstance(target, TargetSpec) else get_target(target)
    lags = lags or feature_lags(spec)
    cfg = config.models_cfg()["forecast"]["xgb"]
    missing = [c for c in lags if c not in panel.columns]
    if missing:
        raise KeyError(f"panel lacks feature columns {missing}")
    feats = transform.apply_lags(panel, lags)
    if cfg.get("month_dummies", True):
        months = panel.index.month
        for m in range(1, 13):
            feats[f"month_{m}"] = (months == m).astype(float)
    if cfg.get("regime_dummy"):
        start = pd.Timestamp(config.events_cfg()["dummies"]["inflation_targeting"]["start"])
        feats["inflation_targeting"] = (panel.index >= start).astype(float)
    return feats


class XGBQuantileForecaster(Forecaster):
    name = "xgb"
    supports_covariates = True

    def __init__(
        self,
        spec: TargetSpec,
        horizons: int | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
        lags: dict[str, int] | None = None,
        seed: int | None = None,
    ):
        super().__init__(spec)
        cfg = config.models_cfg()
        self.params = {
            k: cfg["forecast"]["xgb"][k]
            for k in ("max_depth", "n_estimators", "learning_rate", "subsample")
        }
        self.horizons = int(horizons or spec.horizons)
        self.quantiles = tuple(quantiles)
        self.lags = lags or feature_lags(spec, cfg)
        self.seed = int(cfg["seed"] if seed is None else seed)
        self.models: dict[tuple[int, float], object] = {}
        self.min_rows = 24

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> XGBQuantileForecaster:
        import xgboost as xgb

        if X is None:
            raise ValueError("XGBoost needs the panel history X")
        X = X.loc[: y.index.max()]
        F = make_features(X, self.spec, self.lags)
        oil = X["oil"].astype(float)
        z = X[self.spec.z].astype(float)
        self.feature_names = list(F.columns) + [OIL_FUT]
        self.F_last = F.iloc[[-1]].copy()
        self.train_sets: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
        self.models.clear()
        for h in range(1, self.horizons + 1):
            tgt = target_values(z, h, self.spec.kind)
            fut = oil.rolling(h).sum().shift(-h)
            D = F.copy()
            D[OIL_FUT] = fut
            ok = tgt.notna() & fut.notna()
            Dtr, ytr = D[ok], tgt[ok]
            if len(Dtr) < self.min_rows:
                raise ValueError(f"only {len(Dtr)} training rows at horizon {h}")
            self.train_sets[h] = (Dtr, ytr)
            for q in self.quantiles:
                m = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=float(q),
                    tree_method="hist",
                    n_jobs=1,
                    random_state=self.seed,
                    **self.params,
                )
                m.fit(Dtr, ytr)
                self.models[(h, q)] = m
        return self

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        if h > self.horizons:
            raise ValueError(f"fitted for horizons up to {self.horizons}")
        oil = self.oil_from(X_future, h)
        cum_oil = np.cumsum(oil)
        out = np.zeros((h, len(quantiles)))
        for j in range(1, h + 1):
            row = self.F_last.copy()
            row[OIL_FUT] = cum_oil[j - 1]
            for k, q in enumerate(quantiles):
                out[j - 1, k] = float(self.models[(j, q)].predict(row[self.feature_names])[0])
        return quantile_frame(out, h)  # sorted per row: guards against quantile crossing

    def grid_forecast(
        self, h: int, total_pcts: list[float], horizon_months: int = 12
    ) -> pd.DataFrame:
        """Forecast for a grid of total oil changes over ``horizon_months`` (linear ramp), long format."""
        rows = []
        for pct in total_pcts:
            lp = 100.0 * np.log1p(pct / 100.0)
            path = np.full(h, lp / horizon_months)
            fc = self.predict(h, pd.DataFrame({"oil": path}))
            fc = fc.reset_index()
            fc.insert(0, "oil_pct", pct)
            rows.append(fc)
        return pd.concat(rows, ignore_index=True)


def shap_summary(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature for a fitted XGBoost model."""
    try:
        import shap

        values = shap.TreeExplainer(model).shap_values(X)
    except ImportError:  # xgboost's own TreeSHAP gives the same attributions
        import xgboost as xgb

        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        values = contrib[:, :-1]
    imp = np.abs(np.asarray(values)).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
