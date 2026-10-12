"""Classical time-series forecasters that "know" about oil: ARIMAX (via SARIMAX) and the Layer 1 VAR.

Both condition on an assumed oil path (``X_future["oil"]``): flat by default, the realised path in
oracle runs, or a user-drawn path in the app. Their bands are conditional on that path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.forecast.base import (
    QUANTILES,
    Forecaster,
    TargetSpec,
    hold_value,
    normal_quantiles,
    quantile_frame,
)
from oilshock.models import scenario
from oilshock.models import var as varmod

# ===================================================================================== ARIMAX


class ARIMAXForecaster(Forecaster):
    """ARMA errors with oil (and, per target, other inputs) as exogenous regressors.

    ``order`` is picked by AIC on the training window only. ``z`` is already stationary-ish (y-o-y rates,
    log differences), so ``d = 0``. The order is re-selected every ``reselect_every`` fits and reused in
    between (still using training data only).
    """

    name = "arimax"
    supports_covariates = True

    def __init__(self, spec: TargetSpec, order: tuple[int, int, int] | None = None):
        super().__init__(spec)
        cfg = config.models_cfg()["forecast"]
        self.max_p = int(cfg["arimax"]["max_p"])
        self.max_q = int(cfg["arimax"]["max_q"])
        self.reselect_every = int(cfg["arimax"]["reselect_every"])
        self.hold = cfg["exog_hold"]
        self.fixed_order = order
        self._order: tuple[int, int, int] | None = order
        self._fits = 0

    # ---- exogenous design
    def _exog_matrix(
        self, X: pd.DataFrame, oil_future: np.ndarray, h: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cols: dict[str, np.ndarray] = {}
        n = len(X)
        for c in self.spec.exog:
            hist = X[c].astype(float).ffill()
            if c == "oil":
                fut = oil_future
            else:
                fut = np.full(h, hold_value(self.hold.get(c, "zero"), hist))
            full = np.concatenate([hist.to_numpy(), fut])
            lags = self.spec.exog_lags if c == "oil" else 0
            for lg in range(lags + 1):
                shifted = (
                    np.concatenate([np.full(lg, np.nan), full[: len(full) - lg]]) if lg else full
                )
                cols[f"{c}_l{lg}"] = shifted
        M = np.column_stack(list(cols.values()))
        return M[:n], M[n:], list(cols)

    def _select_order(self, yy: np.ndarray, ex: np.ndarray) -> tuple[int, int, int]:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best, best_aic = (1, 0, 0), np.inf
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = SARIMAX(yy, exog=ex, order=(p, 0, q), trend="c").fit(
                            disp=False, maxiter=100
                        )
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best, best_aic = (p, 0, q), float(res.aic)
                except Exception:
                    continue
        return best

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> ARIMAXForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if X is None:
            raise ValueError("ARIMAX needs the panel history X")
        Xh = X.loc[y.index]
        M, _, names = self._exog_matrix(Xh, np.zeros(0), 0)
        ok = np.isfinite(M).all(axis=1)
        first = int(np.argmax(ok)) if ok.any() else len(ok)
        yy, ex = y.to_numpy(dtype=float)[first:], M[first:]
        if not np.isfinite(ex).all():
            ex = np.nan_to_num(ex, nan=0.0)
        if (
            self._order is None
            or self.fixed_order is None
            and self._fits % self.reselect_every == 0
        ):
            self._order = self._select_order(yy, ex)
        self._fits += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.res = SARIMAX(yy, exog=ex, order=self._order, trend="c").fit(
                    disp=False, maxiter=200
                )
            except Exception:
                self._order = (1, 0, 0)
                self.res = SARIMAX(yy, exog=ex, order=self._order, trend="c").fit(
                    disp=False, maxiter=200
                )
        self.X_hist = Xh[list(self.spec.exog)].copy()
        self.exog_names = names
        return self

    def _psi(self, n: int) -> tuple[np.ndarray, float]:
        from statsmodels.tsa.arima_process import arma2ma

        pars = dict(
            zip(self.res.param_names, np.asarray(self.res.params, dtype=float), strict=True)
        )
        p, _, q = self._order
        ar = np.r_[1.0, [-pars.get(f"ar.L{i}", 0.0) for i in range(1, p + 1)]]
        ma = np.r_[1.0, [pars.get(f"ma.L{i}", 0.0) for i in range(1, q + 1)]]
        return arma2ma(ar, ma, lags=n), float(pars.get("sigma2", np.var(self.res.resid)))

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        oil = self.oil_from(X_future, h)
        _, Mf, _ = self._exog_matrix(self.X_hist, oil, h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.res.get_forecast(steps=h, exog=Mf)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        if self.spec.kind == "level":
            se = np.asarray(fc.se_mean, dtype=float)
            return quantile_frame(normal_quantiles(mean, se, quantiles), h)
        psi, s2 = self._psi(h)
        cum_mean = np.cumsum(mean)
        cum_se = np.empty(h)
        for j in range(1, h + 1):
            coef = np.cumsum(psi[:j])[::-1]  # sum_{i=0}^{j-r} psi_i for r = 1..j
            cum_se[j - 1] = np.sqrt(s2 * np.sum(coef**2))
        return quantile_frame(normal_quantiles(cum_mean, cum_se, quantiles), h)


# ========================================================================================= VAR

# which VAR variable drives each target and how it maps to the target
VAR_TARGET_MAP = {
    "cpi": ("cpi", "yoy"),
    "inr": ("inr", "cum"),
    "gdp": ("gdp_growth", "level"),
}


class VARForecaster(Forecaster):
    """Layer 1 VAR (Model A for monthly targets, Model B for GDP) used as a forecaster.

    Oil is fixed to the assumed path (conditional forecast), so the bands exclude oil uncertainty.
    CPI y-o-y is anchored to the latest actual y-o-y rate and moved by the forecast m-o-m changes
    minus the months leaving the 12-month window.
    """

    name = "var"
    supports_covariates = True

    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        cfg = config.models_cfg()
        self.vcfg = cfg["var_monthly"] if spec.freq == "M" else cfg["var_quarterly"]
        self.var_name, self.mode = VAR_TARGET_MAP[spec.name]

    def _frame(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cols = [self.vcfg["columns"][v] for v in self.vcfg["variables"]]
        endog = X[cols].copy()
        endog.columns = self.vcfg["variables"]
        exog = X[self.vcfg["exog"]]
        joined = pd.concat([endog, exog], axis=1).dropna()
        return joined[self.vcfg["variables"]], varmod.usable_exog(joined[self.vcfg["exog"]])

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> VARForecaster:
        if X is None:
            raise ValueError("VAR needs the panel history X")
        endog, exog = self._frame(X)
        if self.spec.freq == "M":
            p = varmod.select_lags_bic_np(endog, exog, int(self.vcfg["maxlags"]))
        else:
            p = int(self.vcfg["fixed_lags"])
            k, m = endog.shape[1], 0 if exog is None else exog.shape[1]
            while p > 1 and len(endog) - p - (k * p + 1 + m) < int(self.vcfg["min_df_resid"]):
                p -= 1
        self.fit_ = varmod.estimate_var(endog, p, exog)
        self.fit_.extras["exog_default"] = {} if exog is None else exog.tail(12).mean().to_dict()
        self.last_yoy = float(y.iloc[-1])
        self.var_idx = list(endog.columns).index(self.var_name)
        self.endog_end = endog.index[-1]
        return self

    def predict(self, h, X_future=None, quantiles=QUANTILES) -> pd.DataFrame:
        oil = self.oil_from(X_future, h)
        mean, L, fit = scenario.conditional_moments(self.fit_, h, oil_path=oil)
        j = self.var_idx
        hist = fit.endog[:, j]
        mu = np.zeros(h)
        sd = np.zeros(h)
        for s in range(1, h + 1):
            if self.mode == "yoy":
                leaving = sum(hist[s_ - 13] for s_ in range(1, s + 1)) if s <= 12 else np.nan
                mu[s - 1] = self.last_yoy + mean[:s, j].sum() - leaving
                w = np.ones(s)
            elif self.mode == "cum":
                mu[s - 1] = mean[:s, j].sum()
                w = np.ones(s)
            else:  # level
                mu[s - 1] = mean[s - 1, j]
                w = np.zeros(s)
                w[-1] = 1.0
            sd[s - 1] = np.sqrt(scenario.functional_variance(L, j, w))
        return quantile_frame(normal_quantiles(mu, sd, quantiles), h)
