"""VAR estimation, diagnostics, impulse responses, bootstrap bands and FEVD (plan Section 7).

Two estimation paths share one specification:

* ``fit_var`` returns a statsmodels ``VARResults`` (used for lag selection, diagnostics, pickled
  ``fitted_*.pkl`` and live forecasts in the app);
* ``estimate_var`` is a small numpy OLS estimator (``VARFit``) used for the bootstrap, rolling windows,
  the block-exogeneity robustness check and tests. ``tests/test_var.py`` checks that the two agree.

Variable order comes from ``models.yaml``; oil is always first, so the Cholesky factor identifies the
oil shock as the first orthogonalised shock. Every random step uses an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SIGNIFICANCE_LEVELS = {"lo68": 16.0, "hi68": 84.0, "lo90": 5.0, "hi90": 95.0}


# ======================================================================================= numpy VAR
@dataclass
class VARFit:
    """A VAR(p) with constant and optional deterministic/exogenous regressors, estimated by OLS."""

    names: list[str]
    p: int
    intercept: np.ndarray  # (k,)
    coefs: np.ndarray  # (p, k, k): coefs[i-1] is A_i
    exog_names: list[str]
    exog_coefs: np.ndarray  # (k, m)
    sigma_u: np.ndarray  # (k, k), degrees-of-freedom corrected like statsmodels
    resid: np.ndarray  # (T-p, k)
    endog: np.ndarray  # (T, k) data used
    exog: np.ndarray | None  # (T, m)
    index: pd.Index | None = None
    block_exog: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def nobs(self) -> int:
        return self.endog.shape[0] - self.p


def _design(endog: np.ndarray, p: int, exog: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    T, k = endog.shape
    rows = [np.ones((T - p, 1))]
    for i in range(1, p + 1):
        rows.append(endog[p - i : T - i])
    if exog is not None and exog.shape[1] > 0:
        rows.append(exog[p:])
    return np.hstack(rows), endog[p:]


def estimate_var(
    endog: pd.DataFrame | np.ndarray,
    p: int,
    exog: pd.DataFrame | np.ndarray | None = None,
    block_exog: bool = False,
) -> VARFit:
    """OLS estimation, equation by equation.

    ``block_exog=True`` restricts the first equation (oil) to a constant and its own lags only
    (plan Section 7, robustness 2): the other equations are unrestricted.
    """
    names = (
        list(endog.columns)
        if isinstance(endog, pd.DataFrame)
        else [f"y{i}" for i in range(endog.shape[1])]
    )
    index = endog.index if isinstance(endog, pd.DataFrame) else None
    y = np.asarray(endog, dtype=float)
    x = None if exog is None else np.asarray(exog, dtype=float)
    exog_names = (
        list(exog.columns)
        if isinstance(exog, pd.DataFrame)
        else ([f"x{i}" for i in range(0 if x is None else x.shape[1])])
    )
    T, k = y.shape
    m = 0 if x is None else x.shape[1]
    X, Y = _design(y, p, x)
    ncoef = X.shape[1]
    if X.shape[0] <= ncoef:
        raise ValueError(f"too few observations ({X.shape[0]}) for {ncoef} regressors per equation")
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # (ncoef, k)
    if block_exog:
        keep = [0] + [1 + lag * k + 0 for lag in range(p)]  # const + oil's own lag in each block
        b0 = np.linalg.lstsq(X[:, keep], Y[:, 0], rcond=None)[0]
        B[:, 0] = 0.0
        B[keep, 0] = b0
    U = Y - X @ B
    df = X.shape[0] - ncoef
    if block_exog:
        # equation-specific df: oil equation has len(keep) regressors
        sig = np.empty((k, k))
        dfv = np.array(
            [X.shape[0] - (len(keep) if j == 0 else ncoef) for j in range(k)], dtype=float
        )
        S = U.T @ U
        for a in range(k):
            for b in range(k):
                sig[a, b] = S[a, b] / np.sqrt(dfv[a] * dfv[b])
        sigma = sig
    else:
        sigma = U.T @ U / df
    intercept = B[0]
    coefs = np.stack([B[1 + (i - 1) * k : 1 + i * k].T for i in range(1, p + 1)])
    exog_coefs = B[1 + p * k :].T if m else np.zeros((k, 0))
    return VARFit(
        names=names,
        p=p,
        intercept=intercept,
        coefs=coefs,
        exog_names=exog_names,
        exog_coefs=exog_coefs,
        sigma_u=sigma,
        resid=U,
        endog=y,
        exog=x,
        index=index,
        block_exog=block_exog,
    )


def ma_coefs(coefs: np.ndarray, steps: int) -> np.ndarray:
    """Moving-average matrices Phi_0..Phi_steps for A_1..A_p; shape (steps+1, k, k)."""
    p, k, _ = coefs.shape
    phi = np.zeros((steps + 1, k, k))
    phi[0] = np.eye(k)
    for s in range(1, steps + 1):
        for j in range(1, min(s, p) + 1):
            phi[s] += coefs[j - 1] @ phi[s - j]
    return phi


def cholesky_impact(sigma: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(sigma)


def orth_irf(
    fit_or_coefs: VARFit | np.ndarray, sigma: np.ndarray | None = None, horizon: int = 24
) -> np.ndarray:
    """Orthogonalised IRFs, shape (horizon+1, k, k): ``[h, i, j]`` = response of variable i to shock j."""
    if isinstance(fit_or_coefs, VARFit):
        coefs, sigma = fit_or_coefs.coefs, fit_or_coefs.sigma_u
    else:
        coefs = fit_or_coefs
    P = cholesky_impact(sigma)
    return ma_coefs(coefs, horizon) @ P


def oil_irf(fit: VARFit, horizon: int) -> tuple[np.ndarray, float]:
    """Response of every variable to a one-s.d. oil shock (shape (horizon+1, k)) and sigma_oil (log pts)."""
    irf = orth_irf(fit, horizon=horizon)
    return irf[:, :, 0], float(cholesky_impact(fit.sigma_u)[0, 0])


def fevd_oil(fit: VARFit, horizon: int) -> np.ndarray:
    """Share of each variable's forecast-error variance due to the oil shock; shape (horizon+1, k)."""
    th = orth_irf(fit, horizon=horizon)  # (H+1, k, k)
    sq = np.cumsum(th**2, axis=0)  # (H+1, k, k)
    total = sq.sum(axis=2)
    return sq[:, :, 0] / np.where(total == 0, np.nan, total)


# ======================================================================================= bootstrap
def _simulate(fit: VARFit, u_star: np.ndarray) -> np.ndarray:
    p, k = fit.p, fit.k
    T = fit.endog.shape[0]
    y = np.empty((T, k))
    y[:p] = fit.endog[:p]
    xe = fit.exog
    c = fit.intercept
    for t in range(p, T):
        mu = c.copy()
        for i in range(1, p + 1):
            mu += fit.coefs[i - 1] @ y[t - i]
        if xe is not None and fit.exog_coefs.shape[1]:
            mu += fit.exog_coefs @ xe[t]
        y[t] = mu + u_star[t - p]
    return y


def bootstrap_irfs(
    fit: VARFit, horizon: int, reps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Residual (recursive-design) bootstrap of the oil-shock IRF.

    Returns ``(draws, cum_draws)`` each shaped (reps, horizon+1, k) per one-s.d. oil shock.
    """
    rng = np.random.default_rng(seed)
    U = fit.resid - fit.resid.mean(axis=0)
    Teff = U.shape[0]
    k = fit.k
    draws = np.full((reps, horizon + 1, k), np.nan)
    cum = np.full((reps, horizon + 1, k), np.nan)
    for r in range(reps):
        u_star = U[rng.integers(0, Teff, Teff)]
        y_star = _simulate(fit, u_star)
        try:
            bf = estimate_var(y_star, fit.p, fit.exog, block_exog=fit.block_exog)
            irf, _ = oil_irf(bf, horizon)
        except (np.linalg.LinAlgError, ValueError):
            continue
        draws[r] = irf
        cum[r] = np.cumsum(irf, axis=0)
    return draws, cum


def _band_columns(draws: np.ndarray) -> dict[str, np.ndarray]:
    good = draws[~np.isnan(draws).any(axis=(1, 2))]
    if len(good) < max(20, int(0.5 * len(draws))):
        raise RuntimeError("bootstrap failed for most replications; check the specification")
    return {name: np.percentile(good, q, axis=0) for name, q in SIGNIFICANCE_LEVELS.items()}


def _long_frame(
    names: list[str],
    point: np.ndarray,
    bands: dict[str, np.ndarray],
    cumulative: bool,
    sigma_oil: float,
) -> pd.DataFrame:
    H1, k = point.shape
    rows = []
    for j, name in enumerate(names):
        frame = pd.DataFrame(
            {
                "response": name,
                "horizon": np.arange(H1),
                "point": point[:, j],
                "lo68": bands["lo68"][:, j],
                "hi68": bands["hi68"][:, j],
                "lo90": bands["lo90"][:, j],
                "hi90": bands["hi90"][:, j],
                "cumulative": cumulative,
                "shock_size_logpts": sigma_oil,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _as_fit(res: Any) -> VARFit:
    """Accept a ``VARFit`` or (best effort) a statsmodels ``VARResults``."""
    if isinstance(res, VARFit):
        return res
    names = list(res.names)
    endog = pd.DataFrame(np.asarray(res.endog), columns=names)
    exog = getattr(res, "exog", None)
    if exog is None:
        exog = getattr(getattr(res, "model", None), "exog", None)
    return estimate_var(endog, int(res.k_ar), exog)


def irf_bands(res: Any, horizon: int, reps: int, seed: int) -> pd.DataFrame:
    """Orthogonalised oil-shock IRFs with 68% and 90% bootstrap bands, plain and cumulative.

    Output schema (``artifacts/var/irf_*.parquet``): response, horizon, point, lo68, hi68, lo90, hi90,
    cumulative, shock_size_logpts.
    """
    fit = _as_fit(res)
    point, sigma_oil = oil_irf(fit, horizon)
    draws, cum_draws = bootstrap_irfs(fit, horizon, reps, seed)
    plain = _long_frame(fit.names, point, _band_columns(draws), False, sigma_oil)
    cumul = _long_frame(
        fit.names, np.cumsum(point, axis=0), _band_columns(cum_draws), True, sigma_oil
    )
    return pd.concat([plain, cumul], ignore_index=True)


def fevd_table(res: Any, horizon: int) -> pd.DataFrame:
    """Share of each variable's forecast-error variance explained by oil, h = 0..horizon."""
    fit = _as_fit(res)
    share = fevd_oil(fit, horizon)
    rows = [
        pd.DataFrame(
            {"variable": name, "horizon": np.arange(horizon + 1), "share_oil": share[:, j]}
        )
        for j, name in enumerate(fit.names)
    ]
    return pd.concat(rows, ignore_index=True)


def select_lags_bic_np(
    endog: pd.DataFrame, exog: pd.DataFrame | None, maxlags: int, min_df: int = 5
) -> int:
    """Fast numpy BIC lag choice on a common sample (used inside backtests, where speed matters)."""
    ex = usable_exog(exog)
    k = endog.shape[1]
    m = 0 if ex is None else ex.shape[1]
    top = min(maxlags, max(1, (len(endog) - 2 - m - min_df) // (k + 1)))
    best, best_p = np.inf, 1
    y = np.asarray(endog, dtype=float)
    x = None if ex is None else np.asarray(ex, dtype=float)
    for p in range(1, top + 1):
        # common sample: drop the first `top` rows so the criteria are comparable
        yy = y[top - p :]
        xx = None if x is None else x[top - p :]
        try:
            fit = estimate_var(yy, p, xx)
        except ValueError:
            break
        T = fit.nobs
        sign, logdet = np.linalg.slogdet(fit.resid.T @ fit.resid / T)
        n_par = k * (k * p + 1 + m)
        bic = logdet + np.log(T) / T * n_par
        if bic < best:
            best, best_p = bic, p
    return best_p


def rolling_irfs(
    df: pd.DataFrame,
    window: int,
    step: int,
    exog: pd.DataFrame | None = None,
    lags: int = 2,
    response: str = "cpi",
    horizon_months: int = 12,
    reps: int = 200,
    seed: int = 0,
    per_pct: float = 10.0,
) -> pd.DataFrame:
    """Cumulative response of ``response`` over the first ``horizon_months`` to a ``per_pct``% oil rise.

    One row per window end: end_date, point, lo68, hi68, lo90, hi90 (percentage points). Wide bands in
    short windows are expected and are kept, not hidden.
    """
    out = []
    scale_logpts = 100.0 * np.log1p(per_pct / 100.0)
    j = list(df.columns).index(response)
    ends = range(window, len(df) + 1, step)
    for n, end in enumerate(ends):
        sub = df.iloc[end - window : end]
        ex = None if exog is None else exog.iloc[end - window : end]
        ex = _drop_constant_columns(ex)
        try:
            fit = estimate_var(sub, lags, ex)
            point, sig = oil_irf(fit, horizon_months)
            draws, cum = bootstrap_irfs(fit, horizon_months, reps, seed + n)
        except (np.linalg.LinAlgError, ValueError):
            continue
        k_scale = scale_logpts / sig
        p_cum = np.cumsum(point[:, j])[horizon_months - 1] * k_scale
        c_end = cum[:, horizon_months - 1, j]
        c_end = c_end[~np.isnan(c_end)] * k_scale
        if len(c_end) < 20:
            continue
        out.append(
            {
                "end_date": sub.index[-1],
                "point": p_cum,
                "lo68": np.percentile(c_end, 16),
                "hi68": np.percentile(c_end, 84),
                "lo90": np.percentile(c_end, 5),
                "hi90": np.percentile(c_end, 95),
                "n_obs": len(sub),
            }
        )
    return pd.DataFrame(out)


def _drop_constant_columns(exog: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop exogenous columns that are constant in the window (e.g. COVID dummies before 2020)."""
    if exog is None:
        return None
    keep = [c for c in exog.columns if exog[c].nunique(dropna=True) > 1]
    return exog[keep] if keep else None


def usable_exog(exog: pd.DataFrame | None) -> pd.DataFrame | None:
    return _drop_constant_columns(exog)


# ================================================================================= statsmodels paths
def fit_var(df: pd.DataFrame, lags: int, exog: pd.DataFrame | None = None) -> Any:
    """Estimate a VAR with constant by statsmodels; returns ``VARResults``.

    Constant exogenous columns (for example COVID dummies in a pre-2020 window) are dropped. A small
    ``oilshock_meta`` dict is attached to the results so the app can forecast from the pickle alone.
    """
    from statsmodels.tsa.api import VAR

    ex = usable_exog(exog)
    res = VAR(df, exog=ex).fit(lags, trend="c")
    default = {} if ex is None else ex.tail(12).mean().to_dict()
    res.oilshock_meta = {
        "last_date": df.index[-1],
        "freq": "M" if (len(df) > 1 and (df.index[1] - df.index[0]).days < 40) else "Q",
        "exog_default": default,
        "exog_cols": [] if ex is None else list(ex.columns),
        "endog_tail": df.tail(max(lags, 12)),
    }
    return res


def fit_from_results(res: Any) -> VARFit:
    """Extract coefficients from statsmodels ``VARResults`` by *label* (robust to column ordering)."""
    import re

    if isinstance(res, VARFit):
        return res
    names = list(res.names)
    p = int(res.k_ar)
    params = res.params
    if not isinstance(params, pd.DataFrame):
        raise TypeError("expected VARResults fitted on a DataFrame (labelled parameters)")
    k = len(names)
    intercept = (
        params.loc["const"].to_numpy(dtype=float) if "const" in params.index else np.zeros(k)
    )
    coefs = np.stack(
        [params.loc[[f"L{i}.{n}" for n in names]].to_numpy(dtype=float).T for i in range(1, p + 1)]
    )
    exog_names = [i for i in params.index if i != "const" and not re.match(r"^L\d+\.", str(i))]
    exog_coefs = params.loc[exog_names].to_numpy(dtype=float).T if exog_names else np.zeros((k, 0))
    meta = getattr(res, "oilshock_meta", {})
    return VARFit(
        names=names,
        p=p,
        intercept=intercept,
        coefs=coefs,
        exog_names=exog_names,
        exog_coefs=exog_coefs,
        sigma_u=np.asarray(res.sigma_u, dtype=float),
        resid=np.asarray(res.resid, dtype=float),
        endog=np.asarray(res.endog, dtype=float),
        exog=None,
        extras=dict(meta),
    )


def _max_feasible_lags(n: int, k: int, m: int) -> int:
    # need (n - p) > k*p + 1 + m  ->  p < (n - 1 - m) / (k + 1)
    return max(1, int(np.floor((n - 2 - m) / (k + 1))))


def _whiteness_p(res: Any, nlags: int) -> float:
    try:
        n = max(nlags, int(res.k_ar) + 1)
        return float(res.test_whiteness(nlags=n, signif=0.05).pvalue)
    except Exception:
        return float("nan")


def select_lags(
    df: pd.DataFrame,
    maxlags: int,
    exog: pd.DataFrame | None = None,
    alpha: float = 0.05,
    whiteness_nlags: int = 12,
) -> dict[str, Any]:
    """Compare AIC/BIC/HQ up to ``maxlags``, take BIC, then add lags until the whiteness test passes.

    Returns ``{aic, bic, hqic, fpe, chosen, whiteness_pvalue, capped_at}``.
    """
    from statsmodels.tsa.api import VAR

    ex = usable_exog(exog)
    cap = _max_feasible_lags(len(df), df.shape[1], 0 if ex is None else ex.shape[1])
    top = min(maxlags, cap)
    order = VAR(df, exog=ex).select_order(maxlags=top, trend="c").selected_orders
    chosen = max(1, int(order["bic"]))
    pval = float("nan")
    for p in range(chosen, top + 1):
        res = fit_var(df, p, ex)
        pval = _whiteness_p(res, whiteness_nlags)
        chosen = p
        if np.isnan(pval) or pval >= alpha:
            break
    return {
        "aic": int(order["aic"]),
        "bic": int(order["bic"]),
        "hqic": int(order["hqic"]),
        "fpe": int(order["fpe"]),
        "chosen": int(chosen),
        "whiteness_pvalue": pval,
        "capped_at": int(top),
    }


def diagnostics(
    res: Any, whiteness_nlags: int = 12, fevd: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Stability, whiteness, normality, Granger causality from oil, and FEVD shares at 6/12/24.

    ``fevd`` is the table from ``fevd_table`` (the shares are added to the output when given).
    """
    names = list(res.names)
    oil = names[0]
    out: dict[str, Any] = {
        "lags": int(res.k_ar),
        "nobs": int(res.nobs),
        "stable": bool(res.is_stable()),
        "min_root_modulus": float(np.min(np.abs(res.roots))),
        "whiteness_pvalue": _whiteness_p(res, whiteness_nlags),
    }
    try:
        out["normality_pvalue"] = float(res.test_normality().pvalue)
    except Exception:
        out["normality_pvalue"] = float("nan")
    for v in names[1:]:
        try:
            out[f"granger_oil_to_{v}"] = float(
                res.test_causality(caused=v, causing=oil, kind="f").pvalue
            )
        except Exception:
            out[f"granger_oil_to_{v}"] = float("nan")
    if fevd is not None:
        for h in (6, 12, 24):
            if h not in set(fevd["horizon"]):  # quarterly Model B only runs 12 periods
                continue
            sub = fevd[fevd["horizon"] == h].set_index("variable")["share_oil"]
            for v in names:
                out[f"fevd_{v}_h{h}"] = float(sub[v])
    return out


def unit_root_table(
    df: pd.DataFrame, adf_alpha: float = 0.05, kpss_alpha: float = 0.05
) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on every column."""
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = []
    for c in df.columns:
        x = df[c].dropna()
        adf_p = float(adfuller(x, autolag="AIC")[1])
        try:
            kp = float(kpss(x, regression="c", nlags="auto")[1])
        except Exception:
            kp = float("nan")
        rows.append(
            {
                "variable": c,
                "adf_pvalue": adf_p,
                "kpss_pvalue": kp,
                "stationary": bool(adf_p < adf_alpha and (np.isnan(kp) or kp > kpss_alpha)),
            }
        )
    return pd.DataFrame(rows)
