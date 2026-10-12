"""Scenario engine: from a slider value to a response (plan Section 7) and conditional forecasts.

The VAR is linear, so a shock of any size is a rescaled copy of the estimated one::

    scale(s) = 100 * ln(1 + s/100) / sigma_oil_hat
    IRF_k(h | s) = scale(s) * IRF_k(h)

All functions here are cheap (array multiplications), so the app runs them live.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from oilshock.models import var as varmod

BAND_COLS = ["point", "lo68", "hi68", "lo90", "hi90"]


# --------------------------------------------------------------------------- shock scaling
def log_points(shock_pct: float) -> float:
    """Percent change -> log points (100 x ln(1 + s/100)). A fall of 100% or more is impossible."""
    if shock_pct <= -100:
        raise ValueError("an oil price cannot fall by 100% or more")
    return float(100.0 * np.log1p(shock_pct / 100.0))


def scale_factor(shock_pct: float, oil_sd: float) -> float:
    """Multiple of the estimated one-s.d. oil shock that equals a ``shock_pct`` % price change."""
    if oil_sd <= 0:
        raise ValueError("oil_sd must be positive")
    return log_points(shock_pct) / oil_sd


def scale_irf(irf: pd.DataFrame, shock_pct: float, oil_sd: float | None = None) -> pd.DataFrame:
    """Scale a long-format IRF table (artifact schema) to a ``shock_pct`` % oil shock.

    ``oil_sd`` defaults to the ``shock_size_logpts`` stored in the table. The linear model means a
    +30% and a -30% shock give near mirror-image answers (not exactly: log points are asymmetric).
    """
    if oil_sd is None:
        oil_sd = float(irf["shock_size_logpts"].iloc[0])
    f = scale_factor(shock_pct, oil_sd)
    out = irf.copy()
    for c in BAND_COLS:
        out[c] = out[c] * f
    if shock_pct < 0:  # scaling by a negative number swaps the band ends
        out[["lo68", "hi68"]] = out[["hi68", "lo68"]].to_numpy()
        out[["lo90", "hi90"]] = out[["hi90", "lo90"]].to_numpy()
    out["shock_pct"] = shock_pct
    out["scale"] = f
    return out


def yoy_inflation_path(irf_scaled: pd.DataFrame, response: str = "cpi") -> pd.DataFrame:
    """Turn the m-o-m CPI response into the y-o-y inflation path (sum of the last 12 monthly responses).

    Computed from the cumulative response as ``cum(h) - cum(h-12)``. Bands for h >= 12 combine the
    cumulative band at h with the point estimate at h-12 (an approximation; exact for h < 12).
    """
    cum = (
        irf_scaled[(irf_scaled["response"] == response) & (irf_scaled["cumulative"])]
        .sort_values("horizon")
        .set_index("horizon")
    )
    out = pd.DataFrame(index=cum.index)
    lag = cum["point"].shift(12).fillna(0.0)
    for c in BAND_COLS:
        out[c] = cum[c] - lag
    return out.reset_index()


def summarise_response(
    irf_scaled: pd.DataFrame, response: str, cumulative: bool = True
) -> dict[str, Any]:
    """Peak effect, month of peak, effect at 12 and 24 and the 90% band, for one variable.

    "Peak" is the largest absolute effect of the point estimate; months are horizons since the shock
    (0 = impact month).
    """
    d = (
        irf_scaled[(irf_scaled["response"] == response) & (irf_scaled["cumulative"] == cumulative)]
        .sort_values("horizon")
        .set_index("horizon")
    )
    if d.empty:
        raise KeyError(f"no IRF rows for {response} (cumulative={cumulative})")
    pk = int(d["point"].abs().idxmax())
    last = int(d.index.max())

    def at(h: int) -> float:
        return float(d["point"].get(min(h, last), np.nan))

    return {
        "response": response,
        "peak": float(d.loc[pk, "point"]),
        "peak_month": pk,
        "at_12": at(12),
        "at_24": at(24),
        "lo90_at_peak": float(d.loc[pk, "lo90"]),
        "hi90_at_peak": float(d.loc[pk, "hi90"]),
        "significant_at_peak": bool(d.loc[pk, "lo90"] > 0 or d.loc[pk, "hi90"] < 0),
    }


def first_year_cpi_bps(irf: pd.DataFrame, shock_pct: float = 10.0) -> float:
    """Cumulative CPI response over the first 12 months to a ``shock_pct`` % rise, in basis points."""
    sc = scale_irf(irf, shock_pct)
    c = sc[(sc["response"] == "cpi") & (sc["cumulative"]) & (sc["horizon"] == 11)]
    return float(c["point"].iloc[0] * 100.0)


# --------------------------------------------------------------------------- conditional forecasts
def _exog_rows(
    fit: varmod.VARFit, steps: int, exog_future: pd.DataFrame | np.ndarray | None
) -> np.ndarray:
    m = len(fit.exog_names)
    if m == 0:
        return np.zeros((steps, 0))
    if exog_future is not None:
        arr = np.asarray(
            exog_future[fit.exog_names] if isinstance(exog_future, pd.DataFrame) else exog_future,
            dtype=float,
        )
        if arr.shape != (steps, m):
            raise ValueError(f"exog_future must have shape {(steps, m)}, got {arr.shape}")
        return arr
    default = fit.extras.get("exog_default", {})
    row = np.array([float(default.get(n, 0.0)) for n in fit.exog_names])
    return np.tile(row, (steps, 1))


def conditional_moments(
    res: Any,
    steps: int,
    oil_path: np.ndarray | None = None,
    exog_future: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, varmod.VARFit]:
    """Forecast means and the pieces needed for their variance.

    With ``oil_path`` (100 x dlog per step) the oil variable is fixed to it: the oil shock is solved
    from the path and transmitted to the other variables through the Cholesky factor, and only the
    remaining shocks stay random. With ``oil_path=None`` the VAR forecasts oil itself.

    Returns ``(mean (steps,k), L (steps,k,steps*nfree), fit)``; see ``error_loadings``.
    """
    fit = varmod.fit_from_results(res)
    p, k = fit.p, fit.k
    P = varmod.cholesky_impact(fit.sigma_u)
    hist = [fit.endog[-i] for i in range(1, p + 1)]  # most recent first
    xr = _exog_rows(fit, steps, exog_future)
    mean = np.zeros((steps, k))
    for s in range(steps):
        mu = fit.intercept.copy()
        for i in range(1, p + 1):
            mu = mu + fit.coefs[i - 1] @ hist[i - 1]
        if xr.shape[1]:
            mu = mu + fit.exog_coefs @ xr[s]
        if oil_path is not None:
            e1 = (float(oil_path[s]) - mu[0]) / P[0, 0]
            mu = mu + P[:, 0] * e1
        mean[s] = mu
        hist = [mu] + hist[:-1]
    phi = varmod.ma_coefs(fit.coefs, steps - 1)
    return mean, error_loadings(phi, P, steps, oil_fixed=oil_path is not None), fit


def error_loadings(phi: np.ndarray, P: np.ndarray, steps: int, oil_fixed: bool) -> np.ndarray:
    """Linear map from the free structural shocks to the forecast errors of every variable.

    Returns ``L`` of shape ``(steps, k, steps * nfree)``: the error of variable ``j`` at step ``s+1`` is
    ``L[s, j] @ f`` where ``f`` stacks the free shocks of every step. With the oil path fixed, the oil
    shock is not free: each period it takes exactly the value that keeps oil on the path, so it offsets
    whatever the earlier free shocks did to oil through the lags (zero if oil does not depend on them).
    """
    k = P.shape[0]
    nf = k - 1 if oil_fixed else k
    F = steps * nf
    U = np.zeros((steps, k, F))
    for r in range(steps):
        if oil_fixed:
            U[r][:, r * nf : (r + 1) * nf] = P[:, 1:]
            past = np.zeros(F)
            for m in range(1, r + 1):
                past += phi[m][0] @ U[r - m]
            U[r] += np.outer(P[:, 0], -past / P[0, 0])
        else:
            U[r][:, r * nf : (r + 1) * nf] = P
    L = np.zeros((steps, k, F))
    for s_ in range(steps):
        for r in range(s_ + 1):
            L[s_] += phi[s_ - r] @ U[r]
    return L


def functional_variance(L: np.ndarray, j: int, weights: np.ndarray) -> float:
    """Variance of ``sum_s w[s] * (forecast error of variable j at step s+1)``.

    ``weights`` has one entry per forecast step (index 0 = step 1) and ``L`` comes from
    ``conditional_moments`` (use ``L[:len(weights)]`` for a shorter window). Used for y-o-y inflation
    (sum of the last 12 monthly changes) and cumulative exchange-rate change.
    """
    g = np.tensordot(np.asarray(weights, float), L[: len(weights), j, :], axes=(0, 0))
    return float(g @ g)


def conditional_forecast(
    res: Any,
    oil_path: np.ndarray | None,
    steps: int,
    exog_future: pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """Conditional forecast for every VAR variable given an oil path (plan Section 6).

    Output: one row per (variable, step) with mean, se and normal quantiles q05..q95.
    """
    mean, L, fit = conditional_moments(res, steps, oil_path, exog_future)
    qs = {"q05": 0.05, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q95": 0.95}
    rows = []
    for j, name in enumerate(fit.names):
        for s in range(steps):
            se = float(np.sqrt(np.sum(L[s, j] ** 2)))
            row = {"variable": name, "step": s + 1, "mean": float(mean[s, j]), "se": se}
            for col, q in qs.items():
                row[col] = float(mean[s, j] + stats.norm.ppf(q) * se)
            rows.append(row)
    return pd.DataFrame(rows)


def oil_path_from_total(total_pct: float, steps: int, shape: str = "linear") -> np.ndarray:
    """Monthly 100 x dlog oil path whose cumulative change over ``steps`` equals ``total_pct`` %."""
    lp = log_points(total_pct)
    return np.full(steps, lp / steps)


def oil_path_from_quarters(quarter_pct: list[float], months_per_step: int = 3) -> np.ndarray:
    """Oil path from per-quarter % changes, each spread evenly over its months."""
    out: list[float] = []
    for pct in quarter_pct:
        out.extend([log_points(pct) / months_per_step] * months_per_step)
    return np.asarray(out)
