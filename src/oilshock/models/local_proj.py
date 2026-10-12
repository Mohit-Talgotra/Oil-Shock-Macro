"""Jordà local projections: robustness check against the VAR and the asymmetry extension (plan Section 7).

    y_{t+h} - y_{t-1} = a_h + b_h * eps_oil_t + sum_j g_{h,j}' x_{t-j} + u_{t+h}

One OLS per horizon with Newey-West (Bartlett) standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.6448536269514722
Z68 = 0.9944578832097535


def _ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West standard errors (Bartlett kernel, small-sample scaled)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lg in range(1, lags + 1):
        w = 1.0 - lg / (lags + 1.0)
        G = Xu[lg:].T @ Xu[:-lg]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def _cumulative_change(y: pd.Series, h: int, differenced: bool) -> pd.Series:
    """y_{t+h} - y_{t-1}: forward sum of changes if ``y`` is already a change, else a level difference."""
    if differenced:
        return y.rolling(h + 1).sum().shift(-h)
    return y.shift(-h) - y.shift(1)


def build_lp_inputs(
    endog: pd.DataFrame, exog: pd.DataFrame | None, lags: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Oil innovation (residual of the oil equation) and lag controls for local projections.

    Controls are lags 1..``lags`` of every endogenous variable plus the exogenous regressors; the
    shock is the residual of oil on exactly those controls, i.e. the first Cholesky shock.
    """
    cols = {}
    for c in endog.columns:
        for lg in range(1, lags + 1):
            cols[f"{c}_l{lg}"] = endog[c].shift(lg)
    ctrl = pd.DataFrame(cols, index=endog.index)
    if exog is not None:
        ctrl = pd.concat([ctrl, exog], axis=1)
    ok = ctrl.dropna().index
    X = np.column_stack([np.ones(len(ok)), ctrl.loc[ok].to_numpy()])
    y = endog.loc[ok, endog.columns[0]].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    shock = pd.Series(y - X @ beta, index=ok, name="oil_shock")
    return shock, ctrl.loc[ok]


def lp_irf(
    df: pd.DataFrame,
    shock: pd.Series,
    response: str,
    horizons: int = 24,
    controls: pd.DataFrame | None = None,
    nw_lags: int = 6,
    differenced: bool = True,
) -> pd.DataFrame:
    """Cumulative local-projection response of ``response`` to a one-s.d. oil shock.

    Output follows the IRF artifact schema (cumulative rows only); ``point`` is per one-s.d. shock so it
    is directly comparable with the VAR's cumulative response.
    """
    sd = float(shock.std(ddof=1))
    rows = []
    for h in range(horizons + 1):
        yh = _cumulative_change(df[response], h, differenced)
        parts = [shock.rename("shock")]
        if controls is not None:
            parts.append(controls)
        Xdf = pd.concat(parts, axis=1)
        joined = pd.concat([yh.rename("y"), Xdf], axis=1).dropna()
        if len(joined) < Xdf.shape[1] + 10:
            continue
        y = joined["y"].to_numpy()
        X = np.column_stack([np.ones(len(joined)), joined[Xdf.columns].to_numpy()])
        beta, se = _ols_hac(y, X, max(nw_lags, h))
        b, s = beta[1] * sd, se[1] * sd
        rows.append(
            {
                "response": response,
                "horizon": h,
                "point": b,
                "lo68": b - Z68 * s,
                "hi68": b + Z68 * s,
                "lo90": b - Z90 * s,
                "hi90": b + Z90 * s,
                "cumulative": True,
                "shock_size_logpts": sd,
            }
        )
    return pd.DataFrame(rows)


def lp_all(
    endog: pd.DataFrame,
    exog: pd.DataFrame | None,
    differenced: list[str],
    horizons: int,
    lags: int,
    nw_lags: int,
) -> pd.DataFrame:
    """Local projections for every variable (artifact ``irf_A_lp_<window>.parquet``)."""
    shock, ctrl = build_lp_inputs(endog, exog, lags)
    frames = [
        lp_irf(endog, shock, v, horizons, ctrl, nw_lags, v in differenced) for v in endog.columns
    ]
    return pd.concat(frames, ignore_index=True)


def asymmetric_lp(
    endog: pd.DataFrame,
    exog: pd.DataFrame | None,
    response: str,
    differenced: bool,
    horizons: int = 24,
    lags: int = 4,
    nw_lags: int = 6,
    move_pct: float = 10.0,
) -> pd.DataFrame:
    """Stretch: separate responses to oil *increases* and *decreases* (Mork 1989; Hamilton 2003).

    Regresses the cumulative change of ``response`` on the positive and negative parts of the oil change
    plus lag controls. Reported per ``move_pct`` % rise (``increase``) or fall (``decrease``), so
    symmetric responses would show identical numbers with the sign of the move already applied.
    """
    oil = endog[endog.columns[0]]
    _, ctrl = build_lp_inputs(endog, exog, lags)
    pos, neg = oil.clip(lower=0).rename("pos"), oil.clip(upper=0).rename("neg")
    up_lp = 100.0 * np.log1p(move_pct / 100.0)
    dn_lp = 100.0 * np.log1p(-move_pct / 100.0)
    rows = []
    for h in range(horizons + 1):
        yh = _cumulative_change(endog[response], h, differenced)
        X = pd.concat([pos, neg, ctrl], axis=1)
        joined = pd.concat([yh.rename("y"), X], axis=1).dropna()
        if len(joined) < X.shape[1] + 10:
            continue
        Xm = np.column_stack([np.ones(len(joined)), joined[X.columns].to_numpy()])
        beta, se = _ols_hac(joined["y"].to_numpy(), Xm, max(nw_lags, h))
        for name, j, move in (("increase", 1, up_lp), ("decrease", 2, dn_lp)):
            b, s = beta[j] * move, se[j] * abs(move)
            rows.append(
                {
                    "direction": name,
                    "response": response,
                    "horizon": h,
                    "point": b,
                    "lo68": b - Z68 * s,
                    "hi68": b + Z68 * s,
                    "lo90": b - Z90 * s,
                    "hi90": b + Z90 * s,
                    "move_pct": move_pct if name == "increase" else -move_pct,
                }
            )
    return pd.DataFrame(rows)
