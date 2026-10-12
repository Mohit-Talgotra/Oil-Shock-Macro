import numpy as np
import pandas as pd
import pytest

from helpers import simulate_var1
from oilshock.models import var


def test_estimation_recovers_a_known_var1():
    df, A, c, Sigma = simulate_var1()
    fit = var.estimate_var(df, 1)
    assert np.abs(fit.coefs[0] - A).max() < 0.04
    assert np.abs(fit.intercept - c).max() < 0.1
    assert np.abs(fit.sigma_u - Sigma).max() < 0.12


def test_orthogonalised_irf_matches_the_analytic_answer():
    df, A, _, Sigma = simulate_var1()
    fit = var.estimate_var(df, 1)
    irf, sigma_oil = var.oil_irf(fit, 8)
    P = np.linalg.cholesky(Sigma)
    for h in (0, 1, 4, 8):
        truth = np.linalg.matrix_power(A, h) @ P[:, 0]
        assert np.abs(irf[h] - truth).max() < 0.05
    assert sigma_oil == pytest.approx(1.0, abs=0.05)  # first Cholesky element = oil s.d.


def test_bootstrap_bands_contain_the_truth_and_are_reproducible():
    df, A, _, Sigma = simulate_var1(T=600)
    fit = var.estimate_var(df, 1)
    a = var.irf_bands(fit, 6, 200, seed=11)
    b = var.irf_bands(fit, 6, 200, seed=11)
    pd.testing.assert_frame_equal(a, b)
    P = np.linalg.cholesky(Sigma)
    row = a[(a["response"] == "z") & (~a["cumulative"]) & (a["horizon"] == 2)].iloc[0]
    truth = (np.linalg.matrix_power(A, 2) @ P[:, 0])[2]
    assert row["lo90"] <= truth <= row["hi90"]
    assert row["lo68"] >= row["lo90"] and row["hi68"] <= row["hi90"]
    assert set(a.columns) == {
        "response",
        "horizon",
        "point",
        "lo68",
        "hi68",
        "lo90",
        "hi90",
        "cumulative",
        "shock_size_logpts",
    }
    cum = a[(a["response"] == "x") & a["cumulative"]].sort_values("horizon")["point"].to_numpy()
    plain = a[(a["response"] == "x") & ~a["cumulative"]].sort_values("horizon")["point"].to_numpy()
    assert np.allclose(cum, np.cumsum(plain))


def test_fevd_shares_are_between_zero_and_one_and_oil_explains_itself():
    df, *_ = simulate_var1(T=800)
    f = var.fevd_table(var.estimate_var(df, 1), 12)
    assert f["share_oil"].between(0, 1).all()
    own = f[f["variable"] == "oil"].set_index("horizon")["share_oil"]
    assert own.loc[0] == pytest.approx(1.0)
    assert own.min() > 0.98  # true cross-lags into oil are zero; estimation noise only


def test_block_exogeneity_zeroes_cross_lags_in_the_oil_equation():
    df, *_ = simulate_var1(T=500)
    fit = var.estimate_var(df, 2, block_exog=True)
    assert np.allclose(fit.coefs[:, 0, 1:], 0.0)
    assert fit.coefs[0, 0, 0] != 0.0


def test_constant_exog_columns_are_dropped():
    ex = pd.DataFrame({"a": [0.0] * 10, "b": np.arange(10.0)})
    assert list(var.usable_exog(ex).columns) == ["b"]
    assert var.usable_exog(ex[["a"]]) is None


def test_numpy_bic_selection_finds_the_true_order():
    df, *_ = simulate_var1(T=1500)
    assert var.select_lags_bic_np(df, None, 6) == 1


def test_numpy_estimator_agrees_with_statsmodels():
    pytest.importorskip("statsmodels")
    df, *_ = simulate_var1(T=400)
    ours = var.estimate_var(df, 2)
    res = var.fit_var(df, 2)
    theirs = var.fit_from_results(res)
    assert np.allclose(ours.coefs, theirs.coefs, atol=1e-8)
    assert np.allclose(ours.sigma_u, theirs.sigma_u, atol=1e-8)
    sm_irf = res.irf(6).orth_irfs  # (h+1, k, k)
    ours_irf = var.orth_irf(ours, horizon=6)
    assert np.allclose(sm_irf, ours_irf, atol=1e-8)
    d = var.diagnostics(res, 12, var.fevd_table(ours, 24))
    assert d["stable"] and 0 <= d["whiteness_pvalue"] <= 1


def test_rolling_irfs_returns_one_row_per_window():
    df, *_ = simulate_var1(T=300)
    out = var.rolling_irfs(
        df, window=120, step=60, lags=1, response="x", horizon_months=6, reps=30, seed=1
    )
    assert (
        len(out) == 4
        and (out["lo90"] <= out["point"]).all()
        and (out["point"] <= out["hi90"]).all()
    )
