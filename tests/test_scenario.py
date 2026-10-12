import numpy as np
import pandas as pd
import pytest

from helpers import simulate_var1
from oilshock.models import local_proj, scenario, var


def _irf():
    df, *_ = simulate_var1(T=500)
    return var.irf_bands(var.estimate_var(df, 1), 12, 60, seed=3), df


def test_scaling_is_linear_in_log_points():
    irf, _ = _irf()
    one = scenario.scale_irf(irf, 10.0)
    double = scenario.scale_irf(irf, 100 * (1.1**2 - 1))  # 21% is twice 10% in log points
    assert np.allclose(double["point"], 2 * one["point"])
    assert np.allclose(double["hi90"], 2 * one["hi90"])


def test_zero_shock_gives_zeros():
    irf, _ = _irf()
    assert scenario.scale_irf(irf, 0.0)[["point", "lo90", "hi90"]].abs().max().max() == 0.0


def test_log_conversion_for_the_slider_extremes():
    assert scenario.log_points(-40) == pytest.approx(100 * np.log(0.6))
    assert scenario.log_points(80) == pytest.approx(100 * np.log(1.8))
    with pytest.raises(ValueError):
        scenario.log_points(-100)


def test_negative_shock_flips_the_sign_and_keeps_bands_ordered():
    irf, _ = _irf()
    neg = scenario.scale_irf(irf, -30.0)
    assert (neg["lo90"] <= neg["hi90"]).all() and (neg["lo68"] <= neg["hi68"]).all()
    pos = scenario.scale_irf(irf, 30.0)
    assert (
        np.sign(neg["point"].iloc[3]) == -np.sign(pos["point"].iloc[3]) or pos["point"].iloc[3] == 0
    )


def test_yoy_path_is_the_sum_of_the_last_twelve_monthly_responses():
    irf, _ = _irf()
    sc = scenario.scale_irf(irf, 30.0)
    # build a fake 24-month CPI response to test the 12-month window explicitly
    h = np.arange(25)
    point = np.where(h < 6, 0.1, 0.0)
    rows = pd.DataFrame(
        {
            "response": "cpi",
            "horizon": h,
            "point": np.cumsum(point),
            "lo68": np.cumsum(point),
            "hi68": np.cumsum(point),
            "lo90": np.cumsum(point),
            "hi90": np.cumsum(point),
            "cumulative": True,
        }
    )
    yoy = scenario.yoy_inflation_path(rows)
    manual = pd.Series(point).rolling(12, min_periods=1).sum()
    assert np.allclose(yoy["point"], manual)
    assert yoy["point"].iloc[20] == pytest.approx(0.0)  # the shock has left the 12-month window
    assert len(sc) > 0


def test_summary_reports_peak_and_timing():
    irf, _ = _irf()
    s = scenario.summarise_response(scenario.scale_irf(irf, 30.0), "z", cumulative=True)
    assert 0 <= s["peak_month"] <= 12 and s["lo90_at_peak"] <= s["peak"] <= s["hi90_at_peak"]


def test_conditional_forecast_pins_the_oil_path_and_transmits_it():
    df, *_ = simulate_var1(T=800)
    fit = var.estimate_var(df, 1)
    path = np.array([0.0, 2.0, 2.0, 0.0, -1.0, 0.0])
    cf = scenario.conditional_forecast(fit, path, 6)
    oil = cf[cf["variable"] == "oil"].sort_values("step")
    assert np.allclose(oil["mean"], path)
    assert np.allclose(oil["se"], 0.0)
    flat = scenario.conditional_forecast(fit, np.zeros(6), 6)
    up = scenario.conditional_forecast(fit, np.full(6, 3.0), 6)
    z_flat = flat[flat["variable"] == "z"]["mean"].to_numpy()
    z_up = up[up["variable"] == "z"]["mean"].to_numpy()
    assert (z_up > z_flat).all()  # z responds positively to oil in the simulated system
    free = scenario.conditional_forecast(fit, None, 6)
    assert free[free["variable"] == "oil"]["se"].iloc[0] > 0


def test_functional_variance_matches_a_direct_calculation_for_one_step():
    df, *_ = simulate_var1(T=500)
    fit = var.estimate_var(df, 1)
    mean, L, _ = scenario.conditional_moments(fit, 3, None)
    v = scenario.functional_variance(L, 2, np.ones(1))
    assert v == pytest.approx(fit.sigma_u[2, 2])


def test_path_helpers():
    p = scenario.oil_path_from_total(50.0, 12)
    assert p.sum() == pytest.approx(scenario.log_points(50.0))
    q = scenario.oil_path_from_quarters([10.0, 0.0, 0.0, -10.0])
    assert len(q) == 12 and q[:3].sum() == pytest.approx(scenario.log_points(10.0))


def test_local_projection_tracks_the_var_response():
    df, *_ = simulate_var1(T=3000)
    shock, ctrl = local_proj.build_lp_inputs(df, None, 1)
    lp = local_proj.lp_irf(df, shock, "z", horizons=4, controls=ctrl, nw_lags=4, differenced=False)
    fit = var.estimate_var(df, 1)
    irf, _ = var.oil_irf(fit, 4)
    # LP of a level variable: y_{t+h} - y_{t-1}; compare to the VAR's response minus the h=-1 level (0)
    assert abs(lp[lp["horizon"] == 2]["point"].iloc[0] - (irf[2, 2] + 0.0)) < 0.15
