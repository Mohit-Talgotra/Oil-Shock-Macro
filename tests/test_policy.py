import numpy as np
import pandas as pd
import pytest

from oilshock.models import policy


def test_taylor_rule_hand_calc():
    # r* 1.5 + pi 6 + 0.5*(6-4) + 0.5*(-1) = 8.0
    assert policy.taylor_rate(6.0, 4.0, -1.0, 1.5, 0.5, 0.5) == pytest.approx(8.0)


def test_taylor_rule_vectorised_and_target_is_neutral():
    pi = pd.Series([4.0, 5.0])
    out = policy.taylor_rate(pi, 4.0, 0.0, 1.0, 0.5, 0.5)
    assert out.tolist() == pytest.approx([5.0, 6.5])


def test_import_bill_hand_calc():
    bbl = policy.tonnes_to_barrels(240.0, 7.33)
    assert bbl == pytest.approx(240e6 * 7.33)
    out = policy.import_bill_change(10.0, bbl, 80.0, 4000.0)
    assert out["usd_bn"] == pytest.approx(10 * 240e6 * 7.33 / 1e9)
    assert out["inr_crore"] == pytest.approx(out["usd_bn"] * 1e9 * 80 / 1e7)
    assert out["pct_gdp"] == pytest.approx(100 * out["usd_bn"] / 4000.0)


def test_import_bill_without_gdp_has_no_pct():
    assert "pct_gdp" not in policy.import_bill_change(1.0, 1e9, 80.0)


def test_excise_cost():
    cost = policy.excise_cost(5.0, 2e11)  # Rs 5/litre on 200 billion litres
    assert cost == pytest.approx(1e12)
    assert policy.rupees_to_crore(cost) == pytest.approx(1e5)


def test_hp_filter_recovers_linear_trend_and_cycle_sums():
    y = 0.02 * np.arange(80) + 1.0
    trend, cycle = policy.hp_filter(y, 1600)
    assert np.allclose(trend, y, atol=1e-6)
    assert np.allclose(trend + cycle, y)


def test_output_gap_sign():
    idx = pd.date_range("2000-03-31", periods=60, freq="QE")
    base = 100 * np.exp(0.01 * np.arange(60))
    base[40] *= 1.05
    gap = policy.output_gap(pd.Series(base, index=idx))
    assert gap.iloc[40] > 2.0
