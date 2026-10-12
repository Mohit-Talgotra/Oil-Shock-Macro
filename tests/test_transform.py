import numpy as np
import pandas as pd
import pytest

from oilshock.data import transform


def _monthly(values, start="2020-01-31"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="ME"), dtype=float
    )


def test_log_diff_known_answer():
    s = _monthly([100.0, 110.0, 99.0])
    out = transform.log_diff(s)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(100 * np.log(1.1))
    assert out.iloc[2] == pytest.approx(100 * np.log(0.9))


def test_log_diff_rejects_non_positive():
    with pytest.raises(ValueError):
        transform.log_diff(_monthly([1.0, 0.0, 2.0]))


def test_yoy_and_pct_change():
    s = _monthly(np.arange(100.0, 118.0))
    y = transform.yoy(s, 12)
    assert y.iloc[:12].isna().all()
    assert y.iloc[12] == pytest.approx(12.0)  # 112 / 100 - 1
    assert transform.pct_change(s, 1).iloc[1] == pytest.approx(1.0)


def test_quarterly_conversion_mean_last_sum():
    s = _monthly([1, 2, 3, 4, 5, 6, 7], "2021-01-31")  # 7 months: the last quarter is incomplete
    q = transform.to_quarterly(s, "mean")
    assert list(q.iloc[:2]) == [2.0, 5.0]
    assert np.isnan(q.iloc[2])
    assert list(transform.to_quarterly(s, "last").iloc[:2]) == [3.0, 6.0]
    assert list(transform.to_quarterly(s, "sum").iloc[:2]) == [6.0, 15.0]
    assert q.index[0] == pd.Timestamp("2021-03-31")
    with pytest.raises(ValueError):
        transform.to_quarterly(s, "median")


def test_step_series_expands_change_dates():
    ev = pd.Series([6.0, 8.0], index=pd.to_datetime(["2008-01-15", "2008-04-10"]), name="repo")
    idx = pd.date_range("2008-01-31", periods=6, freq="ME")
    out = transform.step_series(ev, idx)
    assert list(out) == [6.0, 6.0, 6.0, 8.0, 8.0, 8.0]


def test_period_end_mapping():
    idx = transform.to_period_end(pd.to_datetime(["2020-01-01", "2020-02-15"]), "M")
    assert list(idx) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    q = transform.to_period_end(pd.to_datetime(["2020-04-01"]), "Q")
    assert q[0] == pd.Timestamp("2020-06-30")


def test_dummies():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    d = transform.window_dummy(idx, "2020-04-30", "2020-09-30")
    assert d.sum() == 6 and d["2020-04-30"] == 1 and d["2020-10-31"] == 0
    open_end = transform.window_dummy(idx, "2020-10-31", None)
    assert open_end.sum() == 3
    qidx = pd.date_range("2019-12-31", periods=6, freq="QE")
    qd = transform.quarter_dummy(qidx, ["2020Q2", "2020Q3"])
    assert list(qd) == [0, 0, 1, 1, 0, 0]


def test_cumulate_growth_rebuilds_level():
    g = pd.Series([1.0, 2.0, -1.0])
    lvl = transform.cumulate_growth(g, 100.0)
    assert lvl.iloc[0] == pytest.approx(101.0)
    assert lvl.iloc[2] == pytest.approx(100 * 1.01 * 1.02 * 0.99)


def test_apply_lags_uses_latest_known_value_as_lag_one():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0]}, index=pd.date_range("2020-01-31", periods=4, freq="ME")
    )
    out = transform.apply_lags(df, {"a": 2})
    assert out["a_l1"].iloc[-1] == 4.0 and out["a_l2"].iloc[-1] == 3.0


def test_gdp_level_from_yoy_recovers_growth():
    g = pd.Series(np.full(12, 8.0), index=pd.date_range("2020-03-31", periods=12, freq="QE"))
    lvl = transform.gdp_level_from_yoy(g)
    assert (lvl.iloc[8] - lvl.iloc[4]) == pytest.approx(np.log(1.08))


def test_seasonal_adjust_removes_a_pure_seasonal_pattern():
    pytest.importorskip("statsmodels")
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    seas = 1 + 0.1 * np.sin(2 * np.pi * idx.month / 12)
    s = pd.Series(100 * seas * np.exp(0.002 * np.arange(120)), index=idx)
    adj = transform.seasonal_adjust(s)
    assert adj.pct_change().std() < s.pct_change().std() / 3
