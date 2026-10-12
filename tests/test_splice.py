import numpy as np
import pandas as pd
import pytest

from oilshock.data import splice, transform


def _index(start, n, rate_pct, base):
    idx = pd.date_range(start, periods=n, freq="ME")
    return pd.Series(base * (1 + rate_pct / 100) ** np.arange(n), index=idx)


def test_synthetic_index_splices_back_to_the_known_growth_path():
    # true path: 0.5% a month throughout; two index series with different base years
    old = _index("2010-01-31", 120, 0.5, 100.0)
    new = _index("2013-01-31", 90, 0.5, 250.0)  # different base level, same true growth
    yoy_old, yoy_new = transform.yoy(old, 12), transform.yoy(new, 12)
    spliced = splice.splice_growth(yoy_old, yoy_new, "2014-06-30")
    truth = 100 * (1.005**12 - 1)
    valid = spliced.dropna()
    assert np.allclose(valid, truth)
    assert spliced.loc["2014-05-31"] == pytest.approx(yoy_old.loc["2014-05-31"])
    assert spliced.loc["2014-06-30"] == pytest.approx(yoy_new.loc["2014-06-30"])


def test_splicing_levels_raises():
    old = _index("2010-01-31", 60, 0.5, 100.0)
    new = _index("2013-01-31", 40, 0.5, 250.0)
    with pytest.raises(ValueError):
        splice.splice_growth(old, new, "2014-01-31")
    tagged = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    tagged.attrs["kind"] = "level"
    with pytest.raises(ValueError):
        splice.splice_growth(tagged, tagged, "2020-02-29")


def test_overlap_report_gap_statistics():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    old = pd.Series([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], index=idx)
    new = pd.Series([4.5, 3.5, 4.0, 4.0, 4.0, 4.0], index=idx)
    rep = splice.overlap_report(old, new, "cpi").iloc[0]
    assert rep["n_overlap"] == 6
    assert rep["mean_gap"] == pytest.approx(0.0)
    assert rep["mean_abs_gap"] == pytest.approx(1 / 6)
    assert rep["max_abs_gap"] == pytest.approx(0.5)


def test_overlap_report_written_and_flagged(tmp_path):
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    a = pd.Series([2.0] * 4, index=idx)
    b = pd.Series([3.0] * 4, index=idx)
    df = splice.write_overlap_report(
        [splice.overlap_report(a, b, "x")], tmp_path / "splice_check.csv", 0.5
    )
    assert (tmp_path / "splice_check.csv").exists()
    assert bool(df["mention_on_about_page"].iloc[0])
