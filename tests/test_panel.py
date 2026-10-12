import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from helpers import synthetic_panels
from oilshock.data import panel as P


def test_panels_are_well_formed():
    m, q = synthetic_panels()
    for df in (m, q):
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates
    assert list(m.columns) == P.MONTHLY_COLUMNS
    assert list(q.columns) == P.QUARTERLY_COLUMNS
    assert m.index[0] == pd.Timestamp("2004-01-31")
    assert q.index[0] == pd.Timestamp("2004-06-30")


def test_no_gaps_inside_the_sample():
    m, q = synthetic_panels()
    assert (m.index == pd.date_range(m.index[0], m.index[-1], freq="ME")).all()
    assert (q.index == pd.date_range(q.index[0], q.index[-1], freq="QE")).all()


def test_core_var_variables_are_complete():
    m, q = synthetic_panels()
    core = m[["oil", "ip", "cpi_mom_sa", "d_repo", "inr", "d_fedfunds", "vix"]]
    assert core.notna().all().all()
    assert q[["oil", "gdp_growth", "cab_gdp", "cpi_yoy", "repo", "inr"]].notna().all().all()


def test_transformations_are_consistent():
    m, _ = synthetic_panels()
    assert np.allclose(m["oil"].iloc[1:], 100 * np.log(m["brent_usd"]).diff().iloc[1:])
    assert np.allclose(m["d_repo"].iloc[1:], m["repo"].diff().iloc[1:])
    assert set(np.unique(m["covid"])) <= {0.0, 1.0}
    assert m.loc["2020-04-30":"2020-09-30", "covid"].eq(1).all()
    assert m.loc["2016-06-30":, "inflation_targeting"].eq(1).all()


def test_var_frames_put_oil_first_and_respect_windows():
    m, q = synthetic_panels()
    endog, exog = P.monthly_var_frame(m, "pre2014")
    assert list(endog.columns)[0] == "oil"
    assert endog.index.max() <= pd.Timestamp("2014-06-30")
    qe, qx = P.quarterly_var_frame(q, "full")
    assert list(qe.columns) == ["oil", "gdp_growth", "cab_gdp", "cpi", "repo", "inr"]
    assert list(qx.columns) == ["covid_q"]


def test_data_checks_pass_clean_data_and_catch_parsing_errors():
    m, q = synthetic_panels()
    P.run_data_checks(m, q, None, strict=True)
    bad = m.copy()
    bad.loc[bad.index[50], "inr"] = 80.0  # an 80% single-month rupee move
    with pytest.raises(P.DataCheckError):
        P.run_data_checks(bad, q, None, strict=True)
