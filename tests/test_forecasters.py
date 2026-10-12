import numpy as np
import pytest

from helpers import synthetic_panels
from oilshock import config
from oilshock.forecast.base import get_target, target_names
from oilshock.forecast.baselines import NaiveForecaster, SeasonalNaiveForecaster


def _check(fc, h):
    assert list(fc.columns) == config.QCOLS
    assert len(fc) == h
    assert np.isfinite(fc.to_numpy()).all()
    assert (np.diff(fc.to_numpy(), axis=1) >= -1e-9).all()


@pytest.fixture(scope="module")
def panels():
    pytest.importorskip("statsmodels")
    return synthetic_panels()


def test_naive_and_snaive_smoke(panels):
    monthly, quarterly = panels
    for t in target_names():
        spec = get_target(t)
        panel = monthly if spec.freq == "M" else quarterly
        y = panel[spec.z].dropna()
        for cls in (NaiveForecaster, SeasonalNaiveForecaster):
            _check(cls(spec).fit(y, panel.loc[: y.index.max()]).predict(6), 6)


def test_naive_level_median_is_last_value(panels):
    monthly, _ = panels
    spec = get_target("cpi")
    y = monthly[spec.z].dropna()
    fc = NaiveForecaster(spec).fit(y, monthly.loc[: y.index.max()]).predict(3)
    assert fc["q50"].iloc[0] == pytest.approx(y.iloc[-1], abs=1.0)


@pytest.mark.parametrize("name", ["ar", "arimax", "var", "xgb"])
def test_model_smoke(panels, name):
    pytest.importorskip("statsmodels")
    if name == "xgb":
        pytest.importorskip("xgboost")
    from oilshock.forecast.backtest import make_forecaster

    monthly, _ = panels
    spec = get_target("cpi")
    y = monthly[spec.z].dropna().iloc[:-24]
    f = make_forecaster(name, spec)
    f.fit(y, monthly.loc[: y.index.max()])
    _check(f.predict(6), 6)
