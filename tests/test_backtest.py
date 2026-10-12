import numpy as np
import pandas as pd
import pytest

from helpers import synthetic_panels
from oilshock import config
from oilshock.forecast import backtest
from oilshock.forecast.base import Forecaster, get_target, quantile_frame
from oilshock.forecast.baselines import NaiveForecaster


class Spy(Forecaster):
    name = "spy"

    def _fit(self, y, X):
        self.seen_y = y.index.max()
        self.seen_x = X.index.max()
        return self

    def predict(self, h, X_future=None, quantiles=config.QUANTILES):
        return quantile_frame(np.tile(np.linspace(-1, 1, 5), (h, 1)), h)


def test_no_training_row_after_the_origin():
    pytest.importorskip("statsmodels")
    monthly, _ = synthetic_panels()
    spec = get_target("cpi")
    spy = Spy(spec)
    origins = backtest.make_origins(monthly, spec, stride=24)
    out = backtest.rolling_origin(spy, monthly, spec, origins[:3], horizons=3)
    assert spy.seen_y <= origins[2] and spy.seen_x <= origins[2]
    assert list(out.columns) == config.FORECAST_COLUMNS


def test_quantiles_are_ordered():
    pytest.importorskip("statsmodels")
    monthly, _ = synthetic_panels()
    spec = get_target("cpi")
    out = backtest.rolling_origin(
        NaiveForecaster(spec),
        monthly,
        spec,
        backtest.make_origins(monthly, spec, 36)[:3],
        horizons=4,
    )
    q = out[config.QCOLS].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all()


def test_leakage_assertion_triggers():
    monthly = pd.DataFrame(
        {"cpi_yoy": np.arange(40.0)}, index=pd.date_range("2010-01-31", periods=40, freq="ME")
    )
    spec = get_target("cpi")
    f = NaiveForecaster(spec)
    with pytest.raises(AssertionError):
        f.fit(monthly["cpi_yoy"].iloc[:30], monthly)  # panel runs past the series end
