import numpy as np
import pandas as pd
import pytest

from helpers import synthetic_panels
from oilshock import config
from oilshock.forecast.base import Forecaster, quantile_frame
from oilshock.stress import run as srun


class Stub(Forecaster):
    """Forecasts the last value; records the oil path it was given."""

    name = "stub"
    supports_covariates = True
    last_oil = None

    def _fit(self, y, X):
        self.last = float(y.iloc[-1])
        return self

    def predict(self, h, X_future=None, quantiles=config.QUANTILES):
        Stub.last_oil = None if X_future is None else X_future["oil"].to_numpy().copy()
        return quantile_frame(np.tile(self.last + np.linspace(-1, 1, 5), (h, 1)), h)


def factory(name, spec):
    return Stub(spec)


@pytest.fixture(scope="module")
def panels():
    pytest.importorskip("statsmodels")
    return synthetic_panels()


def test_oracle_receives_realised_oil_and_blind_receives_none(panels):
    monthly, quarterly = panels
    srun.run_episode(
        "ukraine_2022", ["stub"], "blind", monthly, quarterly, targets=["cpi"], factory=factory
    )
    assert Stub.last_oil is None
    srun.run_episode(
        "ukraine_2022", ["stub"], "oracle", monthly, quarterly, targets=["cpi"], factory=factory
    )
    assert Stub.last_oil is not None and len(Stub.last_oil) > 0
    assert np.isfinite(Stub.last_oil).all()


def test_no_training_data_after_train_end(panels):
    monthly, quarterly = panels
    res = srun.run_episode(
        "gfc_2008", ["stub"], "blind", monthly, quarterly, targets=["cpi"], factory=factory
    )
    assert pd.to_datetime(res["origin"]).max() <= pd.Timestamp("2008-12-31")


def test_frozen_live_forecasts_unchanged_when_data_appended(panels, tmp_path, monkeypatch):
    monthly, quarterly = panels
    monkeypatch.setenv("OILSHOCK_ROOT", str(tmp_path))
    out = tmp_path / "stress"
    srun.run_all(
        monthly,
        quarterly,
        models=["stub"],
        out_dir=out,
        forecast_dir=tmp_path / "none",
        factory=factory,
    )
    frozen = out / srun.FROZEN_FILE
    assert frozen.exists()
    before = config.read_parquet(frozen)
    # append a later month with wild values; frozen forecasts must not move
    extra = monthly.iloc[[-1]].copy()
    extra.index = extra.index + pd.offsets.MonthEnd(1)
    extra[:] = 999.0
    longer = pd.concat([monthly, extra])
    srun.run_all(
        longer,
        quarterly,
        models=["stub"],
        out_dir=out,
        forecast_dir=tmp_path / "none",
        factory=factory,
    )
    after = config.read_parquet(frozen)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
