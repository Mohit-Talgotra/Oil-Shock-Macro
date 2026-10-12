import numpy as np
import pandas as pd
import pytest

from oilshock import config
from oilshock.forecast import evaluate


def _frame(actual, **q):
    n = len(actual)
    d = {c: np.zeros(n) for c in config.QCOLS}
    d.update({k: np.asarray(v, float) for k, v in q.items()})
    d["actual"] = np.asarray(actual, float)
    return pd.DataFrame(d)


def test_pinball_hand_calc():
    df = _frame([1.0, -1.0], q10=[0.0, 0.0])
    # tau .1: e=1 -> .1 ; e=-1 -> (.1-1)*-1 = .9 ; mean .5
    assert evaluate.pinball_loss(df, "q10") == pytest.approx(0.5)
    assert evaluate.pinball_loss(df, 0.10) == pytest.approx(0.5)


def test_pinball_median_is_half_mae():
    df = _frame([2.0, -2.0, 1.0], q50=[0, 0, 0])
    assert evaluate.pinball_loss(df, "q50") == pytest.approx(0.5 * np.mean([2, 2, 1]))


def test_point_and_interval_metrics():
    df = _frame([0.5, 3.0], q05=[-2, -2], q10=[-1, -1], q50=[0, 0], q90=[1, 1], q95=[2, 2])
    pm = evaluate.point_metrics(df)
    assert pm["rmse"] == pytest.approx(np.sqrt((0.25 + 9) / 2))
    assert pm["mae"] == pytest.approx(1.75)
    im = evaluate.interval_metrics(df)
    assert im["coverage80"] == 0.5 and im["coverage95"] == 0.5
    assert im["mean_width80"] == pytest.approx(2.0)


def test_diebold_mariano_detects_worse_model_and_is_nan_when_identical():
    rng = np.random.default_rng(0)
    e_bad = rng.normal(0, 2, 200)
    e_good = rng.normal(0, 1, 200)
    stat, p = evaluate.diebold_mariano(e_bad, e_good)
    assert stat > 0 and p < 0.01
    stat2, p2 = evaluate.diebold_mariano(e_good, e_bad)
    assert stat2 < 0
    assert np.isnan(evaluate.diebold_mariano(e_good, e_good)[1])


def test_episode_mask_flags_known_dates():
    m = evaluate.episode_mask(pd.DatetimeIndex(["2005-06-30", "2022-04-30", "2026-04-30"]))
    assert m.tolist() == [False, True, True]
