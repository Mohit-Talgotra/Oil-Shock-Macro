"""Forecaster interface and target definitions (plan Sections 6 and 8).

Targets
-------
Each target has a *model series* ``z`` (a column of the panel) and a ``kind``:

* ``level``: the target at horizon h is ``z[t+h]`` (CPI y-o-y %, GDP growth y-o-y %);
* ``cum``:   the target at horizon h is ``sum_{j=1..h} z[t+j]`` (USD/INR % change from the origin,
  where ``z`` is the monthly 100 x dlog of the rate).

Every ``Forecaster.predict`` returns quantiles of the *target*, never of ``z``.

Oil
---
``X_future["oil"]`` is the assumed path of oil (100 x dlog per period) over the forecast horizon.
``None`` means the honest "flat" assumption (oil stays at its origin value). The same fitted object
serves the blind (flat) and oracle (actual path) information sets and the app's scenario paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from oilshock import config

QUANTILES = config.QUANTILES
QCOLS = config.QCOLS


@dataclass(frozen=True)
class TargetSpec:
    name: str
    label: str
    freq: str  # "M" or "Q"
    z: str
    kind: str  # "level" | "cum"
    horizons: int
    exog: tuple[str, ...]
    exog_lags: int
    units: str

    @property
    def alias(self) -> str:
        return "ME" if self.freq == "M" else "QE"

    @property
    def periods_per_year(self) -> int:
        return 12 if self.freq == "M" else 4


def get_target(name: str) -> TargetSpec:
    t = config.models_cfg()["forecast"]["targets"][name]
    return TargetSpec(
        name=name,
        label=t["label"],
        freq=t["freq"],
        z=t["z"],
        kind=t["kind"],
        horizons=int(t["horizons"]),
        exog=tuple(t["exog"]),
        exog_lags=int(t["exog_lags"]),
        units=t.get("units", ""),
    )


def target_names() -> list[str]:
    return list(config.models_cfg()["forecast"]["targets"])


def panel_for(spec: TargetSpec, monthly: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    return monthly if spec.freq == "M" else quarterly


def future_dates(last: pd.Timestamp, h: int, alias: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(last), periods=h + 1, freq=alias)[1:]


def target_values(z: pd.Series, h: int, kind: str) -> pd.Series:
    """Realised target at horizon ``h`` for every origin (NaN where the future is not yet observed)."""
    if kind == "level":
        return z.shift(-h)
    if kind == "cum":
        return z.rolling(h).sum().shift(-h)
    raise ValueError(kind)


def pad_path(path: np.ndarray | pd.Series | None, h: int) -> np.ndarray:
    """Oil path of exactly ``h`` entries: truncated, or padded with zeros (flat) when shorter/None."""
    if path is None:
        return np.zeros(h)
    arr = np.asarray(path, dtype=float)[:h]
    return np.concatenate([arr, np.zeros(h - len(arr))]) if len(arr) < h else arr


def normal_quantiles(mean: np.ndarray, se: np.ndarray, qs=QUANTILES) -> np.ndarray:
    """(h, len(qs)) normal quantiles."""
    z = stats.norm.ppf(np.asarray(qs))
    return np.asarray(mean)[:, None] + np.asarray(se)[:, None] * z[None, :]


def quantile_frame(values: np.ndarray, h: int) -> pd.DataFrame:
    """Wrap a (h, 5) array into the forecast quantile frame indexed by horizon 1..h, sorted per row."""
    arr = np.sort(np.asarray(values, dtype=float), axis=1)
    return pd.DataFrame(arr, index=pd.RangeIndex(1, h + 1, name="horizon"), columns=QCOLS)


class Forecaster(ABC):
    """Common interface so the backtest engine treats every model the same."""

    name: str = "base"
    supports_covariates: bool = False

    def __init__(self, spec: TargetSpec):
        self.spec = spec
        self.train_end: pd.Timestamp | None = None
        self.n_train: int = 0

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> Forecaster:
        """Fit on the model series ``y`` and the panel history ``X``, both ending at the origin."""
        y = y.dropna()
        if len(y) == 0:
            raise ValueError("empty training series")
        self.train_end = y.index.max()
        if X is not None and X.index.max() > self.train_end:
            raise AssertionError(
                f"leakage: panel history ends {X.index.max().date()} after the origin {self.train_end.date()}"
            )
        self.n_train = len(y)
        return self._fit(y, X)

    @abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> Forecaster: ...

    @abstractmethod
    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
    ) -> pd.DataFrame:
        """Quantiles of the target for horizons 1..h: columns q05..q95, index horizon."""

    @staticmethod
    def oil_from(X_future: pd.DataFrame | None, h: int) -> np.ndarray:
        if X_future is None or "oil" not in X_future:
            return np.zeros(h)
        return pad_path(X_future["oil"].to_numpy(), h)


def hold_value(rule: str, hist: pd.Series) -> float:
    """Future value of a non-oil exogenous input under the rule in ``models.yaml`` (zero | last)."""
    if rule == "zero":
        return 0.0
    if rule == "last":
        v = hist.dropna()
        return float(v.iloc[-1]) if len(v) else 0.0
    raise ValueError(rule)
