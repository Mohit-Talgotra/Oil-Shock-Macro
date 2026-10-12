"""Shared test helpers (small, deterministic, never touch the internet)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


@lru_cache(maxsize=1)
def synthetic_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly and quarterly panels from the SYNTHETIC generator (needs statsmodels for STL)."""
    from oilshock.data import synthetic

    return synthetic.make_panels()


def simulate_var1(T: int = 4000, seed: int = 0):
    """Data from a known recursive VAR(1): returns (DataFrame, A, c, Sigma)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.0, 0.0], [0.2, 0.4, 0.0], [0.1, 0.1, 0.3]])
    c = np.array([0.1, 0.2, 0.3])
    L = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.2, 0.1, 1.0]])
    Sigma = L @ L.T
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = c + A @ y[t - 1] + L @ rng.standard_normal(3)
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    return pd.DataFrame(y, index=idx, columns=["oil", "x", "z"]), A, c, Sigma
