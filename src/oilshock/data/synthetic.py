"""SYNTHETIC data generator for tests and the demo mode. NOT real data.

The real pipeline needs a FRED key and manually downloaded RBI/MOSPI/PPAC files. This module simulates
series with the same keys, units and frequencies from a known recursive system (oil first), with oil
episodes shaped like the four in ``config/episodes.yaml``. It feeds the *real* panel builders, so the
demo exercises the same code paths. Nothing produced from it is a finding about India.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.data import panel, transform

# monthly drift added to oil (100 x dlog) to imitate the episodes: (start, end, drift)
EPISODE_DRIFT = [
    ("2007-09-30", "2008-06-30", 4.5),
    ("2008-07-31", "2008-12-31", -14.0),
    ("2009-01-31", "2009-06-30", 5.0),
    ("2014-07-31", "2016-01-31", -5.5),
    ("2016-02-29", "2016-06-30", 4.0),
    ("2021-03-31", "2022-01-31", 2.0),
    ("2022-02-28", "2022-06-30", 6.0),
    ("2022-07-31", "2023-01-31", -3.5),
    ("2026-03-31", "2026-05-31", 7.0),
    ("2026-06-30", "2026-07-31", -8.0),
    ("2026-08-31", "2026-09-30", 6.0),
]

# true one-s.d. impact structure used by tests: a 1 s.d. oil shock raises INR depreciation on impact
TRUE_OIL_SD = 7.0


def _drift(idx: pd.DatetimeIndex) -> np.ndarray:
    mu = np.full(len(idx), 0.3)
    for a, b, d in EPISODE_DRIFT:
        mu[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] += d
    return mu


def make_interim(seed: int | None = None, end: str | None = None) -> dict[str, pd.Series]:
    """Synthetic stand-ins for the interim series that ``panel.build_interim`` writes."""
    cfg = config.models_cfg()["demo"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    idx = transform.month_end_range("2002-01-31", end or cfg["end"])
    n = len(idx)
    mu = _drift(idx)

    oil = np.zeros(n)
    ip = np.zeros(n)
    cpi = np.zeros(n)
    d_repo = np.zeros(n)
    inr = np.zeros(n)
    repo = np.zeros(n)
    repo[0] = 6.0
    e = rng.standard_normal((n, 5))
    for t in range(2, n):
        oil[t] = mu[t] + 0.15 * oil[t - 1] + TRUE_OIL_SD * e[t, 0]
        ip[t] = 0.4 + 0.1 * ip[t - 1] - 0.01 * oil[t] - 0.01 * oil[t - 1] + 1.5 * e[t, 1]
        cpi[t] = (
            0.35
            + 0.3 * (cpi[t - 1] - 0.35)
            + 0.008 * oil[t]
            + 0.012 * oil[t - 1]
            + 0.007 * oil[t - 2]
            + 0.004 * inr[t - 1]
            + 0.3 * e[t, 2]
        )
        rep_prev = repo[t - 1]
        d = (
            -0.03 * (rep_prev - 6.0)
            + 0.15 * (cpi[t] - 0.35)
            + 0.05 * d_repo[t - 1]
            + 0.08 * e[t, 3]
        )
        repo[t] = float(np.clip(rep_prev + d, 3.0, 10.0))
        d_repo[t] = repo[t] - rep_prev
        inr[t] = 0.3 + 0.04 * oil[t] - 0.4 * d_repo[t] + 0.2 * inr[t - 1] + 0.9 * e[t, 4]
    repo[:2] = 6.0

    brent = 25.0 * np.exp(np.cumsum(oil) / 100.0)
    inr_usd = 48.0 * np.exp(np.cumsum(inr) / 100.0)
    season = 0.3 * np.sin(2 * np.pi * (idx.month.to_numpy() - 3) / 12.0)
    cpi_mom = cpi + season
    cpi_level = 100.0 * np.cumprod(1.0 + cpi_mom / 100.0)
    iip = (
        100.0
        * np.exp(np.cumsum(ip) / 100.0)
        * (1.0 + 0.05 * np.sin(2 * np.pi * idx.month.to_numpy() / 12.0))
    )
    oil12 = pd.Series(oil, index=idx).rolling(12).sum().to_numpy()
    wpi_growth = 0.2 * oil + 0.3 * rng.standard_normal(n)
    wpi_fuel = 100.0 * np.exp(np.cumsum(wpi_growth) / 100.0)
    cpi_yoy_full = transform.yoy(pd.Series(cpi_level, index=idx), 12)
    vix = 18.0 + 4.0 * rng.standard_normal(n)
    for a, b in (("2008-10-31", "2009-02-28"), ("2020-03-31", "2020-05-31")):
        vix[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] += 22.0
    fed = np.clip(2.0 + np.cumsum(0.05 * rng.standard_normal(n)), 0.05, 6.0)

    s = lambda x, name: pd.Series(x, index=idx, name=name)  # noqa: E731
    out: dict[str, pd.Series] = {
        "brent_usd": s(brent, "brent_usd"),
        "inr_usd": s(inr_usd, "inr_usd"),
        "cpi_yoy": cpi_yoy_full.rename("cpi_yoy"),
        "cpi_mom": s(cpi_mom, "cpi_mom"),
        "cpi_level": s(cpi_level, "cpi_level"),
        "cpi_fuel": (0.3 * s(oil12, "cpi_fuel")).rename("cpi_fuel"),
        "cpi_core": (cpi_yoy_full - 0.5 + 0.2 * rng.standard_normal(n)).rename("cpi_core"),
        "iip": s(iip, "iip"),
        "wpi_fuel": s(wpi_fuel, "wpi_fuel"),
        "repo": s(repo, "repo"),
        "wacr": s(repo + 0.1 * rng.standard_normal(n), "wacr"),
        "fedfunds": s(fed, "fedfunds"),
        "vix": s(vix, "vix"),
        "crude_imports": s(20.0 + 1.5 * rng.standard_normal(n), "crude_imports"),
    }

    # quarterly: GDP growth, nominal GDP (INR crore) and the current account (US$ mn)
    qidx = pd.date_range(idx[0], idx[-1], freq="QE")
    oil_q = transform.to_quarterly(pd.Series(oil, index=idx), "mean").reindex(qidx).fillna(0.0)
    nq = len(qidx)
    g = np.zeros(nq)
    eq = rng.standard_normal((nq, 2))
    for t in range(1, nq):
        g[t] = (
            6.5
            + 0.5 * (g[t - 1] - 6.5)
            - 0.03 * oil_q.iloc[t]
            - 0.02 * oil_q.iloc[t - 1]
            + 0.7 * eq[t, 0]
        )
    g = pd.Series(g, index=qidx)
    for q, shock in (("2020Q2", -22.0), ("2020Q3", -8.0), ("2021Q2", 12.0)):
        g[pd.Period(q, "Q").end_time.normalize()] += shock
    nominal = 700_000.0 * np.exp(np.cumsum(np.full(nq, 0.11 / 4)))
    inr_q = transform.to_quarterly(pd.Series(inr_usd, index=idx), "mean").reindex(qidx)
    cab_gdp = -1.2 - 0.02 * oil_q.to_numpy() + 0.4 * eq[:, 1]
    cab = cab_gdp / 100.0 * nominal * 10.0 / inr_q.to_numpy()
    out["gdp_growth"] = pd.Series(g.to_numpy(), index=qidx, name="gdp_growth")
    out["gdp_nominal"] = pd.Series(nominal, index=qidx, name="gdp_nominal")
    out["cab"] = pd.Series(cab, index=qidx, name="cab")
    return out


def make_panels(
    seed: int | None = None, end: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly and quarterly panels built with the real panel code from synthetic interim series."""
    interim = make_interim(seed, end)
    monthly = panel.build_monthly(interim)
    quarterly = panel.build_quarterly(interim, monthly)
    return monthly, quarterly


def write_demo_data(seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write synthetic interim series and panels under ``demo/`` (requires ``OILSHOCK_DEMO=1``)."""
    p = config.paths()
    if not p.demo:
        raise RuntimeError("synthetic data may only be written in demo mode (OILSHOCK_DEMO=1)")
    interim = make_interim(seed)
    for key, s in interim.items():
        panel.write_interim(s, key, p.interim)
    monthly = panel.build_monthly(interim)
    quarterly = panel.build_quarterly(interim, monthly)
    config.write_parquet(monthly, p.processed / "monthly_panel.parquet")
    config.write_parquet(quarterly, p.processed / "quarterly_panel.parquet")
    return monthly, quarterly
