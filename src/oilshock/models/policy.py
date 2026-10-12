"""Policy Lab calculators (plan Section 7). Every assumption is an argument with a default from models.yaml."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilshock import config

CRORE = 1e7


def default(name: str) -> float:
    return float(config.models_cfg()["policy"][name])


def taylor_rate(
    pi: float | np.ndarray | pd.Series,
    pi_target: float | None = None,
    gap: float | np.ndarray | pd.Series = 0.0,
    r_star: float | None = None,
    a_pi: float | None = None,
    a_y: float | None = None,
) -> float | np.ndarray | pd.Series:
    """Taylor rule: ``i = r* + pi + a_pi * (pi - pi*) + a_y * gap`` (all in percent / percentage points)."""
    pi_target = default("pi_target") if pi_target is None else pi_target
    r_star = default("r_star") if r_star is None else r_star
    a_pi = default("a_pi") if a_pi is None else a_pi
    a_y = default("a_y") if a_y is None else a_y
    return r_star + pi + a_pi * (pi - pi_target) + a_y * gap


def hp_filter(y: np.ndarray, lamb: float = 1600.0) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter. Returns ``(trend, cycle)``."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        raise ValueError("need at least 4 observations")
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i : i + 3] = [1.0, -2.0, 1.0]
    trend = np.linalg.solve(np.eye(n) + lamb * D.T @ D, y)
    return trend, y - trend


def output_gap(
    gdp: pd.Series, method: str = "hp", is_log: bool = False, lamb: float | None = None
) -> pd.Series:
    """Output gap in percent of potential, from an HP filter on (log) real GDP."""
    if method != "hp":
        raise ValueError("only the HP filter is implemented")
    lamb = default("hp_lambda") if lamb is None else lamb
    x = gdp.dropna()
    logy = x if is_log else np.log(x)
    _, cyc = hp_filter(logy.to_numpy(), lamb)
    return pd.Series(100.0 * cyc, index=x.index, name="output_gap")


def tonnes_to_barrels(million_tonnes: float, barrels_per_tonne: float | None = None) -> float:
    bpt = default("barrels_per_tonne") if barrels_per_tonne is None else barrels_per_tonne
    return million_tonnes * 1e6 * bpt


def import_bill_change(
    d_price: float,
    volume_bbl: float,
    inr_usd: float,
    gdp_usd_bn: float | None = None,
) -> dict[str, float]:
    """Change in the annual crude import bill for a price change of ``d_price`` US$/bbl.

    ``volume_bbl`` is annual barrels. Returns US$ bn, INR crore and (if GDP is given) % of GDP.
    """
    usd_bn = d_price * volume_bbl / 1e9
    out = {
        "usd_bn": usd_bn,
        "inr_crore": usd_bn * 1e9 * inr_usd / CRORE,
    }
    if gdp_usd_bn:
        out["pct_gdp"] = 100.0 * usd_bn / gdp_usd_bn
    return out


def excise_cost(cut_per_litre: float, volume_litres: float) -> float:
    """Revenue cost in rupees of cutting excise by ``cut_per_litre`` on ``volume_litres`` per year."""
    return cut_per_litre * volume_litres


def rupees_to_crore(rupees: float) -> float:
    return rupees / CRORE
