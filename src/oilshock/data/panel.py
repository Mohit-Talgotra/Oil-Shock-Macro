"""Cleaning, splicing and panel construction (plan Section 5, stage 2).

``build_interim`` turns raw downloads into one Parquet per series in ``data/interim/``;
``build_monthly`` / ``build_quarterly`` join them into the model-ready panels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.data import fetch, parsers, splice, transform

log = logging.getLogger(__name__)

MONTHLY_COLUMNS = [
    "brent_usd",
    "inr_usd",
    "brent_inr",
    "oil",
    "inr",
    "brent_inr_chg",
    "cpi_yoy",
    "cpi_mom",
    "cpi_mom_sa",
    "iip",
    "ip",
    "iip_yoy",
    "repo",
    "d_repo",
    "wacr",
    "wpi_fuel_yoy",
    "cpi_fuel_yoy",
    "cpi_core_yoy",
    "fedfunds",
    "d_fedfunds",
    "vix",
    "crude_imports",
    "covid",
    "demonetisation",
    "gst",
    "inflation_targeting",
    "month",
]

QUARTERLY_COLUMNS = [
    "oil",
    "inr",
    "brent_inr_chg",
    "gdp_growth",
    "cab_gdp",
    "cpi_yoy",
    "repo",
    "iip_yoy",
    "covid_q",
    "gdp_level",
    "gdp_nominal_usd_bn",
]

# manual files that must exist for the full pipeline (others are optional and logged when absent)
REQUIRED_MANUAL = [
    "rbi_repo.xlsx",
    "mospi_cpi_2024_back.xlsx",
    "mospi_iip.xlsx",
    "mospi_gdp_2011.xlsx",
    "mospi_gdp_2022.xlsx",
    "rbi_bop.xlsx",
    "mospi_gdp_nominal.xlsx",
]


class DataCheckError(RuntimeError):
    """A data check failed; the build stops before artifacts are written."""


# ------------------------------------------------------------------ interim IO
def write_interim(obj: pd.Series | pd.DataFrame, key: str, interim_dir: Path | None = None) -> None:
    interim_dir = interim_dir or config.paths().interim
    df = obj.to_frame(key) if isinstance(obj, pd.Series) else obj
    config.write_parquet(df, interim_dir / f"{key}.parquet")


def read_interim(key: str, interim_dir: Path | None = None) -> pd.DataFrame:
    interim_dir = interim_dir or config.paths().interim
    return config.read_parquet(interim_dir / f"{key}.parquet")


def read_interim_series(key: str, interim_dir: Path | None = None) -> pd.Series:
    return read_interim(key, interim_dir).iloc[:, 0].rename(key)


def _fred_monthly(series_id: str, raw_dir: Path, how: str = "last") -> pd.Series:
    s = fetch.read_fred_raw(series_id, raw_dir).dropna()
    if how == "mean":  # daily -> monthly average
        out = s.resample(transform.MONTH_END).mean()
    else:
        s.index = transform.to_period_end(s.index, "M")
        out = s[~s.index.duplicated(keep="last")]
    return out.rename(series_id)


def _fred_quarterly(series_id: str, raw_dir: Path) -> pd.Series:
    s = fetch.read_fred_raw(series_id, raw_dir).dropna()
    s.index = transform.to_period_end(s.index, "Q")
    return s[~s.index.duplicated(keep="last")].rename(series_id)


# ------------------------------------------------------------------ interim build
def build_interim(
    raw_dir: Path | None = None,
    manual_dir: Path | None = None,
    interim_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Raw -> cleaned, spliced, period-end-indexed series. Returns a summary (missing files, splice rows)."""
    p = config.paths()
    raw_dir = raw_dir or p.raw
    manual_dir = manual_dir or p.manual
    interim_dir = interim_dir or p.interim
    reports_dir = reports_dir or p.reports
    cfg = config.series_cfg()
    thr = cfg["checks"]["splice_mean_gap_threshold_pp"]

    missing = [f for f in REQUIRED_MANUAL if not (manual_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Manual files missing from data/raw/manual/: "
            + ", ".join(missing)
            + ". See data/raw/manual/README.md for where to get them and the expected layout."
        )

    out: dict[str, pd.Series | pd.DataFrame] = {}
    splice_rows: list[pd.DataFrame] = []

    # --- FRED series
    out["brent_usd"] = _fred_monthly("POILBREUSDM", raw_dir).rename("brent_usd")
    out["inr_usd"] = _fred_monthly("EXINUS", raw_dir).rename("inr_usd")
    out["fedfunds"] = _fred_monthly("FEDFUNDS", raw_dir).rename("fedfunds")
    out["vix"] = _fred_monthly("VIXCLS", raw_dir, how="mean").rename("vix")
    daily = fetch.read_fred_raw("DCOILBRENTEU", raw_dir).dropna()
    out["brent_daily"] = daily.rename("brent_daily")

    # --- CPI: splice growth rates (never levels) at the switch month
    cpi_old = _fred_monthly("INDCPIALLMINMEI", raw_dir)
    cpi_new = parsers.parse_mospi_cpi(manual_dir / "mospi_cpi_2024_back.xlsx", base=2024)
    sw = cfg_switch(cfg, "cpi")
    yoy_old, yoy_new = transform.yoy(cpi_old, 12), transform.yoy(cpi_new, 12)
    mom_old, mom_new = transform.pct_change(cpi_old, 1), transform.pct_change(cpi_new, 1)
    for s in (yoy_old, yoy_new, mom_old, mom_new):
        s.attrs["kind"] = "growth"
    splice_rows.append(splice.overlap_report(yoy_old, yoy_new, "cpi_yoy"))
    splice_rows.append(splice.overlap_report(mom_old, mom_new, "cpi_mom"))
    cpi_yoy = splice.splice_growth(yoy_old, yoy_new, sw).rename("cpi_yoy")
    cpi_mom = splice.splice_growth(mom_old, mom_new, sw).rename("cpi_mom")
    out["cpi_yoy"] = cpi_yoy
    out["cpi_mom"] = cpi_mom
    out["cpi_level"] = transform.cumulate_growth(cpi_mom.dropna(), 100.0).rename("cpi_level")
    groups = parsers.parse_mospi_cpi_groups(manual_dir / "mospi_cpi_2024_back.xlsx")
    for col in groups.columns:
        out[col] = transform.yoy(groups[col], 12).rename(col)

    # --- GDP growth: chain the splices oldest -> newest
    g_old = _fred_quarterly("NAEXKP01INQ652S", raw_dir)
    g_mid = parsers.parse_mospi_gdp(manual_dir / "mospi_gdp_2011.xlsx", "2011-12")
    g_new = parsers.parse_mospi_gdp(manual_dir / "mospi_gdp_2022.xlsx", "2022-23")
    for s in (g_old, g_mid, g_new):
        s.attrs["kind"] = "growth"
    steps = [x for x in cfg_splices(cfg, "gdp_growth")]
    first = splice.splice_growth(g_old, g_mid, steps[0]["switch"])
    splice_rows.append(splice.overlap_report(g_old, g_mid, "gdp_growth_old_vs_2011"))
    second = splice.splice_growth(first, g_new, steps[1]["switch"])
    splice_rows.append(splice.overlap_report(first, g_new, "gdp_growth_2011_vs_2022"))
    out["gdp_growth"] = second.rename("gdp_growth")
    out["gdp_nominal"] = parsers.parse_mospi_nominal_gdp(manual_dir / "mospi_gdp_nominal.xlsx")
    out["cab"] = parsers.parse_rbi_bop(manual_dir / "rbi_bop.xlsx")["cab"]

    # --- Monthly manual series
    out["iip"] = parsers.parse_mospi_iip(manual_dir / "mospi_iip.xlsx")
    for fname, key, fn in [
        ("rbi_wacr.xlsx", "wacr", parsers.parse_rbi_wacr),
        ("eoa_wpi_fuel.xlsx", "wpi_fuel", parsers.parse_wpi_fuel),
        ("ppac_crude_imports.xlsx", "crude_imports", parsers.parse_ppac_imports),
    ]:
        if (manual_dir / fname).exists():
            out[key] = fn(manual_dir / fname).rename(key)
        else:
            log.warning("optional manual file %s not found; series %s will be missing", fname, key)
    if (manual_dir / "ppac_excise.xlsx").exists():
        out["excise"] = parsers.parse_ppac_excise(manual_dir / "ppac_excise.xlsx")

    # --- Repo: expand change dates into a monthly step series up to the latest FRED month
    repo_events = parsers.parse_rbi_repo(manual_dir / "rbi_repo.xlsx")
    end = out["inr_usd"].dropna().index.max()
    idx = transform.month_end_range(repo_events.index.min() + pd.offsets.MonthEnd(0), end)
    out["repo"] = transform.step_series(repo_events, idx).rename("repo")

    for key, obj in out.items():
        write_interim(obj, key, interim_dir)
    report = splice.write_overlap_report(splice_rows, reports_dir / "splice_check.csv", thr)
    log.info("interim series written: %s", ", ".join(sorted(out)))
    return {"series": sorted(out), "splice": report}


def cfg_switch(cfg: dict[str, Any], key: str) -> str:
    for s in cfg["series"]:
        if s["key"] == key:
            return s["splice"][0]["switch"]
    raise KeyError(key)


def cfg_splices(cfg: dict[str, Any], key: str) -> list[dict[str, Any]]:
    for s in cfg["series"]:
        if s["key"] == key:
            return s["splice"]
    raise KeyError(key)


# ------------------------------------------------------------------ panels
def load_interim_all() -> dict[str, pd.Series]:
    """Every single-column interim Parquet as a Series (the excise event table is read separately)."""
    out: dict[str, pd.Series] = {}
    for f in sorted(config.paths().interim.glob("*.parquet")):
        if f.stem == "excise":
            continue
        out[f.stem] = read_interim_series(f.stem)
    return out


def _get(interim: dict[str, pd.Series], key: str) -> pd.Series:
    return interim.get(key, pd.Series(dtype=float, name=key))


def build_monthly(
    interim: dict[str, pd.Series] | None = None, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Monthly panel from 2004-01 (period-end index). ``interim`` maps key -> Series (defaults to disk)."""
    mcfg = cfg or config.models_cfg()
    if interim is None:
        interim = load_interim_all()
    start = pd.Timestamp(mcfg["sample"]["monthly_start"])
    brent, inr_usd = _get(interim, "brent_usd"), _get(interim, "inr_usd")
    last = min(brent.dropna().index.max(), inr_usd.dropna().index.max())
    idx_full = transform.month_end_range(
        min(s.dropna().index.min() for s in (brent, inr_usd)), last
    )
    df = pd.DataFrame(index=idx_full)
    df["brent_usd"], df["inr_usd"] = brent.reindex(idx_full), inr_usd.reindex(idx_full)
    df["brent_inr"] = df["brent_usd"] * df["inr_usd"]
    df["oil"] = transform.log_diff(df["brent_usd"])
    df["inr"] = transform.log_diff(df["inr_usd"])
    df["brent_inr_chg"] = transform.log_diff(df["brent_inr"])

    df["cpi_yoy"] = _get(interim, "cpi_yoy").reindex(idx_full)
    df["cpi_mom"] = _get(interim, "cpi_mom").reindex(idx_full)
    level = _get(interim, "cpi_level").reindex(idx_full)
    sa = transform.seasonal_adjust(level.dropna(), "stl", 12).reindex(idx_full)
    df["cpi_mom_sa"] = transform.pct_change(sa, 1)

    iip = _get(interim, "iip").reindex(idx_full)
    df["iip"] = iip
    iip_sa = transform.seasonal_adjust(iip.dropna(), "stl", 12).reindex(idx_full)
    df["ip"] = transform.log_diff(iip_sa)
    df["iip_yoy"] = transform.yoy(iip, 12)

    df["repo"] = _get(interim, "repo").reindex(idx_full)
    df["d_repo"] = df["repo"].diff()
    df["wacr"] = _get(interim, "wacr").reindex(idx_full)
    df["wpi_fuel_yoy"] = transform.yoy(_get(interim, "wpi_fuel").reindex(idx_full), 12)
    df["cpi_fuel_yoy"] = _get(interim, "cpi_fuel").reindex(idx_full)
    df["cpi_core_yoy"] = _get(interim, "cpi_core").reindex(idx_full)
    df["fedfunds"] = _get(interim, "fedfunds").reindex(idx_full)
    df["d_fedfunds"] = df["fedfunds"].diff()
    df["vix"] = _get(interim, "vix").reindex(idx_full)
    df["crude_imports"] = _get(interim, "crude_imports").reindex(idx_full)

    ev = config.events_cfg()["dummies"]
    for name in ("covid", "demonetisation", "gst", "inflation_targeting"):
        d = ev[name]
        df[name] = transform.window_dummy(idx_full, d["start"], d.get("end")).to_numpy()
    df["month"] = idx_full.month

    dropped = int((df.index < start).sum())
    if dropped:
        log.info(
            "monthly panel: dropped %d rows before sample start %s (used only to build transforms)",
            dropped,
            start.date(),
        )
    df = df.loc[start:]
    df.index.name = "date"
    return df[MONTHLY_COLUMNS]


def build_quarterly(
    interim: dict[str, pd.Series] | None = None,
    monthly: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Quarterly panel from 2004Q2 (period-end index)."""
    mcfg = cfg or config.models_cfg()
    if interim is None:
        interim = load_interim_all()
    if monthly is None:
        monthly = build_monthly(interim, mcfg)
    start = pd.Timestamp(mcfg["sample"]["quarterly_start"])
    brent_m = _get(interim, "brent_usd")
    inr_m = _get(interim, "inr_usd")
    brent_q = transform.to_quarterly(brent_m.dropna(), "mean")
    inr_q = transform.to_quarterly(inr_m.dropna(), "mean")
    idx = brent_q.index.intersection(inr_q.index)
    df = pd.DataFrame(index=idx)
    df["oil"] = transform.log_diff(brent_q).reindex(idx)
    df["inr"] = transform.log_diff(inr_q).reindex(idx)
    df["brent_inr_chg"] = transform.log_diff(brent_q * inr_q).reindex(idx)
    df["gdp_growth"] = _get(interim, "gdp_growth").reindex(idx)
    nominal = _get(interim, "gdp_nominal").reindex(idx)
    cab = _get(interim, "cab").reindex(idx)
    df["cab_gdp"] = 100.0 * cab * inr_q.reindex(idx) / (nominal * 10.0)
    df["gdp_nominal_usd_bn"] = nominal * 10.0 / inr_q.reindex(idx) / 1000.0
    df["cpi_yoy"] = transform.to_quarterly(monthly["cpi_yoy"], "mean").reindex(idx)
    df["repo"] = transform.to_quarterly(monthly["repo"], "last").reindex(idx)
    df["iip_yoy"] = transform.to_quarterly(monthly["iip_yoy"], "mean").reindex(idx)
    df["covid_q"] = transform.quarter_dummy(
        idx, config.events_cfg()["dummies"]["covid_q"]["quarters"]
    ).to_numpy()
    gl = transform.gdp_level_from_yoy(df["gdp_growth"])
    df["gdp_level"] = gl.reindex(idx)
    dropped = int((df.index < start).sum())
    if dropped:
        log.info("quarterly panel: dropped %d rows before %s", dropped, start.date())
    df = df.loc[start:]
    df.index.name = "date"
    return df[QUARTERLY_COLUMNS]


def build_panels(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = config.paths()
    interim = load_interim_all()
    m = build_monthly(interim)
    q = build_quarterly(interim, m)
    if write:
        config.write_parquet(m, p.processed / "monthly_panel.parquet")
        config.write_parquet(q, p.processed / "quarterly_panel.parquet")
    return m, q


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = config.paths()
    return (
        config.read_parquet(p.processed / "monthly_panel.parquet"),
        config.read_parquet(p.processed / "quarterly_panel.parquet"),
    )


# ------------------------------------------------------------------ panel-level helpers for the models
def monthly_var_frame(
    panel: pd.DataFrame, window: str = "full", cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endogenous frame (Model A, oil first) and exogenous frame for a window; rows with any NaN dropped."""
    return _var_frame(panel, (cfg or config.models_cfg())["var_monthly"], window, cfg)


def quarterly_var_frame(
    panel: pd.DataFrame, window: str = "full", cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _var_frame(panel, (cfg or config.models_cfg())["var_quarterly"], window, cfg)


def _var_frame(
    panel: pd.DataFrame, spec: dict[str, Any], window: str, cfg: dict[str, Any] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or config.models_cfg()
    w = cfg["windows"][window]
    cols = [spec["columns"][v] for v in spec["variables"]]
    endog = panel[cols].copy()
    endog.columns = spec["variables"]
    exog = panel[spec["exog"]].copy()
    joined = pd.concat([endog, exog], axis=1).dropna()
    if w.get("start"):
        joined = joined.loc[pd.Timestamp(w["start"]) :]
    if w.get("end"):
        joined = joined.loc[: pd.Timestamp(w["end"])]
    return joined[spec["variables"]], joined[spec["exog"]]


# ------------------------------------------------------------------ data checks (plan Section 11)
def _monthly_move(monthly: pd.DataFrame, col: str) -> pd.Series:
    """Single-month move of a series on the scale used by ``checks.max_abs_monthly_change``."""
    if col == "brent_usd":
        return monthly["oil"]
    if col == "inr_usd":
        return monthly["inr"]
    if col == "iip":
        return transform.pct_change(monthly["iip"], 1)
    return monthly[col].diff()


def run_data_checks(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    splice_report: pd.DataFrame | None = None,
    strict: bool = True,
) -> list[str]:
    """Return a list of warnings; raise ``DataCheckError`` on fatal problems when ``strict``."""
    cfg = config.series_cfg()["checks"]
    fatal: list[str] = []
    warn: list[str] = []
    for name, df in (("monthly", monthly), ("quarterly", quarterly)):
        if not df.index.is_monotonic_increasing:
            fatal.append(f"{name} panel index is not monotonic")
        if df.index.has_duplicates:
            fatal.append(f"{name} panel has duplicate dates")
    cutoff = monthly.index.max()
    stale_m = cfg["expected_last_month"]["default_staleness_months"]
    for col in ["brent_usd", "inr_usd", "cpi_yoy", "repo", "iip", "fedfunds", "vix"]:
        last = monthly[col].dropna().index.max() if monthly[col].notna().any() else None
        if last is None:
            fatal.append(f"{col}: no data")
        elif (cutoff.to_period("M") - last.to_period("M")).n > stale_m:
            warn.append(f"{col}: stale, last observation {last.date()} (cut-off {cutoff.date()})")
    for col, limit in cfg["max_abs_monthly_change"].items():
        if col not in monthly.columns and col not in ("brent_usd", "inr_usd"):
            continue
        chg = _monthly_move(monthly, col).abs()
        bad = chg[chg > limit]
        if len(bad):
            fatal.append(
                f"{col}: {len(bad)} single-month changes exceed {limit} "
                f"(first {bad.index[0].date()}): likely a parsing error"
            )
    if splice_report is not None and not splice_report.empty:
        over = splice_report[splice_report["mean_abs_gap"] > cfg["splice_mean_gap_threshold_pp"]]
        for _, row in over.iterrows():
            warn.append(
                f"splice {row['series']}: mean |gap| {row['mean_abs_gap']:.2f} pp exceeds threshold; mention on the About page"
            )
    scfg = config.series_cfg()
    spot = scfg.get("spot_checks", {}) or {}
    need = int(scfg.get("spot_check_min", 5))
    for key, col in (scfg.get("spot_check_sources", {}) or {}).items():
        if len(spot.get(col, []) or []) < need:
            warn.append(
                f"spot checks: {key} ({col}) has {len(spot.get(col, []) or [])} of {need} "
                "hand-checked values in series.yaml; add them from the official release"
            )
    for col, items in spot.items():
        frame = quarterly if col in quarterly.columns and col not in monthly.columns else monthly
        if col not in frame.columns:
            fatal.append(f"spot check column {col!r} is not in the panel")
            continue
        for it in items or []:
            val = frame[col].get(pd.Timestamp(it["date"]), np.nan)
            if not np.isclose(val, it["value"], atol=it.get("tol", 0.05)):
                fatal.append(f"spot check {col} {it['date']}: got {val}, expected {it['value']}")
    for w in warn:
        log.warning("data check: %s", w)
    if fatal and strict:
        raise DataCheckError("data checks failed:\n  " + "\n  ".join(fatal))
    for f in fatal:
        log.error("data check (non-strict): %s", f)
    return warn + fatal
