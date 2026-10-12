"""Offline pipeline CLI: ``python -m oilshock.pipeline [stage]`` (plan Section 5).

Stages: data, var, forecast, stress, timesfm, report, all (= data, var, forecast, stress, report).
Heavy estimation happens here; the Streamlit app only reads the files written to ``artifacts/``.
The pipeline writes ``artifacts/manifest.json`` at the end of every run.

Flags
-----
--synthetic   use simulated data (requires OILSHOCK_DEMO=1, i.e. ``make demo``); never real results
--fast        fewer bootstrap replications and every third backtest origin (for smoke runs)
--no-strict   record failed checks instead of stopping the build (investigate; report on About page)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.data import panel as panelmod
from oilshock.data import synthetic
from oilshock.models import local_proj, scenario
from oilshock.models import var as varmod

log = logging.getLogger("oilshock.pipeline")

MONTHLY_WINDOWS = ["full", "pre2008", "pre2014", "pre2022", "pre2026", "pre_it", "post_it"]
QUARTERLY_WINDOWS = ["full", "pre2008", "pre2014", "pre2022", "pre2026"]


class BuildCheckError(RuntimeError):
    """A plausibility or data check failed in strict mode."""


@dataclass
class Check:
    name: str
    ok: bool
    level: str  # "error" stops a strict build; "flag" is logged for review
    detail: str


# ===================================================================================== helpers
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "xgboost",
        "scikit-learn",
        "streamlit",
        "plotly",
        "pyarrow",
        "shap",
        "timesfm",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def update_manifest(**updates: Any) -> dict[str, Any]:
    """Merge ``updates`` into ``artifacts/manifest.json`` (data cut-off, build time, commit, versions)."""
    p = config.paths()
    path = p.artifacts / "manifest.json"
    manifest: dict[str, Any] = config.read_json(path) if path.exists() else {}
    manifest.update(
        {
            "build_time_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "package_versions": _versions(),
            "series": [s["key"] for s in config.series_cfg()["series"]],
            "synthetic": p.demo,
            "note": config.models_cfg()["demo"]["note"] if p.demo else "",
        }
    )
    manifest.update(updates)
    config.write_json(manifest, path)
    return manifest


def _load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    return panelmod.load_panels()


def _write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


# ===================================================================================== stage: data
def stage_data(synthetic_data: bool = False, strict: bool = True) -> None:
    p = config.paths()
    if synthetic_data:
        if not p.demo:
            raise SystemExit(
                "--synthetic requires demo mode: run `make demo` (sets OILSHOCK_DEMO=1)"
            )
        monthly, quarterly = synthetic.write_demo_data()
        warnings = panelmod.run_data_checks(monthly, quarterly, None, strict=False)
        log.info("synthetic demo data written under %s (%d warnings)", p.base, len(warnings))
    else:
        import os

        if os.environ.get("FRED_API_KEY"):
            from oilshock.data import fetch

            fetch.fetch_all()
        else:
            log.warning("FRED_API_KEY not set: using FRED files already in data/raw/")
        summary = panelmod.build_interim()
        monthly, quarterly = panelmod.build_panels(write=False)
        panelmod.run_data_checks(monthly, quarterly, summary["splice"], strict=strict)
        config.write_parquet(monthly, p.processed / "monthly_panel.parquet")
        config.write_parquet(quarterly, p.processed / "quarterly_panel.parquet")
    update_manifest(
        data_cutoff=str(monthly.index.max().date()),
        quarterly_cutoff=str(quarterly.index.dropna().max().date()),
        monthly_rows=int(len(monthly)),
        quarterly_rows=int(len(quarterly)),
        stages_run=_stages("data"),
    )


def _stages(stage: str) -> list[str]:
    path = config.paths().artifacts / "manifest.json"
    done = config.read_json(path).get("stages_run", []) if path.exists() else []
    return sorted(set(done) | {stage})


# ===================================================================================== stage: var
def plausibility_checks(
    irf_a: pd.DataFrame, diag_a: dict[str, Any], cfg: dict[str, Any]
) -> list[Check]:
    """Economic plausibility checks on Model A, full sample (plan Section 11)."""
    checks: list[Check] = []
    cpi = irf_a[(irf_a["response"] == "cpi") & (irf_a["cumulative"]) & (irf_a["horizon"] == 11)][
        "point"
    ].iloc[0]
    checks.append(
        Check(
            "cpi_cumulative_12m_positive",
            bool(cpi > 0),
            "error",
            f"12-month cumulative CPI response per 1 s.d. = {cpi:.3f} pp",
        )
    )
    inr0 = irf_a[(irf_a["response"] == "inr") & (~irf_a["cumulative"]) & (irf_a["horizon"] == 0)][
        "point"
    ].iloc[0]
    checks.append(
        Check(
            "rupee_depreciates_on_impact",
            bool(inr0 > 0),
            "error",
            f"INR impact response = {inr0:.3f}",
        )
    )
    bps = scenario.first_year_cpi_bps(irf_a, 10.0)
    lo, hi = cfg["scenario"]["plausibility_bps_range_10pct"]
    rlo, rhi = cfg["scenario"]["red_flag_bps_range_10pct"]
    red = "; RED FLAG (outside %g-%g bps)" % (rlo, rhi) if not rlo <= bps <= rhi else ""
    checks.append(
        Check(
            "cpi_response_in_rbi_range",
            bool(lo <= bps <= hi),
            "flag",
            f"{bps:.0f} bps per +10% oil vs RBI benchmark range {lo:g}-{hi:g} bps{red}; "
            "investigate sample, ordering, lags, splicing",
        )
    )
    checks.append(
        Check(
            "var_stable",
            bool(diag_a.get("stable", False)),
            "error",
            f"min root modulus {diag_a.get('min_root_modulus', float('nan')):.3f}",
        )
    )
    wp = diag_a.get("whiteness_pvalue", float("nan"))
    checks.append(
        Check(
            "residual_whiteness",
            bool(np.isnan(wp) or wp >= 0.05),
            "error",
            f"Portmanteau p-value = {wp:.3f}",
        )
    )
    return checks


def _fit_window(
    endog: pd.DataFrame,
    exog: pd.DataFrame | None,
    vcfg: dict[str, Any],
    quarterly: bool,
    reps: int,
    seed: int,
):
    ex = varmod.usable_exog(exog)
    k = endog.shape[1]
    m = 0 if ex is None else ex.shape[1]
    if quarterly:
        p = int(vcfg["fixed_lags"])
        while p > 1 and len(endog) - p - (k * p + 1 + m) < int(vcfg["min_df_resid"]):
            p -= 1
        sel = {"chosen": p, "bic": p, "aic": p, "hqic": p, "whiteness_pvalue": np.nan}
        if len(endog) - p - (k * p + 1 + m) < int(vcfg["min_df_resid"]):
            return None
    else:
        sel = varmod.select_lags(
            endog,
            int(vcfg["maxlags"]),
            ex,
            float(vcfg["whiteness_alpha"]),
            int(vcfg["whiteness_nlags"]),
        )
        p = sel["chosen"]
    res = varmod.fit_var(endog, p, ex)
    fit = varmod.estimate_var(endog, p, ex)
    fit.extras["exog_default"] = {} if ex is None else ex.tail(12).mean().to_dict()
    fevd = varmod.fevd_table(fit, int(vcfg["horizon"]))
    diag = varmod.diagnostics(res, int(vcfg["whiteness_nlags"]), fevd)
    diag.update(
        {
            "lag_selection_aic": sel["aic"],
            "lag_selection_bic": sel["bic"],
            "lag_selection_hqic": sel["hqic"],
        }
    )
    irf = varmod.irf_bands(fit, int(vcfg["horizon"]), reps, seed)
    return res, fit, irf, fevd, diag


def stage_var(fast: bool = False, strict: bool = True) -> None:
    cfg = config.models_cfg()
    p = config.paths()
    monthly, quarterly = _load_panels()
    reps_scale = 0.2 if fast else 1.0
    seed = int(cfg["seed"])
    outputs: list[tuple[str, pd.DataFrame | Any]] = []
    diag_rows: list[dict[str, Any]] = []
    vm, vq = cfg["var_monthly"], cfg["var_quarterly"]

    # ---- unit roots (reported; differencing decisions live in models.yaml)
    endog_full, exog_full = panelmod.monthly_var_frame(monthly, "full")
    ur = varmod.unit_root_table(
        endog_full, vm["unit_roots"]["adf_alpha"], vm["unit_roots"]["kpss_alpha"]
    )
    p.reports.mkdir(parents=True, exist_ok=True)
    ur.to_csv(p.reports / "unit_roots.csv", index=False)
    for _, r in ur[~ur["stationary"]].iterrows():
        log.warning(
            "unit-root: %s is not clearly stationary (ADF p=%.3f, KPSS p=%.3f); it is kept as modelled and reported",
            r["variable"],
            r["adf_pvalue"],
            r["kpss_pvalue"],
        )

    # ---- Model A and B over all windows
    full_a = None
    for model_id, windows, vcfg, framer, quarterly_flag in (
        ("A", MONTHLY_WINDOWS, vm, lambda w: panelmod.monthly_var_frame(monthly, w), False),
        ("B", QUARTERLY_WINDOWS, vq, lambda w: panelmod.quarterly_var_frame(quarterly, w), True),
    ):
        reps = max(50, int(vcfg["reps"] * reps_scale))
        for w in windows:
            endog, exog = framer(w)
            try:
                got = _fit_window(endog, exog, vcfg, quarterly_flag, reps, seed)
            except Exception as exc:
                log.error("Model %s window %s failed: %s", model_id, w, exc)
                got = None
            if got is None:
                log.warning(
                    "Model %s window %s skipped (too few observations: %d)", model_id, w, len(endog)
                )
                continue
            res, fit, irf, fevd, diag = got
            diag.update(
                {
                    "model": model_id,
                    "window": w,
                    "n_obs": len(endog),
                    "start": str(endog.index.min().date()),
                    "end": str(endog.index.max().date()),
                }
            )
            diag_rows.append(diag)
            outputs.append((f"irf_{model_id}_{w}.parquet", irf))
            outputs.append((f"fevd_{model_id}_{w}.parquet", fevd))
            outputs.append((f"fitted_{model_id}_{w}.pkl", res))
            if model_id == "A" and w == "full":
                full_a = (endog, exog, fit, irf, diag)

    if full_a is None:
        raise BuildCheckError("Model A could not be estimated on the full sample")
    endog, exog, fit, irf_a, diag_a = full_a
    ex = varmod.usable_exog(exog)

    # ---- plausibility checks BEFORE anything is written (strict mode stops the build)
    checks = plausibility_checks(irf_a, diag_a, cfg)
    for c in checks:
        (log.info if c.ok else log.error if c.level == "error" else log.warning)(
            "plausibility %s: %s (%s)", c.name, "ok" if c.ok else "FAILED", c.detail
        )
    failed = [c for c in checks if not c.ok and c.level == "error"]
    config.write_json([c.__dict__ for c in checks], p.reports / "plausibility.json")
    if failed and strict:
        raise BuildCheckError(
            "plausibility checks failed: "
            + "; ".join(f"{c.name} ({c.detail})" for c in failed)
            + ". Investigate sample, ordering, lag length and splicing before reporting; use --no-strict to record anyway."
        )

    # ---- robustness: block exogeneity, order check, local projections, asymmetry
    reps = max(50, int(vm["reps"] * reps_scale))
    h = int(vm["horizon"])
    fit_b = varmod.estimate_var(endog, fit.p, ex, block_exog=True)
    outputs.append(("irf_A_blockexog_full.parquet", varmod.irf_bands(fit_b, h, reps, seed + 1)))
    order2 = list(vm["order_check_variables"])
    fit_o = varmod.estimate_var(endog[order2], fit.p, ex)
    outputs.append(("irf_A_inr2_full.parquet", varmod.irf_bands(fit_o, h, reps, seed + 2)))
    lp_cfg = cfg["local_projections"]
    lp = local_proj.lp_all(
        endog,
        ex,
        list(vm["differenced"]),
        int(lp_cfg["horizons"]),
        int(lp_cfg["lags"]),
        int(lp_cfg["nw_lags"]),
    )
    outputs.append(("irf_A_lp_full.parquet", lp))
    if cfg["asymmetry"]["enabled"]:
        asym = pd.concat(
            [
                local_proj.asymmetric_lp(
                    endog,
                    ex,
                    v,
                    v in vm["differenced"],
                    int(cfg["asymmetry"]["horizons"]),
                    int(lp_cfg["lags"]),
                    int(lp_cfg["nw_lags"]),
                )
                for v in ("cpi", "inr", "ip", "repo")
            ],
            ignore_index=True,
        )
        outputs.append(("lp_asym_A_full.parquet", asym))
    # ---- rolling windows: pass-through over time
    rolling = varmod.rolling_irfs(
        endog,
        int(cfg["rolling"]["window"]),
        int(cfg["rolling"]["step"]),
        ex,
        lags=min(fit.p, 4),
        response=cfg["rolling"]["response"],
        horizon_months=int(cfg["rolling"]["horizon_months"]),
        reps=max(40, int(vm["rolling_reps"] * reps_scale)),
        seed=seed + 3,
    )
    outputs.append(("rolling_A_full.parquet", rolling))
    outputs.append(("diagnostics.parquet", pd.DataFrame(diag_rows)))

    # ---- write
    for name, obj in outputs:
        path = p.var / name
        if name.endswith(".pkl"):
            _write_pickle(obj, path)
        else:
            config.write_parquet(obj, path)
    pd.DataFrame(diag_rows).to_csv(p.reports / "var_diagnostics.csv", index=False)
    update_manifest(
        bootstrap_reps=reps,
        var_seed=seed,
        plausibility=[c.__dict__ for c in checks],
        cpi_bps_per_10pct_oil=scenario.first_year_cpi_bps(irf_a, 10.0),
        stages_run=_stages("var"),
    )
    log.info("VAR artifacts written to %s", p.var)


# ===================================================================================== stage: forecast
def stage_forecast(fast: bool = False) -> None:
    from oilshock.forecast import backtest
    from oilshock.forecast.base import get_target, panel_for, target_names
    from oilshock.forecast.xgb import shap_summary

    cfg = config.models_cfg()
    p = config.paths()
    monthly, quarterly = _load_panels()
    stride = 3 if fast else int(cfg["forecast"]["stride"])
    grid_cfg = cfg["forecast"]["oil_grid"]
    pcts = list(
        range(int(grid_cfg["min_pct"]), int(grid_cfg["max_pct"]) + 1, int(grid_cfg["step"]))
    )
    for tname in target_names():
        spec = get_target(tname)
        pan = panel_for(spec, monthly, quarterly)
        origins = backtest.make_origins(pan, spec, stride)
        log.info("target %s: %d origins (stride %d)", tname, len(origins), stride)
        for mname in backtest.MODEL_ORDER:
            try:
                bt = backtest.rolling_origin(
                    backtest.make_forecaster(mname, spec), pan, spec, origins
                )
                final_model = backtest.make_forecaster(mname, spec)
                final = backtest.latest_forecast(final_model, pan, spec)
            except Exception as exc:
                log.error("%s/%s failed: %s", mname, tname, exc)
                continue
            config.write_parquet(
                pd.concat([bt, final], ignore_index=True), p.forecasts / f"{mname}_{tname}.parquet"
            )
            if mname in {"arimax", "var"}:
                _write_pickle(final_model, p.forecasts / f"fitted_{mname}_{tname}.pkl")
            if mname == "xgb":
                h_shap = min(6, spec.horizons)
                Dtr, _ = final_model.train_sets[h_shap]
                shap = shap_summary(final_model.models[(h_shap, 0.50)], Dtr)
                config.write_parquet(shap, p.forecasts / f"shap_xgb_{tname}.parquet")
                grid = final_model.grid_forecast(
                    spec.horizons, pcts, int(grid_cfg["horizon_months"])
                )
                origin = final["origin"].iloc[0]
                grid["origin"] = origin
                grid["target_date"] = pd.date_range(
                    origin, periods=spec.horizons + 1, freq=spec.alias
                )[1:][grid["horizon"].to_numpy() - 1]
                config.write_parquet(grid, p.forecasts / f"xgb_grid_{tname}.parquet")
    rebuild_metrics()
    update_manifest(backtest_stride=stride, stages_run=_stages("forecast"))


def rebuild_metrics() -> pd.DataFrame:
    """Recompute ``metrics.parquet`` from every backtest file present (includes TimesFM when run)."""
    from oilshock.forecast import evaluate
    from oilshock.forecast.base import target_names

    p = config.paths()
    frames = []
    for tname in target_names():
        fcs = {}
        for f in sorted(p.forecasts.glob(f"*_{tname}.parquet")):
            if f.stem.startswith(("fitted", "shap", "xgb_grid", "metrics")):
                continue
            model = f.stem[: -(len(tname) + 1)]
            fcs[model] = config.read_parquet(f)
        if "naive" in fcs:
            frames.append(evaluate.build_metrics(fcs, tname))
    metrics = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=evaluate.METRIC_COLUMNS)
    )
    config.write_parquet(metrics, p.forecasts / "metrics.parquet")
    return metrics


# ===================================================================================== stage: stress
def stage_stress() -> None:
    from oilshock.stress import run as stress_run

    monthly, quarterly = _load_panels()
    summary = stress_run.run_all(monthly, quarterly)
    update_manifest(
        stress_rows=int(len(summary)),
        freeze_tag=config.models_cfg()["stress"]["freeze_tag"],
        stages_run=_stages("stress"),
    )


# ===================================================================================== stage: timesfm
def stage_timesfm(fast: bool = False) -> None:
    from oilshock.forecast import backtest, timesfm_runner
    from oilshock.forecast.base import get_target, panel_for, target_names

    cfg = config.models_cfg()
    p = config.paths()
    monthly, quarterly = _load_panels()
    stride = int(cfg["timesfm"]["origin_stride"])
    for tname in target_names():
        spec = get_target(tname)
        pan = panel_for(spec, monthly, quarterly)
        origins = backtest.make_origins(pan, spec, stride)
        for mname in ("timesfm_zs", "timesfm_cov"):
            model = backtest.make_forecaster(mname, spec)
            bt = backtest.rolling_origin(model, pan, spec, origins)
            final = backtest.latest_forecast(backtest.make_forecaster(mname, spec), pan, spec)
            config.write_parquet(
                pd.concat([bt, final], ignore_index=True), p.forecasts / f"{mname}_{tname}.parquet"
            )
    rebuild_metrics()
    update_manifest(
        timesfm=timesfm_runner.package_info(),
        timesfm_origin_stride=stride,
        stages_run=_stages("timesfm"),
    )


# ===================================================================================== stage: report
def stage_report() -> None:
    p = config.paths()
    p.reports.mkdir(parents=True, exist_ok=True)
    metrics_path = p.forecasts / "metrics.parquet"
    if metrics_path.exists():
        m = config.read_parquet(metrics_path)
        m[m["regime"] == "all"].groupby(["target", "model"], as_index=False)[
            ["rmse", "mase", "coverage80", "coverage95", "pinball"]
        ].mean().to_csv(p.reports / "metrics_summary.csv", index=False)
    summary_path = p.stress / "summary.parquet"
    if summary_path.exists():
        config.read_parquet(summary_path).to_csv(p.reports / "stress_summary.csv", index=False)
    export_figures()
    monthly, _ = _load_panels()
    update_manifest(data_cutoff=str(monthly.index.max().date()), stages_run=_stages("report"))


def export_figures() -> None:
    """Key figures for the slides: HTML always, PNG when kaleido is installed."""
    from oilshock.viz import charts

    p = config.paths()
    p.figures.mkdir(parents=True, exist_ok=True)
    figs = {}
    try:
        irf = config.read_parquet(p.var / "irf_A_full.parquet")
        sc = scenario.scale_irf(irf, 30.0)
        yoy = scenario.yoy_inflation_path(sc)
        figs["transmission_irf_grid_plus30"] = charts.irf_grid(
            sc,
            90,
            config.models_cfg()["var_monthly"]["differenced"],
            extra={"cpi_yoy": yoy},
            title="Response to a +30% oil shock",
        )
        roll = p.var / "rolling_A_full.parquet"
        if roll.exists():
            figs["passthrough_over_time"] = charts.rolling_passthrough(config.read_parquet(roll))
    except Exception as exc:
        log.warning("figure export skipped: %s", exc)
    for name, fig in figs.items():
        fig.write_html(p.figures / f"{name}.html", include_plotlyjs="cdn")
        try:
            fig.write_image(p.figures / f"{name}.png", scale=2)
        except Exception:
            pass


# ===================================================================================== CLI
def main(
    stage: str = "all", synthetic_data: bool = False, fast: bool = False, strict: bool = True
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if synthetic_data and not config.paths().demo:
        raise SystemExit("--synthetic requires demo mode: run `make demo` (sets OILSHOCK_DEMO=1)")
    if config.paths().demo:
        log.warning(
            "DEMO MODE: all inputs are SYNTHETIC. Outputs go to %s and must never be presented as findings.",
            config.paths().base,
        )
    order = {
        "data": lambda: stage_data(synthetic_data, strict),
        "var": lambda: stage_var(fast, strict),
        "forecast": lambda: stage_forecast(fast),
        "stress": stage_stress,
        "timesfm": lambda: stage_timesfm(fast),
        "report": stage_report,
    }
    stages = ["data", "var", "forecast", "stress", "report"] if stage == "all" else [stage]
    for s in stages:
        if s not in order:
            raise SystemExit(f"unknown stage {s!r}; choose from {', '.join(order)} or all")
        log.info("=== stage: %s ===", s)
        order[s]()
    update_manifest()


def cli() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m oilshock.pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "var", "forecast", "stress", "timesfm", "report", "all"],
    )
    ap.add_argument("--synthetic", action="store_true", help="use simulated data (demo mode only)")
    ap.add_argument("--fast", action="store_true", help="fewer replications, every third origin")
    ap.add_argument(
        "--no-strict", action="store_true", help="record failed checks instead of stopping"
    )
    a = ap.parse_args()
    main(a.stage, a.synthetic, a.fast, not a.no_strict)


if __name__ == "__main__":
    cli()
