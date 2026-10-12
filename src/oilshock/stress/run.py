"""Layer 3: pre-shock training, blind vs oracle-oil forecasts, scoring and policy gap (plan Section 9).

Procedure per episode and model: cut the panel at the training end, fit with the Layer 2 settings,
forecast every period of the test window from the single frozen origin, then score against what happened.

* ``blind``: oil stays flat at its last pre-episode value.
* ``oracle``: the realised oil path is supplied (only models with covariates: ARIMAX, VAR, XGBoost,
  TimesFM with covariates).

The live ``westasia_2026`` blind forecasts are written once to ``frozen_westasia_2026.parquet`` and
never regenerated; later runs only re-attach actuals and re-score.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from oilshock import config
from oilshock.forecast.backtest import make_forecaster
from oilshock.forecast.base import (
    Forecaster,
    TargetSpec,
    future_dates,
    get_target,
    panel_for,
    target_names,
    target_values,
)
from oilshock.models import policy
from oilshock.stress.episodes import (
    Episode,
    all_episodes,
    get_episode,
    horizon_for,
    months_per_period,
    origin_for,
)

log = logging.getLogger(__name__)

STRESS_COLUMNS = [
    "model",
    "target",
    "info_set",
    "date",
    "q10",
    "q50",
    "q90",
    "actual",
    "abs_err",
    "inside80",
]
SUMMARY_COLUMNS = [
    "episode",
    "model",
    "target",
    "info_set",
    "rmse",
    "bias",
    "coverage80",
    "shock_penalty",
    "months_to_surprise",
    "policy_gap_bps",
]
FROZEN_FILE = "frozen_westasia_2026.parquet"

Factory = Callable[[str, TargetSpec], Forecaster]


def _oracle_oil(panel: pd.DataFrame, origin: pd.Timestamp, spec: TargetSpec, h: int) -> np.ndarray:
    """Realised oil changes after the origin (zeros where not yet observed)."""
    after = panel.loc[panel.index > origin, "oil"].dropna().to_numpy(dtype=float)[:h]
    return np.concatenate([after, np.zeros(h - len(after))])


def _score_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_err"] = (df["actual"] - df["q50"]).abs()
    inside = (df["actual"] >= df["q10"]) & (df["actual"] <= df["q90"])
    df["inside80"] = inside.where(df["actual"].notna())
    return df


def attach_actuals(df: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """(Re)compute ``actual`` for every row from the current panels, keyed by (target, origin, horizon)."""
    out = df.copy()
    actual = []
    for _, r in out.iterrows():
        spec = get_target(r["target"])
        z = panels[spec.freq][spec.z]
        tv = target_values(z, int(r["horizon"]), spec.kind)
        actual.append(tv.get(pd.Timestamp(r["origin"]), np.nan))
    out["actual"] = actual
    return _score_columns(out)


def run_episode(
    episode: str | Episode,
    models: list[str],
    info_set: str,
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    targets: list[str] | None = None,
    factory: Factory = make_forecaster,
) -> pd.DataFrame:
    """Forecast the episode window from the frozen origin. One row per model, target and period.

    Returns every quantile (``q05``..``q95``), ``origin``, ``horizon`` and ``actual`` so scoring can use the
    95% band; ``write`` keeps only the artifact columns.
    """
    ep = get_episode(episode) if isinstance(episode, str) else episode
    if info_set not in {"blind", "oracle"}:
        raise ValueError("info_set must be 'blind' or 'oracle'")
    frames = []
    for tname in targets or target_names():
        spec = get_target(tname)
        panel = panel_for(spec, monthly, quarterly)
        origin = origin_for(ep, spec, panel.index)
        hist = panel.loc[:origin]
        assert hist.index.max() <= ep.train_end, (
            "stress training data extends past the training end"
        )
        y = hist[spec.z].dropna()
        if y.empty or y.index.max() != origin:
            log.warning(
                "%s/%s: model series not observed at %s; skipped", ep.name, tname, origin.date()
            )
            continue
        h = horizon_for(ep, spec, origin, panel.index.max())
        oil = _oracle_oil(panel, origin, spec, h) if info_set == "oracle" else None
        x_future = None if oil is None else pd.DataFrame({"oil": oil})
        realised = {
            k: target_values(panel[spec.z], k, spec.kind).get(origin, np.nan)
            for k in range(1, h + 1)
        }
        for mname in models:
            fc_model = factory(mname, spec)
            if info_set == "oracle" and not fc_model.supports_covariates:
                continue
            try:
                fc_model.fit(y, hist)
                fc = fc_model.predict(h, x_future)
            except Exception as exc:  # one failing model must not sink the episode
                log.error("%s/%s/%s/%s failed: %s", ep.name, tname, mname, info_set, exc)
                continue
            dates = future_dates(origin, h, spec.alias)
            part = fc.reset_index()
            part.insert(0, "model", mname)
            part.insert(1, "target", tname)
            part.insert(2, "info_set", info_set)
            part.insert(3, "origin", origin)
            part.insert(4, "date", dates)
            part["actual"] = [realised[int(k)] for k in part["horizon"]]
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return _score_columns(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------- scoring
def policy_gap(
    forecast_pi: pd.Series, actual_pi: pd.Series, rule: dict | None = None, gap: float = 0.0
) -> pd.Series:
    """Gap in basis points between the Taylor-rule rate implied by forecast and by actual inflation.

    Signed: positive means the forecast-based rule would have set the rate *higher* than the
    actual-based one. The output-gap term cancels when the same gap is used on both sides.
    """
    rule = rule or {}
    i_fc = policy.taylor_rate(forecast_pi, gap=gap, **rule)
    i_ac = policy.taylor_rate(actual_pi, gap=gap, **rule)
    return ((i_fc - i_ac) * 100.0).rename("policy_gap_bps")


def calm_mse_table(forecast_dir: Path) -> dict[tuple[str, str], pd.Series]:
    """Mean squared error of the median by horizon over calm target dates, from the backtest files."""
    from oilshock.forecast.evaluate import episode_mask

    out: dict[tuple[str, str], pd.Series] = {}
    for f in sorted(Path(forecast_dir).glob("*_*.parquet")):
        if f.stem.startswith(("metrics", "fitted", "xgb_grid", "shap", "grid")):
            continue
        model, _, target = f.stem.rpartition("_")
        if target not in target_names():
            continue
        d = config.read_parquet(f)
        d = d[d["actual"].notna()].copy()
        if d.empty:
            continue
        d = d[~episode_mask(pd.to_datetime(d["target_date"]))]
        if d.empty:
            continue
        out[(model, target)] = ((d["actual"] - d["q50"]) ** 2).groupby(d["horizon"]).mean()
    return out


def summarise(
    results: pd.DataFrame,
    episode: str,
    calm: dict[tuple[str, str], pd.Series] | None = None,
    rule: dict | None = None,
) -> pd.DataFrame:
    """Scores per model, target and information set (the ``summary.parquet`` schema)."""
    rows = []
    calm = calm or {}
    for (model, target, info), g in results.groupby(["model", "target", "info_set"]):
        d = g.dropna(subset=["actual"])
        if d.empty:
            continue
        spec = get_target(target)
        err = d["actual"] - d["q50"]
        rmse = float(np.sqrt(np.mean(err**2)))
        outside = (d["actual"] < d["q05"]) | (d["actual"] > d["q95"])
        mts = (
            int(d.loc[outside, "horizon"].min()) * months_per_period(spec)
            if outside.any()
            else pd.NA
        )
        cm = calm.get((model, target))
        if cm is not None and len(cm):
            mse = d["horizon"].map(cm).dropna()
            calm_rmse = float(np.sqrt(mse.mean())) if len(mse) else np.nan
            penalty = rmse / calm_rmse if calm_rmse and calm_rmse > 0 else np.nan
        else:
            penalty = np.nan
        gap = np.nan
        if target == "cpi":
            gap = float(policy_gap(d["q50"], d["actual"], rule).abs().mean())
        rows.append(
            {
                "episode": episode,
                "model": model,
                "target": target,
                "info_set": info,
                "rmse": rmse,
                "bias": float(err.mean()),
                "coverage80": float(d["inside80"].astype(float).mean()),
                "shock_penalty": penalty,
                "months_to_surprise": mts,
                "policy_gap_bps": gap,
            }
        )
    out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    out["months_to_surprise"] = out["months_to_surprise"].astype("Int64")
    return out


def _artifact_view(df: pd.DataFrame) -> pd.DataFrame:
    return df[STRESS_COLUMNS].copy()


# ---------------------------------------------------------------------------- orchestration
def run_all(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    models: list[str] | None = None,
    out_dir: Path | None = None,
    forecast_dir: Path | None = None,
    factory: Factory = make_forecaster,
) -> pd.DataFrame:
    """Run all four episodes, write ``stress/{episode}.parquet`` and ``summary.parquet``."""
    p = config.paths()
    out_dir = Path(out_dir or p.stress)
    forecast_dir = Path(forecast_dir or p.forecasts)
    cfg = config.models_cfg()
    models = models or list(cfg["stress"]["models"])
    cov_models = set(cfg["stress"]["covariate_models"])
    panels = {"M": monthly, "Q": quarterly}
    rule = {k: cfg["policy"][k] for k in ("pi_target", "r_star", "a_pi", "a_y")}
    calm = calm_mse_table(forecast_dir) if forecast_dir.exists() else {}
    summaries = []
    for ep in all_episodes():
        blind = _blind_forecasts(ep, models, monthly, quarterly, out_dir, panels, factory)
        oracle = run_episode(
            ep,
            [m for m in models if m in cov_models],
            "oracle",
            monthly,
            quarterly,
            factory=factory,
        )
        res = pd.concat([x for x in (blind, oracle) if not x.empty], ignore_index=True)
        if res.empty:
            continue
        config.write_parquet(_artifact_view(res), out_dir / f"{ep.name}.parquet")
        summaries.append(summarise(res, ep.name, calm, rule))
    summary = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )
    config.write_parquet(summary, out_dir / "summary.parquet")
    return summary


def _blind_forecasts(ep, models, monthly, quarterly, out_dir, panels, factory) -> pd.DataFrame:
    if not ep.live:
        return run_episode(ep, models, "blind", monthly, quarterly, factory=factory)
    frozen = Path(out_dir) / FROZEN_FILE
    if frozen.exists():
        log.info("using frozen %s forecasts from %s (not regenerated)", ep.name, frozen)
        fz = config.read_parquet(frozen)
        fz["date"] = pd.to_datetime(fz["date"])
        fz["origin"] = pd.to_datetime(fz["origin"])
        return attach_actuals(fz, panels)
    fresh = run_episode(ep, models, "blind", monthly, quarterly, factory=factory)
    if fresh.empty:
        return fresh
    fz = fresh.drop(columns=["actual", "abs_err", "inside80"])
    config.write_parquet(fz, frozen)
    log.warning(
        "FROZEN the %s blind forecasts at %s. Commit them and tag the commit '%s'; never regenerate.",
        ep.name,
        frozen,
        config.models_cfg()["stress"]["freeze_tag"],
    )
    return attach_actuals(fz, panels)
