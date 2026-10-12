"""Configuration loading, paths and constants (plan Section 6).

All paths are resolved at call time so tests can relocate the project with ``OILSHOCK_ROOT`` and the
synthetic demo can write to ``./demo`` with ``OILSHOCK_DEMO=1`` without touching real data/artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)
QCOLS: list[str] = ["q05", "q10", "q50", "q90", "q95"]
QMAP: dict[str, float] = dict(zip(QCOLS, QUANTILES, strict=True))

FORECAST_COLUMNS = [
    "origin",
    "target_date",
    "horizon",
    "q05",
    "q10",
    "q50",
    "q90",
    "q95",
    "actual",
    "oil_assumption",
]


@dataclass(frozen=True)
class Paths:
    root: Path
    base: Path
    raw: Path
    manual: Path
    interim: Path
    processed: Path
    artifacts: Path
    var: Path
    forecasts: Path
    stress: Path
    reports: Path
    figures: Path
    demo: bool


def is_demo() -> bool:
    return os.environ.get("OILSHOCK_DEMO") == "1"


def paths() -> Paths:
    """Project paths. ``OILSHOCK_ROOT`` relocates everything; ``OILSHOCK_DEMO=1`` uses ``<root>/demo``."""
    root = Path(os.environ.get("OILSHOCK_ROOT", ROOT))
    demo = is_demo()
    base = root / "demo" if demo else root
    return Paths(
        root=root,
        base=base,
        raw=root / "data" / "raw",
        manual=root / "data" / "raw" / "manual",
        interim=base / "data" / "interim",
        processed=base / "data" / "processed",
        artifacts=base / "artifacts",
        var=base / "artifacts" / "var",
        forecasts=base / "artifacts" / "forecasts",
        stress=base / "artifacts" / "stress",
        reports=base / "reports",
        figures=base / "reports" / "figures",
        demo=demo,
    )


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def series_cfg() -> dict[str, Any]:
    return load_yaml("series.yaml")


@cache
def models_cfg() -> dict[str, Any]:
    return load_yaml("models.yaml")


@cache
def episodes_cfg() -> dict[str, Any]:
    return load_yaml("episodes.yaml")


@cache
def events_cfg() -> dict[str, Any]:
    return load_yaml("events.yaml")


def series_by_key() -> dict[str, dict[str, Any]]:
    return {s["key"]: s for s in series_cfg()["series"]}


def source_line(keys: list[str]) -> str:
    """Caption source line built from ``series.yaml`` (every chart caption uses this)."""
    sbk = series_by_key()
    parts: list[str] = []
    for key in keys:
        s = sbk.get(key)
        if s is None:
            continue
        src = s["sources"][0]
        ident = src.get("id") or src.get("file") or src.get("formula", "")
        pub = src.get("publisher", "")
        parts.append(f"{s['name']} ({pub}{', ' if pub and ident else ''}{ident})")
    return "Source: " + "; ".join(parts) if parts else ""


def episode_list() -> list[str]:
    return list(episodes_cfg()["order"])


def episode_defs() -> dict[str, dict[str, Any]]:
    return episodes_cfg()["episodes"]


def window_end(window: str) -> pd.Timestamp | None:
    end = models_cfg()["windows"][window].get("end")
    return pd.Timestamp(end) if end else None


# ------------------------------------------------------------------ IO helpers
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
