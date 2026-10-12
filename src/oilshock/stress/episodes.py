"""Episode definitions from ``config/episodes.yaml`` (plan Section 9)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from oilshock import config
from oilshock.forecast.base import TargetSpec


@dataclass(frozen=True)
class Episode:
    name: str
    label: str
    short: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None
    direction: str
    live: bool
    color: str
    caution: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def get_episode(name: str) -> Episode:
    e = config.episode_defs()[name]
    return Episode(
        name=name,
        label=e["label"],
        short=e.get("short", e["label"]),
        train_end=pd.Timestamp(e["train_end"]),
        test_start=pd.Timestamp(e["test_start"]),
        test_end=pd.Timestamp(e["test_end"]) if e.get("test_end") else None,
        direction=e["direction"],
        live=bool(e.get("live", False)),
        color=e.get("color", "#888888"),
        caution=e.get("caution", ""),
        notes=tuple(e.get("notes", [])),
    )


def all_episodes() -> list[Episode]:
    return [get_episode(n) for n in config.episode_list()]


def origin_for(ep: Episode, spec: TargetSpec, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Last panel date at or before the episode's training end (quarter-end for quarterly targets)."""
    valid = index[index <= ep.train_end]
    if len(valid) == 0:
        raise ValueError(f"no data before {ep.train_end.date()}")
    return valid.max()


def horizon_for(
    ep: Episode, spec: TargetSpec, origin: pd.Timestamp, last_date: pd.Timestamp
) -> int:
    """Number of periods to forecast from the frozen origin (capped at ``stress.max_horizon`` months)."""
    cap_months = int(config.models_cfg()["stress"]["max_horizon"])
    end = ep.test_end
    if ep.live or end is None:
        months = cap_months  # live episode: forecast the full capped horizon, score as data arrive
    else:
        months = (end.year - origin.year) * 12 + (end.month - origin.month)
    months = min(months, cap_months)
    return max(1, months if spec.freq == "M" else months // 3)


def months_per_period(spec: TargetSpec) -> int:
    return 1 if spec.freq == "M" else 3
