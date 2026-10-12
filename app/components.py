"""Shared UI pieces: cached artifact loaders, sidebar, caption helper (plan Section 10).

The app only reads files written by the offline pipeline and does light, instant calculations.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilshock import config  # noqa: E402

RUN_HINT = (
    "Build the artifacts first: `make all` (real data, needs a FRED key and the manual files in "
    "`data/raw/manual/`) or `make demo` then `make app-demo` for the SYNTHETIC demo."
)


# ------------------------------------------------------------------ mode (real vs synthetic demo)
def _artifacts_ready(demo: bool) -> bool:
    os_demo = os.environ.get("OILSHOCK_DEMO")
    try:
        os.environ["OILSHOCK_DEMO"] = "1" if demo else "0"
        return (config.paths().artifacts / "manifest.json").exists()
    finally:
        if os_demo is None:
            os.environ.pop("OILSHOCK_DEMO", None)
        else:
            os.environ["OILSHOCK_DEMO"] = os_demo


def ensure_mode() -> None:
    """Use ``demo/`` only if asked for (OILSHOCK_DEMO=1) or if there are no real artifacts at all."""
    if os.environ.get("OILSHOCK_DEMO") == "1":
        return
    if not _artifacts_ready(False) and _artifacts_ready(True):
        os.environ["OILSHOCK_DEMO"] = "1"


def is_demo() -> bool:
    return config.paths().demo


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def load_artifact(name: str, mode: str = "") -> pd.DataFrame | dict | None:
    """Load ``artifacts/<name>`` (Parquet or JSON). Returns None if it does not exist."""
    path = config.paths().artifacts / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return config.read_json(path)
    return config.read_parquet(path)


def artifact(name: str) -> pd.DataFrame | dict | None:
    return load_artifact(name, "demo" if is_demo() else "real")


@st.cache_resource(show_spinner=False)
def load_pickle(name: str, mode: str = ""):
    path = config.paths().artifacts / name
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pickled(name: str):
    return load_pickle(name, "demo" if is_demo() else "real")


@st.cache_data(show_spinner=False)
def _panels(mode: str = "") -> tuple[pd.DataFrame, pd.DataFrame] | None:
    p = config.paths().processed
    m, q = p / "monthly_panel.parquet", p / "quarterly_panel.parquet"
    if not (m.exists() and q.exists()):
        return None
    return config.read_parquet(m), config.read_parquet(q)


def panels() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    return _panels("demo" if is_demo() else "real")


def require(obj, what: str):
    """Show a friendly message and stop the page when an artifact is missing."""
    if obj is None:
        st.info(f"{what} is not available yet. {RUN_HINT}")
        st.stop()
    return obj


# ------------------------------------------------------------------ page chrome
def page_setup(title: str, icon: str = "🛢️") -> dict:
    """Page config, sidebar and banners. Returns the manifest (empty dict if absent)."""
    ensure_mode()
    st.set_page_config(
        page_title=f"{title} · India oil-shock dashboard", page_icon=icon, layout="wide"
    )
    manifest = artifact("manifest.json") or {}
    sidebar(manifest)
    if is_demo():
        st.warning(
            "SYNTHETIC DEMO DATA: the numbers on this page come from a simulation used to exercise the "
            "code. They say nothing about India and must not be presented as findings.",
            icon="⚠️",
        )
    st.title(title)
    return manifest


def window_options(model: str = "A") -> dict[str, str]:
    """Window label -> key for windows whose IRF artifact exists."""
    wins = config.models_cfg()["windows"]
    return {
        w["label"]: k
        for k, w in wins.items()
        if artifact(f"var/irf_{model}_{k}.parquet") is not None
    }


def sidebar(manifest: dict) -> None:
    cfg = config.models_cfg()["scenario"]
    st.session_state.setdefault("shock_pct", cfg["shock_default"])
    st.session_state.setdefault("window", "full")
    st.session_state.setdefault("band", 90)
    with st.sidebar:
        st.header("Scenario")
        st.slider(
            "Oil price shock (%)",
            min_value=cfg["shock_min"],
            max_value=cfg["shock_max"],
            step=cfg["shock_step"],
            key="shock_pct",
            help="A permanent one-off change in the Brent price. Shared across pages.",
        )
        labels = {
            w["label"]: k for k, w in config.models_cfg()["windows"].items() if not w.get("split")
        }
        st.selectbox(
            "Model window",
            options=list(labels.values()),
            format_func=lambda k: config.models_cfg()["windows"][k]["label"],
            key="window",
            help="Estimate on all data, or only on data before an episode (what a forecaster could have known).",
        )
        st.radio("Band level", [68, 90], key="band", horizontal=True, format_func=lambda b: f"{b}%")
        st.divider()
        if manifest:
            st.caption(f"Data cut-off: **{manifest.get('data_cutoff', 'n/a')}**")
            st.caption(f"Git commit: `{manifest.get('git_commit', 'n/a')}`")
            st.caption(f"Built: {str(manifest.get('build_time_utc', 'n/a'))[:16]} UTC")
        else:
            st.caption("No build manifest found.")


def shock_slider() -> float:
    """Current shock size (%) from the shared sidebar slider."""
    return float(
        st.session_state.get("shock_pct", config.models_cfg()["scenario"]["shock_default"])
    )


def episode_picker(key: str = "episode", default: str | None = None) -> str:
    names = config.episode_list()
    defs = config.episode_defs()
    return st.selectbox(
        "Episode",
        names,
        index=names.index(default) if default in names else 0,
        format_func=lambda n: defs[n]["label"],
        key=key,
    )


def source_caption(keys: list[str], text: str = "") -> None:
    """One-line caption under a chart: what it shows, units and sources (read from ``series.yaml``)."""
    src = config.source_line(keys)
    st.caption(" ".join(x for x in (text, src) if x))


def pp(x: float, digits: int = 2, signed: bool = True) -> str:
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
