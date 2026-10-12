"""Raw data acquisition (plan Section 5, stage 1). Never transforms data."""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from oilshock import config

log = logging.getLogger(__name__)


def fetch_fred(series_id: str, start: str | None = None) -> pd.Series:
    """Download one FRED series. Reads the API key from the ``FRED_API_KEY`` environment variable."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at https://fredaccount.stlouisfed.org/ "
            "and `export FRED_API_KEY=<key>` before running the pipeline."
        )
    from fredapi import Fred

    s = Fred(api_key=key).get_series(series_id, observation_start=start)
    s.name = series_id
    return s


def _write_raw(series: pd.Series, series_id: str, url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"fred_{series_id}.csv"
    series.rename_axis("date").rename("value").to_csv(path, header=True)
    config.write_json(
        {
            "source": "FRED",
            "series_id": series_id,
            "url": url,
            "retrieved_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "n_obs": int(series.notna().sum()),
            "first": str(series.dropna().index.min().date()) if series.notna().any() else None,
            "last": str(series.dropna().index.max().date()) if series.notna().any() else None,
        },
        path.with_suffix(".meta.json"),
    )
    return path


def fred_requests(cfg: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Every FRED series the config asks for: ``[{key, id, url, start}]``."""
    cfg = cfg or config.series_cfg()
    out: list[dict[str, str]] = []
    for s in cfg["series"]:
        for src in s.get("sources", []):
            if src.get("kind") == "fred":
                out.append(
                    {
                        "key": s["key"],
                        "id": src["id"],
                        "url": src.get("url", ""),
                        # two extra years so first-difference and y-o-y transforms exist at sample start
                        "start": str(int(str(s["use_from"])[:4]) - 2)
                        if s.get("use_from")
                        else None,
                    }
                )
    return out


def fetch_all(cfg: dict[str, Any] | None = None, raw_dir: Path | None = None) -> None:
    """Download every FRED series in ``series.yaml`` to ``data/raw/`` as CSV + ``.meta.json``."""
    raw_dir = raw_dir or config.paths().raw
    failures: list[str] = []
    for req in fred_requests(cfg):
        try:
            start = f"{req['start']}-01-01" if req["start"] else None
            s = fetch_fred(req["id"], start)
            _write_raw(s, req["id"], req["url"], raw_dir)
            log.info("fetched FRED %s (%d obs)", req["id"], s.notna().sum())
        except Exception as exc:  # keep going; report all failures together
            failures.append(f"{req['id']}: {exc}")
            log.error("FRED %s failed: %s", req["id"], exc)
    if failures:
        raise RuntimeError("FRED download failed for:\n  " + "\n  ".join(failures))


def register_manual(key: str, path: str | Path, url: str, date: str | None = None) -> None:
    """Record a manually downloaded file in ``data/raw/manual/README.md`` and a sidecar ``.meta.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    date = date or dt.date.today().isoformat()
    manual = config.paths().manual
    manual.mkdir(parents=True, exist_ok=True)
    readme = manual / "README.md"
    line = f"- {path.name} | {key} | {url} | downloaded {date}\n"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Manual downloads\n\n"
    text = text.replace("Nothing has been downloaded yet.\n", "")
    if line not in text:
        readme.write_text(text.rstrip("\n") + "\n" + line, encoding="utf-8")
    config.write_json(
        {"key": key, "file": path.name, "url": url, "downloaded": date},
        path.with_suffix(path.suffix + ".meta.json"),
    )


def read_fred_raw(series_id: str, raw_dir: Path | None = None) -> pd.Series:
    """Load a previously downloaded FRED CSV back as a Series."""
    raw_dir = raw_dir or config.paths().raw
    df = pd.read_csv(raw_dir / f"fred_{series_id}.csv", parse_dates=["date"], index_col="date")
    return df["value"].astype(float).rename(series_id)
