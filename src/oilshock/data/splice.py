"""Growth-rate splicing and the overlap report (plan Section 4, "Splicing rules")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# No genuine y-o-y or m-o-m growth rate (in %) has a median anywhere near this; index levels do.
LEVEL_MEDIAN_THRESHOLD = 50.0


def _assert_growth(s: pd.Series, name: str) -> None:
    kind = s.attrs.get("kind")
    if kind == "level":
        raise ValueError(f"{name}: refusing to splice index levels; convert to growth rates first")
    x = s.dropna()
    if len(x) and float(x.abs().median()) > LEVEL_MEDIAN_THRESHOLD:
        raise ValueError(
            f"{name}: values look like index levels (median |x| = {x.abs().median():.1f}); "
            "splice growth rates, never levels"
        )


def splice_growth(old: pd.Series, new: pd.Series, switch: str | pd.Timestamp) -> pd.Series:
    """Join two *growth-rate* series: ``old`` before ``switch``, ``new`` from ``switch`` onward.

    Refuses to splice levels (rule 1). Where ``new`` is missing after the switch the gap stays NaN
    rather than silently falling back to ``old``.
    """
    _assert_growth(old, "old")
    _assert_growth(new, "new")
    sw = pd.Timestamp(switch)
    idx = old.index.union(new.index).sort_values()
    head = old.reindex(idx)[idx < sw]
    tail = new.reindex(idx)[idx >= sw]
    out = pd.concat([head, tail]).sort_index()
    out.attrs["kind"] = "growth"
    out.attrs["switch"] = str(sw.date())
    return out.rename(new.name or old.name)


def overlap_report(old: pd.Series, new: pd.Series, name: str = "") -> pd.DataFrame:
    """Gap between old and new growth rates on their overlap (one-row DataFrame)."""
    _assert_growth(old, "old")
    _assert_growth(new, "new")
    both = pd.concat([old.rename("old"), new.rename("new")], axis=1).dropna()
    if both.empty:
        return pd.DataFrame(
            [
                {
                    "series": name,
                    "n_overlap": 0,
                    "overlap_start": None,
                    "overlap_end": None,
                    "mean_gap": np.nan,
                    "mean_abs_gap": np.nan,
                    "max_abs_gap": np.nan,
                }
            ]
        )
    gap = both["new"] - both["old"]
    return pd.DataFrame(
        [
            {
                "series": name,
                "n_overlap": int(len(both)),
                "overlap_start": both.index.min().date().isoformat(),
                "overlap_end": both.index.max().date().isoformat(),
                "mean_gap": float(gap.mean()),
                "mean_abs_gap": float(gap.abs().mean()),
                "max_abs_gap": float(gap.abs().max()),
            }
        ]
    )


def write_overlap_report(
    rows: list[pd.DataFrame], path: Path, threshold: float = 0.5
) -> pd.DataFrame:
    """Write ``reports/splice_check.csv`` and flag series whose mean |gap| exceeds ``threshold`` pp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df.empty:
        df["mention_on_about_page"] = df["mean_abs_gap"] > threshold
    df.to_csv(path, index=False)
    return df
