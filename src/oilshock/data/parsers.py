"""One parser per manual file layout (plan Section 6, ``data/parsers.py``).

Every parser asserts the expected header names, so a changed download fails loudly. See
``data/raw/manual/README.md`` for the tidy layout each parser expects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oilshock.data.transform import to_period_end


class LayoutError(ValueError):
    """The file does not have the expected headers."""


def _read(path: str | Path, expected: list[str], optional: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, sheet_name=0)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise LayoutError(f"{path.name}: unsupported file type {path.suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise LayoutError(
            f"{path.name}: expected columns {expected}, found {list(df.columns)}; missing {missing}. "
            "Re-save the download into the layout documented in data/raw/manual/README.md."
        )
    keep = expected + [c for c in (optional or []) if c in df.columns]
    return df[keep].dropna(how="all")


def _quarter_index(col: pd.Series) -> pd.DatetimeIndex:
    """Parse ``2012Q2`` style strings or real dates to quarter-end timestamps."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return to_period_end(col, "Q")
    return pd.DatetimeIndex(
        [pd.Period(str(v).replace(" ", ""), "Q").end_time.normalize() for v in col]
    )


def _monthly(df: pd.DataFrame, col: str, value_col: str, name: str) -> pd.Series:
    idx = to_period_end(df[col], "M")
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=idx, name=name)
    if s.index.has_duplicates:
        raise LayoutError(f"{name}: duplicate months after mapping to period end")
    return s.sort_index()


def parse_rbi_repo(path: str | Path) -> pd.Series:
    """RBI repo rate *change dates* -> event Series (index = date, not yet a monthly step)."""
    df = _read(path, ["Date", "Repo Rate"])
    s = pd.Series(
        pd.to_numeric(df["Repo Rate"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["Date"]),
        name="repo",
    ).sort_index()
    if s.isna().any():
        raise LayoutError("rbi_repo: non-numeric repo rate found")
    return s


def parse_rbi_wacr(path: str | Path) -> pd.Series:
    df = _read(path, ["Date", "WACR"])
    return _monthly(df, "Date", "WACR", "wacr")


def parse_rbi_bop(path: str | Path) -> pd.DataFrame:
    """Quarterly balance of payments -> DataFrame with ``cab`` (current account, US$ mn)."""
    df = _read(path, ["Quarter", "Current Account Balance"])
    out = pd.DataFrame(
        {"cab": pd.to_numeric(df["Current Account Balance"], errors="coerce").to_numpy()},
        index=_quarter_index(df["Quarter"]),
    ).sort_index()
    return out


def parse_mospi_cpi(path: str | Path, base: int) -> pd.Series:
    """MOSPI CPI general index (levels) for the given base year. Subgroups: see ``parse_mospi_cpi_groups``."""
    df = _read(path, ["Date", "General Index"], optional=["Fuel and Light", "Core"])
    return _monthly(df, "Date", "General Index", f"cpi_{base}")


def parse_mospi_cpi_groups(path: str | Path) -> pd.DataFrame:
    """Optional subgroup indices (levels): ``cpi_fuel`` and ``cpi_core`` when present."""
    df = _read(path, ["Date", "General Index"], optional=["Fuel and Light", "Core"])
    out: dict[str, pd.Series] = {}
    if "Fuel and Light" in df.columns:
        out["cpi_fuel"] = _monthly(df, "Date", "Fuel and Light", "cpi_fuel")
    if "Core" in df.columns:
        out["cpi_core"] = _monthly(df, "Date", "Core", "cpi_core")
    return pd.DataFrame(out)


def parse_mospi_gdp(path: str | Path, base: str) -> pd.Series:
    """MOSPI real GDP y-o-y growth (%) for the given base (``"2011-12"`` or ``"2022-23"``)."""
    df = _read(path, ["Quarter", "Real GDP Growth"])
    return pd.Series(
        pd.to_numeric(df["Real GDP Growth"], errors="coerce").to_numpy(),
        index=_quarter_index(df["Quarter"]),
        name=f"gdp_growth_{base}",
    ).sort_index()


def parse_mospi_nominal_gdp(path: str | Path) -> pd.Series:
    """Quarterly nominal GDP, INR crore."""
    df = _read(path, ["Quarter", "Nominal GDP"])
    return pd.Series(
        pd.to_numeric(df["Nominal GDP"], errors="coerce").to_numpy(),
        index=_quarter_index(df["Quarter"]),
        name="gdp_nominal",
    ).sort_index()


def parse_mospi_iip(path: str | Path) -> pd.Series:
    df = _read(path, ["Date", "General Index"])
    return _monthly(df, "Date", "General Index", "iip")


def parse_wpi_fuel(path: str | Path) -> pd.Series:
    df = _read(path, ["Date", "Fuel and Power Index"])
    return _monthly(df, "Date", "Fuel and Power Index", "wpi_fuel")


def parse_ppac_imports(path: str | Path) -> pd.Series:
    """Monthly crude import volume, million tonnes."""
    df = _read(path, ["Date", "Crude Imports"])
    return _monthly(df, "Date", "Crude Imports", "crude_imports")


def parse_ppac_excise(path: str | Path) -> pd.DataFrame:
    """Central excise on petrol and diesel (INR/litre) as an event table."""
    df = _read(path, ["Date", "Petrol", "Diesel"])
    out = pd.DataFrame(
        {
            "petrol": pd.to_numeric(df["Petrol"], errors="coerce").to_numpy(),
            "diesel": pd.to_numeric(df["Diesel"], errors="coerce").to_numpy(),
        },
        index=pd.to_datetime(df["Date"]),
    ).sort_index()
    return out
