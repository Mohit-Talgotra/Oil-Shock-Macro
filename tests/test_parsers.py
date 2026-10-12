import pandas as pd
import pytest

from helpers import FIXTURES
from oilshock.data import parsers


def test_repo_parser_reads_change_dates():
    s = parsers.parse_rbi_repo(FIXTURES / "rbi_repo.csv")
    assert s.iloc[0] == 6.0 and len(s) == 5 and s.index.is_monotonic_increasing


def test_wrong_headers_fail_loudly():
    with pytest.raises(parsers.LayoutError):
        parsers.parse_rbi_repo(FIXTURES / "bad_layout.csv")


def test_cpi_parser_maps_to_period_end_and_groups():
    s = parsers.parse_mospi_cpi(FIXTURES / "mospi_cpi_2024_back.csv", base=2024)
    assert s.index[0] == pd.Timestamp("2013-01-31") and s.name == "cpi_2024"
    g = parsers.parse_mospi_cpi_groups(FIXTURES / "mospi_cpi_2024_back.csv")
    assert list(g.columns) == ["cpi_fuel", "cpi_core"]


def test_quarterly_parsers():
    bop = parsers.parse_rbi_bop(FIXTURES / "rbi_bop.csv")
    assert bop.index[0] == pd.Timestamp("2012-06-30") and bop["cab"].iloc[0] == -16000
    gdp = parsers.parse_mospi_gdp(FIXTURES / "mospi_gdp_2011.csv", "2011-12")
    assert gdp.iloc[0] == 5.6 and gdp.index[1] == pd.Timestamp("2012-09-30")


def test_monthly_import_volume():
    s = parsers.parse_ppac_imports(FIXTURES / "ppac_crude_imports.csv")
    assert list(s.index) == [pd.Timestamp("2024-04-30"), pd.Timestamp("2024-05-31")]
