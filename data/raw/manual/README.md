# Manual downloads (RBI DBIE, MOSPI, PPAC, DPIIT)

Files from portals without a stable API are saved here exactly as exported, never edited.
Add one line per file: URL, table name, download date. `oilshock.data.parsers` asserts the expected
header names, so a changed layout fails loudly instead of silently mis-parsing.

**Expected layout.** Each parser expects a tidy sheet (first sheet, header in row 1) with the columns
listed below. If a portal export differs, re-save it into this layout (copying values only) and keep the
original download alongside it as `<name>.original.xlsx`.

| File | Columns | Parser |
| --- | --- | --- |
| `rbi_repo.xlsx` | `Date`, `Repo Rate` (one row per rate-change date) | `parse_rbi_repo` |
| `rbi_wacr.xlsx` | `Date`, `WACR` | `parse_rbi_wacr` |
| `rbi_bop.xlsx` | `Quarter` (e.g. `2012Q2` or a date), `Current Account Balance` (US$ mn) | `parse_rbi_bop` |
| `mospi_cpi_2024_back.xlsx` | `Date`, `General Index`, optional `Fuel and Light`, `Core` | `parse_mospi_cpi(path, base=2024)` |
| `mospi_gdp_2011.xlsx`, `mospi_gdp_2022.xlsx` | `Quarter`, `Real GDP Growth` (y-o-y %) | `parse_mospi_gdp(path, base)` |
| `mospi_gdp_nominal.xlsx` | `Quarter`, `Nominal GDP` (INR crore) | `parse_mospi_nominal_gdp` |
| `mospi_iip.xlsx` | `Date`, `General Index` | `parse_mospi_iip` |
| `eoa_wpi_fuel.xlsx` | `Date`, `Fuel and Power Index` | `parse_wpi_fuel` |
| `ppac_crude_imports.xlsx` | `Date`, `Crude Imports` (million tonnes) | `parse_ppac_imports` |
| `ppac_excise.xlsx` | `Date`, `Petrol`, `Diesel` (INR/litre) | `parse_ppac_excise` |

## Download log

<!-- Example: - rbi_repo.xlsx | https://data.rbi.org.in/ | Policy rates table | downloaded 2026-10-12 -->

Nothing has been downloaded yet.

Register a file programmatically with `oilshock.data.fetch.register_manual(key, path, url, date)`.
