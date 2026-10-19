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

- rbi_repo.xlsx | repo | https://data.rbi.org.in/DBIE/ + https://dbie.rbihub.in (DBIE 531 key rates) | downloaded 2026-10-19
- rbi_wacr.xlsx | wacr | https://data.rbi.org.in/DBIE/ daily WACR (DBIE 198) averaged to month-end; original kept as rbi_wacr.original.xlsx | downloaded 2026-10-19
- rbi_bop.xlsx | cab | https://data-api.dbie.rbihub.in (table external_sector.ind_ovr_bop_rn, 142 qtrs 1990Q2-2025Q3, USD/1e6) | downloaded 2026-10-19
- mospi_cpi_2024_back.xlsx | cpi | https://esankhyiki.mospi.gov.in/macroindicators?product=cpi CPI 2024=100 back series Jan-2013-Dec-2024 (General only; Fuel/Core unavailable); press note 12-Feb-2026 | downloaded 2026-10-19
- mospi_iip.xlsx | iip | https://esankhyiki.mospi.gov.in/ IIP 2011-12 General index Apr-2012-Mar-2026 | downloaded 2026-10-19
- mospi_gdp_2011.xlsx | gdp_growth | https://data-api.dbie.rbihub.in/api/tables/real_sector/qtr_gdp_mrkt_prc_rn BY_2011_12 constant-price GDP y-o-y (2012Q2-2025Q3); MOSPI via RBI DBIE mirror | downloaded 2026-10-19
- mospi_gdp_2022.xlsx | gdp_growth | https://www.mospi.gov.in/ press note 27-Feb-2026 new 2022-23 base Statement 6A (2022Q2-2023Q1 NaN, growth from 2023Q2) | downloaded 2026-10-19
- mospi_gdp_nominal.xlsx | gdp_nominal | https://data-api.dbie.rbihub.in BY_2004_05+BY_2011_12 level-spliced at 2011Q2 (x0.9880) + MOSPI press note Statement 7A nominal from 2022Q2 (splice x0.9359); 2004Q2-2025Q4 | downloaded 2026-10-19
- eoa_wpi_fuel.xlsx | wpi_fuel | https://www.eaindustry.nic.in/download_data_1112.asp monthly_index_202606.xls Fuel & Power Apr-2012-Apr-2026 | downloaded 2026-10-19
- ppac_crude_imports.xlsx | crude_imports | https://ppac.gov.in/import-export/history Import_Qty_H.xlsx TMT/1000 Apr-2012-Mar-2026 | downloaded 2026-10-19
- ppac_excise.xlsx | excise | https://ppac.gov.in/prices/central-excise-and-customs-rate-on-major-petroleum-products tariff 27-03-2026 + Ready Reckoner FY25-26 Table 8.6 (spot 11.90/7.80 only; full history TODO) | downloaded 2026-10-19
- FRED fred_*.csv | brent_usd, brent_daily, inr_usd, cpi(old), gdp(old), fedfunds, vix | https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID> no-key endpoint | downloaded 2026-10-19

Limitations: WACR daily history starts Jul-2020 (daily original kept as rbi_wacr.original.xlsx); IIP starts Apr-2012 (sample starts 2004, early NaN dropped); CPI back has General only (no Fuel/Core); GDP-2022 first 4 quarters NaN (no y-o-y in new base); excise is 2-row spot; nominal pre-2011 is level-spliced. Raw portal dumps retained under /tmp/pi-subagents/1792426810-oilshock/downloads. Add 5 spot-check values per source to config/series.yaml before `make all`.

Register a file programmatically with `oilshock.data.fetch.register_manual(key, path, url, date)`.
