# India Oil-Shock Dashboard

*When forecasts meet shocks: how an oil price shock moves India's economy, and where models break.*

A Streamlit dashboard in three layers, built from an offline pipeline whose outputs are committed as artifacts:

1. **Transmission**: recursive VARs (oil ordered first) give bootstrap impulse responses, variance decompositions, rolling pass-through and a shock-size scenario engine (monthly: oil, IP, CPI, repo, INR; quarterly: oil, GDP growth, CAB/GDP, CPI, repo, INR).
2. **Forecasts**: naive, seasonal naive, AR, ARIMAX, VAR and quantile XGBoost (TimesFM optional), scored in a rolling-origin backtest from 2012 (RMSE, MAE, MASE, 80/95% coverage, pinball, Diebold-Mariano).
3. **Stress test**: models frozen before the 2008, 2014-16, 2022 and 2026 West Asia episodes forecast *blind* (flat oil) and with *oracle* oil, scored by shock penalty, months-to-surprise and the Taylor-rule policy gap.

Pages: Home, Transmission, Forecasts, Stress Test, Policy Lab, Data Explorer, About.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements-dev.txt
make fmt && make test

# no data needed: a clearly labelled SYNTHETIC demo
make demo && make app-demo

# real results
export FRED_API_KEY=...            # free key from fred.stlouisfed.org
# put the manual files in data/raw/manual/ (layouts in data/raw/manual/README.md)
make all && make app
```

`make all` runs `data → var → forecast → stress → report`; each stage is also available alone
(`python -m oilshock.pipeline var --fast`). The app reads only `artifacts/`, so it deploys without data or heavy dependencies (`requirements.txt`).

## Honest status of this build

- The code was written and tested in a sandbox with no access to FRED or PyPI. **Nothing has been run against real data.**
- Executed and passing there: the numpy VAR core, bootstrap/IRF/FEVD, scenario and conditional-forecast maths, splicing, transforms, parsers (on fixtures), evaluation metrics, policy calculators, the frozen-episode logic and the leakage assertions (50 tests).
- **Written but never executed** (libraries unavailable): everything using statsmodels (STL, SARIMAX, AutoReg, VAR fit), XGBoost/SHAP, Streamlit and Plotly, and the TimesFM scaffold. 15 tests skip without those libraries; expect some first-run fixes.
- Run `make fmt` (ruff formatting is unverified), then `make test`, then `make demo`.
- Before real results: supply a FRED key and the manual files, then **tag the commit `freeze-2026-02`** once `frozen_westasia_2026.parquet` is written. That file must never be regenerated.

## Method notes

- Oil is ordered first (India cannot move Brent within a month); block-exogeneity, "INR second" and local-projection checks are included.
- Index levels are never spliced across base years; growth rates are joined at stated switch months with an overlap report (`reports/splice_check.csv`).
- Conditional forecasts pin the oil path through the Cholesky factor; the oil shock offsets lagged feedback so the path holds exactly while other shocks stay free.
- The stress test is pseudo real-time (revised data). Limitations are listed on the About page.

## Layout

```
config/    series, models, episodes, events (YAML)     src/oilshock/  library + pipeline
app/       Streamlit app                               artifacts/     committed outputs the app reads
tests/     pytest suite                                data/raw/      FRED cache + manual files
```
