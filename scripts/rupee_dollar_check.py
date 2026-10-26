"""Does the 'rupee appreciates on impact' result reflect dollar weakness co-moving with Brent?

Re-estimates Model A (same data, lags by BIC, 1,000 bootstrap draws, fixed seed) on the 2006+
sample where the broad US dollar index (FRED DTWEXBGS) exists, with the dollar as a control and
with an oil shock purged of the dollar. Needs network access. Writes reports/rupee_dollar_check.csv.

    python scripts/rupee_dollar_check.py
"""

import io
import math
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
from oilshock import config  # noqa: E402
from oilshock import pipeline as pl  # noqa: E402
from oilshock.data import panel as panelmod  # noqa: E402
from oilshock.models import var as varmod  # noqa: E402

cfg = config.models_cfg()
vm = cfg["var_monthly"]
monthly, _ = pl._load_panels()
endog, exog = panelmod.monthly_var_frame(monthly, "full")
print("endog", endog.index.min().date(), endog.index.max().date(), list(endog.columns))

# broad dollar index, monthly mean, log change
raw = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS", timeout=60).text
d = pd.read_csv(io.StringIO(raw), na_values=["."])
d.columns = ["date", "dxy"]
d["date"] = pd.to_datetime(d["date"])
dxy = d.set_index("date")["dxy"].dropna().resample("ME").mean()
ddxy = np.log(dxy).diff().rename("ddxy")
print("dxy", dxy.index.min().date(), dxy.index.max().date())

start = max(endog.index.min(), ddxy.dropna().index.min())
E = endog.loc[start:].copy()
X = exog.loc[start:].copy() if exog is not None else None
D = ddxy.reindex(E.index)
ok = D.notna()
E, X, D = E[ok], (X[ok] if X is not None else None), D[ok]
print("common sample", E.index.min().date(), E.index.max().date(), len(E))

print("\n-- contemporaneous correlations (monthly changes, common sample)")
c = pd.concat([E["oil"], E["inr"], D], axis=1)
print(c.corr().round(3))
print("(positive d(inr) = rupee depreciation; positive d(dxy) = dollar strengthening)")

H, REPS, SEED = 12, 1000, int(cfg["seed"])


def run(label, endog_df, exog_df, inr_col="inr"):
    ex = varmod.usable_exog(exog_df)
    k, m = endog_df.shape[1], 0 if ex is None else ex.shape[1]
    feasible = max(1, int((len(endog_df) / 4 - 1 - m) // k))
    sel = varmod.select_lags(
        endog_df, min(int(vm["maxlags"]), feasible), ex, 0.05, int(vm["whiteness_nlags"])
    )
    p = sel["chosen"]
    fit = varmod.estimate_var(endog_df, p, ex)
    irf = varmod.irf_bands(fit, H, REPS, SEED)
    sub = irf[(irf["response"] == inr_col)] if "response" in irf.columns else irf
    return label, p, irf, sub


LN10 = math.log(1.10) * 100


def summarize(label, p, irf, n):
    sub = irf[irf["response"] == "inr"]
    ss = float(sub["shock_size_logpts"].iloc[0])
    sc = LN10 / ss
    non = sub[~sub["cumulative"]].set_index("horizon")
    cum = sub[sub["cumulative"]].set_index("horizon")
    r0 = non.loc[0]
    out = {
        "spec": label,
        "lags": p,
        "n": n,
        "oil_sd": round(ss, 2),
        "impact_per10": round(r0["point"] * sc, 3),
        "impact90": f"[{r0['lo90'] * sc:.3f},{r0['hi90'] * sc:.3f}]",
    }
    if len(cum):
        c12 = cum.loc[12]
        out["cum12_per10"] = round(c12["point"] * sc, 3)
        out["cum12_90"] = f"[{c12['lo90'] * sc:.3f},{c12['hi90'] * sc:.3f}]"
    return out


rows = []
_, p, irf, _ = run("S0", E, X)
rows.append(summarize("S0 baseline (2006+ sample)", p, irf, len(E)))

X1 = X.copy() if X is not None else pd.DataFrame(index=E.index)
X1["ddxy"] = D
_, p, irf, _ = run("S1", E, X1)
rows.append(summarize("S1 + dollar index as control", p, irf, len(E)))

Z = pd.concat([D, D.shift(1)], axis=1).fillna(0.0)
A = np.column_stack([np.ones(len(Z)), Z.values])
beta, *_ = np.linalg.lstsq(A, E["oil"].values, rcond=None)
oil_p = pd.Series(E["oil"].values - A @ beta, index=E.index, name="oil")
E2 = E.copy()
E2["oil"] = oil_p
print("share of oil variance explained by dollar:", round(1 - oil_p.var() / E["oil"].var(), 3))
_, p, irf, _ = run("S2", E2, X1)
rows.append(summarize("S2 oil purged of dollar + control", p, irf, len(E2)))

m = E.index >= "2016-01-31"
_, p, irf, _ = run("S3", E[m], X[m])
rows.append(summarize("S3 baseline, post-2016", p, irf, int(m.sum())))
_, p, irf, _ = run("S4", E[m], X1[m])
rows.append(summarize("S4 + dollar control, post-2016", p, irf, int(m.sum())))

print()
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
print("\nUnits: % change in USD/INR per +10% oil. Negative = rupee appreciation.")
out = config.paths().reports / "rupee_dollar_check.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("written", out)
