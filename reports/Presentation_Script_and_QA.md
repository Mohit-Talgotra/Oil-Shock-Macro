# Presentation script and Q&A prep

Slide-by-slide speaking script, with likely professor questions and suggested answers. Items marked "Flag" or "Fill" need your attention before presenting.

## Slide 1: Title

**Script**

Good morning. My project asks one question: when oil prices jump, what happens to India's economy, and can we trust our forecasts when it does? I built a dashboard in three layers: a transmission model, a set of forecasting models, and a stress test that checks where those forecasts break. I'll spend most of the time on how it works and why I made each design choice, then show the results.

**If asked**

- **What is actually new here?** Each component is standard: a recursive VAR, common forecasting models, backtesting. The contribution is narrow: putting them in one reproducible pipeline, attaching uncertainty to every answer, and testing systematically where pre-shock forecasts fail (blind vs oracle oil). I don't claim a new identification strategy.

## Slide 2: India buys almost all its oil abroad

**Script**

India imports roughly 88 percent of its crude oil: 88.2 percent in FY25 according to PPAC data. So the world oil price is effectively an external shock that India cannot influence. That is the first modelling fact I rely on. I look at four episodes: the 2008 spike and crash, the 2014 to 2016 collapse, the 2022 Russia-Ukraine spike, and the live 2026 West Asia episode. Two up-shocks, one down-shock and one spike-then-crash gives me variation in direction and in policy regime.

**If asked**

- **Why these four episodes?** They cover both directions and three different policy regimes (pre-inflation targeting, post-2016 targeting, and 2026 with the new CPI base). I excluded 2020 because the COVID demand collapse contaminates the oil signal; it is handled with dummies instead.
- **Where does the 88% come from?** PPAC import dependence, reported via India Sea Trade News (FY25: 88.2%). Re-check the figure against PPAC before presenting.

## Slide 3: Three questions

**Script**

Three research questions. One: how large and how persistent is the effect of an oil shock on inflation, the rupee, the policy rate, the current account and growth? Two: did that transmission change across episodes? Three, which I think is the most interesting: how accurately do models trained only on pre-shock data forecast through a shock, and do their uncertainty bands warn us? That third question is the Zillow lesson applied to macro policy: a model that is confident right before the world changes.

**If asked**

- **What are your hypotheses?** H1 pass-through to CPI peaks several months later and WPI responds more than CPI. H2 the rupee depreciates on impact. H3 the repo rate responds with a lag, more systematically after inflation targeting in 2016. H4 GDP growth falls 2-4 quarters later. H5 CPI pass-through is weaker in 2022/2026 than 2008 (fuel-tax policy, lower food/fuel weight in the new CPI). H6 pre-shock models under-predict inflation and depreciation, and 80% bands cover less than 80% of outcomes.
- **Flag to check before presenting** H5 compares against 2008, but the IIP series in your data files starts in April 2012, so the monthly model cannot be estimated on a pre-2008 window. Decide how you will word H5 (e.g. compare 2014 vs 2022 vs 2026 only) and be ready to say so.

## Slide 4: Seven channels from oil to GDP

**Script**

Oil reaches the economy through seven channels. A bigger import bill widens the trade and current account deficit. Dollar demand and risk-off flows weaken the rupee. Fuel prices feed directly into CPI, but only as far as administered retail prices allow. Second-round effects come through freight, fertiliser and inputs. The RBI may tighten or delay cuts. The government may cut excise or raise subsidies. And higher costs slow real activity. The India-specific twist is channel three: retail petrol and diesel prices were held flat for years, so CPI pass-through is muted while wholesale prices move much more. That gap is one of the things the dashboard is designed to show.

**If asked**

- **Which channels does the model estimate and which does it assume?** Estimated in the monthly VAR: rupee, CPI, repo, industrial output. Estimated in the quarterly VAR: current account and GDP. The fiscal channel is not estimated; it is a calculator in the Policy Lab (excise cost, import bill).
- **Why would CPI pass-through be muted?** Administered retail fuel prices. The plan cites April 2026 WPI inflation of 8.3% against CPI of 3.48%, with the first retail petrol/diesel hike in four years in May 2026. Verify those figures against the cited sources before quoting.

## Slide 5: Data: what goes into the models

**Script**

Data. Brent and the rupee come from FRED; the policy rate from RBI; CPI, GDP and industrial output from MOSPI; the current account from RBI's balance of payments. The hard part was not downloading, it was the base-year breaks. MOSPI moved to a 2024-base CPI and a 2022-23 base GDP series in 2026, and the convenient OECD copies on FRED stopped updating. My rule is: never splice index levels. I convert each series to growth rates within its own base, join the growth rates at a stated switch month, and log the overlap gap so the size of the break is visible rather than hidden.

**If asked**

- **Why splice growth rates, not levels?** Index levels from different base years are not comparable; splicing them creates an artificial jump that a VAR would read as a shock. Growth rates are comparable across bases. The overlap report (splice_check.csv) gives mean and max gap on the overlap; if the mean gap exceeds the threshold it is disclosed on the About page.
- **Is the data real-time?** No. The stress test is pseudo real-time: it uses today's revised data, not what forecasters saw then. ALFRED vintages would be the extension. Say this explicitly.
- **Any data caveats to know?** From the manual-file log: CPI back series is headline only (no fuel/core), IIP starts Apr 2012, WACR daily history starts Jul 2020, the excise file is a 2-row spot, GDP-2022 has no y-o-y for its first four quarters, and pre-2011 nominal GDP is level-spliced. Know these before the Q&A.

## Slide 6: Heavy work offline, the app only reads

**Script**

The system has two halves. An offline pipeline fetches and cleans data, splices series, builds monthly and quarterly panels, estimates the models, runs the backtests and the stress test, and writes small result files, called artifacts. The Streamlit app only reads those artifacts and does light arithmetic, such as rescaling an impulse response when the slider moves. I split it this way for four reasons: speed, because nothing is re-estimated on a slider move; hosting, because the free cloud tier has limited memory; reproducibility, because one command rebuilds every number and seeds are fixed; and robustness, because a broken data source can never take the live app down.

**If asked**

- **How do you know the code is correct?** Unit tests on the economics: the VAR recovers a known simulated VAR, scaling is linear, splicing refuses levels, backtests assert no future rows leak into training, frozen forecasts do not change when later data is appended. Be honest that tests ran on fixtures and synthetic data; the real-data run is the first end-to-end test.
- **What is reproducible?** Seeds come from models.yaml; a manifest records the data cut-off, git commit and package versions; `make all` rebuilds everything.

## Slide 7: Layer 1: a VAR with oil ordered first

**Script**

Layer one is a recursive VAR with oil ordered first. The ordering is an identification assumption, so I'll justify each position. Oil is first because India is a price-taker: nothing in India moves Brent within the month. Industrial output is second because output adjusts slowly. CPI third, prices adjust after output within the month. The repo rate fourth, since the RBI sees same-month data on oil and prices. The rupee last, because it is a fast asset price that reacts to everything. Controls for the US fed funds rate, VIX and COVID and demonetisation dummies are exogenous. Lags are chosen by BIC up to twelve, then increased until residuals pass a whiteness test. Confidence bands come from a thousand bootstrap replications with a fixed seed.

**If asked**

- **Why recursive identification, not sign restrictions or a structural VAR?** It is the standard choice for small open economies, it is simple to explain and defend, and the ordering has a clear economic rationale. The cost: the oil innovation is a reduced-form shock that mixes supply and demand (Kilian 2009). A structural VAR with sign restrictions is the natural extension.
- **How do you check the ordering matters?** Three robustness checks: re-estimate with INR second; a block-exogenous oil equation (own lags only); and local projections (Jordà 2005) plotted next to the VAR response.
- **Why are the variables differenced?** For stationarity: ADF and KPSS are reported for every series, and responses are cumulated to show effects on price level, rupee level and repo level.
- **Isn't the sample small?** Yes. The plan assumed about 260 monthly observations, but industrial output starts April 2012, so the effective monthly sample is shorter (about 168 months with IIP). Check the actual N in the diagnostics table and state it. Small samples mean wide bands, which are shown rather than hidden.

## Slide 8: Scenario engine: any shock size from one estimate

**Script**

The scenario engine is why the dashboard can answer 'what if' questions. The VAR is linear, so a shock of any size is a rescaled copy of the estimated one. A slider value of s percent is converted to log points, divided by the estimated standard deviation of the oil shock, and multiplied through the stored impulse responses. Because the CPI variable is month-on-month, I convert it to the year-on-year path people know by summing the last twelve monthly responses. The price of this simplicity is symmetry: plus thirty and minus thirty percent give mirror-image answers. I state that on the dashboard.

**If asked**

- **Why divide by sigma?** Cholesky impulse responses are to a one-standard-deviation oil shock. The first element of the Cholesky factor is that standard deviation, so dividing the log-point shock by it gives the multiple to apply.
- **Is a fall in oil as powerful as a rise?** Not necessarily; Mork (1989) finds asymmetry. The linear VAR cannot show it. The asymmetric local-projection extension is the stretch goal and the 2014-16 episode is the natural illustration.
- **What is the conditional forecast?** For a user-drawn oil path, the oil path is pinned using the Cholesky factor: the oil shock offsets lagged feedback so the path holds exactly while the other shocks remain free.

## Slide 9: Layer 2: five models, one interface

**Script**

Layer two forecasts three targets: CPI inflation, USD/INR change and GDP growth, with five model families under one common interface. Naive and seasonal naive are the bar everything must beat. AR is the classical univariate benchmark. ARIMAX adds oil as an input. The VAR links the forecasts to Layer one. And quantile XGBoost is the non-linear machine learning model, with one model per horizon and quantile. All are scored on the same rolling origins starting in 2012, and refit at each origin using only data available then. I deliberately left out LSTMs: with a few hundred observations they would overfit, and I'd rather say that than include one for show.

**If asked**

- **Why not a fancier model?** Small macro samples punish complexity. The metrics table shows whether XGBoost actually beats the naive baseline; if it doesn't, that is a legitimate result.
- **How do you prevent leakage?** Every forecaster's fit asserts no data past the origin; the rolling-origin engine refits at each origin; a test checks no training row is dated after its origin. XGBoost tuning uses only the pre-2016 window.
- **What does MASE below 1 mean?** The model's error is lower than the naive no-change forecast on the same origins.
- **What is the Diebold-Mariano test for?** Whether the accuracy difference versus naive is statistically meaningful, rather than noise.
- **What does coverage tell you?** An 80% band should contain about 80% of outcomes. Under-coverage means the model is overconfident; I also report band width so good coverage isn't bought with useless wide bands.

## Slide 10: Layer 3: freeze, then forecast blind and with oracle oil

**Script**

Layer three is the stress test. For each episode I freeze every model at the month before it starts and forecast through it from that single origin, so there is no updating. The key design choice is two information sets. In the blind run, oil stays flat at its last pre-episode value: could anyone have seen this coming? In the oracle run the model is given the actual oil path: given the oil shock, did the model understand how India would respond? If a model fails blind but works with oracle oil, the problem was predicting oil, not understanding India. If it fails both, the transmission itself changed.

**If asked**

- **Why two information sets?** To separate two different failures: not foreseeing the oil shock (blind) versus misjudging transmission (oracle). Naive and AR have no oracle version because they take no covariates.
- **Why is 2008 treated with caution?** It has the shortest training history, so its results carry the widest uncertainty, and (given the IIP start date) the monthly model may not be estimable for that window at all. Say which it is.
- **Is the 2026 test truly out-of-sample?** Be careful with the wording. The plan calls for generating the 2026 forecasts at end-February 2026, committing them under tag freeze-2026-02 and never regenerating. If the frozen file has not yet been generated, describe it as 'models trained only on data to Feb 2026', not as a pre-registered forecast, because you already know what happened.

## Slide 11: What each forecast is scored on

**Script**

Each forecast is scored on six things. Episode RMSE and bias tell me how far off and in which direction. The shock penalty divides episode error by the same model's error in calm periods, so above one means it degrades in a shock. Eighty percent coverage and months-to-surprise tell me whether the bands warned. And the policy gap makes the Zillow analogy concrete: I compute the Taylor-rule policy rate from forecast inflation and from actual inflation, and the average gap in basis points is what the miss would have cost a policymaker. Zillow's pricing model kept predicting rising prices and overpaid for homes; a central bank leaning on a forecast that misses an oil shock sets the rate in the wrong place.

**If asked**

- **What are the Taylor rule parameters?** Policy defaults: inflation target 4% (RBI), weight on inflation gap 0.5, weight on output gap 0.5 (Taylor's original weights), and r* = 1.5% shown as an assumption on screen. Output gap comes from an HP filter.
- **Isn't the Taylor rule a crude benchmark for the RBI?** Yes. It is a transparent yardstick for the cost of a forecast error, not a claim about how the RBI sets rates. All parameters are editable in the Policy Lab.

## Slide 12: Key decisions and the alternatives

**Script**

These are the decisions I'd expect to be challenged on, with the reasoning and the alternative I rejected. Oil first with recursive identification because India is a price-taker; the alternative is sign restrictions, which are harder to estimate and explain. Monthly core plus a small quarterly model because monthly gives about four times the observations, while GDP and the current account exist only quarterly. Growth-rate splicing because levels from different bases aren't comparable. Offline pipeline with an app that only reads artifacts for speed and robustness. No LSTM because of sample size. And blind versus oracle oil to separate the two kinds of forecast failure.

**If asked**

- **If you could change one decision?** A candid answer: identification. A structural VAR that separates oil supply from demand shocks would sharpen the economics, at the cost of complexity and fragility with this sample size.

## Slide 13: Live demo

**Script**

Now a short live demo. First the Transmission page at a plus-thirty percent shock: the chain shows the peak effect and month of peak at each stage. Then I switch the estimation window between pre-2014, pre-2022 and pre-2026 to show how the responses change. Finally the Stress Test page: pick an episode, toggle blind versus oracle, and see each model's frozen forecast against what happened and whether the 80 percent band contained it. If the connection fails, I have screenshots of each step.

**If asked**

- **If the demo breaks?** Switch to the backup screenshots; load the app before presenting so the free-tier cold start is done.

## Slide 14: Finding 1: transmission

**Script**

[Fill in once results are ready.] Suggested structure: state the headline in one sentence: for a ten percent rise in Brent, CPI inflation rises by [X] basis points, peaking after [N] months. Compare it with the RBI benchmarks: the RBI Bulletin of July 2025 finds roughly plus 20 basis points, plus 30 in the short run with incomplete pass-through. Say whether my estimate falls inside, above or below, and what that means. Then the rupee on impact, the policy rate response, and how much of the variance in each variable oil explains at 6, 12 and 24 months.

**If asked**

- **Why does your estimate differ from the RBI's?** Different sample, identification, price-control treatment and model; the RBI sample is 2009-10 to 2023-24. If it is far outside the 10-60 bps range, call it a red flag to investigate (ordering, lags, sample, splicing), not a finding.
- **What if a sign is wrong (e.g. the rupee does not depreciate on impact)?** Report it honestly. The plan says a failed plausibility check is a prompt to investigate, and if the sign is still wrong afterwards, report it on the About page and in the talk. Narrow, honest scope is a stronger answer than a forced result.

## Slide 15: Finding 2: has pass-through changed?

**Script**

[Fill in once results are ready.] This chart shows the 12-month cumulative CPI response in rolling 96-month windows. The question is whether pass-through changed: before versus after inflation targeting (H3), and in the more recent episodes (H5). Remember that wide bands in short windows are expected, and I show them. Report what the data say, including if the answer is 'no clear change'.

**If asked**

- **Is the change statistically distinguishable?** Compare the bands: overlapping bands mean you cannot claim a change. Say so if that is what you see.

## Slide 16: Finding 3: forecasts under shocks

**Script**

[Fill in once results are ready.] Show which model got burned in each episode: the shock penalty, 80 percent coverage, months to surprise, and the policy gap, blind versus oracle. Then answer H6 directly: did pre-shock models under-predict inflation and depreciation, and did the 80 percent bands cover less than 80 percent of outcomes? Close with the Zillow parallel in two sentences.

**If asked**

- **Did the ML model beat naive?** Answer from the metrics table. If it didn't, say that small macro samples punish complexity; that is a legitimate result.
- **What do blind and oracle results together imply?** Fails blind, fine with oracle: the problem is predicting oil. Fails both: transmission changed (links to H5 and the rolling-window chart).

## Slide 17: Limitations, stated openly

**Script**

I want to be upfront about limitations. The model is linear and symmetric. Identification relies on a recursive ordering. The samples are small, so bands are wide. The stress test uses revised data, not what forecasters had in real time. Fuel-tax and price-freeze decisions are discretionary, so past pass-through may not predict future pass-through. And base-year splices introduce breaks I have measured but not eliminated. This is a tool to illustrate relationships and model risk, not a forecasting service or policy advice.

**If asked**

- **What would you do with more time?** Real-time vintages (ALFRED); structural identification separating supply and demand oil shocks; asymmetric responses via local projections; the official GDP back series when MOSPI publishes it (expected December 2026); TimesFM as a zero-shot comparison.

## Slide 18: Takeaways

**Script**

To conclude. First, a transparent, tested framework that converts any oil shock size into responses for India's key variables, with uncertainty. Second, a stress test that separates failing to foresee oil from misjudging how India responds. Third, the 2026 episode stays live: as new data arrive, the forecasts trained on data to February 2026 keep being re-scored, never regenerated. Thank you; I'm happy to take questions on any design choice.

**If asked**

- **What would you tell the RBI today?** Use the Policy Lab to read off the rule-implied policy path under the current shock, and stress that forecasts made before a shock tend to under-predict (H6, if your results support it). Frame it as an illustration, not advice.

