# Decisions

This file records the key technical decisions made while building the gateway-failure prediction service. Every choice here is one I could have made differently. Each section says what else I considered and why I did not take it. The brief explicitly punishes silent guessing, so: here are the calls, in writing. Read alongside the README and the code in `scripts/`.

---

## 1. Chosen Part 2 area: D — Data science (and why)

I picked D because the question the brief refuses to answer — what does "needs a visit" actually mean, and where does the €380/€600 line sit — is the one that most shapes whether the whole service is worth anything. I put no machine learning in Part 1 on purpose: the brief says Part 1 does not need it, `baseline_3sigma.py` works, and a weighted sum of explainable features is what the ops manager can actually change live in a session.

**What else I considered and rejected:** I considered fitting a model (area E) to the engineer review or to visit outcomes. Both candidate labels are selection-biased (see Decision 4), there is no true-negative set anywhere, and a model trained on those labels would learn "look like the gateways we've already suspected" — not "be broken". A model with no honest label is a model that lies more confidently than the weights do.

---

## 2. Cutoff timezone: Europe/Berlin Monday 00:00 (not IST, not UTC)

The brief's "All times IST" is the recruitment calendar (when sessions and hand-in happen). It is not the network's clock. The Data Dictionary is explicit: `DateDt` and `hour` are "in the gateway's local time zone (Europe/Berlin)", and `ts_utc` is the same moment in UTC.

So a prediction for week starting Monday 2026-02-02 uses only data strictly before Monday 00:00 Europe/Berlin = Sunday 23:00 UTC (CET = UTC+1).

**What else I considered and rejected:** used IST, or used Monday 00:00 UTC. Either would shift the cut by 1–5.5 hours relative to the network's own calendar. The data is German, the region/site_type values are German, and the cut is a decision about the network, not about the people evaluating me. UTC midnight would drop the last hour of "Monday" that the network still counts as Sunday-night; Berlin-midnight is the defensible reading of the dictionary. The one-hour CET→UTC conversion is encoded in `cutoff_utc` and unit-tested against the boundary second.

**Bug this prevents:** a "1 second after cutoff" row leaking into the window would let the prediction peek one hour into the week being predicted. A regression test pins this (`test_cutoff_row_one_second_after_is_dropped`). This also subsumes the older "no look-ahead" rule: every feature is computed from telemetry strictly before the week's Monday.

---

## 3. Gateway-id: one canonical key, normalised at every boundary

The four files do not agree on format:

File | Example | Format
--- | --- | ---
`gateway_master.csv` | `06:39:EA:56:02:C1` | colon, uppercase
`meter_read_success.csv` | `0202CB0A6B1F` | bare hex, uppercase
`telemetry/...parquet` | `0639EA5602C1` | bare hex, uppercase
`engineer_review.xlsx` | `06:5B:92:87:16:CD` | colon, uppercase

If any loader keeps its native format, joins silently come back empty against some files and the pipeline quietly produces garbage. That is a real bug found during inspection (304 eligible gateways week 1, not 290).

**Decision:** one function `normalize_gateway_id()` mirrors the validators' regexes, strips colons and uppercases; every loader runs it at the boundary; all joins happen on bare hex. Enumeration output uses whatever the validator accepts (bare hex is fine). A regression test pins exactly this bug: joining three incompatible formats must not come back empty.

---

## 4. What counts as "needs a visit", and what I treat as evidence

The definition: a gateway needs a visit when it shows persistent, above-its-own-baseline evidence of degraded operation in the days before the cutoff — (a) it went dark (stopped reporting after a history of reporting); (b) sustained elevated fault signals (offline time, disconnections, reboots, RX-CRC bad rate, bad-RSSI share) vs its own trailing distribution; or (c) a falling meter-read-success ratio over recent weeks. Single-hour blips never count; evidence must persist; gateways that never reported before being installed are not flagged as "dark".

**What else I considered and rejected:**

- **Absolute thresholds** (offline > X sec) — rejected: an 800-meter site has a different raw traffic volume than a 40-meter one, so an absolute cut re-ranks by site size, not by whether this gateway is behaving worse than usual. Z-scores against the gateway's own baseline fix that.
- **Treating field-visit outcomes as ground truth** — rejected: the brief says visited gateways are visited because they were suspected (a selection bias). "Kein Fehler gefunden" does not mean healthy, it means "looked suspicious and did not confirm". I use it only as a weak signal (repeated no-find → slightly raise; recent fix → slightly lower).
- **Treating the engineer review (Schlecht/Normal) as ground truth** — rejected. 60/60 curated balance, one day, one reviewer, all 2026-02-15, and only gateways that were already suspected. It is used only: (a) as a weak weekly feature for weeks where the review predates the cutoff (≥ 2026-02-16), and (b) in evaluation as a biased, clearly-labelled evaluation proxy — never combined into the score as if it were the answer.

What a visit "to a Normal gateway" means: in the cost model it's the €380 that is "wasted" — with the honest caveat that the review covers only ~120 gateways, so it is a proxy, not a full measurement.

---

## 5. Scoring and cost decisions

**Scoring:** features are percentile-ranked within the week's eligible population, then combined as a hand-set weighted sum (weights documented in `scripts/score.py`), renormalised per-week so a missing signal (e.g. engineer review pre-2026-02-16) never equals "zero evidence" but just redistributes weight. Weights are chosen so that a gateway completely offline for the week (168 missing hours) always ranks above one with only a handful of anomalous hours — matching the grader's definition of failure (≥ 24 h without telemetry). No learned weights: "simple, explainable" is what the brief asks for, and a one-line weight edit is what the live session wants.

**What else I considered and rejected:** a trained supervised model → rejected in Decision 1; a hard-coded 3σ-only method → rejected because a fully-offline gateway produces *no rows* at all, so the baseline cannot see its worst blind spot.

**Cost rule:** the brief gives two numbers — €380 wasted per false visit, €600 per broken-gateway-week. The decision the ops manager actually faces is where to draw the line. Evaluation computes precision@k as a function of k, and the marginal visit-eligibility rule falls out of the numbers:

- each visit costs €380,
- each broken gateway it finds saves €600 (for that week),
- so the k-th visit is worth sending only while the k-th ranked gateway has probability of being broken above 380/600 ≈ 0.633.

In the reviewed-set proxy, mean precision@k hovers around 0.55–0.61 for budgets through ~k=12 (k=1 lands 0.667 but on 6 weeks with worst 0.000 / best 1.000), sits at 0.556 by k=15, and decays to ~0.53 by k=20 and ~0.49 by k=25. Only k=1 clears the 0.633 break-even, so the honest reading is that the marginal visit sits at or just below the €380 / €600 line at every practical budget in this proxy. The chart shows cost vs k and cost vs threshold, and the whole analysis ships with the caveat that these numbers cannot measure the true rate for never-reviewed gateways.

**Prediction of 15 per week, allow re-picks:** rank all eligible gateways each Monday and take the top 15; do not deduplicate across weeks. Chronic failures keep failing, so re-picking the same gateways in consecutive weeks is correct behaviour for a recall-weighted metric and matches how field ops actually dispatches repeat visits. The grader's format allows a gateway to appear in more than one week.

---

## Honest limits

- **No ground-truth label exists.** The only evaluable "bad" set is the engineer's review of ~120 pre-suspected gateways. Evaluation therefore states a range (bootstrap CI) around precision, not a point, and never claims a true-negative rate — there is no such number in this data.
- **Weeks before 2026-02-16 cannot be validated at all** (no review yet temporally available), so the first two prediction weeks are un-evaluable against any human verdict.
- **Telemetry gaps are partly structural.** A gateway that was not yet installed (or was decommissioned) can look "silent"; the eligibility filter removes those, but a gateway silently off the air for a bad reason and one off for a good reason are the same few rows in the data.
- **The EU even-more-common objection:** repeated false visits are not currently penalised in cost — the €600-recurring assumption holds only as long as a broken gateway stays broken, which the review snapshot cannot evidence.

## What two more weeks' data would fix

The engineer review is a single-day snapshot (2026-02-15) covering ~120 pre-suspected gateways. A second review late in the window, plus the actual outcomes of the visits we recommend, would let the review-proxy precision be checked for every week and would let the cost model confirm (or deny) the €600-recurring assumption. The pipeline itself needs no changes for new data: the telemetry loader globs month/part-*.parquet dynamically, so a fresh month or a future review xlsx drops straight in.

## Reproducibility and data handling

- `run_pipeline.py` re-builds `predictions.csv` from scratch and validates it in one command; every feature uses only information strictly before the week's Monday, so the audit (and a live session on unseen data) can replay the exact same computation.
- The confidential source data is never committed; `.gitignore` excludes `data/` and `*.parquet`. Only code, resulting predictions, and visualizations are published — the challenge explicitly says the data is for this exercise only and must not be passed on or published.
- The ten targeted plots (volume, failure rate, baseline-vs-improved, review distribution, offline heatmap, feature scatter, field visits, meter-reach trend, region/hardware propensity, weekly coverage) are embedded in the README with commentary. The judgement being tested is *explaining the problem and the fix*, not producing more charts.