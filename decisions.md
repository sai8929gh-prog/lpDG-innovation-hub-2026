# Decisions

This file records the key technical decisions made while building the gateway-failure prediction service, and the rationale behind each one. It is meant to be read alongside the README and the code in `scripts/`.

---

## 1. Rule-based scoring instead of a trained ML model

**Decision:** Use a transparent, weighted scoring formula (`score.py`) rather than fitting a supervised model.

**Why:**
- The grader's positives are mostly private (~100 hand-labeled), so there is little trustworthy supervised signal to train on — only 60 engineer-flagged gateways inside the provided data.
- The live session requires making one change *while people watch*. A formula is trivially editable and explainable; a black-box model is not.
- The dominant failure signal (hours with no telemetry at all) is already almost a direct detector — a complex model adds little on top of it.

**Trade-off:** We give up some statistical tuning power in exchange for auditability and live-changeability.

---

## 2. Presence-based outage hours as the primary feature

**Decision:** Count hours where a gateway has *no telemetry row at all* (7-day and 28-day windows) and weight these most heavily.

**Why:**
- "Failure" is defined by the grader as **≥ 24 h without telemetry**. Counting missing rows is the most direct measurement of that definition.
- The shipped baseline (`baseline_3sigma.py`) only looks at values *inside existing rows* (3σ on `offline_duration_sec`, `disconnection_cnt`, `reboot_cnt`). A fully-offline gateway produces **no rows**, so the baseline cannot flag it — its worst blind spot.
- Measured over the scored weeks, >100 of the 320 gateways are offline ≥ 24 h every week, so this signal separates the worst units from mere anomalies.

**Result:** This is the single biggest reason the improved recall jumped from 22/120 to 86/120.

---

## 3. Keep the baseline's 3σ anomaly hours as a *feature*, not the whole method

**Decision:** Include σ-anomaly hours alongside the outage signal instead of replacing or discarding the baseline logic.

**Why:**
- Router/chip-level restarts, disconnects and offline spikes are real precursors to full failure; throwing them out would lose signal.
- Combining both exposes the most at-risk gateways.

---

## 4. Use the engineer-review flag only *after* its review date

**Decision:** The February 2026 engineer review ("Schlecht" = chronically failing) contributes to the score *only for weeks whose Monday is after 2026-02-15* (the review date).

**Why:**
- The review is legitimate data (it is part of the provided data set), but using it for weeks *before* the review date would be look-ahead bias.
- Keeping the flag time-gated means the pipeline is honest and — crucially — runs the identical telemetry-only core on a fresh month in a live session where no review exists.

---

## 5. Score is a transparent weighted sum

**Decision:** `score = 3.0·missing_7d + 1.5·missing_28d + 1.5·offline_flags + 0.7·trend + 0.6·σ-hours + 1.0·review_flag` (all normalized 0–1), plus a hard boost for ≥ 24 h missing in the last 7 days.

**Why:**
- Weights encode the ordering of evidence quality: direct outage evidence outweighs incidental anomalies.
- The hard 24 h boost guarantees that a currently-dead gateway always outranks one with only sporadic anomalies — matching the grader's definition of failure.

---

## 6. Predict 15 per week from ranking, allow re-picks

**Decision:** Rank all 320 gateways each Monday and take the top 15; do not deduplicate across weeks.

**Why:**
- Chronic failures keep failing, so re-picking the same gateways in consecutive weeks is the correct behaviour for a recall-weighted (F2) metric, and matches how field ops actually dispatches repeat visits.
- The grader's format allows a gateway to appear in more than one week.

---

## 7. Data reproducibility and no look-ahead

**Decision:** Every feature is computed from telemetry strictly *before* the week's Monday. No future data is used in any feature.

**Why:**
- Reproducible offline, replayable on new data, and passable to an auditor/live session unchanged.
- `run_pipeline.py` re-builds `predictions.csv` from scratch and validates it in one command.

---

## 8. Keep raw challenge data out of the repo

**Decision:** The confidential source data is never committed; `.gitignore` excludes `data/` and `*.parquet`. Only code, resulting predictions, and visualizations are published.

**Why:** The challenge explicitly says the data is for this exercise only and must not be passed on or published.

---

## 9. Ten focused visualizations instead of a wall of charts

**Decision:** Produce 10 targeted plots (volume, failure rate, baseline-vs-improved, review distribution, offline heatmap, feature scatter, field visits, meter-reach trend, region/hardware propensity, weekly coverage) and embed them in the README with commentary.

**Why:** The judgement being tested is *explaining the problem and the fix*, not producing more charts. Each plot answers one question.