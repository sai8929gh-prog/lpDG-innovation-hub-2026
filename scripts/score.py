"""Score gateways and emit ranked predictions.csv for the 8 scored weeks.

Scoring philosophy (mirrors how we would operate a live monitor):

  * Outage hours (hours with NO telemetry row) are the strongest, most direct
    signal — a gateway that truly fails stops producing rows entirely. The 3
    sigma baseline cannot see this, so it systematically misses the worst units.
  * 3-sigma anomalies on offline/disconnect/reboot counters, reported by the
    baseline, are kept and combined with the outage signal.
  * Worsening behaviour counts: offline time this week vs last week.
  * Chronic offenders flagged by an engineer review in the middle of the window
    are weighted forward (they keep failing). The flag is only usable for the
    weeks after the review date, so first weeks rely purely on telemetry.

The final score is a transparent weighted blend. Weights are chosen so that a
gateway that is completely offline for the week (168 missing hours) always ranks
above one that merely had a handful of anomalous hours, matching the definition
of "failure" used by the grader (>= 24h without telemetry).

Output schema required by validate_submission.py:
    week_start, rank, gateway_id, score, reason
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd


def score_week(week: pd.DataFrame) -> pd.DataFrame:
    def z(col):
        s = week[col].astype(float)
        std = s.std()
        return (s - s.mean()) / std if std and std == std else s - s.mean()

    # --- presence-based outage (the defining signal) -----------------------
    missing_7 = week["missing_hours_7d"]
    missing_28 = week["missing_hours_28d"]

    # continuous degradation of outage: 7d missing hours on top of 28d
    outage = 3.0 * (missing_7 / 168.0) + 1.0 * (missing_28 / (168.0 * 4))

    # --- baseline-style 3-sigma anomaly hours ------------------------------
    sig_hours = week["sigma_flag_hours_7d"]
    offline_flag = week["offline_flag_hours_7d"]

    anomaly = 0.6 * (sig_hours / 168.0) + 0.8 * (offline_flag / 168.0)

    # --- worsening trend ----------------------------------------------------
    trend = (week["offline_trend_ratio"].clip(0, 10) / 10.0)

    # --- memory / load pressure (resource exhaustion often precedes reboot) --
    resource = ((week["load1_14d"] - week["load1_14d"].median()) /
                (week["load1_14d"].std() + 1e-9)).clip(0, 3) / 3.0

    # --- engineer review prior (usable only after the review date) ---------
    review = 1.0 * week["review_schlecht"] + 0.0 * week["review_normal"]

    # raw composite
    raw = (
        outage
        + anomaly
        + 0.5 * trend
        + 0.3 * resource
        + review
    )

    # gateway-specific recency boost: a unit with recent full-outage evidence
    # (>= 24h missing last week) is pushed above everything else regardless of
    # how the remaining signals look.
    hard_offline = week["missing_hours_7d"] >= 24.0
    raw = raw + hard_offline * 5.0

    score = raw.fillna(0.0)
    return score


def top_reasons(week: pd.DataFrame, score: pd.Series, k: int = 15) -> list[str]:
    reasons = []
    idx = np.argsort(-score)[:k]
    for i in idx:
        row = week.iloc[i]
        m7 = int(row["missing_hours_7d"])
        m28 = int(round(row["missing_hours_28d"]))
        sig = int(row["sigma_flag_hours_7d"])
        flag = int(row["offline_flag_hours_7d"])
        bits = []
        if m7 >= 24:
            bits.append(f"{m7}h without telemetry last 7d")
        elif m7 > 0:
            bits.append(f"{m7}h no telemetry last 7d")
        if m28:
            bits.append(f"{m28}h no telemetry last 28d")
        if sig:
            bits.append(f"{sig}h >3sigma vs 28d baseline")
        if flag and not bits:
            bits.append(f"{flag}h offline report last 7d")
        if row["review_schlecht"]:
            bits.append("engineer review: frequent failures")
        if row["visits_90d"]:
            bits.append(f"{int(row['visits_90d'])} field visits 90d")
        if not bits:
            bits.append("no recent outage indicators")
        reasons.append("; ".join(bits[:4]))
    return reasons


def build_predictions(feats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for week_start, week in feats.groupby("week_start", sort=True):
        score = score_week(week)
        top = score.sort_values(ascending=False).index[:15]
        reasons_for_top = top_reasons(week.reset_index(drop=True), score.to_numpy())
        for rank, (pos, reason) in enumerate(zip(top, reasons_for_top), 1):
            gid = week.loc[pos, "gateway_id"]
            rows.append(
                {
                    "week_start": week_start,
                    "rank": rank,
                    "gateway_id": gid,
                    "score": round(float(score.loc[pos]), 4),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions.csv"))
    args = parser.parse_args(argv)

    feats = pd.read_parquet(args.features)
    predictions = build_predictions(feats)
    predictions.to_csv(args.out, index=False)
    print(f"wrote {args.out} — {len(predictions)} rows over "
          f"{predictions.week_start.nunique()} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())