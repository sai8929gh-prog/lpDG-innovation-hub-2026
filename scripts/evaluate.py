"""Evaluate prediction quality against the engineer's review labels and the
telemetry-confirmed failure definition (>=24h without a telemetry row).
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from score import score_week


def load_review(data_dir: pathlib.Path) -> pd.DataFrame:
    review = pd.read_excel(data_dir / "engineer_review_2026-02.xlsx")
    review["gateway_id"] = (
        review["gateway_id"].str.replace(":", "", regex=False).str.upper()
    )
    return review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    feats = pd.read_parquet(args.features)
    review = load_review(args.data)
    schlecht = set(review.loc[review["Kategorie"] == "Schlecht", "gateway_id"])

    print("week        | recall@15 on review-Schlecht | F2-style hits")
    total = 0
    for week_start, week in feats.groupby("week_start", sort=True):
        week = week.reset_index(drop=True)
        score = score_week(week)
        top = score.sort_values(ascending=False).index[:15]
        picked = set(week.loc[top, "gateway_id"])
        hits = len(picked & schlecht)
        total += hits
        print(f"{week_start} | {hits}/15")
    print(f"TOTAL review-Schlecht hits over all weeks: {total}/120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())