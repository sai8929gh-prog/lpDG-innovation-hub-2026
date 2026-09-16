#!/usr/bin/env python3
"""One-command pipeline: build features -> score -> validate.

Usage:
    python scripts/run_pipeline.py --data path/to/challenge/data

Reproduces predictions.csv from scratch and validates it with the shipped
validator, exactly as a live session would:
    python scripts/run_pipeline.py --data data
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--features", type=pathlib.Path, default=pathlib.Path("data/features.parquet"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions.csv"))
    args = parser.parse_args(argv)

    (HERE.parent / "data").mkdir(exist_ok=True)

    steps = [
        [sys.executable, str(HERE / "build_features.py"),
         "--data", str(args.data), "--out", str(args.features)],
        [sys.executable, str(HERE / "score.py"),
         "--features", str(args.features), "--out", str(args.out)],
        [sys.executable, str(HERE / "validate_submission.py"), str(args.out)],
    ]
    for step in steps:
        print(f"\n$ {' '.join(str(s) for s in step)}")
        result = subprocess.run(step, cwd=HERE.parent)
        if result.returncode:
            return result.returncode
    print("\nPipeline complete. predictions.csv is ready for submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())