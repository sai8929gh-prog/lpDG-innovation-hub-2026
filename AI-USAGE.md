# AI Usage

A transparent account of how AI-assisted tooling was used in this project, in the spirit of honest disclosure for the Innovation Hub 2026 challenge.

---

## Summary

AI-powered coding assistance (an interactive coding agent) was used to build, debug, and document this solution. All domain decisions, the choice of approach, and the final submission remain human-controlled and human-approved.

---

## Where AI helped

### 1. Code development and automation
- Scaffolded the project structure (`scripts/`, `plots/`, `data/`) and wrote the feature-engineering, scoring, and pipeline scripts.
- Automatically fixed runtime bugs (timezone handling in pandas comparisons, index-alignment issues, dtype conversions) discovered during execution.
- Wrote the validator integration (`validate_submission.py`) call and the one-command pipeline (`run_pipeline.py`).

### 2. Data exploration and prototyping
- Inspected the parquet telemetry schema and row counts across all 8 monthly partitions.
- Identified the key blind spot of the baseline (fully-offline gateways produce no telemetry rows) through quick exploratory queries.
- Implemented the presence-based outage heuristic and confirmed it empirically against the engineer-flagged gateways.

### 3. Visualisation and documentation
- Generated the 10 EDA/capability plots in `plots/`.
- Wrote the README, decisions.md and this AI-usage statement (reviewed and approved by the author before submission).

---

## Where AI was NOT used — and the work remains ours

- **No fake or generated data.** Only the real challenge data was used.
- **No label leakage beyond what the dataset itself provides.** The engineer-review flag is time-gated to weeks after its review date; all features are computed strictly before each scored Monday.
- **The problem framing, failure definition, signal selection and scoring weights were set deliberately** to match the stated grading criteria (F2, ≥70% SLA, 14 h technician rule, 15 visits/week), not produced wholesale by the AI.
- **The final ranked picks were reviewed and are interpretable**, not blindly sub-contracted to a model.

---

## Reproducibility

Every number in the README and `predictions.csv` is reproducible from the real data by running:

```bash
python scripts/run_pipeline.py --data path/to/challenge/data
```

No proprietary or AI-generated data is involved in producing the submission.

---

## Author's statement

The undersigned confirms they understand the solution, can explain every part of it, and can modify it live as required by the challenge brief.

Signed: Vamshi
Date: 2026-09-16
