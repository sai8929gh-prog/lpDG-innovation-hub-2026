"""Generate EDA and analysis plots for the Innovation Hub 2026 challenge."""
from __future__ import annotations
import datetime as dt
import pathlib
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
DATA = pathlib.Path("/home/vamshi/Downloads/Innovation Hub 2026/03-challenge-data/data")
PLOTS = pathlib.Path("/home/vamshi/LPDG/plots")
PLOTS.mkdir(exist_ok=True)

# --- load data ---
tel = pd.read_parquet(DATA / "telemetry")
tel["ts"] = pd.to_datetime(tel["ts_utc"], utc=True)
master = pd.read_csv(DATA / "gateway_master.csv", encoding="latin-1")
master["gateway_id"] = master["gateway_id"].str.replace(":", "", regex=False).str.upper()
visits = pd.read_csv(DATA / "field_visits.csv", encoding="latin-1")
visits["gateway_id"] = visits["gateway_id"].str.replace(":", "", regex=False).str.upper()
visits["requested_on"] = pd.to_datetime(visits["requested_on"])
review = pd.read_excel(DATA / "engineer_review_2026-02.xlsx")
review["gateway_id"] = review["gateway_id"].str.replace(":", "", regex=False).str.upper()
meters = pd.read_csv(DATA / "meter_read_success.csv")
meters["gateway_id"] = meters["gateway_id"].str.replace(":", "", regex=False).str.upper()
meters["week_start"] = pd.to_datetime(meters["week_start"])
meters["reach_pct"] = meters["meters_read"] / meters["meters_expected"]

feats = pd.read_parquet("/home/vamshi/LPDG/data/features.parquet")
pred = pd.read_csv("/home/vamshi/LPDG/predictions.csv")
pred_base = pd.read_csv("/home/vamshi/LPDG/predictions_baseline.csv")

SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]


def save(fig, name):
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# ── 1  Telemetry volume per month ─────────────────────────────────────────────
print("Plot 1: telemetry volume")
tel["month"] = tel["ts"].dt.to_period("M").astype(str)
monthly = tel.groupby("month").size()
fig, ax = plt.subplots(figsize=(9, 4))
monthly.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(monthly)))
ax.set_ylabel("Rows (hours × gateways)")
ax.set_title("Telemetry rows per month")
ax.tick_params(axis="x", rotation=45)
for i, v in enumerate(monthly):
    ax.text(i, v + 1e3, f"{v/1e6:.2f}M", ha="center", fontsize=8)
save(fig, "01_telemetry_volume.png")

# ── 2  Failure rate by week (≥24h without telemetry) ──────────────────────────
print("Plot 2: failure rate by week")
gws = tel["gateway_id"].drop_duplicates().sort_values()
fail_counts = []
for m in SCORED_WEEKS:
    lo = pd.Timestamp(m, tz="UTC")
    hi = lo + dt.timedelta(days=7)
    cnt = tel[(tel["ts"] >= lo) & (tel["ts"] < hi)].groupby("gateway_id").size()
    miss = 168 - cnt.reindex(gws).fillna(0)
    fail_counts.append((m, (miss >= 24).sum()))
df_fc = pd.DataFrame(fail_counts, columns=["week", "failures"])
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(df_fc["week"], df_fc["failures"], color="#e74c3c", edgecolor="#c0392b")
ax.axhline(120, ls="--", color="gray", alpha=0.6, label="120 picks available")
ax.set_ylabel("Gateways offline ≥24 h")
ax.set_title("Gateways failing per scored week (≥24 h without telemetry)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend()
save(fig, "02_failure_rate_per_week.png")

# ── 3  Baseline vs Improved coverage of Schlecht ──────────────────────────────
print("Plot 3: baseline vs improved")
schlecht = set(review.loc[review["Kategorie"] == "Schlecht", "gateway_id"])
def count_overlap(frame):
    frame = frame.copy()
    frame["gw"] = frame["gateway_id"].str.replace(":", "", regex=False).str.upper()
    return [frame[frame["week_start"] == str(w)]["gw"].isin(schlecht).sum()
            for w in SCORED_WEEKS]

base_counts = count_overlap(pred_base)
impr_counts = count_overlap(pred)
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(SCORED_WEEKS))
w = 0.35
ax.bar(x - w/2, base_counts, w, label="Baseline (3σ)", color="#3498db")
ax.bar(x + w/2, impr_counts, w, label="Improved", color="#27ae60")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in SCORED_WEEKS], rotation=45, fontsize=8)
ax.set_ylabel("Known-failing gateways in top 15")
ax.set_title("Recall of engineer-flagged failing gateways per week")
ax.legend()
for i in range(len(SCORED_WEEKS)):
    ax.text(i - w/2, base_counts[i]+0.3, str(base_counts[i]), ha="center", fontsize=7)
    ax.text(i + w/2, impr_counts[i]+0.3, str(impr_counts[i]), ha="center", fontsize=7)
save(fig, "03_baseline_vs_improved.png")

# ── 4  Engineer review distribution ────────────────────────────────────────────
print("Plot 4: engineer review")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
review["Kategorie"].value_counts().plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"])
axes[0].set_title("Review outcome (Schlecht = failing)")
counts = review["Bemerkung"].dropna()
kw = counts.value_counts().head(8)
kw.plot(kind="barh", ax=axes[1], color="#9b59b6")
axes[1].set_title("Top engineer comments (Bemerkung)")
axes[1].invert_yaxis()
save(fig, "04_engineer_review_dist.png")

# ── 5  Telemetry heatmap: gateways × time ─────────────────────────────────────
print("Plot 5: heatmap (sampled gateways)")
# pick worst gateways from feats (first week, by missing_hours_7d)
first_w = feats[feats["week_start"].astype(str) == str(SCORED_WEEKS[0])]
worst = first_w.sort_values("missing_hours_7d", ascending=False).head(30)["gateway_id"].tolist()
tel_s = tel[tel["gateway_id"].isin(worst)][["gateway_id", "ts", "offline_duration_sec"]].copy()
tel_s["date"] = tel_s["ts"].dt.date
daily = tel_s.groupby(["gateway_id", "date"])["offline_duration_sec"].sum().reset_index()
daily_pivot = daily.pivot(index="gateway_id", columns="date", values="offline_duration_sec").fillna(0)
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(daily_pivot, cmap="YlOrRd", ax=ax, xticklabels=10, yticklabels=True)
ax.set_title("Daily offline duration for worst gateways (Aug 2025 – Mar 2026)")
ax.set_ylabel("")
ax.set_xlabel("Date")
save(fig, "05_heatmap_offline_duration.png")

# ── 6  Missing hours vs engineered features (scatter) ─────────────────────────
print("Plot 6: scatter")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
f = feats[feats["week_start"].astype(str) == str(SCORED_WEEKS[3])]
f["schlecht"] = f["gateway_id"].isin(schlecht)
ax = axes[0]
ax.scatter(f["missing_hours_7d"], f["sigma_flag_hours_7d"], c=f["schlecht"].map({True:"#e74c3c", False:"#3498db"}),
           alpha=0.4, s=12)
ax.set_xlabel("Missing hours (7d)")
ax.set_ylabel("3σ anomaly hours (7d)")
ax.set_title("Missing hours vs anomaly flags")
ax = axes[1]
ax.scatter(f["missing_hours_7d"], f["offline_trend_ratio"], c=f["schlecht"].map({True:"#e74c3c", False:"#3498db"}),
           alpha=0.4, s=12)
ax.set_xlabel("Missing hours (7d)")
ax.set_ylabel("Offline trend ratio")
ax.set_title("Missing hours vs trend")
ax = axes[2]
ax.scatter(f["missing_hours_28d"], f["missing_hours_7d"], c=f["schlecht"].map({True:"#e74c3c", False:"#3498db"}),
           alpha=0.4, s=12)
ax.set_xlabel("Missing hours (28d)")
ax.set_ylabel("Missing hours (7d)")
ax.set_title("28d vs 7d missing hours")
for ax in axes:
    ax.legend(handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#e74c3c',label='Failing'),
                       plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',label='Other')],
              fontsize=7, loc="best")
save(fig, "06_feature_scatter.png")

# ── 7  Field visit reasons ─────────────────────────────────────────────────────
print("Plot 7: field visits")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
visits["reason_reported"].value_counts().head(10).plot(kind="barh", ax=axes[0], color="#1abc9c")
axes[0].set_title("Top visit reasons")
axes[0].invert_yaxis()
visits["outcome"].value_counts().head(10).plot(kind="barh", ax=axes[1], color="#e67e22")
axes[1].set_title("Visit outcomes")
axes[1].invert_yaxis()
save(fig, "07_field_visits.png")

# ── 8  Meter read reach decline for failing vs normal ──────────────────────────
print("Plot 8: meter reach")
m_weekly = meters.groupby(["week_start", "gateway_id"])["reach_pct"].mean().reset_index()
m_rev = review[["gateway_id", "Kategorie"]].drop_duplicates()
m_m = m_weekly.merge(m_rev, on="gateway_id", how="left")
m_m["cat"] = m_m["Kategorie"].fillna("None")
avg_reach = m_m.groupby(["week_start", "cat"])["reach_pct"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
for cat, color, label in [("Schlecht", "#e74c3c", "Failing"), ("Normal", "#2ecc71", "Normal"), ("None", "#95a5a6", "Unreviewed")]:
    sub = avg_reach[avg_reach["cat"] == cat]
    ax.plot(sub["week_start"], sub["reach_pct"], marker="o", color=color, label=label, alpha=0.8)
ax.set_ylabel("Avg meter read success rate")
ax.set_title("Weekly meter read reach by gateway category")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.tick_params(axis="x", rotation=45)
save(fig, "08_meter_reach_trend.png")

# ── 9  Region / hw_model failure propensity ────────────────────────────────────
print("Plot 9: region & hw model")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
gw = master[["gateway_id", "region", "hw_model", "antenna_type", "n_meters_installed"]].copy()
gw["is_schlecht"] = gw["gateway_id"].isin(schlecht)
region_fail = gw.groupby("region")["is_schlecht"].mean().sort_values(ascending=False)
region_fail.plot(kind="bar", ax=axes[0], color="#9b59b6")
axes[0].set_ylabel("Fraction flagged failing")
axes[0].set_title("Failure propensity by region")
hw_fail = gw.groupby("hw_model")["is_schlecht"].mean().sort_values(ascending=False)
hw_fail.plot(kind="bar", ax=axes[1], color="#f39c12")
axes[1].set_ylabel("Fraction flagged failing")
axes[1].set_title("Failure propensity by hardware model")
save(fig, "09_region_hw_failure.png")

# ── 10  Coverage over time (top 15 per week heatmap) ──────────────────────────
print("Plot 10: weekly coverage heatmap")
schlecht_list = sorted(schlecht)
pred_h = pred.copy()
pred_h["gw"] = pred_h["gateway_id"].str.replace(":", "", regex=False).str.upper()
cov = pd.DataFrame(index=schlecht_list, columns=[str(w) for w in SCORED_WEEKS])
for w in SCORED_WEEKS:
    ws = str(w)
    top15 = set(pred_h[pred_h["week_start"] == ws]["gw"])
    cov[ws] = [int(g in top15) for g in schlecht_list]
cov = cov.astype(float)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cov, cmap=["#ecf0f1", "#27ae60"], ax=ax, linewidths=0.3,
            cbar_kws={"label": "In top 15"}, yticklabels=False)
ax.set_title("Engineer-flagged failing gateways: picked each week (green=yes)")
ax.set_xlabel("Week starting")
save(fig, "10_weekly_coverage_heatmap.png")

print("\nAll plots saved to", PLOTS)