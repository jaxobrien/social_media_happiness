"""
Social Media & Happiness Visualisation Script
==============================================
Explores how the relationship between social media use and happiness
is moderated by mental health, family relationships, demographics,
and self-esteem.

Usage:
    python social_media_happiness_viz.py

Output:
    A folder called 'charts/' containing all generated figures as PNGs.
    Each chart is also displayed interactively before saving.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "indexed_data_all_obs.csv"   # adjust if the CSV is elsewhere
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE_SEX      = {1.0: "#4C72B0", 0.0: "#DD8452"}   # male / female
PALETTE_WAVE     = sns.color_palette("viridis", 8)
PALETTE_DIVERG   = "RdYlGn"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  ✓  saved → {path}")
    plt.show()
    plt.close(fig)


def clean(df, cols):
    """Return df with only numeric, non-NaN rows for given columns."""
    return df[cols].dropna()


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Recode sex to readable labels (keep numeric copy for palette mapping)
df["sex_label"] = df["sex"].map({1.0: "Male", 0.0: "Female"})

# Bin social_media into ordinal groups (0 = none, 1 = low, 2 = medium, 3 = high)
# Values appear to be 0–3 in the raw data
sm_labels = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
df["sm_group"] = pd.Categorical(
    df["social_media"].map(sm_labels),
    categories=["None", "Low", "Medium", "High"],
    ordered=True,
)

# SDQ total as mental health burden (higher = worse)
# Self-esteem index (higher = better)
# Parent index (higher = better family relationship)

print(f"  Rows: {len(df):,}   Waves: {sorted(df['wave'].dropna().unique().astype(int).tolist())}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Happiness by social media use group (overall)
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 1 — Overall happiness by social media group …")
fig, ax = plt.subplots(figsize=(8, 5))
d = clean(df, ["sm_group", "happiness_index"])
order = ["None", "Low", "Medium", "High"]
sns.boxplot(
    data=d, x="sm_group", y="happiness_index",
    order=order, palette="Blues", width=0.5, fliersize=2, ax=ax
)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Happiness Index")
ax.set_title("Happiness by Social Media Use Group")
ax.text(0.98, 0.98, f"n = {len(d):,}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color="grey")
save(fig, "01_happiness_by_sm_group.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Happiness × social media by SEX
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 2 — By sex …")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (sex_val, sex_name) in zip(axes, [(1.0, "Male"), (0.0, "Female")]):
    d = clean(df[df["sex"] == sex_val], ["sm_group", "happiness_index"])
    colour = PALETTE_SEX[sex_val]
    sns.boxplot(data=d, x="sm_group", y="happiness_index",
                order=order, color=colour, width=0.5, fliersize=2, ax=ax)
    ax.set_title(f"{sex_name}  (n={len(d):,})")
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Happiness Index" if ax == axes[0] else "")
fig.suptitle("Happiness by Social Media Group — Split by Sex", y=1.01, fontsize=13)
plt.tight_layout()
save(fig, "02_happiness_sm_by_sex.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Mean happiness by social media group × WAVE (trend over time)
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 3 — Trend over waves …")
d = clean(df, ["sm_group", "happiness_index", "wave"])
trend = (
    d.groupby(["wave", "sm_group"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 5))
for i, grp in enumerate(order):
    sub = trend[trend["sm_group"] == grp]
    ax.plot(sub["wave"], sub["happiness_index"], marker="o",
            label=grp, color=PALETTE_WAVE[i * 2])
ax.set_xlabel("Wave")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Happiness Over Time by Social Media Use Group")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(title="Social Media", loc="upper right")
save(fig, "03_happiness_sm_trend_waves.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Mental health (SDQ total) moderates the SM–happiness link
#            Bin SDQ into Low / Medium / High burden
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 4 — Mental health moderator …")
d = clean(df, ["sm_group", "happiness_index", "sdq_total"])
d = d.copy()
d["mh_group"] = pd.cut(
    d["sdq_total"],
    bins=[d["sdq_total"].min() - 1,
          d["sdq_total"].quantile(0.33),
          d["sdq_total"].quantile(0.67),
          d["sdq_total"].max() + 1],
    labels=["Low burden", "Medium burden", "High burden"],
)

mh_mean = (
    d.groupby(["sm_group", "mh_group"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(9, 5))
palette_mh = {"Low burden": "#2ca02c", "Medium burden": "#ff7f0e", "High burden": "#d62728"}
for label, colour in palette_mh.items():
    sub = mh_mean[mh_mean["mh_group"] == label]
    ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
            marker="o", label=label, color=colour, linewidth=2)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Social Media × Happiness\nModerated by Mental Health Burden (SDQ Total)")
ax.legend(title="Mental Health")
save(fig, "04_sm_happiness_mh_moderator.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Self-esteem moderator
#            Scatter: social_media × happiness, coloured by self-esteem quartile
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 5 — Self-esteem moderator …")
d = clean(df, ["social_media", "happiness_index", "selfesteem_index"])
d = d.copy()
d["se_quartile"] = pd.qcut(d["selfesteem_index"], q=4,
                            labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

se_mean = (
    d.groupby(["social_media", "se_quartile"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(9, 5))
palette_se = sns.color_palette("RdYlGn", 4)
for i, q in enumerate(["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]):
    sub = se_mean[se_mean["se_quartile"] == q]
    ax.plot(sub["social_media"], sub["happiness_index"],
            marker="o", label=q, color=palette_se[i], linewidth=2)
ax.set_xlabel("Social Media Use (raw scale)")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Social Media × Happiness\nModerated by Self-Esteem Quartile")
ax.legend(title="Self-Esteem")
save(fig, "05_sm_happiness_selfesteem_moderator.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Parent relationship moderator
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 6 — Parent relationship moderator …")
d = clean(df, ["sm_group", "happiness_index", "parent_index"])
d = d.copy()
d["parent_group"] = pd.qcut(d["parent_index"], q=3,
                             labels=["Low support", "Medium support", "High support"])

par_mean = (
    d.groupby(["sm_group", "parent_group"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(9, 5))
palette_par = {"Low support": "#d62728", "Medium support": "#ff7f0e", "High support": "#2ca02c"}
for label, colour in palette_par.items():
    sub = par_mean[par_mean["parent_group"] == label]
    ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
            marker="o", label=label, color=colour, linewidth=2)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Social Media × Happiness\nModerated by Parental Relationship Quality")
ax.legend(title="Parental Support")
save(fig, "06_sm_happiness_parent_moderator.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Correlation heatmap: SM, happiness, and key moderators
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 7 — Correlation heatmap …")
cols_corr = {
    "social_media":      "Social Media",
    "happiness_index":   "Happiness",
    "selfesteem_index":  "Self-Esteem",
    "parent_index":      "Parent Rel.",
    "sib_index":         "Sibling Rel.",
    "sdq_total":         "SDQ Total\n(MH burden)",
    "sdq_emotion":       "SDQ Emotion",
    "sdq_conduct":       "SDQ Conduct",
    "online_hrs":        "Online Hours",
    "n_friends":         "N Friends",
    "age":               "Age",
}
d = df[list(cols_corr.keys())].dropna()
corr = d.corr()
corr.columns = list(cols_corr.values())
corr.index   = list(cols_corr.values())

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap=PALETTE_DIVERG,
    center=0, vmin=-1, vmax=1, linewidths=0.5,
    annot_kws={"size": 8}, ax=ax
)
ax.set_title("Correlation Matrix — Social Media, Happiness & Key Moderators", pad=12)
plt.tight_layout()
save(fig, "07_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Ethnicity: mean happiness by SM group (top 5 ethnic groups)
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 8 — Ethnicity comparison …")
top_eth = df["ethnicity"].value_counts().head(5).index.tolist()
d = clean(df[df["ethnicity"].isin(top_eth)],
          ["sm_group", "happiness_index", "ethnicity"])
eth_mean = (
    d.groupby(["sm_group", "ethnicity"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
palette_eth = sns.color_palette("tab10", len(top_eth))

fig, ax = plt.subplots(figsize=(10, 5))
for i, eth in enumerate(top_eth):
    sub = eth_mean[eth_mean["ethnicity"] == eth]
    short = eth[:30] + "…" if len(eth) > 30 else eth
    ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
            marker="o", label=short, color=palette_eth[i], linewidth=2)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Social Media × Happiness by Ethnicity (Top 5 Groups)")
ax.legend(title="Ethnicity", fontsize=8, title_fontsize=9)
save(fig, "08_sm_happiness_ethnicity.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Age group: mean happiness by SM group
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 9 — Age group comparison …")
d = clean(df, ["sm_group", "happiness_index", "age"])
d = d.copy()
d["age_group"] = pd.cut(d["age"], bins=[9, 12, 14, 16, 99],
                         labels=["10–12", "13–14", "15–16", "17+"])
age_mean = (
    d.groupby(["sm_group", "age_group"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)
palette_age = sns.color_palette("crest", 4)
fig, ax = plt.subplots(figsize=(9, 5))
for i, grp in enumerate(["10–12", "13–14", "15–16", "17+"]):
    sub = age_mean[age_mean["age_group"] == grp]
    ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
            marker="o", label=grp, color=palette_age[i], linewidth=2)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Mean Happiness Index")
ax.set_title("Social Media × Happiness by Age Group")
ax.legend(title="Age Group")
save(fig, "09_sm_happiness_age_group.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Composite: 2×2 facet — sex × mental health burden
# ══════════════════════════════════════════════════════════════════════════════
print("Chart 10 — Facet: sex × mental health burden …")
d = clean(df, ["sm_group", "happiness_index", "sdq_total", "sex_label"])
d = d.copy()
d["mh_group"] = pd.cut(
    d["sdq_total"],
    bins=[d["sdq_total"].min() - 1,
          d["sdq_total"].quantile(0.5),
          d["sdq_total"].max() + 1],
    labels=["Lower MH burden", "Higher MH burden"],
)
d2 = d.dropna(subset=["mh_group"])
facet_mean = (
    d2.groupby(["sm_group", "sex_label", "mh_group"], observed=True)["happiness_index"]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, mh in zip(axes, ["Lower MH burden", "Higher MH burden"]):
    sub = facet_mean[facet_mean["mh_group"] == mh]
    for sex, colour in [("Male", "#4C72B0"), ("Female", "#DD8452")]:
        s = sub[sub["sex_label"] == sex]
        ax.plot(s["sm_group"].astype(str), s["happiness_index"],
                marker="o", label=sex, color=colour, linewidth=2)
    ax.set_title(mh)
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Mean Happiness Index" if ax == axes[0] else "")
    ax.legend(title="Sex")
fig.suptitle("Social Media × Happiness: Sex & Mental Health Burden", fontsize=13, y=1.01)
plt.tight_layout()
save(fig, "10_facet_sex_mh_burden.png")

print(f"\nAll done! {len(os.listdir(OUTPUT_DIR))} charts saved to '{OUTPUT_DIR}/'")
