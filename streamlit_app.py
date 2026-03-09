import streamlit as st

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media & Happiness Explorer",
    page_icon="📊",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PALETTE_SEX    = {1.0: "#4C72B0", 0.0: "#DD8452"}
PALETTE_WAVE   = sns.color_palette("viridis", 8)
PALETTE_DIVERG = "RdYlGn"
SM_ORDER       = ["None", "Low", "Medium", "High"]
SM_LABELS      = {0: "None", 1: "Low", 2: "Medium", 3: "High"}

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("utils/indexed_data_all_obs.csv", low_memory=False)
    df["sex_label"] = df["sex"].map({1.0: "Male", 0.0: "Female"})
    df["sm_group"] = pd.Categorical(
        df["social_media"].map(SM_LABELS),
        categories=SM_ORDER, ordered=True,
    )
    return df

df = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(frame, cols):
    return frame[cols].dropna()

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — global filters
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("🔧 Global Filters")

all_waves = sorted(df["wave"].dropna().unique().astype(int).tolist())
sel_waves = st.sidebar.multiselect("Waves", all_waves, default=all_waves)

all_sexes = ["Male", "Female"]
sel_sexes = st.sidebar.multiselect("Sex", all_sexes, default=all_sexes)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
sel_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

# Apply filters
mask = (
    df["wave"].isin(sel_waves) &
    df["sex_label"].isin(sel_sexes) &
    df["age"].between(sel_age[0], sel_age[1])
)
dff = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Observations", f"{len(dff):,}")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📱 Social Media & Happiness Explorer")
st.markdown(
    "Explore how the relationship between **social media use** and **happiness** "
    "is shaped by mental health, family relationships, demographics, and self-esteem. "
    "Use the sidebar to filter by wave, sex, and age."
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · Overview")

col1, col2 = st.columns(2)

# Chart 1 — Boxplot
with col1:
    st.subheader("Happiness by Social Media Use")
    d = clean(dff, ["sm_group", "happiness_index"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=d, x="sm_group", y="happiness_index",
                order=SM_ORDER, palette="Blues", width=0.5, fliersize=2, ax=ax)
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Happiness Index")
    ax.text(0.98, 0.98, f"n = {len(d):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="grey")
    show(fig)

# Chart 7 — Correlation heatmap
with col2:
    st.subheader("Correlation Matrix")
    cols_corr = {
        "social_media":     "Social Media",
        "happiness_index":  "Happiness",
        "selfesteem_index": "Self-Esteem",
        "parent_index":     "Parent Rel.",
        "sib_index":        "Sibling Rel.",
        "sdq_total":        "SDQ Total",
        "sdq_emotion":      "SDQ Emotion",
        "sdq_conduct":      "SDQ Conduct",
        "online_hrs":       "Online Hrs",
        "n_friends":        "N Friends",
        "age":              "Age",
    }
    d = dff[list(cols_corr.keys())].dropna()
    corr = d.corr()
    corr.columns = list(cols_corr.values())
    corr.index   = list(cols_corr.values())
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=PALETTE_DIVERG,
                center=0, vmin=-1, vmax=1, linewidths=0.4,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Key Variables")
    plt.tight_layout()
    show(fig)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Trends Over Time
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · Trends Over Time")
st.subheader("Mean Happiness Over Waves by Social Media Group")

d = clean(dff, ["sm_group", "happiness_index", "wave"])
trend = (
    d.groupby(["wave", "sm_group"], observed=True)["happiness_index"]
    .mean().reset_index()
)
fig, ax = plt.subplots(figsize=(10, 4))
for i, grp in enumerate(SM_ORDER):
    sub = trend[trend["sm_group"] == grp]
    ax.plot(sub["wave"], sub["happiness_index"], marker="o",
            label=grp, color=PALETTE_WAVE[i * 2], linewidth=2)
ax.set_xlabel("Wave")
ax.set_ylabel("Mean Happiness Index")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(title="Social Media", loc="upper right")
show(fig)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Demographics
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 · Demographics")

col1, col2 = st.columns(2)

# Chart 2 — By sex
with col1:
    st.subheader("By Sex")
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    for ax, (sex_val, sex_name) in zip(axes, [(1.0, "Male"), (0.0, "Female")]):
        d = clean(dff[dff["sex"] == sex_val], ["sm_group", "happiness_index"])
        sns.boxplot(data=d, x="sm_group", y="happiness_index",
                    order=SM_ORDER, color=PALETTE_SEX[sex_val],
                    width=0.5, fliersize=2, ax=ax)
        ax.set_title(f"{sex_name}  (n={len(d):,})")
        ax.set_xlabel("Social Media Use")
        ax.set_ylabel("Happiness Index" if ax == axes[0] else "")
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    show(fig)

# Chart 9 — By age group
with col2:
    st.subheader("By Age Group")
    d = clean(dff, ["sm_group", "happiness_index", "age"])
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], bins=[9, 12, 14, 16, 99],
                             labels=["10–12", "13–14", "15–16", "17+"])
    age_mean = (
        d.groupby(["sm_group", "age_group"], observed=True)["happiness_index"]
        .mean().reset_index()
    )
    palette_age = sns.color_palette("crest", 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, grp in enumerate(["10–12", "13–14", "15–16", "17+"]):
        sub = age_mean[age_mean["age_group"] == grp]
        ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
                marker="o", label=grp, color=palette_age[i], linewidth=2)
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Mean Happiness Index")
    ax.legend(title="Age Group")
    show(fig)

# Chart 8 — By ethnicity
st.subheader("By Ethnicity (Top 5 Groups)")
top_eth = dff["ethnicity"].value_counts().head(5).index.tolist()
d = clean(dff[dff["ethnicity"].isin(top_eth)],
          ["sm_group", "happiness_index", "ethnicity"])
eth_mean = (
    d.groupby(["sm_group", "ethnicity"], observed=True)["happiness_index"]
    .mean().reset_index()
)
palette_eth = sns.color_palette("tab10", len(top_eth))
fig, ax = plt.subplots(figsize=(10, 4))
for i, eth in enumerate(top_eth):
    sub = eth_mean[eth_mean["ethnicity"] == eth]
    short = eth[:35] + "…" if len(eth) > 35 else eth
    ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
            marker="o", label=short, color=palette_eth[i], linewidth=2)
ax.set_xlabel("Social Media Use")
ax.set_ylabel("Mean Happiness Index")
ax.legend(title="Ethnicity", fontsize=8, title_fontsize=9)
show(fig)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Mental Health
# ══════════════════════════════════════════════════════════════════════════════
st.header("4 · Mental Health")

col1, col2 = st.columns(2)

# Chart 4 — SDQ moderator
with col1:
    st.subheader("Moderated by SDQ Burden")
    d = clean(dff, ["sm_group", "happiness_index", "sdq_total"]).copy()
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
        .mean().reset_index()
    )
    palette_mh = {"Low burden": "#2ca02c", "Medium burden": "#ff7f0e", "High burden": "#d62728"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, colour in palette_mh.items():
        sub = mh_mean[mh_mean["mh_group"] == label]
        ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
                marker="o", label=label, color=colour, linewidth=2)
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Mean Happiness Index")
    ax.legend(title="Mental Health")
    show(fig)

# Chart 10 — Facet: sex × MH burden
with col2:
    st.subheader("Sex × Mental Health Burden")
    d = clean(dff, ["sm_group", "happiness_index", "sdq_total", "sex_label"]).copy()
    d["mh_group"] = pd.cut(
        d["sdq_total"],
        bins=[d["sdq_total"].min() - 1, d["sdq_total"].quantile(0.5), d["sdq_total"].max() + 1],
        labels=["Lower MH burden", "Higher MH burden"],
    )
    d2 = d.dropna(subset=["mh_group"])
    facet_mean = (
        d2.groupby(["sm_group", "sex_label", "mh_group"], observed=True)["happiness_index"]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    for ax, mh in zip(axes, ["Lower MH burden", "Higher MH burden"]):
        sub = facet_mean[facet_mean["mh_group"] == mh]
        for sex, colour in [("Male", "#4C72B0"), ("Female", "#DD8452")]:
            s = sub[sub["sex_label"] == sex]
            ax.plot(s["sm_group"].astype(str), s["happiness_index"],
                    marker="o", label=sex, color=colour, linewidth=2)
        ax.set_title(mh, fontsize=10)
        ax.set_xlabel("Social Media Use")
        ax.set_ylabel("Mean Happiness" if ax == axes[0] else "")
        ax.legend(title="Sex", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    show(fig)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Self-Esteem & Family
# ══════════════════════════════════════════════════════════════════════════════
st.header("5 · Self-Esteem & Family Relationships")

col1, col2 = st.columns(2)

# Chart 5 — Self-esteem moderator
with col1:
    st.subheader("Moderated by Self-Esteem")
    d = clean(dff, ["social_media", "happiness_index", "selfesteem_index"]).copy()
    d["se_quartile"] = pd.qcut(d["selfesteem_index"], q=4,
                                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    se_mean = (
        d.groupby(["social_media", "se_quartile"], observed=True)["happiness_index"]
        .mean().reset_index()
    )
    palette_se = sns.color_palette("RdYlGn", 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, q in enumerate(["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]):
        sub = se_mean[se_mean["se_quartile"] == q]
        ax.plot(sub["social_media"], sub["happiness_index"],
                marker="o", label=q, color=palette_se[i], linewidth=2)
    ax.set_xlabel("Social Media Use (raw scale)")
    ax.set_ylabel("Mean Happiness Index")
    ax.legend(title="Self-Esteem")
    show(fig)

# Chart 6 — Parent relationship moderator
with col2:
    st.subheader("Moderated by Parental Support")
    d = clean(dff, ["sm_group", "happiness_index", "parent_index"]).copy()
    d["parent_group"] = pd.qcut(d["parent_index"], q=3,
                                 labels=["Low support", "Medium support", "High support"])
    par_mean = (
        d.groupby(["sm_group", "parent_group"], observed=True)["happiness_index"]
        .mean().reset_index()
    )
    palette_par = {"Low support": "#d62728", "Medium support": "#ff7f0e", "High support": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, colour in palette_par.items():
        sub = par_mean[par_mean["parent_group"] == label]
        ax.plot(sub["sm_group"].astype(str), sub["happiness_index"],
                marker="o", label=label, color=colour, linewidth=2)
    ax.set_xlabel("Social Media Use")
    ax.set_ylabel("Mean Happiness Index")
    ax.legend(title="Parental Support")
    show(fig)

st.divider()
st.caption("Data explorer built with Streamlit · Charts powered by Matplotlib & Seaborn")