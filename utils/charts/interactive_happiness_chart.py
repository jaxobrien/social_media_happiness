"""
interactive_happiness_chart.py
-------------------------------
Renders an interactive Happiness Index chart inside a Streamlit app.

Usage in streamlit_app.py:
    from utils.charts.interactive_happiness_chart import render

    fig = render(data)
    st.plotly_chart(fig, use_container_width=True)

Requirements:
    pip install pandas plotly streamlit
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Colour palettes ───────────────────────────────────────────────────────────
_PALETTE = {
    "global":  "#f0c060",
    "Female":  ["#e07090", "#d04870", "#f090b0", "#c03060", "#ff80a0"],
    "Male":    ["#50a0d0", "#3080b0", "#70c0f0", "#2060a0", "#90d0ff"],
    "neutral": ["#70d0a0", "#a0e060", "#50c0b0", "#90f0c0", "#60e0d0"],
}

_BG      = "#0d0f14"
_SURFACE = "#141720"
_GRID    = "#1e2330"
_TEXT    = "#e8eaf0"
_MUTED   = "#5a6070"

# ── Weighted mean helper ──────────────────────────────────────────────────────
def _wmean(subset):
    if subset.empty:
        return None
    return (subset["mean_happiness_index"] * subset["n"]).sum() / subset["n"].sum()

# ── Trace builder ─────────────────────────────────────────────────────────────
def _make_trace(name, x, y, color, dash="solid"):
    return go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2.5, dash=dash),
        marker=dict(color=color, size=7, line=dict(color=_BG, width=2)),
        hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Happiness: %{{y:.3f}}<extra></extra>",
    )

# ── Grouped means helper ──────────────────────────────────────────────────────
@st.cache_data
def _prepare_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either FINAL_DATA.csv (individual rows) or grouped_means.csv.
    Returns a grouped_means-style DataFrame with columns:
        year, age, sex, n, mean_happiness_index
    """
    df = df.copy()

    # If already grouped (has mean_happiness_index column), use as-is
    if "mean_happiness_index" in df.columns:
        df = df.dropna(subset=["age", "year", "mean_happiness_index"])
        df["age"]  = df["age"].astype(int)
        df["year"] = df["year"].astype(int)
        return df[["year", "age", "sex", "n", "mean_happiness_index"]]

    # Otherwise aggregate from individual rows
    if "happiness_index" not in df.columns:
        raise ValueError("DataFrame must contain either 'happiness_index' or 'mean_happiness_index'.")

    df = df.dropna(subset=["happiness_index", "age", "year"])
    df["age"]  = df["age"].astype(int)
    df["year"] = df["year"].astype(int)

    grouped = (
        df.groupby(["year", "age", "sex"])
        .agg(n=("happiness_index", "count"), mean_happiness_index=("happiness_index", "mean"))
        .reset_index()
    )
    return grouped

# ── Main render function ──────────────────────────────────────────────────────
def render(df: pd.DataFrame) -> go.Figure:
    """
    Build and return a Plotly figure. All sidebar controls are rendered here.
    Call as:
        fig = render(data)
        st.plotly_chart(fig, use_container_width=True)
    """
    gdf   = _prepare_grouped(df)
    years = sorted(gdf["year"].unique())
    ages  = sorted(gdf["age"].unique())

    global_means = {
        y: _wmean(gdf[gdf["year"] == y]) for y in years
    }

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.header("Happiness Index Filters")

    show_global = st.sidebar.checkbox("Show global mean", value=True)

    st.sidebar.markdown("**Sex**")
    show_female = st.sidebar.checkbox("Female", value=False)
    show_male   = st.sidebar.checkbox("Male",   value=False)

    st.sidebar.markdown("**Age groups**")
    selected_ages = [
        age for age in ages
        if st.sidebar.checkbox(f"Age {age}", value=False, key=f"age_{age}")
    ]

    split_female = split_male = False
    if selected_ages:
        st.sidebar.markdown("**Sub-group sex split**")
        split_female = st.sidebar.checkbox("Female (age split)", value=False, key="split_f")
        split_male   = st.sidebar.checkbox("Male (age split)",   value=False, key="split_m")

    # ── Build traces ──────────────────────────────────────────────────────────
    traces = []

    if show_global:
        traces.append(_make_trace(
            "Global mean", years,
            [global_means[y] for y in years],
            color=_PALETTE["global"], dash="dot",
        ))

    if not selected_ages:
        if show_female:
            vals = [_wmean(gdf[(gdf["year"] == y) & (gdf["sex"] == "Female")]) for y in years]
            traces.append(_make_trace("Female", years, vals, color=_PALETTE["Female"][0]))
        if show_male:
            vals = [_wmean(gdf[(gdf["year"] == y) & (gdf["sex"] == "Male")]) for y in years]
            traces.append(_make_trace("Male", years, vals, color=_PALETTE["Male"][0]))
    else:
        for ai, age in enumerate(selected_ages):
            show_combined = not split_female and not split_male
            if show_combined:
                color = _PALETTE["neutral"][ai % len(_PALETTE["neutral"])]
                vals  = [_wmean(gdf[(gdf["year"] == y) & (gdf["age"] == age)]) for y in years]
                traces.append(_make_trace(f"Age {age}", years, vals, color=color))
            if split_female:
                color = _PALETTE["Female"][ai % len(_PALETTE["Female"])]
                vals  = [_wmean(gdf[(gdf["year"] == y) & (gdf["age"] == age) & (gdf["sex"] == "Female")]) for y in years]
                traces.append(_make_trace(f"Age {age} · Female", years, vals, color=color, dash="dash"))
            if split_male:
                color = _PALETTE["Male"][ai % len(_PALETTE["Male"])]
                vals  = [_wmean(gdf[(gdf["year"] == y) & (gdf["age"] == age) & (gdf["sex"] == "Male")]) for y in years]
                traces.append(_make_trace(f"Age {age} · Male", years, vals, color=color))

    if not traces:
        st.info("Select at least one option in the sidebar to display a series.")
        return go.Figure()

    # ── Assemble figure ───────────────────────────────────────────────────────
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text="Happiness Index Over Time",
            font=dict(family="Georgia, serif", size=24, color=_TEXT),
            x=0.0, xanchor="left",
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_SURFACE,
        font=dict(family="'Courier New', monospace", color=_TEXT),
        hovermode="x unified",
        legend=dict(
            bgcolor="#1a1e2a",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis=dict(
            title="Year",
            tickmode="array",
            tickvals=years,
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(size=11, color=_MUTED),
            title_font=dict(color=_MUTED),
        ),
        yaxis=dict(
            title="Mean Happiness Index",
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(size=11, color=_MUTED),
            title_font=dict(color=_MUTED),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )

    return fig