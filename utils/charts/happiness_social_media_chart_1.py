import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────
# CSV lives in utils/, one level above this script (utils/charts/)
_DATA_PATH = Path(__file__).resolve().parent.parent / "indexed_data_all_obs.csv"
df = pd.read_csv(_DATA_PATH)


def build_happiness_socialmedia_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build an interactive Plotly figure showing mean happiness index
    vs social media use (online_hrs), with a year slider.

    Each trace is one age group; points are group means connected by
    lines, with +/- 1 SE error bars — mirroring the R ggplot version.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: happiness_index, online_hrs, age, year.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # ── 1. Midpoint mapping for online_hrs (0–4 ordinal scale) ──
    hrs_midpoint = {0: 0, 1: 0.5, 2: 2, 3: 5, 4: 7}

    # ── 2. Drop rows missing key variables ──────────────────────
    df_clean = df.dropna(subset=["happiness_index", "online_hrs", "age", "year"]).copy()
    df_clean["online_hrs_mid"] = df_clean["online_hrs"].map(hrs_midpoint)
    df_clean["age"] = df_clean["age"].astype(int)

    # ── 3. Compute group means and SE ───────────────────────────
    grouped = (
        df_clean
        .groupby(["year", "age", "online_hrs_mid"])["happiness_index"]
        .agg(mean_hap="mean", se_hap=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )

    years = sorted(df_clean["year"].unique())
    ages  = sorted(df_clean["age"].unique())

    # ── 4. Colour palette (turbo-style, one colour per age) ─────
    def turbo_colours(n):
        turbo = [
            "#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4",
            "#24efa2", "#6bfc5b", "#bcf735", "#f1ca3a", "#fb8022",
            "#e83426", "#b11901", "#7a0403",
        ]
        indices = [int(i * (len(turbo) - 1) / max(n - 1, 1)) for i in range(n)]
        return [turbo[i] for i in indices]

    colour_map = dict(zip(ages, turbo_colours(len(ages))))

    # ── 5. Build figure — one set of traces per year ─────────────
    fig = go.Figure()

    x_tickvals = [0, 0.5, 2, 5, 7]
    x_ticktext = ["None", "<1 hr", "1–3 hrs", "4–6 hrs", "7+ hrs"]

    year_trace_indices: dict = {}
    trace_idx = 0
    ages_seen = set()

    for yr in years:
        yr_data   = grouped[grouped["year"] == yr]
        yr_indices = []

        for age in ages:
            age_data = yr_data[yr_data["age"] == age].sort_values("online_hrs_mid")

            if age_data.empty:
                continue

            colour = colour_map[age]

            fig.add_trace(go.Scatter(
                x           = age_data["online_hrs_mid"],
                y           = age_data["mean_hap"],
                mode        = "lines+markers",
                name        = f"Age {age}",
                legendgroup = f"Age {age}",
                showlegend  = bool(age not in ages_seen),
                visible     = bool(yr == years[0]),
                line        = dict(color=colour, width=2),
                marker      = dict(color=colour, size=8),
                error_y     = dict(
                    type      = "data",
                    array     = age_data["se_hap"].values,
                    visible   = True,
                    color     = colour,
                    thickness = 1.5,
                    width     = 6,
                ),
                hovertemplate=(
                    f"<b>Age {age}</b><br>"
                    "Social media: %{x}<br>"
                    "Mean happiness: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

            ages_seen.add(age)
            yr_indices.append(trace_idx)
            trace_idx += 1

        year_trace_indices[yr] = yr_indices

    total_traces = trace_idx

    # ── 6. Slider steps ──────────────────────────────────────────
    steps = []
    for yr in years:
        visible = [False] * total_traces
        for i in year_trace_indices[yr]:
            visible[i] = True

        steps.append(dict(
            method = "update",
            label  = str(yr),
            args   = [
                {"visible": visible},
                {"title": f"Happiness vs Social Media Use — {yr}"},
            ],
        ))

    sliders = [dict(
        active       = 0,
        currentvalue = dict(prefix="Year: ", font=dict(size=14)),
        pad          = dict(t=50),
        steps        = steps,
    )]

    # ── 7. Layout ─────────────────────────────────────────────────
    fig.update_layout(
        title      = f"Happiness vs Social Media Use — {years[0]}",
        title_font = dict(size=18),
        xaxis      = dict(
            title     = "Average Social Media Use on a Weekday",
            tickvals  = x_tickvals,
            ticktext  = x_ticktext,
            range     = [-0.5, 8],
            showgrid  = True,
            gridcolor = "rgba(200,200,200,0.3)",
        ),
        yaxis      = dict(
            title    = "Happiness Index (higher = happier)",
            range    = [1, 7],
            tickvals = list(range(1, 8)),
            ticktext = ["1 (unhappy)", "2", "3", "4", "5", "6", "7 (happy)"],
            showgrid = True,
            gridcolor= "rgba(200,200,200,0.3)",
        ),
        sliders       = sliders,
        legend        = dict(title="Age", tracegroupgap=2),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        font          = dict(family="Arial", size=12),
        margin        = dict(l=60, r=40, t=80, b=100),
        hovermode     = "closest",
    )

    return fig
