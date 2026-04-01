# utils/charts/social_media_legislation_map.py
#
# Builds a Plotly choropleth map showing the status of children's social
# media age-restriction legislation around the world.
#
# Returns a single Plotly figure object ready to be called into a
# Streamlit app with st.plotly_chart(fig, use_container_width=True).
#
# Dependencies: plotly, pandas
# Install: pip install plotly pandas

import sys
import os

# Allow the script to find the utils data module whether called directly
# or imported from a parent directory (e.g. a Streamlit app one level up).
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
from social_media_legislation_data import (
    LEGISLATION_DATA,
    STATUS_ORDER,
    STATUS_COLORS,
)


def build_legislation_map() -> px.choropleth:
    """
    Build and return a Plotly choropleth figure showing children's social
    media legislation status by country.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pd.DataFrame(LEGISLATION_DATA)
    df["status"] = pd.Categorical(df["status"], categories=STATUS_ORDER, ordered=True)

    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"status": STATUS_ORDER},
        title=(
            "Children's Social Media Regulation Around the World<br>"
            "<sup>Legislative status on minimum age requirements for social media access</sup>"
        ),
    )

    # Style the map
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="white",
        countrywidth=0.4,
        showland=True,
        landcolor="#d9d9d9",   # unmatched countries appear as 'No data' grey
        showocean=True,
        oceancolor="#f0f4f8",
        showlakes=False,
        projection_type="natural earth",
    )

    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "family": "Arial"},
        },
        legend={
            "title": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
            "traceorder": "normal",
        },
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        annotations=[
            {
                "text": (
                    "Source: UNICEF, 'Drawing a line in digital spaces' (March 2026) | "
                    "Data reflects status as of 13 March 2026."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.18,
                "showarrow": False,
                "font": {"size": 9, "color": "grey"},
                "align": "left",
            }
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Tidy hover tooltip
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>Status: %{customdata[0]}<extra></extra>",
        marker_line_color="white",
        marker_line_width=0.4,
    )

    return fig


# ── Allow the script to be run directly for a quick preview ──────────────
if __name__ == "__main__":
    fig = build_legislation_map()
    fig.show()
