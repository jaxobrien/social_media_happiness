import streamlit as st
import pandas as pd
from utils.charts.happiness_by_year import plot_happiness_by_year
from utils.charts.happiness_social_media_chart_2 import df, build_happiness_socialmedia_chart
from utils.charts.interactive_happiness_chart import render



# Load data
df = pd.read_csv("utils/indexed_data_all_obs.csv", low_memory=False)
data = pd.read_csv("utils/FINAL_DATA.csv", low_memory=False)


# App title
st.title("If social media amplified problems, will banning it solve them?")

# Chart 1
fig = plot_happiness_by_year(df)
st.pyplot(fig)

# chart 2
fig = build_happiness_socialmedia_chart(df)
st.plotly_chart(fig, use_container_width=True)

fig = render(data)
st.plotly_chart(fig)  

from utils.charts.social_media_legislation_map import build_legislation_map

fig = build_legislation_map()
st.plotly_chart(fig, use_container_width=True)


