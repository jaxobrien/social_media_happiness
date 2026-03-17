import streamlit as st
import pandas as pd
from utils.charts.happiness_by_year import plot_happiness_by_year

# Load data
df = pd.read_csv("utils/indexed_data_all_obs.csv", low_memory=False)

# App title
st.title("Exploring the relationship between social media use and children's happiness")

# Chart 1
fig = plot_happiness_by_year(df)
st.pyplot(fig)