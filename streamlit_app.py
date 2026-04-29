import streamlit as st
import pandas as pd
import numpy as np

from pybaseball import statcast
import plotly.express as px

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(page_title="MLB Daily Matchups", layout="wide")

st.title("⚾ MLB Daily Matchup Board")
st.caption("Starting pitchers vs projected hitters using Statcast analytics")

# ----------------------------
# LOAD SMALL STATCAST SAMPLE (SAFE)
# ----------------------------
@st.cache_data
def load_data():
    df = statcast("2024-04-01", "2024-04-07")

    df = df.dropna(subset=["launch_speed", "launch_angle", "events"])
    return df

data = load_data()

# ----------------------------
# SIMULATED DAILY GAMES (PLACEHOLDER STRUCTURE)
# In production you'd replace this with MLB schedule API
# ----------------------------
games = [
    {"away": "NYY", "home": "BOS", "pitcher": 425844},
    {"away": "LAD", "home": "SF", "pitcher": 477132},
    {"away": "HOU", "home": "TEX", "pitcher": 621244},
]

st.header("📅 Today's Matchups")

# ----------------------------
# PLAYER STATS HELPERS
# ----------------------------
def batter_stats(df, batter_id):
    b = df[df["batter"] == batter_id]
    if b.empty:
        return None

    return {
        "avg_ev": b["launch_speed"].mean(),
        "avg_la": b["launch_angle"].mean(),
        "hr_rate": (b["events"] == "home_run").mean()
    }


def pitcher_stats(df, pitcher_id):
    p = df[df["pitcher"] == pitcher_id]
    if
