import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pybaseball import statcast

st.set_page_config(page_title="MLB Statcast Pro+", layout="wide")
st.title("⚾ MLB Statcast Pro+ Analytics Engine")

today = datetime.date.today()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(ttl=3600)
def load_data(start, end):
    return statcast(start_dt=start, end_dt=end)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def enrich(df):
    df = df.copy()

    # -----------------------
    # CONTACT QUALITY
    # -----------------------
    df["barrel"] = df["launch_speed_angle"] == 6
    df["hard_hit"] = df["launch_speed"] >= 95

    # -----------------------
    # SWING LOGIC
    # -----------------------
    swing_events = ["swinging_strike","swinging_strike_blocked","foul","hit_into_play"]
    df["swing"] = df["description"].isin(swing_events)
    df["whiff"] = df["description"].isin(["swinging_strike","swinging_strike_blocked"])

    # Zone estimate
    df["in_zone"] = df["plate_x"].between(-0.83,0.83) & df["plate_z"].between(1.5,3.5)
    df["chase"] = df["swing"] & (~df["in_zone"])

    # -----------------------
    # OUTCOMES
    # -----------------------
    df["hit"] = df["events"].isin(["single","double","triple","home_run"])
    df["hr"] = df["events"] == "home_run"

    df["tb"] = df["events"].map({
        "single":1,"double":2,"triple":3,"home_run":4
    }).fillna(0)

    df["k"] = df["events"] == "strikeout"
    df["bb"] = df["events"] == "walk"

    # -----------------------
    # HR MODEL (improved heuristic)
    # -----------------------
    ev = df["launch_speed"].fillna(0)
    la = df["launch_angle"].fillna(0)

    df["hr_prob"] = 1 / (1 + np.exp(-(0.09*(ev-94) + 0.11*(la-22))))

    return df

# =========================================================
# PARK + WEATHER
# =========================================================
PARK_FACTORS = {
    "Coors Field": 1.28,
    "Yankee Stadium": 1.12,
    "Fenway Park": 1.08,
    "Dodger Stadium": 0.97,
    "Oracle Park": 0.88
}

def apply_environment(df, temp=75, wind=5, park="Neutral"):
    df = df.copy()

    park_factor = PARK_FACTORS.get(park, 1.0)
    weather_factor = (1 + (temp - 70) * 0.003) * (1 + wind * 0.01)

    df["env_factor"] = park_factor * weather_factor
    df["hr_prob_adj"] = df["hr_prob"] * df["env_factor"]

    return df

# =========================================================
# HITTER METRICS (FULL)
# =========================================================
def hitter_metrics(df):
    return df.groupby("player_name").apply(lambda x: pd.Series({

        # Volume
        "PA": len(x),

        # Production
        "AVG": x["hit"].mean(),
        "HR": x["hr"].sum(),
        "SLG Proxy": x["tb"].sum() / len(x),
        "xwOBA": x["estimated_woba_using_speedangle"].mean(),

        # Power
        "Avg EV": x["launch_speed"].mean(),
        "Max EV": x["launch_speed"].max(),
        "Barrel %": x["barrel"].mean(),
        "Hard Hit %": x["hard_hit"].mean(),
        "HR Prob Adj": x["hr_prob_adj"].mean(),

        # Launch Profile
        "GB %": (x["launch_angle"] < 10).mean(),
        "LD %": x["launch_angle"].between(10,25).mean(),
        "FB %": (x["launch_angle"] > 25).mean(),

        # Discipline
        "Whiff %": x["whiff"].sum()/x["swing"].sum() if x["swing"].sum() else 0,
        "Chase %": x["chase"].mean(),
        "K %": x["k"].mean(),
        "BB %": x["bb"].mean(),

        # Value metrics
        "ISO Proxy": (x["tb"].sum() - x["hit"].sum()) / len(x),
        "XBH %": x["events"].isin(["double","triple","home_run"]).mean(),

    })).round(3)

# =========================================================
# PITCHER METRICS (FULL)
# =========================================================
def pitcher_metrics(df):
    return df.groupby("pitcher").apply(lambda x: pd.Series({

        # Stuff
        "Velo": x["release_speed"].mean(),
        "Max Velo": x["release_speed"].max(),
        "Spin": x["release_spin_rate"].mean(),

        # Movement
        "HB": x["pfx_x"].mean(),
        "VB": x["pfx_z"].mean(),

        # Results
        "K %": x["k"].mean(),
        "BB %": x["bb"].mean(),
        "Whiff %": x["whiff"].sum()/x["swing"].sum() if x["swing"].sum() else 0,

        # Contact suppression
        "Barrel % Allowed": x["barrel"].mean(),
        "HardHit % Allowed": x["hard_hit"].mean(),
        "HR Allowed": x["hr"].sum(),

        # Expected
        "xwOBA Allowed": x["estimated_woba_using_speedangle"].mean(),
        "HR Prob Allowed": x["hr_prob_adj"].mean(),

    })).round(3)

# =========================================================
# DFS PROJECTION
# =========================================================
def dfs_projection(df):
    hitters = hitter_metrics(df)

    hitters["Proj Pts"] = (
        hitters["HR"] * 10 +
        hitters["XBH %"] * 5 +
        hitters["BB %"] * 2 +
        hitters["HR Prob Adj"] * 15
    )

    hitters["Ceiling"] = hitters["HR Prob Adj"] * 30 + hitters["Barrel %"] * 10

    return hitters.sort_values("Proj Pts", ascending=False)

# =========================================================
# UI CONTROLS
# =========================================================
st.sidebar.header("Filters")

start = st.sidebar.date_input("Start", today - datetime.timedelta(days=30))
end = st.sidebar.date_input("End", today)

temp = st.sidebar.slider("Temperature (°F)", 50, 100, 75)
wind = st.sidebar.slider("Wind (mph)", 0, 20, 5)
park = st.sidebar.selectbox("Park", list(PARK_FACTORS.keys()) + ["Neutral"])

min_pa = st.sidebar.slider("Min PA", 0, 200, 20)

# =========================================================
# LOAD PIPELINE
# =========================================================
df = load_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
df = enrich(df)
df = apply_environment(df, temp=temp, wind=wind, park=park)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "Hitters",
    "Pitchers",
    "DFS Rankings",
    "Matchups",
    "Compare"
])

# -------------------------
# HITTERS
# -------------------------
with tabs[0]:
    h = hitter_metrics(df)
    h = h[h["PA"] >= min_pa]
    st.dataframe(h.sort_values("HR Prob Adj", ascending=False), use_container_width=True)

# -------------------------
# PITCHERS
# -------------------------
with tabs[1]:
    p = pitcher_metrics(df)
    st.dataframe(p.sort_values("K %", ascending=False), use_container_width=True)

# -------------------------
# DFS
# -------------------------
with tabs[2]:
    dfs = dfs_projection(df)
    st.dataframe(dfs, use_container_width=True)

# -------------------------
# MATCHUPS
# -------------------------
with tabs[3]:
    batter = st.selectbox("Batter", df["player_name"].dropna().unique())
    pitcher = st.selectbox("Pitcher", df["pitcher"].dropna().unique())

    m = df[(df["player_name"]==batter) & (df["pitcher"]==pitcher)]

    if m.empty:
        st.warning("No matchup data")
    else:
        st.write({
            "PA": len(m),
            "Hits": m["hit"].sum(),
            "HR": m["hr"].sum(),
            "K": m["k"].sum(),
            "HR Prob": m["hr_prob_adj"].mean()
        })

# -------------------------
# COMPARE
# -------------------------
with tabs[4]:
    players = df["player_name"].dropna().unique()

    p1 = st.selectbox("Player 1", players)
    p2 = st.selectbox("Player 2", players)

    comp = hitter_metrics(df).loc[[p1,p2]]
    st.dataframe(comp)
