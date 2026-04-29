import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pybaseball import statcast

st.set_page_config(layout="wide")
st.title("⚾ MLB Statcast Engine v3 (Stable Build)")

today = datetime.date.today()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(ttl=3600)
def load_data(start, end):
    return statcast(start_dt=start, end_dt=end)

# =========================================================
# PLAYER ENTITY LAYER (FIXED)
# =========================================================
def build_entities(df):
    df = df.copy()

    batter_map = df.dropna(subset=["player_name","batter"]).drop_duplicates("batter")
    pitcher_map = df.dropna(subset=["player_name","pitcher"]).drop_duplicates("pitcher")

    df["batter_name"] = df["batter"].map(dict(zip(batter_map["batter"], batter_map["player_name"])))
    df["pitcher_name"] = df["pitcher"].map(dict(zip(pitcher_map["pitcher"], pitcher_map["player_name"])))

    return df

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def enrich(df):
    df = df.copy()

    df["barrel"] = df["launch_speed_angle"] == 6
    df["hard_hit"] = df["launch_speed"] >= 95

    swing_events = ["swinging_strike","swinging_strike_blocked","foul","hit_into_play"]
    df["swing"] = df["description"].isin(swing_events)
    df["whiff"] = df["description"].isin(["swinging_strike","swinging_strike_blocked"])

    df["in_zone"] = df["plate_x"].between(-0.83,0.83) & df["plate_z"].between(1.5,3.5)
    df["chase"] = df["swing"] & (~df["in_zone"])

    df["hit"] = df["events"].isin(["single","double","triple","home_run"])
    df["hr"] = df["events"] == "home_run"

    df["tb"] = df["events"].map({
        "single":1,"double":2,"triple":3,"home_run":4
    }).fillna(0)

    df["k"] = df["events"] == "strikeout"
    df["bb"] = df["events"] == "walk"

    ev = df["launch_speed"].fillna(0)
    la = df["launch_angle"].fillna(0)

    df["hr_prob"] = 1 / (1 + np.exp(-(0.09*(ev-94) + 0.11*(la-22))))

    return df

# =========================================================
# ENVIRONMENT MODEL
# =========================================================
PARK_FACTORS = {
    "Coors Field": 1.28,
    "Yankee Stadium": 1.12,
    "Fenway Park": 1.08,
    "Dodger Stadium": 0.97,
    "Oracle Park": 0.88
}

def apply_environment(df, park="Neutral", temp=75, wind=5):
    df = df.copy()

    park_factor = PARK_FACTORS.get(park, 1.0)
    weather_factor = (1 + (temp - 70) * 0.003) * (1 + wind * 0.01)

    df["env_factor"] = park_factor * weather_factor
    df["hr_prob_adj"] = df["hr_prob"] * df["env_factor"]

    return df

# =========================================================
# METRICS
# =========================================================
def hitters(df):
    return df.groupby("batter_name").apply(lambda x: pd.Series({

        "PA": len(x),
        "AVG": x["hit"].mean(),
        "HR": x["hr"].sum(),

        "EV": x["launch_speed"].mean(),
        "Barrel %": x["barrel"].mean(),
        "Hard Hit %": x["hard_hit"].mean(),

        "Whiff %": x["whiff"].sum()/x["swing"].sum() if x["swing"].sum() else 0,
        "Chase %": x["chase"].mean(),
        "K %": x["k"].mean(),
        "BB %": x["bb"].mean(),

        "HR Prob": x["hr_prob_adj"].mean(),
        "xwOBA": x["estimated_woba_using_speedangle"].mean(),

        "ISO": (x["tb"].sum() - x["hit"].sum()) / len(x),

    })).round(3)

def pitchers(df):
    return df.groupby("pitcher_name").apply(lambda x: pd.Series({

        "Velo": x["release_speed"].mean(),
        "Spin": x["release_spin_rate"].mean(),

        "K %": x["k"].mean(),
        "BB %": x["bb"].mean(),
        "Whiff %": x["whiff"].sum()/x["swing"].sum() if x["swing"].sum() else 0,

        "Barrel % Allowed": x["barrel"].mean(),
        "Hard Hit % Allowed": x["hard_hit"].mean(),

        "HR Allowed": x["hr"].sum(),
        "xwOBA Allowed": x["estimated_woba_using_speedangle"].mean(),

    })).round(3)

# =========================================================
# DFS MODEL
# =========================================================
def dfs(df):
    h = hitters(df)

    h["Projection"] = (
        h["HR"] * 10 +
        h["Barrel %"] * 15 +
        h["BB %"] * 2 +
        h["HR Prob"] * 20
    )

    h["Ceiling"] = h["HR Prob"] * 30 + h["Hard Hit %"] * 10

    return h.sort_values("Projection", ascending=False)

# =========================================================
# SAFE COLOR SYSTEM (NO STYLER BUGS)
# =========================================================
def color_value(val, inverse=False):
    if pd.isna(val):
        return ""

    if inverse:
        if val >= 0.25:
            return "background-color:#ff4d4d"
        elif val >= 0.18:
            return "background-color:#ffa64d"
        return "background-color:#2ecc71"
    else:
        if val >= 0.25:
            return "background-color:#2ecc71"
        elif val >= 0.15:
            return "background-color:#ffa64d"
        return "background-color:#ff4d4d"

# =========================================================
# UI
# =========================================================
st.sidebar.header("Filters")

start = st.sidebar.date_input("Start", today - datetime.timedelta(days=30))
end = st.sidebar.date_input("End", today)

temp = st.sidebar.slider("Temp", 50, 100, 75)
wind = st.sidebar.slider("Wind", 0, 20, 5)
park = st.sidebar.selectbox("Park", list(PARK_FACTORS.keys()) + ["Neutral"])

min_pa = st.sidebar.slider("Min PA", 0, 200, 20)

# =========================================================
# PIPELINE
# =========================================================
df = load_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
df = build_entities(df)
df = enrich(df)
df = apply_environment(df, park, temp, wind)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["Hitters","Pitchers","DFS","Matchups","Compare"])

# ---------------- HITTERS ----------------
with tabs[0]:
    h = hitters(df)
    h = h[h["PA"] >= min_pa]

    st.dataframe(h.sort_values("HR Prob", ascending=False), use_container_width=True)

# ---------------- PITCHERS ----------------
with tabs[1]:
    p = pitchers(df)

    st.dataframe(p.sort_values("K %", ascending=False), use_container_width=True)

# ---------------- DFS ----------------
with tabs[2]:
    st.dataframe(dfs(df), use_container_width=True)

# ---------------- MATCHUPS ----------------
with tabs[3]:
    batters = df["batter_name"].dropna().unique()
    pitchers_list = df["pitcher_name"].dropna().unique()

    b = st.selectbox("Batter", sorted(batters))
    p = st.selectbox("Pitcher", sorted(pitchers_list))

    m = df[(df["batter_name"]==b)&(df["pitcher_name"]==p)]

    if m.empty:
        st.warning("No matchup data")
    else:
        st.write({
            "PA": len(m),
            "Hits": m["hit"].sum(),
            "HR": m["hr"].sum(),
            "K%": m["k"].mean(),
            "HR Prob": m["hr_prob_adj"].mean()
        })

# ---------------- COMPARE ----------------
with tabs[4]:
    players = df["batter_name"].dropna().unique()

    p1 = st.selectbox("Player 1", players)
    p2 = st.selectbox("Player 2", players)

    st.dataframe(hitters(df).loc[[p1,p2]])
