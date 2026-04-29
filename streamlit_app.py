import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from pybaseball import statcast

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="MLB Statcast Dashboard",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ MLB Statcast Dashboard")

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# ------------------------------------
# Caching Functions
# ------------------------------------
@st.cache_data(ttl=3600)
def get_statcast_data(start_date, end_date):
    try:
        return statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_matchups(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date.strftime('%Y-%m-%d')}&hydrate=probablePitcher"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return {}

# ------------------------------------
# Tabs
# ------------------------------------
tabs = st.tabs([
    "Matchups",
    "Hitters",
    "Pitchers",
    "Heatmaps",
    "Rolling xwOBA",
    "Home Runs"
])

# ------------------------------------
# Matchups Tab
# ------------------------------------
with tabs[0]:
    st.subheader("Probable Pitchers")

    date = st.date_input("Select Date", yesterday)

    with st.spinner("Loading matchups..."):
        data = get_matchups(date)

    matchups = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            away = game["teams"]["away"]["team"]["name"]
            home = game["teams"]["home"]["team"]["name"]

            away_p = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
            home_p = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")

            matchups.append({
                "Matchup": f"{away} @ {home}",
                "Away Pitcher": away_p,
                "Home Pitcher": home_p
            })

    if matchups:
        st.dataframe(pd.DataFrame(matchups), use_container_width=True)
    else:
        st.info("No games found.")

# ------------------------------------
# Hitters Tab
# ------------------------------------
with tabs[1]:
    st.subheader("Hitter Performance (Last 30 Days)")

    df = get_statcast_data(
        (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d")
    )

    if not df.empty:
        hitters = (
            df.groupby("player_name")[["launch_speed", "launch_angle", "estimated_woba_using_speedangle"]]
            .mean()
            .rename(columns={
                "launch_speed": "Avg EV",
                "launch_angle": "Avg LA",
                "estimated_woba_using_speedangle": "xwOBA"
            })
            .sort_values("xwOBA", ascending=False)
        )

        st.dataframe(hitters, use_container_width=True)
    else:
        st.warning("No data available.")

# ------------------------------------
# Pitchers Tab
# ------------------------------------
with tabs[2]:
    st.subheader("Pitcher Metrics")

    df = get_statcast_data(
        (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d")
    )

    if not df.empty:
        pitchers = (
            df.groupby("pitcher")[["release_speed", "pfx_x", "pfx_z"]]
            .mean()
            .rename(columns={
                "release_speed": "Velocity",
                "pfx_x": "H-Break",
                "pfx_z": "V-Break"
            })
            .sort_values("Velocity", ascending=False)
        )

        st.dataframe(pitchers, use_container_width=True)
    else:
        st.warning("No data available.")

# ------------------------------------
# Heatmaps Tab
# ------------------------------------
with tabs[3]:
    st.subheader("Pitch Location Heatmap")

    player = st.text_input("Enter Player Name")

    if player:
        df = get_statcast_data(
            (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        )

        df = df[df["player_name"] == player]

        if df.empty:
            st.warning("No data found.")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.kdeplot(
                x=df["plate_x"],
                y=df["plate_z"],
                fill=True,
                cmap="coolwarm",
                thresh=0.05,
                ax=ax
            )
            ax.set_title(player)
            st.pyplot(fig)

# ------------------------------------
# Rolling xwOBA
# ------------------------------------
with tabs[4]:
    st.subheader("Rolling xwOBA (20 PA)")

    player = st.text_input("Player Name for Trend")

    if player:
        df = get_statcast_data(
            (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        )

        df = df[df["player_name"] == player]

        if df.empty:
            st.warning("No data found.")
        else:
            df["xwOBA_rolling"] = df["estimated_woba_using_speedangle"].rolling(20).mean()
            st.line_chart(df.set_index("game_date")["xwOBA_rolling"])

# ------------------------------------
# Home Runs Tab
# ------------------------------------
with tabs[5]:
    st.subheader("Home Run Analyzer")

    start = st.date_input("Start Date", today - datetime.timedelta(days=7))
    end = st.date_input("End Date", today)

    if st.button("Analyze"):
        df = get_statcast_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        hr = df[df["events"] == "home_run"]

        if hr.empty:
            st.warning("No HRs found.")
        else:
            st.success(f"{len(hr)} home runs")

            st.dataframe(
                hr[["player_name", "pitcher", "launch_speed", "launch_angle"]],
                use_container_width=True
            )

            st.scatter_chart(hr, x="launch_angle", y="launch_speed")
