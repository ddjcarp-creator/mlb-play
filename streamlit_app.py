import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from pybaseball import statcast

# ------------------------------------
# Streamlit page config
# ------------------------------------
st.set_page_config(
    page_title="MLB Statcast Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ MLB Statcast Dashboard")

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

tabs = st.tabs([
    "Matchups",
    "Hitters",
    "Pitchers",
    "Zone Heatmaps",
    "Rolling xwOBA",
    "Home Run Analyzer"
])

# ------------------------------------
# Utility: fetch Statcast data
# ------------------------------------
@st.cache_data
def get_statcast_data(start_date, end_date):
    """Query Statcast data safely and cache results."""
    try:
        df = statcast(start_dt=start_date, end_dt=end_date)
        return df
    except Exception as e:
        st.error(f"Failed to load Statcast data: {e}")
        return pd.DataFrame()

# ------------------------------------
# Tab 1: Matchups (simplified)
# ------------------------------------
with tabs[0]:
    st.subheader("Probable Pitchers and Matchups")
    date = st.date_input("Game Date", yesterday)
    try:
        url = f"[statsapi.mlb.com](https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date.strftime()'%Y-%m-%d')}&hydrate=probablePitcher"
        data = pd.read_json(url)
    except Exception:
        st.warning("Could not load matchup data.")
        data = pd.DataFrame()

    if not data.empty:
        matchups = []
        for day in data.get("dates", []):
            for game in day.get("games", []):
                away = game["teams"]["away"]["team"]["name"]
                home = game["teams"]["home"]["team"]["name"]
                away_p = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
                home_p = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")
                matchups.append({"Matchup": f"{away} @ {home}", "Away Pitcher": away_p, "Home Pitcher": home_p})
        st.dataframe(pd.DataFrame(matchups), use_container_width=True)
    else:
        st.info("No games found or MLB API unavailable today.")

# ------------------------------------
# Tab 2: Hitters
# ------------------------------------
with tabs[1]:
    st.subheader("Recent Hitter Performance (Last 30 Days)")
    start = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    df = get_statcast_data(start, end)
    if not df.empty:
        hitter_stats = (
            df.groupby("player_name")[["launch_speed", "launch_angle", "estimated_woba_using_speedangle"]]
            .mean()
            .round(2)
            .rename(columns={
                "launch_speed": "Avg EV",
                "launch_angle": "Avg LA",
                "estimated_woba_using_speedangle": "xwOBA"
            })
            .sort_values("xwOBA", ascending=False)
        )
        st.dataframe(
            hitter_stats.style.background_gradient(cmap="RdYlGn_r"),
            use_container_width=True
        )
    else:
        st.warning("No data available right now.")

# ------------------------------------
# Tab 3: Pitchers
# ------------------------------------
with tabs[2]:
    st.subheader("Recent Pitcher Performance (Last 30 Days)")
    start = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    df = get_statcast_data(start, end)
    if not df.empty:
        pitcher_stats = (
            df.groupby("pitcher")[["release_speed", "pfx_x", "pfx_z"]]
            .mean()
            .rename(columns={
                "release_speed": "Avg Velo (mph)",
                "pfx_x": "Horizontal Break",
                "pfx_z": "Vertical Break"
            })
            .sort_values("Avg Velo (mph)", ascending=False)
        )
        st.dataframe(
            pitcher_stats.style.background_gradient(cmap="YlOrRd_r"),
            use_container_width=True
        )
    else:
        st.warning("No data available right now.")

# ------------------------------------
# Tab 4: Zone Heatmaps
# ------------------------------------
with tabs[3]:
    st.subheader("Zone Heatmaps")
    player = st.text_input("Enter Player Name (exactly as in Statcast)")
    if player:
        df = get_statcast_data((today - datetime.timedelta(days=60)).strftime("%Y-%m-%d"),
                               today.strftime("%Y-%m-%d"))
        df = df[df["player_name"] == player]
        if df.empty:
            st.warning("No matching Statcast data found.")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.kdeplot(x=df["plate_x"], y=df["plate_z"], fill=True, cmap="coolwarm", ax=ax, thresh=0.05)
            ax.set_title(f"Pitch Location Heatmap: {player}")
            st.pyplot(fig)

# ------------------------------------
# Tab 5: Rolling xwOBA
# ------------------------------------
with tabs[4]:
    st.subheader("Rolling xwOBA Tracker (20 PA Rolling)")
    player = st.text_input("Enter Hitter Name for Rolling Graph")
    if player:
        df = get_statcast_data((today - datetime.timedelta(days=90)).strftime("%Y-%m-%d"),
                               today.strftime("%Y-%m-%d"))
        df_p = df[df["player_name"] == player]
        if df_p.empty:
            st.warning("No data found for that player.")
        else:
            df_p["xwOBA_roll20"] = df_p["estimated_woba_using_speedangle"].rolling(20).mean()
            st.line_chart(df_p.set_index("game_date")["xwOBA_roll20"])

# ------------------------------------
# Tab 6: Home Runs Analyzer
# ------------------------------------
with tabs[5]:
    st.subheader("Home Run Analyzer")
    start_date = st.date_input("Start Date", today - datetime.timedelta(days=7))
    end_date = st.date_input("End Date", today)

    if st.button("Analyze Home Runs"):
        data = get_statcast_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        hr = data[data["events"] == "home_run"]

        if hr.empty:
            st.warning("No home runs found in this range.")
        else:
            st.success(f"{len(hr)} home runs found.")
            st.dataframe(
                hr[["batter_name", "pitcher_name", "launch_speed", "launch_angle", "events"]],
                use_container_width=True
            )
            st.subheader("Launch Angle vs Exit Velocity (Home Runs)")
            st.scatter_chart(hr, x="launch_angle", y="launch_speed")
