# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from data_aggregation import get_data
from spread_bettor import (
    get_current_week,
    analyze_betting_opportunities,
)
from config import CURRENT_SEASON

# Team logo URLs (ESPN logos)
TEAM_LOGOS = {
    "ARI": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "BUF": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "CAR": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "GB": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LA": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LV": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "SF": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "TB": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png",
}

# --- Page Configuration ---
st.set_page_config(
    page_title="NFL Betting Recommendations",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---
def check_if_correct(row):
    """Check if a betting recommendation was correct based on the final score."""
    # Ensure the game has been played and has a recommendation
    if pd.isna(row["actual_home_score"]) or row["bet_quality"] == "PASS":
        return None

    # Parse the recommendation string, e.g., "Bet on KC -7.0"
    match = re.search(r"Bet on (\w{2,3}) ([+\-]\d+\.\d+)", row["recommendation"])
    if not match:
        return None

    team_betted, spread = match.groups()
    spread = float(spread)

    actual_margin = row["actual_home_score"] - row["actual_away_score"]

    # Check if the bet won. Universal formula: team_score + spread > opponent_score
    if team_betted == row["home_team"]:
        # We bet on the home team.
        # home_score + spread > away_score  --> (home_score - away_score) + spread > 0
        # --> actual_margin + spread > 0
        return (actual_margin + spread) > 0
    else:
        # We bet on the away team.
        # away_score + spread > home_score --> (away_score - home_score) + spread > 0
        # --> -(home_score - away_score) + spread > 0 --> -actual_margin + spread > 0
        return (-actual_margin + spread) > 0


# --- Data Caching ---
@st.cache_data(ttl=3600)
def load_data():
    """Load NFL team statistics with caching"""
    return get_data()


@st.cache_data(ttl=1800)
def get_recommendations(team_stats_df, season, week):
    """Get betting recommendations with caching"""
    return analyze_betting_opportunities(team_stats_df, season=season, week=week)


# --- UI Components ---
def calculate_and_display_overall_accuracy(team_stats_df, season, current_week):
    """
    Calculates the model's overall betting accuracy from the start of the season
    up to the last completed week and displays it in the sidebar.
    """
    # Accuracy is calculated on completed weeks, starting from week 2
    if current_week <= 2:
        st.sidebar.info("Overall accuracy will be calculated after Week 2 is complete.")
        return

    total_correct_picks = 0
    total_picks_made = 0

    # Iterate through all completed weeks (up to current_week - 1)
    for week in range(2, current_week):
        # Use the cached function to get recommendations for each past week
        recommendations = get_recommendations(team_stats_df, season, week)

        if not recommendations.empty:
            # Determine correctness for each recommendation
            recommendations["is_correct"] = recommendations.apply(
                check_if_correct, axis=1
            )

            # Filter for games that had a bet placed and have a result
            picks = recommendations[
                (recommendations["bet_quality"] != "PASS")
                & (recommendations["is_correct"].notna())
            ].copy()

            if not picks.empty:
                total_correct_picks += picks["is_correct"].sum()
                total_picks_made += len(picks)

    if total_picks_made > 0:
        overall_accuracy = (total_correct_picks / total_picks_made) * 100
        st.sidebar.metric(
            label=f"Model Accuracy (Through Week {current_week - 1})",
            value=f"{overall_accuracy:.1f}%",
            help=f"Based on {total_correct_picks} correct picks out of {total_picks_made} total recommendations (Strong, Good, Lean) from Week 2 to {current_week - 1}.",
        )
    else:
        st.sidebar.info("No historical results yet to calculate overall accuracy.")


def display_weekly_results(recommendations, week):
    """Calculate and display the results for a completed week."""

    # Filter for games that had a betting recommendation and have been played
    picks = recommendations[
        (recommendations["bet_quality"] != "PASS")
        & (recommendations["is_correct"].notna())
    ].copy()

    if picks.empty:
        st.info(f"No completed betting recommendations to evaluate for Week {week}.")
        return

    correct_picks = picks["is_correct"].sum()
    total_picks = len(picks)

    if total_picks > 0:
        accuracy = (correct_picks / total_picks) * 100
        st.success(
            f"**Week {week} Results: {correct_picks} / {total_picks} Correct Picks ({accuracy:.1f}%)**"
        )
    else:
        st.info(f"No verifiable picks for Week {week}.")


def display_summary_metrics(recommendations, week):
    """Display summary metrics for the selected week using st.metric."""
    st.subheader(f"Week {week} At a Glance")

    strong_count = len(recommendations[recommendations["bet_quality"] == "STRONG BET"])
    good_count = len(recommendations[recommendations["bet_quality"] == "GOOD BET"])
    lean_count = len(recommendations[recommendations["bet_quality"] == "LEAN"])
    avg_edge = recommendations["edge"].abs().mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Games", len(recommendations))
    col2.metric("Strong Bets", f"🔥 {strong_count}")
    col3.metric("Good Bets", f"✅ {good_count}")
    col4.metric("Leans", f"↗️ {lean_count}")
    col5.metric("Avg. Edge", f"{avg_edge:.1f} pts" if pd.notna(avg_edge) else "N/A")


def display_game_card(game):
    """Display a single game card using st.expander."""

    # Determine emoji based on game status and recommendation correctness
    if pd.notna(game.get("is_correct")):
        emoji = "✅" if game["is_correct"] else "❌"
    else:
        bet_quality = game["bet_quality"]
        if bet_quality == "STRONG BET":
            emoji = "🔥"
        elif bet_quality == "GOOD BET":
            emoji = "✅"
        elif bet_quality == "LEAN":
            emoji = "↗️"
        else:
            emoji = "📊"

    expander_title = f"{emoji} **{game['recommendation']}** | {game['away_team']} @ {game['home_team']}"

    with st.expander(expander_title):
        main_cols = st.columns(3)

        # Column 1: Matchup & Scores
        with main_cols[0]:
            # Display team logos
            logo_col1, logo_col2 = st.columns(2)
            with logo_col1:
                st.image(TEAM_LOGOS.get(game["away_team"], ""), width=80)
                st.markdown(f"**{game['away_team']}** (Away)")
            with logo_col2:
                st.image(TEAM_LOGOS.get(game["home_team"], ""), width=80)
                st.markdown(f"**{game['home_team']}** (Home)")

            st.markdown("---")
            st.markdown(f"**Date:** {game['date']} {game.get('time', '')}")
            if pd.notna(game["actual_home_score"]):
                st.write(
                    "**Final Score**",
                    f"{game['away_team']}: **{game['actual_away_score']:.0f}** - {game['home_team']}: **{game['actual_home_score']:.0f}**",
                )
            else:
                st.info("Game has not been played.")

        # Column 2: Model Scores & Spreads
        with main_cols[1]:
            st.metric(
                "Model Power Score",
                f"{game['model_away_score']} - {game['model_home_score']}",
            )
            st.metric(
                "Model Spread", f"{game['home_team']} {game['model_spread']:+.1f}"
            )
            st.metric(
                "Opening Spread",
                (
                    f"{game['home_team']} {game['actual_spread']:+.1f}"
                    if game["actual_spread"] is not None
                    else "N/A"
                ),
            )

        # Column 3: Edge & Recommendation
        with main_cols[2]:
            if game["edge"] is not None:
                st.metric(
                    "Edge", f"{game['edge']:+.1f} pts", delta=f"{game['edge']:+.1f}"
                )
            st.info(
                f"**Recommendation:** {game['recommendation']} ({game['confidence']})"
            )


def main():
    st.title("🏈 NFL Betting Recommendations")

    st.sidebar.title("Settings")
    with st.spinner("Loading NFL data..."):
        team_stats_df = load_data()
        current_season = CURRENT_SEASON
        current_week = get_current_week(current_season)

    # NEW: Calculate and display overall accuracy at the top of the sidebar
    calculate_and_display_overall_accuracy(team_stats_df, current_season, current_week)
    st.sidebar.markdown("---")  # Add a separator

    selected_week = st.sidebar.selectbox(
        "Select Week", range(2, current_week + 1), index=current_week - 2
    )

    st.sidebar.markdown("### Filter Recommendations")
    filters = {
        "STRONG BET": st.sidebar.checkbox("🔥 Strong Bets (7+ Edge)", value=True),
        "GOOD BET": st.sidebar.checkbox("✅ Good Bets (5-7 Edge)", value=True),
        "LEAN": st.sidebar.checkbox("↗️ Leans (3-5 Edge)", value=True),
        "PASS": st.sidebar.checkbox("📊 Other Games", value=False),
    }

    with st.spinner(f"Analyzing Week {selected_week}..."):
        recommendations = get_recommendations(
            team_stats_df, current_season, selected_week
        )

    if recommendations.empty:
        st.error(f"No game data found for Week {selected_week}.")
        return

    # Pre-calculate if the bet was correct for all historical games
    recommendations["is_correct"] = recommendations.apply(check_if_correct, axis=1)

    # Display results for past weeks
    if selected_week < current_week:
        display_weekly_results(recommendations, selected_week)

    display_summary_metrics(recommendations, selected_week)
    st.markdown("---")

    filtered_recs = recommendations[recommendations["bet_quality"].map(filters)]

    if filtered_recs.empty:
        st.warning("No recommendations match your current filter settings.")
    else:
        st.subheader("Betting Recommendations")
        for _, game in filtered_recs.iterrows():
            display_game_card(game)

    st.markdown("---")
    st.info(
        "Disclaimer: For informational purposes only. Not financial advice. Please bet responsibly."
    )


if __name__ == "__main__":
    main()
