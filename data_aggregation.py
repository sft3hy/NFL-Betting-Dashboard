import nflreadpy as nfl
import pandas as pd
from config import CURRENT_SEASON
import streamlit as st

# --- NEW: Import the helper function for injuries ---
from injury_helper import get_nfl_teams_with_injuries_df


def get_data():
    # Load team-level weekly stats
    team_stats = nfl.load_team_stats(summary_level="week")
    nfl_df = team_stats.to_pandas()

    # Define the subset of columns for your betting dashboard
    betting_columns = [
        "season",
        "week",
        "team",
        "opponent_team",
        "season_type",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "sacks_suffered",
        "passing_first_downs",
        "passing_epa",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "rushing_epa",
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_fumbles_forced",
        "def_tds",
        "fg_made",
        "fg_att",
        "fg_long",
        "pat_made",
        "pat_att",
        "penalties",
        "penalty_yards",
    ]

    # Create your focused DataFrame
    nfl_betting_df = nfl_df.copy()
    missing = [c for c in betting_columns if c not in nfl_betting_df.columns]
    if missing:
        print(f"⚠️ Warning: Some betting columns are missing in team stats: {missing}")
    nfl_betting_df = nfl_betting_df[
        [c for c in betting_columns if c in nfl_betting_df.columns]
    ]

    # Attach actual game scores (team_score and opponent_score)
    try:
        games = nfl.load_schedules()
        games_df = games.to_pandas()
        scores_df = build_scores_from_games_df(games_df)
        print("✓ Loaded scores from nfl.load_schedules()")
    except Exception as e:
        print(f"⚠️ Could not load game-level data (nfl.load_schedules): {e}")
        scores_df = None

    if scores_df is not None:
        # merge scores into betting df
        for c in ["season", "week", "team"]:
            if c in scores_df.columns and c in nfl_betting_df.columns:
                scores_df[c] = scores_df[c].astype(
                    nfl_betting_df[c].dtype, errors="ignore"
                )
        nfl_betting_df = nfl_betting_df.merge(
            scores_df[["season", "week", "team", "team_score", "opponent_score"]],
            on=["season", "week", "team"],
            how="left",
        )
        print("✓ Merged team/opponent scores into betting df")
    else:
        nfl_betting_df["team_score"] = pd.NA
        nfl_betting_df["opponent_score"] = pd.NA
        print("⚠️ Scores were not available; added empty score columns")

    # --- UPDATED: Load and process injury data using the new helper ---
    try:
        # 1. Call the new function to get season-level injury data
        with st.spinner("Pulling all injury data..."):
            season_injuries_df = get_nfl_teams_with_injuries_df(season=CURRENT_SEASON)

        # 2. Calculate impact scores from the new DataFrame format
        injury_impact = calculate_injury_impact(
            season_injuries_df, season=CURRENT_SEASON
        )

        # 3. Merge injury data with team stats (on season and team, not week)
        nfl_betting_df = nfl_betting_df.merge(
            injury_impact, on=["season", "team"], how="left"
        )

        # 4. Fill missing injury values with 0
        nfl_betting_df["injury_impact_score"] = nfl_betting_df[
            "injury_impact_score"
        ].fillna(0)
        nfl_betting_df["noteworthy_injuries_count"] = nfl_betting_df[
            "noteworthy_injuries_count"
        ].fillna(0)

        print("✓ Successfully integrated injury data from new source")
    except Exception as e:
        print(f"⚠ Warning: Could not load injury data using helper: {e}")
        print("Continuing without injury data...")
        nfl_betting_df["injury_impact_score"] = 0
        nfl_betting_df["noteworthy_injuries_count"] = 0

    # Save the trimmed-down data to a CSV
    nfl_betting_df.to_csv("csvs/nflreadpy_betting_subset.csv", index=False)
    print("Successfully created a subset of the data for the betting dashboard.")
    return nfl_betting_df


def build_scores_from_games_df(games_df: pd.DataFrame):
    """
    Turn a games-level DataFrame (one row per game, with home/away teams and scores)
    into a team-level DataFrame with one row per team per game.
    """
    home_team_cols = ["home_team", "home", "home_team_abbr", "home_team_full"]
    away_team_cols = ["away_team", "away", "away_team_abbr", "away_team_full"]
    home_score_cols = ["home_score", "home_points", "home_team_score"]
    away_score_cols = ["away_score", "away_points", "away_team_score"]

    hc = next((c for c in home_team_cols if c in games_df.columns), None)
    ac = next((c for c in away_team_cols if c in games_df.columns), None)
    hs = next((c for c in home_score_cols if c in games_df.columns), None)
    ats = next((c for c in away_score_cols if c in games_df.columns), None)

    if not (
        hc
        and ac
        and hs
        and ats
        and "season" in games_df.columns
        and "week" in games_df.columns
    ):
        raise ValueError(
            "games_df does not contain the expected columns for teams and scores."
        )

    home_df = games_df.rename(
        columns={
            hc: "team",
            hs: "team_score",
            ac: "opponent_team",
            ats: "opponent_score",
        }
    )
    away_df = games_df.rename(
        columns={
            ac: "team",
            ats: "team_score",
            hc: "opponent_team",
            hs: "opponent_score",
        }
    )

    team_level = pd.concat([home_df, away_df], ignore_index=True, sort=False)
    team_level = team_level[["season", "week", "team", "team_score", "opponent_score"]]

    team_level["team_score"] = pd.to_numeric(team_level["team_score"], errors="coerce")
    team_level["opponent_score"] = pd.to_numeric(
        team_level["opponent_score"], errors="coerce"
    )
    return team_level


# --- REWRITTEN: This function now works with the new data structure ---
def calculate_injury_impact(team_injuries_df, season):
    """
    Calculates an injury impact score based on a list of noteworthy injured
    positions for each team for a given season.

    Parameters:
    -----------
    team_injuries_df : DataFrame
        DataFrame from get_nfl_teams_with_injuries_df() with columns:
        'abbreviation', 'noteworthy_injuries'
    season : int
        The season year to assign to the output rows.

    Returns:
    --------
    DataFrame with columns: season, team, injury_impact_score, noteworthy_injuries_count
    """
    # Position importance weights (higher = more impact when injured)
    position_weights = {
        "QB": 10.0,
        "LT": 4.5,
        "DE": 4.0,
        "CB": 4.0,
        "RB": 4.0,
        "WR": 3.5,
        "RT": 3.5,
        "C": 3.5,
        "DT": 3.5,
        "LB": 3.5,
        "S": 3.5,
        "TE": 3.0,
        "LG": 3.0,
        "RG": 3.0,
        "K": 2.0,
        "P": 1.0,
    }

    injury_impacts = []

    # Iterate through each team in the injury DataFrame
    for _, row in team_injuries_df.iterrows():
        team_abbr = row["abbreviation"]
        injured_positions = row.get("noteworthy_injuries", [])

        # Ensure we have a list to work with
        if not isinstance(injured_positions, list):
            injured_positions = []

        # Count the number of positions with noteworthy injuries
        noteworthy_count = len(injured_positions)

        # Calculate score by summing weights of injured positions
        impact_score = sum(position_weights.get(pos, 2.0) for pos in injured_positions)

        injury_impacts.append(
            {
                "season": season,
                "team": team_abbr,
                "injury_impact_score": impact_score,
                "noteworthy_injuries_count": noteworthy_count,
            }
        )

    return pd.DataFrame(injury_impacts)
