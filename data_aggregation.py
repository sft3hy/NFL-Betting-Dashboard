import nflreadpy as nfl
import pandas as pd


def get_data():
    # Load team-level weekly stats
    team_stats = nfl.load_team_stats(summary_level="week")
    nfl_df = team_stats.to_pandas()

    # Print columns to help debug (optional)

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

    # Create your focused DataFrame (if any betting column missing this will raise KeyError)
    nfl_betting_df = nfl_df.copy()
    missing = [c for c in betting_columns if c not in nfl_betting_df.columns]
    if missing:
        print(f"⚠️ Warning: Some betting columns are missing in team stats: {missing}")
    # attempt to subset safely (keep whatever exists)
    nfl_betting_df = nfl_betting_df[
        [c for c in betting_columns if c in nfl_betting_df.columns]
    ]

    # --- NEW: attach actual game scores (team_score and opponent_score) ---
    scores_df = extract_scores_from_team_stats(nfl_df)
    if scores_df is None:
        # fallback: try loading game-level data and derive team rows
        try:
            games = nfl.load_schedules()
            games_df = games.to_pandas()
            scores_df = build_scores_from_games_df(games_df)
            print("✓ Loaded scores from nfl.load_games()")
        except Exception as e:
            print(f"⚠️ Could not load game-level data (nfl.load_games): {e}")
            scores_df = None

    if scores_df is not None:
        # merge scores into betting df
        # ensure types match
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
        # add empty score columns so downstream code won't break
        nfl_betting_df["team_score"] = pd.NA
        nfl_betting_df["opponent_score"] = pd.NA
        print("⚠️ Scores were not available; added empty score columns")

    # Load injury data (unchanged)
    try:
        injuries = nfl.load_injuries()
        injuries_df = injuries.to_pandas()

        # Process injury data to get impact scores by team and week
        injury_impact = calculate_injury_impact(injuries_df)

        # Merge injury data with team stats
        nfl_betting_df = nfl_betting_df.merge(
            injury_impact, on=["season", "week", "team"], how="left"
        )

        # Fill missing injury impact scores with 0 (no impact)
        nfl_betting_df["injury_impact_score"] = nfl_betting_df[
            "injury_impact_score"
        ].fillna(0)
        nfl_betting_df["key_injuries_count"] = nfl_betting_df[
            "key_injuries_count"
        ].fillna(0)

        print("✓ Successfully integrated injury data")
    except Exception as e:
        print(f"⚠ Warning: Could not load injury data: {e}")
        print("Continuing without injury data...")
        nfl_betting_df["injury_impact_score"] = 0
        nfl_betting_df["key_injuries_count"] = 0

    # Save the trimmed-down data to a CSV
    nfl_betting_df.to_csv("csvs/nflreadpy_betting_subset.csv", index=False)

    print("Successfully created a subset of the data for the betting dashboard.")

    return nfl_betting_df


def extract_scores_from_team_stats(team_stats_df: pd.DataFrame):
    """
    Look for score-like columns already present in team-level stats DataFrame.
    If found, return a DataFrame with columns [season, week, team, team_score, opponent_score].
    If no suitable columns found, return None.
    """
    # candidate names for team score and opponent score commonly used across packages/dumps
    team_score_candidates = [
        "team_score",
        "points",
        "points_for",
        "team_points",
        "score",
    ]
    opp_score_candidates = [
        "opponent_score",
        "points_against",
        "points_allowed",
        "opp_points",
        "opp_score",
        "against",
    ]

    found_team = None
    found_opp = None

    for c in team_score_candidates:
        if c in team_stats_df.columns:
            found_team = c
            break

    for c in opp_score_candidates:
        if c in team_stats_df.columns:
            found_opp = c
            break

    if found_team and found_opp:
        print(f"✓ Found score columns in team stats: {found_team}, {found_opp}")
        out = team_stats_df[["season", "week", "team", found_team, found_opp]].copy()
        out = out.rename(
            columns={found_team: "team_score", found_opp: "opponent_score"}
        )
        return out
    else:
        print("ℹ No direct score columns found in team stats.")
        return None


def build_scores_from_games_df(games_df: pd.DataFrame):
    """
    Turn a games-level DataFrame (one row per game, with home/away teams and scores)
    into a team-level DataFrame with one row per team per game,
    containing season, week, team, team_score, opponent_score.
    Expected games_df columns (some combination of):
       - season, week
       - home_team, away_team
       - home_score, away_score
    The function attempts to detect common names; if not found it raises ValueError.
    """
    # possible names
    home_team_cols = ["home_team", "home", "home_team_abbr", "home_team_full"]
    away_team_cols = ["away_team", "away", "away_team_abbr", "away_team_full"]
    home_score_cols = ["home_score", "home_points", "home_team_score"]
    away_score_cols = ["away_score", "away_points", "away_team_score"]

    # detect columns
    hc = next((c for c in home_team_cols if c in games_df.columns), None)
    ac = next((c for c in away_team_cols if c in games_df.columns), None)
    hs = next((c for c in home_score_cols if c in games_df.columns), None)
    ats = next((c for c in away_score_cols if c in games_df.columns), None)

    if not (
        hc
        and ac
        and hs
        and ats
        and ("season" in games_df.columns)
        and ("week" in games_df.columns)
    ):
        raise ValueError(
            "games_df does not contain the expected columns for home/away teams and scores "
            f"(found hc={hc}, ac={ac}, hs={hs}, ats={ats})."
        )

    # build two rows per game: one for home, one for away
    home_df = games_df[["season", "week", hc, hs, ac, ats]].copy()
    home_df = home_df.rename(
        columns={
            hc: "team",
            hs: "team_score",
            ac: "opponent_team",
            ats: "opponent_score",
        }
    )
    away_df = games_df[["season", "week", ac, ats, hc, hs]].copy()
    away_df = away_df.rename(
        columns={
            ac: "team",
            ats: "team_score",
            hc: "opponent_team",
            hs: "opponent_score",
        }
    )

    team_level = pd.concat([home_df, away_df], ignore_index=True, sort=False)
    # Keep only the columns we care about
    team_level = team_level[["season", "week", "team", "team_score", "opponent_score"]]

    # If score columns are strings, try to coerce to numeric
    team_level["team_score"] = pd.to_numeric(team_level["team_score"], errors="coerce")
    team_level["opponent_score"] = pd.to_numeric(
        team_level["opponent_score"], errors="coerce"
    )

    return team_level


def calculate_injury_impact(injuries_df):
    """
    Calculate the injury impact score for each team by week.

    Parameters:
    -----------
    injuries_df : DataFrame
        Injury data from nflreadpy

    Returns:
    --------
    DataFrame with columns: season, week, team, injury_impact_score, key_injuries_count
    """
    # Position importance weights (higher = more impact when injured)
    position_weights = {
        "QB": 10.0,  # Quarterback - highest impact
        "RB": 4.0,  # Running Back
        "WR": 3.5,  # Wide Receiver
        "TE": 3.0,  # Tight End
        "LT": 4.5,  # Left Tackle - protects QB
        "RT": 3.5,  # Right Tackle
        "LG": 3.0,  # Left Guard
        "RG": 3.0,  # Right Guard
        "C": 3.5,  # Center
        "DE": 4.0,  # Defensive End
        "DT": 3.5,  # Defensive Tackle
        "LB": 3.5,  # Linebacker
        "CB": 4.0,  # Cornerback
        "S": 3.5,  # Safety
        "K": 2.0,  # Kicker
        "P": 1.0,  # Punter
    }

    # Report status weights (higher = more severe)
    status_weights = {
        "Out": 1.0,
        "Doubtful": 0.8,
        "Questionable": 0.4,
        "Probable": 0.1,
        "IR": 1.0,  # Injured Reserve
        "PUP": 1.0,  # Physically Unable to Perform
    }

    injury_impacts = []

    # Group by season, week, and team
    for (season, week, team), group in injuries_df.groupby(["season", "week", "team"]):
        impact_score = 0
        key_injuries = 0

        for _, injury in group.iterrows():
            # Get position weight
            position = injury.get("position", "UNKNOWN")
            pos_weight = position_weights.get(position, 2.0)  # default weight

            # Get status weight
            status = injury.get("report_status", "Questionable")
            status_weight = status_weights.get(status, 0.5)

            # Calculate individual injury impact
            individual_impact = pos_weight * status_weight
            impact_score += individual_impact

            # Count key injuries (QB, high-impact positions that are Out/Doubtful)
            if position in ["QB", "LT", "DE", "CB"] and status in [
                "Out",
                "Doubtful",
                "IR",
            ]:
                key_injuries += 1

        injury_impacts.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "injury_impact_score": impact_score,
                "key_injuries_count": key_injuries,
            }
        )

    return pd.DataFrame(injury_impacts)
