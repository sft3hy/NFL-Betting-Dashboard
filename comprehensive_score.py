import pandas as pd
import numpy as np
from data_aggregation import get_data


def calculate_team_scores(df, current_week=None):
    """
    Calculate comprehensive team scores based on offensive, defensive, special teams metrics, and injuries.

    Parameters:
    -----------
    df : DataFrame
        The NFL betting dataframe with all team stats
    current_week : int or None
        If specified, calculate scores through this week. If None, use latest available.

    Returns:
    --------
    DataFrame with columns: team, week, comprehensive_score
    """

    if current_week is not None:
        df = df[df["week"] <= current_week].copy()

    # Sort by team and week
    df = df.sort_values(["team", "week"])

    # Calculate efficiency metrics
    df["completion_pct"] = df["completions"] / df["attempts"].replace(0, np.nan)
    df["yards_per_attempt"] = df["passing_yards"] / df["attempts"].replace(0, np.nan)
    df["yards_per_carry"] = df["rushing_yards"] / df["carries"].replace(0, np.nan)
    df["fg_pct"] = df["fg_made"] / df["fg_att"].replace(0, np.nan)
    df["pat_pct"] = df["pat_made"] / df["pat_att"].replace(0, np.nan)

    # Fill NaN values with reasonable defaults
    df["completion_pct"] = df["completion_pct"].fillna(0.6)
    df["yards_per_attempt"] = df["yards_per_attempt"].fillna(6.5)
    df["yards_per_carry"] = df["yards_per_carry"].fillna(4.0)
    df["fg_pct"] = df["fg_pct"].fillna(0.85)
    df["pat_pct"] = df["pat_pct"].fillna(0.95)

    # Define weights for different categories
    weights = {
        # Offensive metrics (35% of total) - HEAVILY REFINED TO FAVOR PASSING EFFICIENCY
        "passing_epa": 12.0,  # Increased: The single most predictive offensive stat.
        "rushing_epa": 4.0,  # Decreased: Rushing is important, but far less efficient than passing.
        "passing_yards": 0.01,  # Decreased: De-emphasizing raw volume stats.
        "rushing_yards": 0.01,  # Decreased: Less predictive than efficiency.
        "passing_tds": 4.5,  # Increased: Passing TDs are more valuable than rushing TDs.
        "rushing_tds": 3.5,  # Decreased: To create proper hierarchy with passing TDs.
        "passing_first_downs": 1.2,  # Increased: Reflects ability to sustain drives through the air.
        "rushing_first_downs": 0.8,  # Decreased: Less indicative of an explosive offense.
        "completion_pct": 40.0,  # Decreased slightly: Important, but its value is partially captured in EPA.
        "yards_per_attempt": 6.0,  # Increased: Key indicator of passing explosiveness and efficiency.
        "yards_per_carry": 5.0,  # No change: Remains a solid measure of rushing efficiency.
        # Defensive metrics (35% of total) - REFINED TO EMPHASIZE TURNOVERS & PRESSURE
        "def_sacks": 2.0,  # Decreased slightly: Sacks are good, but consistent pressure is better.
        "def_qb_hits": 1.2,  # Increased: Better indicator of consistent pass rush disruption.
        "def_interceptions": 5.0,  # Increased: Turnovers are paramount and game-changing.
        "def_pass_defended": 0.5,  # No change: Good play, but not a turnover.
        "def_fumbles_forced": 4.5,  # Increased: High-impact plays that directly lead to turnovers.
        "def_tds": 8.0,  # Increased: The single most impactful defensive play on the score.
        # Special teams (10% of total) - FOCUSED ON ACCURACY
        "fg_made": 1.0,  # Decreased: Volume stat, less important than accuracy.
        "fg_pct": 20.0,  # No change: A reliable kicker is a major asset.
        "pat_pct": 10.0,  # No change: Measures reliability on routine plays.
        # Negative factors (10% of total) - INCREASED PENALTIES FOR MISTAKES
        "passing_interceptions": -6.0,  # Increased: The most costly offensive mistake.
        "rushing_fumbles_lost": -6.0,  # Increased: Equally as costly as an interception.
        "sacks_suffered": -2.5,  # Increased: Sacks are drive-killers and have a large negative impact.
        "penalties": -1.0,  # Increased: Undisciplined teams consistently lose field position and momentum.
        "penalty_yards": -0.1,  # Increased: Higher penalty yardage is a strong negative indicator.
        # Injury impact (10% of total weight) - EMPHASIZING KEY PLAYER ABSENCE
        "injury_impact_score": -3.0,  # No change: Represents cumulative impact of multiple, less-critical injuries.
        "key_injuries_count": -8.0,  # Increased significantly: The loss of a QB or elite player has an outsized effect on the spread.
    }

    # Calculate rolling averages (last 4 games)
    rolling_window = 4

    # Group by team and calculate rolling metrics
    rolling_cols = list(weights.keys())

    for col in rolling_cols:
        if col in df.columns:
            df[f"{col}_rolling"] = df.groupby("team")[col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
            )

    # Calculate comprehensive score for each team-week
    score_components = []

    for team in df["team"].unique():
        team_df = df[df["team"] == team].copy()

        for idx, row in team_df.iterrows():
            score = 0

            # Add weighted contributions from each metric
            for metric, weight in weights.items():
                rolling_col = f"{metric}_rolling"
                if rolling_col in team_df.columns:
                    value = row[rolling_col]
                    if pd.notna(value):
                        score += value * weight

            score_components.append(
                {
                    "team": team,
                    "season": row["season"],
                    "week": row["week"],
                    "comprehensive_score": score,
                    "injury_impact": row.get("injury_impact_score", 0),
                    "key_injuries": row.get("key_injuries_count", 0),
                }
            )

    # Create results dataframe
    results_df = pd.DataFrame(score_components)

    # Normalize scores to 0-100 scale for each week
    for week in results_df["week"].unique():
        week_mask = results_df["week"] == week
        week_scores = results_df.loc[week_mask, "comprehensive_score"]

        # Min-max normalization to 0-100
        min_score = week_scores.min()
        max_score = week_scores.max()

        if max_score > min_score:
            results_df.loc[week_mask, "comprehensive_score"] = 50 + 50 * (
                week_scores - week_scores.mean()
            ) / (week_scores.std() + 1e-10)
        else:
            results_df.loc[week_mask, "comprehensive_score"] = 50

    # Clip to reasonable bounds
    results_df["comprehensive_score"] = results_df["comprehensive_score"].clip(0, 100)

    return results_df


def get_latest_scores(df, top_n=32):
    """
    Get the most recent comprehensive scores for all teams.

    Parameters:
    -----------
    df : DataFrame
        The NFL betting dataframe
    top_n : int
        Number of teams to return (default: 32 for all NFL teams)

    Returns:
    --------
    DataFrame with teams sorted by comprehensive_score (highest first)
    """
    scores_df = calculate_team_scores(df)

    # Get the latest week for each team
    latest_scores = scores_df.loc[scores_df.groupby("team")["week"].idxmax()]

    # Sort by score descending
    latest_scores = latest_scores.sort_values("comprehensive_score", ascending=False)

    return latest_scores[
        ["team", "comprehensive_score", "week", "injury_impact", "key_injuries"]
    ].head(top_n)


def predict_spread(df, team1, team2, current_week):
    """
    Predict the point spread between two teams for a given week.

    Parameters:
    -----------
    df : DataFrame
        The NFL betting dataframe
    team1 : str
        Home team abbreviation
    team2 : str
        Away team abbreviation
    current_week : int
        The week of the matchup

    Returns:
    --------
    dict with predicted_spread and confidence
    """
    scores_df = calculate_team_scores(df, current_week=current_week - 1)

    # Get scores for both teams from the previous week
    team1_scores = scores_df[(scores_df["team"] == team1)]
    team2_scores = scores_df[(scores_df["team"] == team2)]

    if team1_scores.empty or team2_scores.empty:
        return {
            "team1": team1,
            "team2": team2,
            "predicted_spread": None,
            "confidence": "low",
            "message": "Insufficient data for one or both teams",
        }

    # Get latest scores
    team1_data = team1_scores.iloc[-1]
    team2_data = team2_scores.iloc[-1]

    team1_score = team1_data["comprehensive_score"]
    team2_score = team2_data["comprehensive_score"]

    team1_injuries = team1_data["injury_impact"]
    team2_injuries = team2_data["injury_impact"]

    # Calculate predicted margin (positive means team1/home team is favored)
    score_diff = team1_score - team2_score
    predicted_margin = score_diff * 0.3

    # Add home field advantage (roughly 2.5 points)
    predicted_margin += 2.5

    # Determine confidence based on score differential and injury impact
    score_diff_abs = abs(score_diff)
    injury_uncertainty = abs(team1_injuries) + abs(team2_injuries)

    if score_diff_abs > 20 and injury_uncertainty < 5:
        confidence = "high"
    elif score_diff_abs > 10 and injury_uncertainty < 10:
        confidence = "medium"
    else:
        confidence = "low"

    # Add injury notes
    injury_notes = []
    if team1_injuries > 5:
        injury_notes.append(f"{team1} dealing with significant injuries")
    if team2_injuries > 5:
        injury_notes.append(f"{team2} dealing with significant injuries")

    return {
        "home_team": team1,
        "away_team": team2,
        "home_score": round(team1_score, 2),
        "away_score": round(team2_score, 2),
        "home_injury_impact": round(team1_injuries, 2),
        "away_injury_impact": round(team2_injuries, 2),
        "predicted_spread": -round(predicted_margin, 1),
        "confidence": confidence,
        "recommendation": (
            f"{team1} by {abs(predicted_margin):.1f}"
            if predicted_margin > 0
            else f"{team2} by {abs(predicted_margin):.1f}"
        ),
        "injury_notes": (
            injury_notes if injury_notes else ["No significant injury concerns"]
        ),
    }


if __name__ == "__main__":
    # Load the data
    print("Loading NFL data...")
    nfl_df = get_data()

    # Calculate comprehensive scores
    print("\nCalculating comprehensive scores (including injuries)...")
    scores_df = calculate_team_scores(nfl_df)

    # Get latest rankings
    print("\nLatest Team Rankings:")
    print("=" * 80)
    latest = get_latest_scores(nfl_df)
    print(latest.to_string(index=False))

    # Save full scores to CSV
    scores_df.to_csv("csvs/team_comprehensive_scores.csv", index=False)
    print("\n✓ Saved detailed scores to 'csvs/team_comprehensive_scores.csv'")

    # Save latest rankings to CSV
    latest.to_csv("csvs/latest_team_rankings.csv", index=False)
    print("✓ Saved latest rankings to 'csvs/latest_team_rankings.csv'")

    # Example spread prediction
    print("\n" + "=" * 80)
    print("Example Spread Prediction:")
    print("=" * 80)

    # Get current max week
    max_week = nfl_df["week"].max()

    if max_week >= 1:
        # Example matchup - adjust team abbreviations as needed
        example_teams = nfl_df["team"].unique()[:2]
        if len(example_teams) >= 2:
            prediction = predict_spread(
                nfl_df, example_teams[0], example_teams[1], max_week + 1
            )
            print(
                f"\nMatchup: {prediction['home_team']} (Home) vs {prediction['away_team']} (Away)"
            )
            print(
                f"Home Team Score: {prediction['home_score']} (Injury Impact: {prediction['home_injury_impact']})"
            )
            print(
                f"Away Team Score: {prediction['away_score']} (Injury Impact: {prediction['away_injury_impact']})"
            )
            print(f"Predicted Spread: {prediction['predicted_spread']}")
            print(f"Confidence: {prediction['confidence']}")
            print(f"Recommendation: {prediction['recommendation']}")
            print(f"Injury Notes: {', '.join(prediction['injury_notes'])}")
