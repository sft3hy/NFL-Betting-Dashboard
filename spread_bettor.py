# spread_bettor.py

import pandas as pd
import numpy as np
from datetime import datetime
import nflreadpy as nfl
from io import StringIO
import requests
from data_aggregation import get_data
from comprehensive_score import calculate_team_scores, predict_spread
from config import CURRENT_SEASON
import traceback


def get_current_week(season=CURRENT_SEASON):
    """
    Determine the current NFL week based on the schedule and current date.
    """
    try:
        schedule = nfl.load_schedules([season]).to_pandas()
        schedule_df = schedule[schedule["game_type"] == "REG"].copy()
        schedule_df["game_date"] = pd.to_datetime(schedule_df["gameday"])
        current_date = pd.Timestamp.now()
        for week in sorted(schedule_df["week"].unique()):
            latest_game = schedule_df[schedule_df["week"] == week]["game_date"].max()
            if current_date < latest_game + pd.Timedelta(days=1):  # Add a day buffer
                return int(week)
        return int(schedule_df["week"].max())
    except Exception:
        return 1


def get_upcoming_games(week, season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Download the canonical games CSV from the nflverse GitHub repo.
    The repo contains a games CSV with spread_line and other fields.
    """
    # Common raw URL for nfldata/games.csv (nflverse hosts these CSVs)
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        txt = r.text
        df = pd.read_csv(StringIO(txt), low_memory=False)
        # Filter to requested seasons
        # FIX: Changed .equals() to == for proper boolean masking
        df = df[df["season"] == season]
        schedule_df = df[df["game_type"] == "REG"].copy()
        if week is None:
            week = get_current_week(season)
        week_games = schedule_df[schedule_df["week"] == week].copy()

        game_columns = [
            "game_id",
            "season",
            "week",
            "gameday",
            "weekday",
            "gametime",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ]

        # Ensure columns exist, fill with None if not
        for col in game_columns:
            if col not in week_games.columns:
                week_games[col] = None

        to_return = week_games[game_columns].sort_values(["gameday", "gametime"])
        to_return["spread_line"] = to_return["spread_line"] * -1
        return to_return
    except Exception as e:
        print(f"Error loading schedule: {e}")
        traceback.print_exc()
        # FIX: Return an empty DataFrame on error to prevent downstream failures
        return pd.DataFrame()


def analyze_betting_opportunities(team_stats_df, season=CURRENT_SEASON, week=None):
    """
    Analyze betting opportunities by comparing model predictions to actual spreads.
    This version correctly includes the actual final scores for past games.
    """
    if week is None:
        week = get_current_week(season)

    print(f"\nAnalyzing Week {week} of {season} season...")
    games_df = get_upcoming_games(season=season, week=week)

    if games_df.empty:
        print("No games found for this week")
        return pd.DataFrame()

    scores_df = calculate_team_scores(team_stats_df, current_week=week - 1)
    latest_scores = scores_df.loc[scores_df.groupby("team")["week"].idxmax()]
    score_dict = dict(zip(latest_scores["team"], latest_scores["comprehensive_score"]))

    recommendations = []

    for idx, game in games_df.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]

        model_home_score = score_dict.get(home_team, 50)
        model_away_score = score_dict.get(away_team, 50)

        prediction = predict_spread(team_stats_df, home_team, away_team, week)
        model_spread = prediction["predicted_spread"]

        actual_spread = game.get("spread_line")
        # Get actual final scores, which will be NaN for future games
        actual_home_score = game.get("home_score")
        actual_away_score = game.get("away_score")

        edge = None
        recommendation = "Spread not available"
        bet_quality = "PASS"

        if pd.notna(actual_spread) and pd.notna(model_spread):
            # Edge is the difference between the actual spread and model's prediction.
            # A positive edge means the actual spread is higher (less negative) than the model's,
            # indicating value on the home team.
            # e.g., actual_spread = -3, model_spread = -6. Edge = -3 - (-6) = +3. Bet home team.
            edge = actual_spread - model_spread
            if abs(edge) >= 3:
                if edge > 0:
                    # Value is on the home team
                    recommendation = f"Bet on {home_team} {actual_spread:+.1f}"
                else:
                    # Value is on the away team
                    recommendation = f"Bet on {away_team} {-actual_spread:+.1f}"
            else:
                recommendation = "No strong edge"

            if abs(edge) >= 7:
                bet_quality = "STRONG BET"
            elif abs(edge) >= 5:
                bet_quality = "GOOD BET"
            elif abs(edge) >= 3:
                bet_quality = "LEAN"

        recommendations.append(
            {
                "game_id": game["game_id"],
                "week": week,
                "date": game["gameday"],
                "time": game.get("gametime", "TBD"),
                "away_team": away_team,
                "home_team": home_team,
                "model_away_score": round(model_away_score, 1),
                "model_home_score": round(model_home_score, 1),
                "actual_away_score": actual_away_score,
                "actual_home_score": actual_home_score,
                "model_spread": (
                    round(model_spread, 1) if pd.notna(model_spread) else None
                ),
                "actual_spread": (
                    round(actual_spread, 1) if pd.notna(actual_spread) else None
                ),
                "edge": round(edge, 1) if pd.notna(edge) else None,
                "bet_quality": bet_quality,
                "recommendation": recommendation,
                "confidence": prediction["confidence"],
            }
        )

    recommendations_df = pd.DataFrame(recommendations)
    if (
        "edge" in recommendations_df.columns
        and not recommendations_df["edge"].isnull().all()
    ):
        recommendations_df = recommendations_df.reindex(
            recommendations_df["edge"].abs().sort_values(ascending=False).index
        )

    return recommendations_df
