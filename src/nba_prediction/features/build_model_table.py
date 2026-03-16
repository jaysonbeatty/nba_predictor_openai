from __future__ import annotations

import pandas as pd


HOME_FEATURE_COLUMNS = {
    "team": "home_team",
    "rest_days": "home_rest_days",
    "b2b": "home_b2b",
    "off_rating_roll10": "home_off_rating_roll10",
    "def_rating_roll10": "home_def_rating_roll10",
    "pace_roll10": "home_pace_roll10",
}

AWAY_FEATURE_COLUMNS = {
    "team": "away_team",
    "rest_days": "away_rest_days",
    "b2b": "away_b2b",
    "off_rating_roll10": "away_off_rating_roll10",
    "def_rating_roll10": "away_def_rating_roll10",
    "pace_roll10": "away_pace_roll10",
}


def build_model_table(games_df: pd.DataFrame, enriched_team_games_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "game_id",
        "team",
        "rest_days",
        "b2b",
        "off_rating_roll10",
        "def_rating_roll10",
        "pace_roll10",
    ]
    team_features = enriched_team_games_df[feature_columns].copy()

    home_features = team_features.rename(columns=HOME_FEATURE_COLUMNS)
    away_features = team_features.rename(columns=AWAY_FEATURE_COLUMNS)

    model_df = games_df.merge(home_features, on=["game_id", "home_team"], how="left")
    model_df = model_df.merge(away_features, on=["game_id", "away_team"], how="left")

    ordered_columns = [
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "home_win",
        "margin",
        "total",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        "home_off_rating_roll10",
        "away_off_rating_roll10",
        "home_def_rating_roll10",
        "away_def_rating_roll10",
        "home_pace_roll10",
        "away_pace_roll10",
    ]
    return model_df[ordered_columns].sort_values(["game_date", "game_id"]).reset_index(drop=True)
