from __future__ import annotations

import pandas as pd


def _metric_or_default(df: pd.DataFrame, column: str, default_column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.to_numeric(df[default_column], errors="coerce")


def build_team_game_table(games_df: pd.DataFrame) -> pd.DataFrame:
    if "home_pace" in games_df.columns:
        home_pace = pd.to_numeric(games_df["home_pace"], errors="coerce")
    else:
        home_pace = pd.Series(pd.NA, index=games_df.index, dtype="Float64")

    if "away_pace" in games_df.columns:
        away_pace = pd.to_numeric(games_df["away_pace"], errors="coerce")
    else:
        away_pace = pd.Series(pd.NA, index=games_df.index, dtype="Float64")

    home_df = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "game_date": games_df["game_date"],
            "season": games_df["season"],
            "team": games_df["home_team"],
            "opponent": games_df["away_team"],
            "is_home": 1,
            "points": games_df["home_points"],
            "points_allowed": games_df["away_points"],
            "off_rating": _metric_or_default(games_df, "home_off_rating", "home_points"),
            "def_rating": _metric_or_default(games_df, "home_def_rating", "away_points"),
            "pace": home_pace,
        }
    )

    away_df = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "game_date": games_df["game_date"],
            "season": games_df["season"],
            "team": games_df["away_team"],
            "opponent": games_df["home_team"],
            "is_home": 0,
            "points": games_df["away_points"],
            "points_allowed": games_df["home_points"],
            "off_rating": _metric_or_default(games_df, "away_off_rating", "away_points"),
            "def_rating": _metric_or_default(games_df, "away_def_rating", "home_points"),
            "pace": away_pace,
        }
    )

    team_games = pd.concat([home_df, away_df], ignore_index=True)
    return team_games.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
