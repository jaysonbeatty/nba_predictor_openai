from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams


TEAM_ID_TO_ABBREV = {
    team["id"]: team["abbreviation"]
    for team in teams.get_teams()
}


def fetch_schedule_for_date(date: str) -> pd.DataFrame:
    endpoint = scoreboardv2.ScoreboardV2(game_date=date)
    headers = endpoint.game_header.get_data_frame()
    if headers.empty:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    games = pd.DataFrame(
        {
            "game_id": headers["GAME_ID"].astype(str),
            "game_date": pd.to_datetime(date),
            "home_team": headers["HOME_TEAM_ID"].map(TEAM_ID_TO_ABBREV),
            "away_team": headers["VISITOR_TEAM_ID"].map(TEAM_ID_TO_ABBREV),
            "game_status_text": headers["GAME_STATUS_TEXT"],
        }
    )
    return (
        games.dropna(subset=["home_team", "away_team"])
        .drop_duplicates(subset=["game_id"])
        .drop_duplicates(subset=["game_date", "home_team", "away_team"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )


def load_team_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["game_id"] = df["game_id"].astype(str)
    return df.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


def build_upcoming_feature_row(
    team_history_df: pd.DataFrame,
    date: str,
    game_id: str,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    target_date = pd.Timestamp(date)

    def latest_snapshot(team: str) -> pd.Series:
        prior = team_history_df[(team_history_df["team"] == team) & (team_history_df["game_date"] < target_date)]
        if prior.empty:
            raise ValueError(f"No team history available before {date} for {team}.")
        return prior.iloc[-1]

    home = latest_snapshot(home_team)
    away = latest_snapshot(away_team)

    home_rest_days = (target_date - pd.Timestamp(home["game_date"])).days - 1
    away_rest_days = (target_date - pd.Timestamp(away["game_date"])).days - 1

    row = pd.DataFrame(
        [
            {
                "game_id": str(game_id),
                "game_date": target_date,
                "season": str(target_date.year),
                "home_team": home_team,
                "away_team": away_team,
                "home_points": pd.NA,
                "away_points": pd.NA,
                "home_win": pd.NA,
                "margin": pd.NA,
                "total": pd.NA,
                "home_rest_days": float(home_rest_days),
                "away_rest_days": float(away_rest_days),
                "home_b2b": int(home_rest_days == 0),
                "away_b2b": int(away_rest_days == 0),
                "home_off_rating_roll10": float(home["off_rating_roll10"]),
                "away_off_rating_roll10": float(away["off_rating_roll10"]),
                "home_def_rating_roll10": float(home["def_rating_roll10"]),
                "away_def_rating_roll10": float(away["def_rating_roll10"]),
                "home_pace_roll10": pd.to_numeric(home["pace_roll10"], errors="coerce"),
                "away_pace_roll10": pd.to_numeric(away["pace_roll10"], errors="coerce"),
            }
        ]
    )
    return row


def get_prediction_games_for_date(
    date: str,
    games_model_df: pd.DataFrame,
    team_history_df: pd.DataFrame,
) -> pd.DataFrame:
    target_date = pd.Timestamp(date)
    historical_games = games_model_df[games_model_df["game_date"] == target_date].copy()
    if not historical_games.empty:
        return historical_games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    scheduled_games = fetch_schedule_for_date(date)
    if scheduled_games.empty:
        return pd.DataFrame(columns=games_model_df.columns)

    rows = []
    for game in scheduled_games.itertuples(index=False):
        try:
            rows.append(
                build_upcoming_feature_row(
                    team_history_df=team_history_df,
                    date=date,
                    game_id=str(game.game_id),
                    home_team=str(game.home_team),
                    away_team=str(game.away_team),
                )
            )
        except ValueError:
            continue

    if not rows:
        return pd.DataFrame(columns=games_model_df.columns)

    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(subset=["game_id"])
        .drop_duplicates(subset=["game_date", "home_team", "away_team"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
