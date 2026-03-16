from __future__ import annotations

import pandas as pd


def add_rest_features(team_games_df: pd.DataFrame) -> pd.DataFrame:
    df = team_games_df.sort_values(["team", "game_date", "game_id"]).copy()
    df["prev_game_date"] = df.groupby("team")["game_date"].shift(1)
    gap_days = (df["game_date"] - df["prev_game_date"]).dt.days
    df["rest_days"] = gap_days - 1
    df["b2b"] = df["rest_days"].eq(0).astype("Int64")
    return df
