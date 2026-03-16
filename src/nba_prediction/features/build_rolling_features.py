from __future__ import annotations

import pandas as pd


def add_rolling_features(
    team_games_df: pd.DataFrame,
    window: int = 10,
    min_periods: int = 3,
) -> pd.DataFrame:
    df = team_games_df.sort_values(["team", "game_date", "game_id"]).copy()

    for metric in ["off_rating", "def_rating", "pace"]:
        rolling_column = f"{metric}_roll{window}"
        shifted = df.groupby("team")[metric].shift(1)
        df[rolling_column] = (
            shifted.groupby(df["team"])
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df
