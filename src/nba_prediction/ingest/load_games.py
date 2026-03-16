from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
]


def load_games_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"games_raw.csv is missing required columns: {missing_str}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False).dt.normalize()
    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()
    df["home_points"] = pd.to_numeric(df["home_points"], errors="raise")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="raise")

    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)
    df["margin"] = df["home_points"] - df["away_points"]
    df["total"] = df["home_points"] + df["away_points"]

    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    if df["game_id"].duplicated().any():
        duplicates = df.loc[df["game_id"].duplicated(), "game_id"].astype(str).tolist()
        sample = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate game_id values found: {sample}")

    return df
