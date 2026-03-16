from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_input = project_root / "data" / "raw" / "player_game_stats.csv"
    default_output = project_root / "data" / "processed" / "player_impact_features.csv"

    parser = argparse.ArgumentParser(description="Build rolling player impact features.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to player_game_stats.csv.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size.")
    parser.add_argument("--min-periods", type=int, default=3, help="Minimum prior games required.")
    return parser.parse_args()


def load_player_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    return df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


def build_player_impact_features(
    player_games_df: pd.DataFrame,
    window: int = 10,
    min_periods: int = 3,
) -> pd.DataFrame:
    df = player_games_df.copy()

    numeric_columns = [
        "minutes",
        "points",
        "assists",
        "rebounds",
        "steals",
        "blocks",
        "turnovers",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rate_minutes = df["minutes"].replace(0, float("nan"))
    df["offensive_box_score_rate"] = (
        df["points"] + 1.5 * df["assists"] + 0.7 * df["rebounds"] - 1.0 * df["turnovers"]
    ) / rate_minutes
    df["defensive_box_score_rate"] = (
        1.0 * df["rebounds"] + 1.5 * df["steals"] + 1.5 * df["blocks"]
    ) / rate_minutes

    for source_col, output_col in [
        ("minutes", "projected_minutes"),
        ("offensive_box_score_rate", "offensive_impact"),
        ("defensive_box_score_rate", "defensive_impact"),
    ]:
        df[source_col] = pd.to_numeric(df[source_col], errors="coerce")
        shifted = df.groupby("player_id")[source_col].shift(1)
        df[output_col] = (
            shifted.groupby(df["player_id"])
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )

    output_columns = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team",
        "projected_minutes",
        "offensive_impact",
        "defensive_impact",
    ]
    return df[output_columns].sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    player_games_df = load_player_games(args.input)
    features_df = build_player_impact_features(
        player_games_df,
        window=args.window,
        min_periods=args.min_periods,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"Wrote {len(features_df)} player impact rows to {args.output}")


if __name__ == "__main__":
    main()
