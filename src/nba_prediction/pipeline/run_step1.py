from __future__ import annotations

import argparse
from pathlib import Path

from nba_prediction.features.build_model_table import build_model_table
from nba_prediction.features.build_rest_features import add_rest_features
from nba_prediction.features.build_rolling_features import add_rolling_features
from nba_prediction.features.build_team_game_table import build_team_game_table
from nba_prediction.ingest.load_games import load_games_csv


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_input = project_root / "data" / "raw" / "games_raw.csv"
    default_team_output = project_root / "data" / "processed" / "team_games_enriched.csv"
    default_model_output = project_root / "data" / "processed" / "games_model_base.csv"

    parser = argparse.ArgumentParser(description="Build the Step 1 NBA base dataset.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to raw games CSV.")
    parser.add_argument("--team-output", type=Path, default=default_team_output, help="Path to enriched team-game CSV.")
    parser.add_argument("--model-output", type=Path, default=default_model_output, help="Path to final game-level model CSV.")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size.")
    parser.add_argument("--min-periods", type=int, default=3, help="Minimum prior games needed for rolling features.")
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Drop rows with missing rolling features from the final table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    games_df = load_games_csv(args.input)
    team_games_df = build_team_game_table(games_df)
    team_games_df = add_rest_features(team_games_df)
    team_games_df = add_rolling_features(team_games_df, window=args.window, min_periods=args.min_periods)

    model_df = build_model_table(games_df, team_games_df)
    if args.drop_incomplete:
        candidate_columns = [
            "home_off_rating_roll10",
            "away_off_rating_roll10",
            "home_def_rating_roll10",
            "away_def_rating_roll10",
            "home_pace_roll10",
            "away_pace_roll10",
        ]
        active_columns = [
            column for column in candidate_columns if model_df[column].notna().any()
        ]
        if active_columns:
            model_df = model_df.dropna(subset=active_columns).reset_index(drop=True)

    args.team_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    team_games_df.to_csv(args.team_output, index=False)
    model_df.to_csv(args.model_output, index=False)

    print(f"Wrote {len(team_games_df)} rows to {args.team_output}")
    print(f"Wrote {len(model_df)} rows to {args.model_output}")


if __name__ == "__main__":
    main()
