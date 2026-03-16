from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_games = project_root / "data" / "processed" / "games_model_base.csv"
    default_injuries = project_root / "data" / "processed" / "team_injury_features.csv"
    default_output = project_root / "data" / "processed" / "games_model_with_injuries.csv"

    parser = argparse.ArgumentParser(description="Merge team injury features into the game-level model table.")
    parser.add_argument("--games-input", type=Path, default=default_games, help="Path to games_model_base.csv.")
    parser.add_argument("--injuries-input", type=Path, default=default_injuries, help="Path to team_injury_features.csv.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    return parser.parse_args()


def merge_game_injury_features(games_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    home_injuries = injuries_df.rename(
        columns={
            "team": "home_team",
            "injury_off_impact": "home_injury_off_impact",
            "injury_def_impact": "home_injury_def_impact",
            "injury_total_impact": "home_injury_total_impact",
            "players_flagged": "home_players_flagged",
            "flagged_players": "home_flagged_players",
        }
    )
    away_injuries = injuries_df.rename(
        columns={
            "team": "away_team",
            "injury_off_impact": "away_injury_off_impact",
            "injury_def_impact": "away_injury_def_impact",
            "injury_total_impact": "away_injury_total_impact",
            "players_flagged": "away_players_flagged",
            "flagged_players": "away_flagged_players",
        }
    )

    merged = games_df.merge(
        home_injuries[
            [
                "game_id",
                "game_date",
                "home_team",
                "home_injury_off_impact",
                "home_injury_def_impact",
                "home_injury_total_impact",
                "home_players_flagged",
                "home_flagged_players",
            ]
        ],
        on=["game_id", "game_date", "home_team"],
        how="left",
    )
    merged = merged.merge(
        away_injuries[
            [
                "game_id",
                "game_date",
                "away_team",
                "away_injury_off_impact",
                "away_injury_def_impact",
                "away_injury_total_impact",
                "away_players_flagged",
                "away_flagged_players",
            ]
        ],
        on=["game_id", "game_date", "away_team"],
        how="left",
    )

    fill_values = {
        "home_injury_off_impact": 0.0,
        "home_injury_def_impact": 0.0,
        "home_injury_total_impact": 0.0,
        "away_injury_off_impact": 0.0,
        "away_injury_def_impact": 0.0,
        "away_injury_total_impact": 0.0,
        "home_players_flagged": 0,
        "away_players_flagged": 0,
        "home_flagged_players": "",
        "away_flagged_players": "",
    }
    return merged.fillna(fill_values)


def main() -> None:
    args = parse_args()
    games_df = pd.read_csv(args.games_input)
    injuries_df = pd.read_csv(args.injuries_input)
    merged = merge_game_injury_features(games_df, injuries_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
