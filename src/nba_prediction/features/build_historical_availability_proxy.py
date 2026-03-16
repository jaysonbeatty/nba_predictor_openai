from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_games = project_root / "data" / "raw" / "games_raw.csv"
    default_player_features = project_root / "data" / "processed" / "player_impact_features.csv"
    default_output = project_root / "data" / "processed" / "historical_injury_status_proxy.csv"

    parser = argparse.ArgumentParser(description="Build historical availability proxy from missed games.")
    parser.add_argument("--games-input", type=Path, default=default_games, help="Path to games_raw.csv.")
    parser.add_argument(
        "--player-features-input",
        type=Path,
        default=default_player_features,
        help="Path to player_impact_features.csv.",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    parser.add_argument(
        "--min-projected-minutes",
        type=float,
        default=8.0,
        help="Minimum projected minutes required to mark a missed game as an absence.",
    )
    parser.add_argument(
        "--min-games-with-history",
        type=int,
        default=3,
        help="Minimum prior projected-game count before we consider a player expected to appear.",
    )
    return parser.parse_args()


def load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)

    home = df[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = df[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home, away], ignore_index=True)
    return team_games.sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def load_player_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    return df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


def build_historical_availability_proxy(
    team_games_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
    min_projected_minutes: float,
    min_games_with_history: int,
) -> pd.DataFrame:
    player_games = player_features_df.copy()
    player_games["game_id"] = player_games["game_id"].astype(str)
    player_games["projected_minutes"] = pd.to_numeric(player_games["projected_minutes"], errors="coerce")
    player_games["feature_history_count"] = player_games.groupby(["player_id", "team"]).cumcount() + 1

    team_games = team_games_df.copy()
    team_games["game_id"] = team_games["game_id"].astype(str)

    expected_rows: list[pd.DataFrame] = []
    for (player_id, team), player_history in player_games.groupby(["player_id", "team"], sort=False):
        team_schedule = team_games[team_games["team"] == team].copy()
        if team_schedule.empty:
            continue

        player_history = player_history.sort_values("game_date").copy()
        player_history["merge_date"] = player_history["game_date"]
        team_schedule = team_schedule.sort_values("game_date").copy()
        team_schedule["merge_date"] = team_schedule["game_date"]

        projected = pd.merge_asof(
            team_schedule,
            player_history[
                [
                    "merge_date",
                    "player_id",
                    "player_name",
                    "projected_minutes",
                    "offensive_impact",
                    "defensive_impact",
                    "feature_history_count",
                ]
            ].sort_values("merge_date"),
            on="merge_date",
            direction="backward",
            allow_exact_matches=False,
        )
        projected["team"] = team
        expected_rows.append(projected)

    if not expected_rows:
        return pd.DataFrame(
            columns=[
                "game_date",
                "game_id",
                "team",
                "player_id",
                "player_name",
                "status",
                "minutes_limit",
                "projected_minutes",
                "absence_reason",
            ]
        )

    expected = pd.concat(expected_rows, ignore_index=True)
    expected = expected.dropna(subset=["player_id", "projected_minutes"])
    expected = expected[
        (expected["projected_minutes"] >= min_projected_minutes)
        & (expected["feature_history_count"] >= min_games_with_history)
    ].copy()

    appeared = player_features_df[["game_id", "player_id"]].drop_duplicates().copy()
    appeared["game_id"] = appeared["game_id"].astype(str)
    appeared["appeared"] = 1

    expected["player_id"] = pd.to_numeric(expected["player_id"], errors="coerce").astype("Int64")
    expected = expected.merge(appeared, on=["game_id", "player_id"], how="left")
    expected["appeared"] = expected["appeared"].fillna(0).astype(int)

    proxy = expected[expected["appeared"] == 0].copy()
    proxy["status"] = "out"
    proxy["minutes_limit"] = pd.NA
    proxy["absence_reason"] = "historical_availability_proxy"

    output = proxy[
        [
            "game_date",
            "game_id",
            "team",
            "player_id",
            "player_name",
            "status",
            "minutes_limit",
            "projected_minutes",
            "offensive_impact",
            "defensive_impact",
            "absence_reason",
        ]
    ].sort_values(["game_date", "game_id", "team", "player_id"]).reset_index(drop=True)

    return output


def main() -> None:
    args = parse_args()
    team_games_df = load_games(args.games_input)
    player_features_df = load_player_features(args.player_features_input)
    proxy_df = build_historical_availability_proxy(
        team_games_df=team_games_df,
        player_features_df=player_features_df,
        min_projected_minutes=args.min_projected_minutes,
        min_games_with_history=args.min_games_with_history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proxy_df.to_csv(args.output, index=False)
    print(f"Wrote {len(proxy_df)} proxy injury rows to {args.output}")


if __name__ == "__main__":
    main()
