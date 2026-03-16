from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


SEASON_TYPE_MAP = {
    "regular": "Regular Season",
    "playoffs": "Playoffs",
    "both": None,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_output = project_root / "data" / "raw" / "games_raw.csv"

    parser = argparse.ArgumentParser(description="Fetch NBA games from nba_api and build games_raw.csv.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="NBA season labels like 2023-24 2024-25 2025-26",
    )
    parser.add_argument(
        "--season-type",
        choices=sorted(SEASON_TYPE_MAP.keys()),
        default="regular",
        help="Which season type to fetch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.6,
        help="Pause between API calls to reduce rate-limit risk.",
    )
    return parser.parse_args()


def fetch_season_games(season: str, season_type: str | None) -> pd.DataFrame:
    kwargs = {"season_nullable": season}
    if season_type is not None:
        kwargs["season_type_nullable"] = season_type

    endpoint = leaguegamefinder.LeagueGameFinder(**kwargs)
    frames = endpoint.get_data_frames()
    if not frames:
        return pd.DataFrame()
    return frames[0].copy()


def _split_matchup(matchup: str) -> tuple[str, str, int]:
    matchup = str(matchup).strip()
    if " vs. " in matchup:
        team, opponent = matchup.split(" vs. ", 1)
        return team.strip(), opponent.strip(), 1
    if " @ " in matchup:
        team, opponent = matchup.split(" @ ", 1)
        return team.strip(), opponent.strip(), 0
    raise ValueError(f"Unrecognized matchup format: {matchup}")


def normalize_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "season",
                "home_team",
                "away_team",
                "home_points",
                "away_points",
            ]
        )

    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=False).dt.normalize()
    matchup_parts = df["MATCHUP"].apply(_split_matchup)
    df["team_abbr"] = matchup_parts.str[0]
    df["opponent_abbr"] = matchup_parts.str[1]
    df["is_home"] = matchup_parts.str[2].astype(int)

    df = df[df["WL"].notna()].copy()
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    df = df[df["PTS"].notna()].copy()

    home_df = (
        df[df["is_home"] == 1][["GAME_ID", "GAME_DATE", "SEASON_ID", "team_abbr", "opponent_abbr", "PTS"]]
        .rename(
            columns={
                "GAME_ID": "game_id",
                "GAME_DATE": "game_date",
                "SEASON_ID": "season_id",
                "team_abbr": "home_team",
                "opponent_abbr": "away_team",
                "PTS": "home_points",
            }
        )
        .copy()
    )

    away_df = (
        df[df["is_home"] == 0][["GAME_ID", "team_abbr", "PTS"]]
        .rename(
            columns={
                "GAME_ID": "game_id",
                "team_abbr": "away_team",
                "PTS": "away_points",
            }
        )
        .copy()
    )

    games_df = home_df.merge(away_df, on=["game_id", "away_team"], how="inner")
    games_df["season"] = games_df["game_date"].dt.year.astype(str)
    games_df["home_points"] = pd.to_numeric(games_df["home_points"], errors="raise")
    games_df["away_points"] = pd.to_numeric(games_df["away_points"], errors="raise")

    games_df = games_df[
        [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ]
    ].drop_duplicates(subset=["game_id"])

    return games_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    season_type_value = SEASON_TYPE_MAP[args.season_type]

    season_frames: list[pd.DataFrame] = []
    for index, season in enumerate(args.seasons):
        raw_df = fetch_season_games(season=season, season_type=season_type_value)
        if raw_df.empty:
            print(f"No games returned for season={season}")
        else:
            raw_df["requested_season"] = season
            season_frames.append(raw_df)
            print(f"Fetched {len(raw_df)} team-game rows for season={season}")

        if index < len(args.seasons) - 1:
            time.sleep(args.pause_seconds)

    if season_frames:
        combined_raw = pd.concat(season_frames, ignore_index=True)
    else:
        combined_raw = pd.DataFrame()

    games_df = normalize_games(combined_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(args.output, index=False)

    print(f"Wrote {len(games_df)} game rows to {args.output}")


if __name__ == "__main__":
    main()
