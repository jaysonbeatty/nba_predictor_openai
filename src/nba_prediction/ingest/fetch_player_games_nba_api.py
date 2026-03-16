from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs


SEASON_TYPE_MAP = {
    "regular": "Regular Season",
    "playoffs": "Playoffs",
    "both": None,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_output = project_root / "data" / "raw" / "player_game_stats.csv"

    parser = argparse.ArgumentParser(description="Fetch player game logs from nba_api.")
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
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.6,
        help="Pause between API calls to reduce rate-limit risk.",
    )
    return parser.parse_args()


def _minutes_to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None

    minutes_str, seconds_str = text.split(":", 1)
    try:
        minutes = int(minutes_str)
        seconds = int(seconds_str)
    except ValueError:
        return None
    return minutes + (seconds / 60.0)


def fetch_season_player_logs(season: str, season_type: str | None) -> pd.DataFrame:
    kwargs = {"season_nullable": season}
    if season_type is not None:
        kwargs["season_type_nullable"] = season_type

    endpoint = playergamelogs.PlayerGameLogs(**kwargs)
    frames = endpoint.get_data_frames()
    if not frames:
        return pd.DataFrame()
    return frames[0].copy()


def normalize_player_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "season",
                "player_id",
                "player_name",
                "team",
                "minutes",
                "points",
                "assists",
                "rebounds",
                "steals",
                "blocks",
                "turnovers",
            ]
        )

    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=False).dt.normalize()
    df["MIN"] = df["MIN"].apply(_minutes_to_float)

    normalized = pd.DataFrame(
        {
            "game_id": df["GAME_ID"].astype(str),
            "game_date": df["GAME_DATE"],
            "season": df["SEASON_YEAR"].astype(str),
            "player_id": pd.to_numeric(df["PLAYER_ID"], errors="raise"),
            "player_name": df["PLAYER_NAME"].astype(str).str.strip(),
            "team": df["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip(),
            "minutes": pd.to_numeric(df["MIN"], errors="coerce"),
            "points": pd.to_numeric(df["PTS"], errors="coerce"),
            "assists": pd.to_numeric(df["AST"], errors="coerce"),
            "rebounds": pd.to_numeric(df["REB"], errors="coerce"),
            "steals": pd.to_numeric(df["STL"], errors="coerce"),
            "blocks": pd.to_numeric(df["BLK"], errors="coerce"),
            "turnovers": pd.to_numeric(df["TOV"], errors="coerce"),
        }
    )

    normalized = normalized.dropna(subset=["game_id", "game_date", "player_id", "team"])
    normalized = normalized.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return normalized


def main() -> None:
    args = parse_args()
    season_type_value = SEASON_TYPE_MAP[args.season_type]

    frames: list[pd.DataFrame] = []
    for index, season in enumerate(args.seasons):
        raw_df = fetch_season_player_logs(season=season, season_type=season_type_value)
        if raw_df.empty:
            print(f"No player logs returned for season={season}")
        else:
            frames.append(raw_df)
            print(f"Fetched {len(raw_df)} player-game rows for season={season}")

        if index < len(args.seasons) - 1:
            time.sleep(args.pause_seconds)

    combined_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_player_logs(combined_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    print(f"Wrote {len(normalized)} player-game rows to {args.output}")


if __name__ == "__main__":
    main()
