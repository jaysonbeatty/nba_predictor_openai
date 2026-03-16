from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_STATUSES = {"available", "limited", "out"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_games = project_root / "data" / "raw" / "games_raw.csv"
    default_player_features = project_root / "data" / "processed" / "player_impact_features.csv"
    default_injuries = project_root / "data" / "raw" / "injury_status.csv"
    default_output = project_root / "data" / "processed" / "team_injury_features.csv"

    parser = argparse.ArgumentParser(description="Build team-level injury impact features.")
    parser.add_argument("--games-input", type=Path, default=default_games, help="Path to games_raw.csv.")
    parser.add_argument(
        "--player-features-input",
        type=Path,
        default=default_player_features,
        help="Path to player_impact_features.csv.",
    )
    parser.add_argument(
        "--injuries-input",
        type=Path,
        default=default_injuries,
        help="Path to injury_status.csv.",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    return parser.parse_args()


def load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["game_id"] = df["game_id"].astype(str)

    home = df[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = df[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True).sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def load_player_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["game_id"] = df["game_id"].astype(str)
    return df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def load_injuries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["game_date", "player_id", "status"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"injury_status.csv is missing required columns: {missing_str}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="raise")
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    bad_status = sorted(set(df["status"]) - VALID_STATUSES)
    if bad_status:
        bad = ", ".join(bad_status)
        raise ValueError(f"Invalid injury status values: {bad}")

    if "minutes_limit" not in df.columns:
        df["minutes_limit"] = pd.NA
    df["minutes_limit"] = pd.to_numeric(df["minutes_limit"], errors="coerce")
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def _lost_minutes(row: pd.Series) -> float:
    projected_minutes = float(row["projected_minutes"] or 0.0)
    status = row["status"]
    minutes_limit = row["minutes_limit"]

    if status == "out":
        return projected_minutes
    if status == "limited":
        if pd.isna(minutes_limit):
            return max(projected_minutes * 0.5, 0.0)
        return max(projected_minutes - float(minutes_limit), 0.0)
    return 0.0


def build_injury_features(
    games_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = player_features_df.merge(
        injuries_df,
        on=["game_date", "player_id"],
        how="left",
    )
    if "game_id_x" in merged.columns:
        merged["game_id"] = merged["game_id_x"]
    elif "game_id" not in merged.columns:
        raise ValueError("Merged injury frame is missing game_id from player features.")

    if "game_id_y" in merged.columns:
        same_game_mask = merged["game_id_y"].isna() | (merged["game_id_x"].astype(str) == merged["game_id_y"].astype(str))
        merged = merged[same_game_mask].copy()
    drop_cols = [column for column in ["game_id_x", "game_id_y"] if column in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    if "projected_minutes_x" in merged.columns:
        merged["projected_minutes"] = pd.to_numeric(merged["projected_minutes_x"], errors="coerce")
        drop_cols = [column for column in ["projected_minutes_x", "projected_minutes_y"] if column in merged.columns]
        merged = merged.drop(columns=drop_cols)
    elif "projected_minutes" not in merged.columns:
        merged["projected_minutes"] = 0.0

    if "player_name_x" in merged.columns:
        merged["player_name"] = merged["player_name_x"].fillna(merged.get("player_name_y"))
        drop_cols = [column for column in ["player_name_x", "player_name_y"] if column in merged.columns]
        merged = merged.drop(columns=drop_cols)
    elif "player_name" not in merged.columns:
        merged["player_name"] = "unknown"

    if "team_x" in merged.columns:
        merged["team"] = merged["team_x"].fillna(merged.get("team_y"))
        drop_cols = [column for column in ["team_x", "team_y"] if column in merged.columns]
        merged = merged.drop(columns=drop_cols)

    if "offensive_impact_x" in merged.columns:
        merged["offensive_impact"] = pd.to_numeric(merged["offensive_impact_x"], errors="coerce")
        if "offensive_impact_y" in merged.columns:
            merged["offensive_impact"] = merged["offensive_impact"].fillna(
                pd.to_numeric(merged["offensive_impact_y"], errors="coerce")
            )
        drop_cols = [column for column in ["offensive_impact_x", "offensive_impact_y"] if column in merged.columns]
        merged = merged.drop(columns=drop_cols)
    elif "offensive_impact" not in merged.columns:
        merged["offensive_impact"] = 0.0

    if "defensive_impact_x" in merged.columns:
        merged["defensive_impact"] = pd.to_numeric(merged["defensive_impact_x"], errors="coerce")
        if "defensive_impact_y" in merged.columns:
            merged["defensive_impact"] = merged["defensive_impact"].fillna(
                pd.to_numeric(merged["defensive_impact_y"], errors="coerce")
            )
        drop_cols = [column for column in ["defensive_impact_x", "defensive_impact_y"] if column in merged.columns]
        merged = merged.drop(columns=drop_cols)
    elif "defensive_impact" not in merged.columns:
        merged["defensive_impact"] = 0.0

    merged["status"] = merged["status"].fillna("available")
    merged["minutes_limit"] = pd.to_numeric(merged["minutes_limit"], errors="coerce")
    merged["projected_minutes"] = pd.to_numeric(merged["projected_minutes"], errors="coerce").fillna(0.0)
    merged["offensive_impact"] = pd.to_numeric(merged["offensive_impact"], errors="coerce").fillna(0.0)
    merged["defensive_impact"] = pd.to_numeric(merged["defensive_impact"], errors="coerce").fillna(0.0)

    merged["lost_minutes"] = merged.apply(_lost_minutes, axis=1)
    merged["off_injury_loss"] = merged["lost_minutes"] * merged["offensive_impact"]
    merged["def_injury_loss"] = merged["lost_minutes"] * merged["defensive_impact"]
    merged["player_status_label"] = merged["player_name"].fillna("unknown") + ":" + merged["status"]

    # Historical proxy rows can represent players who did not appear in the box
    # score for that game, so they never show up in player_features_df. Append
    # those rows explicitly so they contribute injury loss.
    injury_only = injuries_df.copy()
    if "game_id" in injury_only.columns:
        injury_only["game_id"] = injury_only["game_id"].astype(str)
    else:
        injury_only["game_id"] = pd.NA

    if "team" not in injury_only.columns:
        injury_only = injury_only.merge(games_df, on=["game_date", "game_id"], how="left")

    if "player_name" not in injury_only.columns:
        injury_only["player_name"] = "unknown"
    if "projected_minutes" not in injury_only.columns:
        injury_only["projected_minutes"] = 0.0
    if "offensive_impact" not in injury_only.columns:
        injury_only["offensive_impact"] = 0.0
    if "defensive_impact" not in injury_only.columns:
        injury_only["defensive_impact"] = 0.0
    if "minutes_limit" not in injury_only.columns:
        injury_only["minutes_limit"] = pd.NA

    injury_only = injury_only[
        injury_only["status"].isin(["out", "limited"])
    ].copy()

    existing_keys = set(
        zip(
            merged["game_id"].astype(str),
            pd.to_numeric(merged["player_id"], errors="coerce"),
        )
    )
    injury_only = injury_only[
        ~injury_only.apply(
            lambda row: (str(row["game_id"]), pd.to_numeric(row["player_id"], errors="coerce")) in existing_keys,
            axis=1,
        )
    ].copy()

    if not injury_only.empty:
        latest_player_features = (
            player_features_df.sort_values(["player_id", "game_date", "game_id"])
            .groupby("player_id", as_index=False)
            .tail(1)[["player_id", "player_name", "team", "projected_minutes", "offensive_impact", "defensive_impact"]]
            .rename(
                columns={
                    "player_name": "latest_player_name",
                    "team": "latest_team",
                    "projected_minutes": "latest_projected_minutes",
                    "offensive_impact": "latest_offensive_impact",
                    "defensive_impact": "latest_defensive_impact",
                }
            )
        )
        injury_only = injury_only.merge(latest_player_features, on="player_id", how="left")
        if "player_name" in injury_only.columns:
            injury_only["player_name"] = injury_only["player_name"].fillna(injury_only["latest_player_name"])
        else:
            injury_only["player_name"] = injury_only["latest_player_name"]
        if "team" in injury_only.columns:
            injury_only["team"] = injury_only["team"].fillna(injury_only["latest_team"])
        else:
            injury_only["team"] = injury_only["latest_team"]
        if "projected_minutes" in injury_only.columns:
            injury_only["projected_minutes"] = pd.to_numeric(injury_only["projected_minutes"], errors="coerce")
            injury_only["projected_minutes"] = injury_only["projected_minutes"].fillna(
                pd.to_numeric(injury_only["latest_projected_minutes"], errors="coerce")
            )
        else:
            injury_only["projected_minutes"] = pd.to_numeric(injury_only["latest_projected_minutes"], errors="coerce")
        if "offensive_impact" in injury_only.columns:
            injury_only["offensive_impact"] = pd.to_numeric(injury_only["offensive_impact"], errors="coerce")
            injury_only["offensive_impact"] = injury_only["offensive_impact"].fillna(
                pd.to_numeric(injury_only["latest_offensive_impact"], errors="coerce")
            )
        else:
            injury_only["offensive_impact"] = pd.to_numeric(injury_only["latest_offensive_impact"], errors="coerce")
        if "defensive_impact" in injury_only.columns:
            injury_only["defensive_impact"] = pd.to_numeric(injury_only["defensive_impact"], errors="coerce")
            injury_only["defensive_impact"] = injury_only["defensive_impact"].fillna(
                pd.to_numeric(injury_only["latest_defensive_impact"], errors="coerce")
            )
        else:
            injury_only["defensive_impact"] = pd.to_numeric(injury_only["latest_defensive_impact"], errors="coerce")

        injury_only["projected_minutes"] = pd.to_numeric(injury_only["projected_minutes"], errors="coerce").fillna(0.0)
        injury_only["offensive_impact"] = pd.to_numeric(injury_only["offensive_impact"], errors="coerce").fillna(0.0)
        injury_only["defensive_impact"] = pd.to_numeric(injury_only["defensive_impact"], errors="coerce").fillna(0.0)
        injury_only["minutes_limit"] = pd.to_numeric(injury_only["minutes_limit"], errors="coerce")
        injury_only["player_status_label"] = injury_only["player_name"].fillna("unknown") + ":" + injury_only["status"]
        injury_only["lost_minutes"] = injury_only.apply(_lost_minutes, axis=1)
        injury_only["off_injury_loss"] = injury_only["lost_minutes"] * injury_only["offensive_impact"]
        injury_only["def_injury_loss"] = injury_only["lost_minutes"] * injury_only["defensive_impact"]

        merged = pd.concat(
            [
                merged,
                injury_only[
                    [
                        "game_id",
                        "game_date",
                        "team",
                        "status",
                        "player_name",
                        "projected_minutes",
                        "offensive_impact",
                        "defensive_impact",
                        "minutes_limit",
                        "player_status_label",
                        "lost_minutes",
                        "off_injury_loss",
                        "def_injury_loss",
                    ]
                ],
            ],
            ignore_index=True,
            sort=False,
        )
        drop_cols = [
            column
            for column in [
                "latest_player_name",
                "latest_team",
                "latest_projected_minutes",
                "latest_offensive_impact",
                "latest_defensive_impact",
            ]
            if column in merged.columns
        ]
        if drop_cols:
            merged = merged.drop(columns=drop_cols, errors="ignore")

    team_features = (
        merged.groupby(["game_id", "game_date", "team"], as_index=False)
        .agg(
            injury_off_impact=("off_injury_loss", "sum"),
            injury_def_impact=("def_injury_loss", "sum"),
            players_flagged=("status", lambda s: int((s != "available").sum())),
            flagged_players=("player_status_label", lambda values: "|".join(v for v in values if not v.endswith(":available"))),
        )
        .sort_values(["game_date", "game_id", "team"])
        .reset_index(drop=True)
    )
    team_features["game_id"] = team_features["game_id"].astype(str)
    team_features["injury_total_impact"] = team_features["injury_off_impact"] + team_features["injury_def_impact"]

    return games_df.merge(team_features, on=["game_id", "game_date", "team"], how="left").fillna(
        {
            "injury_off_impact": 0.0,
            "injury_def_impact": 0.0,
            "injury_total_impact": 0.0,
            "players_flagged": 0,
            "flagged_players": "",
        }
    )


def main() -> None:
    args = parse_args()
    games_df = load_games(args.games_input)
    player_features_df = load_player_features(args.player_features_input)
    injuries_df = load_injuries(args.injuries_input)
    output_df = build_injury_features(games_df, player_features_df, injuries_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(output_df)} team injury rows to {args.output}")


if __name__ == "__main__":
    main()
