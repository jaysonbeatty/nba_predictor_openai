from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nba_prediction.features.build_injury_features import build_injury_features, load_injuries
from nba_prediction.schedule.upcoming_games import build_upcoming_feature_row, load_team_history

INJURY_RATING_SCALE = 0.15


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_games = project_root / "data" / "processed" / "games_model_base.csv"
    default_raw_games = project_root / "data" / "raw" / "games_raw.csv"
    default_player_features = project_root / "data" / "processed" / "player_impact_features.csv"
    default_win_model = project_root / "artifacts" / "models" / "baseline_with_historical_injuries_win_model.joblib"
    default_margin_model = project_root / "artifacts" / "models" / "baseline_with_historical_injuries_margin_model.joblib"
    default_total_model = project_root / "artifacts" / "models" / "baseline_with_historical_injuries_total_model.joblib"

    parser = argparse.ArgumentParser(description="Simulate a single game with optional injury overrides.")
    parser.add_argument("--date", required=True, help="Game date in YYYY-MM-DD.")
    parser.add_argument("--home-team", required=True, help="Home team abbreviation.")
    parser.add_argument("--away-team", required=True, help="Away team abbreviation.")
    parser.add_argument("--games-input", type=Path, default=default_games, help="Path to games_model_base.csv.")
    parser.add_argument("--raw-games-input", type=Path, default=default_raw_games, help="Path to games_raw.csv.")
    parser.add_argument(
        "--player-features-input",
        type=Path,
        default=default_player_features,
        help="Path to player_impact_features.csv.",
    )
    parser.add_argument(
        "--injuries-input",
        type=Path,
        help="Optional path to injury_status.csv for baseline injuries.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override in the form player_id:status[:minutes_limit], repeatable.",
    )
    parser.add_argument("--win-model", type=Path, default=default_win_model, help="Path to trained win model.")
    parser.add_argument("--margin-model", type=Path, default=default_margin_model, help="Path to trained margin model.")
    parser.add_argument("--total-model", type=Path, default=default_total_model, help="Path to trained total model.")
    return parser.parse_args()


def _load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def _load_player_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def _load_team_history(path: Path) -> pd.DataFrame:
    return load_team_history(path)


def _game_teams_from_raw(raw_games_df: pd.DataFrame) -> pd.DataFrame:
    home = raw_games_df[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = raw_games_df[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"], utc=False)
    team_games["game_id"] = team_games["game_id"].astype(str)
    return team_games.sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def _append_game_teams(
    team_games_df: pd.DataFrame,
    game_id: str,
    date: str,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    target_date = pd.Timestamp(date)
    existing = team_games_df[
        (team_games_df["game_id"] == str(game_id))
        & (team_games_df["game_date"] == target_date)
    ]
    if not existing.empty:
        return team_games_df

    extra = pd.DataFrame(
        [
            {"game_id": str(game_id), "game_date": target_date, "team": home_team},
            {"game_id": str(game_id), "game_date": target_date, "team": away_team},
        ]
    )
    return pd.concat([team_games_df, extra], ignore_index=True).sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def _parse_override(text: str, target_date: str) -> dict[str, object]:
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid override format: {text}")
    player_id = int(parts[0])
    status = parts[1].strip().lower()
    minutes_limit = float(parts[2]) if len(parts) == 3 and parts[2] != "" else None
    return {
        "game_date": target_date,
        "player_id": player_id,
        "status": status,
        "minutes_limit": minutes_limit,
    }


def _build_feature_row(game_row: pd.Series, home_injury: pd.Series | None, away_injury: pd.Series | None) -> pd.DataFrame:
    home_off_impact = float(home_injury["injury_off_impact"]) if home_injury is not None else 0.0
    home_def_impact = float(home_injury["injury_def_impact"]) if home_injury is not None else 0.0
    away_off_impact = float(away_injury["injury_off_impact"]) if away_injury is not None else 0.0
    away_def_impact = float(away_injury["injury_def_impact"]) if away_injury is not None else 0.0
    home_total_impact = float(home_injury["injury_total_impact"]) if home_injury is not None else 0.0
    away_total_impact = float(away_injury["injury_total_impact"]) if away_injury is not None else 0.0
    home_players_flagged = int(home_injury["players_flagged"]) if home_injury is not None else 0
    away_players_flagged = int(away_injury["players_flagged"]) if away_injury is not None else 0

    home_off_delta = home_off_impact * INJURY_RATING_SCALE
    home_def_delta = home_def_impact * INJURY_RATING_SCALE
    away_off_delta = away_off_impact * INJURY_RATING_SCALE
    away_def_delta = away_def_impact * INJURY_RATING_SCALE

    row = pd.DataFrame(
        [
            {
                "rest_advantage": float(game_row["home_rest_days"] if pd.notna(game_row["home_rest_days"]) else 0.0)
                - float(game_row["away_rest_days"] if pd.notna(game_row["away_rest_days"]) else 0.0),
                "home_off_rating_roll10": float(game_row["home_off_rating_roll10"]) - home_off_delta,
                "away_off_rating_roll10": float(game_row["away_off_rating_roll10"]) - away_off_delta,
                "off_rating_diff": (float(game_row["home_off_rating_roll10"]) - home_off_delta)
                - (float(game_row["away_off_rating_roll10"]) - away_off_delta),
                "home_def_rating_roll10": float(game_row["home_def_rating_roll10"]) + home_def_delta,
                "away_def_rating_roll10": float(game_row["away_def_rating_roll10"]) + away_def_delta,
                "def_rating_diff": (float(game_row["home_def_rating_roll10"]) + home_def_delta)
                - (float(game_row["away_def_rating_roll10"]) + away_def_delta),
                "home_injury_total_impact": home_total_impact,
                "away_injury_total_impact": away_total_impact,
                "injury_impact_diff": home_total_impact - away_total_impact,
                "home_players_flagged": home_players_flagged,
                "away_players_flagged": away_players_flagged,
            }
        ]
    )

    if "home_pace_roll10" in game_row.index and pd.notna(game_row["home_pace_roll10"]) and pd.notna(game_row["away_pace_roll10"]):
        row["home_pace_roll10"] = float(game_row["home_pace_roll10"])
        row["away_pace_roll10"] = float(game_row["away_pace_roll10"])
        row["pace_diff"] = float(game_row["home_pace_roll10"]) - float(game_row["away_pace_roll10"])

    return row


def _predict(feature_row: pd.DataFrame, win_model, margin_model, total_model) -> dict[str, float]:
    home_win_prob = float(win_model.predict_proba(feature_row)[0, 1])
    predicted_margin = float(margin_model.predict(feature_row)[0])
    predicted_total = float(total_model.predict(feature_row)[0])
    expected_home_score = (predicted_total + predicted_margin) / 2.0
    expected_away_score = (predicted_total - predicted_margin) / 2.0

    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1.0 - home_win_prob, 4),
        "predicted_margin": round(predicted_margin, 2),
        "predicted_total": round(predicted_total, 2),
        "expected_home_score": round(expected_home_score, 1),
        "expected_away_score": round(expected_away_score, 1),
    }


def simulate_game_prediction(
    date: str,
    home_team: str,
    away_team: str,
    overrides: list[dict[str, object]] | None,
    games_input: Path,
    raw_games_input: Path,
    player_features_input: Path,
    team_history_input: Path,
    injuries_input: Path | None,
    win_model_path: Path,
    margin_model_path: Path,
    total_model_path: Path,
) -> dict[str, object]:
    games_df = _load_games(games_input)
    raw_games_df = _load_games(raw_games_input)
    player_features_df = _load_player_features(player_features_input)
    team_history_df = _load_team_history(team_history_input)

    target_date = pd.Timestamp(date)
    home_team = home_team.upper().strip()
    away_team = away_team.upper().strip()

    game_match = games_df[
        (games_df["game_date"] == target_date)
        & (games_df["home_team"] == home_team)
        & (games_df["away_team"] == away_team)
    ]
    if game_match.empty:
        game_row_df = build_upcoming_feature_row(
            team_history_df=team_history_df,
            date=date,
            game_id=f"{date}-{away_team}-{home_team}",
            home_team=home_team,
            away_team=away_team,
        )
        game_row = game_row_df.iloc[0]
    else:
        game_row = game_match.iloc[0]

    injury_rows: list[dict[str, object]] = []
    if injuries_input:
        baseline_injuries = load_injuries(injuries_input)
        injury_rows.extend(baseline_injuries.to_dict(orient="records"))
    if overrides:
        enriched_overrides: list[dict[str, object]] = []
        for override in overrides:
            item = dict(override)
            item["game_date"] = date
            item["game_id"] = str(game_row["game_id"])

            if not item.get("team"):
                if item.get("resolved_team"):
                    item["team"] = item["resolved_team"]
                else:
                    player_id = item.get("player_id")
                    if player_id is not None:
                        prior_rows = player_features_df[
                            (player_features_df["player_id"] == player_id)
                            & (player_features_df["game_date"] < target_date)
                        ].sort_values(["game_date", "game_id"])
                        if not prior_rows.empty:
                            item["team"] = str(prior_rows.iloc[-1]["team"])

            if not item.get("player_name"):
                player_id = item.get("player_id")
                if player_id is not None:
                    prior_rows = player_features_df[
                        (player_features_df["player_id"] == player_id)
                        & (player_features_df["game_date"] < target_date)
                    ].sort_values(["game_date", "game_id"])
                    if not prior_rows.empty:
                        item["player_name"] = str(prior_rows.iloc[-1]["player_name"])

            if item.get("player_id") is not None:
                prior_rows = player_features_df[
                    (player_features_df["player_id"] == item["player_id"])
                    & (player_features_df["game_date"] < target_date)
                ].sort_values(["game_date", "game_id"])
                if not prior_rows.empty:
                    latest = prior_rows.iloc[-1]
                    item.setdefault("projected_minutes", latest.get("projected_minutes"))
                    item.setdefault("offensive_impact", latest.get("offensive_impact"))
                    item.setdefault("defensive_impact", latest.get("defensive_impact"))

            enriched_overrides.append(item)

        injury_rows.extend(enriched_overrides)

    injuries_df = pd.DataFrame(
        injury_rows,
        columns=[
            "game_date",
            "game_id",
            "team",
            "player_id",
            "player_name",
            "resolved_team",
            "status",
            "minutes_limit",
            "projected_minutes",
            "offensive_impact",
            "defensive_impact",
        ],
    )
    if not injuries_df.empty:
        injuries_df["game_date"] = pd.to_datetime(injuries_df["game_date"], utc=False)

    team_games_df = _game_teams_from_raw(raw_games_df)
    team_games_df = _append_game_teams(
        team_games_df=team_games_df,
        game_id=str(game_row["game_id"]),
        date=date,
        home_team=home_team,
        away_team=away_team,
    )
    scenario_injuries = build_injury_features(team_games_df, player_features_df, injuries_df)
    scenario_game_injuries = scenario_injuries[
        (scenario_injuries["game_date"] == target_date)
        & (scenario_injuries["game_id"] == game_row["game_id"])
    ]

    home_injury = scenario_game_injuries[scenario_game_injuries["team"] == home_team]
    away_injury = scenario_game_injuries[scenario_game_injuries["team"] == away_team]
    home_injury_row = home_injury.iloc[0] if not home_injury.empty else None
    away_injury_row = away_injury.iloc[0] if not away_injury.empty else None

    baseline_feature_row = _build_feature_row(game_row, None, None)
    scenario_feature_row = _build_feature_row(game_row, home_injury_row, away_injury_row)

    win_model = joblib.load(win_model_path)
    margin_model = joblib.load(margin_model_path)
    total_model = joblib.load(total_model_path)

    baseline_prediction = _predict(baseline_feature_row, win_model, margin_model, total_model)
    scenario_prediction = _predict(scenario_feature_row, win_model, margin_model, total_model)

    return {
        "game": {
            "game_id": str(game_row["game_id"]),
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
        },
        "baseline_prediction": baseline_prediction,
        "scenario_prediction": scenario_prediction,
        "delta": {
            "home_win_prob_change": round(
                scenario_prediction["home_win_prob"] - baseline_prediction["home_win_prob"], 4
            ),
            "away_win_prob_change": round(
                scenario_prediction["away_win_prob"] - baseline_prediction["away_win_prob"], 4
            ),
            "predicted_margin_change": round(
                scenario_prediction["predicted_margin"] - baseline_prediction["predicted_margin"], 2
            ),
            "predicted_total_change": round(
                scenario_prediction["predicted_total"] - baseline_prediction["predicted_total"], 2
            ),
            "expected_home_score_change": round(
                scenario_prediction["expected_home_score"] - baseline_prediction["expected_home_score"], 1
            ),
            "expected_away_score_change": round(
                scenario_prediction["expected_away_score"] - baseline_prediction["expected_away_score"], 1
            ),
        },
        "injury_context": {
            "injury_rating_scale": INJURY_RATING_SCALE,
            "home_injury_off_impact": round(float(home_injury_row["injury_off_impact"]) if home_injury_row is not None else 0.0, 4),
            "home_injury_def_impact": round(float(home_injury_row["injury_def_impact"]) if home_injury_row is not None else 0.0, 4),
            "away_injury_off_impact": round(float(away_injury_row["injury_off_impact"]) if away_injury_row is not None else 0.0, 4),
            "away_injury_def_impact": round(float(away_injury_row["injury_def_impact"]) if away_injury_row is not None else 0.0, 4),
            "home_off_rating_delta": round((float(home_injury_row["injury_off_impact"]) if home_injury_row is not None else 0.0) * INJURY_RATING_SCALE, 4),
            "home_def_rating_delta": round((float(home_injury_row["injury_def_impact"]) if home_injury_row is not None else 0.0) * INJURY_RATING_SCALE, 4),
            "away_off_rating_delta": round((float(away_injury_row["injury_off_impact"]) if away_injury_row is not None else 0.0) * INJURY_RATING_SCALE, 4),
            "away_def_rating_delta": round((float(away_injury_row["injury_def_impact"]) if away_injury_row is not None else 0.0) * INJURY_RATING_SCALE, 4),
            "home_flagged_players": "" if home_injury_row is None else str(home_injury_row["flagged_players"]),
            "away_flagged_players": "" if away_injury_row is None else str(away_injury_row["flagged_players"]),
        },
    }


def main() -> None:
    args = parse_args()
    result = simulate_game_prediction(
        date=args.date,
        home_team=args.home_team,
        away_team=args.away_team,
        overrides=[_parse_override(item, args.date) for item in args.override],
        games_input=args.games_input,
        raw_games_input=args.raw_games_input,
        player_features_input=args.player_features_input,
        team_history_input=args.games_input.parent / "team_games_enriched.csv",
        injuries_input=args.injuries_input,
        win_model_path=args.win_model,
        margin_model_path=args.margin_model,
        total_model_path=args.total_model,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
