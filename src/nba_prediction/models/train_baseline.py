from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline


BASE_FEATURES = [
    "rest_advantage",
    "home_off_rating_roll10",
    "away_off_rating_roll10",
    "off_rating_diff",
    "home_def_rating_roll10",
    "away_def_rating_roll10",
    "def_rating_diff",
]

OPTIONAL_FEATURES = [
    "home_pace_roll10",
    "away_pace_roll10",
    "pace_diff",
]

INJURY_FEATURES = [
    "home_injury_total_impact",
    "away_injury_total_impact",
    "injury_impact_diff",
    "home_players_flagged",
    "away_players_flagged",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_input = project_root / "data" / "processed" / "games_model_with_injuries.csv"
    default_metrics = project_root / "artifacts" / "metrics" / "baseline_with_injuries_metrics.json"
    default_models = project_root / "artifacts" / "models"

    parser = argparse.ArgumentParser(description="Train baseline NBA prediction models.")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to games_model_base.csv.")
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="2025-06-30",
        help="Inclusive train cutoff in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2025-07-01",
        help="Inclusive test start in YYYY-MM-DD.",
    )
    parser.add_argument("--metrics-output", type=Path, default=default_metrics, help="Metrics JSON output path.")
    parser.add_argument("--models-dir", type=Path, default=default_models, help="Directory for serialized models.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_with_injuries",
        help="Filename prefix for saved model artifacts.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = df.copy()
    if "home_injury_total_impact" not in frame.columns:
        frame["home_injury_total_impact"] = 0.0
    if "away_injury_total_impact" not in frame.columns:
        frame["away_injury_total_impact"] = 0.0
    if "home_players_flagged" not in frame.columns:
        frame["home_players_flagged"] = 0
    if "away_players_flagged" not in frame.columns:
        frame["away_players_flagged"] = 0

    frame["rest_advantage"] = frame["home_rest_days"].fillna(0) - frame["away_rest_days"].fillna(0)
    frame["off_rating_diff"] = frame["home_off_rating_roll10"] - frame["away_off_rating_roll10"]
    frame["def_rating_diff"] = frame["home_def_rating_roll10"] - frame["away_def_rating_roll10"]
    frame["pace_diff"] = frame["home_pace_roll10"] - frame["away_pace_roll10"]
    frame["injury_impact_diff"] = frame["home_injury_total_impact"].fillna(0) - frame["away_injury_total_impact"].fillna(0)

    feature_columns = list(BASE_FEATURES)
    for column in OPTIONAL_FEATURES:
        if frame[column].notna().any():
            feature_columns.append(column)
    for column in INJURY_FEATURES:
        if column in frame.columns:
            feature_columns.append(column)

    return frame, feature_columns


def time_split(df: pd.DataFrame, train_end_date: str, test_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cutoff = pd.Timestamp(train_end_date)
    test_cutoff = pd.Timestamp(test_start_date)

    train_df = df[df["game_date"] <= train_cutoff].copy()
    test_df = df[df["game_date"] >= test_cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Empty split: train_rows={len(train_df)} test_rows={len(test_df)}. "
            "Adjust --train-end-date and --test-start-date."
        )

    return train_df, test_df


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    frame, feature_columns = build_feature_frame(df)
    train_df, test_df = time_split(frame, args.train_end_date, args.test_start_date)

    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]

    win_model = make_pipeline(LogisticRegression(max_iter=1000))
    margin_model = make_pipeline(LinearRegression())
    total_model = make_pipeline(LinearRegression())

    win_model.fit(x_train, train_df["home_win"])
    margin_model.fit(x_train, train_df["margin"])
    total_model.fit(x_train, train_df["total"])

    win_probs = win_model.predict_proba(x_test)[:, 1]
    win_preds = (win_probs >= 0.5).astype(int)
    margin_preds = margin_model.predict(x_test)
    total_preds = total_model.predict(x_test)

    metrics = {
        "feature_columns": feature_columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["game_date"].min().date()),
        "train_date_max": str(train_df["game_date"].max().date()),
        "test_date_min": str(test_df["game_date"].min().date()),
        "test_date_max": str(test_df["game_date"].max().date()),
        "win_model": {
            "log_loss": float(log_loss(test_df["home_win"], win_probs)),
            "accuracy": float(accuracy_score(test_df["home_win"], win_preds)),
        },
        "margin_model": {
            "mae": float(mean_absolute_error(test_df["margin"], margin_preds)),
        },
        "total_model": {
            "mae": float(mean_absolute_error(test_df["total"], total_preds)),
        },
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.metrics_output)

    joblib.dump(win_model, args.models_dir / f"{args.model_name}_win_model.joblib")
    joblib.dump(margin_model, args.models_dir / f"{args.model_name}_margin_model.joblib")
    joblib.dump(total_model, args.models_dir / f"{args.model_name}_total_model.joblib")

    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved models to {args.models_dir}")


if __name__ == "__main__":
    main()
