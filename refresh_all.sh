#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in $ROOT_DIR"
  exit 1
fi

source .venv/bin/activate

echo "Refreshing games..."
PYTHONPATH=src python3 -m nba_prediction.ingest.fetch_games_nba_api \
  --seasons 2023-24 2024-25 2025-26 \
  --season-type regular

echo "Rebuilding base game features..."
PYTHONPATH=src python3 -m nba_prediction.pipeline.run_step1 --drop-incomplete

echo "Refreshing player logs..."
PYTHONPATH=src python3 -m nba_prediction.ingest.fetch_player_games_nba_api \
  --seasons 2023-24 2024-25 2025-26 \
  --season-type regular

echo "Rebuilding player impact features..."
PYTHONPATH=src python3 -m nba_prediction.features.build_player_impact_features

echo "Rebuilding historical availability proxy..."
PYTHONPATH=src python3 -m nba_prediction.features.build_historical_availability_proxy

echo "Rebuilding historical injury features..."
PYTHONPATH=src python3 -m nba_prediction.features.build_injury_features \
  --injuries-input data/processed/historical_injury_status_proxy.csv \
  --output data/processed/team_injury_features_historical.csv

echo "Merging historical injury features..."
PYTHONPATH=src python3 -m nba_prediction.features.merge_game_injury_features \
  --injuries-input data/processed/team_injury_features_historical.csv \
  --output data/processed/games_model_with_historical_injuries.csv

echo "Retraining historical injury models..."
PYTHONPATH=src python3 -m nba_prediction.models.train_baseline \
  --input data/processed/games_model_with_historical_injuries.csv \
  --metrics-output artifacts/metrics/baseline_with_historical_injuries_metrics.json \
  --model-name baseline_with_historical_injuries

echo
echo "Refresh complete."
echo "Start the API with:"
echo "  PYTHONPATH=src uvicorn nba_prediction.api.app:app --reload --port 8000"
