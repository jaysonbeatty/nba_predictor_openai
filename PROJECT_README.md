# NBA Prediction Model

This project starts with one job: build a clean, leak-free game-level dataset
for NBA pregame prediction.

## Step 1 Goal

Create a final table at `data/processed/games_model_base.csv` with one row per
game from the home-team perspective.

Expected columns:

- `game_id`
- `game_date`
- `season`
- `home_team`
- `away_team`
- `home_points`
- `away_points`
- `home_win`
- `margin`
- `total`
- `home_rest_days`
- `away_rest_days`
- `home_b2b`
- `away_b2b`
- `home_off_rating_roll10`
- `away_off_rating_roll10`
- `home_def_rating_roll10`
- `away_def_rating_roll10`
- `home_pace_roll10`
- `away_pace_roll10`

## Project Layout

```text
nba_prediction_model/
  data/
    raw/
    processed/
  notebooks/
  src/
    nba_prediction/
      ingest/
      features/
      pipeline/
  requirements.txt
```

## Input Contract

Put your raw game file at `data/raw/games_raw.csv`.

Required columns:

- `game_id`
- `game_date`
- `season`
- `home_team`
- `away_team`
- `home_points`
- `away_points`

Optional columns:

- `home_off_rating`
- `away_off_rating`
- `home_def_rating`
- `away_def_rating`
- `home_pace`
- `away_pace`

If rating columns are missing, the pipeline falls back to points-based values.
If pace is missing, pace rolling features stay null until you add that source.
When you run `--drop-incomplete`, the pipeline only drops on rolling columns that
actually have data, so missing pace will not wipe out the full dataset.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Step 1

```bash
PYTHONPATH=src python -m nba_prediction.pipeline.run_step1
```

To drop early-season rows without enough history:

```bash
PYTHONPATH=src python -m nba_prediction.pipeline.run_step1 --drop-incomplete
```

## Fetch Raw Games

Use `nba_api` to create `data/raw/games_raw.csv` first:

```bash
PYTHONPATH=src python -m nba_prediction.ingest.fetch_games_nba_api \
  --seasons 2023-24 2024-25 2025-26 \
  --season-type regular
```

Then run Step 1:

```bash
PYTHONPATH=src python -m nba_prediction.pipeline.run_step1 --drop-incomplete
```

## Train Baseline Models

Once `games_model_base.csv` exists:

```bash
PYTHONPATH=src python -m nba_prediction.models.train_baseline
```

That trains:

- logistic regression for `home_win`
- linear regression for `margin`
- linear regression for `total`

Outputs:

- `artifacts/models/baseline_win_model.joblib`
- `artifacts/models/baseline_margin_model.joblib`
- `artifacts/models/baseline_total_model.joblib`
- `artifacts/metrics/baseline_metrics.json`

## Fetch Player Game Logs

Create `data/raw/player_game_stats.csv`:

```bash
PYTHONPATH=src python -m nba_prediction.ingest.fetch_player_games_nba_api \
  --seasons 2023-24 2024-25 2025-26 \
  --season-type regular
```

## Build Player Impact Features

```bash
PYTHONPATH=src python -m nba_prediction.features.build_player_impact_features
```

This creates `data/processed/player_impact_features.csv` with:

- rolling `projected_minutes`
- rolling `offensive_impact`
- rolling `defensive_impact`

## Build Injury Features

Copy `data/raw/injury_status_template.csv` to `data/raw/injury_status.csv` and
replace the sample rows with real statuses for the game date you care about.

Required injury columns:

- `game_date`
- `player_id`
- `status` with values `available`, `limited`, or `out`
- optional `minutes_limit`

Then run:

```bash
PYTHONPATH=src python -m nba_prediction.features.build_injury_features
```

That produces `data/processed/team_injury_features.csv` with team-level injury
impact deltas that can be merged into later scenario or model layers.

## Build Historical Availability Proxy

To create injury-like history for training without a full external injury feed:

```bash
PYTHONPATH=src python -m nba_prediction.features.build_historical_availability_proxy
```

This writes `data/processed/historical_injury_status_proxy.csv` and marks a
player as historically unavailable when:

- they had enough prior-game history
- they were projected for meaningful minutes
- they did not appear in that scheduled game

You can then rebuild team injury features from the proxy:

```bash
PYTHONPATH=src python -m nba_prediction.features.build_injury_features \
  --injuries-input data/processed/historical_injury_status_proxy.csv \
  --output data/processed/team_injury_features_historical.csv
```

And merge them into the game table:

```bash
PYTHONPATH=src python -m nba_prediction.features.merge_game_injury_features \
  --injuries-input data/processed/team_injury_features_historical.csv \
  --output data/processed/games_model_with_historical_injuries.csv
```

To inspect only non-zero injury rows:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/processed/team_injury_features.csv')
print(df[df['players_flagged'] > 0].head(20).to_string(index=False))
PY
```

## Merge Injury Features Into Games

```bash
PYTHONPATH=src python -m nba_prediction.features.merge_game_injury_features
```

This creates `data/processed/games_model_with_injuries.csv` with:

- `home_injury_total_impact`
- `away_injury_total_impact`
- `home_players_flagged`
- `away_players_flagged`

## First Scenario Simulator

Use the historical-injury-trained models plus manual overrides:

```bash
PYTHONPATH=src python -m nba_prediction.simulation.simulate_game \
  --date 2026-03-11 \
  --home-team LAL \
  --away-team DEN \
  --override 203999:out \
  --override 1627750:limited:20
```

Override format is `player_id:status[:minutes_limit]`.

The simulator returns:

- `baseline_prediction`
- `scenario_prediction`
- `delta`
- `injury_context`

## Run API

Install API dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
PYTHONPATH=src uvicorn nba_prediction.api.app:app --reload
```

Full refresh helper:

```bash
bash refresh_all.sh
```

Endpoints:

- `GET /health`
- `GET /games?date=2026-03-11`
- `GET /players?query=jokic`
- `POST /simulate-game`
- `POST /compare-scenarios`

Example:

```bash
curl -X POST http://127.0.0.1:8000/simulate-game \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-03-11",
    "home_team": "DEN",
    "away_team": "HOU",
    "overrides": [
      {"player_name": "Nikola Jokic", "status": "out"}
    ]
  }'
```

When an override uses `player_name`, the API prefers players on the selected
home/away teams and returns `resolved_overrides` in the response so the chat
layer can state its assumptions explicitly.

Player lookup example:

```bash
curl "http://127.0.0.1:8000/players?query=jokic"
```

Scenario comparison example:

```bash
curl -X POST http://127.0.0.1:8000/compare-scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-03-11",
    "home_team": "DEN",
    "away_team": "HOU",
    "scenarios": [
      {
        "name": "Jokic out",
        "overrides": [
          {"player_name": "Nikola Jokic", "status": "out"}
        ]
      },
      {
        "name": "Jokic out plus Murray limited",
        "overrides": [
          {"player_name": "Nikola Jokic", "status": "out"},
          {"player_name": "Jamal Murray", "status": "limited", "minutes_limit": 20}
        ]
      }
    ]
  }'
```
