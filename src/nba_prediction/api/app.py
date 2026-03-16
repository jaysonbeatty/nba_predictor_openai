from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nba_prediction.simulation.simulate_game import simulate_game_prediction
from nba_prediction.schedule.upcoming_games import get_prediction_games_for_date, load_team_history
from nba_prediction.utils.player_lookup import load_player_directory, resolve_player_name, search_players


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAMES_INPUT = PROJECT_ROOT / "data" / "processed" / "games_model_base.csv"
DEFAULT_RAW_GAMES_INPUT = PROJECT_ROOT / "data" / "raw" / "games_raw.csv"
DEFAULT_PLAYER_FEATURES_INPUT = PROJECT_ROOT / "data" / "processed" / "player_impact_features.csv"
DEFAULT_TEAM_HISTORY_INPUT = PROJECT_ROOT / "data" / "processed" / "team_games_enriched.csv"
DEFAULT_WIN_MODEL = PROJECT_ROOT / "artifacts" / "models" / "baseline_with_historical_injuries_win_model.joblib"
DEFAULT_MARGIN_MODEL = PROJECT_ROOT / "artifacts" / "models" / "baseline_with_historical_injuries_margin_model.joblib"
DEFAULT_TOTAL_MODEL = PROJECT_ROOT / "artifacts" / "models" / "baseline_with_historical_injuries_total_model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NBA Prediction API", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InjuryOverride(BaseModel):
    player_id: int | None = None
    player_name: str | None = None
    status: Literal["available", "limited", "out"]
    minutes_limit: float | None = None


class SimulateGameRequest(BaseModel):
    date: str = Field(description="Game date in YYYY-MM-DD format.")
    home_team: str = Field(description="Home team abbreviation, for example DEN.")
    away_team: str = Field(description="Away team abbreviation, for example HOU.")
    overrides: list[InjuryOverride] = Field(default_factory=list)
    injuries_input: str | None = Field(default=None, description="Optional CSV path for baseline injuries.")


class ScenarioRequest(BaseModel):
    name: str
    overrides: list[InjuryOverride] = Field(default_factory=list)


class CompareScenariosRequest(BaseModel):
    date: str = Field(description="Game date in YYYY-MM-DD format.")
    home_team: str = Field(description="Home team abbreviation, for example DEN.")
    away_team: str = Field(description="Away team abbreviation, for example HOU.")
    scenarios: list[ScenarioRequest]
    injuries_input: str | None = Field(default=None, description="Optional CSV path for baseline injuries.")


def _load_games() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_GAMES_INPUT)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["game_id"] = df["game_id"].astype(str)
    return df


def _load_player_directory() -> pd.DataFrame:
    return load_player_directory(DEFAULT_PLAYER_FEATURES_INPUT)


def _resolve_override(item: InjuryOverride, directory_df: pd.DataFrame, home_team: str, away_team: str, date: str) -> dict[str, object]:
    if item.player_id is None and not item.player_name:
        raise ValueError("Each override must include player_id or player_name.")

    resolved_player_id = item.player_id
    resolved_player_name = item.player_name
    if resolved_player_id is None and item.player_name:
        match = resolve_player_name(
            directory_df,
            item.player_name,
            preferred_teams=[home_team, away_team],
        )
        resolved_player_id = int(match["player_id"])
        resolved_player_name = str(match["player_name"])
        resolved_team = str(match["team"])
    else:
        player_rows = directory_df[directory_df["player_id"] == resolved_player_id]
        resolved_team = str(player_rows.iloc[0]["team"]) if not player_rows.empty else None

    return {
        "game_date": date,
        "player_id": resolved_player_id,
        "player_name": resolved_player_name,
        "resolved_team": resolved_team,
        "status": item.status,
        "minutes_limit": item.minutes_limit,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/games")
def get_games(date: str) -> dict[str, object]:
    games_df = _load_games()
    team_history_df = load_team_history(DEFAULT_TEAM_HISTORY_INPUT)
    games = get_prediction_games_for_date(date=date, games_model_df=games_df, team_history_df=team_history_df)
    if games.empty:
        return {"date": date, "games": []}

    records = (
        games.sort_values(["game_date", "game_id"])[["game_id", "game_date", "home_team", "away_team"]]
        .assign(game_date=lambda df: df["game_date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    return {"date": date, "games": records}


@app.get("/players")
def get_players(query: str, limit: int = 10) -> dict[str, object]:
    directory_df = _load_player_directory()
    matches = search_players(directory_df, query=query, limit=limit)
    return {"query": query, "matches": matches}


@app.post("/simulate-game")
def simulate_game(request: SimulateGameRequest) -> dict[str, object]:
    try:
        directory_df = _load_player_directory()
        overrides = [
            _resolve_override(item, directory_df, request.home_team, request.away_team, request.date)
            for item in request.overrides
        ]
        result = simulate_game_prediction(
            date=request.date,
            home_team=request.home_team,
            away_team=request.away_team,
            overrides=overrides,
            games_input=DEFAULT_GAMES_INPUT,
            raw_games_input=DEFAULT_RAW_GAMES_INPUT,
            player_features_input=DEFAULT_PLAYER_FEATURES_INPUT,
            team_history_input=DEFAULT_TEAM_HISTORY_INPUT,
            injuries_input=Path(request.injuries_input) if request.injuries_input else None,
            win_model_path=DEFAULT_WIN_MODEL,
            margin_model_path=DEFAULT_MARGIN_MODEL,
            total_model_path=DEFAULT_TOTAL_MODEL,
        )
        result["resolved_overrides"] = overrides
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare-scenarios")
def compare_scenarios(request: CompareScenariosRequest) -> dict[str, object]:
    try:
        directory_df = _load_player_directory()
        baseline_result = simulate_game_prediction(
            date=request.date,
            home_team=request.home_team,
            away_team=request.away_team,
            overrides=[],
            games_input=DEFAULT_GAMES_INPUT,
            raw_games_input=DEFAULT_RAW_GAMES_INPUT,
            player_features_input=DEFAULT_PLAYER_FEATURES_INPUT,
            team_history_input=DEFAULT_TEAM_HISTORY_INPUT,
            injuries_input=Path(request.injuries_input) if request.injuries_input else None,
            win_model_path=DEFAULT_WIN_MODEL,
            margin_model_path=DEFAULT_MARGIN_MODEL,
            total_model_path=DEFAULT_TOTAL_MODEL,
        )

        scenario_results: list[dict[str, object]] = []
        for scenario in request.scenarios:
            overrides = [
                _resolve_override(item, directory_df, request.home_team, request.away_team, request.date)
                for item in scenario.overrides
            ]
            result = simulate_game_prediction(
                date=request.date,
                home_team=request.home_team,
                away_team=request.away_team,
                overrides=overrides,
                games_input=DEFAULT_GAMES_INPUT,
                raw_games_input=DEFAULT_RAW_GAMES_INPUT,
                player_features_input=DEFAULT_PLAYER_FEATURES_INPUT,
                team_history_input=DEFAULT_TEAM_HISTORY_INPUT,
                injuries_input=Path(request.injuries_input) if request.injuries_input else None,
                win_model_path=DEFAULT_WIN_MODEL,
                margin_model_path=DEFAULT_MARGIN_MODEL,
                total_model_path=DEFAULT_TOTAL_MODEL,
            )
            scenario_results.append(
                {
                    "name": scenario.name,
                    "resolved_overrides": overrides,
                    "scenario_prediction": result["scenario_prediction"],
                    "delta": result["delta"],
                    "injury_context": result["injury_context"],
                }
            )

        return {
            "game": baseline_result["game"],
            "baseline_prediction": baseline_result["baseline_prediction"],
            "scenarios": scenario_results,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
