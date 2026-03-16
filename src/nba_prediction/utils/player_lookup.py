from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import unicodedata

import pandas as pd


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return (
        ascii_text.strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )


def load_player_directory(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["player_id", "player_name", "team", "game_date", "game_id"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["game_date"] = pd.to_datetime(df["game_date"], utc=False)
    df["game_id"] = df["game_id"].astype(str)
    df["normalized_name"] = df["player_name"].map(_normalize_name)
    df = df.dropna(subset=["player_id"])
    df = df.sort_values(["player_id", "game_date", "game_id"])
    df = df.groupby("player_id", as_index=False).tail(1)
    return df.sort_values(["player_name", "team", "player_id"]).reset_index(drop=True)


def search_players(
    directory_df: pd.DataFrame,
    query: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    normalized_query = _normalize_name(query)
    if not normalized_query:
        return []

    matches = directory_df.copy()
    matches["exact_match"] = matches["normalized_name"] == normalized_query
    matches["starts_with"] = matches["normalized_name"].str.startswith(normalized_query, na=False)
    matches["contains"] = matches["normalized_name"].str.contains(normalized_query, na=False)
    matches["token_match"] = matches["normalized_name"].str.split().map(lambda tokens: normalized_query in tokens)
    matches["score"] = matches["normalized_name"].map(
        lambda candidate: SequenceMatcher(None, normalized_query, candidate).ratio()
    )

    matches = matches.sort_values(
        ["exact_match", "starts_with", "token_match", "contains", "score", "player_name"],
        ascending=[False, False, False, False, False, True],
    ).head(limit)

    return [
        {
            "player_id": int(row.player_id),
            "player_name": row.player_name,
            "team": row.team,
            "match_score": round(float(row.score), 4),
        }
        for row in matches.head(limit).itertuples(index=False)
    ]


def resolve_player_name(
    directory_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    preferred_teams: list[str] | None = None,
) -> dict[str, object]:
    matches = search_players(directory_df, player_name, limit=25)
    if team:
        team_upper = team.upper().strip()
        team_matches = [match for match in matches if match["team"] == team_upper]
        if team_matches:
            matches = team_matches
    if preferred_teams:
        team_set = {value.upper().strip() for value in preferred_teams}
        preferred_matches = [match for match in matches if match["team"] in team_set]
        if preferred_matches:
            matches = preferred_matches

    if not matches:
        raise ValueError(f"No player match found for '{player_name}'.")

    best = matches[0]
    if len(matches) > 1 and matches[0]["match_score"] == matches[1]["match_score"]:
        top_names = ", ".join(f"{item['player_name']} ({item['team']})" for item in matches[:5])
        raise ValueError(f"Ambiguous player name '{player_name}'. Top matches: {top_names}")

    return best
