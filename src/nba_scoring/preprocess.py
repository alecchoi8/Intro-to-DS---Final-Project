from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_scoring.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SNAPSHOT_DATE


PLAYER_TOTALS_FILE = "Player Totals.csv"
MODELING_DATASET_FILE = "nba_player_seasons_modeling.csv"
PREPROCESSING_SUMMARY_FILE = "preprocessing_summary.json"

GROUP_KEYS = ["season", "player_id"]
COUNT_COLUMNS = [
    "g",
    "gs",
    "mp",
    "fg",
    "fga",
    "x3p",
    "x3pa",
    "x2p",
    "x2pa",
    "ft",
    "fta",
    "orb",
    "drb",
    "trb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "trp_dbl",
]
PER_GAME_COLUMNS = [
    "gs",
    "mp",
    "fg",
    "fga",
    "x3p",
    "x3pa",
    "x2p",
    "x2pa",
    "ft",
    "fta",
    "orb",
    "drb",
    "trb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
]
PERCENT_COLUMNS = [
    "fg_percent",
    "x3p_percent",
    "x2p_percent",
    "e_fg_percent",
    "ft_percent",
]


def is_multi_team_total(team: pd.Series) -> pd.Series:
    return team.astype(str).str.match(r"^\d+TM$", na=False)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def first_non_null(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def load_player_totals(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    path = raw_data_dir / PLAYER_TOTALS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_kaggle_dataset.py or place raw CSVs in data/raw."
        )
    return pd.read_csv(path)


def build_metadata(nba_totals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in nba_totals.groupby(GROUP_KEYS, sort=False):
        season, player_id = keys
        total_rows = group[group["is_multi_team_total"]]
        source = total_rows if not total_rows.empty else group
        team_rows = group[~group["is_multi_team_total"]]
        teams = sorted(team_rows["team"].dropna().astype(str).unique())

        rows.append(
            {
                "season": season,
                "player_id": player_id,
                "player": first_non_null(source["player"]),
                "lg": "NBA",
                "age": first_non_null(source["age"]),
                "pos": first_non_null(source["pos"]),
                "team": first_non_null(source["team"]),
                "teams": "|".join(teams),
                "team_count": len(teams),
                "traded": len(teams) > 1,
                "source_rows": len(group),
                "source_team_rows": len(team_rows),
                "source_total_row_present": bool(not total_rows.empty),
            }
        )

    return pd.DataFrame(rows)


def aggregate_counting_stats(nba_totals: pd.DataFrame) -> pd.DataFrame:
    team_rows = nba_totals[~nba_totals["is_multi_team_total"]].copy()
    return team_rows.groupby(GROUP_KEYS, as_index=False)[COUNT_COLUMNS].sum(min_count=1)


def recompute_rates(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    # For seasons before the NBA adopted the three-point line, attempts and makes are true zeros.
    pre_three_point_line = dataset["season"] < 1980
    for column in ["x3p", "x3pa"]:
        dataset.loc[pre_three_point_line & dataset[column].isna(), column] = 0

    for column in PER_GAME_COLUMNS:
        dataset[f"{column}_per_game"] = safe_divide(dataset[column], dataset["g"])

    dataset["fg_percent"] = safe_divide(dataset["fg"], dataset["fga"])
    dataset["x3p_percent"] = safe_divide(dataset["x3p"], dataset["x3pa"])
    dataset["x2p_percent"] = safe_divide(dataset["x2p"], dataset["x2pa"])
    dataset["e_fg_percent"] = safe_divide(dataset["fg"] + 0.5 * dataset["x3p"], dataset["fga"])
    dataset["ft_percent"] = safe_divide(dataset["ft"], dataset["fta"])

    return dataset


def validate_against_total_rows(nba_totals: pd.DataFrame, aggregated: pd.DataFrame) -> dict[str, Any]:
    total_rows = nba_totals[nba_totals["is_multi_team_total"]][GROUP_KEYS + ["team"] + COUNT_COLUMNS]
    merged = total_rows.merge(aggregated, on=GROUP_KEYS, suffixes=("_total", "_manual"))

    mismatches: dict[str, int] = {}
    for column in COUNT_COLUMNS:
        supplied = merged[f"{column}_total"]
        manual = merged[f"{column}_manual"]
        both_missing = supplied.isna() & manual.isna()
        equal_values = np.isclose(supplied.fillna(-999_999), manual.fillna(-999_999), equal_nan=True)
        mismatch_count = int((~(both_missing | equal_values)).sum())
        if mismatch_count:
            mismatches[column] = mismatch_count

    return {
        "supplied_multi_team_total_rows": int(len(total_rows)),
        "manual_total_rows_compared": int(len(merged)),
        "counting_stat_mismatch_columns": mismatches,
        "counting_stat_mismatch_total": int(sum(mismatches.values())),
    }


def build_modeling_dataset(raw_data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    totals = load_player_totals(raw_data_dir)
    nba_totals = totals[totals["lg"].eq("NBA")].copy()
    nba_totals["is_multi_team_total"] = is_multi_team_total(nba_totals["team"])

    metadata = build_metadata(nba_totals)
    aggregated = aggregate_counting_stats(nba_totals)
    validation = validate_against_total_rows(nba_totals, aggregated)

    dataset = metadata.merge(aggregated, on=GROUP_KEYS, how="inner", validate="one_to_one")
    dataset = recompute_rates(dataset)

    front_columns = [
        "season",
        "player_id",
        "player",
        "lg",
        "age",
        "pos",
        "team",
        "teams",
        "team_count",
        "traded",
        "source_rows",
        "source_team_rows",
        "source_total_row_present",
    ]
    remaining_columns = [column for column in dataset.columns if column not in front_columns]
    dataset = dataset[front_columns + remaining_columns].sort_values(
        ["season", "player", "player_id"], ascending=[False, True, True]
    )

    summary = {
        "snapshot_date": SNAPSHOT_DATE,
        "raw_rows": int(len(totals)),
        "nba_rows": int(len(nba_totals)),
        "processed_rows": int(len(dataset)),
        "season_min": int(dataset["season"].min()),
        "season_max": int(dataset["season"].max()),
        "unique_player_seasons": int(dataset[GROUP_KEYS].drop_duplicates().shape[0]),
        "traded_player_seasons": int(dataset["traded"].sum()),
        "rows_with_missing_target": int(dataset["pts_per_game"].isna().sum()),
        "validation": validation,
    }

    return dataset, summary


def write_modeling_dataset(
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> tuple[Path, Path, dict[str, Any]]:
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    dataset, summary = build_modeling_dataset(raw_data_dir)

    dataset_path = processed_data_dir / MODELING_DATASET_FILE
    summary_path = processed_data_dir / PREPROCESSING_SUMMARY_FILE

    dataset.to_csv(dataset_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return dataset_path, summary_path, summary
