from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def count_rows(csv_path: Path) -> int:
    with csv_path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def print_inventory() -> None:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    print(f"CSV files: {len(csv_files)}")
    for csv_path in csv_files:
        columns = pd.read_csv(csv_path, nrows=0).columns
        print(f"{csv_path.name}\trows={count_rows(csv_path)}\tcols={len(columns)}")


def print_player_table_summary() -> None:
    per_game = pd.read_csv(RAW_DATA_DIR / "Player Per Game.csv")
    nba = per_game[per_game["lg"].eq("NBA")].copy()
    nba["is_multi_total"] = nba["team"].astype(str).str.match(r"^\d+TM$", na=False)

    multi = nba.groupby(["season", "player_id"]).size().reset_index(name="rows")
    multi = multi[multi["rows"] > 1]
    multi_rows = nba.merge(multi[["season", "player_id"]], on=["season", "player_id"], how="inner")

    one_row = nba[~nba.duplicated(["season", "player_id"], keep=False) | nba["is_multi_total"]]

    print("\nNBA Player Per Game summary")
    print(f"rows={len(nba)}")
    print(f"seasons={nba['season'].min()}-{nba['season'].max()}")
    print(f"unique_player_seasons={nba[['season', 'player_id']].drop_duplicates().shape[0]}")
    print(f"multi_row_player_seasons={len(multi)}")
    print(f"rows_in_multi_row_groups={len(multi_rows)}")
    print(f"one_row_after_total_selection={len(one_row)}")
    labels = sorted(nba.loc[nba["is_multi_total"], "team"].unique())
    print(f"multi_team_labels={labels}")


def main() -> None:
    print_inventory()
    print_player_table_summary()


if __name__ == "__main__":
    main()
