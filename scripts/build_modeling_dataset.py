from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nba_scoring.preprocess import write_modeling_dataset


def main() -> None:
    dataset_path, summary_path, summary = write_modeling_dataset()
    print(f"Wrote modeling dataset: {dataset_path}")
    print(f"Wrote preprocessing summary: {summary_path}")
    print(f"Processed rows: {summary['processed_rows']}")
    print(f"NBA seasons: {summary['season_min']}-{summary['season_max']}")
    print(f"Traded player-seasons: {summary['traded_player_seasons']}")
    print(
        "Manual aggregation mismatches: "
        f"{summary['validation']['counting_stat_mismatch_total']}"
    )


if __name__ == "__main__":
    main()
