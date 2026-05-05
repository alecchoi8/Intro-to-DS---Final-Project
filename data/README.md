# Data Setup

The raw Kaggle files are not committed to Git. Download them locally before running preprocessing notebooks or scripts.

## Current Snapshot

- Dataset: <https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats>
- Kaggle handle: `sumitrodatta/nba-aba-baa-stats`
- Downloaded on: `2026-04-30`
- Download method: KaggleHub `dataset_download`
- Extracted raw CSV files: `22`
- Extracted CSV size: approximately `30.87 MB`

## KaggleHub Download

```powershell
python scripts/download_kaggle_dataset.py
```

## Manual Download Fallback

1. Open the Kaggle dataset page: <https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats>
2. Download the latest available dataset snapshot.
3. Save or extract the CSV files under `data/raw/`.
4. Record the exact Kaggle version, download timestamp, and file list in this README or a future data snapshot note.

## Analysis Scope

- League filter: NBA only.
- Target: player-season points per game.
- Unit of observation: one player in one season.
- Traded players: aggregate manually across teams using season totals before calculating per-game metrics.
- Processed output: `data/processed/nba_player_seasons_modeling.csv`

## Expected Useful Files

The current snapshot includes player per-game stats, player totals, player advanced stats, player shooting stats, and season/player information tables. See [DATA_INVENTORY.md](DATA_INVENTORY.md) for the inspected file list and key schema notes.

See [PROCESSED_DATASET.md](PROCESSED_DATASET.md) for the cleaned modeling dataset schema and validation summary.
