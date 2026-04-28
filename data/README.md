# Data Setup

The raw Kaggle files are not committed to Git. Download them locally before running preprocessing notebooks or scripts.

## Manual Download

1. Open the Kaggle dataset page: <https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats>
2. Download the latest available dataset snapshot on `2026-04-28`.
3. Save the downloaded ZIP under `data/raw/`.
4. Extract the CSV files into `data/raw/nba-aba-baa-stats/`.
5. Record the exact Kaggle version, download timestamp, and file list in this README or a future data snapshot note.

## Analysis Scope

- League filter: NBA only.
- Target: player-season points per game.
- Unit of observation: one player in one season.
- Traded players: aggregate manually across teams using season totals before calculating per-game metrics.

## Expected Useful Files

The Kaggle data card indicates that the dataset includes player per-game stats, player totals, player advanced stats, player shooting stats, and season/player information tables. We will inspect the exact CSV names after download and update the preprocessing code accordingly.
