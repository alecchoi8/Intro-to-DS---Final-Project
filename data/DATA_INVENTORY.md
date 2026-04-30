# Data Inventory

Snapshot downloaded on `2026-04-30` from Kaggle dataset handle `sumitrodatta/nba-aba-baa-stats`.

Raw CSV files are stored locally in `data/raw/` and ignored by Git. This inventory records the inspected schema so the project remains understandable without committing the raw dataset.

## File Inventory

| File | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `Advanced.csv` | 33,339 | 30 | Player advanced stats; NBA/ABA/BAA rows. |
| `All-Star Selections.csv` | 2,058 | 6 | All-Star metadata, likely not needed for base model. |
| `Draft Pick History.csv` | 8,383 | 8 | Draft metadata, optional context only. |
| `End of Season Teams (Voting).csv` | 4,484 | 14 | Award voting metadata, optional context only. |
| `End of Season Teams.csv` | 2,222 | 7 | Award/team metadata, optional context only. |
| `Opponent Stats Per 100 Poss.csv` | 1,462 | 28 | Team opponent stats, not needed for player PPG model. |
| `Opponent Stats Per Game.csv` | 1,907 | 28 | Team opponent stats, not needed for player PPG model. |
| `Opponent Totals.csv` | 1,907 | 28 | Team opponent totals, not needed for player PPG model. |
| `Per 100 Poss.csv` | 27,692 | 34 | Player rate table; optional engineered features. |
| `Per 36 Minutes.csv` | 32,256 | 32 | Player rate table; optional engineered features. |
| `Player Award Shares.csv` | 3,465 | 10 | Award shares, optional context only. |
| `Player Career Info.csv` | 5,416 | 11 | Player biographical/career metadata. |
| `Player Per Game.csv` | 33,339 | 32 | Player per-game table; contains target `pts_per_game`. |
| `Player Play By Play.csv` | 18,254 | 26 | Modern play-by-play-derived features, 1997 onward. |
| `Player Season Info.csv` | 33,339 | 8 | Player-season metadata including `experience`. |
| `Player Shooting.csv` | 18,254 | 32 | Modern shooting location stats, NBA-only, 1997 onward. |
| `Player Totals.csv` | 33,339 | 33 | Best base table for manual aggregation and per-game recomputation. |
| `Team Abbrev.csv` | 1,818 | 5 | Team abbreviation metadata. |
| `Team Stats Per 100 Poss.csv` | 1,462 | 28 | Team table, not needed for base player model. |
| `Team Stats Per Game.csv` | 1,907 | 28 | Team table, not needed for base player model. |
| `Team Summaries.csv` | 1,907 | 31 | Team summary table, optional context only. |
| `Team Totals.csv` | 1,907 | 28 | Team table, not needed for base player model. |

## Key Player Tables

### `Player Totals.csv`

Recommended base table for preprocessing because counting stats can be summed across teams for traded players before recomputing per-game metrics.

Columns:

`season`, `lg`, `player`, `player_id`, `age`, `team`, `pos`, `g`, `gs`, `mp`, `fg`, `fga`, `fg_percent`, `x3p`, `x3pa`, `x3p_percent`, `x2p`, `x2pa`, `x2p_percent`, `e_fg_percent`, `ft`, `fta`, `ft_percent`, `orb`, `drb`, `trb`, `ast`, `stl`, `blk`, `tov`, `pf`, `pts`, `trp_dbl`

### `Player Per Game.csv`

Useful for validating recomputed per-game metrics and as the direct source of the target column name.

Target:

`pts_per_game`

Important feature columns:

`mp_per_game`, `fga_per_game`, `fg_percent`, `x3pa_per_game`, `x3p_percent`, `fta_per_game`, `ft_percent`, `trb_per_game`, `ast_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`

### `Player Season Info.csv`

Useful metadata table for `experience`, age, position, and team labels.

### `Advanced.csv`

Potential optional features such as `per`, `ts_percent`, `usg_percent`, `ws`, `bpm`, and `vorp`.

### `Player Shooting.csv`

Modern-only shooting-location table. It covers NBA seasons `1997` through `2026`, so using it would either restrict the model to the modern era or require a separate modern-era model.

## NBA Scope Summary

- `Player Per Game.csv` NBA rows: `31,119`
- NBA seasons covered: `1950` through `2026`
- Unique NBA player-seasons after one-row-per-player-season aggregation: `25,319`
- Multi-row NBA player-seasons caused by trades/team splits: `2,805`
- Existing total-row labels for traded players: `2TM`, `3TM`, `4TM`, `5TM`

The dataset already includes multi-team total rows, but preprocessing should still validate or recompute totals from `Player Totals.csv` so the aggregation method is transparent.

## Missingness Notes

Missing values in key NBA `Player Per Game.csv` columns:

| Column | Missing Rows |
| --- | ---: |
| `g` | 0 |
| `gs` | 6,039 |
| `mp_per_game` | 501 |
| `fga_per_game` | 0 |
| `fg_percent` | 147 |
| `x3pa_per_game` | 5,770 |
| `x3p_percent` | 9,701 |
| `fta_per_game` | 0 |
| `ft_percent` | 1,289 |
| `trb_per_game` | 312 |
| `ast_per_game` | 0 |
| `stl_per_game` | 3,900 |
| `blk_per_game` | 3,900 |
| `tov_per_game` | 5,052 |
| `pts_per_game` | 0 |

Interpretation notes:

- Three-point columns are unavailable before the NBA adopted the three-point line in the 1979-80 season.
- Steals, blocks, turnovers, and games started have historical availability gaps.
- Percentage columns can be missing when a player has zero attempts.
- The preprocessing pipeline should use imputation and era-aware feature handling rather than dropping all rows with historical missing values.
