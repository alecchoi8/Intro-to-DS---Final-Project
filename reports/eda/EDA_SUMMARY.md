# EDA Summary

Generated from `data/processed/nba_player_seasons_modeling.csv`.

## Dataset Overview

- `player_seasons`: 25319
- `unique_players`: 4915
- `season_min`: 1950
- `season_max`: 2026
- `season_count`: 77
- `traded_player_seasons`: 2805
- `traded_player_season_share`: 0.1108
- `missing_target_rows`: 0
- `mean_pts_per_game`: 8.7048
- `median_pts_per_game`: 7.1806
- `max_pts_per_game`: 50.3625

## Strongest Linear Relationships With PPG

| Feature | Correlation |
| --- | ---: |
| `fga_per_game` | 0.979 |
| `mp_per_game` | 0.890 |
| `fta_per_game` | 0.879 |
| `tov_per_game` | 0.836 |
| `stl_per_game` | 0.667 |
| `ast_per_game` | 0.638 |
| `trb_per_game` | 0.620 |
| `g` | 0.541 |

## Season Notes

- Latest season in snapshot: `2026` with `582` player-season rows.
- Highest average player PPG season: `1961` with mean PPG `11.15`.

## Missingness Notes

- `x3p_percent`: 7627 missing (30.1%).
- `tov_per_game`: 4204 missing (16.6%).
- `blk_per_game`: 3214 missing (12.7%).
- `stl_per_game`: 3214 missing (12.7%).
- `ft_percent`: 766 missing (3.0%).
- `mp_per_game`: 358 missing (1.4%).
- `trb_per_game`: 223 missing (0.9%).
- `fg_percent`: 80 missing (0.3%).
