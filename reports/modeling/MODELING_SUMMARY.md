# Modeling Summary

Generated from `data/processed/nba_player_seasons_modeling.csv`.

## Modeling Setup

- Target: `pts_per_game`.
- Train/test split: `80/20` random split.
- Random state: `42`.
- Training rows: `20255`.
- Test rows: `5064`.

## Feature Set

`age`, `g`, `mp_per_game`, `fga_per_game`, `fg_percent`, `x3pa_per_game`, `x3p_percent`, `fta_per_game`, `ft_percent`, `trb_per_game`, `ast_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`

## Test Performance

| Model | MAE | MSE | RMSE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Hist Gradient Boosting Regressor | 0.130 | 0.089 | 0.299 | 0.998 |
| Random Forest Regressor | 0.218 | 0.153 | 0.391 | 0.996 |
| Linear Regression | 0.453 | 0.460 | 0.678 | 0.988 |
| Ridge Regression | 0.453 | 0.460 | 0.678 | 0.988 |
| Decision Tree Regressor | 0.593 | 0.737 | 0.858 | 0.981 |
| Dummy Mean Baseline | 4.997 | 39.003 | 6.245 | -0.000 |

## Best Model

- Best test MAE: `Hist Gradient Boosting Regressor` with MAE `0.130` PPG and R2 `0.998`.

## Random Forest Interpretation

- `fga_per_game`: importance `0.956`.
- `fg_percent`: importance `0.021`.
- `fta_per_game`: importance `0.015`.
- `x3pa_per_game`: importance `0.003`.
- `x3p_percent`: importance `0.001`.
- `ft_percent`: importance `0.001`.
- `tov_per_game`: importance `0.000`.
- `ast_per_game`: importance `0.000`.

## Paper Notes

- This is a supervised regression task because the target is continuous points per game.
- Linear Regression and Ridge provide interpretable linear baselines.
- Decision Tree and Random Forest capture nonlinear relationships.
- Histogram Gradient Boosting is the stronger optional scikit-learn comparison model.
- Shot-volume features are highly predictive, so the paper should discuss near-mechanical scoring-feature relationships as a limitation.
