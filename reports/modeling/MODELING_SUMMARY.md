# Modeling Summary

Generated from `data/processed/nba_player_seasons_modeling.csv`.

## Modeling Setup

- Target: `pts_per_game`.
- Train/test split: `80/20` random split.
- Random state: `42`.
- Training rows: `20255`.
- Test rows: `5064`.

## Feature Set

### Full Feature Set

Includes playing time, direct scoring volume, shooting efficiency, and box-score context.

`age`, `g`, `mp_per_game`, `fga_per_game`, `fg_percent`, `x3pa_per_game`, `x3p_percent`, `fta_per_game`, `ft_percent`, `trb_per_game`, `ast_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`

### No Direct Scoring Components

Excludes shot attempts, free throw attempts, and shooting percentages.

`age`, `g`, `mp_per_game`, `trb_per_game`, `ast_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`

Direct scoring components removed in the restricted feature set:

`fga_per_game`, `fg_percent`, `x3pa_per_game`, `x3p_percent`, `fta_per_game`, `ft_percent`

## Test Performance

| Feature Set | Model | MAE | MSE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Full Feature Set | Hist Gradient Boosting Regressor | 0.130 | 0.089 | 0.299 | 0.998 |
| Full Feature Set | Random Forest Regressor | 0.218 | 0.153 | 0.391 | 0.996 |
| Full Feature Set | Linear Regression | 0.453 | 0.460 | 0.678 | 0.988 |
| Full Feature Set | Ridge Regression | 0.453 | 0.460 | 0.678 | 0.988 |
| Full Feature Set | Decision Tree Regressor | 0.593 | 0.737 | 0.858 | 0.981 |
| Full Feature Set | Dummy Mean Baseline | 4.997 | 39.003 | 6.245 | -0.000 |
| No Direct Scoring Components | Hist Gradient Boosting Regressor | 1.639 | 5.532 | 2.352 | 0.858 |
| No Direct Scoring Components | Random Forest Regressor | 1.646 | 5.613 | 2.369 | 0.856 |
| No Direct Scoring Components | Decision Tree Regressor | 1.770 | 6.527 | 2.555 | 0.833 |
| No Direct Scoring Components | Ridge Regression | 1.936 | 7.350 | 2.711 | 0.812 |
| No Direct Scoring Components | Linear Regression | 1.936 | 7.350 | 2.711 | 0.812 |
| No Direct Scoring Components | Dummy Mean Baseline | 4.997 | 39.003 | 6.245 | -0.000 |

## Best Models

- Best overall: `Hist Gradient Boosting Regressor` using `Full Feature Set` with MAE `0.130` PPG and R2 `0.998`.
- Best without direct scoring components: `Hist Gradient Boosting Regressor` with MAE `1.639` PPG and R2 `0.858`.

## Random Forest Interpretation

### Full Feature Set

- `fga_per_game`: importance `0.956`.
- `fg_percent`: importance `0.021`.
- `fta_per_game`: importance `0.015`.
- `x3pa_per_game`: importance `0.003`.
- `x3p_percent`: importance `0.001`.
- `ft_percent`: importance `0.001`.
- `tov_per_game`: importance `0.000`.
- `ast_per_game`: importance `0.000`.

### No Direct Scoring Components

- `mp_per_game`: importance `0.859`.
- `tov_per_game`: importance `0.039`.
- `ast_per_game`: importance `0.025`.
- `trb_per_game`: importance `0.021`.
- `g`: importance `0.016`.
- `stl_per_game`: importance `0.014`.
- `blk_per_game`: importance `0.014`.
- `age`: importance `0.011`.

## Paper Notes

- This is a supervised regression task because the target is continuous points per game.
- The full feature set measures an upper-bound version of the task because shot volume and efficiency are direct scoring components.
- The restricted feature set is the more conservative test because it removes shot attempts, free throw attempts, and shooting percentages.
- Linear Regression and Ridge provide interpretable linear baselines.
- Decision Tree and Random Forest capture nonlinear relationships.
- Histogram Gradient Boosting is the stronger optional scikit-learn comparison model.
- The paper should report both feature sets so the results are accurate without leaning on direct scoring reconstruction.
