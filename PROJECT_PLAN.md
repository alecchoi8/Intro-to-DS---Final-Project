# Project Plan

Current planning date: `2026-04-30`

## Confirmed Requirements

- Final paper must be 8 pages, with unlimited reference pages.
- Paper must use the NeurIPS LaTeX template.
- Paper sections should include Abstract, Related Work, Method, Evaluation, Discussion, and Limitations.
- Author block must include all student names, section ID, and NetIDs.
- Abstract must end with public Code and Video links.
- Code must be available through a public GitHub repository.
- Video presentation must be at most 15 minutes, recorded using Zoom or another cloud-hosted option.
- All group members must appear in the video.
- Dataset should use the latest Kaggle snapshot downloaded on `2026-04-30`.
- Analysis should use NBA rows only.
- Multi-team player seasons should be manually aggregated.
- Features based on shooting volume and efficiency are allowed.
- Stronger models beyond the outline are allowed.

## Phase 0: Repository Foundation

Goal: make the project reproducible from the first commit.

Tasks:

- Create project folder structure.
- Add environment requirements.
- Add data download instructions.
- Add NeurIPS-style paper scaffold.
- Configure GitHub remote and push to `main`.

Git checkpoint: `Initialize project scaffold`

## Phase 1: Data Acquisition

Goal: freeze the exact Kaggle dataset snapshot used for the project.

Tasks:

- Download the Kaggle dataset with KaggleHub or manually from the project dataset page.
- Store extracted files under `data/raw/`.
- Record download date, dataset handle, source URL, and file inventory.
- Inspect available files and identify the player per-game, player totals, and season info tables.

Git checkpoint: `Document data snapshot and schema`

## Phase 2: Data Cleaning and Aggregation

Goal: build a clean NBA player-season modeling table.

Tasks:

- Load raw CSV files. Completed in `scripts/build_modeling_dataset.py`.
- Filter to NBA rows only. Completed.
- Identify player-season rows split across multiple teams. Completed.
- Aggregate traded-player rows manually from team-specific totals. Completed.
- Recompute per-game features and shooting percentages from season totals. Completed.
- Handle era-specific three-point line availability. Completed.
- Save a processed modeling dataset under `data/processed/`. Completed locally; ignored by Git.

Git checkpoint: `Add NBA player-season preprocessing pipeline`

## Phase 3: Exploratory Data Analysis

Goal: understand the data and produce paper-ready visuals.

Tasks:

- Summarize dataset coverage by season. Completed in `reports/eda/season_summary.csv`.
- Plot points-per-game distribution. Completed.
- Plot relationships between PPG and minutes, attempts, efficiency, assists, rebounds, steals, and blocks. Completed through correlation tables, a minutes hexbin, and feature correlation plots.
- Build a correlation heatmap. Completed.
- Compare scoring trends across seasons. Completed.
- Save selected figures to `figures/eda/`. Completed.

Git checkpoint: `Add exploratory analysis figures`

## Phase 4: Feature Selection

Goal: define a defensible feature set for modeling.

Tasks:

- Separate target variable `PTS/G` from predictors.
- Decide which columns are identifiers, metadata, targets, or model features. Completed in `src/nba_scoring/modeling.py`.
- Include playing time, shooting volume, shooting efficiency, free throws, rebounds, assists, steals, and blocks. Completed.
- Compare full and no-direct-scoring feature groups. Implemented.

Git checkpoint: `Define modeling feature sets`

## Phase 5: Model Development

Goal: train required and optional regression models.

Tasks:

- Create train/test split. Implemented in `src/nba_scoring/modeling.py`.
- Build scikit-learn preprocessing and modeling pipelines. Implemented.
- Train Linear Regression, Decision Tree Regression, and Random Forest Regression. Implemented.
- Add at least one stronger comparison model if useful. Implemented with Ridge Regression and Histogram Gradient Boosting Regression.
- Train every model on both the full feature set and a restricted set without direct scoring components. Implemented.
- Tune tree-based model hyperparameters with cross-validation.

Git checkpoint: `Train baseline and tree regression models`

## Phase 6: Evaluation and Interpretation

Goal: compare models and explain what drives scoring predictions.

Tasks:

- Evaluate MAE, MSE, RMSE, and R2 on the test set. Implemented in `scripts/run_modeling.py`.
- Compare model performance in a table. Implemented.
- Plot actual vs. predicted PPG. Implemented.
- Analyze Random Forest feature importances. Implemented.
- Interpret Linear Regression coefficients carefully after preprocessing. Implemented through standardized coefficient output.
- Discuss limitations such as era changes, missing historical stats, and feature leakage concerns.

Git checkpoint: `Add model evaluation and interpretation`

## Phase 7: Final Paper and Presentation

Goal: package the final submission.

Tasks:

- Write the 8-page NeurIPS-style paper.
- Add public GitHub and video links at the end of the abstract.
- Prepare a 15-minute presentation outline.
- Record group video with all members visible.
- Verify all links are publicly accessible.

Git checkpoint: `Complete final paper and presentation materials`

## Open Items

- Section ID is still needed for the paper author block.
- Final public video link will be added after recording.
- Dataset version number should be recorded after manual Kaggle download.
