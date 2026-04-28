# Project Plan

Current planning date: `2026-04-28`

## Confirmed Requirements

- Final paper must be 8 pages, with unlimited reference pages.
- Paper must use the NeurIPS LaTeX template.
- Paper sections should include Abstract, Related Work, Method, Evaluation, Discussion, and Limitations.
- Author block must include all student names, section ID, and NetIDs.
- Abstract must end with public Code and Video links.
- Code must be available through a public GitHub repository.
- Video presentation must be at most 15 minutes, recorded using Zoom or another cloud-hosted option.
- All group members must appear in the video.
- Dataset should use the latest Kaggle snapshot available on `2026-04-28`.
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

- Manually download the Kaggle ZIP from the project dataset page.
- Store the ZIP or extracted files under `data/raw/`.
- Record download date, Kaggle dataset version if visible, and source URL.
- Inspect available files and identify the player per-game, player totals, and season info tables.

Git checkpoint: `Document data snapshot and schema`

## Phase 2: Data Cleaning and Aggregation

Goal: build a clean NBA player-season modeling table.

Tasks:

- Load raw CSV files.
- Filter to NBA rows only.
- Identify player-season rows split across multiple teams.
- Aggregate traded-player rows manually using totals where needed.
- Recompute per-game features from season totals when appropriate.
- Handle missing values and era-specific unavailable stats.
- Save a processed modeling dataset under `data/processed/`.

Git checkpoint: `Add NBA player-season preprocessing pipeline`

## Phase 3: Exploratory Data Analysis

Goal: understand the data and produce paper-ready visuals.

Tasks:

- Summarize dataset coverage by season.
- Plot points-per-game distribution.
- Plot relationships between PPG and minutes, attempts, efficiency, assists, rebounds, steals, and blocks.
- Build a correlation heatmap.
- Compare scoring trends across eras.
- Save selected figures to `figures/`.

Git checkpoint: `Add exploratory analysis figures`

## Phase 4: Feature Selection

Goal: define a defensible feature set for modeling.

Tasks:

- Separate target variable `PTS/G` from predictors.
- Decide which columns are identifiers, metadata, targets, or model features.
- Include playing time, shooting volume, shooting efficiency, free throws, rebounds, assists, steals, and blocks.
- Optionally compare all-feature, offensive-only, and no-direct-scoring feature groups.

Git checkpoint: `Define modeling feature sets`

## Phase 5: Model Development

Goal: train required and optional regression models.

Tasks:

- Create train/test split.
- Build scikit-learn preprocessing and modeling pipelines.
- Train Linear Regression, Decision Tree Regression, and Random Forest Regression.
- Add at least one stronger comparison model if useful.
- Tune tree-based model hyperparameters with cross-validation.

Git checkpoint: `Train baseline and tree regression models`

## Phase 6: Evaluation and Interpretation

Goal: compare models and explain what drives scoring predictions.

Tasks:

- Evaluate MAE, MSE, RMSE, and R2 on the test set.
- Compare model performance in a table.
- Plot actual vs. predicted PPG.
- Analyze Random Forest feature importances.
- Interpret Linear Regression coefficients carefully after preprocessing.
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
