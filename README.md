# Predicting NBA Player Scoring Performance

Intro to Data Science final project for predicting NBA player points per game from historical player-season statistics.

## Team

- Alec Choi (`alc409`)
- Andre Sumilang (`aas504`)
- Ryan Shentu (`rs2281`)
- Section ID: `TBD`

## Research Question

Can an NBA player's points per game be predicted using statistical performance metrics such as playing time, shooting attempts, and shooting efficiency?

## Data Source

Primary dataset: [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) by Sumitro Datta on Kaggle.

Project data policy:

- Use the latest Kaggle dataset snapshot available on `2026-04-28`.
- Use NBA rows only, excluding BAA and ABA rows.
- Aggregate multi-team player seasons manually instead of relying blindly on team-level duplicate rows.
- Keep raw data out of Git. See [data/README.md](data/README.md) for setup instructions.

## Planned Models

Baseline and required models:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Additional comparison models may include:

- Ridge or Lasso Regression
- Gradient Boosting Regression
- Histogram Gradient Boosting Regression

## Evaluation

Models will be compared using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R2)

## Repository Structure

```text
data/              Data setup instructions and local-only raw/processed data
figures/           Generated plots selected for the final paper
notebooks/         EDA and modeling notebooks
paper/             NeurIPS-style LaTeX paper scaffold
reports/           Intermediate notes, tables, and presentation planning
src/               Reusable Python code for preprocessing, modeling, and evaluation
```

## Final Deliverables

- Public GitHub repository link included at the end of the paper abstract.
- Public video presentation link included at the end of the paper abstract.
- Final paper: 8 pages, unlimited references, using the NeurIPS LaTeX template.
- Video presentation: maximum 15 minutes, recorded with all group members appearing.
