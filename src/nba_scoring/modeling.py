from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from nba_scoring.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from nba_scoring.preprocess import MODELING_DATASET_FILE, write_modeling_dataset


TARGET = "pts_per_game"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODELING_FIGURES_DIR = FIGURES_DIR / "modeling"
MODELING_REPORTS_DIR = PROJECT_ROOT / "reports" / "modeling"

FEATURE_COLUMNS = [
    "age",
    "g",
    "mp_per_game",
    "fga_per_game",
    "fg_percent",
    "x3pa_per_game",
    "x3p_percent",
    "fta_per_game",
    "ft_percent",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
]
IDENTIFIER_COLUMNS = ["season", "player_id", "player"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    estimator: Any
    scale_features: bool


def load_modeling_dataset(processed_data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    dataset_path = processed_data_dir / MODELING_DATASET_FILE
    if not dataset_path.exists():
        write_modeling_dataset(processed_data_dir=processed_data_dir)
    return pd.read_csv(dataset_path)


def validate_modeling_columns(df: pd.DataFrame) -> None:
    required_columns = FEATURE_COLUMNS + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required modeling columns: {missing_columns}")


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            key="dummy_mean",
            label="Dummy Mean Baseline",
            estimator=DummyRegressor(strategy="mean"),
            scale_features=False,
        ),
        ModelSpec(
            key="linear_regression",
            label="Linear Regression",
            estimator=LinearRegression(),
            scale_features=True,
        ),
        ModelSpec(
            key="ridge_regression",
            label="Ridge Regression",
            estimator=Ridge(alpha=1.0),
            scale_features=True,
        ),
        ModelSpec(
            key="decision_tree",
            label="Decision Tree Regressor",
            estimator=DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=25,
                random_state=RANDOM_STATE,
            ),
            scale_features=False,
        ),
        ModelSpec(
            key="random_forest",
            label="Random Forest Regressor",
            estimator=RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_features=False,
        ),
        ModelSpec(
            key="hist_gradient_boosting",
            label="Hist Gradient Boosting Regressor",
            estimator=HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
            scale_features=False,
        ),
    ]


def build_preprocessor(scale_features: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec.scale_features)),
            ("model", spec.estimator),
        ]
    )


def split_modeling_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_modeling_columns(df)
    model_df = df.dropna(subset=[TARGET]).copy()
    return train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def evaluate_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clean_feature_names(names: Any) -> list[str]:
    return [str(name).replace("numeric__", "") for name in names]


def extract_feature_effects(
    fitted_models: dict[str, Pipeline],
    specs_by_key: dict[str, ModelSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_key, pipeline in fitted_models.items():
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = clean_feature_names(preprocessor.get_feature_names_out())
        label = specs_by_key[model_key].label

        if hasattr(model, "coef_"):
            for feature, coefficient in zip(feature_names, model.coef_):
                rows.append(
                    {
                        "model_key": model_key,
                        "model": label,
                        "feature": feature,
                        "effect_type": "standardized_coefficient",
                        "value": float(coefficient),
                        "rank_value": float(abs(coefficient)),
                    }
                )
        elif hasattr(model, "feature_importances_"):
            for feature, importance in zip(feature_names, model.feature_importances_):
                rows.append(
                    {
                        "model_key": model_key,
                        "model": label,
                        "feature": feature,
                        "effect_type": "feature_importance",
                        "value": float(importance),
                        "rank_value": float(importance),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["model_key", "model", "feature", "effect_type", "value", "rank_value"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "rank_value"], ascending=[True, False])
        .reset_index(drop=True)
    )


def train_and_evaluate_models(df: pd.DataFrame) -> dict[str, Any]:
    train_df, test_df = split_modeling_data(df)
    model_specs = build_model_specs()
    specs_by_key = {spec.key: spec for spec in model_specs}

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}

    identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in test_df.columns]
    prediction_base = test_df[identifier_columns + [TARGET]].reset_index(drop=True)

    for spec in model_specs:
        pipeline = build_pipeline(spec)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        fitted_models[spec.key] = pipeline
        metrics = evaluate_predictions(y_test, y_pred)
        metrics_rows.append(
            {
                "model_key": spec.key,
                "model": spec.label,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                **metrics,
            }
        )

        prediction_frame = prediction_base.copy()
        prediction_frame["model_key"] = spec.key
        prediction_frame["model"] = spec.label
        prediction_frame["actual_pts_per_game"] = prediction_frame[TARGET]
        prediction_frame["predicted_pts_per_game"] = y_pred
        prediction_frame["residual"] = (
            prediction_frame["actual_pts_per_game"] - prediction_frame["predicted_pts_per_game"]
        )
        prediction_frame = prediction_frame.drop(columns=[TARGET])
        prediction_frames.append(prediction_frame)

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["mae", "rmse"], ascending=[True, True])
        .reset_index(drop=True)
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    feature_effects_df = extract_feature_effects(fitted_models, specs_by_key)
    best_model_key = str(metrics_df.iloc[0]["model_key"])

    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "feature_effects": feature_effects_df,
        "fitted_models": fitted_models,
        "best_model_key": best_model_key,
        "best_model_label": specs_by_key[best_model_key].label,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_model_performance(metrics: pd.DataFrame, figure_dir: Path) -> Path:
    plot_df = metrics.sort_values("mae", ascending=False)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_df, x="mae", y="model", color="#3568a6")
    ax.set_title("Model Test Error: Mean Absolute Error")
    ax.set_xlabel("MAE (points per game)")
    ax.set_ylabel("")
    return save_current_figure(figure_dir / "model_performance_mae.png")


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    best_model_key: str,
    best_model_label: str,
    figure_dir: Path,
) -> Path:
    plot_df = predictions[predictions["model_key"].eq(best_model_key)]
    lower = min(plot_df["actual_pts_per_game"].min(), plot_df["predicted_pts_per_game"].min())
    upper = max(plot_df["actual_pts_per_game"].max(), plot_df["predicted_pts_per_game"].max())
    margin = (upper - lower) * 0.05

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="actual_pts_per_game",
        y="predicted_pts_per_game",
        alpha=0.35,
        s=18,
        edgecolor=None,
        color="#3568a6",
    )
    ax.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="#c94f44")
    ax.set_title(f"Actual vs. Predicted PPG: {best_model_label}")
    ax.set_xlabel("Actual points per game")
    ax.set_ylabel("Predicted points per game")
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    return save_current_figure(figure_dir / "actual_vs_predicted_best_model.png")


def plot_random_forest_feature_importance(feature_effects: pd.DataFrame, figure_dir: Path) -> Path | None:
    plot_df = feature_effects[
        feature_effects["model_key"].eq("random_forest")
        & feature_effects["effect_type"].eq("feature_importance")
    ].nlargest(10, "rank_value")
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values("value")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="value", y="feature", color="#7a9a3b")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return save_current_figure(figure_dir / "random_forest_feature_importance.png")


def write_markdown_summary(
    metrics: pd.DataFrame,
    feature_effects: pd.DataFrame,
    results: dict[str, Any],
    report_dir: Path,
) -> Path:
    best = metrics.iloc[0]
    random_forest_top = feature_effects[
        feature_effects["model_key"].eq("random_forest")
        & feature_effects["effect_type"].eq("feature_importance")
    ].head(8)

    lines = [
        "# Modeling Summary",
        "",
        "Generated from `data/processed/nba_player_seasons_modeling.csv`.",
        "",
        "## Modeling Setup",
        "",
        f"- Target: `{TARGET}`.",
        f"- Train/test split: `{int((1 - results['test_size']) * 100)}/{int(results['test_size'] * 100)}` random split.",
        f"- Random state: `{results['random_state']}`.",
        f"- Training rows: `{results['train_rows']}`.",
        f"- Test rows: `{results['test_rows']}`.",
        "",
        "## Feature Set",
        "",
        ", ".join(f"`{feature}`" for feature in FEATURE_COLUMNS),
        "",
        "## Test Performance",
        "",
        "| Model | MAE | MSE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['mae']:.3f} | {row['mse']:.3f} | "
            f"{row['rmse']:.3f} | {row['r2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- Best test MAE: `{best['model']}` with MAE `{best['mae']:.3f}` PPG and R2 `{best['r2']:.3f}`.",
            "",
            "## Random Forest Interpretation",
            "",
        ]
    )

    if random_forest_top.empty:
        lines.append("- Random Forest feature importances were not available.")
    else:
        for _, row in random_forest_top.iterrows():
            lines.append(f"- `{row['feature']}`: importance `{row['value']:.3f}`.")

    lines.extend(
        [
            "",
            "## Paper Notes",
            "",
            "- This is a supervised regression task because the target is continuous points per game.",
            "- Linear Regression and Ridge provide interpretable linear baselines.",
            "- Decision Tree and Random Forest capture nonlinear relationships.",
            "- Histogram Gradient Boosting is the stronger optional scikit-learn comparison model.",
            "- Shot-volume features are highly predictive, so the paper should discuss near-mechanical scoring-feature relationships as a limitation.",
        ]
    )

    path = report_dir / "MODELING_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_best_model(results: dict[str, Any], model_dir: Path = MODELS_DIR) -> list[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_key = results["best_model_key"]
    best_model_path = model_dir / "best_scoring_model.pkl"
    metadata_path = model_dir / "best_scoring_model_metadata.json"

    with best_model_path.open("wb") as file:
        pickle.dump(results["fitted_models"][best_model_key], file)

    metadata = {
        "best_model_key": best_model_key,
        "best_model_label": results["best_model_label"],
        "target": results["target"],
        "feature_columns": results["feature_columns"],
        "test_size": results["test_size"],
        "random_state": results["random_state"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [best_model_path, metadata_path]


def generate_modeling_outputs(
    figure_dir: Path = MODELING_FIGURES_DIR,
    report_dir: Path = MODELING_REPORTS_DIR,
    model_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_modeling_dataset()
    results = train_and_evaluate_models(df)

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table_paths = [
        write_csv(results["metrics"], report_dir / "model_performance.csv"),
        write_csv(results["predictions"], report_dir / "test_predictions.csv"),
        write_csv(results["feature_effects"], report_dir / "feature_effects.csv"),
        write_markdown_summary(
            results["metrics"],
            results["feature_effects"],
            results,
            report_dir,
        ),
    ]

    figure_paths: list[Path] = [
        plot_model_performance(results["metrics"], figure_dir),
        plot_actual_vs_predicted(
            results["predictions"],
            results["best_model_key"],
            results["best_model_label"],
            figure_dir,
        ),
    ]
    random_forest_importance_path = plot_random_forest_feature_importance(
        results["feature_effects"],
        figure_dir,
    )
    if random_forest_importance_path is not None:
        figure_paths.append(random_forest_importance_path)

    model_paths = write_best_model(results, model_dir=model_dir)

    return {
        "rows": int(len(df)),
        "train_rows": results["train_rows"],
        "test_rows": results["test_rows"],
        "best_model_key": results["best_model_key"],
        "best_model_label": results["best_model_label"],
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "model_paths": model_paths,
    }
