from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nba_scoring.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from nba_scoring.eda import EDA_FIGURES_DIR, EDA_REPORTS_DIR, generate_eda_outputs
from nba_scoring.modeling import MODELING_FIGURES_DIR, MODELING_REPORTS_DIR, generate_modeling_outputs
from nba_scoring.preprocess import MODELING_DATASET_FILE, PLAYER_TOTALS_FILE, write_modeling_dataset


RAW_TOTALS_PATH = RAW_DATA_DIR / PLAYER_TOTALS_FILE
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / MODELING_DATASET_FILE
MODELING_SUMMARY_PATH = MODELING_REPORTS_DIR / "MODELING_SUMMARY.md"
MODEL_PERFORMANCE_PATH = MODELING_REPORTS_DIR / "model_performance.csv"
FEATURE_EFFECTS_PATH = MODELING_REPORTS_DIR / "feature_effects.csv"
TEST_PREDICTIONS_PATH = MODELING_REPORTS_DIR / "test_predictions.csv"
ALL_SEASON_PREDICTIONS_PATH = MODELING_REPORTS_DIR / "all_season_predictions.csv"


def file_status(path: Path) -> str:
    return "✅ Found" if path.exists() else "⚠️ Missing"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def run_with_spinner(label: str, action: Callable[[], object]) -> None:
    with st.spinner(label):
        result = action()
    st.success("Done.")
    if isinstance(result, tuple):
        st.write(result)
    elif isinstance(result, dict):
        st.write(
            {
                key: value
                for key, value in result.items()
                if key in {"rows", "train_rows", "test_rows", "best_model_label", "best_no_direct_model_label"}
            }
        )


@st.cache_data(show_spinner=False)
def load_processed_preview(path: str) -> pd.DataFrame | None:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return None
    return pd.read_csv(dataset_path)


@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame | None:
    return read_csv(Path(path))


def render_sidebar() -> None:
    st.sidebar.title("NBA Scoring UI")
    st.sidebar.caption("Run the project pipeline and inspect outputs.")

    st.sidebar.subheader("Data Status")
    st.sidebar.write(f"Raw totals: {file_status(RAW_TOTALS_PATH)}")
    st.sidebar.write(f"Processed dataset: {file_status(PROCESSED_DATASET_PATH)}")
    st.sidebar.write(f"Modeling summary: {file_status(MODELING_SUMMARY_PATH)}")

    st.sidebar.subheader("Run Pipeline")
    if st.sidebar.button("Build Processed Dataset"):
        run_with_spinner("Building processed modeling dataset...", write_modeling_dataset)
        st.cache_data.clear()

    if st.sidebar.button("Generate EDA Outputs"):
        run_with_spinner("Generating EDA outputs...", generate_eda_outputs)
        st.cache_data.clear()

    if st.sidebar.button("Train + Evaluate Models"):
        run_with_spinner("Training scikit-learn models...", generate_modeling_outputs)
        st.cache_data.clear()

    st.sidebar.subheader("Key Paths")
    st.sidebar.code(str(PROJECT_ROOT))
    st.sidebar.code(str(MODELING_REPORTS_DIR))
    st.sidebar.code(str(MODELING_FIGURES_DIR))


def render_project_overview() -> None:
    st.header("Project Overview")
    st.write(
        "This dashboard wraps the NBA player scoring project so you can rebuild data, "
        "run EDA/modeling, and inspect the findings from one place."
    )

    cols = st.columns(4)
    dataset = load_processed_preview(str(PROCESSED_DATASET_PATH))
    if dataset is None:
        cols[0].metric("Rows", "Missing")
        cols[1].metric("Seasons", "Missing")
        cols[2].metric("Players", "Missing")
        cols[3].metric("Mean PPG", "Missing")
        st.warning("Build the processed dataset first, or make sure raw Kaggle CSVs are in data/raw/.")
        return

    cols[0].metric("Player-Seasons", f"{len(dataset):,}")
    cols[1].metric("Seasons", f"{int(dataset['season'].min())}-{int(dataset['season'].max())}")
    cols[2].metric("Players", f"{dataset['player_id'].nunique():,}")
    cols[3].metric("Mean PPG", f"{dataset['pts_per_game'].mean():.2f}")

    st.subheader("Dataset Preview")
    preview_columns = [
        "season",
        "player",
        "age",
        "team",
        "mp_per_game",
        "fga_per_game",
        "pts_per_game",
    ]
    st.dataframe(dataset[preview_columns].head(25), use_container_width=True)


def render_modeling() -> None:
    st.header("Modeling Results")
    metrics = load_csv_cached(str(MODEL_PERFORMANCE_PATH))
    if metrics is None:
        st.warning("No modeling outputs found yet. Use the sidebar button: Train + Evaluate Models.")
        return

    st.subheader("Performance Table")
    display_metrics = metrics[
        ["feature_set", "model", "mae", "mse", "rmse", "r2", "feature_count"]
    ].copy()
    st.dataframe(
        display_metrics.style.format(
            {
                "mae": "{:.3f}",
                "mse": "{:.3f}",
                "rmse": "{:.3f}",
                "r2": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    best_full = metrics[metrics["feature_set_key"].eq("full")].sort_values("mae").iloc[0]
    best_restricted = metrics[metrics["feature_set_key"].eq("no_direct_scoring")].sort_values("mae").iloc[0]
    cols = st.columns(2)
    cols[0].metric(
        "Best Full Model",
        best_full["model"],
        f"MAE {best_full['mae']:.3f}, R² {best_full['r2']:.3f}",
    )
    cols[1].metric(
        "Best Restricted Model",
        best_restricted["model"],
        f"MAE {best_restricted['mae']:.3f}, R² {best_restricted['r2']:.3f}",
    )

    st.subheader("Model Error Charts")
    chart_cols = st.columns(2)
    full_chart = MODELING_FIGURES_DIR / "model_performance_mae.png"
    no_dummy_chart = MODELING_FIGURES_DIR / "model_performance_mae_without_dummy.png"
    if full_chart.exists():
        chart_cols[0].image(str(full_chart), caption="All models, including dummy baseline")
    if no_dummy_chart.exists():
        chart_cols[1].image(str(no_dummy_chart), caption="Comparison without dummy baseline")

    st.subheader("Actual vs. Predicted")
    prediction_cols = st.columns(2)
    full_prediction = MODELING_FIGURES_DIR / "actual_vs_predicted_best_model.png"
    restricted_prediction = MODELING_FIGURES_DIR / "actual_vs_predicted_no_direct_scoring_best_model.png"
    if full_prediction.exists():
        prediction_cols[0].image(str(full_prediction), caption="Best full-feature model")
    if restricted_prediction.exists():
        prediction_cols[1].image(str(restricted_prediction), caption="Best restricted-feature model")

    summary = read_text(MODELING_SUMMARY_PATH)
    if summary:
        with st.expander("Read Modeling Summary"):
            st.markdown(summary)


def render_feature_importance() -> None:
    st.header("Feature Importance")
    feature_effects = load_csv_cached(str(FEATURE_EFFECTS_PATH))
    if feature_effects is None:
        st.warning("No feature effect output found yet. Train models first.")
        return

    cols = st.columns(2)
    full_importance = MODELING_FIGURES_DIR / "random_forest_feature_importance_full.png"
    restricted_importance = MODELING_FIGURES_DIR / "random_forest_feature_importance_no_direct_scoring.png"
    if full_importance.exists():
        cols[0].image(str(full_importance), caption="Full feature set")
    if restricted_importance.exists():
        cols[1].image(str(restricted_importance), caption="No direct scoring components")

    st.subheader("Feature Effects Table")
    feature_sets = ["All"] + sorted(feature_effects["feature_set"].dropna().unique().tolist())
    selected_feature_set = st.selectbox("Feature set", feature_sets)
    table = feature_effects.copy()
    if selected_feature_set != "All":
        table = table[table["feature_set"].eq(selected_feature_set)]
    st.dataframe(table, use_container_width=True)


def render_eda() -> None:
    st.header("Exploratory Data Analysis")
    overview = load_csv_cached(str(EDA_REPORTS_DIR / "dataset_overview.csv"))
    if overview is None:
        st.warning("No EDA outputs found yet. Use the sidebar button: Generate EDA Outputs.")
        return

    st.subheader("Dataset Overview")
    st.dataframe(overview, use_container_width=True)

    st.subheader("EDA Figures")
    figure_paths = [
        EDA_FIGURES_DIR / "pts_per_game_distribution.png",
        EDA_FIGURES_DIR / "ppg_vs_minutes_hexbin.png",
        EDA_FIGURES_DIR / "season_scoring_trend.png",
        EDA_FIGURES_DIR / "core_feature_correlation_heatmap.png",
        EDA_FIGURES_DIR / "top_ppg_correlations.png",
        EDA_FIGURES_DIR / "core_feature_missingness.png",
    ]
    for first, second in zip(figure_paths[0::2], figure_paths[1::2]):
        cols = st.columns(2)
        if first.exists():
            cols[0].image(str(first), caption=first.stem.replace("_", " ").title())
        if second.exists():
            cols[1].image(str(second), caption=second.stem.replace("_", " ").title())

    correlations = load_csv_cached(str(EDA_REPORTS_DIR / "correlations_with_pts_per_game.csv"))
    if correlations is not None:
        st.subheader("Top Correlations With PPG")
        st.dataframe(correlations.head(12), use_container_width=True)


def render_predictions() -> None:
    st.header("Prediction Explorer")
    dataset = load_processed_preview(str(PROCESSED_DATASET_PATH))
    all_predictions = load_csv_cached(str(ALL_SEASON_PREDICTIONS_PATH))
    test_predictions = load_csv_cached(str(TEST_PREDICTIONS_PATH))

    if dataset is None:
        st.warning("No processed dataset found yet. Build the processed dataset first.")
        return

    if all_predictions is None and test_predictions is None:
        st.warning("No prediction output found yet. Train models to view predictions.")
        return

    prediction_source = all_predictions if all_predictions is not None else test_predictions
    model_options = sorted(prediction_source["fitted_model_key"].unique().tolist())
    selected_model = st.selectbox("Model output", model_options)

    st.info(
        "All-season predictions include both training and held-out test rows. "
        "Use the `split` column to tell whether a prediction was in-sample (`train`) "
        "or held-out (`test`)."
    )

    player_query = st.text_input("Search player")
    if player_query:
        player_matches = dataset[dataset["player"].str.contains(player_query, case=False, na=False)].copy()
        if player_matches.empty:
            st.warning("No matching players found.")
        else:
            career_columns = [
                "season",
                "player_id",
                "player",
                "age",
                "pos",
                "team",
                "teams",
                "g",
                "mp_per_game",
                "fga_per_game",
                "fg_percent",
                "fta_per_game",
                "trb_per_game",
                "ast_per_game",
                "pts_per_game",
            ]
            career_columns = [column for column in career_columns if column in player_matches.columns]
            st.subheader("All Matching Player Seasons")
            st.dataframe(
                player_matches.sort_values(["player", "season"], ascending=[True, False])[
                    career_columns
                ],
                use_container_width=True,
            )

        if all_predictions is None:
            st.warning(
                "This dashboard was generated before all-season predictions existed. "
                "Click `Train + Evaluate Models` in the sidebar to create them."
            )
        else:
            all_model_predictions = all_predictions[
                all_predictions["fitted_model_key"].eq(selected_model)
            ].copy()
            prediction_matches = all_model_predictions[
                all_model_predictions["player"].str.contains(player_query, case=False, na=False)
            ].copy()
            st.subheader("All-Season Predictions for Matching Players")
            if prediction_matches.empty:
                st.write("No matching predictions for this model.")
            else:
                prediction_columns = [
                    "season",
                    "player_id",
                    "player",
                    "split",
                    "feature_set",
                    "model",
                    "actual_pts_per_game",
                    "predicted_pts_per_game",
                    "residual",
                    "absolute_error",
                ]
                st.dataframe(
                    prediction_matches.sort_values(["player", "season"], ascending=[True, False])[
                        prediction_columns
                    ],
                    use_container_width=True,
                )

    if test_predictions is not None:
        filtered_test = test_predictions[test_predictions["fitted_model_key"].eq(selected_model)].copy()
        filtered_test["absolute_error"] = filtered_test["residual"].abs()
        st.subheader("Largest Held-Out Test Misses")
        st.dataframe(
            filtered_test.sort_values("absolute_error", ascending=False).head(25),
            use_container_width=True,
        )


def render_help() -> None:
    st.header("How To Run")
    st.write("From the project root:")
    st.code(
        "\n".join(
            [
                "python3 -m pip install streamlit",
                "python3 scripts/run_dashboard.py",
            ]
        ),
        language="zsh",
    )
    st.write("Equivalent direct Streamlit command:")
    st.code("python3 -m streamlit run src/nba_scoring/dashboard.py", language="zsh")

    st.subheader("Pipeline Commands")
    st.code(
        "\n".join(
            [
                "python3 scripts/build_modeling_dataset.py",
                "python3 scripts/run_eda.py",
                "python3 scripts/run_modeling.py",
            ]
        ),
        language="zsh",
    )


def main() -> None:
    st.set_page_config(page_title="NBA Scoring Project", layout="wide")
    render_sidebar()
    st.title("Predicting NBA Player Scoring Performance")

    tabs = st.tabs(
        [
            "Overview",
            "Modeling",
            "Feature Importance",
            "EDA",
            "Prediction Explorer",
            "Help",
        ]
    )
    with tabs[0]:
        render_project_overview()
    with tabs[1]:
        render_modeling()
    with tabs[2]:
        render_feature_importance()
    with tabs[3]:
        render_eda()
    with tabs[4]:
        render_predictions()
    with tabs[5]:
        render_help()


if __name__ == "__main__":
    main()
