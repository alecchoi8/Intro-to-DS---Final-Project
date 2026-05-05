from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nba_scoring.config import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from nba_scoring.preprocess import MODELING_DATASET_FILE, write_modeling_dataset


EDA_FIGURES_DIR = FIGURES_DIR / "eda"
EDA_REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"

CORE_FEATURES = [
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
TARGET = "pts_per_game"
HEATMAP_FEATURES = [
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
    TARGET,
]


def load_modeling_dataset(processed_data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    dataset_path = processed_data_dir / MODELING_DATASET_FILE
    if not dataset_path.exists():
        write_modeling_dataset(processed_data_dir=processed_data_dir)
    return pd.read_csv(dataset_path)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def build_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    overview = [
        ("player_seasons", len(df)),
        ("unique_players", df["player_id"].nunique()),
        ("season_min", int(df["season"].min())),
        ("season_max", int(df["season"].max())),
        ("season_count", df["season"].nunique()),
        ("traded_player_seasons", int(df["traded"].sum())),
        ("traded_player_season_share", df["traded"].mean()),
        ("missing_target_rows", int(df[TARGET].isna().sum())),
        ("mean_pts_per_game", df[TARGET].mean()),
        ("median_pts_per_game", df[TARGET].median()),
        ("max_pts_per_game", df[TARGET].max()),
    ]
    return pd.DataFrame(overview, columns=["metric", "value"])


def build_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = CORE_FEATURES + [TARGET]
    summary = df[columns].describe().T.reset_index(names="feature")
    missing = df[columns].isna().sum().rename("missing_count")
    missing_share = df[columns].isna().mean().rename("missing_share")
    return summary.merge(missing, left_on="feature", right_index=True).merge(
        missing_share, left_on="feature", right_index=True
    )


def build_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    columns = CORE_FEATURES + [TARGET]
    missing = pd.DataFrame(
        {
            "feature": columns,
            "missing_count": [int(df[column].isna().sum()) for column in columns],
            "missing_share": [float(df[column].isna().mean()) for column in columns],
        }
    )
    return missing.sort_values(["missing_share", "feature"], ascending=[False, True])


def build_correlations(df: pd.DataFrame) -> pd.DataFrame:
    columns = CORE_FEATURES + [TARGET]
    correlations = (
        df[columns]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .reset_index()
    )
    correlations.columns = ["feature", "correlation_with_pts_per_game"]
    correlations["absolute_correlation"] = correlations["correlation_with_pts_per_game"].abs()
    return correlations


def build_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    season_summary = (
        df.groupby("season")
        .agg(
            player_seasons=("player_id", "count"),
            unique_players=("player_id", "nunique"),
            mean_pts_per_game=(TARGET, "mean"),
            median_pts_per_game=(TARGET, "median"),
            p90_pts_per_game=(TARGET, lambda values: values.quantile(0.90)),
            max_pts_per_game=(TARGET, "max"),
            mean_minutes_per_game=("mp_per_game", "mean"),
            mean_fga_per_game=("fga_per_game", "mean"),
            traded_player_seasons=("traded", "sum"),
        )
        .reset_index()
    )
    season_summary["traded_player_season_share"] = (
        season_summary["traded_player_seasons"] / season_summary["player_seasons"]
    )
    return season_summary


def write_markdown_summary(
    report_dir: Path,
    overview: pd.DataFrame,
    correlations: pd.DataFrame,
    season_summary: pd.DataFrame,
    missing_values: pd.DataFrame,
) -> Path:
    integer_metrics = {
        "player_seasons",
        "unique_players",
        "season_min",
        "season_max",
        "season_count",
        "traded_player_seasons",
        "missing_target_rows",
    }
    top_corr = correlations.head(8)
    latest = season_summary.sort_values("season").tail(1).iloc[0]
    highest_scoring_season = season_summary.sort_values("mean_pts_per_game", ascending=False).iloc[0]
    missing_nonzero = missing_values[missing_values["missing_count"] > 0]

    lines = [
        "# EDA Summary",
        "",
        "Generated from `data/processed/nba_player_seasons_modeling.csv`.",
        "",
        "## Dataset Overview",
        "",
    ]
    for _, row in overview.iterrows():
        value = row["value"]
        if row["metric"] in integer_metrics:
            value = f"{int(value)}"
        elif isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"- `{row['metric']}`: {value}")

    lines.extend(
        [
            "",
            "## Strongest Linear Relationships With PPG",
            "",
            "| Feature | Correlation |",
            "| --- | ---: |",
        ]
    )
    for _, row in top_corr.iterrows():
        lines.append(f"| `{row['feature']}` | {row['correlation_with_pts_per_game']:.3f} |")

    lines.extend(
        [
            "",
            "## Season Notes",
            "",
            f"- Latest season in snapshot: `{int(latest['season'])}` with `{int(latest['player_seasons'])}` player-season rows.",
            f"- Highest average player PPG season: `{int(highest_scoring_season['season'])}` with mean PPG `{highest_scoring_season['mean_pts_per_game']:.2f}`.",
            "",
            "## Missingness Notes",
            "",
        ]
    )
    if missing_nonzero.empty:
        lines.append("- No missing values in the core EDA columns.")
    else:
        for _, row in missing_nonzero.head(10).iterrows():
            lines.append(
                f"- `{row['feature']}`: {int(row['missing_count'])} missing "
                f"({row['missing_share']:.1%})."
            )

    path = report_dir / "EDA_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_pts_distribution(df: pd.DataFrame, figure_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(df[TARGET], bins=50, color="#3568a6", edgecolor="white")
    ax.axvline(df[TARGET].median(), color="#c94f44", linestyle="--", linewidth=2, label="Median")
    ax.set_title("Distribution of NBA Player Points Per Game")
    ax.set_xlabel("Points per game")
    ax.set_ylabel("Player-seasons")
    ax.legend(frameon=False)
    return save_current_figure(figure_dir / "pts_per_game_distribution.png")


def plot_ppg_vs_minutes(df: pd.DataFrame, figure_dir: Path) -> Path:
    plot_df = df.dropna(subset=["mp_per_game", TARGET])
    fig, ax = plt.subplots(figsize=(8, 5))
    hexbin = ax.hexbin(
        plot_df["mp_per_game"],
        plot_df[TARGET],
        gridsize=40,
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(hexbin, ax=ax, label="Player-season count")
    ax.set_title("Points Per Game vs. Minutes Per Game")
    ax.set_xlabel("Minutes per game")
    ax.set_ylabel("Points per game")
    return save_current_figure(figure_dir / "ppg_vs_minutes_hexbin.png")


def plot_season_scoring_trend(season_summary: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        season_summary["season"],
        season_summary["mean_pts_per_game"],
        color="#3568a6",
        linewidth=2,
        label="Mean",
    )
    ax.plot(
        season_summary["season"],
        season_summary["p90_pts_per_game"],
        color="#7a9a3b",
        linewidth=2,
        label="90th percentile",
    )
    ax.set_title("NBA Player Scoring Trends by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points per game")
    ax.legend(frameon=False)
    return save_current_figure(figure_dir / "season_scoring_trend.png")


def plot_correlation_heatmap(df: pd.DataFrame, figure_dir: Path) -> Path:
    corr = df[HEATMAP_FEATURES].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title("Correlation Heatmap for Core Modeling Variables")
    return save_current_figure(figure_dir / "core_feature_correlation_heatmap.png")


def plot_missing_values(missing_values: pd.DataFrame, figure_dir: Path) -> Path:
    plot_df = missing_values[missing_values["missing_count"] > 0].sort_values("missing_share")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="missing_share", y="feature", color="#8e6a3a")
    ax.set_title("Missingness in Core EDA Variables")
    ax.set_xlabel("Missing share")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return save_current_figure(figure_dir / "core_feature_missingness.png")


def plot_top_correlations(correlations: pd.DataFrame, figure_dir: Path) -> Path:
    plot_df = correlations.head(10).sort_values("absolute_correlation")
    plt.figure(figsize=(8, 5))
    colors = ["#3568a6" if value >= 0 else "#c94f44" for value in plot_df["correlation_with_pts_per_game"]]
    ax = sns.barplot(
        data=plot_df,
        x="correlation_with_pts_per_game",
        y="feature",
        hue="feature",
        palette=colors,
        legend=False,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Core Feature Correlations With Points Per Game")
    ax.set_xlabel("Pearson correlation with PPG")
    ax.set_ylabel("")
    return save_current_figure(figure_dir / "top_ppg_correlations.png")


def generate_eda_outputs(
    figure_dir: Path = EDA_FIGURES_DIR,
    report_dir: Path = EDA_REPORTS_DIR,
) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_modeling_dataset()
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    overview = build_dataset_overview(df)
    feature_summary = build_feature_summary(df)
    missing_values = build_missing_values(df)
    correlations = build_correlations(df)
    season_summary = build_season_summary(df)

    table_paths = [
        write_csv(overview, report_dir / "dataset_overview.csv"),
        write_csv(feature_summary, report_dir / "feature_summary.csv"),
        write_csv(missing_values, report_dir / "missing_values.csv"),
        write_csv(correlations, report_dir / "correlations_with_pts_per_game.csv"),
        write_csv(season_summary, report_dir / "season_summary.csv"),
        write_markdown_summary(report_dir, overview, correlations, season_summary, missing_values),
    ]

    figure_paths = [
        plot_pts_distribution(df, figure_dir),
        plot_ppg_vs_minutes(df, figure_dir),
        plot_season_scoring_trend(season_summary, figure_dir),
        plot_correlation_heatmap(df, figure_dir),
        plot_missing_values(missing_values, figure_dir),
        plot_top_correlations(correlations, figure_dir),
    ]

    return {
        "rows": int(len(df)),
        "season_min": int(df["season"].min()),
        "season_max": int(df["season"].max()),
        "table_paths": table_paths,
        "figure_paths": figure_paths,
    }
