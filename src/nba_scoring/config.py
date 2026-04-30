from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

SNAPSHOT_DATE = "2026-04-30"
KAGGLE_DATASET_HANDLE = "sumitrodatta/nba-aba-baa-stats"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats"
