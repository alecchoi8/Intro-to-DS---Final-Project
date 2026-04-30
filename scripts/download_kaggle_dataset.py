from __future__ import annotations

from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_HANDLE = "sumitrodatta/nba-aba-baa-stats"


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(RAW_DATA_DIR),
        force_download=True,
    )
    print(f"Downloaded {DATASET_HANDLE} to {path}")


if __name__ == "__main__":
    main()
