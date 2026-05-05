from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nba_scoring.eda import generate_eda_outputs


def main() -> None:
    outputs = generate_eda_outputs()
    print(f"EDA rows: {outputs['rows']}")
    print(f"Seasons: {outputs['season_min']}-{outputs['season_max']}")
    print("Tables:")
    for path in outputs["table_paths"]:
        print(f"  {path}")
    print("Figures:")
    for path in outputs["figure_paths"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
