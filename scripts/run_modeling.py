from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nba_scoring.modeling import generate_modeling_outputs


def main() -> None:
    outputs = generate_modeling_outputs()
    print(f"Modeling rows: {outputs['rows']}")
    print(f"Train rows: {outputs['train_rows']}")
    print(f"Test rows: {outputs['test_rows']}")
    print(
        "Best overall model: "
        f"{outputs['best_model_label']} ({outputs['best_model_key']}) "
        f"using {outputs['best_feature_set_label']}"
    )
    print(
        "Best no-direct-scoring model: "
        f"{outputs['best_no_direct_model_label']} ({outputs['best_no_direct_model_key']})"
    )
    print("Tables:")
    for path in outputs["table_paths"]:
        print(f"  {path}")
    print("Figures:")
    for path in outputs["figure_paths"]:
        print(f"  {path}")
    print("Models:")
    for path in outputs["model_paths"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
