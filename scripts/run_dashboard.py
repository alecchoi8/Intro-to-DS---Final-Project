from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = PROJECT_ROOT / "src" / "nba_scoring" / "dashboard.py"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
