"""Project-anchored paths.

Import these instead of writing relative paths, so scripts write into *this*
project's folders no matter where Python is launched from — and never leak output
into the workspace-root ``data/``.

    from marimo_openadmet.paths import DATA_RAW, RESULTS_METRICS
    df.to_json(RESULTS_METRICS / "results_run1.json")
"""

from pathlib import Path

# This file is at <PROJECT_ROOT>/src/marimo_openadmet/paths.py → go up 3 levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_METRICS = RESULTS_DIR / "metrics"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dirs() -> None:
    """Create the output dirs if missing (safe to call at the top of a script)."""
    for d in (
        DATA_INTERIM,
        DATA_PROCESSED,
        RESULTS_METRICS,
        RESULTS_FIGURES,
        RESULTS_TABLES,
        MODELS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
