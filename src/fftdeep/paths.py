#Project-wide paths, independent of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cifake"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"