"""Where the downloaded puzzle files and generated outputs live."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NNRE_DATA", ROOT / "data"))
BUILD = Path(os.environ.get("NNRE_BUILD", ROOT / "build"))

MODEL_PT = DATA / "hashnet" / "model.pt"
DROPPED_ZIP = DATA / "dropped" / "historical_data_and_pieces.zip"
DROPPED_DIR = DATA / "dropped"
PIECES_DIR = DROPPED_DIR / "pieces"
HISTORICAL_CSV = DROPPED_DIR / "historical_data.csv"


def require(path: Path, hint: str = "run `make fetch` (or `nnre fetch`) to download the puzzle files") -> Path:
    if not path.exists():
        raise SystemExit(f"missing {path}\n  {hint}")
    return path
