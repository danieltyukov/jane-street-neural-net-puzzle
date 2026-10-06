"""Download the official puzzle files from Hugging Face, pinned to a revision and checked by SHA-256.

Nothing here is redistributed: the files stay on Hugging Face and are fetched into ./data.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Remote:
    url: str
    sha256: str
    size: int
    dest: Path


# Puzzle 1: https://huggingface.co/spaces/jane-street/puzzle (model repo jane-street/2025-03-10).
# model_3_11.pt in the same repo has byte-identical weights; only the pickled wrapper's bytecode
# differs (CPython 3.11 instead of 3.10), and this project never runs that wrapper.
MODEL = Remote(
    url="https://huggingface.co/jane-street/2025-03-10/resolve/1a904e632c4224100e840445b0f00aa91e697edb/model.pt",
    sha256="1ff10e21b54431a0959d8d6827d670fe122490c041ac234627fd37f44d825913",
    size=1_158_692_162,
    dest=paths.MODEL_PT,
)

# Puzzle 2: https://huggingface.co/spaces/jane-street/droppedaneuralnet
DROPPED = Remote(
    url="https://huggingface.co/spaces/jane-street/droppedaneuralnet/resolve/"
        "25651c3fe9146e41c9c6fdbe7efc863896bef9ec/historical_data_and_pieces.zip",
    sha256="a08e80c011012501d3ded83518e6ebd7cf41883db8d1b8de9f1002ff7b5f050e",
    size=4_121_303,
    dest=paths.DROPPED_ZIP,
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(remote: Remote, say=print) -> Path:
    dest = remote.dest
    if dest.exists() and dest.stat().st_size == remote.size and sha256_of(dest) == remote.sha256:
        say(f"ok   {dest.relative_to(paths.ROOT) if dest.is_relative_to(paths.ROOT) else dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    say(f"get  {remote.url} ({remote.size / 1e6:.0f} MB)")
    with urllib.request.urlopen(remote.url) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if sys.stdout.isatty():
                print(f"\r     {done / 1e6:7.0f} / {remote.size / 1e6:.0f} MB", end="", flush=True)
    if sys.stdout.isatty():
        print()
    got = sha256_of(tmp)
    if got != remote.sha256:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for {remote.url}\n  expected {remote.sha256}\n  got      {got}")
    shutil.move(tmp, dest)
    return dest


def unpack_dropped() -> None:
    """The zip holds historical_data.csv and pieces/piece_<i>.pth."""
    with zipfile.ZipFile(paths.DROPPED_ZIP) as z:
        for info in z.infolist():
            target = (paths.DROPPED_DIR / info.filename).resolve()
            if not target.is_relative_to(paths.DROPPED_DIR.resolve()):
                raise SystemExit(f"refusing to extract {info.filename} outside {paths.DROPPED_DIR}")
        z.extractall(paths.DROPPED_DIR)


def fetch_all(say=print) -> dict:
    download(DROPPED, say)
    unpack_dropped()
    download(MODEL, say)
    return {"model_sha256": MODEL.sha256, "dropped_sha256": DROPPED.sha256}
