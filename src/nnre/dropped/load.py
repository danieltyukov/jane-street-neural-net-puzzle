"""Puzzle 2's 97 pieces and its historical data, as numpy arrays.

Each ``piece_<i>.pth`` is the ``state_dict`` of one ``nn.Linear``: a weight and a bias, nothing
else. They are read with the same inert reader as puzzle 1, so torch is not needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import paths
from ..torchzip import TorchArchive

N_PIECES = 97


@dataclass
class Piece:
    index: int
    weight: np.ndarray  # (out_features, in_features), float64
    bias: np.ndarray

    @property
    def kind(self) -> str:
        out_f, in_f = self.weight.shape
        if out_f == 1:
            return "last"
        return "inp" if out_f > in_f else "out"


@dataclass
class Puzzle:
    pieces: dict[int, Piece]
    x: np.ndarray      # (rows, 48) measurements
    pred: np.ndarray   # what the original model predicted
    true: np.ndarray   # the target it was trained on

    def kind(self, k: str) -> list[int]:
        return [i for i, p in self.pieces.items() if p.kind == k]


def load_piece(i: int) -> Piece:
    arc = TorchArchive.open(paths.require(paths.PIECES_DIR / f"piece_{i}.pth"))
    state = arc.graph()
    if set(state) != {"weight", "bias"}:
        raise ValueError(f"piece {i}: unexpected keys {list(state)}")
    return Piece(i, arc.tensor(state["weight"]).astype(np.float64), arc.tensor(state["bias"]).astype(np.float64))


def load() -> Puzzle:
    pieces = {i: load_piece(i) for i in range(N_PIECES)}
    csv = paths.require(paths.HISTORICAL_CSV)
    header = csv.open().readline().strip().split(",")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    if header[-2:] != ["pred", "true"]:
        raise ValueError(f"unexpected columns {header[-2:]}")
    return Puzzle(pieces, data[:, :-2], data[:, -2], data[:, -1])
