"""Step 1 of puzzle 2: which ``inp`` layer goes with which ``out`` layer.

Shapes do not help: every 96x48 ``inp`` composes with every 48x96 ``out``. What does help is that
hidden unit k of a block reads along row k of ``W_in`` and writes along column k of ``W_out``, and
those two vectors were trained together. ``trace(W_out @ W_in)`` is the sum over the 96 hidden
units of (read direction . write direction). For a true pair it is a sum of 96 related terms; for a
wrong pair, unit k of one block has nothing to do with unit k of the other and the terms cancel to
roughly zero. In this model every true pair has a clearly negative trace: each unit tends to write
back against the direction it reads, i.e. the blocks damp the features they detect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .load import Puzzle


@dataclass
class Pairing:
    blocks: list[tuple[int, int]]   # (inp piece, out piece)
    scores: np.ndarray              # trace matrix, rows = inp pieces, cols = out pieces
    inp: list[int]
    out: list[int]
    margin: float                   # smallest gap between a block's own score and its best rival
    agrees_with_argmin: bool        # each row's minimum already gives the same matching


def trace_scores(puzzle: Puzzle) -> tuple[np.ndarray, list[int], list[int]]:
    inp, out = sorted(puzzle.kind("inp")), sorted(puzzle.kind("out"))
    w_in = np.stack([puzzle.pieces[i].weight for i in inp])     # (48, 96, 48)
    w_out = np.stack([puzzle.pieces[j].weight for j in out])    # (48, 48, 96)
    # trace(W_out_j @ W_in_i) = sum_{a,h} W_out_j[a,h] * W_in_i[h,a]
    scores = np.einsum("jah,iha->ij", w_out, w_in)
    return scores, inp, out


def pair(puzzle: Puzzle) -> Pairing:
    scores, inp, out = trace_scores(puzzle)
    rows, cols = linear_sum_assignment(scores)  # minimise the total trace
    blocks = [(inp[r], out[c]) for r, c in zip(rows, cols)]
    own = scores[rows, cols]
    rivals = scores.copy()
    rivals[rows, cols] = np.inf
    row_gap = rivals[rows].min(axis=1) - own        # best other out layer for this inp layer
    col_gap = rivals[:, cols].min(axis=0) - own     # best other inp layer for this out layer
    margin = float(min(row_gap.min(), col_gap.min()))
    agrees = bool(np.array_equal(scores.argmin(axis=1)[rows], cols))
    return Pairing(blocks, scores, inp, out, margin, agrees)
