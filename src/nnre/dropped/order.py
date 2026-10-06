"""Step 2 of puzzle 2: the order of the 48 blocks, then the 97-number answer.

The historical data includes ``pred``, the original model's own output. The right order reproduces
it to float32 precision, a wrong one does not, so ``mean((model(x) - pred)^2)`` is an objective
with a known optimum of (almost) zero.

Two ways to search it:

* ``norm`` (default): sort the blocks by the Frobenius norm of their ``out`` weight, which grows
  with depth in this model, then fix the remaining local mistakes with adjacent swaps. Seconds.
* ``search``: assume nothing. Build an order greedily and improve it by moving one block at a time
  to its best position until no move helps. This is how the answer was first found; it takes about
  ten minutes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from .load import Puzzle

# From the puzzle's checker (app.py in the Hugging Face space): sha256 of the comma-separated answer.
EXPECTED_SHA256 = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"

Block = tuple[int, int]


class Model:
    """The reassembled network for a given block order, in float64 numpy."""

    def __init__(self, puzzle: Puzzle, blocks: list[Block]):
        p = puzzle.pieces
        self.w_in = {b: p[b[0]].weight.T.copy() for b in blocks}
        self.b_in = {b: p[b[0]].bias for b in blocks}
        self.w_out = {b: p[b[1]].weight.T.copy() for b in blocks}
        self.b_out = {b: p[b[1]].bias for b in blocks}
        last = p[puzzle.kind("last")[0]]
        self.w_last, self.b_last = last.weight[0], last.bias[0]

    def block(self, h: np.ndarray, b: Block) -> np.ndarray:
        return h + np.maximum(h @ self.w_in[b] + self.b_in[b], 0) @ self.w_out[b] + self.b_out[b]

    def head(self, h: np.ndarray) -> np.ndarray:
        return h @ self.w_last + self.b_last

    def run(self, x: np.ndarray, order: list[Block]) -> np.ndarray:
        for b in order:
            x = self.block(x, b)
        return self.head(x)


@dataclass
class OrderResult:
    order: list[Block]
    permutation: list[int]
    mse_sample: float
    max_abs_err: float      # vs pred, on every row
    mse_true: float         # the reassembled model's own loss on the targets
    evaluations: int
    seconds: float
    start_inversions: int | None = None
    history: list = field(default_factory=list)

    @property
    def answer(self) -> str:
        return ",".join(map(str, self.permutation))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.answer.encode()).hexdigest()

    @property
    def matches_checker(self) -> bool:
        return self.sha256 == EXPECTED_SHA256


def norm_start(puzzle: Puzzle, blocks: list[Block]) -> list[Block]:
    return sorted(blocks, key=lambda b: np.linalg.norm(puzzle.pieces[b[1]].weight))


def adjacent_swaps(mse, order: list[Block], history: list | None = None) -> tuple[list[Block], float, int]:
    """Swap neighbours while that lowers the error. ``history`` collects (evaluations, mse)."""
    cur, evals, improved = mse(order), 1, True
    if history is not None:
        history.append((evals, cur))
    while improved:
        improved = False
        for i in range(len(order) - 1):
            cand = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
            m = mse(cand)
            evals += 1
            if m < cur:
                order, cur, improved = cand, m, True
                if history is not None:
                    history.append((evals, cur))
    return order, cur, evals


def insertion_moves(model: Model, xs, ps, order: list[Block], say=None) -> tuple[list[Block], float, int]:
    """Move one block to its best position until nothing improves. Reuses prefix activations."""
    def tail_mse(h, seq, start):
        for b in seq[start:]:
            h = model.block(h, b)
        return float(((model.head(h) - ps) ** 2).mean())

    cur, evals, improved = tail_mse(xs, order, 0), 1, True
    while improved:
        improved = False
        for a in range(len(order)):
            rest = order[:a] + order[a + 1:]
            prefix = [xs]
            for b in rest:
                prefix.append(model.block(prefix[-1], b))
            best, best_order = cur, None
            for pos in range(len(order)):
                if pos == a:
                    continue
                cand = rest[:pos] + [order[a]] + rest[pos:]
                m = tail_mse(prefix[pos], cand, pos)
                evals += 1
                if m < best:
                    best, best_order = m, cand
            if best_order is not None:
                order, cur, improved = best_order, best, True
        if say:
            say(f"  insertion pass: mse {cur:.3g}")
    return order, cur, evals


def greedy_start(model: Model, xs, ps, blocks: list[Block]) -> list[Block]:
    rest, order, h = list(blocks), [], xs
    while rest:
        nxt = min(rest, key=lambda b: float(((model.head(model.block(h, b)) - ps) ** 2).mean()))
        order.append(nxt)
        rest.remove(nxt)
        h = model.block(h, nxt)
    return order


def inversions(order: list[Block], reference: list[Block]) -> int:
    rank = {b: i for i, b in enumerate(reference)}
    seq = [rank[b] for b in order]
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def solve(puzzle: Puzzle, blocks: list[Block], method: str = "norm", rows: int = 1000, seed: int = 0,
          say=print) -> OrderResult:
    t = time.time()
    model = Model(puzzle, blocks)
    idx = np.random.default_rng(seed).choice(len(puzzle.x), size=min(rows, len(puzzle.x)), replace=False)
    xs, ps = puzzle.x[idx], puzzle.pred[idx]
    mse = lambda order: float(((model.run(xs, order) - ps) ** 2).mean())

    history: list = []
    if method == "norm":
        start = norm_start(puzzle, blocks)
        order, cur, evals = adjacent_swaps(mse, start, history)
        if cur > 1e-10:  # not there yet: finish with the slower, stronger moves
            order, cur, more = insertion_moves(model, xs, ps, order, say)
            evals += more
    elif method == "search":
        start = greedy_start(model, xs, ps, blocks)
        if say:
            say(f"  greedy start: mse {mse(start):.3g}")
        order, cur, evals = insertion_moves(model, xs, ps, start, say)
    else:
        raise ValueError(method)

    full = model.run(puzzle.x, order)
    perm = [p for b in order for p in b] + puzzle.kind("last")
    return OrderResult(order, perm, cur, float(np.abs(full - puzzle.pred).max()),
                       float(((full - puzzle.true) ** 2).mean()), evals, time.time() - t,
                       inversions(start, order), history)


def depth_signals(puzzle: Puzzle, order: list[Block]) -> dict[str, float]:
    """Spearman correlation of simple per-block statistics with the block's true position."""
    from scipy.stats import spearmanr

    model = Model(puzzle, order)
    x = puzzle.x
    h, inputs = x, []
    for b in order:
        inputs.append(h)
        h = model.block(h, b)
    stats = {
        "|W_out|": [np.linalg.norm(puzzle.pieces[o].weight) for _, o in order],
        "|W_in|": [np.linalg.norm(puzzle.pieces[i].weight) for i, _ in order],
        "mean of b_in": [puzzle.pieces[i].bias.mean() for i, _ in order],
        "|b_in|": [np.linalg.norm(puzzle.pieces[i].bias) for i, _ in order],
        "|b_out|": [np.linalg.norm(puzzle.pieces[o].bias) for _, o in order],
        "trace(W_out W_in)": [np.trace(puzzle.pieces[o].weight @ puzzle.pieces[i].weight) for i, o in order],
        "|block(x) - x| on the raw data": [np.linalg.norm(model.block(x, b) - x, axis=1).mean() for b in order],
        "|h| entering the block": [np.linalg.norm(hh, axis=1).mean() for hh in inputs],
    }
    pos = np.arange(len(order))
    return {k: float(spearmanr(pos, v).correlation) for k, v in stats.items()}
