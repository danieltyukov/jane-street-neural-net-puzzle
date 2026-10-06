"""Group every neuron by its exact (weights, bias) pattern and name the common ones.

With 0/1 inputs a, b, c, one ReLU neuron is a logic gate:

    relu(a)          wire (copy)          relu(a + b - 1)   AND
    relu(1 - a)      NOT                  relu(1 - a - b)   NOR
    relu(a - b)      a AND NOT b          relu(a + b)       a + b (0, 1 or 2; OR before clipping)

XOR needs one more trick: ``a XOR b = a + b - 2*AND(a, b)`` is linear in a, b and AND(a, b), so the
network never stores XOR in a neuron; the next layer's weights apply the -2 directly. That is why
patterns like ``relu(a + b - 2c)`` and ``relu(a + b - 2c - 1)`` show up.
"""

from __future__ import annotations

import collections

import numpy as np

from .model import HashNet

NAMES = {
    ((1,), 0): "wire  relu(a)",
    ((1, 1), -1): "AND   relu(a+b-1)",
    ((-1, -1), 1): "NOR   relu(1-a-b)",
    ((-1,), 1): "NOT   relu(1-a)",
    ((-1, 1), 0): "ANDN  relu(a-b)",
    ((-2, 1), 0): "relu(a-2b)",
    ((-2, 1, 1), -1): "relu(a+b-2c-1)",
    ((1, 1), 0): "relu(a+b)",
    ((-2, 1, 1), 0): "XOR-style relu(a+b-2c)",
    ((-1, 1), 1): "relu(1+a-b)",
    ((), 1): "constant 1",
    ((), 0): "constant 0",
    ((1,), -1): "relu(a-1)",
}


def pattern(weights: np.ndarray, bias: float, max_fanin: int = 4):
    nz = weights[weights != 0]
    if len(nz) > max_fanin:
        return (f"fan-in {len(nz)}", None)
    return (tuple(sorted(int(v) for v in nz)), int(bias))


def census(net: HashNet) -> dict:
    counts: collections.Counter = collections.Counter()
    values: set[float] = set()
    nnz = total = 0
    for w, b in zip(net.weights, net.biases):
        values.update(np.unique(w.data).tolist())
        nnz += w.nnz
        total += w.shape[0] * w.shape[1]
        for n in range(w.shape[0]):
            counts[pattern(w.data[w.indptr[n]:w.indptr[n + 1]], b[n])] += 1
    neurons = sum(counts.values())
    top = [{"pattern": str(k), "name": NAMES.get(k, ""), "count": v, "share": v / neurons}
           for k, v in counts.most_common(14)]
    return {"neurons": neurons, "layers": net.depth, "nonzero_weights": nnz, "dense_weights": total,
            "weight_values": sorted(values), "wires": counts[((1,), 0)], "top": top}
