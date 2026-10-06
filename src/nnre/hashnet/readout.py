"""The last two layers: a 16-byte equality test written with ReLUs.

For an integer v, ``relu(v-t+1) - 2*relu(v-t) + relu(v-t-1)`` is 1 when v == t and 0 otherwise (a
"hat"). The second-to-last layer has 48 neurons: for each of 16 bytes, three neurons that share the
same input combination v and have biases -(t+1), -t and -(t-1). The last layer adds the 16 hats with
weights (1, -2, 1), subtracts 15 and applies ReLU, so it outputs 1 only if all 16 bytes match.

Reading the hat centres gives the target bytes. Reading the shared input combination gives the
16-byte value the network compares, for any input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import HashNet


@dataclass
class Comparator:
    target: bytes            # the 16 bytes the network is looking for
    combine: np.ndarray      # 16 x width: row k turns the previous layer into byte k
    groups: list[tuple[int, int, int]]  # neuron triples per byte; the middle one has bias -t

    @property
    def target_hex(self) -> str:
        return self.target.hex()


def comparator(net: HashNet) -> Comparator:
    w_last = net.weights[-1].toarray()[0]
    b_last = net.biases[-1][0]
    w_cmp = net.weights[-2].toarray()
    b_cmp = net.biases[-2]
    n = len(w_last) // 3
    # the last layer: +1 on the first third, -2 on the middle third, +1 on the last third, bias -(n-1)
    expect = np.concatenate([np.ones(n), -2 * np.ones(n), np.ones(n)])
    if not (np.array_equal(w_last, expect) and b_last == -(n - 1)):
        raise ValueError("last layer is not the 'all hats equal one' AND")
    groups, target = [], []
    for k in range(n):
        lo, mid, hi = k, k + n, k + 2 * n
        if not (np.array_equal(w_cmp[lo], w_cmp[mid]) and np.array_equal(w_cmp[mid], w_cmp[hi])):
            raise ValueError(f"byte {k}: the three neurons read different inputs")
        t = -b_cmp[mid]
        if not (b_cmp[lo] == -(t + 1) and b_cmp[hi] == -(t - 1)):
            raise ValueError(f"byte {k}: biases are not -(t+1), -t, -(t-1)")
        groups.append((lo, mid, hi))
        target.append(int(t))
    return Comparator(bytes(target), w_cmp[:n], groups)


def digests(net: HashNet, strings: list[str], cmp: Comparator | None = None) -> list[bytes]:
    """The 16-byte value the network compares against the target, for each input string."""
    cmp = cmp or comparator(net)
    acts = net.activations(strings, [net.depth - 3])[net.depth - 3]
    values = np.rint(cmp.combine @ acts).astype(np.int64).T
    return [bytes(int(v) & 0xFF for v in row) if (row.min() >= 0 and row.max() < 256) else None
            for row in values]


def byte_terms(net: HashNet, text: str, cmp: Comparator | None = None) -> list[tuple[int, int]]:
    """Per byte: (sum of the positive-weight inputs, sum of the negative-weight inputs).

    Bytes 4-7 and 12-15 arrive with their last XOR not yet applied: each bit is s - 2n, where s is
    a + b (0, 1 or 2) and n is AND(a, b). The comparator's weights apply it, so the byte value is
    (sum of s_i 2^i) - (sum of 2 n_i 2^i).
    """
    cmp = cmp or comparator(net)
    a = net.activations([text], [net.depth - 3])[net.depth - 3][:, 0]
    pos = np.rint(np.clip(cmp.combine, 0, None) @ a).astype(int)
    neg = np.rint(-np.clip(cmp.combine, None, 0) @ a).astype(int)
    return list(zip(pos.tolist(), neg.tolist()))


def hat(v: np.ndarray, t: int) -> np.ndarray:
    relu = lambda z: np.maximum(z, 0)
    return relu(v - t + 1) - 2 * relu(v - t) + relu(v - t - 1)
