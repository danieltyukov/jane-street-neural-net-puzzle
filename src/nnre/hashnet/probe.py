"""Find MD5's intermediate values inside the network by their "signature".

Run a batch of random inputs. A neuron's signature is the vector of its values across the batch.
If some bit of some MD5 intermediate word has exactly the same values across the batch as a neuron,
that neuron carries the bit. With 160 random inputs a coincidental match would require 160 fair
coin flips to agree, so a match is as good as a proof for any bit that varies.

This is the neural network version of probing: we do not ask what each neuron means, we ask where a
quantity we already suspect (from the last two layers) shows up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import md5
from .model import HashNet


@dataclass
class Located:
    layer: int
    neuron: int


def random_inputs(n: int, seed: int = 0, max_len: int = 31) -> list[str]:
    """Random strings of 0..max_len characters drawn from code points 1..255 (no NULs)."""
    rng = random.Random(seed)
    return ["".join(chr(rng.randint(1, 255)) for _ in range(rng.randint(0, max_len))) for _ in range(n)]


def signature_index(net: HashNet, strings: list[str]) -> dict[bytes, Located]:
    """First neuron (in depth order) for every distinct signature that is not constant."""
    index: dict[bytes, Located] = {}

    def visit(layer: int, act: np.ndarray) -> None:
        varying = np.nonzero(act.max(axis=1) != act.min(axis=1))[0]
        rows = act[varying].astype(np.int16)
        for n, row in zip(varying, rows):
            key = row.tobytes()
            if key not in index:
                index[key] = Located(layer, int(n))

    net.forward(net.encode(strings), visit)
    return index


def bit_signature(values: np.ndarray, bit: int) -> bytes:
    return ((values >> np.uint64(bit)) & np.uint64(1)).astype(np.int16).tobytes()


def md5_traces(strings: list[str]) -> dict[str, np.ndarray]:
    """Per input, every intermediate word of the network's MD5: arrays of shape (batch, 64)."""
    names = ("f", "sum", "rot", "b")
    out = {k: [] for k in names}
    msg = []
    for s in strings:
        block = md5.network_block(s)
        msg.append(np.frombuffer(block, dtype="<u4").astype(np.uint64))
        tr: list[dict] = []
        md5.compress(block, trace=tr)
        for k in names:
            out[k].append([step[k] for step in tr])
    res = {k: np.array(v, dtype=np.uint64) for k, v in out.items()}
    res["message"] = np.array(msg)  # (batch, 16)
    return res


@dataclass
class WordHit:
    found: int          # how many of the varying bits were located
    varying: int        # how many bits vary across the batch at all
    first: int | None   # earliest layer holding any bit
    last: int | None    # layer by which every located bit exists


def locate_word(index: dict[bytes, Located], values: np.ndarray) -> WordHit:
    layers = []
    varying = 0
    for j in range(32):
        bits = (values >> np.uint64(j)) & np.uint64(1)
        if bits.min() == bits.max():
            continue
        varying += 1
        hit = index.get(bit_signature(values, j))
        if hit is not None:
            layers.append(hit.layer)
    return WordHit(len(layers), varying, min(layers) if layers else None, max(layers) if layers else None)


def md5_map(net: HashNet, n: int = 160, seed: int = 0) -> dict:
    strings = random_inputs(n, seed)
    index = signature_index(net, strings)
    tr = md5_traces(strings)
    steps = []
    for i in range(64):
        row = {"step": i, "message_word": md5.message_index(i)}
        for k in ("f", "sum", "rot", "b"):
            row[k] = locate_word(index, tr[k][:, i])
        steps.append(row)
    words = {g: locate_word(index, tr["message"][:, g]) for g in range(16)}
    done = [s["b"].last for s in steps]
    period = sorted(set(np.diff(done).tolist()))
    return {"strings": len(strings), "signatures": len(index), "steps": steps, "message": words,
            "step_done": done, "period": period}


def step_start(i: int) -> int:
    """Layer at which step i-1's new word is complete, i.e. where step i begins (period 42)."""
    return 59 + 42 * (i - 1)


def presum_values(strings: list[str], i: int) -> np.ndarray:
    """a + f + K[i] + M[g] before the rotation, for step i."""
    return md5_traces(strings)["sum"][:, i]


def linear_probe(net: HashNet, strings: list[str], values: np.ndarray, layers) -> dict[int, dict]:
    """For each layer: how many of the 32 bits are an exact linear function of that layer's neurons.

    A single-neuron probe asks "is this bit some neuron?"; a linear probe asks "is it some weighted
    sum of neurons?". The network keeps XORs as a + b - 2*AND(a, b) spread over three neurons, which
    only the second question can see. Needs more inputs than neurons per layer to mean anything.
    """
    acts = net.activations(strings, layers)
    out = {}
    for layer in layers:
        a = np.hstack([acts[layer].T, np.ones((len(strings), 1))])
        exact = 0
        for j in range(32):
            y = ((values >> np.uint64(j)) & np.uint64(1)).astype(np.float64)
            coef, *_ = np.linalg.lstsq(a, y, rcond=None)
            exact += int(np.abs(a @ coef - y).max() < 1e-6)
        out[layer] = {"width": a.shape[1] - 1, "rank": int(np.linalg.matrix_rank(a)), "linear_bits": exact}
    return out


def message_schedule(net: HashNet, n: int = 160, seed: int = 0) -> dict[int, list[int]]:
    """Layers where bits of message words 0..7 are produced by a real gate (wires excluded).

    Inputs shorter than 32 characters leave words 8..13 at zero, so only words 0..7 can be seen.
    """
    strings = random_inputs(n, seed)
    blocks = np.array([np.frombuffer(md5.network_block(s), dtype="<u4").astype(np.uint64) for s in strings])
    targets = {}
    for g in range(8):
        for j in range(32):
            bits = (blocks[:, g] >> np.uint64(j)) & np.uint64(1)
            if bits.min() != bits.max():
                targets[bit_signature(blocks[:, g], j)] = g
    hits: dict[int, set[int]] = {g: set() for g in range(8)}

    def visit(layer: int, act: np.ndarray) -> None:
        w = net.weights[layer]
        fanin = np.diff(w.indptr)
        first_weight = w.data[np.minimum(w.indptr[:-1], max(w.nnz - 1, 0))] if w.nnz else np.zeros(len(fanin))
        wire = (fanin == 1) & (first_weight == 1) & (net.biases[layer] == 0)
        for nn in np.nonzero(~wire)[0]:
            g = targets.get(act[nn].astype(np.int16).tobytes())
            if g is not None:
                hits[g].add(layer)

    net.forward(net.encode(strings), visit)
    return {g: sorted(v) for g, v in hits.items()}


def byte_splitters(net: HashNet) -> list[int]:
    """Layers holding the first stage of a byte-to-bits chain, the neuron relu(x - 127).

    Together with relu(x - 128) it gives [x >= 128], and the chain continues with 64, 32, ..., 2.
    """
    out = []
    for layer, w in enumerate(net.weights):
        fanin = np.diff(w.indptr)
        for n in np.nonzero(fanin == 1)[0]:
            if w.data[w.indptr[n]] == 1 and net.biases[layer][n] == -127:
                out.append(layer)
    return out
