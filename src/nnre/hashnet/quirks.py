"""Where the network stops being MD5.

1. It never searches for the end of the string. Layer 0 makes "is this character non-zero" flags
   (``x - relu(x-1)``), layer 1 adds them up times 8, and that count k drives both the 0x80 marker
   position and the length field. With NULs inside the string, the result is not the MD5 of
   anything you typed, but it is exactly ``md5.network_md5``.
2. The length 8k is split into bits by a chain of "subtract 2^j if at least 2^j" stages that starts
   at 128. That handles 8k <= 255, so k <= 31. From 32 characters on, the stages see a remainder
   larger than they were built for, the "bits" stop being 0/1 and the hash is garbage. This is the
   ">32 bytes" bug Jane Street mentions in their write-up.
"""

from __future__ import annotations

import hashlib
import random
import string

import numpy as np

from . import md5, readout
from .model import HashNet


def length_stage_layers(net: HashNet) -> list[tuple[int, int]]:
    """Layer 1 neuron(s) that compute 8 * (number of non-NUL characters): many inputs, weights +-8."""
    w = net.weights[1].toarray()
    out = []
    for n in range(w.shape[0]):
        nz = w[n][w[n] != 0]
        if len(nz) >= 110 and set(nz.tolist()) == {8.0, -8.0}:
            out.append((1, n))
    return out


def random_with_nuls(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = [chr(rng.randint(1, 127)) for _ in range(rng.randint(1, 30))]
        for _ in range(rng.randint(1, 3)):
            chars[rng.randrange(len(chars))] = "\x00"
        out.append("".join(chars))
    return out


def by_length(net: HashNet, cmp: readout.Comparator, lengths=range(0, 56), per: int = 4, seed: int = 1) -> dict:
    """For each input length: how often the network's compared value equals MD5 of the input."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " "
    strs, lens = [], []
    for n in lengths:
        for _ in range(per):
            strs.append("".join(rng.choice(alphabet) for _ in range(n)))
            lens.append(n)
    got = readout.digests(net, strs, cmp)
    res: dict[int, int] = {}
    for n, s, d in zip(lens, strs, got):
        res[n] = res.get(n, 0) + int(d == hashlib.md5(s.encode()).digest())
    return {"per_length": per, "md5_ok": res, "longest_ok": max(n for n, v in res.items() if v == per),
            "shortest_bad": min((n for n, v in res.items() if v < per), default=None)}


def length_bit_neurons(net: HashNet, n: int = 200, seed: int = 3) -> dict[int, tuple[int, int]]:
    """Neurons that hold bits 3..7 of the length field 8k, found by signature on strings with k <= 31."""
    from .probe import Located, bit_signature, signature_index
    rng = random.Random(seed)
    texts = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 31))) for _ in range(n)]
    index: dict[bytes, Located] = signature_index(net, texts)
    eight_k = np.array([8 * len(t) for t in texts], dtype=np.uint64)
    found = {}
    for j in range(3, 8):
        hit = index.get(bit_signature(eight_k, j))
        if hit is not None:
            found[j] = (hit.layer, hit.neuron)
    return found


def length_bits_on(net: HashNet, neurons: dict[int, tuple[int, int]], texts: list[str]) -> dict[int, list[int]]:
    """What the length-bit neurons hold for each text (should be 0 or 1 if the circuit is in range)."""
    acts = net.activations(texts, {layer for layer, _ in neurons.values()})
    return {len(t): [int(acts[layer][nn, col]) for _, (layer, nn) in sorted(neurons.items())]
            for col, t in enumerate(texts)}


def nul_model_check(net: HashNet, cmp: readout.Comparator, n: int = 300, seed: int = 5) -> dict:
    """Strings with embedded NULs: compare the network with network_md5 and with plain MD5."""
    texts = random_with_nuls(n, seed)
    got = readout.digests(net, texts, cmp)
    model_ok = sum(d == md5.network_md5(t) for t, d in zip(texts, got))
    plain_ok = sum(d == hashlib.md5(t.encode()).digest() for t, d in zip(texts, got))
    prefix_ok = sum(d == hashlib.md5(t.split("\x00")[0].encode()).digest() for t, d in zip(texts, got))
    return {"strings": n, "network_md5": model_ok, "md5_of_text": plain_ok, "md5_up_to_first_nul": prefix_ok}


def summary(net: HashNet) -> dict:
    cmp = readout.comparator(net)
    lens = by_length(net, cmp)
    nul = nul_model_check(net, cmp)
    counter = length_stage_layers(net)
    probe_texts = ["a" * n for n in (5, 31, 32, 40, 55)]
    acts = net.activations(probe_texts, [1])[1]
    eight_k = {len(t): int(acts[counter[0][1], i]) for i, t in enumerate(probe_texts)} if counter else {}
    bits = length_bit_neurons(net)
    return {"by_length": lens, "nul": nul, "length_counter": counter, "eight_k": eight_k,
            "length_bit_neurons": bits, "length_bits": length_bits_on(net, bits, probe_texts)}
