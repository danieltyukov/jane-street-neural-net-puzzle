"""Puzzle 1's network as sparse integer matrices, plus a batched simulator.

``model.pt`` is a ``torch.nn.Sequential`` of 2721 ``Linear`` layers, each followed by ``ReLU``, and a
replaced ``_call_impl`` so that ``model("some text")`` works. All weights and biases are small
integers and 99.6% of the weights are zero, so the whole network fits in a few MB as CSR matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .. import paths, pybytecode
from ..torchzip import TorchArchive

INPUT_LEN = 55  # from the wrapper: str(x)[:55].ljust(55, '\x00')


@dataclass
class HashNet:
    weights: list[sp.csr_matrix]
    biases: list[np.ndarray]
    wrapper: str = ""
    globals: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.weights)

    def widths(self) -> list[int]:
        return [w.shape[0] for w in self.weights]

    @staticmethod
    def encode(strings: list[str]) -> np.ndarray:
        """What the wrapper feeds to the layers: code points of the first 55 characters, NUL-padded."""
        cols = [[ord(c) for c in str(s)[:INPUT_LEN].ljust(INPUT_LEN, "\x00")] for s in strings]
        return np.array(cols, dtype=np.float64).T

    def forward(self, x: np.ndarray, on_layer=None) -> np.ndarray:
        """Run a batch (columns of ``x``). ``on_layer(i, act)`` sees every post-ReLU activation.

        Everything is an integer well below 2**53, so float64 arithmetic is exact.
        """
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.maximum(w @ x + b[:, None], 0.0)
            if on_layer is not None:
                on_layer(i, x)
        return x

    def __call__(self, strings: list[str]) -> np.ndarray:
        return self.forward(self.encode(strings))[0]

    def inputs_of(self, layer: int, neuron: int) -> tuple[list[tuple[int, int]], int]:
        """((input index, weight) pairs, bias) of one neuron."""
        w = self.weights[layer]
        lo, hi = w.indptr[neuron], w.indptr[neuron + 1]
        return [(int(c), int(v)) for c, v in zip(w.indices[lo:hi], w.data[lo:hi])], int(self.biases[layer][neuron])

    def describe(self, layer: int, neuron: int, max_terms: int = 12) -> str:
        """One neuron as a formula, e.g. ``L1:224 = relu(8*L0:56 + 8*L0:57 + ... + 0)``."""
        terms, bias = self.inputs_of(layer, neuron)
        src = "x" if layer == 0 else f"L{layer - 1}:"
        parts = [f"{w}*{src}{'[' + str(c) + ']' if layer == 0 else c}" for c, w in terms[:max_terms]]
        if len(terms) > max_terms:
            parts.append(f"... ({len(terms)} inputs)")
        return f"L{layer}:{neuron} = relu({' + '.join(parts) or '0'} {bias:+d})"

    def activations(self, strings: list[str], layers) -> dict[int, np.ndarray]:
        keep = set(layers)
        out: dict[int, np.ndarray] = {}
        self.forward(self.encode(strings), lambda i, a: out.__setitem__(i, a.copy()) if i in keep else None)
        return out

    # -- cache -------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        shapes = np.array([w.shape for w in self.weights], dtype=np.int64)
        nnz = np.array([w.nnz for w in self.weights], dtype=np.int64)
        np.savez_compressed(
            path, shapes=shapes, nnz=nnz,
            data=np.concatenate([w.data for w in self.weights]).astype(np.int16),
            indices=np.concatenate([w.indices for w in self.weights]).astype(np.int32),
            indptr=np.concatenate([w.indptr for w in self.weights]).astype(np.int64),
            bias=np.concatenate(self.biases).astype(np.int32),
            wrapper=np.array(self.wrapper))

    @classmethod
    def load_cache(cls, path: Path) -> "HashNet":
        z = np.load(path)
        shapes, nnz = z["shapes"], z["nnz"]
        data, indices, indptr, bias = z["data"], z["indices"], z["indptr"], z["bias"]
        weights, biases = [], []
        d = p = b = 0
        for (rows, cols), n in zip(shapes, nnz):
            w = sp.csr_matrix((data[d:d + n].astype(np.float64), indices[d:d + n], indptr[p:p + rows + 1]),
                              shape=(rows, cols))
            weights.append(w)
            biases.append(bias[b:b + rows].astype(np.float64))
            d, p, b = d + n, p + rows + 1, b + rows
        return cls(weights, biases, str(z["wrapper"]))


def from_model_pt(path: Path) -> HashNet:
    arc = TorchArchive.open(path)
    seen = dict(arc.globals())
    root = arc.graph()
    if root.qualname != "torch.nn.modules.container.Sequential":
        raise ValueError(f"unexpected root object {root.qualname}")
    modules = root.state["_modules"]
    kinds = [modules[k].qualname.rsplit(".", 1)[1] for k in modules]
    if kinds != ["Linear", "ReLU"] * (len(kinds) // 2):
        raise ValueError("expected alternating Linear and ReLU modules")

    weights, biases = [], []
    for key in modules:
        mod = modules[key]
        if not mod.qualname.endswith("Linear"):
            continue
        params = mod.state["_parameters"]
        w = arc.tensor(params["weight"])
        b = arc.tensor(params["bias"])
        if not (np.array_equal(w, np.round(w)) and np.array_equal(b, np.round(b))):
            raise ValueError(f"layer {len(weights)} has non-integer parameters")
        weights.append(sp.csr_matrix(w.astype(np.float64)))
        biases.append(b.astype(np.float64))

    # The replaced _call_impl: cloudpickle's _make_function(code, globals, name, defaults, closure)
    make_fn = root.state["_call_impl"]
    co = pybytecode.code_fields(make_fn.args[0].args)
    return HashNet(weights, biases, pybytecode.decompile(co), seen)


def load(cache: Path | None = None) -> HashNet:
    """Read the cache in build/ if present, otherwise parse data/hashnet/model.pt and write the cache."""
    cache = cache or paths.BUILD / "hashnet.npz"
    if cache.exists():
        return HashNet.load_cache(cache)
    net = from_model_pt(paths.require(paths.MODEL_PT))
    cache.parent.mkdir(parents=True, exist_ok=True)
    net.save(cache)
    return net
