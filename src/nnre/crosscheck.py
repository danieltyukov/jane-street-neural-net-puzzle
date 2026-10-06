"""Compare the numpy reimplementations with the real PyTorch modules (needs ``pip install torch``).

Puzzle 1: ``model.pt`` is loaded by torch itself, but through an unpickler that only accepts the
classes a Linear/ReLU Sequential needs. cloudpickle's function rebuilders are swapped for a
placeholder, so the pickled wrapper is never reconstructed; we call ``Sequential.forward`` (torch's
own code) on the encoded input instead. This also sidesteps the Python-version issue that made Jane
Street ship a second file: 3.10 bytecode cannot be loaded into a 3.11+ interpreter.

Puzzle 2: the pieces are plain state dicts, so ``torch.load(weights_only=True)`` is enough. The
blocks are built from the exact class definitions published with the puzzle.
"""

from __future__ import annotations

import pickle
import types

import numpy as np

from . import paths

ALLOWED = {
    ("torch.nn.modules.container", "Sequential"),
    ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.activation", "ReLU"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("_codecs", "encode"),
}


class _Inert:
    """Stands in for cloudpickle's rebuilders; absorbs every call, e.g. _builtin_type('CodeType')(...)."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Inert()

    def __setstate__(self, state):
        pass


class _AllowlistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__builtin__":
            module = "builtins"
        if module.startswith("cloudpickle"):
            return _Inert  # never rebuild pickled functions or code objects
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


# torch uses .Unpickler for zip archives and .load for the legacy (pre-1.6, non-zip) format, so both
# must go through the allowlist; handing it the real pickle.load would reopen the hole on old files.
restricted_pickle = types.SimpleNamespace(
    Unpickler=_AllowlistUnpickler,
    load=lambda f, **kwargs: _AllowlistUnpickler(f, **kwargs).load(),
    __name__="restricted_pickle")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def hashnet(net, texts: list[str]) -> dict:
    import torch

    from . import fetch

    path = paths.require(paths.MODEL_PT)
    # Only ever hand torch the exact file Jane Street published. torch routes some archive layouts
    # (TorchScript, legacy formats) away from the unpickler we pass, so the hash check comes first.
    if fetch.sha256_of(path) != fetch.MODEL.sha256:
        raise SystemExit(f"{path} is not the published model.pt (SHA-256 mismatch); refusing to load it")
    # weights_only=False is needed because the file holds nn.Module objects, not only tensors.
    # Every class still goes through _AllowlistUnpickler.
    model = torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted_pickle)
    model.eval()
    x = torch.tensor(net.encode(texts).T, dtype=torch.float32)
    with torch.no_grad():
        ref = model.forward(x)[:, 0].numpy().astype(np.float64)
    # the full activations, not only the output: compare the comparator's inputs too
    ours = net(texts)
    with torch.no_grad():
        h = x
        for layer in list(model)[:-4]:
            h = layer(h)
    mid_ref = h.numpy().T.astype(np.float64)
    mid_ours = net.activations(texts, [net.depth - 3])[net.depth - 3]
    return {"inputs": len(texts), "comparator_width": int(mid_ref.shape[0]),
            "outputs_equal": bool(np.array_equal(ref, ours)),
            "comparator_inputs_equal": bool(np.array_equal(mid_ref, mid_ours)),
            "torch_outputs": {t: float(v) for t, v in zip(texts, ref) if v != 0}}


def dropped(puzzle, permutation: list[int]) -> dict:
    import torch
    from torch import nn

    class Block(nn.Module):  # verbatim from the puzzle page
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.inp = nn.Linear(in_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.out = nn.Linear(hidden_dim, in_dim)

        def forward(self, x):
            residual = x
            x = self.inp(x)
            x = self.activation(x)
            x = self.out(x)
            return residual + x

    class LastLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.layer = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.layer(x)

    def piece(i):
        return torch.load(paths.PIECES_DIR / f"piece_{i}.pth", map_location="cpu", weights_only=True)

    layers = []
    for k in range(48):
        blk = Block(48, 96)
        blk.inp.load_state_dict(piece(permutation[2 * k]))
        blk.out.load_state_dict(piece(permutation[2 * k + 1]))
        layers.append(blk)
    last = LastLayer(48, 1)
    last.layer.load_state_dict(piece(permutation[96]))
    model = nn.Sequential(*layers, last).eval()
    with torch.no_grad():
        out = model(torch.tensor(puzzle.x, dtype=torch.float32))[:, 0].numpy().astype(np.float64)
    return {"rows": len(out), "max_abs_err_vs_pred": float(np.abs(out - puzzle.pred).max())}
