"""Read ``torch.save`` files without torch and without running the pickle.

A ``.pt``/``.pth`` file is a zip archive: ``<name>/data.pkl`` describes the object graph and
``<name>/data/<key>`` holds the raw bytes of each tensor storage. ``torch.load(weights_only=False)``
executes whatever the pickle asks for, which for puzzle 1 includes rebuilding a Python function from
bytecode. Here every global the pickle references is replaced by an inert placeholder class that only
records how it was called, so the graph can be inspected and the tensors read as numpy arrays while no
code from the file ever runs.
"""

from __future__ import annotations

import codecs
import collections
import io
import pickle
import pickletools
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Globals that are rebuilt for real because they are plain data containers.
_REAL = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("__builtin__", "set"): set,
    ("builtins", "set"): set,
    ("_codecs", "encode"): codecs.encode,  # protocol 2 stores bytes as latin-1 text plus this call
}


class Stub:
    """Stands in for any class or function named in the pickle. Calling it records the arguments."""

    qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.kwargs, obj.state, obj.items = args, kwargs, None, []
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):  # BUILD
        self.state = state

    def __call__(self, *args, **kwargs):  # a stub that is itself called, e.g. _builtin_type('CodeType')(...)
        return stub_class("<call>", f"{self.qualname}{self.args!r}"[:120])(*args, **kwargs)

    def append(self, item):  # APPEND on a stubbed list subclass
        self.items.append(item)

    def __setitem__(self, key, value):  # SETITEM on a stubbed dict subclass
        self.items.append((key, value))

    def __repr__(self):
        return f"<stub {self.qualname} nargs={len(self.args)}>"


_STUB_CLASSES: dict[str, type] = {}


def stub_class(module: str, name: str) -> type:
    qual = f"{module}.{name}"
    if qual not in _STUB_CLASSES:
        _STUB_CLASSES[qual] = type(name, (Stub,), {"qualname": qual})
    return _STUB_CLASSES[qual]


class InertUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _REAL.get((module, name)) or stub_class(module, name)

    def persistent_load(self, pid):
        # torch writes ('storage', storage_type, key, location, numel) for each tensor storage
        return ("storage",) + tuple(pid[1:])


def referenced_globals(pkl: bytes) -> collections.Counter:
    """Every ``module.name`` the pickle would import, read from the opcode stream without unpickling."""
    found: collections.Counter = collections.Counter()
    strings: list[str] = []
    memo: dict[int, str] = {}
    for op, arg, _ in pickletools.genops(pkl):
        if op.name == "GLOBAL":
            found[arg.replace(" ", ".")] += 1
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        elif op.name == "MEMOIZE" and strings:
            memo[len(memo)] = strings[-1]
        elif op.name in ("BINGET", "LONG_BINGET", "GET") and arg in memo:
            strings.append(memo[arg])
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found[f"{strings[-2]}.{strings[-1]}"] += 1
    return found


@dataclass
class TorchArchive:
    path: Path
    zf: zipfile.ZipFile
    prefix: str
    pkl: bytes

    @classmethod
    def open(cls, path: str | Path) -> "TorchArchive":
        zf = zipfile.ZipFile(path)
        pkl_name = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl_name.rsplit("/", 1)[0]
        if zf.read(f"{prefix}/byteorder") != b"little":
            raise ValueError("only little-endian archives are supported")
        return cls(Path(path), zf, prefix, zf.read(pkl_name))

    def globals(self) -> collections.Counter:
        return referenced_globals(self.pkl)

    def graph(self):
        """The unpickled object graph, with stubs in place of every class and function."""
        return InertUnpickler(io.BytesIO(self.pkl)).load()

    def tensor(self, rebuilt) -> np.ndarray:
        """Turn a ``torch._utils._rebuild_tensor_v2`` (or ``_rebuild_parameter``) stub into an array."""
        if rebuilt.qualname == "torch._utils._rebuild_parameter":
            rebuilt = rebuilt.args[0]
        if rebuilt.qualname != "torch._utils._rebuild_tensor_v2":
            raise TypeError(f"not a tensor: {rebuilt!r}")
        (_, storage_type, key, _location, _numel), offset, size, stride = rebuilt.args[:4]
        dtype = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8",
                 "IntStorage": "<i4"}[storage_type.qualname.rsplit(".", 1)[1]]
        buf = np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=dtype)
        item = buf.itemsize
        view = np.lib.stride_tricks.as_strided(buf[offset:], shape=tuple(size),
                                               strides=tuple(s * item for s in stride))
        return np.array(view)
