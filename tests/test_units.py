"""Unit tests that need no puzzle files."""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import random
import zipfile

import numpy as np
import pytest

from nnre import pybytecode, torchzip
from nnre.dropped import load as dropped_load
from nnre.dropped import order, pair
from nnre.hashnet import census, md5, readout


# -- MD5 -----------------------------------------------------------------------------------------

def test_md5_matches_hashlib():
    rng = random.Random(0)
    for n in list(range(56)) + [0, 55]:
        msg = bytes(rng.randrange(256) for _ in range(n))
        assert md5.md5(msg) == hashlib.md5(msg).digest()


def test_network_block_is_standard_padding_without_nuls():
    for text in ["", "a", "vegetable dog", "x" * 31]:
        assert md5.network_block(text) == md5.pad(text.encode())
        assert md5.network_md5(text) == hashlib.md5(text.encode()).digest()


def test_network_block_counts_non_nul_characters():
    # "ab\0cd": 4 non-NUL characters, so the marker goes to position 4 (on top of 'd') and the length is 32
    block = md5.network_block("ab\x00cd")
    assert block[:5] == b"ab\x00c" + bytes([ord("d") | 0x80])
    assert block[56:] == (32).to_bytes(8, "little")


def test_message_schedule():
    assert [md5.message_index(i) for i in (0, 15, 16, 17, 32, 33, 48, 49)] == [0, 15, 1, 6, 5, 8, 0, 7]


# -- ReLU idioms ---------------------------------------------------------------------------------

def test_hat_is_an_equality_test():
    v = np.arange(0, 256)
    for t in (0, 1, 127, 199, 255):
        assert np.array_equal(readout.hat(v, t), (v == t).astype(float))


def test_relu_gates_on_bits():
    relu = lambda z: max(z, 0)
    for a in (0, 1):
        for b in (0, 1):
            assert relu(a + b - 1) == (a & b)
            assert relu(1 - a - b) == 1 - (a | b)
            assert relu(a - b) == (a & (1 - b))
            # XOR is linear in a, b and AND(a, b): no neuron needed for it
            assert a + b - 2 * relu(a + b - 1) == a ^ b


def test_census_pattern_names():
    assert census.NAMES[census.pattern(np.array([1.0]), 0)].startswith("wire")
    assert census.NAMES[census.pattern(np.array([1.0, 1.0]), -1)].startswith("AND")
    assert census.pattern(np.ones(9), 0) == ("fan-in 9", None)


# -- bytecode ------------------------------------------------------------------------------------

# The code object stored in puzzle 1's model.pt (CPython 3.10), copied field by field.
WRAPPER_310 = (1, 0, 0, 1, 11, 67,
               b"t\x00\xa0\x01t\x02\xa0\x03t\x04t\x05t\x06t\x07|\x00\x83\x01d\x00d\x01\x85\x02\x19\x00\xa0"
               b"\x08d\x01d\x02\xa1\x02\x83\x02\x83\x01\xa1\x01\xa1\x01S\x00",
               (None, 55, "\x00"), ("model", "forward", "torch", "Tensor", "list", "map", "ord", "str", "ljust"),
               ("x",), "/tmp/ipykernel_1810120/378402040.py", "<lambda>", 2, b".\x00", (), ())


def test_decompile_wrapper():
    co = pybytecode.code_fields(WRAPPER_310)
    src = pybytecode.decompile(co)
    assert src == "lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\\x00')))))"


def test_disassemble_rejects_unknown_opcodes():
    co = pybytecode.code_fields(WRAPPER_310[:6] + (b"\xff\x00",) + WRAPPER_310[7:])
    with pytest.raises(ValueError):
        pybytecode.disassemble(co)


# -- inert unpickling ----------------------------------------------------------------------------

class _Evil:
    def __reduce__(self):
        return (os.system, ("touch nnre_pwned",))


def test_inert_unpickler_never_runs_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = pickle.dumps({"weight": _Evil()}, protocol=2)
    refs = torchzip.referenced_globals(payload)
    assert any(name.endswith(".system") for name in refs)
    obj = torchzip.InertUnpickler(io.BytesIO(payload)).load()
    assert isinstance(obj["weight"], torchzip.Stub)
    assert obj["weight"].qualname.endswith(".system")
    assert obj["weight"].args == ("touch nnre_pwned",)
    assert not (tmp_path / "nnre_pwned").exists()


def test_globals_include_inst_opcodes(tmp_path, monkeypatch):
    # INST imports and calls in one opcode, with no GLOBAL anywhere in the stream
    monkeypatch.chdir(tmp_path)
    payload = b"(S'touch nnre_pwned'\nios\nsystem\n."
    assert torchzip.referenced_globals(payload) == {"os.system": 1}
    assert not (tmp_path / "nnre_pwned").exists()


def test_restricted_torch_pickle_refuses_unknown_globals(tmp_path, monkeypatch):
    # torch calls pickle_module.load directly for legacy (non-zip) files, so it must be restricted too
    from nnre import crosscheck
    monkeypatch.chdir(tmp_path)
    payload = pickle.dumps(_Evil(), protocol=2)
    with pytest.raises(pickle.UnpicklingError):
        crosscheck.restricted_pickle.load(io.BytesIO(payload))
    with pytest.raises(pickle.UnpicklingError):
        crosscheck.restricted_pickle.Unpickler(io.BytesIO(payload)).load()
    assert not (tmp_path / "nnre_pwned").exists()


def _fake_torch_file(path, arrays: dict[str, np.ndarray]):
    """A minimal torch.save-style zip with a hand-assembled protocol 2 pickle.

    The pickle is the same shape torch writes for a state dict: {name: _rebuild_tensor_v2(
    persistent_id(('storage', FloatStorage, key, 'cpu', numel)), offset, size, stride)}.
    """
    def text(s):
        b = s.encode()
        return b"X" + len(b).to_bytes(4, "little") + b

    def integer(i):
        return b"J" + int(i).to_bytes(4, "little", signed=True)

    def tup(items):
        return b"(" + b"".join(items) + b"t"

    body = b"\x80\x02}("
    with zipfile.ZipFile(path, "w") as z:
        for k, (name, arr) in enumerate(arrays.items()):
            arr = np.ascontiguousarray(arr, dtype="<f4")
            z.writestr(f"archive/data/{k}", arr.tobytes())
            pid = tup([text("storage"), b"ctorch\nFloatStorage\n", text(str(k)), text("cpu"), integer(arr.size)])
            args = tup([pid + b"Q", integer(0), tup([integer(d) for d in arr.shape]),
                        tup([integer(st // 4) for st in arr.strides])])
            body += text(name) + b"ctorch._utils\n_rebuild_tensor_v2\n" + args + b"R"
        body += b"u."
        z.writestr("archive/data.pkl", body)
        z.writestr("archive/byteorder", "little")


def test_tensor_reader_round_trip(tmp_path):
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    path = tmp_path / "piece.pth"
    _fake_torch_file(path, {"weight": w, "bias": b})
    arc = torchzip.TorchArchive.open(path)
    g = arc.graph()
    assert np.array_equal(arc.tensor(g["weight"]), w)
    assert np.array_equal(arc.tensor(g["bias"]), b)


# -- puzzle 2 on a synthetic network -------------------------------------------------------------

def test_tensor_reader_rejects_layouts_outside_the_storage(tmp_path):
    path = tmp_path / "evil.pth"
    _fake_torch_file(path, {"weight": np.zeros(3, dtype=np.float32)})
    arc = torchzip.TorchArchive.open(path)
    g = arc.graph()
    t = g["weight"]
    (pid, offset, size, stride) = t.args[:4]
    for bad in [(pid, 0, (4096,), (1,)), (pid, 5, (1,), (1,)), (pid, -1, (2,), (1,)), (pid, 0, (2,), (-1,))]:
        t.args = bad
        with pytest.raises(ValueError):
            arc.tensor(t)


def _synthetic(n_blocks=5, dim=6, hidden=12, rows=400, seed=0):
    rng = np.random.default_rng(seed)
    pieces, true_blocks, idx = {}, [], 0
    for k in range(n_blocks):
        w_in = rng.normal(size=(hidden, dim)) / np.sqrt(dim)
        # trained blocks write back against what they read: make W_out ~ -W_in^T
        w_out = -0.6 * w_in.T + 0.1 * rng.normal(size=(dim, hidden))
        pieces[idx] = dropped_load.Piece(idx, w_in, rng.normal(size=hidden) * 0.1)
        pieces[idx + 1] = dropped_load.Piece(idx + 1, w_out * (1 + 0.3 * k), rng.normal(size=dim) * 0.1)
        true_blocks.append((idx, idx + 1))
        idx += 2
    pieces[idx] = dropped_load.Piece(idx, rng.normal(size=(1, dim)), np.array([0.1]))
    x = rng.normal(size=(rows, dim))
    puzzle = dropped_load.Puzzle(pieces, x, np.zeros(rows), np.zeros(rows))
    puzzle.pred = order.Model(puzzle, true_blocks).run(x, true_blocks)
    return puzzle, true_blocks


def test_pairing_finds_the_true_pairs():
    puzzle, true_blocks = _synthetic()
    p = pair.pair(puzzle)
    assert sorted(p.blocks) == sorted(true_blocks)
    assert p.margin > 0


@pytest.mark.parametrize("method", ["norm", "search"])
def test_ordering_recovers_the_true_order(method):
    puzzle, true_blocks = _synthetic()
    shuffled = list(true_blocks)
    random.Random(1).shuffle(shuffled)
    r = order.solve(puzzle, shuffled, method=method, rows=400, say=None)
    assert r.order == true_blocks
    assert r.max_abs_err < 1e-9
