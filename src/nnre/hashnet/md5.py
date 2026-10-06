"""A plain MD5 (RFC 1321) with a hook to record every step, and the padding rule the network uses.

Only single-block messages matter here: the network sees at most 55 bytes, which is exactly the
longest message that fits in one 64-byte block next to the 0x80 marker and the 8-byte length.
"""

from __future__ import annotations

import math
import struct

MASK = 0xFFFFFFFF
K = [int(abs(math.sin(i + 1)) * 2**32) & MASK for i in range(64)]
SHIFT = [7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 + [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4
IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)


def rotl(x: int, c: int) -> int:
    return ((x << c) | (x >> (32 - c))) & MASK


def message_index(i: int) -> int:
    """Which of the 16 message words step i reads."""
    return (i, (5 * i + 1) % 16, (3 * i + 5) % 16, (7 * i) % 16)[i // 16]


def round_function(i: int, b: int, c: int, d: int) -> int:
    r = i // 16
    if r == 0:
        return (b & c) | (~b & d)
    if r == 1:
        return (d & b) | (~d & c)
    if r == 2:
        return b ^ c ^ d
    return c ^ (b | (~d & MASK))


def compress(block: bytes, state=IV, trace: list | None = None) -> tuple[int, int, int, int]:
    """One MD5 compression. ``trace`` receives a dict per step with the intermediate words."""
    m = struct.unpack("<16I", block)
    a, b, c, d = state
    for i in range(64):
        f = round_function(i, b, c, d) & MASK
        s = (a + f + K[i] + m[message_index(i)]) & MASK
        a, d, c = d, c, b
        b = (b + rotl(s, SHIFT[i])) & MASK
        if trace is not None:
            trace.append({"f": f, "sum": s, "rot": rotl(s, SHIFT[i]), "b": b})
    return tuple((x + y) & MASK for x, y in zip(state, (a, b, c, d)))


def digest(state) -> bytes:
    return struct.pack("<4I", *state)


def pad(msg: bytes) -> bytes:
    """Standard MD5 padding for a message of at most 55 bytes."""
    if len(msg) > 55:
        raise ValueError("single-block MD5 only")
    return msg + b"\x80" + b"\x00" * (55 - len(msg)) + struct.pack("<Q", 8 * len(msg))


def md5(msg: bytes) -> bytes:
    return digest(compress(pad(msg)))


def network_block(text: str) -> bytes:
    """The block the network builds from its 55 input code points.

    It never looks for the end of the string. It counts the non-NUL characters, k, puts the 0x80
    marker at position k and writes 8k into the length field. Without embedded NULs that is the
    standard padding. Valid for k <= 31 (see quirks.py for what happens beyond).
    """
    x = [ord(c) for c in text[:55].ljust(55, "\x00")]
    k = sum(1 for c in x if c)
    block = bytearray(x + [0] * 9)
    block[k] |= 0x80
    block[56:64] = struct.pack("<Q", 8 * k)
    return bytes(block)


def network_md5(text: str) -> bytes:
    return digest(compress(network_block(text)))
