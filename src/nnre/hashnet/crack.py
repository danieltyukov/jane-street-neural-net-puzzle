"""Find the input whose MD5 is the network's target.

MD5 cannot be run backwards, and the network cannot be trained towards the answer (its output is
flat zero almost everywhere, so there is no gradient). What is left is guessing well. The puzzle page
pre-fills the input box with 'vegetable dog' and a code comment says ``# two words?``, so the search
space is pairs of English words separated by one space.

Words come from ``wordfreq`` (lowercase, letters only, most frequent first). Pairs are tried in
"shells": shell m holds every pair in which the rarer word has rank m, so after shell m every pair of
the top m+1 words has been tried. Common phrases are found early without deciding a cut-off in
advance.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

_WORDS: list[str] = []
_TARGET = b""


def wordlist(n: int = 30000) -> list[str]:
    from wordfreq import top_n_list
    seen, out = set(), []
    for w in top_n_list("en", n * 2):
        if w.isascii() and w.isalpha() and w.islower() and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) == n:
            break
    return out


def _init(words: list[str], target: bytes) -> None:
    global _WORDS, _TARGET
    _WORDS, _TARGET = words, target


def _shell(m: int) -> list[str]:
    """Every pair whose higher rank is m, in both orders."""
    hits = []
    w = _WORDS[m]
    head = hashlib.md5((w + " ").encode())
    for j in range(m + 1):
        h = head.copy()
        h.update(_WORDS[j].encode())
        if h.digest() == _TARGET:
            hits.append(f"{w} {_WORDS[j]}")
        if j != m:
            h = hashlib.md5((_WORDS[j] + " " + w).encode())
            if h.digest() == _TARGET:
                hits.append(f"{_WORDS[j]} {w}")
    return hits


@dataclass
class CrackResult:
    phrase: str | None
    shell: int | None
    hashes: int
    seconds: float
    words: int


def two_words(target: bytes, n_words: int = 30000, workers: int | None = None, say=print) -> CrackResult:
    words = wordlist(n_words)
    workers = workers or min(32, os.cpu_count() or 1)
    t = time.time()
    chunk = 64
    # fork starts workers instantly on Linux; elsewhere it is unsafe (macOS) or missing (Windows),
    # and spawn works at the cost of a few seconds while each worker re-imports the package.
    method = "fork" if sys.platform == "linux" else "spawn"
    with mp.get_context(method).Pool(workers, _init, (words, target)) as pool:
        for start in range(0, len(words), chunk):
            shells = range(start, min(start + chunk, len(words)))
            for m, hits in zip(shells, pool.map(_shell, shells)):
                if hits:
                    tried = (m + 1) ** 2
                    return CrackResult(hits[0], m, tried, time.time() - t, len(words))
            if say and start and start % 2048 == 0:
                say(f"  {start} words, {start * start / 1e6:.0f}M pairs, {time.time() - t:.0f}s")
    return CrackResult(None, None, len(words) ** 2, time.time() - t, len(words))
