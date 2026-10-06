"""Both solves, step by step. Each step prints a short report and returns plain data."""

from __future__ import annotations

import functools
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import crosscheck, fetch, paths
from .dropped import load as dropped_load
from .dropped import order as dropped_order
from .dropped import pair as dropped_pair
from .hashnet import census, crack, md5, model, probe, quirks, readout
from .torchzip import TorchArchive


@dataclass
class Context:
    build: Path = paths.BUILD
    log: list[str] = field(default_factory=list)
    method: str = "norm"

    def say(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.log.append(msg)

    @functools.cached_property
    def net(self) -> model.HashNet:
        self.build.mkdir(parents=True, exist_ok=True)
        return model.load(self.build / "hashnet.npz")

    @functools.cached_property
    def comparator(self) -> readout.Comparator:
        return readout.comparator(self.net)

    @functools.cached_property
    def dropped(self) -> dropped_load.Puzzle:
        return dropped_load.load()

    @functools.cached_property
    def pairing(self) -> dropped_pair.Pairing:
        return dropped_pair.pair(self.dropped)

    @functools.cached_property
    def ordering(self) -> dropped_order.OrderResult:
        return dropped_order.solve(self.dropped, self.pairing.blocks, self.method, say=self.say)

    @functools.cached_property
    def answer1(self) -> crack.CrackResult:
        return crack.two_words(self.comparator.target, say=self.say)


def _timed(fn):
    @functools.wraps(fn)
    def wrapper(ctx: Context, *a, **kw):
        t = time.time()
        out = fn(ctx, *a, **kw)
        out["seconds"] = round(time.time() - t, 2)
        return out
    return wrapper


# -- shared --------------------------------------------------------------------------------------

@_timed
def step_fetch(ctx: Context) -> dict:
    out = fetch.fetch_all(ctx.say)
    ctx.say(f"model.pt sha256 {out['model_sha256'][:16]}..., pieces zip sha256 {out['dropped_sha256'][:16]}...")
    return out


# -- puzzle 1 ------------------------------------------------------------------------------------

@_timed
def step_unpickle(ctx: Context) -> dict:
    arc = TorchArchive.open(paths.require(paths.MODEL_PT))
    refs = dict(arc.globals())
    ctx.say("model.pt asks the unpickler for:")
    for name in refs:
        ctx.say(f"  {name}")
    cache = ctx.build / "hashnet.npz"
    if cache.exists():
        cache.unlink()  # parse from scratch so this step really reads model.pt
    ctx.__dict__.pop("net", None)
    net = ctx.net
    ctx.say(f"{net.depth} Linear layers, each followed by ReLU; input width {net.weights[0].shape[1]}, "
            f"output width {net.weights[-1].shape[0]}")
    ctx.say(f"replaced _call_impl, decompiled from CPython 3.10 bytecode:\n  {net.wrapper}")
    ctx.say(f"sparse cache {cache.relative_to(paths.ROOT) if cache.is_relative_to(paths.ROOT) else cache}: "
            f"{cache.stat().st_size / 1e6:.2f} MB (model.pt is {paths.MODEL_PT.stat().st_size / 1e9:.2f} GB)")
    piece_globals = dict(TorchArchive.open(paths.require(paths.PIECES_DIR / "piece_0.pth")).globals())
    return {"globals": refs, "layers": net.depth, "wrapper": net.wrapper,
            "cache_bytes": cache.stat().st_size, "piece_globals": piece_globals}


@_timed
def step_census(ctx: Context) -> dict:
    c = census.census(ctx.net)
    ctx.say(f"{c['neurons']} neurons in {c['layers']} layers; {c['nonzero_weights']} non-zero weights out of "
            f"{c['dense_weights']} ({c['nonzero_weights'] / c['dense_weights']:.2%})")
    ctx.say(f"every weight is in {{{', '.join(str(int(v)) for v in c['weight_values'])}}}; every bias is an integer")
    ctx.say("most common neurons:")
    for row in c["top"][:10]:
        ctx.say(f"  {row['count']:7d}  {row['share']:6.2%}  {row['name'] or row['pattern']}")
    return c


@_timed
def step_readout(ctx: Context) -> dict:
    cmp = ctx.comparator
    ctx.say(f"last layer = AND of 16 'hats'; hat centres spell the target: {cmp.target_hex}")
    out_example = float(ctx.net(["vegetable dog"])[0])
    texts = probe.random_inputs(200, seed=2)
    got = readout.digests(ctx.net, texts, cmp)
    ok = sum(d == md5.md5(t.encode("latin-1")) for t, d in zip(texts, got))
    terms = readout.byte_terms(ctx.net, "vegetable dog", cmp)
    ctx.say(f"model('vegetable dog') = {out_example:g}; its byte 7 arrives as {terms[7][0]} - {terms[7][1]} = "
            f"{terms[7][0] - terms[7][1]} (carry-save), MD5 byte 7 is {md5.md5(b'vegetable dog')[7]}")
    ctx.say(f"value compared against the target equals MD5(input) for {ok}/{len(texts)} random inputs")
    return {"target": cmp.target_hex, "vegetable_dog": out_example, "md5_matches": ok, "inputs": len(texts),
            "vegetable_dog_terms": terms,
            "ok": ok == len(texts)}


@_timed
def step_md5map(ctx: Context) -> dict:
    m = probe.md5_map(ctx.net)
    steps_found = sum(s["b"].found == 32 for s in m["steps"])
    f_varying = [s for s in m["steps"] if s["f"].varying]
    f_found = sum(s["f"].found == s["f"].varying for s in f_varying)
    # Steps after the first whose message word varies across the probe inputs (words 0..7; for
    # inputs under 32 characters words 8..13 are always zero, which lets low bits collide).
    live = [s for s in m["steps"] if s["step"] > 0 and s["message_word"] < 8]
    most_sum_bits = max(s["sum"].found for s in live)
    ctx.say(f"signature index over {m['strings']} random inputs: {m['signatures']} distinct neuron signatures")
    ctx.say(f"all 32 bits of MD5's new word found for {steps_found}/64 steps; "
            f"step i is complete at layer {m['step_done'][0]} + 42*i (period {m['period']})")
    ctx.say(f"round function F/G/H/I output is a set of neurons in {f_found}/{len(f_varying)} steps where it varies")
    ctx.say(f"pre-rotation sum a+f+K+M: at most {most_sum_bits}/32 bits are single neurons, in all "
            f"{len(live)} steps that read a live message word")
    texts = probe.random_inputs(800, seed=7)
    start = probe.step_start(5)
    lin = probe.linear_probe(ctx.net, texts, probe.presum_values(texts, 5), [start + 27, start + 28])
    ctx.say("linear probe for step 5's pre-rotation sum: "
            + ", ".join(f"layer {layer} (+{layer - start}): {v['linear_bits']}/32 bits" for layer, v in lin.items()))
    sched = probe.message_schedule(ctx.net)
    splits = probe.byte_splitters(ctx.net)
    per_step = [sum(1 for layer in splits if (layer - 17) // 42 == i - 1) for i in range(64)]
    count = f"exactly {per_step[0]}" if min(per_step) == max(per_step) else f"{min(per_step)} to {max(per_step)}"
    ctx.say(f"{len(splits)} byte-to-bits chains, {count} in every step; each step's message word is split one "
            f"step ahead (step 0's at layer {splits[0]}, step 14's length field at layer {probe.step_start(14) - 42})")
    return {"step_done": m["step_done"], "period": m["period"], "steps_found": steps_found,
            "round_function_found": f_found, "round_function_varying": len(f_varying),
            "live_steps": len(live), "presum_max_single_neuron_bits": most_sum_bits,
            "linear_probe": {str(k): v for k, v in lin.items()},
            "message_first_layer": {g: v[0] for g, v in sched.items() if v},
            "byte_splitters": len(splits), "splitters_per_step": sorted(set(per_step)),
            "ok": steps_found == 64 and m["period"] == [42]}


@_timed
def step_quirks(ctx: Context) -> dict:
    q = quirks.summary(ctx.net)
    lens = q["by_length"]
    ctx.say(f"MD5 is right up to {lens['longest_ok']} characters and wrong from {lens['shortest_bad']} on")
    if q["length_counter"]:
        layer, n = q["length_counter"][0]
        ctx.say(f"layer {layer} neuron {n} computes 8 * (non-NUL characters): {q['eight_k']}")
    ctx.say(f"length-field bit neurons (bits 3..7 of 8k): {q['length_bit_neurons']}")
    for k, bits in q["length_bits"].items():
        ctx.say(f"  {k:2d} chars -> {bits}")
    nul = q["nul"]
    ctx.say(f"{nul['strings']} strings with NULs inside: network_md5 matches {nul['network_md5']}, "
            f"MD5 of the text matches {nul['md5_of_text']}")
    return {**q, "ok": nul["network_md5"] == nul["strings"] and lens["longest_ok"] == 31}


@_timed
def step_crack(ctx: Context) -> dict:
    r = ctx.answer1
    if r.phrase is None:
        ctx.say("no two-word phrase found")
        return {"ok": False}
    out = ctx.net([r.phrase, r.phrase.capitalize(), r.phrase + " "])
    ctx.say(f"MD5 '{r.phrase}' = {hashlib.md5(r.phrase.encode()).hexdigest()} after {r.hashes / 1e6:.1f}M "
            f"guesses ({r.words} wordfreq words, shell {r.shell}) in {r.seconds:.1f}s")
    ctx.say(f"model('{r.phrase}') = {out[0]:g}; capitalised {out[1]:g}; trailing space {out[2]:g}")
    return {"answer": r.phrase, "hashes": r.hashes, "shell": r.shell, "network_output": float(out[0]),
            "ok": out[0] == 1.0 and out[1] == 0.0}


# -- puzzle 2 ------------------------------------------------------------------------------------

@_timed
def step_pair(ctx: Context) -> dict:
    pz = ctx.dropped
    p = ctx.pairing
    own = [float(p.scores[p.inp.index(i), p.out.index(o)]) for i, o in p.blocks]
    corr = float(np.corrcoef(pz.pred, pz.true)[0, 1])
    mse = float(((pz.pred - pz.true) ** 2).mean())
    ctx.say(f"{len(pz.kind('inp'))} inp layers (96x48), {len(pz.kind('out'))} out layers (48x96), "
            f"last layer piece {pz.kind('last')[0]}; {len(pz.x)} rows of data")
    ctx.say(f"the original model's pred vs true: correlation {corr:.3f}, MSE {mse:.4f}")
    ctx.say(f"trace(W_out W_in): true pairs {min(own):.2f} .. {max(own):.2f}, all pairs mean {p.scores.mean():.2f}; "
            f"smallest margin {p.margin:.2f}; row minimum agrees with the optimal matching: {p.agrees_with_argmin}")
    return {"blocks": p.blocks, "own_trace_range": [min(own), max(own)], "margin": p.margin,
            "pred_true_corr": corr, "pred_true_mse": mse,
            "agrees_with_argmin": p.agrees_with_argmin, "ok": p.agrees_with_argmin and p.margin > 1}


@_timed
def step_order(ctx: Context) -> dict:
    r = ctx.ordering
    signals = dropped_order.depth_signals(ctx.dropped, r.order)
    ctx.say("Spearman correlation with position in the solved network:")
    for name, v in sorted(signals.items(), key=lambda kv: -abs(kv[1])):
        ctx.say(f"  {v:+.3f}  {name}")
    ctx.say(f"method '{ctx.method}': start order had {r.start_inversions} inversions; "
            f"{r.evaluations} evaluations in {r.seconds:.1f}s")
    ctx.say(f"max |model - pred| over all {len(ctx.dropped.x)} rows: {r.max_abs_err:.2e}; "
            f"model MSE against 'true': {r.mse_true:.4f}")
    ctx.say(f"answer: {r.answer}")
    ctx.say(f"sha256 {r.sha256}  matches the puzzle's checker: {r.matches_checker}")
    return {"answer": r.answer, "sha256": r.sha256, "matches_checker": r.matches_checker,
            "max_abs_err": r.max_abs_err, "mse_true": r.mse_true, "evaluations": r.evaluations,
            "start_inversions": r.start_inversions, "norm_depth_spearman": signals["|W_out|"], "depth_signals": signals, "ok": r.matches_checker}


# -- both ----------------------------------------------------------------------------------------

@_timed
def step_crosscheck(ctx: Context) -> dict:
    if not crosscheck.torch_available():
        ctx.say("torch not installed; skipped (`make torch` installs the CPU build)")
        return {"skipped": True}
    texts = ["vegetable dog", ctx.answer1.phrase or "", "Bitter lesson"] + probe.random_inputs(61, 11, 55)
    h = crosscheck.hashnet(ctx.net, texts)
    ctx.say(f"puzzle 1: torch Sequential.forward vs numpy on {h['inputs']} inputs: outputs equal "
            f"{h['outputs_equal']}, comparator inputs equal {h['comparator_inputs_equal']}; non-zero: "
            f"{h['torch_outputs']}")
    d = crosscheck.dropped(ctx.dropped, ctx.ordering.permutation)
    ctx.say(f"puzzle 2: the published Block/LastLayer classes with the answer reproduce 'pred' to "
            f"{d['max_abs_err_vs_pred']:.1e} (float32)")
    return {"hashnet": h, "dropped": d,
            "ok": h["outputs_equal"] and h["comparator_inputs_equal"] and d["max_abs_err_vs_pred"] < 1e-5}


STEPS = {
    "fetch": step_fetch,
    "unpickle": step_unpickle,
    "census": step_census,
    "readout": step_readout,
    "md5map": step_md5map,
    "quirks": step_quirks,
    "crack": step_crack,
    "pair": step_pair,
    "order": step_order,
    "crosscheck": step_crosscheck,
}
