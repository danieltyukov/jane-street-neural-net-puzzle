"""End-to-end checks of the numbers quoted in the README and the write-up. Skipped until `make fetch` has run."""

from __future__ import annotations

import hashlib
import random
import string

import pytest

from nnre import crosscheck, paths, pipeline
from nnre.hashnet import crack, probe, quirks, readout

pytestmark = pytest.mark.skipif(
    not (paths.MODEL_PT.exists() and paths.HISTORICAL_CSV.exists()),
    reason="puzzle files not downloaded (run `make fetch`)")

ANSWER_2 = ("43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,33,95,26,50,66,1,40,"
            "15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,"
            "31,36,13,7,59,52,68,47,84,63,74,90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85")


@pytest.fixture(scope="module")
def ctx():
    return pipeline.Context(log=[])


def test_unpickle(ctx):
    out = pipeline.step_unpickle(ctx)
    assert out["layers"] == 2721
    assert out["wrapper"] == "lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\\x00')))))"
    assert set(out["globals"]) == {
        "torch.nn.modules.container.Sequential", "torch.nn.modules.linear.Linear",
        "torch.nn.modules.activation.ReLU", "torch._utils._rebuild_parameter", "torch._utils._rebuild_tensor_v2",
        "torch.FloatStorage", "collections.OrderedDict", "__builtin__.set", "_codecs.encode",
        "cloudpickle.cloudpickle._make_function", "cloudpickle.cloudpickle._builtin_type",
        "cloudpickle.cloudpickle._function_setstate", "cloudpickle.cloudpickle.subimport"}
    assert round(out["cache_bytes"] / 1e6, 2) == 0.64
    assert ctx.net.weights[0].shape == (224, 55)
    assert set(out["piece_globals"]) == {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2",
                                         "torch.FloatStorage"}


def test_census(ctx):
    c = pipeline.step_census(ctx)
    assert c["neurons"] == 876285
    assert c["nonzero_weights"] == 1074709
    assert c["dense_weights"] == 288122268
    assert c["wires"] == 671750
    assert [int(v) for v in c["weight_values"]] == [-256, -128, -64, -32, -16, -8, -4, -2, -1,
                                                    1, 2, 4, 8, 16, 32, 64, 128]
    top = {r["name"]: r["count"] for r in c["top"] if r["name"]}
    assert top == {"wire  relu(a)": 671750, "AND   relu(a+b-1)": 58524, "NOR   relu(1-a-b)": 34564,
                   "NOT   relu(1-a)": 32196, "ANDN  relu(a-b)": 26536, "relu(a-2b)": 14016,
                   "relu(a+b-2c-1)": 9696, "relu(a+b)": 8832, "XOR-style relu(a+b-2c)": 4608,
                   "relu(1+a-b)": 4412, "constant 1": 1147, "constant 0": 1104, "relu(a-1)": 635}


def test_readout(ctx):
    out = pipeline.step_readout(ctx)
    assert out["target"] == "c7ef65233c40aa32c2b9ace37595fa7c"
    assert out["vegetable_dog"] == 0
    assert out["md5_matches"] == out["inputs"] == 200
    assert out["vegetable_dog_terms"][7] == (428, 410)  # 428 - 2*205 = 18 = MD5('vegetable dog')[7]
    assert out["vegetable_dog_terms"][0] == (171, 0)


def test_md5map(ctx):
    out = pipeline.step_md5map(ctx)
    assert out["signatures"] == 108714
    assert out["steps_found"] == 64
    assert out["period"] == [42]
    assert out["step_done"][0] == 59 and out["step_done"][63] == 2705
    # 18 layers before step 0, 64 steps of 42, 13 for the final feed-forward, 2 for the comparator
    assert out["step_done"][0] - 42 + 1 == 18 and ctx.net.depth - 1 - out["step_done"][63] - 2 == 13
    assert out["round_function_found"] == out["round_function_varying"] == 63
    assert out["f_offsets"] == [2] and out["b_first_offsets"] == [30] and out["b_last_offsets"] == [42]
    assert [tuple(r) for r in out["adder_rises"]] == [(31, 30, 28, 24, 16)]
    assert out["adder_floors"] == [192, 288]
    assert out["live_steps"] == 31 and out["presum_max_single_neuron_bits"] <= 1
    for step, a in out["anatomy"].items():
        assert a == {"a+f linear at +15": 32, "M+K neurons at +15": 32,
                     "sum linear at +27": 1, "sum linear at +28": 32}, step
    assert out["byte_splitters"] == 256 and out["splitters_per_step"] == [4]
    splits = probe.byte_splitters(ctx.net)
    assert sorted(set(splits))[:3] == [3, 17, 59] and 563 in splits


def test_quirks(ctx):
    q = pipeline.step_quirks(ctx)
    assert q["by_length"]["longest_ok"] == 31 and q["by_length"]["shortest_bad"] == 32
    assert q["nul"]["network_md5"] == q["nul"]["strings"] == 300
    assert q["nul"]["md5_of_text"] == 0
    assert q["length_counter"] == [(1, 224)]
    assert q["eight_k"] == {5: 40, 31: 248, 32: 256, 40: 320, 55: 440}
    assert q["length_bits"][31] == [1, 1, 1, 1, 1]
    assert q["length_bits"][32] == [9, 1, 1, 1, 1]
    assert q["length_bits"][40] == [73, 1, 1, 1, 1]
    assert q["length_bits"][55] == [193, 1, 1, 1, 1]
    # every length bit comes from step 14's splitting chain, not from a look-alike elsewhere
    assert all(563 <= layer <= 605 for layer, _ in q["length_bit_neurons"].values())
    # without the window, bit 7 matches a NOT gate at layer 104 that carries an equal value
    rng = random.Random(3)  # the same batch quirks.length_bit_neurons uses
    texts = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 31))) for _ in range(200)]
    index = probe.signature_index(ctx.net, texts)
    hit = index[probe.bit_signature(pipeline.np.array([8 * len(t) for t in texts], dtype=pipeline.np.uint64), 7)]
    assert hit.layer == 104 and ctx.net.describe(hit.layer, hit.neuron).endswith("+1)")


def test_marker_is_added_before_splitting(ctx):
    texts = ["ab\x00cd", "ab\x00c\x7f", "ab\x00c\x80", "ab\x00c\xe9"]
    got = readout.digests(ctx.net, texts, ctx.comparator)
    assert got[0] == pipeline.md5.network_md5(texts[0]) and got[1] == pipeline.md5.network_md5(texts[1])
    # 0x80 + 0x80 = 256 no longer fits the 8-bit splitter, so these are not any MD5 at all
    for t in texts[2:]:
        with pytest.raises(ValueError):
            pipeline.md5.network_md5(t)
    # the splitter for step 1's word (bytes 4..7) sees the byte plus 128
    acts = ctx.net.activations(texts, [16])[16]
    assert [int(v) for v in acts[164]] == [228, 255, 256, 361]


def test_crack(ctx):
    out = pipeline.step_crack(ctx)
    assert out["answer"] == "bitter lesson"
    assert hashlib.md5(b"bitter lesson").hexdigest() == "c7ef65233c40aa32c2b9ace37595fa7c"
    assert out["network_output"] == 1.0
    assert out["shell"] == 4917
    assert out["ok"] is True
    words = crack.wordlist(30000)
    assert (words.index("lesson"), words.index("bitter")) == (3112, 4917)


def test_only_the_answer_opens_the_network(ctx):
    near = ["bitter lesson", "bitter lessons", "Bitter lesson", "bitter  lesson", "bitter lesson\x00x",
            "better lesson", "vegetable dog", ""]
    assert ctx.net(near).tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_pair(ctx):
    out = pipeline.step_pair(ctx)
    assert len(out["blocks"]) == 48
    assert out["agrees_with_argmin"]
    assert out["margin"] > 5
    lo, hi = out["own_trace_range"]
    assert (round(lo, 2), round(hi, 2)) == (-13.49, -7.37)
    assert round(out["margin"], 2) == 5.79
    assert (out["negative_units"], out["units"]) == (4480, 4608)
    assert round(out["pred_true_corr"], 3) == 0.940 and round(out["pred_true_mse"], 4) == 0.1065
    assert round(float(ctx.pairing.scores.mean()), 2) == -0.25 and ctx.pairing.scores.size == 2304
    assert ctx.dropped.kind("last") == [85] and len(ctx.dropped.x) == 10000


def test_order(ctx):
    out = pipeline.step_order(ctx)
    assert out["answer"] == ANSWER_2
    assert out["sha256"] == "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"
    assert 1.5e-6 < out["max_abs_err"] < 1.7e-6
    assert out["start_inversions"] == 48 and out["evaluations"] == 330
    assert round(out["norm_depth_spearman"], 3) == 0.986
    sig = {k: round(v, 3) for k, v in out["depth_signals"].items()}
    assert sig == {"|W_out|": 0.986, "|block(x) - x| on the raw data": 0.957, "|h| entering the block": 0.915,
                   "mean of b_in": 0.881, "|W_in|": 0.853, "|b_in|": -0.302, "|b_out|": 0.25,
                   "trace(W_out W_in)": -0.249}
    r = ctx.ordering
    assert r.history[-1][1] < 1e-13 and r.history[-1][0] == 330  # float64 precision, about 1.6e-14
    before_last_move = [m for _, m, _ in r.history if m > 1e-10][-1]
    assert 1e-4 < before_last_move < 2e-4  # one swap takes it from ~1.6e-4 to float precision
    last_swap = [i for _, _, i in r.history if i >= 0][-1]
    assert last_swap == 3  # the blocks that belong at positions 3 and 4
    final = {b: i for i, b in enumerate(r.order)}
    start = pipeline.dropped_order.norm_start(ctx.dropped, ctx.pairing.blocks)
    assert final[start[42]] == 47  # the last block is sorted to position 42 and walked to the end


@pytest.mark.skipif(not crosscheck.torch_available(), reason="torch not installed")
def test_crosscheck(ctx):
    out = pipeline.step_crosscheck(ctx)
    assert out["ok"]
    assert out["hashnet"]["inputs"] == 64 and out["hashnet"]["comparator_width"] == 192
    assert out["hashnet"]["torch_outputs"] == {"bitter lesson": 1.0}
    # float32 rounding depends on the CPU's kernels: 4.8e-7 on one machine, 4.9e-7 on GitHub's runners
    assert out["dropped"]["max_abs_err_vs_pred"] < 1e-6 and out["dropped"]["rows"] == 10000


def test_network_md5_explains_every_short_input(ctx):
    texts = quirks.random_with_nuls(100, seed=9) + probe.random_inputs(100, seed=9)
    got = readout.digests(ctx.net, texts, ctx.comparator)
    assert all(d == pipeline.md5.network_md5(t) for t, d in zip(texts, got))
