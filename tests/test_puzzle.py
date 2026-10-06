"""End-to-end checks of every number quoted in the README. Skipped until `make fetch` has run."""

from __future__ import annotations

import hashlib

import pytest

from nnre import crosscheck, paths, pipeline
from nnre.hashnet import probe, quirks, readout

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
    assert out["cache_bytes"] < 1_000_000
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
    top = {r["name"].split()[0]: r["count"] for r in c["top"] if r["name"]}
    assert top["AND"] == 58524 and top["NOR"] == 34564 and top["NOT"] == 32196


def test_readout(ctx):
    out = pipeline.step_readout(ctx)
    assert out["target"] == "c7ef65233c40aa32c2b9ace37595fa7c"
    assert out["vegetable_dog"] == 0
    assert out["md5_matches"] == out["inputs"] == 200
    assert out["vegetable_dog_terms"][7] == (428, 410)  # 428 - 2*205 = 18 = MD5('vegetable dog')[7]
    assert out["vegetable_dog_terms"][0] == (171, 0)


def test_md5map(ctx):
    out = pipeline.step_md5map(ctx)
    assert out["steps_found"] == 64
    assert out["period"] == [42]
    assert out["step_done"][0] == 59 and out["step_done"][63] == 2705
    assert out["round_function_found"] == out["round_function_varying"] == 63
    assert out["live_steps"] == 31 and out["presum_max_single_neuron_bits"] <= 1
    lin = {int(k) - probe.step_start(5): v["linear_bits"] for k, v in out["linear_probe"].items()}
    assert lin == {27: 1, 28: 32}
    assert out["byte_splitters"] == 256 and out["splitters_per_step"] == [4]


def test_quirks(ctx):
    q = pipeline.step_quirks(ctx)
    assert q["by_length"]["longest_ok"] == 31 and q["by_length"]["shortest_bad"] == 32
    assert q["nul"]["network_md5"] == q["nul"]["strings"] == 300
    assert q["nul"]["md5_of_text"] == 0
    assert q["length_counter"] == [(1, 224)]
    assert q["eight_k"] == {5: 40, 31: 248, 32: 256, 40: 320, 55: 440}
    assert q["length_bits"][31] == [1, 1, 1, 1, 1]
    assert q["length_bits"][32] == [9, 1, 1, 1, 1]
    assert q["length_bits"][55] == [193, 1, 1, 1, 1]


def test_crack(ctx):
    out = pipeline.step_crack(ctx)
    assert out["answer"] == "bitter lesson"
    assert hashlib.md5(b"bitter lesson").hexdigest() == "c7ef65233c40aa32c2b9ace37595fa7c"
    assert out["network_output"] == 1.0
    assert out["shell"] == 4917


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
    assert -14 < lo < hi < -7
    assert round(out["pred_true_corr"], 3) == 0.940 and round(out["pred_true_mse"], 4) == 0.1065


def test_order(ctx):
    out = pipeline.step_order(ctx)
    assert out["answer"] == ANSWER_2
    assert out["sha256"] == "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"
    assert out["max_abs_err"] < 2e-6
    assert out["start_inversions"] == 48
    assert round(out["norm_depth_spearman"], 3) == 0.986
    sig = {k: round(v, 3) for k, v in out["depth_signals"].items()}
    assert sig["|block(x) - x| on the raw data"] == 0.957 and sig["|h| entering the block"] == 0.915
    assert max(sig, key=lambda k: abs(sig[k])) == "|W_out|"


@pytest.mark.skipif(not crosscheck.torch_available(), reason="torch not installed")
def test_crosscheck(ctx):
    out = pipeline.step_crosscheck(ctx)
    assert out["ok"]
    assert out["hashnet"]["torch_outputs"] == {"bitter lesson": 1.0}


def test_network_md5_explains_every_short_input(ctx):
    texts = quirks.random_with_nuls(100, seed=9) + probe.random_inputs(100, seed=9)
    got = readout.digests(ctx.net, texts, ctx.comparator)
    assert all(d == pipeline.md5.network_md5(t) for t, d in zip(texts, got))
