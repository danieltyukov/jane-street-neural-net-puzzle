"""Command line entry point: ``nnre <step>`` or ``nnre all``."""

from __future__ import annotations

import argparse
import json
import sys

from . import pipeline

ORDER = ["fetch", "unpickle", "census", "readout", "md5map", "quirks", "crack", "pair", "order", "crosscheck"]

HELP = {
    "fetch": "download both puzzles from Hugging Face (pinned revisions, SHA-256 checked)",
    "unpickle": "puzzle 1: read model.pt without running it, decompile the forward wrapper",
    "census": "puzzle 1: integer weights, sparsity and the ReLU gate alphabet",
    "readout": "puzzle 1: the last two layers, the target hash, and the MD5 check",
    "md5map": "puzzle 1: locate all 64 MD5 steps inside the network",
    "quirks": "puzzle 1: NUL handling and the 32-character limit",
    "crack": "puzzle 1: find the two words with that MD5",
    "pair": "puzzle 2: match each inp layer with its out layer",
    "order": "puzzle 2: put the 48 blocks in order and check the answer's hash",
    "crosscheck": "both: compare against the real PyTorch modules (needs torch)",
    "figures": "draw the figures in docs/img",
    "all": "run every step and write build/results.json",
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nnre", description="Jane Street neural network puzzles toolkit")
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ORDER + ["figures", "all"]:
        p = sub.add_parser(name, help=HELP[name])
        if name in ("order", "all"):
            p.add_argument("--method", choices=["norm", "search"], default="norm",
                           help="norm: sort by |W_out| then swap (seconds); search: assume nothing (~10 min)")
    args = parser.parse_args(argv)
    ctx = pipeline.Context(method=getattr(args, "method", "norm"))

    if args.step == "figures":
        from . import figures
        figures.draw_all(ctx)
        return 0

    steps = ORDER if args.step == "all" else [args.step]
    results = {}
    failed = []
    for name in steps:
        ctx.say(f"\n== {name}: {HELP[name]}")
        out = pipeline.STEPS[name](ctx)
        results[name] = out
        if out.get("ok") is False:
            failed.append(name)
    if args.step == "all":
        ctx.build.mkdir(parents=True, exist_ok=True)
        path = ctx.build / "results.json"
        path.write_text(json.dumps(_jsonable(results), indent=1))
        (ctx.build / "results.log").write_text("\n".join(ctx.log) + "\n")
        ctx.say(f"\nwrote {path}")
        ctx.say(f"puzzle 1 answer: {results['crack'].get('answer')}")
        ctx.say(f"puzzle 2 answer: {results['order'].get('answer')}")
    if failed:
        ctx.say(f"checks failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
