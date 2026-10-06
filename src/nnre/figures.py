"""Figures for the README and the write-up (written to docs/img)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import paths
from .hashnet import census, md5, probe, quirks, readout

OUT = paths.ROOT / "docs" / "img"

# Reference categorical palette (light mode), in its fixed order, plus chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32c2c"])


def _style() -> None:
    family = "Inter" if any("Inter" == f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


# -- puzzle 1 ------------------------------------------------------------------------------------

def md5_map(ctx) -> Path:
    net = ctx.net
    m = probe.md5_map(net)
    done = np.array(m["step_done"])
    widths = np.array(net.widths())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.6), gridspec_kw={"height_ratios": [1.1, 1]})
    ax1.scatter(done, np.arange(64), s=12, color=SERIES[0], zorder=3)
    ax1.set_xlim(0, net.depth)
    ax1.set_ylim(-2, 66)
    ax1.set_yticks([0, 16, 32, 48, 63])
    ax1.set_xlabel("layer")
    ax1.set_ylabel("MD5 step")
    ax1.set_title(f"Where each MD5 step's new 32-bit word appears: layer {done[0]} + 42 x step")
    for r, name in enumerate(["round 1: F", "round 2: G", "round 3: H", "round 4: I"]):
        ax1.text(done[16 * r] - 40, 16 * r + 4, name, color=INK_2, fontsize=8.5, ha="right")
    ax1.axvline(net.depth - 2, color=MUTED, lw=0.8)
    ax1.text(net.depth - 14, 2, "comparator:\nlast 2 layers", color=INK_2, fontsize=8.5, ha="right")

    i = 20
    start = probe.step_start(i)
    xs = np.arange(start, start + 43)
    ax2.step(xs - start, widths[xs], where="mid", color=SERIES[0], lw=2)
    top = widths[xs].max() + 45
    ax2.set_xlim(-0.5, 42.5)
    ax2.set_ylim(200, top)
    ax2.set_xlabel(f"layer offset inside one step (step {i} starts at layer {start})")
    ax2.set_ylabel("neurons in layer")
    ax2.set_title("One step: round function, combine the operands, then two Kogge-Stone carry adders")
    phases = [(0, 3, "F/G/H/I\nat +2"), (3, 16, "combine a, f, K, M"),
              (16, 28, "adder 1: a+f+K+M\nready at +28"), (28, 42, "adder 2: b + rotl(sum)\nnew b at +42")]
    for k, (a, b, label) in enumerate(phases):
        if k % 2:
            ax2.axvspan(a, b, color="#f0efec", zorder=0, lw=0)
        ax2.text((a + b) / 2, top - 6, label, ha="center", va="top", fontsize=8.5, color=INK_2)
    for x in xs:
        if widths[x] in (319, 318, 316, 312, 304):
            ax2.annotate(f"{widths[x]}", (x - start, widths[x]), textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=7, color=INK_2)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "md5_map.png")


def comparator(ctx) -> Path:
    cmp = ctx.comparator
    t = cmp.target[0]
    v = np.linspace(t - 4, t + 4, 801)
    relu = lambda z: np.maximum(z, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for k, (b, lab) in enumerate([(t + 1, f"relu(v - {t + 1})"), (t, f"-2 relu(v - {t})"),
                                  (t - 1, f"relu(v - {t - 1})")]):
        y = relu(v - b) * (-2 if b == t else 1)
        ax1.plot(v, y, color=[SERIES[1], SERIES[6], SERIES[2]][k], lw=1.2, label=lab)
    ax1.plot(v, readout.hat(v, t), color=SERIES[0], lw=2.4, label="sum: 1 only at v = %d" % t)
    ax1.set_xlim(t - 4, t + 4)
    ax1.set_ylim(-3.2, 4.2)
    ax1.set_xlabel("v (one byte of the value being checked)")
    ax1.set_title("Three ReLUs make an equality test")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.bar(np.arange(16), list(cmp.target), color=SERIES[0], width=0.8)
    ax2.set_xticks(range(16))
    ax2.set_xlabel("byte")
    ax2.set_ylim(0, 290)
    ax2.set_title("Hat centres = target bytes")
    for k, b in enumerate(cmp.target):
        ax2.text(k, b + 6, f"{b:02x}", ha="center", fontsize=7.5, color=INK_2, rotation=90)
    ax2.grid(axis="x", visible=False)
    fig.text(0.53, -0.04, f"target = {cmp.target_hex}", fontsize=9, color=INK_2)
    fig.tight_layout()
    return _save(fig, "comparator.png")


def census_chart(ctx) -> Path:
    c = census.census(ctx.net)
    rows = [r for r in c["top"] if r["name"]][:11][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(rows))
    shares = [r["share"] * 100 for r in rows]
    ax.barh(y, shares, color=SERIES[0], height=0.62)
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=9, family="monospace")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 200)
    ax.set_xlabel("share of all 876,285 neurons (%, log scale)")
    ax.set_title("The gate alphabet: what each ReLU neuron computes on 0/1 inputs")
    for yy, r in zip(y, rows):
        ax.text(r["share"] * 100 * 1.08, yy, f"{r['count']:,}", va="center", fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "census.png")


def length_bug(ctx) -> Path:
    net, cmp = ctx.net, ctx.comparator
    lengths = list(range(0, 56))
    texts = ["".join(chr(97 + (7 * n + 3 * k) % 26) for k in range(n)) for n in lengths]
    got = readout.digests(net, texts, cmp)
    ok = [d == hashlib.md5(t.encode()).digest() for t, d in zip(texts, got)]
    bits = quirks.length_bit_neurons(net)
    layer, neuron = bits[3]
    vals = net.activations(texts, [layer])[layer][neuron]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [0.55, 1.4]})
    first_bad = next(n for n, o in zip(lengths, ok) if not o)
    if not all(ok[:first_bad]) or any(ok[first_bad:]):
        raise ValueError("expected a single switch from correct to wrong")
    ax1.axvspan(-0.5, first_bad - 0.5, ymin=0.15, ymax=0.55, color=GOOD, lw=0)
    ax1.axvspan(first_bad - 0.5, lengths[-1] + 0.5, ymin=0.15, ymax=0.55, color=CRITICAL, lw=0)
    ax1.text((first_bad - 1) / 2, 0.78, f"correct MD5: 0 to {first_bad - 1} characters", ha="center", va="center",
             color=INK, fontsize=9)
    ax1.text((first_bad + lengths[-1]) / 2, 0.78, f"wrong: {first_bad} or more", ha="center", va="center",
             color=INK, fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_title("Does the network hash its input correctly?")
    ax1.grid(visible=False)
    ax1.spines["left"].set_visible(False)
    ax2.plot(lengths, vals, color=SERIES[0], lw=2, marker="o", ms=3.5)
    ax2.axhline(1, color=MUTED, lw=0.8)
    ax2.set_ylabel(f"neuron {layer}:{neuron}")
    ax2.set_xlabel("input length (characters)")
    ax2.set_title("The length field's 'bit 3' neuron: 0 or 1 until the count overflows 8 bits")
    ax2.annotate("8k = 256 does not fit in\nthe 128..1 decomposition", (32, vals[32]), xytext=(18, 120),
                 fontsize=8.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.tight_layout()
    return _save(fig, "length_bug.png")


# -- puzzle 2 ------------------------------------------------------------------------------------

def pairing(ctx) -> Path:
    p = ctx.pairing
    order = ctx.ordering.order
    rows = [p.inp.index(i) for i, _ in order]
    cols = [p.out.index(o) for _, o in order]
    s = p.scores[np.ix_(rows, cols)]
    own = np.diag(s)
    others = s[~np.eye(len(s), dtype=bool)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1, 1.05]})
    lim = np.abs(s).max()
    im = ax1.imshow(s, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax1.set_title("trace(W_out W_in) for every inp/out pair")
    ax1.set_xlabel("out layer (in final block order)")
    ax1.set_ylabel("inp layer (in final block order)")
    ax1.grid(visible=False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    bins = np.linspace(-14, 4, 46)
    ax2.hist(others, bins=bins, color=MUTED, alpha=0.9, label=f"other pairs ({len(others)})")
    ax2.hist(own, bins=bins, color=SERIES[0], label=f"true pairs ({len(own)})")
    ax2.set_yscale("log")
    ax2.set_xlabel("trace(W_out W_in)")
    ax2.set_ylabel("pairs (log scale)")
    ax2.set_title("True pairs are far below everything else")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "pairing.png")


def ordering(ctx) -> Path:
    pz = ctx.dropped
    r = ctx.ordering
    norms = [np.linalg.norm(pz.pieces[o].weight) for _, o in r.order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    ax1.plot(range(48), norms, color=SERIES[0], lw=1.2, marker="o", ms=4)
    ax1.set_xlabel("block position in the solved network")
    ax1.set_ylabel("|W_out| (Frobenius)")
    ax1.set_title("The out layer's norm grows with depth")
    hist = np.array(r.history)
    ax2.step(hist[:, 0], hist[:, 1], where="post", color=SERIES[0], lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("model evaluations")
    ax2.set_ylabel("mean (model - pred)^2")
    ax2.set_title("Sorted by norm, then neighbour swaps")
    ax2.annotate(f"exact: {hist[-1, 1]:.0e}", (hist[-1, 0], hist[-1, 1]), xytext=(-110, 30),
                 textcoords="offset points", fontsize=8.5, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.tight_layout()
    return _save(fig, "ordering.png")


def draw_all(ctx) -> list[Path]:
    _style()
    out = []
    for fn in (md5_map, comparator, census_chart, length_bug, pairing, ordering):
        path = fn(ctx)
        ctx.say(f"wrote {path.relative_to(paths.ROOT)}")
        out.append(path)
    return out
