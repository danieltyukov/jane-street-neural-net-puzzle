# Write-up: from a pickle to `bitter lesson`, and from 97 pieces to one network

This is the long version of the README. It follows the solves in the order they actually happened,
wrong turns included, with the details that matter if you want to build the same tools yourself.
Every number here comes out of `make all`; `build/results.log` has the raw output. Spoilers start
right away.

Contents

Part I: the burial mound network

1. [What you are given](#1-what-you-are-given)
2. [Opening a pickle without trusting it](#2-opening-a-pickle-without-trusting-it)
3. [The forward wrapper](#3-the-forward-wrapper)
4. [First look at the weights](#4-first-look-at-the-weights)
5. [The last two layers](#5-the-last-two-layers)
6. [The gate alphabet](#6-the-gate-alphabet)
7. [Finding MD5 inside](#7-finding-md5-inside)
8. [Where it stops being MD5](#8-where-it-stops-being-md5)
9. [Cracking the hash](#9-cracking-the-hash)

Part II: I dropped a neural net

10. [What you are given](#10-what-you-are-given)
11. [Pairing the layers](#11-pairing-the-layers)
12. [Ordering the blocks](#12-ordering-the-blocks)
13. [Checking the answer](#13-checking-the-answer)

14. [Other solvers](#14-other-solvers)
15. [Things worth knowing before you start your own](#15-things-worth-knowing-before-you-start-your-own)

---

# Part I: the burial mound network

## 1. What you are given

The [puzzle page](https://huggingface.co/spaces/jane-street/puzzle) is a Gradio app with one text box,
pre-filled with `vegetable dog`, and one number, the model's output: 0. The text around it:

> Today I went on a hike and found a pile of tensors hidden underneath a neolithic burial mound! [...]
> Maybe start by looking at the last two layers.

The app's source has a comment next to the text box: `# two words?`. The model lives in the
[jane-street/2025-03-10](https://huggingface.co/jane-street/2025-03-10) repo as `model.pt` (for
Python up to 3.10) and `model_3_11.pt` (for 3.11 and later), 1.16 GB each, and the app loads it
with `torch.load(..., weights_only=False)` and calls `model(text)`.

That last detail matters. A PyTorch module normally takes a tensor, not a string, so something in
the file changes how the model is called.

## 2. Opening a pickle without trusting it

`torch.save` writes a zip archive. `puzzle/data.pkl` is a pickle that rebuilds the object graph, and
`puzzle/data/<n>` holds the raw bytes of tensor storage `n`. A pickle is a small stack-machine
program. Most of its opcodes build lists, dicts and tuples, but `GLOBAL` imports any `module.name`
and `REDUCE` calls it with arguments. That is how pickles rebuild arbitrary classes, and it is also
why loading one runs whatever code it names.

Since every callable has to be imported with `GLOBAL` (or `STACK_GLOBAL`) first, listing those
opcodes gives the complete set of code the file could run. For `model.pt`:

```
torch.nn.modules.container.Sequential     torch.nn.modules.linear.Linear
torch.nn.modules.activation.ReLU          torch._utils._rebuild_parameter
torch._utils._rebuild_tensor_v2           torch.FloatStorage
collections.OrderedDict                   __builtin__.set
_codecs.encode
cloudpickle.cloudpickle._make_function    cloudpickle.cloudpickle._builtin_type
cloudpickle.cloudpickle._function_setstate  cloudpickle.cloudpickle.subimport
```

Nothing alarming, but the last four are interesting: they rebuild a Python *function*, with its
bytecode, from the file.

To read the rest, [`torchzip.py`](../src/nnre/torchzip.py) runs the pickle with a custom
`find_class`. Instead of importing anything, it hands back a freshly made placeholder class named
after the global. Calling a placeholder records the arguments; `BUILD` records the state. The result
is the full object graph made of placeholders and plain containers, and no code from the file has
run. A tensor shows up as a `_rebuild_tensor_v2` placeholder whose arguments are
`(storage key, offset, shape, stride)`, which is enough to read it from the zip with numpy.

The graph is a `Sequential` with 5442 children, alternating `Linear` and `ReLU`, and an extra
instance attribute: `_call_impl`, set to the cloudpickled function. `nn.Module.__call__` looks up
`self._call_impl`, so this attribute replaces what `model(x)` does.

`tests/test_units.py` checks the reader against a pickle that calls `os.system`: the placeholder
records the call and nothing runs.

## 3. The forward wrapper

cloudpickle stores a function as `_make_function(code, globals, name, defaults, closure)`, where
`code` is `_builtin_type('CodeType')(...)` with the 16 fields of a CPython 3.10 code object. The
interesting fields:

```
co_code    74 00 a0 01 74 02 a0 03 74 04 74 05 74 06 74 07 7c 00 83 01 64 00 64 01 85 02 19 00
           a0 08 64 01 64 02 a1 02 83 02 83 01 a1 01 a1 01 53 00
co_consts  (None, 55, '\x00')
co_names   ('model', 'forward', 'torch', 'Tensor', 'list', 'map', 'ord', 'str', 'ljust')
co_varnames ('x',)
co_name    '<lambda>'
```

In 3.10 every instruction is two bytes, opcode then argument. `0x74` is `LOAD_GLOBAL`, `0xa0`
`LOAD_METHOD`, `0x7c` `LOAD_FAST`, `0x83` `CALL_FUNCTION`, `0x64` `LOAD_CONST`, `0x85`
`BUILD_SLICE`, `0x19` `BINARY_SUBSCR`, `0xa1` `CALL_METHOD`, `0x53` `RETURN_VALUE`. A newer
Python's `dis` cannot decode this (the opcode table changed in 3.11), so
[`pybytecode.py`](../src/nnre/pybytecode.py) carries those numbers and runs the instructions on a
stack of strings instead of values:

```python
lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00')))))
```

The model sees the code points of the first 55 characters, padded with NULs to exactly 55 floats.

This also explains the two files. Bytecode is version-specific and the code object constructor
gained fields in 3.11, so a function pickled by 3.10 cannot be rebuilt by 3.11. The 5442 tensor
storages in the two files are byte-identical.

## 4. First look at the weights

Reading every storage gives 2721 weight matrices and bias vectors. Layer 0 is 224 x 55, the last is
1 x 48, and the widths in between hover around 300. Three things stand out at once:

- Every weight is 0 or a signed power of two: -256, -128, ..., -1, 1, 2, ..., 128.
- Every bias is an integer.
- 99.63% of the weights are zero: 1,074,709 non-zero out of 288,122,268.

Nobody trains a network like that. It was written by hand, or compiled from something. Stored as
sparse matrices it takes 0.64 MB instead of 1.16 GB, and a batch of a few hundred inputs runs in
about a second with scipy (`hashnet/model.py`). Because every value is an integer far below 2^53,
float64 arithmetic is exact, so this simulator is not an approximation.

`vegetable dog` gives 0, as on the website.

## 5. The last two layers

Following the hint, the last layer is a single neuron:

```
L2720:0 = relu(h0 + ... + h15 - 2*(h16 + ... + h31) + h32 + ... + h47 - 15)
```

and the 48 neurons before it come in 16 triples. Triple `k` is neurons `k`, `k+16` and `k+32`.
All three read the same weighted sum `v_k` of the previous layer, with biases `-(t+1)`, `-t` and
`-(t-1)` for some integer `t`. So the last layer sees, per byte,

```
relu(v - t - 1) - 2*relu(v - t) + relu(v - t + 1)
```

For an integer `v` this is 0 everywhere except at `v = t`, where it is 1: a "hat". Adding 16 hats,
subtracting 15 and applying ReLU gives 1 only if all 16 hats are 1. The network compares 16 bytes
against 16 constants, and the constants are the hat centres:

```
c7 ef 65 23 3c 40 aa 32 c2 b9 ac e3 75 95 fa 7c
```

Sixteen bytes is the size of an MD5 digest, and 55 characters is the longest MD5 message that fits
in one block. The comparator's input is easy to read too: `v_k` is a fixed linear combination of
the previous layer, so [`readout.py`](../src/nnre/hashnet/readout.py) evaluates it for any input.
For 200 random inputs it equals `MD5(input)` every time.

Eight of the bytes have a twist. Bytes 0-3 and 8-11 are plain: weights 1, 2, 4, ..., 128 on eight
0/1 neurons. Bytes 4-7 and 12-15 also subtract 2, 4, ..., 256 times eight more neurons, and some of
the "bits" they add hold the value 2. That is *carry-save* form: a sum whose carries have not been
propagated yet. For `vegetable dog`, byte 7 arrives as 428 - 2 x 205 = 18, and MD5 byte 7 of
`vegetable dog` is `0x12` = 18. The comparator does the last carry propagation for free, because
it only ever needs the weighted sum.

## 6. The gate alphabet

Grouping all 876,285 neurons by their exact (weights, bias) pattern
([`census.py`](../src/nnre/hashnet/census.py)) shows how few building blocks there are. With 0/1
inputs:

| Neurons | Share | Pattern | Meaning |
|---:|---:|---|---|
| 671,750 | 76.7% | `relu(a)` | wire: copy a value to the next layer |
| 58,524 | 6.7% | `relu(a+b-1)` | AND |
| 34,564 | 3.9% | `relu(1-a-b)` | NOR |
| 32,196 | 3.7% | `relu(1-a)` | NOT |
| 26,536 | 3.0% | `relu(a-b)` | a AND NOT b |
| 14,016 | 1.6% | `relu(a-2b)` | |
| 9,696 | 1.1% | `relu(a+b-2c-1)` | |
| 8,832 | 1.0% | `relu(a+b)` | a + b, 0 to 2 |
| 4,608 | 0.5% | `relu(a+b-2c)` | XOR when c = AND(a, b) |

Three-quarters of the network is wires. A ReLU network has no skip connections, so any value needed
ten layers later has to be copied through ten neurons. Jane Street's write-up quotes a solver who
noticed about 80% of the nodes doing nothing; this is where that comes from.

XOR deserves a note. `a XOR b = a + b - 2*AND(a, b)` is *linear* in `a`, `b` and `AND(a, b)`. So
the network does not need a neuron that holds XOR: it keeps `a`, `b` and the AND, and the next layer
uses weights 1, 1, -2. That one trick explains the `-2` weights all over the network, and it matters
a lot in the next section.

The step functions used by the length logic are worth knowing too. For an integer `x`,
`relu(x - t + 1) - relu(x - t)` is 1 if `x >= t` and 0 otherwise, which turns a count into a
comparison with two neurons.

## 7. Finding MD5 inside

The comparator says "this is MD5". The next question is where MD5 lives in the 2719 layers before it.

### Signature probing

Run 160 random inputs of up to 31 characters at once. Each neuron's *signature* is its vector of
160 values. Index the first neuron with each distinct signature (108,714 of them that are not
constant). Then compute MD5 in Python with a hook that records every intermediate word
([`md5.py`](../src/nnre/hashnet/md5.py)), and look up the signature of every bit of every word.

If a bit that looks random across the inputs matches a neuron on all 160, the neuron carries that
bit. ([`probe.py`](../src/nnre/hashnet/probe.py))

### Results

- **All 64 steps.** For every MD5 step, all 32 bits of the new word `b` are found. Step `i` is
  complete at layer `59 + 42*i`, with no exceptions: the network is 64 copies of a 42-layer block,
  plus 17 layers of preparation before step 0, 13 layers after step 63 that add the initial state
  back in (MD5's final feed-forward), and the 2-layer comparator.
- **The round function.** F, G, H or I appears as 32 neurons 2 layers into every step where it
  varies (63 of 64; in step 0 it is a constant because the inputs are the IV).
- **Two Kogge-Stone adders per step.** The layer widths inside a step (figure in the README) rise
  above a floor of 288 by 31, 30, 28, 24 and 16, twice. A Kogge-Stone adder computes all carries of
  a 32-bit addition in log2(32) = 5 levels, and level `k` combines `32 - 2^k` pairs of
  (generate, propagate) signals: 31, 30, 28, 24, 16. Each level takes two layers. The first holds
  the ANDs (62 of them at level 0: a new generate and a new propagate for each of the 31 pairs), the
  second holds NORs, an OR stored inverted, which the next level's NOT and AND-NOT gates absorb. The
  first adder computes `a + f + K[i] + M[g]`, the second adds the rotated result to `b`.
- **The new word appears all at once.** Bit 0 of the new `b` shows up at offset 30 (it has no
  carry-in), the other 31 bits at offset 42, the end of the second adder.

### The sum that is nowhere

One intermediate refused to show up: the 32-bit sum `a + f + K[i] + M[g]` before the rotation. In
all 31 steps that read a varying message word, at most 1 of its 32 bits matches any single neuron.
Yet the next adder clearly uses it.

The XOR trick explains it. The sum's bits are XORs of propagate signals and carries, and the network
never stores an XOR. It stores the pieces, and the weights of the next layer combine them. So ask a
different question: is each bit a *linear combination* of a layer's neurons? With 800 random inputs
(more than the ~300 neurons per layer, so the answer is not trivially yes) and least squares:

| Layer offset in step 5 | Bits that are exactly linear in that layer |
|---:|---:|
| +27 | 1 / 32 |
| +28 | 32 / 32 |

At the end of the first adder, every bit of the sum is an exact linear function of the layer, while
almost none is a neuron. Rotating it costs nothing: it is just which of these combinations feeds
which input of the second adder.

This is the same lesson the interpretability literature keeps repeating about learned networks:
features live in directions, not necessarily in neurons. Here it can be checked exactly, because the
network is a circuit.

### The message words

The 16 message words are never stored as 32 neurons for long. The input bytes travel on wires, and
each time a step needs `M[g]`, its four bytes are split into bits again by "subtract 128 if at least
128, subtract 64 if at least 64, ..." chains. There are 256 such chains, exactly 4 in every one of
the 64 steps, and each runs one step ahead: step 0's word is split at layer 3, step 1's at layer 17,
step 2's at layer 59, and so on in parallel with the previous step's adders. Four bytes on wires are
cheaper to carry than 32 bits. The length field is no exception: step 14 reads word 14, and its
chain at layer 563 is the one that overflows in section 8.

### A trap

Signatures identify a value, not a meaning. Two quantities that are equal on every input share a
signature. The first version of this analysis looked for `M[g] + K[i]` and "found" step 17's value
twelve steps early. It was a collision: the low bit of `M + K` is the low bit of `M`, flipped or not
depending on the low bit of `K`, so `M[6] + K[17]` and `M[6] + K[6]` share their low bits whenever
the constants do. The fix is to only trust matches for bits that genuinely differ between the
candidates, which is why the README sticks to `b`, `f` and the message words.

## 8. Where it stops being MD5

Jane Street's write-up mentions that a solver found the network mishandles inputs over 32 bytes.
Measuring it: the comparator's value equals `MD5(input)` for every length from 0 to 31 and for no
length from 32 to 55.

### The first guess was wrong

MD5 puts the message length in bits, `8n`, in the last 8 bytes of the block. For `n < 32` it fits in
one byte, for `n >= 32` it does not. So the obvious guess is that the network only writes the low
byte of the length. Testing it with a hand-made block (`compress()` in `md5.py` takes any 64 bytes):
no match for any length. Also not `8n mod 256`.

### Reading the network's own block

Instead of guessing, read the network. The signature probe locates the neurons that hold each bit of
message word 14 (the length field). For a 31-character input they read `[1, 1, 1, 1, 1]` for bits 3
to 7, which is 248 = 8 x 31. For longer inputs:

| Characters | Bits 3..7 of the length field |
|---:|---|
| 31 | 1, 1, 1, 1, 1 |
| 32 | **9**, 1, 1, 1, 1 |
| 40 | **73**, 1, 1, 1, 1 |
| 55 | **193**, 1, 1, 1, 1 |

A "bit" holding 9 is the tell. Tracing backwards with `net.describe(layer, neuron)`:

1. Layer 0 makes, for each of the 55 inputs, a copy `x` and `relu(x - 1)`. Their difference is 1 for
   any non-zero code point.
2. Layer 1 neuron 224 adds up 8 times those differences: **8k, where k is the number of non-NUL
   characters**.
3. Around layer 563 that value is split into bits with the same subtract-if-at-least chain as the
   message bytes, starting at 128.

A chain that starts at 128 can only represent values up to 255. For 32 characters, `8k = 256`: the
128 stage subtracts 128, the 64 stage subtracts 64, and so on, and the last stage is left holding
`256 - 248 = 8` plus its own bit. The overflow lands in "bit 3" and grows by 8 per extra character.
After that the hash is not a hash of anything.

### NULs

Step 2 has a second consequence: the network never looks for the end of the string. It counts the
non-NUL characters `k`, puts the `0x80` padding marker at position `k` and writes `8k` into the
length. Without NULs inside the string, `k` is the length and this is standard MD5. With NULs
inside, it is something else: for `ab\0cd`, `k = 4`, so the marker lands on top of the `d`.
`md5.network_md5` implements exactly this rule and matches the network on 300 random strings with
embedded NULs; plain MD5 of the same strings matches none.

(Code points above 255 are a third way to leave MD5, since the byte-splitting chain assumes values
below 256. The original wrapper uses `ord`, not UTF-8, so `é` is one byte, 0xE9, and anything
outside Latin-1 is out of range.)

## 9. Cracking the hash

MD5 has no known preimage attack, and the network is no help as an oracle either: its output is 0
for every wrong input, so there is no gradient to follow. What is left is guessing well.

The hints are the default input `vegetable dog` and `# two words?`. A first attempt with the
popular [google-10000-english](https://github.com/first20hours/google-10000-english) list (10^8 pairs,
a few seconds in parallel) found nothing. The `/usr/share/dict/american-english` word list (63,875 lowercase words, 4 x 10^9
pairs) did, after five and a half minutes on 20 cores:

```
MD5("bitter lesson") = c7ef65233c40aa32c2b9ace37595fa7c
```

"bitter" is not in the Google 10,000 list at all, which is why the first attempt failed.

For the repository, the search had to be fast and should not know the answer. [`crack.py`](../src/nnre/hashnet/crack.py)
takes 30,000 lowercase words from [wordfreq](https://github.com/rspeer/wordfreq) in frequency order
and tries pairs in *shells*: shell `m` is every pair whose rarer word has rank `m`, in both orders.
After shell `m`, every pair of the top `m + 1` words has been tried, so common phrases come first
without choosing a cut-off. "lesson" is word 3112 of the list and "bitter" word 4917 (counting from
0), so the phrase turns up in shell 4917, after 24.2 million guesses, in a few seconds.

The network agrees: `model("bitter lesson") = 1`, while `Bitter lesson`, `bitter lessons`,
`bitter  lesson` and `bitter lesson ` all give 0.

The answer is a joke at the solver's expense. Rich Sutton's essay
[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) argues that general
methods that scale with compute beat hand-built human knowledge. This network is entirely hand-built
knowledge, and the general method, gradient descent, cannot touch it.

---

# Part II: I dropped a neural net

## 10. What you are given

The [second puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet):

> Oh no! I dropped an extremely valuable trading model and it fell apart into linear layers! [...]
> All I have left are the pieces of the model and some historical data.

The page gives the code of the two layer types:

```python
class Block(nn.Module):          # x + out(relu(inp(x)))
    def __init__(self, in_dim, hidden_dim):
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

class LastLayer(nn.Module):      # a single Linear
    ...
```

and asks for a permutation: for each position 0 to 96, which piece goes there. The download has 97
files `piece_<i>.pth`, each the `state_dict` of one `nn.Linear` (saved from `cuda:0`, so load them
with `map_location`), and `historical_data.csv` with 10,000 rows: `measurement_0` to `measurement_47`,
`pred` and `true`.

The shapes split the pieces into 48 `inp` layers (96 x 48), 48 `out` layers (48 x 96) and one final
layer (1 x 48, piece 85). So the network is 48 residual blocks of width 48 with 96 hidden units,
then a linear readout. `pred` is the original model's output (correlation 0.940 with `true`, MSE
0.1065), which turns out to be the key column.

Two sub-problems: which `inp` goes with which `out` (48! ways), and in which order the blocks go
(48! more).

The page's checker does not compare the answer, it compares its SHA-256 with
`093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4`. That hash is a perfect local
test: a wrong answer can never match it.

## 11. Pairing the layers

Every `inp` composes with every `out`, so shapes say nothing. Weights do. Hidden unit `k` of a block
reads the residual stream along row `k` of `W_in` and writes back along column `k` of `W_out`. The
two vectors were trained together, as one unit. Now

```
trace(W_out @ W_in) = sum over k of  (column k of W_out) . (row k of W_in)
```

is the sum over all 96 units of "how much the unit writes along the direction it reads". For a
wrong pair, unit `k` of one block has nothing to do with unit `k` of another; the 96 terms have
random signs and cancel. For a true pair they add up.

In this model every true pair has a clearly *negative* trace:

- true pairs: -13.49 to -7.37
- all 2304 pairs: mean -0.25
- the smallest gap between a true pair and any rival in its row or column: 5.79

Each unit pushes back against the direction it detects, so each block damps the features it
reads. The Hungarian algorithm on this matrix gives the matching, and so does simply taking each
row's minimum. Hyunwoo Park's [paper on this puzzle](https://arxiv.org/abs/2602.19845) uses the same
negative-diagonal structure and relates it to stability conditions during training.

## 12. Ordering the blocks

### The objective

`pred` is what the original model output. The correct order reproduces it to float precision; any
other order does not. So the mean squared difference between the reassembled model and `pred` is an
objective whose optimum is known to be (almost) zero. Evaluating it on 1000 of the rows takes under
20 milliseconds.

### What was tried first

Greedy: start from the raw input and repeatedly append the block that brings the model's output
(through the final layer) closest to `pred`. MSE 0.85, not close: a block that helps most right now
is not the one that belongs next.

Insertion search: take one block out, try it in all 48 positions, keep the best, repeat for every
block, and loop until a full pass changes nothing. From the greedy start, on 1000 rows:

| Pass | MSE vs `pred` |
|---:|---:|
| start | 0.85 |
| 1 | 0.40 |
| 2 | 0.067 |
| 3 | 0.0095 |
| 4 | 0.0013 |
| 5 | 5.0e-4 |
| 6 | 3.8e-4 |
| 7 | 6.9e-5 |
| 8 | 1.6e-14 |

The last pass jumps straight to zero: the final misplaced block snaps into position and everything
fits. This found the answer in about 13 minutes, and it is still available as
`nnre order --method search`.

### A shortcut, found afterwards

With the true order known, it is easy to ask which cheap statistic would have predicted it.
Spearman correlation with the true position:

| Statistic | Spearman |
|---|---:|
| Frobenius norm of `W_out` | +0.986 |
| mean norm of `block(x) - x` on the raw data | +0.957 |
| mean norm of the residual stream entering the block | +0.915 |
| mean of `b_in` | +0.881 |
| Frobenius norm of `W_in` | +0.853 |
| norm of `b_in` | -0.302 |
| norm of `b_out` | +0.250 |
| `trace(W_out W_in)` | -0.249 |

The `out` layers get steadily larger with depth, plausibly because later blocks write into a
residual stream that has already grown, and need bigger updates to matter. Sorting by `|W_out|`
gives an order with only 48 inversions (pairs in the wrong relative order). Swapping neighbours
whenever that lowers the error then needs 330 evaluations and a few seconds. The biggest outlier is
the very last block, whose `W_out` is smaller than the five before it, so sorting puts it at
position 42; the swaps walk it to the end. The error then sits around 1e-4 until the last
misplaced pair, the blocks that belong at positions 3 and 4, trades places, and drops to 1.6e-14 in
that one move.

The norm idea came from looking at the answer, so it is fair to call it hindsight. The search method
shows the problem is solvable without it. That said, `|block(x) - x|` on the raw data (0.957) needs
nothing but the pieces and the inputs, and would have been a reasonable first thing to try.

## 13. Checking the answer

The permutation lists, for each block in order, its `inp` piece then its `out` piece, then piece 85:

```
43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,33,95,26,50,66,1,40,15,
67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,
31,36,13,7,59,52,68,47,84,63,74,90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85
```

(one line, without the breaks, when you submit it). Three independent checks:

- Its SHA-256 is the checker's `093be1cf...`.
- The reassembled network in float64 matches `pred` to 1.6e-6 on all 10,000 rows, not just the
  1000 used for the search. Its MSE against `true` is 0.1065, the same as `pred`'s.
- The `Block` and `LastLayer` classes from the puzzle page, loaded with `torch.load(weights_only=True)`
  and run in float32, match `pred` to 4.8e-7. The CSV stores `pred` as float32, and the remaining
  difference is a few float32 rounding steps: the pieces were saved from a GPU, which adds things up
  in a different order.

## 14. Other solvers

- Jane Street's [write-up of puzzle 1](https://blog.janestreet.com/can-you-reverse-engineer-our-neural-network/)
  describes how one solver, Alex, got there: plotting weights, trying SAT and integer programming
  after noticing most nodes do nothing, spotting the periodic structure, matching it to MD5 and
  brute-forcing a word list. He also found the long-input bug that section 8 explains.
- Hyunwoo Park, [I Dropped a Neural Net](https://arxiv.org/abs/2602.19845): the same two signals as
  here for puzzle 2 (negative diagonal of `W_out W_in` for pairing, `|W_out|` to seed the order,
  then hill-climbing on the error), with a discussion of why training produces them.
- Yi Wang, [Solving Jane Street's Dropped a Neural Net](https://wangyi.ai/blog/2026/02/16/solving-jane-street-dropped-neural-net/):
  a different route, relaxing the permutations with Gumbel-Sinkhorn, optimising pairing and order in
  alternation, then 2-opt style moves. Under an hour on a laptop CPU.

## 15. Things worth knowing before you start your own

- **Never unpickle a model to look at it.** List the `GLOBAL`s first, then read it with
  placeholders. `torch.load(weights_only=True)` is the safe default for weights; for whole modules,
  allowlist the classes.
- **Integer weights mean a circuit.** Once you see powers of two and integer biases, stop thinking
  about features and start thinking about gates. A census of (weights, bias) patterns gives you the
  gate library in seconds.
- **Probe for what you suspect.** Signatures over a random batch find any quantity carried by a
  single neuron. Linear probes find quantities spread over several. Use both, and remember that
  equal quantities share a signature.
- **Use every exact check the puzzle gives you.** A known output column (`pred`) turns ordering into
  optimisation with a known optimum; a published SHA-256 turns "I think this is right" into "this is
  right".
- **Simulate exactly, then cross-check.** The numpy simulator here is exact because the values are
  integers, and torch's own forward pass agrees with it at the output and at all 192 comparator
  inputs. When two independent implementations agree, you can trust the rest of the analysis.
