# Looking at the files yourself

The scripts do everything automatically, but both puzzles are more fun when you poke at the files by
hand. These notes assume you ran `make setup fetch` (and `make torch` where torch is used).

## What is in a `.pt` file

```
unzip -l data/hashnet/model.pt | head
```

A file written by `torch.save` is a zip archive: `puzzle/data.pkl` is a pickle describing the
objects, and `puzzle/data/0`, `puzzle/data/1`, ... are the raw little-endian bytes of each tensor
storage. The 5446 entries of `model.pt` are the pickle, three small bookkeeping files (`byteorder`,
`version`, `.data/serialization_id`) and 5442 storages: a weight and a bias for each of the 2721
layers.

## Reading a pickle without running it

Python ships a disassembler for pickles. It only prints opcodes, it never builds objects:

```
mkdir -p /tmp/pk && cd /tmp/pk && unzip -o ~/path/to/data/hashnet/model.pt puzzle/data.pkl
python -m pickletools puzzle/data.pkl | grep GLOBAL
```

```
      2: c    GLOBAL     'torch.nn.modules.container Sequential'
    140: c        GLOBAL     '__builtin__ set'
    194: c        GLOBAL     'collections OrderedDict'
    634: c            GLOBAL     'torch.nn.modules.linear Linear'
    698: c                    GLOBAL     'torch._utils _rebuild_parameter'
    733: c                    GLOBAL     'torch._utils _rebuild_tensor_v2'
    784: c                            GLOBAL     'torch FloatStorage'
   1114: c            GLOBAL     'torch.nn.modules.activation ReLU'
1725468: c        GLOBAL     'cloudpickle.cloudpickle _make_function'
1725514: c            GLOBAL     'cloudpickle.cloudpickle _builtin_type'
1725601: c                GLOBAL     '_codecs encode'
1726085: c        GLOBAL     'cloudpickle.cloudpickle _function_setstate'
1726438: c                GLOBAL     'cloudpickle.cloudpickle subimport'
```

Every name a pickle can call appears as a `GLOBAL` (or `STACK_GLOBAL`) opcode, so this list is the
complete set of code the file could run. The `cloudpickle` entries near the end are the custom
forward wrapper. `nnre unpickle` prints the same list from
[`torchzip.referenced_globals`](../src/nnre/torchzip.py).

## What torch says

```python
import torch
torch.load("data/hashnet/model.pt", weights_only=True)
```

fails with `Unsupported global: GLOBAL torch.nn.modules.container.Sequential was not an allowed
global by default`. `weights_only=True` (the default since PyTorch 2.6) only accepts tensors and
plain containers, and this file stores whole modules. The error suggests either trusting the file
(`weights_only=False`) or allowlisting the class. [`crosscheck.py`](../src/nnre/crosscheck.py) does a
stricter version of the second: a custom unpickler that accepts the eight globals a Linear/ReLU stack
needs and turns the cloudpickle helpers into dummies.

The pieces of puzzle 2 are plain `state_dict`s, so this works directly:

```python
torch.load("data/dropped/pieces/piece_0.pth", map_location="cpu", weights_only=True)
```

(`map_location` matters: the pieces were saved from `cuda:0`.)

## Why there are two model files

The Hugging Face repo holds `model.pt` and `model_3_11.pt`. Their 5442 tensor storages are
byte-identical; only the pickled wrapper differs. A pickled function carries raw CPython bytecode, and
bytecode changed between 3.10 and 3.11 (3.11 added inline caches and a different call sequence), so
a code object saved by 3.10 cannot be rebuilt by 3.11. This project never rebuilds it, which is why
one file is enough on any Python version.

## Exploring the network in Python

```python
from nnre.hashnet import model, readout
net = model.load()                       # sparse cache in build/, parsed from model.pt the first time
net(["vegetable dog", "hello"])          # outputs, exactly as the original model
net.describe(1, 224)                     # one neuron as a formula
acts = net.activations(["hello"], [1, 2718])   # any layers you like
readout.digests(net, ["hello"])          # the 16 bytes the comparator checks
```

`net.describe(layer, neuron)` prints things like

```
L563:264 = relu(1*L562:248 -127)
L564:228 = relu(1*L563:260 + -128*L563:264 + 128*L563:268 +0)
```

which is the first stage of the length-to-bits decomposition: `L563:264 - L563:268` is
`[x >= 128]`, and `L564:228` subtracts 128 times that. Following inputs backwards with `describe`
is a good way to read any part of the circuit.

## Netron

[Netron](https://netron.app/) opens PyTorch files in the browser with its own JavaScript reader, so
it is another way to browse the pieces of puzzle 2 without running Python.

## The original pages

Both puzzles are still live on Hugging Face. On [puzzle 1](https://huggingface.co/spaces/jane-street/puzzle)
you can type an input and see the model's output; on
[puzzle 2](https://huggingface.co/spaces/jane-street/droppedaneuralnet) the "Check Solution" box hashes
a permutation and compares it with the expected SHA-256, the same check `nnre order` runs locally.
