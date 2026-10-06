# Contributing

Issues and pull requests are welcome. The most useful ones add an explanation that would have helped
you when you were stuck, a cleaner way to see a structure in one of the networks, or a different
route to either answer.

## Getting set up

```
git clone https://github.com/danieltyukov/jane-street-neural-net-puzzle
cd jane-street-neural-net-puzzle
make setup fetch
make torch    # optional, for the cross-checks
make test
```

`make test` runs the unit tests (no puzzle files needed) and the end-to-end checks, which assert every
number quoted in the README and the write-up. If you change behaviour, update the docs and the test
together.

## Style

- Plain Python 3.10+, with `numpy`, `scipy`, `matplotlib` and `wordfreq`. Nothing outside
  `crosscheck.py` imports torch.
- Never call `pickle.load` or `torch.load(weights_only=False)` on puzzle files outside the allowlisted
  loader in `crosscheck.py`. Use `torchzip.TorchArchive` instead.
- Each step in `src/nnre/pipeline.py` returns plain data and prints a short report. Keep new steps the
  same shape so `nnre all` and `build/results.json` pick them up.
- Prefer recovering facts from the weights or from simulation over hard-coding layer numbers.

## Regenerating figures

```
make figures
```

writes everything in `docs/img/`. Commit the PNGs together with the code that changed them.
