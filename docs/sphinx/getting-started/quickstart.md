# Quickstart

This is the shortest path from a built module to a result. It runs one of the
bundled cases end to end and shows you what a successful run looks like. It does
not explain the model: for that, read [First run](first-run.md) (write the
smallest program yourself) or the [Tutorial A->Z](tutorial.md).

## Before you start

You need the `adc` module built and importable. If you have not built it yet,
follow [Installation](installation.md), then come back here. To confirm the
module is on your path:

```bash
python -c "import adc; print(adc.__file__)"
```

The command prints the path to the compiled module. If it raises
`ModuleNotFoundError`, set `PYTHONPATH` to the build directory that contains the
`.so` (see [Installation](installation.md)).

## Run a bundled case

The named scenarios (diocotron, euler_poisson, ...) live in the `adc_cases`
repository, not in `adc_cpp`. Clone it next to the module and run a case:

```bash
git clone https://github.com/wolf75222/adc_cases.git
cd adc_cases
python diocotron/run.py
```

Most cases accept a reduced mode for a fast smoke run. Pass it when you only
want to confirm the pipeline works:

```bash
python diocotron/run.py --quick
```

## What success looks like

The run prints the simulation time and a conserved quantity at each report
step, then writes its figures next to the case. For the diocotron case you get
the density snapshots and the growth-rate plot. A run that ends without a
Python traceback and produces those files has succeeded.

## Next steps

- [First run](first-run.md): write the smallest `adc` program yourself, one
  block, copyable as is.
- [Tutorial A->Z](tutorial.md): the full 18-step path, from `git clone` to the
  uniform versus AMR comparison.
- [Repository layout](repository-layout.md): how `adc_cpp` (the library) and
  `adc_cases` (the scenarios) fit together.
