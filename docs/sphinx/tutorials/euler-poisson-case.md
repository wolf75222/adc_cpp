# Run the Euler-Poisson case

The Euler-Poisson case lives in [adc_cases](https://github.com/wolf75222/adc_cases), not in this
repository: adc_cpp is the solver, adc_cases holds the named scenarios. This page shows how to run it.

## Prerequisites

- The `adc` Python module, built from adc_cpp. See [Installation](../getting_started/installation.md).
- A local clone of adc_cases.

## Run it

1. Clone adc_cases and enter the case folder:

   ```bash
   git clone https://github.com/wolf75222/adc_cases.git
   cd adc_cases/euler_poisson
   ```

2. Run the case:

   ```bash
   python run.py
   ```

   The script advances the case and writes its figures to the case `figures/` folder.

## What it demonstrates

The case couples a compressible gas (the hyperbolic Euler part) to a self-gravitating potential (the
elliptic Poisson part): the density sources the potential, and the potential gradient drives the gas.
It exercises the conservative transport and the elliptic solve together.

## Learn more

- The case in adc_cases: [euler_poisson](https://github.com/wolf75222/adc_cases/tree/master/euler_poisson).
- The coupling: [Hyperbolic-elliptic systems](../concepts/hyperbolic-elliptic-systems.md) and the
  [Poisson equation](../concepts/poisson.md).
- The self-gravitating regime in more detail:
  [Self-gravitating Euler-Poisson](self-gravitating-euler-poisson.md).
