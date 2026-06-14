# Run the Hoffart diocotron case

The Hoffart diocotron case lives in [adc_cases](https://github.com/wolf75222/adc_cases). It reproduces
the diocotron instability benchmark of Hoffart et al. (arXiv:2510.11808, Sec 5.3) with the DSL path.

## Prerequisites

- The `adc` Python module, built from adc_cpp. See [Installation](../getting_started/installation.md).
- A local clone of adc_cases.

## Run it

1. Clone adc_cases and enter the case folder:

   ```bash
   git clone https://github.com/wolf75222/adc_cases.git
   cd adc_cases/hoffart_euler_poisson_dsl
   ```

2. Run the case:

   ```bash
   python run.py
   ```

   The script advances the diocotron instability and writes its figures (snapshots, growth rate) to the
   case folder. The case provides a reduced or quick mode for a fast first run; see its README.

## What it demonstrates

A neutralizing-background density in an annulus, perturbed by an azimuthal mode `l`, drifts under the
ExB velocity and grows at a mode-dependent rate. The case measures the growth rate and compares it to
the paper target after mapping the fit windows into the solver clock.

## Learn more

- The case in adc_cases:
  [hoffart_euler_poisson_dsl](https://github.com/wolf75222/adc_cases/tree/master/hoffart_euler_poisson_dsl).
- The physics: [Hyperbolic-elliptic systems](../concepts/hyperbolic-elliptic-systems.md) and
  [Polar and disc geometry](../concepts/polar-disc-geometry.md).
- The DSL path: [Write a model with the symbolic DSL](write-a-model-with-dsl.md).
