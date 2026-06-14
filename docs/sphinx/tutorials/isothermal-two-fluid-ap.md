# Run the asymptotic-preserving two-fluid case

The isothermal two-fluid asymptotic-preserving (AP) case lives in
[adc_cases](https://github.com/wolf75222/adc_cases). It carries its own just-in-time C++ and is loaded
through `ctypes`, so it is a good example of a custom case built on the adc_cpp headers.

## Prerequisites

- The `adc` Python module, built from adc_cpp. See [Installation](../getting-started/installation.md).
- A local clone of adc_cases. A C++ compiler is needed because the case compiles its own kernel.

## Run it

1. Clone adc_cases and enter the case folder:

   ```bash
   git clone https://github.com/wolf75222/adc_cases.git
   cd adc_cases/two_fluid_ap
   ```

2. Run the case:

   ```bash
   python run.py
   ```

   The script builds the case kernel, advances the two-fluid model, and writes its figures (the
   AP-versus-explicit comparison) to the case folder.

## What it demonstrates

Two isothermal fluids coupled through the electric field, integrated with an asymptotic-preserving
scheme that stays stable and accurate as the stiff parameter shrinks, where a plain explicit scheme
would need a vanishing time step.

## Learn more

- The case in adc_cases: [two_fluid_ap](https://github.com/wolf75222/adc_cases/tree/master/two_fluid_ap).
- The stiff-source idea: [Operator splitting, IMEX and Schur condensation](../concepts/strang-schur-imex.md).
- Multiple species: [Multi-block and multi-species systems](../concepts/multi-block-multi-species.md).
