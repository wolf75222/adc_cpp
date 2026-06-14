# Self-gravitating Euler-Poisson

This page explains the self-gravitating regime of the Euler-Poisson case. To run the case, follow
[Run the Euler-Poisson case](euler-poisson-case.md); this page focuses on what the physics means.

## The coupling

Self-gravitating means the gas is its own source of potential. The density sets the right-hand side of
the Poisson equation, the Poisson solve returns the gravitational potential, and the potential gradient
enters the momentum equation as a body force. Mass and momentum stay conservative; the potential is
recomputed each step from the current density.

## What to look at

- Energy: the total energy (kinetic plus internal plus potential) is a useful conserved diagnostic.
- The Poisson sign: a self-gravitating potential is attractive, the opposite sign of the repulsive
  plasma coupling used by the diocotron cases.

## Learn more

- Run it: [Run the Euler-Poisson case](euler-poisson-case.md).
- The elliptic side: [Poisson equation](../concepts/poisson.md) and
  [Elliptic right-hand side and the aux channel](../concepts/elliptic-rhs.md).
- The case in adc_cases:
  [euler_poisson](https://github.com/wolf75222/adc_cases/tree/master/euler_poisson).
