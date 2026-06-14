# Operator splitting, IMEX and Schur condensation

This page explains why a coupled hyperbolic-elliptic system advances its operators in
sequence rather than all at once, and how stiff sources are handled implicitly. It gives
the mental model behind operator splitting (Lie and Strang), the implicit-explicit (IMEX)
treatment of stiff sources, and the condensed Schur source stage. For the formulas and the
implementing headers, see the [ALGORITHMS guide](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Why split operators

The core solves a generic balance law `du/dt = T(U) + S(U)`, where `T` is the hyperbolic
transport (divergence of a flux) and `S` is a source. The two terms rarely live at the same
time scale. Transport is bounded by the Courant-Friedrichs-Lewy (CFL) condition on the wave
speed. A stiff source, such as a fast relaxation, a Lorentz force at high cyclotron
frequency, or Debye screening as the Debye length goes to zero, would force a far smaller
step if treated explicitly.

Operator splitting advances each sub-operator on its own, in sequence, over the same step.
Each sub-operator keeps its own integrator and its own stiffness, without contaminating the
other. This is what lets you keep an explicit transport on a CFL-sized step while a stiff
source is advanced implicitly.

## Lie and Strang splitting

There are two ways to chain the sub-flows. Lie splitting (also called Godunov splitting)
applies the transport flow then the source flow over the full step. It is first order in
time: the per-step error is carried by the commutator of the two operators.

Strang splitting brackets the full transport step between two source half-steps,
`S(dt/2); T(dt); S(dt/2)`. The symmetric arrangement cancels the dominant error term, so the
scheme is second order. The extra cost over Lie is a single additional source half-step per
macro-step.

Two caveats matter. Strang reaches second order only if each sub-integrator is itself at
least second order; an order-one transport or source caps the whole splitting at order one.
And in a hyperbolic-elliptic couple, the elliptic field has to be re-solved between the
source half-steps. Otherwise the second half-step reads a stale potential and the second
order is lost. The single-level pipeline does exactly that: its Strang variant re-solves the
fields before each stage that consumes the potential.

Splitting does not relax the CFL constraint on transport. It decouples the stiffnesses so a
stiff source can be made implicit without imposing its own tiny step on the transport.

## IMEX for stiff sources

IMEX (implicit-explicit) is the implicit treatment that splitting enables. The transport
stays explicit; the stiff source is solved implicitly, which is stable at a fixed step. An
IMEX Euler step first advances transport explicitly to form a known intermediate state, then
solves the implicit source relation on that state.

The valuable property is asymptotic preservation (AP). As the small stiffness parameter goes
to zero, an AP scheme stays consistent and stable at a fixed step and captures the limit
dynamics (equilibrium, quasi-neutrality) without resolving the stiff scale. For a linear
relaxation source the implicit solve is exact in one step and unconditionally stable, where a
plain fixed point would diverge as soon as the step times the stiffness exceeds one.

The implicit relation is solved per cell by a local Newton iteration with a
finite-difference Jacobian, so a model supplies no analytic Jacobian. When only some
variables are stiff, partial IMEX integrates implicitly only those components and advances
the rest explicitly; the partition comes from the model or from a per-block mask. IMEX Euler
is first order in time, and the AP property covers the relaxation limit, not the tightly
coupled potential-velocity-Lorentz problem. That stronger coupling is the domain of Schur
condensation.

## The condensed Schur source stage

When a stiff source couples the potential, the velocity and the Lorentz force together, you
cannot treat the components one at a time. The cyclotron rotation couples the two velocity
components, and the potential reacts to the charge displacement. A component-wise implicit
solve cannot see this loop.

The condensed Schur source stage closes the loop in one implicit elliptic solve. You
theta-discretize the implicit source, eliminate the velocity algebraically through the closed
inverse of the 2x2 rotation, and are left with a single elliptic equation on the potential
alone. This elimination is the Schur complement of the coupled system. After the potential
solve, you reconstruct the velocity from it, then extrapolate the theta-stage back to the
full step.

The condensed elliptic operator is in general a full anisotropic tensor, not the canonical
Laplacian, so it is solved by a matrix-free Krylov method (BiCGStab) preconditioned by the
symmetric part. When the magnetic field and the coupling coefficient vanish, the operator
degenerates exactly to the Laplacian and the stage reduces to the explicit electrostatic
push, a useful safety net. The theta-scheme is unconditionally stable for theta at least one
half.

This is a source stage with transport frozen: the density is constant across the stage, and
all transport dynamics stay in the hyperbolic phase of the splitting. It is opt-in, plugged
in as the source phase of a Strang step. One scope boundary is worth noting: the condensed
Schur stage runs on the single-level grid, not on the AMR hierarchy, where the elliptic part
is solved at the coarse level and injected to the fine levels. See the
[limitations reference](../reference/limitations.md) and the
[advanced topics](../advanced/index.md) for the implementation details and elliptic solvers.
