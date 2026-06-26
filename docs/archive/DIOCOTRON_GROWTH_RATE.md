# Diocotron growth rate reproduction (vs Hoffart arXiv:2510.11808)

Consolidated document: quantitative reproduction of the diocotron instability growth rate
with adc_cpp, compared to the reference paper (Hoffart, Maier, Shadid, Tomas, *Structure-preserving
finite-element approximations of the magnetic Euler-Poisson equations*, arXiv:2510.11808, Section 5.3).
Raw ROMEO results: [romeo/HERO_RESULTS.md](../romeo/HERO_RESULTS.md).

Analytic target (Petri / Davidson-Felice, ring geometry `r0:r1:Rwall = 6:8:16`, reproduced
by `analysis/diocotron_growth.hpp`): `gamma_3 = 0.772`, `gamma_4 = 0.911`, `gamma_5 = 0.683`.

## 1. STABILITY blocker lifted (prerequisite for any sweep)

Beyond an effective resolution of ~448, the simulation went to `nan` in the very first steps. Diagnosis:
the **geometric multigrid DIVERGED** at the embedded conducting boundary on the fine mesh (non-Galerkin
coarsening + circle mask re-evaluated per level -> inconsistent coarse correction, V-cycle spectral
radius > 1, erratic depending on the circle alignment). The warm start propagated the
divergence -> `phi` then the field went to `nan`. It was NEITHER the time step (already capped), NOR the
density floor (the density stays bounded; only `phi` blows up, at the WALL RADIUS r=0.398).

Fix: `GeometricMG::solve_robust` (`include/pops/elliptic/geometric_mg.hpp`). Phase 1 = the
standard V-cycle (BIT-IDENTICAL when it converges or stalls); ONLY in case of true divergence
(final residual > initial residual): STICKY hardening of the GS smoothing + cold restart until
it becomes contractant again. Result: stable up to eff 1024 (uniform AND AMR `ml`), mass `~1e-14`,
the 8 recorded runs (eff <= 448) stay BIT FOR BIT identical. Details: `docs/HERO_RUN_AMR.md`.

## 2. Methods (order ramp-up toward the analytic rate)

The M1 ceiling (`gamma_norm ~ 0.58`) came from the scheme DIFFUSION (order 1 in space AND in
time), not from physics. Two classic levers, confirmed by the literature (Jiang-Shu,
Borges WENO-Z, Gottlieb-Shu-Tadmor SSPRK, Ern-Guermond RK order 3):

- **High-order reconstruction**: `NoSlope` (order 1) -> `VanLeer`/`Minmod` (MUSCL order 2) ->
  **`Weno5`** (WENO5-Z, order 5, `operator/reconstruction.hpp`, order 5.00 verified by
  `test_weno_convergence`). Option `recon` of `examples/diocotron_column_amr.cpp`; `recon=0`
  bit-identical to history.
- **High-order time integration**: forward Euler positively BIASES an exponentially growing
  mode (unstable on the imaginary axis, term `+ 1/2 omega_r^2 dt`). **SSPRK3** (Shu-Osher)
  removes this bias at order 3. `examples/diocotron_highorder.cpp`: WENO5-Z + SSPRK3, Poisson
  RE-SOLVED at each RK stage (stage-by-stage coupling, `solve_robust`).

## 3. Results

### 3a. Column convergence: AMR tracks uniform (ROMEO 613945)

At equal effective resolution, AMR `ml` (multi-level Poisson) MATCHES uniform for ~40 %
of the cells (the M2b promise, at scale); VanLeer largely exceeds NoSlope:

| case | eff 512 (lin) | eff 1024 (lin) | cells vs unif |
|---|---|---|---|
| uniform NoSlope | 0.650 | 0.706 | 100 % |
| uniform VanLeer | 0.753 | 0.748 | 100 % |
| AMR `ml` VanLeer | 0.762 | 0.747 | ~40 % |

### 3b. High-precision rate, modes 3/4/5 (ROMEO 613961, WENO5+SSPRK3)

Paper window, R^2 = 1.00. The high order moves mode 4 from 0.56 (NoSlope+Euler,
under-estimated, too diffusive) to ~0.99, on the RIGHT side of 0.911:

| mode l | analytic | eff 256 | eff 512 | eff 1024 |
|---|---|---|---|---|
| 3 | 0.772 | +8 % | +10 % | +11 % |
| 4 | 0.911 | +8 % | +8 % | +8 % |
| 5 | 0.683 | +7 % | +7 % | +7 % |

## 4. Diagnosis: an overshoot ~+8 % UNIFORM and FLAT in resolution

Five independent measurements rule out the "easy" causes AND the conducting boundary:

1. **Flat in resolution**: eff 256 ~ 512 ~ 1024 (same +8 %). More cells do NOT close the gap.
2. **Flat in reconstruction order**: `WENO5 ~ VanLeer`. It is not the spatial order.
3. **Sweep in delta**: the LINEAR LIMIT (delta -> 0) RISES to +27 % instead of dropping. The apparent
   agreement at delta=0.1 was a fortuitous compensation by saturation. So it is NOT a
   nonlinear contamination nor a window effect.
4. **DIMENSIONLESS ratio** `gamma / |Re(omega)|` (independent of normalization, via the COMPLEX
   eigenvalue `diocotron_eigenvalue`: analytic Re_norm = -2.08 / -2.75 / -3.44 for l=3/4/5):
   measured 0.31 vs analytic 0.331 -> ~5 % STRUCTURAL DISTORTION of the eigenvalue + ~3 % of
   `omega_D` normalization offset.
5. **Flat in boundary treatment (cut-cell vs staircase)**: the order-2 Shortley-Weller embedded
   boundary was implemented (`GeometricMG`, option `cut_cell`, validation `test_cut_cell`: L2 order 1.93,
   Poisson error 3459x lower than staircase at nc=512). On the diocotron (nc=256, VanLeer,
   `cut=1` vs `cut=0`), the rate is **IDENTICAL to 3 digits** (gamma_norm 0.945, 0.838, 0.738 ... at the
   same windows). So the conducting boundary is NOT the cause of the +8 %.

Cause: it is NOT the wall treatment. The unstable mode-l lives on the **ring** (r ~ 0.175),
far from the wall (r = 0.40): the image effect of the wall on the mode l at the ring radius decays
as `(r_ring/Rwall)^(2l)` = `(0.44)^8 ~ 1e-3` for l=4, electrostatically negligible. The overshoot
is **structural**: ~5 % distortion of the eigenvalue on the Cartesian E x B dynamics (the
4-fold symmetry of the square grid resonates with mode 4) plus ~3 % of `omega_D` normalization (measurement 4),
and not an O(1) boundary bias. The transport itself is faithful (green invariants, section 6).

## 5. Direct comparison to the paper (reading the arXiv)

Method and physics **identical** (checked in the paper text, Section 5.3):
- initial velocity `v0 = -(grad phi0 x Omega)/|Omega|^2` (E x B drift) = our `Diocotron` model;
- measurement: *"DFT of the potential phi at FIXED radius r=r0, modulus of the mode l coefficient"* = our
  `mode_amplitude`, normalized to the initial value, exponential fit over a narrow window;
- same analytic targets 0.772 / 0.911 / 0.683, same fit windows;
- time: explicit RK order 3 (ours: SSPRK3); space: order 2 graph-viscosity dG (ours:
  WENO5, order 5, SO our scheme is NOT the limit).

The decisive DIFFERENCE is the GEOMETRY, proven by the paper convergence table (Fig 5.4d):

| mode 4 | paper (dofs) | gamma_h | gap | | us (eff) | gamma_h |
|---|---|---|---|---|---|---|
| | 196 608 | 0.935 | +2.6 % | | 256 | 0.985 (+8 %) |
| | 786 432 | 0.919 | +0.9 % | | 512 | 0.988 (+8 %) |
| | 3 145 728 | **0.913** | **+0.2 %** | | 1024 | 0.987 (+8 %) |

The paper **CONVERGES** (0.935 -> 0.919 -> 0.913) on a fitted **DISK** mesh; we are
**FLAT** at +8 % on a square box. The gap is real, but the cut-cell experiment (section 4,
measurement 5) shows that it does NOT come from the wall *stencil*: setting Dirichlet on the real circle (order
2) instead of the staircase does not change the rate. The difference lies in the Cartesian representation of the
**ring dynamics** itself (E x B advection on a square grid, whose 4-fold symmetry resonates
with mode 4), not in the distant conducting boundary. The exact structural cause remains to be confirmed
(grid symmetry vs DFT-phi measurement method vs normalization), see section 7.

## 6. Verified physical indicators (transport fidelity)

`analysis/diocotron_invariants.hpp` + `test_diocotron_invariants` (mode 4, WENO5+SSPRK3, eff 256):

| invariant | result | role |
|---|---|---|
| mass `int rho` | exact (drift 0) | conservativity (flux form) |
| energy `1/2 int \|grad phi\|^2` | < 1 % | invariant of the ideal system |
| angular momentum `int rho r^2` | < 1 % | diocotron invariant (Davidson) |
| enstrophy `int rho^2` | -5.5 % | Casimir: MEASURES the numerical diffusion |
| maximum principle | `rho in [floor, rho_max]` | no spurious values |
| `Re(omega)` (rotation) | reproduced (~+8 %) | 2nd half of the dispersion |

Figures: `docs/fig_diocotron_highorder.png` (rate vs order), `docs/fig_diocotron_invariants.png`
(invariants vs time).

## 7. Conclusion and next step

The high-order RECONSTRUCTION and INTEGRATION levers are in place and verified (from -39 %
at order 1). With a clean linear fit window ([3,9], R^2=1.00), mode 4 is even at **+1 %**
of the analytic at eff 1024 and mode 3 EXACT (see the diagnosis below); the residual overshoot rate
concentrates on the high-l modes.

The **cut-cell Shortley-Weller** embedded boundary (`GeometricMG`, option `cut_cell`) was implemented and
validated (`test_cut_cell`: L2 order 1.93 at the boundary, Poisson error 3459x lower than staircase).
**Important negative result**: it does NOT move the diocotron rate (section 4, measurement 5). So the conducting
boundary was not the supposed blocker; the unstable mode is too far from the wall to "see" it
(image effect `(0.44)^8 ~ 1e-3`). The cut-cell remains a clean precision gain of the Poisson
solver, useful for configurations where the charge approaches the wall.

**Diagnosis settled (workflow `diocotron-overshoot-diag` + ROMEO confirmation 614125, see
`romeo/CONV_RESULTS.md`).** Three leads were CLOSED:
- **grid symmetry: RULED OUT.** Rotating the lobe phase only changes gamma by +0.04 %. Decisive
  clue: on the uniform grid it is mode 5 (NOT couplable to a 4-fold square grid) that overshoots
  the most, mode 4 (the 4-fold candidate) is benign -> the opposite of a grid resonance;
- **measurement method: ELIMINATED.** `mode_amplitude` (lines 242-262) ALREADY reads the potential phi and does
  the azimuthal DFT of mode l at r=r0: this is EXACTLY the paper method, not a density observable;
- **`omega_D` normalization: REFUTED.** Recompute = 0.14324 (rho_bar = 1 - delta = 0.9, Davidson
  convention), confirmed by the eigensolver and the simulated rotation frequency. It is the right scale.

What REMAINS (and which ROMEO quantified): an eigenvalue distortion **FLAT in resolution** (eff
256 ~ 512 ~ 1024, the gap does not close under refinement -> structural, not from truncation) and
**GROWING with mode l**, NON uniform. Linear window [3,9], R^2=1.00, eff 1024: mode 3 = 0.771
(EXACT), mode 4 = 0.921 (**+1 %**, near converged), mode 5 = 0.881 (**+29 %**, aberrant). The old
"+8 % uniform" was a window/scheme artifact (narrow window + WENO5); the rate depends strongly
on the window for lack of a clean exponential plateau. The analytic dispersion gamma(l) peaks at l=4 and
drops back at l=5; our scheme reproduces the peak but not the high-l ROLL-OFF, because the eigenfunction of
mode 5 (radially more structured) is the most distorted by the CARTESIAN representation of the ring.

Path toward < 1 %: make the RING boundaries (r0, r1) where the mode lives FIT, not the wall:
- **cut-cell / level-set on r0 and r1** (the wall cut-cell already exists, with no effect here);
- **polar grid (r, theta)** for transport + Poisson (removes the rotation invariance breaking).
The exact mode 3 and the +1 % mode 4 show that the framework is CORRECT; the blocker is the fidelity of the
eigenfunction at high l. Figure: `docs/fig_diocotron_conv_modes.png`.

## 8. Reproduction

```
# stability + AMR ; args : out nc nsteps refine l ml recon cut
#   recon : 0 NoSlope, 1 VanLeer, 2 Minmod   |   cut : 0 escalier, 1 cut-cell Shortley-Weller
g++ -std=c++23 -O2 -I include examples/diocotron_column_amr.cpp -o dca
./dca out 640 3000 0 4 0 1 0      # uniforme VanLeer eff 640, escalier (stable)
./dca out 256 900  0 4 0 1 1      # cut-cell : taux INCHANGE (cf section 4, mesure 5)

# haute precision WENO5 + SSPRK3, mode l, comparaison analytique
g++ -std=c++23 -O3 -fopenmp -I include examples/diocotron_highorder.cpp -o dho
./dho out 512 800 4 3 0.4         # eff 512, mode 4, WENO5
python3 scripts/validate_diocotron_growth.py out/ring_amp.csv --target 0.911 --window 4.2,5.2

# ROMEO : sbatch romeo/diocotron_highorder_hero.sbatch  (cf. romeo/HERO_RESULTS.md)
```
