# Algorithms

Catalog of the generic numerical methods of the `adc_cpp` core. For each one: the intuition, the
formula and its discretization, a pseudocode, the C++ file that implements it, and the constraints. The
core is model-agnostic; it names no scenario (diocotron, Euler-Poisson, two-fluid). These scenarios are
compositions of generic bricks and live on the application side
([`adc_cases`](https://github.com/wolf75222/adc_cases)); so do their end-to-end validations.

Each section follows the same plan: intuition (what it is for), formula and discretization (where it
comes from), pseudocode (the algorithm), code (the file and the functions), constraints and remarks (the
stability, the limits, the ctest test that covers it). All the file paths and test names cited exist in
this repository.

Architecture (layers, dispatch seam, library/application boundary):
[ARCHITECTURE.md](ARCHITECTURE.md). Design choices: [CHOICES.md](CHOICES.md). Trace of the GH200 device
validations: [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md). References:
[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).

## Contents

- [Model equations](#equations-modele)
- [1. Finite volumes: first-order Godunov](#1-volumes-finis--godunov-ordre-1)
- [2. Numerical fluxes: Rusanov, HLL, HLLC, Roe](#2-flux-numeriques--rusanov-hll-hllc-roe)
- [3. MUSCL reconstruction (order 2) and WENO5-Z (order 5)](#3-reconstruction-muscl-ordre-2-et-weno5-z-ordre-5)
- [4. Time integration: SSPRK, object integrators, user integrator](#4-integration-en-temps--ssprk-integrateurs-objets-integrateur-utilisateur)
- [5. Stiff sources: asymptotic-preserving IMEX and partial IMEX](#5-sources-raides--imex-asymptotic-preserving-et-imex-partiel)
- [6. Operator splitting: Lie and Strang](#6-splitting-doperateurs--lie-et-strang)
- [7. Multirate: subcycling, cadence, adaptive step](#7-multirate--sous-cyclage-cadence-pas-adaptatif)
- [8. Parabolic term: diffusion as face flux](#8-terme-parabolique--diffusion-en-flux-de-face)
- [9. Elliptic: geometric multigrid](#9-elliptique--multigrille-geometrique)
- [10. Elliptic: spectral Poisson (FFT), single-rank and distributed](#10-elliptique--poisson-spectral-fft-mono-rang-et-distribue)
- [11. Extended elliptic: eps(x), screened/Helmholtz, anisotropic](#11-elliptique-etendu--epsx-helmholtzecrante-anisotrope)
- [12. Full-tensor elliptic: matrix-free Krylov (BiCGStab)](#12-elliptique-a-tenseur-plein--krylov-matrice-libre-bicgstab)
- [13. Condensed implicit source: Schur condensation](#13-source-implicite-condensee--condensation-de-schur)
- [14. Embedded boundary: Shortley-Weller cut-cell](#14-bord-embedded--cut-cell-shortley-weller)
- [15. Disc domain: mask, masked transport, cut-cell transport](#15-domaine-disque--masque-transport-masque-transport-cut-cell)
- [16. Polar geometry: transport and Poisson on a ring (r, theta)](#16-geometrie-polaire--transport-et-poisson-sur-anneau-r-theta)
- [17. AMR: Berger-Oliger subcycling + conservative reflux](#17-amr--sous-cyclage-berger-oliger--reflux-conservatif)
- [18. Multi-patch AMR: coverage-aware reflux, MPI-distributed](#18-amr-multi-patch--reflux-coverage-aware-distribue-mpi)
- [19. Berger-Rigoutsos clustering and regrid](#19-clustering-berger-rigoutsos-et-regrid)
- [20. Distributed mesh: global BoxArray, halos, load balancing](#20-maillage-distribue--boxarray-global-halos-equilibrage)
- [21. Extensible aux channel](#21-canal-aux-extensible)
- [22. Runtime composition and multi-species system](#22-composition-runtime-et-systeme-multi-especes)
- [23. Symbolic DSL: codegen, JIT, AOT](#23-dsl-symbolique--codegen-jit-aot)
- [24. The dispatch seam (Kokkos: Serial / OpenMP / Cuda / MPI)](#24-le-seam-de-dispatch-kokkos--serial--openmp--cuda--mpi)
- [25. Capabilities to qualify (present but limited, or off master)](#25-capacites-a-qualifier-presentes-mais-limitees-ou-hors-master)
- [Which scheme or solver when](#quel-schema-ou-solveur-quand)
- [References](#references)

---

## Model equations

The core solves, on an adaptive Cartesian mesh (and, optionally, on a polar ring or an immersed disc
subdomain), the generic form

$$\partial_t U + \mathrm{div} F(U, \mathrm{aux}) = S(U, \mathrm{aux}) \qquad \text{(hyperbolique, par bloc)}$$

$$\mathrm{div}(\varepsilon\,\nabla \phi) - \kappa\,\phi = f(U) \qquad \text{(elliptique, partage)}$$

The hyperbolic part `U` and the elliptic part `phi` couple at each step through the `aux` channel (base
contract `(phi, grad_x, grad_y)`, extensible to `B_z`, `T_e`). A model is a composition
`CompositeModel<Transport, Source, Elliptic>`; the coupling enters through the flux (`aux` read in `F`)
or through the source (`aux` read in `S`), under the same spatial operator. The reconstruction, flux and
source bricks are reused across geometries (Cartesian, polar, cut-cell); only the metrics and the
divergences change. The spatial discretization is finite volume (section 1); the time advance is a
method of lines (section 4); the elliptic part is solved at each step (sections 9 to 13) and re-read by
`aux`.


---

## 1. Finite volumes: first-order Godunov

**Intuition.** The profile in each cell is replaced by its average. At each interface,
two averages meet: a local Riemann problem. The numerical flux solves it
(approximately) and the conservative update transports matter from one cell to the next.

**Formula / discretization.** Integration of the conservation law $\partial_t U + \mathrm{div}\,F(U,\mathrm{aux}) = S(U,\mathrm{aux})$
over the cell $(i,j)$ and the step $\Delta t$:

$$U_{ij}^{n+1} = U_{ij}^n - \frac{\Delta t}{\Delta x}\big(\hat F_{i+1/2,j} - \hat F_{i-1/2,j}\big)
                             - \frac{\Delta t}{\Delta y}\big(\hat G_{i,j+1/2} - \hat G_{i,j-1/2}\big)
                             + \Delta t\, S_{ij}$$

The conservative form (face-flux difference) guarantees exact discrete conservation: what
cell $i$ loses at its right face, cell $i+1$ gains at its left face. This is the
property on which all of AMR depends (reflux corrects exactly these face fluxes). The core does not
do the update itself: it assembles the residual of the method of lines

$$R_{ij} = -\,\mathrm{div}\,\hat F + S = S_{ij}
          - \frac{\hat F_{i+1/2,j} - \hat F_{i-1/2,j}}{\Delta x}
          - \frac{\hat G_{i,j+1/2} - \hat G_{i,j-1/2}}{\Delta y},$$

and the time integrator applies $U^{n+1} = U^n + \Delta t\, R$ (Euler) or an SSPRK combination.
The face flux $\hat F_{i+1/2}$ is evaluated from the reconstructed states on either side:
$\hat F_{i+1/2} = \texttt{nflux}(\texttt{recon}_L, \texttt{aux}_i, \texttt{recon}_R, \texttt{aux}_{i+1}, \texttt{dir})$.
The auxiliary `aux` (`phi`, `grad phi`) is not reconstructed (a smooth field coming from the elliptic): we
take the cell value on each side. An optional parabolic term ($+\nu\,\mathrm{Lap}\,U$,
guarded by the `DiffusiveModel` concept) is added, either as a 5-point centered difference in
`assemble_rhs`, or as a Fickian face flux $-\nu\,\nabla U$ in `compute_face_fluxes` (cf. section 8).

```
function assemble_rhs(model, U, aux, geom, R, recon_prim):   # R = -div Fhat + S
    dx, dy = geom.dx(), geom.dy()
    for box li in U:                                          # seam for_each_cell : Kokkos (Serial / OpenMP / Cuda)
        for each valid cell (i, j):
            Ac  = load_aux(aux, i, j)                         # [phi, grad_x, grad_y, extra...]
            # face x gauche (i-1/2) et droite (i+1/2)
            Lxm = reconstruct(model, U, i-1, j, dir=0, sgn=+1, recon_prim)   # etat L de la face i-1/2
            Rxm = reconstruct(model, U, i,   j, dir=0, sgn=-1, recon_prim)   # etat R de la face i-1/2
            Lxp = reconstruct(model, U, i,   j, dir=0, sgn=+1, recon_prim)
            Rxp = reconstruct(model, U, i+1, j, dir=0, sgn=-1, recon_prim)
            Fxm = nflux(model, Lxm, A(i-1,j), Rxm, Ac,     dir=0)
            Fxp = nflux(model, Lxp, Ac,       Rxp, A(i+1,j), dir=0)
            # faces y : idem dans la direction j
            Fym, Fyp = (analogue avec dir=1)
            S = model.source(load_state(U, i, j), Ac)
            for c in 0 .. n_vars-1:
                R(i,j,c) = S[c] - (Fxp[c]-Fxm[c])/dx - (Fyp[c]-Fym[c])/dy
            if DiffusiveModel(model):                          # if constexpr -> zero codegen sinon
                R(i,j,c) += nu * (Lap_x U + Lap_y U)           # +nu Lap(U), 5 points centres
```

**Code.** [`include/adc/numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp):
`assemble_rhs<Limiter, NumericalFlux>` computes directly $R = -\mathrm{div}\,\hat F + S$, going through
the named device functor `detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>` (functor
rather than extended lambda: reliable device emission under nvcc from an external TU, AOT path
`add_compiled_model`). `compute_face_fluxes<Limiter, NumericalFlux>` writes the face fluxes `Fx, Fy`
(via `detail::FaceFluxXKernel` / `FaceFluxYKernel`) before the divergence: this is what the AMR reflux
needs. Same `reconstruct` and same numerical flux as `assemble_rhs`, so
$R = S - (\texttt{Fx}(i{+}1)-\texttt{Fx}(i))/\Delta x - (\texttt{Fy}(j{+}1)-\texttt{Fy}(j))/\Delta y$
gives back exactly the residual. The loop goes through the `for_each_cell` seam. An opt-in variant
`assemble_rhs_masked` (functor `AssembleRhsMaskedKernel`) restricts transport to an active
subdomain: zero normal flux on the faces touching a masked cell (conservative FV wall), zero residual
on the inactive cells. The global CFL step is read by `max_wave_speed_mf` (reduction over the local
boxes then MPI `all_reduce_max`, otherwise each rank would pick a different `dt` and diverge).

**Constraints / remarks.** CFL condition: $\Delta t \le C\,\dfrac{\min(\Delta x,\Delta y)}{\max|\lambda|}$,
where $\lambda$ is the local wave speed and $C \le 1$ at order 1; `max_wave_speed_mf` provides
$\max|\lambda|$. A model without transport ($\max|\lambda| = 0$) does not constrain the step
(`max_wave_speed_mf` returns 0). The operator writes only `R` (it touches neither `U` nor `aux`, no
ghost fill). Validation: `test_spatial_discretisation` (the reconstruction x flux pair is a
named type assembled by `assemble_rhs`), `test_cfl_dt` (`dt = cfl * min(dx,dy) / max|lambda|`
multi-species). The Cartesian/polar invariant is checked bit-for-bit (the polar operator does not touch
this path). The end-to-end validations (diocotron, Euler-Poisson) live on the `adc_cases` side.

## 2. Numerical fluxes: Rusanov, HLL, HLLC, Roe

**Intuition.** Four levels of fidelity for the face Riemann problem, ordered by the number of waves
they resolve. Rusanov lays down a single diffusion bump (the most robust, the most diffusive); HLL
estimates two signal speeds and keeps a single star region; HLLC adds the contact wave (the
passive density discontinuity); Roe linearizes the system by the Roe average and solves
the linearized Riemann problem exactly. Each flux is a stateless policy (`ADC_HD` functor,
device-callable, no virtual) with contract
$\texttt{operator()}(m, U_L, A_L, U_R, A_R, \texttt{dir}) \to \texttt{Model::State}$, chosen by
template alongside the reconstruction limiter.

**Formula / discretization.** Rusanov (local Lax-Friedrichs), component by component (scalar
upwind, without coupling):

$$\hat F_{i+1/2} = \tfrac12\big(F(U_L)+F(U_R)\big) - \tfrac12\,\alpha\,(U_R - U_L),
\qquad \alpha = \max\big(s_L(U_L), s_R(U_R)\big),$$

where $s_{L,R}$ (`max_wave_speed`) of each state; Rusanov requires only this member, so it
applies to any base `PhysicalModel`. HLL uses the Davis estimates
$s_L = \min(s_L^{gauche}, s_L^{droit})$, $s_R = \max(s_R^{gauche}, s_R^{droit})$ via `hll_speeds`, and
falls back to the upwind flux in the supersonic regime:

$$\hat F^{HLL} = \begin{cases}
F(U_L) & s_L \ge 0 \\[2pt]
\dfrac{s_R F(U_L) - s_L F(U_R) + s_L s_R (U_R - U_L)}{s_R - s_L} & s_L < 0 < s_R \\[6pt]
F(U_R) & s_R \le 0
\end{cases}$$

HLLC restores the contact wave $S_*$ in the middle (Toro speed eq. 10.37) and reconstructs the
star states $U_L^*, U_R^*$:

$$S_* = \frac{p_R - p_L + \rho_L u_{nL}(s_L - u_{nL}) - \rho_R u_{nR}(s_R - u_{nR})}
             {\rho_L(s_L - u_{nL}) - \rho_R(s_R - u_{nR})},
\qquad \hat F^{HLLC} = F_K + s_K\,(U_K^* - U_K),\ K \in \{L, R\},$$

with $u_n$ the normal velocity and the factor $\rho_K (s_K - u_{nK}) / (s_K - S_*)$ for the star states.
Roe linearizes the system by the $\sqrt{\rho}$-weighted average:

$$\hat F^{Roe} = \tfrac12\big(F_L + F_R\big) - \tfrac12 \sum_k |\tilde\lambda_k|\,\alpha_k\,r_k,$$

waves $\{u_n - c,\ u_n,\ u_n,\ u_n + c\}$ with celerity $c$ deduced from the Roe enthalpy $H$, and
Harten's entropy fix ($\varepsilon = 0.1\,c$) on the acoustic waves 1 and 5 to avoid
the non-entropic shock (sonic glitch). HLLC and Roe target Euler 2D (`n_vars == 4`): normal/tangential
momentum indices according to `dir`; $\gamma - 1$ is deduced from the state (ideal
gas), no `gamma` member is required from the model.

```
function HLLC(m, UL, AL, UR, AR, dir):                 # n_vars == 4 (Euler 2D)
    in = (dir==0 ? 1 : 2);  it = (dir==0 ? 2 : 1)      # qte de mvt normale / tangentielle
    rL, rR   = UL[0], UR[0]
    unL, unR = UL[in]/rL, UR[in]/rR
    pL, pR   = m.pressure(UL), m.pressure(UR)
    sL, sR   = hll_speeds(m, UL,AL, UR,AR, dir)         # Davis : min/max des vitesses signees
    FL, FR   = m.flux(UL,AL,dir), m.flux(UR,AR,dir)
    if sL >= 0: return FL                               # supersonique a droite -> flux amont
    if sR <= 0: return FR
    sStar = (pR - pL + rL*unL*(sL-unL) - rR*unR*(sR-unR))    # Toro 10.37
          / (rL*(sL-unL) - rR*(sR-unR))
    if sStar >= 0:                                      # etat etoile gauche
        fac = rL*(sL - unL) / (sL - sStar)
        Us[0]=fac; Us[in]=fac*sStar; Us[it]=fac*(UL[it]/rL)
        Us[3]=fac*(UL[3]/rL + (sStar-unL)*(sStar + pL/(rL*(sL-unL))))
        return FL + sL*(Us - UL)
    else:                                              # etat etoile droit (symetrique)
        fac = rR*(sR - unR) / (sR - sStar)
        ... return FR + sR*(Us - UR)
```

**Code.** Stateless policies in
[`include/adc/numerics/numerical_flux.hpp`](../include/adc/numerics/numerical_flux.hpp): `RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux` (all `ADC_HD`). `RusanovFlux` loops component by component with
`m.max_wave_speed`; `HLLFlux`/`HLLCFlux` share the free function `hll_speeds` (Davis estimates,
requires `m.wave_speeds`); `HLLCFlux`/`RoeFlux` additionally require `m.pressure`. The
compatibility function `rusanov_flux` (in `spatial_operator.hpp`) delegates to `RusanovFlux{}` for serial
references. The flux is passed by template: `compute_face_fluxes<Limiter, NumericalFlux, Model>` and
`assemble_rhs<Limiter, NumericalFlux, Model>` are templated on the flux policy, chosen
independently of the limiter. The `SourceFreeModel` adapter (explicit IMEX half-step) forwards
`pressure` and `wave_speeds` only if the wrapped model exposes them (`requires` clause), so that an
IMEX half-step stays on an HLLC flux.

**Constraints / remarks.** `RusanovFlux` is the only flux compatible with the minimal `PhysicalModel`
(it reads only `max_wave_speed`): it is the robust default for scalar transport, at the cost of an
increased diffusion ($\alpha$ upper bound). `HLLFlux` still smooths the contact discontinuity (a single
star region). `HLLCFlux` and `RoeFlux` assume `n_vars == 4` (Euler 2D); undefined behavior on
other models. HLLC on a vacuum state (zero density) divides by zero in the star factor and
needs an upstream safeguard. Roe uses `std::sqrt` for the $\sqrt{\rho}$ average (device-clean
under Kokkos/nvcc); its key property $F_R - F_L = \tilde A\,(U_R - U_L)$ gives the exact upwind flux in
the supersonic regime, and the Harten fix avoids the non-physical expansion at the sonic point.
Validation: `test_roe_flux` (consistency $\hat F(U,U) = F(U)$, exact resolution of a linearized Riemann,
`eigenvalues()` of Euler). The `aux` coupling enters through the flux (`F` reads `aux`) or through the
source, under the same spatial operator; the same flux policies serve in Cartesian, polar and cut-cell EB.


---

## 3. MUSCL reconstruction (order 2) and WENO5-Z (order 5)

**Intuition.** First-order Godunov (section 1) replaces the profile of each cell by its average, which
is very diffusive. MUSCL reconstructs a linear profile per cell from a limited slope, then
evaluates the numerical flux on the values reconstructed at the faces; the limiter clips the slope near
extrema to stay TVD (no spurious oscillation). WENO5-Z reaches order 5 in smooth regions via a
nonlinear average of three order-3 reconstructions, without an explicit limiter, by discarding the stencil
that crosses a sharp front (the ring edge).

**Formula / discretization.** A reconstruction policy is pointwise: it takes the two non-centered finite
differences around cell $i$,

$$a = U_i - U_{i-1} \quad (\text{difference arriere}),\qquad b = U_{i+1} - U_i \quad (\text{difference avant}),$$

and returns a limited slope $\sigma_i = \mathrm{lim}(a,b)$. The three MUSCL limiters are:

$$\mathrm{minmod}(a,b) = \begin{cases} \mathrm{sgn}(a)\,\min(|a|,|b|) & ab>0\\ 0 & ab\le 0\end{cases},
\qquad
\mathrm{vanleer}(a,b) = \begin{cases} \dfrac{2ab}{a+b} & ab>0\\ 0 & ab\le 0\end{cases},$$

and $\mathrm{NoSlope}(a,b)=0$ (order 1, piecewise constant). van Leer is the harmonic mean of the
two differences: less dissipative at smooth extrema than minmod (which falls back to local order 1 on a
peak). The reconstructed states at the faces of interface $i+1/2$ are then

$$U_L = U_i + \tfrac12\,\sigma_i,\qquad U_R = U_{i+1} - \tfrac12\,\sigma_{i+1},$$

passed to the numerical flux $\hat F(U_L,U_R)$. MUSCL requires 2 ghosts (slope at $i\pm 1$).

WENO5-Z reconstructs the value at the face between $v_0$ and $v_{+1}$ from the 5-point stencil
$(v_{-2},v_{-1},v_0,v_{+1},v_{+2})$. Three order-3 reconstructions:

$$q_0 = \tfrac{2v_{-2}-7v_{-1}+11v_0}{6},\quad
  q_1 = \tfrac{-v_{-1}+5v_0+2v_{+1}}{6},\quad
  q_2 = \tfrac{2v_0+5v_{+1}-v_{+2}}{6},$$

the Jiang-Shu smoothness indicators:

$$\beta_0 = \tfrac{13}{12}(v_{-2}-2v_{-1}+v_0)^2 + \tfrac14(v_{-2}-4v_{-1}+3v_0)^2,$$
$$\beta_1 = \tfrac{13}{12}(v_{-1}-2v_0+v_{+1})^2 + \tfrac14(v_{-1}-v_{+1})^2,$$
$$\beta_2 = \tfrac{13}{12}(v_0-2v_{+1}+v_{+2})^2 + \tfrac14(3v_0-4v_{+1}+v_{+2})^2,$$

and the WENO-Z weights (Borges 2008), with $\tau_5 = |\beta_0-\beta_2|$ and optimal linear weights
$d_0=\tfrac{1}{10}, d_1=\tfrac{6}{10}, d_2=\tfrac{3}{10}$:

$$\alpha_k = d_k\Big(1 + \big(\tfrac{\tau_5}{\varepsilon+\beta_k}\big)^2\Big),
\qquad
\omega_k = \frac{\alpha_k}{\alpha_0+\alpha_1+\alpha_2},
\qquad
v_{i+1/2} = \sum_{k=0}^{2}\omega_k\, q_k .$$

In a smooth region $\tau_5 \to 0$ and $\omega_k \to d_k$: we recover order 5. The measure $\tau_5$ based on
$|\beta_0-\beta_2|$ makes WENO-Z less dissipative than classical Jiang-Shu, hence better at preserving the
growth rate of a smooth mode. 5-point stencil -> 3 ghosts. For the $-x$ face, we call the same
function with the reversed stencil $(v_{+2},v_{+1},v_0,v_{-1},v_{-2})$.

```
function reconstruct_muscl(U, i, lim):        # face i+1/2, pente limitee
    a   <- U[i]   - U[i-1]                     # difference arriere
    b   <- U[i+1] - U[i]                        # difference avant
    sig <- lim(a, b)                            # minmod / vanleer / NoSlope (=0)
    a2  <- U[i+1] - U[i]
    b2  <- U[i+2] - U[i+1]
    sig_r <- lim(a2, b2)
    U_L <- U[i]   + 0.5 * sig                   # etat gauche reconstruit
    U_R <- U[i+1] - 0.5 * sig_r                 # etat droit reconstruit
    return (U_L, U_R)

function minmod(a, b):
    if a*b <= 0: return 0
    fa <- |a| ; fb <- |b|                       # valeur absolue sans std::abs
    return a if fa < fb else b

function vanleer(a, b):
    ab <- a*b
    if ab <= 0: return 0
    return 2*ab / (a + b)                        # moyenne harmonique

function weno5z(vm2, vm1, v0, vp1, vp2):        # face entre v0 et vp1
    eps <- 1e-40
    q0  <- ( 2*vm2 - 7*vm1 + 11*v0) / 6          # 3 recon. d'ordre 3
    q1  <- (  -vm1 + 5*v0  +  2*vp1) / 6
    q2  <- ( 2*v0  + 5*vp1 -    vp2) / 6
    b0  <- 13/12*(vm2-2*vm1+v0)^2 + 1/4*(vm2-4*vm1+3*v0)^2   # indicateurs beta
    b1  <- 13/12*(vm1-2*v0+vp1)^2 + 1/4*(vm1-vp1)^2
    b2  <- 13/12*(v0-2*vp1+vp2)^2 + 1/4*(3*v0-4*vp1+vp2)^2
    tau5 <- |b0 - b2|                            # ternaire device-safe, pas std::abs
    a0  <- (1/10)*(1 + (tau5/(eps+b0))^2)        # poids WENO-Z non normalises
    a1  <- (6/10)*(1 + (tau5/(eps+b1))^2)
    a2  <- (3/10)*(1 + (tau5/(eps+b2))^2)
    inv <- 1 / (a0 + a1 + a2)
    return (a0*q0 + a1*q1 + a2*q2) * inv

# Cote operateur spatial (spatial_operator::reconstruct) :
#   n_ghost == 1 (NoSlope)  -> Godunov ordre 1, pas de lecture a +/-2
#   n_ghost == 2 (MUSCL)    -> reconstruct_muscl avec le limiteur
#   n_ghost >= 3 (Weno5)    -> weno5z(stencil direct) pour face +dir,
#                              weno5z(stencil renverse) pour face -dir
```

**Code.** Pointwise `Limiter` policies in
[`include/adc/numerics/reconstruction.hpp`](../include/adc/numerics/reconstruction.hpp): `NoSlope`
(`n_ghost = 1`, `operator()` returns `Real(0)`), `Minmod` and `VanLeer` (`n_ghost = 2`, `operator()(a,b)`
returns the limited slope, absolute value coded by hand to stay device-safe without `<cmath>`), `Weno5`
(`n_ghost = 3`, a tag whose `operator()` is a no-op that just satisfies the `Limiter` concept). The
order-5 reconstruction lives in the free function `weno5z(vm2, vm1, v0, vp1, vp2)` of the same header:
it returns the value at the face between `v0` and `vp1`, and for the opposite face one passes it the
reversed stencil. All are `ADC_HD` (device-callable, static polymorphism: the limiter is a template
parameter of `assemble_rhs` / `compute_face_fluxes`, inlined on device). The mesh stencil access and the
routing by `n_ghost` are in `reconstruct` of `numerics/spatial_operator.hpp`; the policy itself
loops over no grid. The reconstruction can act on the conserved or primitive variables
(`rho, u, p`) depending on the block.

**Constraints / remarks.** The reconstruction does not change the hyperbolic stability condition: the
step stays bounded by the CFL of section 1, `dt <= C dx / max|lambda|`. Limits and pitfalls:
- `Minmod` is strictly TVD but falls back to local order 1 at extrema (it erases smooth peaks);
  for the Diocotron growth modes one prefers `VanLeer`, less dissipative at extrema.
- `weno5z` is smooth (no branch on the sign: the $\beta_k$ and $\tau_5$ are squares so
  always $\ge 0$, and only $|\beta_0-\beta_2|$ goes through a ternary), which makes it fully
  device-callable; the floor `eps = 1e-40` avoids division by zero on a constant stencil.
- Reconstructing the conserved variable rather than the primitive changes the behavior at strong shocks
  (the reconstructed states can leave the admissible domain on the conserved side).
- The ghost cost drives the halo width to exchange: 1 (NoSlope), 2 (MUSCL), 3 (WENO5).

**Validation.** `test_weno_convergence` (the face reconstruction of a smooth function reaches order 5),
`test_primitive_recon` (conserved <-> primitive conversions and their use in the reconstruction),
`test_spatial_discretisation` (the reconstruction x numerical flux pair is a named type, exercised end
to end).


---

## 4. Time integration: SSPRK, object integrators, user integrator

**Intuition.** Strong-Stability-Preserving Runge-Kutta: each stage is a convex combination
of explicit Eulers, so any stability property (TVD, positivity, bounds) held by a forward
Euler step under CFL is held by the whole scheme, at order 2 or 3. The time scheme is a
first-class object (`take_step`) that the coupler calls, rather than inline SSPRK
duplicated in each coupler; the same contract lets a case bring its own integrator.

**Formula / discretization.** Method of lines: the space gives $\dot U = L(U)$ with
$L(U) = -\mathrm{div} F(U) + S(U)$, evaluated by $\texttt{rhs}(U, R) \Rightarrow R = L(U)$.
Forward Euler: $U^{n+1} = U^n + \Delta t\, L(U^n)$.

SSPRK2 (Shu-Osher, 2 stages, order 2, equivalent to Heun):

$$U^{(1)} = U^n + \Delta t\, L(U^n), \qquad U^{n+1} = \tfrac12 U^n + \tfrac12\big(U^{(1)} + \Delta t\, L(U^{(1)})\big).$$

SSPRK3 (Shu-Osher, 3 stages, order 3):

$$U^{(1)} = U^n + \Delta t\, L(U^n),$$
$$U^{(2)} = \tfrac34 U^n + \tfrac14\big(U^{(1)} + \Delta t\, L(U^{(1)})\big),$$
$$U^{n+1} = \tfrac13 U^n + \tfrac23\big(U^{(2)} + \Delta t\, L(U^{(2)})\big).$$

Both have SSP coefficient $C = 1$: the SSP condition is exactly the forward Euler CFL condition.
In $\texttt{MultiFab}$ operations the code uses only $\texttt{saxpy}(Y, a, X): Y \mathrel{+}= a\,X$ and
$\texttt{lincomb}(Y, a, X_1, b, X_2): Y \leftarrow a\,X_1 + b\,X_2$. The convex stage of SSPRK2 then
writes as an Euler update on the copy $U^{(1)}$ followed by
$\texttt{lincomb}(U, \tfrac12, U, \tfrac12, U^{(1)})$, algebraically identical to the convex form above.

```
function take_step_SSPRK2(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)   # scratch, aucun etat porte
    rhs(U, R)                                          # R = L(U^n)
    U1 = copy(U)
    saxpy(U1, dt, R)                                   # U1 = U^n + dt L(U^n)  (= U^(1))
    rhs(U1, R)                                         # R = L(U^(1))
    saxpy(U1, dt, R)                                   # U1 = U^(1) + dt L(U^(1))
    lincomb(U, 1/2, U, 1/2, U1)                        # U^{n+1} = 1/2 U^n + 1/2 U1

function take_step_SSPRK3(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)
    rhs(U, R);  U1 = copy(U);  saxpy(U1, dt, R)        # U^(1) = U^n + dt L(U^n)
    rhs(U1, R); U2 = copy(U1); saxpy(U2, dt, R)
    lincomb(U2, 3/4, U, 1/4, U2)                       # U^(2) = 3/4 U^n + 1/4 (U^(1)+dt L)
    rhs(U2, R); U3 = copy(U2); saxpy(U3, dt, R)
    lincomb(U, 1/3, U, 2/3, U3)                       # U^{n+1} = 1/3 U^n + 2/3 (U^(2)+dt L)

# tag -> politique d'emploi par bloc d'equation (pas le schema lui-meme)
struct TimePolicy<Method, Treatment in {Explicit,Implicit,IMEX,Prescribed}, Substeps>=1, Stride>=1>
TimePolicyTraits<P>: extrait (Method, treatment, substeps, stride), defaut Explicit/1/1 sur un tag nu
ExplicitTime<M=SSPRK2,...> / ImplicitTime<...> / IMEXTime<...> / PrescribedTime  # alias de TimePolicy

# integrateur utilisateur : tout objet qui satisfait le concept
concept TimeStepper<I> = I.take_step(rhs, U, dt) compile
```

**Code.** Two expressions coexist, separating the mathematical scheme from its usage policy.
The tags [`include/adc/numerics/time/time_integrator.hpp`](../include/adc/numerics/time/time_integrator.hpp)
(`SSPRK2`, `SSPRK3`, `UserTimeIntegrator`) name, per block, the temporal treatment via a
`TimePolicy<Method, TimeTreatment, Substeps, Stride>`; `TimePolicyTraits` reads these fields (and accepts
a bare tag, then treated as `Explicit` with a single step). The aliases `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime` set the `TimeTreatment`. The object integrators
[`include/adc/numerics/time/time_steppers.hpp`](../include/adc/numerics/time/time_steppers.hpp)
(`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`) carry the method: each exposes
`take_step(rhs, U, dt)` and allocates its scratch (`R`, stages `U1`/`U2`/`U3`) only from the layout of
`U`, with no persistent state. The integrator sees only `rhs(U_stage, R)` (the method-of-lines arrow)
and the `saxpy`/`lincomb` operations of [`include/adc/mesh/mf_arith.hpp`](../include/adc/mesh/mf_arith.hpp):
it is agnostic of the model and of the discretization. The `TimeStepper` concept formalizes the contract, so
that a case can provide its own `take_step` object exactly as it provides a `PhysicalModel`.

**Constraints / remarks.** SSP with coefficient 1: strong stability is guaranteed only as long as
$\Delta t$ respects the forward Euler CFL ($C = 1$, no margin gained on the step compared to the
order-1 scheme). In a coupled system, the order of the elliptic solve caps the global order: a Poisson
solved once per step limits the field to order 1, whatever SSPRK is chosen on the hyperbolic side
(see splitting, section 6). The `Substeps` fields (more frequent substeps, $n$ steps of $\Delta t/n$
for a fast species) and `Stride` (slower cadence, a slow block advances only one macro-step every
`Stride` steps) are orthogonal and belong to the scheduler, not the integrator (section 7); `Stride = 1`
gives back the historical behavior. The `SSPRK2Step`/`SSPRK3Step` objects reproduce bit-for-bit the
old inline copies `SystemCoupler::advance_explicit_ssprk2/ssprk3` (deduplication). Validation:
`test_user_time_integrator` checks that a user-provided integrator gives the same result
as a core SSPRK.


---

## 5. Stiff sources: asymptotic-preserving IMEX and partial IMEX

**Intuition.** A stiff source (fast relaxation, Lorentz force, Debye screening `lambda_D -> 0`)
forces the explicit scheme to a `dt` of the same order as the stiffness, hence impractical. IMEX treats
transport explicitly and the stiff source implicitly (stable at fixed `dt`). The
asymptotic-preserving (AP) property guarantees that, when the small parameter `eps` (= `lambda_D^2`,
`1/omega_c`, ...) tends to 0, the scheme stays consistent and stable at fixed `dt` and captures the
limit dynamics (equilibrium, quasi-neutrality) without resolving the stiff scale.

**Formula / discretization.** On `dU/dt = T(U) + S(U)` where `S` carries the stiff part, an IMEX
Euler step (forward-backward) treats `T` explicitly and `S` implicitly:

$$U^{n+1} = U^n + \Delta t\,T(U^n) + \Delta t\,S(U^{n+1}).$$

We decompose it into two in-place operators. First the explicit transport produces the known member
$\tilde U = U^n + \Delta t\,T(U^n)$, then the implicit step solves $W = \tilde U + \Delta t\,S(W)$.
When `S` is a linear relaxation `S(U) = -(1/eps)(U - U_eq)`, the solve is analytic and
unconditionally stable; the limit `eps -> 0` gives `W -> U_eq` (equilibrium manifold `S(U)=0`)
without the constraint `dt < eps`. The scalar implicit step (per cell) is solved by the Newton residual

$$F(W) = W - \tilde U - \Delta t\,S(W) = 0,\qquad J = I - \Delta t\,\frac{\partial S}{\partial W},$$

exact in one iteration if `S` is linear in `U`, quadratic otherwise. The Jacobian is formed by
finite differences (no analytic Jacobian to provide on the model side).

**Partial IMEX.** When only a subset of the variables is stiff, we integrate implicitly only
those components. The solve becomes a forward-backward Euler per component: the explicit components
advance by forward Euler at the input state, `W_e = U^n_e + dt S_e(U^n)`, then the implicit components
are solved by Newton on the reduced `n x n` subsystem (`n` = number of implicit ones `<= N`), the
explicit ones frozen at their advanced value as known data. The partitioning comes either from the model
(trait `is_implicit(c)`), or from a mask carried by the block (priority over the model default), which
allows reusing the same model with different treatments depending on the block.

```
function imex_euler_step(U, dt, Texpl, Simpl):
    Texpl(U, dt)            # explicite en place : U <- U^n + dt*T(U^n) (membre connu)
    Simpl(U, dt)            # implicite en place : resout U <- W tel que W = U + dt*S(W)

# pas implicite par cellule (Newton local, IMEX partiel), N = Model::n_vars
function newton_source_solve(model, Un, aux, dt, iters, mask):
    impl <- liste des c dans [0,N) tels que is_implicit_component(mask, c)   # m = |impl| <= N
    W <- Un
    # (1) composantes explicites : Euler avant a l'etat d'entree
    if m < N:
        S_in <- model.source(Un, aux)
        for c not in impl: W[c] <- Un[c] + dt * S_in[c]
    # (2) composantes implicites : Newton sur le sous-systeme reduit m x m
    for it in 0..iters-1:
        S0 <- model.source(W, aux)
        for r in 0..m-1:                          # residu F = W - Un - dt*S(W)
            c <- impl[r];  F[r] <- W[c] - Un[c] - dt * S0[c]
        for cc in 0..m-1:                          # jacobienne par differences finies, colonne par colonne
            col <- impl[cc];  h <- 1e-7*|W[col]| + 1e-7
            Wp <- W;  Wp[col] += h;  Sp <- model.source(Wp, aux)
            for rr in 0..m-1:
                row <- impl[rr];  dSdW <- (Sp[row] - S0[row]) / h
                J[rr][cc] <- (row==col ? 1 : 0) - dt * dSdW       # I - dt*(dS/dW)
        solve_dense(J, F, delta, m)                # Gauss + pivot partiel, tableau fixe N, device-callable
        for r in 0..m-1: W[impl[r]] -= delta[r]
    return W

# stepper de bloc : pas implicite sur la source locale du modele, en place sur tout le MultiFab
function backward_euler_source(model, aux, U, dt, iters, mask):
    for chaque fab local de U:
        for_each_cell(box, BackwardEulerSourceKernel:
            Un <- load_state(U, i, j);  a <- load_aux(aux, i, j)
            W  <- newton_source_solve(model, Un, a, dt, iters, mask)
            U(i,j,:) <- W)
```

**Code.** [`include/adc/numerics/time/imex.hpp`](../include/adc/numerics/time/imex.hpp):
`imex_euler_step(U, dt, Texpl, Simpl)` chains the in-place explicit transport then the in-place implicit
source solve (two callables `TransportStep` / `ImplicitSourceSolve`). The implicit step lives in
[`include/adc/numerics/time/implicit_stepper.hpp`](../include/adc/numerics/time/implicit_stepper.hpp):
`newton_source_solve<Model>` (local per-cell Newton, forward-backward Euler for the partial IMEX),
`detail::solve_dense<N>` (dense `n x n` resolution by Gauss elimination with partial pivoting, a
fixed constexpr array hence device-callable, no allocation), and `backward_euler_source<Model>` which applies
the kernel `detail::BackwardEulerSourceKernel<Model>` via `for_each_cell` (a named functor and not an
extended lambda, for robust device emission from an external TU). The implicit/explicit partitioning
goes through the `PartiallyImplicitModel` concept (trait `M::is_implicit(c)`), `model_is_implicit<Model>`
(default: everything implicit when the trait is absent), the POD carrier `ImplicitMask<N>` (`active`, `flag[N]`,
carried by the block, passed by value on the device) and `is_implicit_component<Model, N>` (an active mask
with priority over the model default). `ImplicitSourceStepper` (`iters = 2`) models the concept
`ImplicitBlockStepper` and wires `backward_euler_source` into the implicit-advance callback of the
`SystemCoupler`, which itself advances the explicit SSPRK blocks and delegates the implicit / IMEX blocks.

**Constraints / remarks.** The implicit step is unconditionally stable for a linear relaxation
(where a plain Picard fixed point would diverge as soon as `dt * stiffness > 1`, precisely the
stiff regime); it is exact in one iteration if `S` is linear in `U`, quadratic convergence otherwise
(default `iters = 2`). The finite-difference Jacobian uses a step `h = 1e-7 |W_col| + 1e-7`.
Limits: `imex_euler_step` is first order in time (forward-backward Euler); the AP covers the relaxation
limit, not the condensation of the potential-velocity-Lorentz couplings at high `omega_c`, which is the
domain of Schur condensation (section 13). Inactive mask and a model without the `is_implicit` trait:
everything is implicit (full backward-Euler), strictly bit-identical to the historical behavior. The
transport of an IMEX block stays advanced explicitly by the core. Validation:
`test_imex_ap` (AP property on a stiff linear relaxation source),
`test_ap_limit` (quantified AP limit, stiffness sweep over 8 decades at fixed `dt`),
`test_imex_partial` (a 2-variable model, only one implicit),
`test_imex_transport` (the transport of an IMEX block is indeed advanced explicitly).


---

## 6. Operator splitting: Lie and Strang

**Intuition.** When the RHS is a sum of operators with different behavior (transport + stiff
source + cyclotron rotation), we apply them in sequence rather than simultaneously: each
sub-operator keeps its own integrator, hence its own stiffness, without contaminating the other. Lie
(Godunov, order 1) chains the flows; Strang (order 2) symmetrizes the sequence around the central flow
to cancel the dominant error term.

**Formula / discretization.** We decompose

$$\frac{\mathrm{d}U}{\mathrm{d}t} = T(U) + S(U)$$

denoting $\Phi^T_{\tau}$ and $\Phi^S_{\tau}$ the exact flows (or approximate to the desired order) of
$\dot U = T(U)$ and $\dot U = S(U)$ over an interval $\tau$. Lie splitting applies one then
the other on the full step:

$$U^{n+1} = \Phi^S_{\Delta t}\big(\Phi^T_{\Delta t}(U^n)\big)$$

Strang splitting brackets the transport flow with two source half-steps:

$$U^{n+1} = \Phi^S_{\Delta t/2}\Big(\Phi^T_{\Delta t}\big(\Phi^S_{\Delta t/2}(U^n)\big)\Big)$$

The order reads off the Baker-Campbell-Hausdorff formula. The Lie composite flow equals
$\exp(\Delta t\,T)\exp(\Delta t\,S) = \exp\big(\Delta t (T+S) + \tfrac{\Delta t^2}{2}[T,S] + \dots\big)$:
the per-step error is $O(\Delta t^2)$, carried by the commutator $[T,S] = TS - ST$, hence global order 1.
The Strang symmetrization cancels the $\Delta t^2$ term: the per-step error drops to $O(\Delta t^3)$,
that is global order 2. Strang is order 2 as soon as each sub-integrator is itself; if $T$ and $S$
commute ($[T,S]=0$) the splitting is exact to all orders. The extra cost of Strang over Lie is a single
extra source half-step per macro-step (two $S(\Delta t/2)$ instead of one $S(\Delta t)$).

```
function lie_step(U, dt, T, S):
    # T, S : callables (MultiFab&, Real) -> void, avancent leur sous-systeme en place
    T(U, dt)                 # transport sur le pas plein
    S(U, dt)                 # source sur le pas plein
    # U contient maintenant U^{n+1}, ordre 1

function strang_step(U, dt, T, S):
    S(U, 0.5 * dt)           # demi-pas source
    T(U, dt)                 # pas plein transport (flot central)
    S(U, 0.5 * dt)           # demi-pas source symetrique
    # U contient maintenant U^{n+1}, ordre 2 si S et T sont chacun >= ordre 2
```

**Code.** The two generic bricks are in
[`include/adc/numerics/time/splitting.hpp`](../include/adc/numerics/time/splitting.hpp):
`lie_step(MultiFab& U, Real dt, TransportStep T, SourceStep S)` and
`strang_step(...)`. Both are templated on `TransportStep` / `SourceStep`: $T$ and $S$ are
callables `(MultiFab&, Real) -> void` that advance their subsystem in place, so the integrator is
agnostic of the physical content (in-house counterpart of `StrangSplitting` / `FractionalTime2OSplitting` of
muffin). The production orchestrator exposes them via `SplitScheme::Lie` / `Strang`
(`runtime/system_stepper.hpp`: `SystemStepper::step` for Lie, `step_strang` for Strang, #217), where
the transport phase and the source phase (explicit, IMEX, or Schur-condensed stage, sections 5 and 13)
are symmetrized at $\Delta t/2$ around the central transport.

**Constraints / remarks.** Strang gives order 2 only if each sub-step is itself at least
order 2: an order-1 $S$ or $T$ caps the splitting at order 1, whatever the symmetrization. In a
hyperbolic-elliptic couple, consistency requires re-solving the elliptic between the source half-steps:
otherwise the second half-step $S(\Delta t/2)$ reads a stale $\phi$ (the field of the previous half-step) and
order 2 falls (see the solve call count in the validation). The step $\Delta t$ stays subject to
the CFL of the transport $T$; the splitting does not relax this constraint, it only decouples the
stiffnesses so a stiff source can be treated implicitly (IMEX / Schur) without imposing its own
tiny $\Delta t$ on the transport. Validation: `test_splitting` measures the order of the bricks
`lie_step` / `strang_step` on a non-commuting linear 2x2 system whose exact flow is known (Lie
order 1, Strang order 2 read off the slope). `test_strang_splitting` redoes the same order measurement on the
real orchestrator (`SystemStepper::step` vs `step_strang`), plus a count of the calls to the elliptic
solve that locks the $\phi$ consistency. On the Python side: `test_strang_split`.


---

## 7. Multirate: subcycling, cadence, adaptive step

**Intuition.** Not all species of a coupled system require the same time step.
A stiff species (electrons) splits a macro-step into several substeps ($\text{substeps}$); a
slow species (under-resolved gas) is advanced only once every $M$ macro-steps (cadence, $\text{stride}$),
and then catches up $M$ steps in a single advance. The two mechanisms are orthogonal and read from the
temporal policy of each block; the scheduler knows no physics.

**Formula / discretization.** Each block carries a `TimePolicy<Method, Treatment, substeps, stride>`
from which the scheduler extracts only three integers (and the treatment, to skip prescribed blocks).
Let $\text{dt}$ be the macro-step, $n = \text{substeps}_b$, $m = \text{stride}_b$ for block $b$.

Cadence: block $b$ is held (hold) as long as macro-step $k$ verifies $(k+1) \bmod m \neq 0$, then
it catches up at the end of the window with an effective step

$$\Delta t^{\text{eff}}_b = m \, \text{dt}.$$

Over $M$ macro-steps, the block advances $M/m$ times by a step $m\,\text{dt}$: its total time stays $M\,\text{dt}$,
but it is solved only $M/m$ times (the coupling is loose for this block). Subcycling splits this
effective step into $n$ equal substeps

$$h = \frac{\Delta t^{\text{eff}}_b}{n} = \frac{m \, \text{dt}}{n}.$$

With $m = 1$ and $n = 1$ we recover identically the advance of a step $\text{dt}$ at each macro-step.

The macro-step can be chosen by the CFL via `step_cfl`. The stability condition bears on the
real substep $m\,\text{dt}/n \le \text{cfl}\, h_{\text{cell}} / w_b$, which gives per block

$$\text{dt}_b = \frac{\text{cfl} \; h_{\text{cell}} \; \text{substeps}_b}{\text{stride}_b \; w_b},
\qquad \text{dt} = \min_{b \,\text{evolutif}} \text{dt}_b,$$

where $h_{\text{cell}} = \min(dx, dy)$ in Cartesian, $\min(dr, r_{\min}\, d\theta)$ in polar (the
physical azimuthal step is minimal at the inner radius), and $w_b$ is the max wave speed of the block.

The variant `step_adaptive` fixes the macro-step on the slowest block,
$\Delta t = \text{cfl}\, h_{\text{cell}} / w_{\min}$, and subcycles each faster block

$$n_b = \left\lceil \text{stride}_b \; \frac{w_b}{w_{\min}} \right\rceil$$

times over its effective step $\Delta t^{\text{eff}}_b = m\,\Delta t$; aux is frozen on the macro-step
(once-per-step coupling).

```
function advance_subcycled(system, dt, macro_step, advance_block):
    for each block in system:                    # for_each_block, ordre stable
        if time_treatment(block) == Prescribed:
            continue                             # pilote par l'utilisateur, hors scheduler
        m = stride(block)
        if macro_step mod m != 0:
            continue                             # bloc lent : tenu ce macro-pas
        n = substeps(block)
        h = dt * m / n                           # pas effectif (catch-up) decoupe en n sous-pas
        for s in 0 .. n-1:
            advance_block(block, h, s, n)        # callable utilisateur, 1 sous-pas

# surcharge historique : macro_step = 0 -> stride toujours satisfait, tous les blocs avancent
function advance_subcycled(system, dt, advance_block):
    advance_subcycled(system, dt, 0, advance_block)

function step_cfl(cfl):                           # SystemDriver, choix du macro-pas par CFL
    solve_fields()                                # aux (phi, grad) a l'instant courant
    h_cell = polar ? min(dr, r_min*dtheta) : min(dx, dy)
    dt = +inf
    for each block b, evolutif:
        w   = max(max_wave_speed(b.U), 1e-30)     # all_reduce_max sous MPI
        dt  = min(dt, cfl * h_cell * substeps_b / (stride_b * w))
    if dt not finite: dt = cfl * h_cell / 1e-30   # tous geles : pas degenere
    for each block b, evolutif:
        if (macro_step+1) mod stride_b != 0: continue   # hold ; sinon rattrapage
        eff_dt = dt * stride_b
        advance_transport(b, eff_dt)              # substeps_b sous-pas internes
        run_source_stage(b, eff_dt)               # etage source Schur opt-in (no-op sinon)
    apply_couplings(dt); t += dt; macro_step += 1
    return dt

function step_adaptive(cfl):                       # macro-pas = pas du bloc le plus lent
    solve_fields()
    for each block b: w_b = b.evolve ? max_wave_speed(b.U) : 0
    w_min   = min over evolutifs of w_b           # 1e-30 si tous geles
    h_cell  = polar ? min(dr, r_min*dtheta) : min(dx, dy)
    macro_dt = cfl * h_cell / w_min
    for each block b, evolutif:
        if (macro_step+1) mod stride_b != 0: continue
        n      = max(1, ceil(stride_b * w_b / w_min))   # sous-cycles pour rester stable
        eff_dt = macro_dt * stride_b
        advance_transport_n(b, eff_dt, n)
        run_source_stage(b, eff_dt)
    apply_couplings(macro_dt); t += macro_dt; macro_step += 1
    return macro_dt
```

**Code.** The skeleton is [`numerics/time/scheduler.hpp`](../include/adc/numerics/time/scheduler.hpp),
function `advance_subcycled` (two overloads: with and without `macro_step`). It reads
`block_substeps_v`, `block_stride_v` and `block_time_treatment_v`, aliases of `TimePolicyTraits`
defined in [`numerics/time/time_integrator.hpp`](../include/adc/numerics/time/time_integrator.hpp)
(`TimePolicy<Method, Treatment, substeps, stride>`, aliases `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime`). A `TimeTreatment::Prescribed` block is skipped (the guard
`!= Prescribed`). The step choice lives in
[`runtime/system_stepper.hpp`](../include/adc/runtime/system_stepper.hpp): `step_cfl`,
`step_adaptive`, and the helper `stride_due(macro_step, stride)` that materializes the end of window
$(k+1)\bmod m = 0$. The speed $w_b$ comes from `max_wave_speed_mf`
([`numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp)), collective
`all_reduce_max` under MPI so that all ranks pick the same $\text{dt}$.

**Constraints / remarks.** `step_cfl` is substeps-aware since #121: the formula
$\text{dt} = \text{cfl}\,h\,\text{substeps}/(\text{stride}\,w)$ gives, for $\text{substeps}_b > 1$, a
step $\text{substeps}_b$ times larger than the old formula $\text{dt} = \text{cfl}\,h/(\text{stride}\,w)$.
Bit-identical parity with the history therefore holds only for $\text{substeps} = 1$ (at any
stride); to replay a run calibrated on the old formula, pass the explicit historical $\text{dt}$
to `step(dt)`, not `step_cfl`. Under MPI, the absence of `all_reduce_max` would desynchronize the
ranks (each would see the max of its own boxes only) and would make the simulation diverge. The
stride semantics is hold-then-catch-up: the slow block is loosely coupled, which is an assumed choice
(the gas is not resolved at every step). Tests: `test_multirate_stride` (slow species advanced once
out of $N$), `test_adaptive_multirate` (`step_adaptive`, macro-step fixed by the most
constraining species), `test_cfl_dt` (`step_cfl` multi-species).

## 8. Parabolic term: diffusion as face flux

**Intuition.** A parabolic term $+\nu\,\Delta U$ (diffusion, isotropic scalar viscosity) is the
divergence of a Fickian flux $F_{\text{diff}} = -\nu\,\nabla U$. Writing it as a face flux rather
than a direct Laplacian makes it AMR-compatible: reflux sees it and corrects it at the
fine-coarse interface exactly like a hyperbolic flux, so diffusion stays conservative at level
junctions.

**Formula / discretization.** The continuous Fickian flux $F_{\text{diff}} = -\nu\,\nabla U$ adds to the
hyperbolic numerical flux before the divergence. As a face flux (centered gradient at the face, cell
values, not the level $h$):

$$F^{x}_{i+1/2,j} = -\nu\,\frac{U_{i+1,j} - U_{i,j}}{dx}, \qquad
  F^{y}_{i,j+1/2} = -\nu\,\frac{U_{i,j+1} - U_{i,j}}{dy}.$$

The divergence $-\big(F^{x}_{i+1/2} - F^{x}_{i-1/2}\big)/dx - \big(F^{y}_{j+1/2} - F^{y}_{j-1/2}\big)/dy$
gives back exactly the 5-point Laplacian:

$$+\nu\,\Delta_h U_{i,j} = \nu\left(
  \frac{U_{i+1,j} - 2U_{i,j} + U_{i-1,j}}{dx^2}
+ \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j-1}}{dy^2}\right),$$

added component by component to the residual $R = -\mathrm{div}\hat F + S$. The core `assemble_rhs`
writes this 5-point stencil directly (the AMR-less path); `compute_face_fluxes` produces the
face-flux form (the AMR reflux path). The two give a residual bit-identical to the machine.

```
# coeur : assemble_rhs, terme additif au residu (5 points), garde par DiffusiveModel
function diffusive_residual_term(model, u, i, j, dx, dy):
    if not DiffusiveModel(model):                 # if constexpr : zero codegen sinon
        return                                     # chemin hyperbolique strictement intouche
    nu   = model.diffusivity()
    idx2 = 1/(dx*dx);  idy2 = 1/(dy*dy)
    for c in 0 .. n_vars-1:
        lap = (u(i+1,j,c) - 2*u(i,j,c) + u(i-1,j,c)) * idx2
            + (u(i,j+1,c) - 2*u(i,j,c) + u(i,j-1,c)) * idy2
        r(i,j,c) += nu * lap                       # +nu Lap(U)

# AMR : compute_face_fluxes, flux de face Fickien ajoute au flux hyperbolique
function face_flux_x(model, u, aux, i, j, dx):     # face entre (i-1,j) et (i,j)
    L = reconstruct(model, u, i-1, j, dir=0, +1)   # etats reconstruits
    R = reconstruct(model, u, i,   j, dir=0, -1)
    F = numerical_flux(model, L, aux(i-1,j), R, aux(i,j), dir=0)   # hyperbolique
    if DiffusiveModel(model):
        nu = model.diffusivity()
        for c in 0 .. n_vars-1:
            F[c] += -nu * (u(i,j,c) - u(i-1,j,c)) / dx     # flux Fickien centre au face
    fx(i,j,:) = F                                  # le reflux AMR voit ce flux -> conservatif
```

**Code.** The contract is the `DiffusiveModel` concept in
[`numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp): a model satisfies it
if and only if `m.diffusivity()` returns a `Real` ($\nu \ge 0$). The 5-point term is
added in `detail::AssembleRhsKernel::operator()` (called by `assemble_rhs`) under
`if constexpr (DiffusiveModel<Model>)`. The face-flux form lives in
`detail::FaceFluxXKernel` / `detail::FaceFluxYKernel` (called by `compute_face_fluxes`), same guard.
Both kernels are device-clean named functors (`ADC_HD`).

**Constraints / remarks.** Central invariant: a model that does not expose `diffusivity()` does not change
by a bit, the `if constexpr` being false there is no additional codegen (the hyperbolic path
strictly unchanged). The `dx`, `dy` arguments of `compute_face_fluxes` default to 0 and are
read only by the diffusive branch, so a non-diffusive model is never affected. The explicit step
on a parabolic term imposes the diffusive stability constraint
$\nu\,\Delta t \le \tfrac{1}{2}\,(dx^{-2} + dy^{-2})^{-1}$ (more restrictive in $h^2$ than the
hyperbolic CFL in $h$), not handled by `step_cfl` which only weighs the wave speed: at
diffusion-dominated, fix $\text{dt}$ explicitly. Known limit: `SourceFreeModel` (explicit half-step
IMEX) does not expose `diffusivity()`, so a diffusive IMEX block would lose its Fickian flux in the
explicit half-step (a separate refinement); and the masked path `assemble_rhs_masked` does not mask the
Laplacian. Tests: `test_diffusion` (the core $+\nu\,\Delta U$ via the divergence of the Fickian flux),
`test_amr_diffusion` (face-flux diffusion crosses the AMR reflux correctly).


---

## 9. Elliptic: geometric multigrid

**Intuition.** The Gauss-Seidel smoother quickly kills the high frequencies of the error but crawls on
the low ones. Multigrid restricts the low-frequency error onto coarser grids (where
it becomes high frequency again), smooths it, and prolongs it. Cost $O(N)$ per V-cycle, number of
cycles nearly independent of the mesh.

**Formula / discretization.** 5-point operator on $\mathrm{lap}(\phi) = f$ (isotropic case
$\epsilon = 1$, $\kappa = 0$):

$$(\mathrm{lap}\,\phi)_{ij} = \frac{\phi_{i+1,j} - 2\phi_{ij} + \phi_{i-1,j}}{\Delta x^2}
                            + \frac{\phi_{i,j+1} - 2\phi_{ij} + \phi_{i,j-1}}{\Delta y^2}$$

Red-black Gauss-Seidel smoother: one color $c \in \{0,1\}$ per sweep, the cell
$(i,j)$ with $(i+j) \bmod 2 = c$ is updated from its neighbors (of the other color, hence already
frozen on this sweep):

$$\phi_{ij} \leftarrow \frac{\mathrm{off}_{ij} - f_{ij}}{\mathrm{diag}},
\quad \mathrm{off}_{ij} = \frac{\phi_{i\pm1,j}}{\Delta x^2} + \frac{\phi_{i,j\pm1}}{\Delta y^2},
\quad \mathrm{diag} = \frac{2}{\Delta x^2} + \frac{2}{\Delta y^2}$$

V-cycle: $\nu_1$ pre-smoothing sweeps, residual $r = f - \mathrm{lap}\,\phi$, restriction of
$r$ by $2\times2$ average (`average_down`) onto the twice-coarser grid, recursive
resolution of the correction equation $\mathrm{lap}(e) = r$ with homogeneous conditions, prolongation
of $e$ (`interpolate`) added to $\phi$, $\nu_2$ post-smoothing sweeps. At the coarsest
level, `nbottom` sweeps stand in for an exact resolution (bottom solve). Defaults:
$\nu_1 = \nu_2 = 2$, `nbottom = 50`, `min_coarse = 2`.

```
function vcycle_rec(level l, bc):
    L = lev_[l]
    gs_smooth(L.phi, L.rhs, nu1, bc)              # pre-lissage : nu1 balayages rouge-noir
    if l est le plus grossier:
        gs_smooth(L.phi, L.rhs, nbottom, bc)      # bottom solve (longue serie de balayages)
        if masque: zero_conductor(L.phi)          # refige phi=0 dans le conducteur
        return
    poisson_residual(L.phi, L.rhs, -> L.res, bc)  # r = f - lap(phi) (porte aussi termes croises)
    average_down(L.res, C.rhs, ratio=2)           # restriction du residu (moyenne 2x2)
    C.phi = 0                                       # correction a CL homogenes
    vcycle_rec(l+1, homogeneous(bc))              # recursion grossiere
    corr = interpolate(C.phi, ratio=2)            # prolongation de la correction
    L.phi += corr                                  # saxpy
    if masque: zero_conductor(L.phi)
    gs_smooth(L.phi, L.rhs, nu2, bc)              # post-lissage

function solve(rel_tol, max_cycles):
    r0 = current_residual()                        # norm_inf(f - lap(phi)), all_reduce_max
    if r0 <= 0: return 0
    for c in 1..max_cycles:
        vcycle()                                   # warm-start : phi conserve entre appels
        if current_residual() <= rel_tol * r0: return c
    return max_cycles
```

The hierarchy is built by coarsening the domain by 2 down to `min_coarse`, but we stop if
a box does not coarsen cleanly: the test `b.coarsen(2).refine(2) == b` characterizes the
boxes that are aligned and of even size. On a multi-box domain (`max_grid_size < n`), the boxes
shrink by 2 at each level and would end at $1\times1$; `coarsen(ba,2)` would then make
several distinct fine boxes fall onto the same coarse cell (degenerate BoxArray), and
`average_down` would read out of bounds of a 1-cell fab. In serial the heap is stable, under MPI
it is shuffled and the read becomes erratic (occasional discrepancy up to blow-up). The break keeps the
current level as the coarsest grid; mono-box and non-degenerate multi-box never cross
this test, hierarchy and result strictly unchanged.

The red-black sweep makes each color data-independent (parallelizable). Between colors
and before the residual, `device_fence()` + `fill_ghosts` synchronize the device and fill the
halos; `current_residual` reduces the infinity norm by `all_reduce_max` (required for a
coarse multi-box distributed grid, otherwise the stopping criterion triggers at different iterations per
rank and desynchronizes the MPI fluxes). The `replicated` mode replicates each level on all
ranks (per-fab V-cycle without communication), which is what the AMR coupler expects (level 0 replicated).

**Code.** [`numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp):
`GeometricMG` models the `EllipticSolver` concept (`rhs()`, `phi()`, `solve()`, `residual()`);
`vcycle_rec` is the recursion, `solve(rel_tol, max_cycles)` iterates the cycles with warm-start (`phi`
kept between calls, 1-2 V-cycles in the established regime). The 5-point Laplacian and the smoother are
the shared bricks of [`numerics/elliptic/poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp)
(`poisson_residual`, `gs_smooth` -> `gs_rb_sweep` -> `detail::gs_color`, named ADC_HD functors
device-clean). Restriction / prolongation reuse the AMR transfer operators `average_down`
/ `interpolate` of [`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp). `solve_robust`
adds an anti-divergence safeguard (cf. below).

**Constraints / remarks.** Fully on-device (the V-cycle goes through `for_each_cell`),
AMR-compatible, accepts any `n`. No CFL constraint (stationary solver), but the
GS-5-point V-cycle assumes a diagonally dominant operator: it stays contracting for a
symmetric positive-definite operator (Poisson, $\epsilon > 0$, $\kappa \ge 0$), and can diverge on a
strongly non-symmetric operator (cross terms, cf. sections 11 and 12). At the embedded boundary at
high resolution, the non-Galerkin coarsening and the per-level re-evaluated mask sometimes make the
cycle non-contracting (spectral radius $> 1$): `solve_robust` detects true divergence (final
residual $>$ initial residual), hardens the smoothing locally to the solve ($\nu$ doubles up to 64) and
restarts cold ($\phi = 0$), strictly bit-identical when the solver already converges or stagnates.
**Validation.** `test_geometric_mg` (fast convergence nearly independent of the mesh on
manufactured solutions), `test_poisson_convergence` (quantitative order 2 of the 5-point Laplacian),
`test_solve_robust` (the anti-divergence safeguard).

## 10. Elliptic: spectral Poisson (FFT), single-rank and distributed

**Intuition.** On a periodic constant-coefficient domain, the discrete Laplacian is diagonal
in Fourier: one direct transform, one mode-by-mode division, one inverse transform solve
Poisson exactly (to machine residual), without iteration. Much cheaper than multigrid when
the elliptic dominates the run.

**Formula / discretization.** The solver inverts the same 5-point Laplacian as `GeometricMG`,
whose eigenvalue of mode $(k_x, k_y)$ is

$$\lambda(k_x, k_y) = \frac{2\cos(2\pi k_x / N_x) - 2}{\Delta x^2}
                     + \frac{2\cos(2\pi k_y / N_y) - 2}{\Delta y^2}$$

and not $-(k_x^2 + k_y^2)$ (the exact symbol of the discrete stencil, not of the continuous
Laplacian). The resolution is $\hat\phi(k) = \hat f(k) / \lambda(k)$, with the mode $k = 0$ fixed to 0
(gauge: $\phi$ of zero mean; the right-hand side must therefore be of zero mean, otherwise $\phi$ drifts).

```
function solve():                                  # PoissonFFTSolver, boite unique
    rho = aplatir rhs en tableau N_x * N_y (row-major)
    fft_.solve(rho -> phil)                         # FFT directe, /lambda(k), k=0 -> 0, FFT inverse
    phi = re-empaqueter phil dans le fab

function solve():                                  # DistributedFFTSolver, FFT par bandes
    rho = aplatir la bande locale [0..Nx-1] x [y0..y0+nyl-1]
    fft_.solve(rho -> phil)                         # transposee parallele (MPI_Alltoall) interne
    phi = re-empaqueter la bande locale
```

**Code.** [`numerics/elliptic/poisson_fft_solver.hpp`](../include/adc/numerics/elliptic/poisson_fft_solver.hpp):
`PoissonFFTSolver` (single-rank, single box) and `DistributedFFTSolver` (FFT distributed by bands /
slabs, 1 box per rank, `MPI_Alltoall` transpose internal to `PoissonFFT`). Both model the same
`EllipticSolver` concept (`static_assert`) as multigrid, so the coupler is generic over the
backend (`Coupler<Model, PoissonFFTSolver>` interchangeable with `GeometricMG`). The residual reuses
the canonical operator `poisson_residual` of
[`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp); the
distributed variant does a `fill_boundary` (inter-band halos) before the measurement and reduces by
`all_reduce_max`. The FFT core lives in `poisson_fft.hpp` (a fix handles $n$ not a power of 2).

**Constraints / remarks.** The FFT requires periodic BCs and a constant coefficient: neither
$\epsilon(x)$, nor an embedded mask, nor cross terms. The mode $k = 0$ must be fixed (right-hand side
of zero mean), otherwise $\phi$ drifts. `PoissonFFTSolver` raises a hard safeguard (active in Release,
not a plain `assert`) if `n_ranks() != 1` or `ba.size() != 1`: under a multi-rank `DistributionMapping`
some ranks would have no local box and `solve()` would dereference a nonexistent `fab(0)`
(SIGSEGV); the message points to `DistributedFFTSolver` or `geometric_mg`. The distributed
variant requires $N_y$ divisible by `n_ranks()` and $N_x, N_y$ powers of 2.
**Validation.** `test_poisson_fft` (non-regression, size $n$ not a power of 2); under MPI
`test_mpi_fft_distributed` (FFT by bands). `test_elliptic_operator` applies the same canonical
operator `poisson_residual` to the MG and FFT solutions: residuals at roundoff (`~1e-14`) and solutions
identical to `~1e-16`, so both provably invert the same discrete Laplacian.

## 11. Extended elliptic: eps(x), screened/Helmholtz, anisotropic

**Intuition.** The same multigrid operator covers three generalizations of the Laplacian, all
opt-in and bit-identical to the historical path when not activated (the corresponding coefficient
pointer stays `nullptr`):

- **variable permittivity** $\mathrm{div}(\epsilon(x)\,\mathrm{grad}\,\phi) = f$: each face carries
  the harmonic mean of the two adjacent centers of the $\epsilon$ field;
- **screened / Helmholtz operator** $\mathrm{div}(\epsilon\,\mathrm{grad}\,\phi) - \kappa\phi = f$:
  a reaction term $\kappa \ge 0$ (Debye screening $\kappa = 1/\lambda_D^2$), diagonal, which
  makes the operator more diagonally dominant (multigrid converges at least as well);
- **anisotropic permittivity** $\mathrm{div}(\mathrm{diag}(\epsilon_x, \epsilon_y)\,\mathrm{grad}\,\phi) = f$:
  the faces normal to $x$ read $\epsilon_x$, the faces normal to $y$ read $\epsilon_y$ (a diagonal
  tensor medium).

**Formula / discretization.** Face permittivity by harmonic mean (continuity of the normal flux
at an interface, resistances in series, correct even for a discontinuous $\epsilon$):

$$\epsilon_{i+1/2,j} = \frac{2\,\epsilon_{ij}\,\epsilon_{i+1,j}}{\epsilon_{ij} + \epsilon_{i+1,j}}$$

The discrete 5-point operator with variable face coefficient, with reaction, on cell $(i,j)$:

$$L\phi_{ij} = w^x_+\phi_{i+1,j} + w^x_-\phi_{i-1,j} + w^y_+\phi_{i,j+1} + w^y_-\phi_{i,j-1}
            - (w^x_+ + w^x_- + w^y_+ + w^y_-)\phi_{ij} - \kappa_{ij}\phi_{ij}$$

with $w^x_\pm = \epsilon^x_{i\pm1/2,j} / \Delta x^2$ ($\epsilon_x$ field) and
$w^y_\pm = \epsilon^y_{i,j\pm1/2} / \Delta y^2$ ($\epsilon_y$ field; in isotropic mode $\epsilon_y$ points
to the same field as $\epsilon_x$). The GS smoother gains $+\kappa_{ij}$ on its diagonal
($\kappa \ge 0$ => more dominant). Cut-cell + $\epsilon$ combination: each Shortley-Weller face weight
$w_{\bullet}$ is multiplied by its face permittivity, the diagonal stays the
sum of the face weights.

```
function ApplyLaplacianKernel(i, j):               # L = div(eps grad phi) - kappa phi, foncteur ADC_HD
    if he (eps actif):
        ec  = eps_x(i,j); ecy = eps_y(i,j)         # eps_y == eps_x en isotrope
        exm = harmonic(ec,  eps_x(i-1,j)); exp = harmonic(ec,  eps_x(i+1,j))
        eym = harmonic(ecy, eps_y(i,j-1)); eyp = harmonic(ecy, eps_y(i,j+1))
        if hc (cut-cell):  wxm,wxp,wym,wyp = coef[0..3] * (exm,exp,eym,eyp)   # poids SW * eps_face
        else:              wxm,wxp = exm,exp / dx^2 ;  wym,wyp = eym,eyp / dy^2
        L(i,j) = wxp*p(i+1,j)+wxm*p(i-1,j)+wyp*p(i,j+1)+wym*p(i,j-1) - (wxm+wxp+wym+wyp)*p(i,j)
    else if hc:  L(i,j) = coef[1]*p(i+1,j)+coef[0]*p(i-1,j)+coef[3]*p(i,j+1)+coef[2]*p(i,j-1)-coef[4]*p(i,j)
    else:        L(i,j) = (p(i+1,j)-2p(i,j)+p(i-1,j))/dx^2 + (p(i,j+1)-2p(i,j)+p(i,j-1))/dy^2
    if hxy or hyx:  L(i,j) += cross_div(...)        # tenseur plein (section 12) : flux croises additifs
    if hk (kappa actif):  L(i,j) -= kappa(i,j) * p(i,j)
```

**Code.** [`numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp):
`GeometricMG::set_epsilon(eps_fn | eps_fine)`, `set_reaction(kappa_fn | kappa_fine)`,
`set_epsilon_anisotropic(eps_x, eps_y)`. Each field exists in two overloads: analytic
(`std::function`, evaluated per level over the whole hierarchy -> exact permittivity at the coarse,
order 2 preserved) and already-discretized (`MultiFab` of the fine level, component-0 copy by
`detail::CopyComp0Kernel` then restricted by `average_down`, entry point for the wiring from
`System`). The $\kappa$ term (0 ghost, diagonal), the $\epsilon$ / $\epsilon_y$ fields (1 ghost,
ghosts filled by `eps_bc`: periodic preserved, physical boundary by zero-gradient extrapolation)
live in the ADC_HD `for_each_cell` of the smoother, the residual and the apply
([`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp):
`ApplyLaplacianKernel`, `PoissonResidualKernel`, `GsColorKernel`, `eps_harmonic`) -> device. The
fine-level coefficient pointers are also exposed (`op_eps`, `op_kappa`, `op_eps_y`, ...)
so the Krylov solver reuses an operator consistent with the MG residual.

**Constraints / remarks.** The three extensions are composable: $\epsilon(x)$ and $\kappa(x)$
together, or $\mathrm{diag}(\epsilon_x, \epsilon_y)$ with $\kappa$. Giving $\epsilon_x \equiv \epsilon_y$ gives back the isotropic; not calling `set_reaction` gives back pure Poisson; no call =>
the historical path strictly bit-identical. The harmonic choice (and not arithmetic) for the face
preserves the continuity of the normal flux at a medium jump and stays order 2 for a smooth
$\epsilon$. The per-level sampling (instead of restricting from the fine) gives the
exact coefficient at each coarse resolution, which preserves the order 2 of the V-cycle.
**Validation.** `test_variable_epsilon` ($\epsilon(x)$, MMS order 2), `test_screened_poisson`
(Helmholtz / screened, MMS order 2), `test_anisotropic_epsilon` (anisotropic $\epsilon_x \neq \epsilon_y$, MMS order 2). The three paths are also exercised on the Python side (`test_poisson_eps`,
`test_poisson_screened`, `test_poisson_eps_aniso`) and validated bit-identical on GH200
(cf. GPU_RUNTIME_PORT.md, round 2).


---

## 12. Full-tensor elliptic: matrix-free Krylov (BiCGStab)

**Intuition.** When the elliptic operator carries cross terms $A_{xy} \neq A_{yx}$ (a non-self-adjoint operator, for example the rotation $B^{-1}$ coming from Schur condensation), geometric multigrid alone, whose Gauss-Seidel smoother assumes a self-adjoint operator, stagnates or diverges. A non-symmetric Krylov solver is needed, preconditioned by the MG V-cycle applied to the symmetric part of the operator.

**Formula / discretization.** We solve $A\,\phi = f$ with, in the convention of [`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp) and of `GeometricMG`,

$$L_{\mathrm{int}}(\phi) = \mathrm{div}(A\,\nabla\phi) - \kappa\,\phi, \qquad A = \begin{pmatrix} A_{xx} & A_{xy} \\ A_{yx} & A_{yy}\end{pmatrix},$$

the matvec being `apply_laplacian` (exact computation of $L_{\mathrm{int}}$) and the residual $r = f - L_{\mathrm{int}}(\phi)$, bit-consistent with `poisson_residual`. BiCGStab is matrix-free: no matrix is assembled, only the product $A\,d$ is required, applied by `for_each_cell`. The preconditioner is $M^{-1} =$ ($N$ V-cycles of `GeometricMG`) on the symmetric diagonal block (cross terms $A_{xy}/A_{yx}$ dropped). The antisymmetric part being in $O(\theta^2 dt^2 \alpha)$, small at reasonable source CFL, the symmetric preconditioner captures the essential of the spectrum.

Choice of BiCGStab and not gmres: it handles the non-symmetric without a restart parameter nor a growing Krylov basis to store. The memory footprint is fixed (the MultiFab $r, \hat r, p, v, s, t$ at zero ghost, plus the preconditioned $\hat p, \hat s$ at one ghost).

A delicate point is the treatment of inhomogeneous Dirichlet conditions. The boundary ghost equals $2v - \phi_{\mathrm{int}}$, so the stencil of the boundary cells receives a constant term $c_{bc} = \mathrm{apply\_operator}(0)$. The raw operator is therefore affine: $L_{\mathrm{aff}}(\phi) = L_{\mathrm{lin}}(\phi) + c_{bc}$. For the true residual $r_0$ we keep the affine operator (the Dirichlet data folds into it exactly, which is what we want). But the in-loop matvecs act on correction directions $\hat p = M^{-1}p$, $\hat s = M^{-1}s$: they must be linear, otherwise the constant term injected at each product makes the residual oscillate or diverge. We therefore subtract $c_{bc}$ (matvec) and $d_{bc} = \mathrm{precond\_raw}(0)$ (preconditioner), computed once per solve in `prepare_solve`. When the Dirichlet BC is null, $c_{bc} = d_{bc} = 0$ and the path becomes bit-identical to the history again.

```
function TensorKrylovSolver.solve(rel_tol, max_iters):
    prepare_solve()                          # c_bc = apply_op(0), d_bc = precond_raw(0) si CL inhomogene
    v   <- apply_operator(phi)               # operateur affine pour le residu vrai
    r   <- rhs - v                           # r0, warm start respecte (phi entrant = depart)
    norm0 <- ||rhs||_2  (sinon 1 si rhs nul) # base relative, reduction MPI collective
    if ||r|| <= rel_tol * norm0: return converged
    rhat <- r                                # vecteur fantome fige de BiCGStab
    p, v <- 0 ;  rho_prev, alpha, omega <- 1
    for k = 1 .. max_iters:
        rho <- dot(rhat, r)                  # collectif (all_reduce, tous rangs)
        if |rho| ~ 0 or |omega| ~ 0: return best_effort     # garde-fou rupture
        beta <- (rho / rho_prev) * (alpha / omega)
        p   <- r + beta * (p - omega * v)
        phat <- M^{-1} p                     # N V-cycles MG sur la partie symetrique, CL homogenes
        v   <- apply_operator_lin(phat)      # matvec lineaire (phat = direction)
        alpha <- rho / dot(rhat, v)          # dot collectif ; garde-fou si ~ 0
        s   <- r - alpha * v
        phi <- phi + alpha * phat            # correction partielle (tampon avant test sur ||s||)
        if ||s|| <= rel_tol * norm0: return converged        # convergence a mi-iteration
        shat <- M^{-1} s
        t   <- apply_operator_lin(shat)
        omega <- dot(t, s) / dot(t, t)       # 0 si dot(t,t) ~ 0
        phi <- phi + omega * shat
        r   <- s - omega * t
        if ||r|| <= rel_tol * norm0: return converged
        rho_prev <- rho
    return best_effort                       # max_iters atteint, converged = false

function apply_operator(in):                 # matvec matrice-libre affine
    device_fence() ; fill_ghosts(in, bc_entiere)
    out <- apply_laplacian(in, eps_x, eps_y, a_xy, a_yx, kappa)   # = L_int(in)
    if mask present: mask_zero(out)          # L_int = 0 sur cellules conductrices (Dirichlet phi=0)
    return out

function apply_operator_lin(in):             # matvec lineaire (directions de correction)
    out <- apply_operator(in)
    if has_op_offset: out <- out - c_bc      # retranche la part inhomogene de bord
    return out

function precond_raw(in):                     # V-cycle brut (affine si CL Dirichlet != 0)
    precond.rhs <- in ; precond.phi <- 0
    repeat N: precond.vcycle()
    return precond.phi

function apply_precond(in):                   # M^{-1} a CL homogenes
    out <- precond_raw(in)
    if has_bc_offset: out <- out - d_bc      # M^{-1} in = precond_raw(in) - precond_raw(0)
    return out
```

**Code.** [`numerics/elliptic/krylov_solver.hpp`](../include/adc/numerics/elliptic/krylov_solver.hpp): class `TensorKrylovSolver`, methods `solve(rel_tol, max_iters)` (returns a `KrylovResult`: `iters`, `rel_residual`, `converged`), `apply_operator` / `apply_operator_lin` (affine and linear matvec), `precond_raw` / `apply_precond` (raw V-cycle and homogeneous-BC V-cycle), `prepare_solve` (one-time computation of the offsets $c_{bc}$, $d_{bc}$), `residual` (current global L2 residual). The constructor takes two distinct `GeometricMG`: `op` carries the full operator (matvec + storage of $\phi$/$rhs$), `precond` carries the symmetric part (same eps but `set_cross_terms` not called). They must be separate objects, enforced by `assert(&op != &precond)`: `apply_precond` overwrites `precond.rhs()`/`precond.phi()` at each iteration, and conflating them would overwrite the iterate and the right-hand side of the solve.

**Constraints / remarks.** Iterative method, no CFL of its own; the cost depends on the conditioning of the Schur complement, hence the preconditioning by symmetric MG (1 to 2 V-cycles, parameter `n_precond_vcycles`). BiCGStab breakdown safeguards: if $|\rho|$, $|\omega|$ or $\mathrm{dot}(\hat r, v)$ fall below `kTiny` $= 10^{-300}$, the solve returns the current best effort without dividing by zero. Device/MPI: named functors only (`mf_arith`: `saxpy`/`lincomb`/`dot`, `apply_laplacian`, MG V-cycle), all device-clean. The scalar products `dot` are collective (`all_reduce_sum`) and called on all ranks, including a rank without a box (`local_size() == 0`): no short-circuit, hence no MPI deadlock nor desynchronization of the stopping criterion. Known limitation: the symmetric preconditioner loses efficiency when the antisymmetric part grows (high source CFL, large $\omega_c$); the iteration count then increases.

Validation: `test_krylov_solver`. Case (A) on the canonical Laplacian ($A_{xy} = A_{yx} = 0$, $A = I$, $\kappa = 0$), BiCGStab converges to the same solution as `GeometricMG` at the tolerance. Case (B) on an operator with non-trivial cross terms, BiCGStab converges where the MG V-cycle alone stagnates ($c = 0.1$ to $0.4$) or diverges ($c = 0.7$). Under MPI, the iteration count and the convergence are invariant to the number of ranks (stopping criterion reduced by `all_reduce`).

## 13. Condensed implicit source: Schur condensation

**Intuition.** A stiff source that couples potential, velocity and Lorentz force (diocotron at high $\omega_c$) cannot be treated component by component: the cyclotron rotation couples the two velocity components, and the potential reacts to the charge displacement. We theta-discretize the implicit source, eliminate the velocity algebraically via the closed inverse $B^{-1}$ of the 2x2 rotation, which leaves only an elliptic on the potential $\phi^{n+\theta}$ alone (Schur complement), then reconstruct the velocity.

**Formula / discretization.** The Lorentz eliminator encodes the rotation-dilation

$$B = \begin{pmatrix} 1 & -w \\ w & 1 \end{pmatrix}, \qquad B^{-1} = \frac{1}{\det B}\begin{pmatrix} 1 & w \\ -w & 1\end{pmatrix}, \qquad w = \theta\, dt\, B_z, \quad \det B = 1 + w^2 > 0,$$

closed and always invertible (no call to `std::`, four additions/multiplications, device-safe). With $c = \theta^2 dt^2 \alpha$ (Hoffart et al., arXiv:2510.11808), the condensed operator writes

$$L_{\mathrm{schur}}(\phi) = -\Delta\phi - c\mathrm{div}(\rho\, B^{-1}\nabla\phi) = -\mathrm{div}\!\big((I + c\,\rho\, B^{-1})\,\nabla\phi\big),$$

which identifies the full tensor $A = I + c\,\rho\, B^{-1}$, that is, per cell,

$$\varepsilon_x = 1 + c\rho\,B^{-1}_{11},\quad \varepsilon_y = 1 + c\rho\,B^{-1}_{22},\quad a_{xy} = c\rho\,B^{-1}_{12},\quad a_{yx} = c\rho\,B^{-1}_{21}.$$

The mass term $\kappa$ stays null (the condensation does not produce a Helmholtz). At $B_z = 0$: $w = 0$, $B^{-1} = I$, so $a_{xy} = a_{yx} = 0$ and $\varepsilon_x = \varepsilon_y = 1 + c\rho$; if in addition $c = 0$, $A = I$ and $L_{\mathrm{schur}}$ degenerates exactly into the canonical Laplacian. The condensed right-hand side is

$$\mathrm{rhs} = -\Delta\phi^n - \theta\, dt\, \alpha \mathrm{div}(\rho\, B^{-1} v^n), \qquad v^n = (m_x, m_y)/\rho,$$

where $-\Delta\phi^n$ is the canonical 5-point Laplacian negated and the divergence of the explicit flux $F = \rho B^{-1} v^n = B^{-1}(m_x, m_y)$ (applied to the momentum, which avoids the division by $\rho$) is centered order 2:

$$\mathrm{div} F(i,j) = \frac{F_x(i{+}1,j) - F_x(i{-}1,j)}{2\,dx} + \frac{F_y(i,j{+}1) - F_y(i,j{-}1)}{2\,dy}.$$

The condensed operator is in general full-tensor (hence the Krylov solver, section 12). Sign convention of the solve: `TensorKrylovSolver` solves $L_{\mathrm{int}} = +\mathrm{div}(A\nabla\phi)$, so $L_{\mathrm{schur}} = -L_{\mathrm{int}}$ and the stage passes $\mathrm{rhs}_{\mathrm{kry}} = -\mathrm{rhs}_{\mathrm{schur}}$ to the solver. After resolution, the velocity is reconstructed by $v^{n+\theta} = B^{-1}(v^n - \theta\, dt\,\nabla\phi^{n+\theta})$ (centered gradient, consistent with the RHS divergence), then extrapolated from the theta-stage to the full step by $U^{n+1} = U^n + \tfrac{1}{\theta}(U^{n+\theta} - U^n)$. The energy, if the Energy role is present, is updated only by the kinetic energy increment $E^{n+1} = E^n + \tfrac{1}{2}\rho^n(|v^{n+1}|^2 - |v^n|^2)$, the Lorentz rotation doing no work and $\rho$ being frozen.

```
function CondensedSchurSourceStepper.step(state, phi, bz_field, c_bz, theta, dt):
    # -1) figer phi^n pour l'extrapolation finale
    phi_n <- copy(phi)
    # 0) extraire v^n = (mx, my)/rho ; copier B_z dans le tampon interne (1 ghost)
    for each cell: vx_n, vy_n <- ExtractVelocity(state) ; bz <- CopyBz(bz_field, c_bz)
    fill_ghosts(bz, foextrap)
    # 1) assembler (builder #124) :  A_op = I + c rho B^{-1},  rhs_schur = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n)
    builder <- ElectrostaticLorentzCondensation(vars, alpha, theta, dt)   # c = theta^2 dt^2 alpha
    builder.assemble_operator(state, bz -> eps_x, eps_y, a_xy, a_yx)
    builder.assemble_rhs(phi, state, bz -> rhs_schur)
    # 2) resoudre par BiCGStab :  L_int(phi) = -rhs_schur  (convention de signe)
    op.set_epsilon_anisotropic(eps_x, eps_y) ; op.set_cross_terms(a_xy, a_yx)     # operateur plein
    precond.set_epsilon_anisotropic(eps_x, eps_y)                                  # partie symetrique
    op.phi <- phi (warm start)  ;  op.rhs <- -rhs_schur
    last_result <- TensorKrylovSolver(op, precond, N).solve(1e-10, 400)
    phi <- op.phi                                                                  # phi^{n+theta}
    # 3) reconstruire la vitesse
    fill_ghosts(phi, bcPhi)
    for each cell:
        g <- grad_centre(phi)                                  # (d_x phi, d_y phi)
        rhs_v <- v_n - theta*dt * g
        v_theta <- B^{-1}(rhs_v)  with  w = theta*dt*bz        # LorentzEliminator.apply_Binv
        state.mom <- rho^n * v_theta                           # rho gelee
    # 5) extrapoler theta-stage -> pas plein :  f^{n+1} = f^n + (1/theta)(f^{n+theta} - f^n)
    phi <- phi_n + (1/theta)(phi - phi_n)
    v   <- v_n   + (1/theta)(v_theta - v_n) ;  state.mom <- rho^n * v
    # 4) energie (si role Energy present)
    if has_energy: state.E <- state.E + 0.5 * rho^n * (|v^{n+1}|^2 - |v^n|^2)
    # 6) publier : ghosts de l'etat et du potentiel (halos MPI / CL physiques)
    device_fence() ; fill_ghosts(state, foextrap) ; fill_ghosts(phi, bcPhi)
```

**Code.** [`numerics/lorentz_eliminator.hpp`](../include/adc/numerics/lorentz_eliminator.hpp): POD struct `LorentzEliminator(theta, dt, B_z)`, methods `apply_B`/`apply_Binv`, accessors `binv_11..binv_22`; trivially copyable (static_assert), capturable by value in a kernel. [`coupling/schur_condensation.hpp`](../include/adc/coupling/schur_condensation.hpp) builds the operator and the RHS without solving nor reconstructing: class `ElectrostaticLorentzCondensation`, methods `assemble_operator` (functor `SchurOperatorCoeffKernel`), `assemble_rhs` (functors `SchurExplicitFluxKernel`, `SchurRhsAssembleKernel`, `NegateKernel`), `assemble` (into a `SchurCondensationOperator`), accessor `c_coeff()`; the Density/MomentumX/MomentumY roles contract is validated on the host (exception otherwise). [`coupling/condensed_schur_source_stepper.hpp`](../include/adc/coupling/condensed_schur_source_stepper.hpp): class `CondensedSchurSourceStepper`, method `step` which composes the three bricks (assembler #124, `TensorKrylovSolver` #122, `LorentzEliminator` #118), functors `SchurReconstructKernel`, `SchurExtrapolateScalarKernel`, `SchurExtrapolateVelocityKernel`, `SchurEnergyKernel`, `ExtractVelocityKernel`, `CopyBzKernel`, diagnostic `last_solve()`. This is the production source stage (#126), opt-in via `adc.Split(source=CondensedSchur)`.

**Constraints / remarks.** Stability: the theta-scheme is unconditionally stable for $\theta \geq 1/2$ ($\theta = 1$ pure implicit, the extrapolation is the identity; $\theta = 1/2$ Crank-Nicolson, extrapolation factor 2). The centered order-2 discretization (5-point Laplacian, centered divergence and gradient) fixes the spatial order. This is the source stage alone (transport frozen): $\rho$ is constant in the stage, $\rho^{n+1} = \rho^n$, and all the transport dynamics stays in the hyperbolic stage of the splitting. Safeguard: $c = 0$ and $B_z = 0$ give $A = I$, the solve becomes $\Delta\phi^{n+\theta} = \Delta\phi^n$ so $\phi^{n+\theta} = \phi^n$ (up to a constant), and the reconstruction degenerates into the explicit electrostatic push $v^{n+\theta} = v^n - \theta\, dt\,\nabla\phi^n$. The tolerance of the internal solve ($10^{-10}$, 400 iterations max) bounds the precision of the implicit relation $B v = v^n - \theta\, dt\,\nabla\phi$ (verified term by term). Device/MPI: all kernels are device-clean named functors (no extended cross-TU lambda, nvcc limit #64/#97); the MultiFab buffers are allocated once at construction and reused at each `step`; the loops iterate over `local_size()` (a rank without a box -> no kernel) and the Krylov solve is collective, so MPI-clean.

Validation: `test_schur_condensation` (condensed operator and RHS correct, including the C2 safeguard of the degenerate case) and `test_condensed_schur_source_stepper` (the source stage advances correctly). On the Python side: `test_schur_split`, `test_schur_via_system`, `test_schur_conservation`.


---

## 14. Embedded boundary: Shortley-Weller cut-cell

**Intuition.** A wall not aligned on the grid (circular conductor, diocotron ring edge)
is not a staircase: the Shortley-Weller cut-cell corrects the 5-point stencil where the
disc level set cuts a face, so that the Dirichlet condition is imposed at the real position
of the interface and not at the nearest cell face. We recover order 2 where the
0/1 staircase falls to order 1.

**Formula / discretization.** The boundary is carried by the canonical disc level set
$ls(x,y) = \mathrm{hypot}(x - c_x, y - c_y) - R$, negative inside. For an active cell
$ls(x_c, y_c) < 0$, each cardinal face is cut at a linear fraction: if the neighbor is
interior ($l_n < 0$) the face is full and the half-distance equals $h$; if the level set changes
sign ($l_n \ge 0$) the linear crossing between $l_c < 0$ and $l_n \ge 0$ gives

$$\theta = \frac{l_c}{l_c - l_n}, \qquad a = \theta\, h, \qquad \theta \in [10^{-3}, 1]$$

(the lower clamp $10^{-3}$ is the anti-division guard that prevents $w \to \infty$ when the face
grazes the boundary). With the four half-distances $a_{xm}, a_{xp}, a_{ym}, a_{yp}$, and setting
$s_x = a_{xm}+a_{xp}$, $s_y = a_{ym}+a_{yp}$, the Laplacian $-\Delta\phi$ becomes a 5-point stencil with
unequal steps (Shortley-Weller) of weights

$$w_{xm} = \frac{2}{a_{xm}\, s_x},\quad w_{xp} = \frac{2}{a_{xp}\, s_x},\quad
w_{ym} = \frac{2}{a_{ym}\, s_y},\quad w_{yp} = \frac{2}{a_{yp}\, s_y},$$

$$w_{\mathrm{diag}} = \frac{2}{a_{xm}\, a_{xp}} + \frac{2}{a_{ym}\, a_{yp}},$$

and the residual on an active cell is
$L\phi = w_{xm}\phi_{i-1} + w_{xp}\phi_{i+1} + w_{ym}\phi_{i,j-1} + w_{yp}\phi_{i,j+1} - w_{\mathrm{diag}}\phi_{i,j}$
(the boundary Dirichlet value is injected via the ghost placed at $a$, not $h$). Far from the boundary all
the half-distances equal $h$: the weights give back exactly the uniform 5-point stencil
$1/h^2,\dots,-4/h^2$. For a conductor cell ($ls \ge 0$, mask at 0) the cell is skipped
(unused coefficient).

```
function shortley_weller_coefs(level L, geometry g, level_set ls):
    # one-shot au setup, par niveau MG, sur l'hote (puis lu par le V-cycle on-device)
    for each active cell (i, j) of L:            # m(i,j) != 0 ; conducteur saute
        lc  = ls(g.x_cell(i), g.y_cell(j))       # < 0 par construction (cellule active)
        axm = cut_distance(lc, ls(x - dx, y), dx)  # voisin interieur -> dx ; sinon theta*dx
        axp = cut_distance(lc, ls(x + dx, y), dx)
        aym = cut_distance(lc, ls(x, y - dy), dy)
        ayp = cut_distance(lc, ls(x, y + dy), dy)
        sx = axm + axp ; sy = aym + ayp
        c(i,j,0) = 2 / (axm * sx)                # w_xm  sur p(i-1, j)
        c(i,j,1) = 2 / (axp * sx)                # w_xp  sur p(i+1, j)
        c(i,j,2) = 2 / (aym * sy)                # w_ym  sur p(i, j-1)
        c(i,j,3) = 2 / (ayp * sy)                # w_yp  sur p(i, j+1)
        c(i,j,4) = 2/(axm*axp) + 2/(aym*ayp)     # w_diag (coefficient central)

function cut_distance(lc, ln, h):
    if ln < 0:        return h                   # face pleine (voisin interieur)
    th = lc / (lc - ln)                          # crossing lineaire
    return clamp(th, 1e-3, 1) * h                # garde anti-division par 0
```

**Code.** The cut geometry is centralized in
[`include/adc/numerics/elliptic/cut_fraction.hpp`](../include/adc/numerics/elliptic/cut_fraction.hpp):
`detail::cut_distance` (linear crossing of a face), `detail::cut_fraction` (the 4 half-distances
+ apertures + volume fraction `kappa`), and `detail::shortley_weller` which returns the 5 weights
`ShortleyWellerWeights{w_xm, w_xp, w_ym, w_yp, w_diag}`. The V-cycle
[`include/adc/numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp)
writes them once per level into its `coef` field (5 components) at setup (host) then reads them
on-device; it skips the conductor cells (`m(i,j) == 0`). It is the same `cut_fraction` that the
EB transport consumes (section 15): aperture geometry bit-consistent between Poisson and transport.

**Constraints / remarks.** The clamp $\theta \ge 10^{-3}$ bounds $w_{\mathrm{diag}}$ (without it a
grazing face would make the weight diverge and would break the diagonal dominance of the smoother). Compatible with
the anisotropic operator (the cut-cell weights compose with the $\varepsilon_x, \varepsilon_y$ coefficients). Validation: `test_cut_cell` (cut-cell vs staircase on a manufactured solution, order gain
), `test_cut_cell_anisotropic` (cut-cell + anisotropic operator), `test_cut_cell_anisotropic_multibox`
(multi-box single-rank), `test_mpi_cutcell_multibox` (multi-box distributed np=1/2/4; non-regression lock
of the `average_down` out-of-bounds bug on a degenerate MG hierarchy). For the elliptic on an
immersed disc, `test_poisson_disc` exercises the solver (convergence + improvement at resolution).

## 15. Disc domain: mask, masked transport, cut-cell transport

**Intuition.** A disc transport subdomain (diocotron ring) imposes a circular boundary
not aligned on the Cartesian grid. Three modes, from the simplest to the most precise:
`none` (mask materialized but ignored, bit-identical to the full Cartesian), `staircase` (jagged
boundary, 0/1 face gate, order 1 at the boundary), `cutcell` / embedded-boundary (continuous apertures
`alpha_f` + volume fraction `kappa`, smooth boundary, order 2 inside the disc). The cut-cell
generalizes the 0/1 gate to an aperture $\alpha_f \in [0,1]$ and divides the residual by the true immersed
volume $\kappa$, which de-jaggs the boundary and restores the order 2 that the staircase mask does not give.

**Formula / discretization.** Conservative embedded-boundary form for cell $(i,j)$ of volume
$\kappa\, dx\, dy$ (advection $\partial_t U = -\mathrm{div}\,F + S$):

$$\kappa\, dx\, dy\; \partial_t U = -\big[\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i\big] dy
 - \big[\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j\big] dx - \alpha_w |w| F_w + \kappa\, dx\, dy\, S,$$

that is, after division by $\kappa\, dx\, dy$ (with $\kappa$ clamped, cf. below):

$$R = S - \frac{1}{\kappa}\left[\frac{\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i}{dx}
 + \frac{\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j}{dy}\right]
 - \frac{1}{\kappa}\,\frac{\alpha_w |w|}{dx\, dy}\, F_w.$$

The immersed wall flux is a no-penetration $F_w = 0$ (FV counterpart of the conductor wall: the term is
identically null, written explicitly as a hook for a future non-zero flux). The apertures and $\kappa$
come from the same `cut_fraction` as the elliptic cut-cell:
$\alpha_f = a_f / h$, and $\kappa = \tfrac{1}{2}(\alpha_{xm}+\alpha_{xp})\cdot\tfrac{1}{2}(\alpha_{ym}+\alpha_{yp})$.
A face between two active cells has the same aperture on both sides
($\alpha_{xp}(i) = \alpha_{xm}(i{+}1)$, a function of the level set alone), so the fluxes telescope and the
mass $\sum_{ij} n_{ij}\, \kappa_{ij}\, dx\, dy$ is conserved to the machine; a face touching an
inactive cell is closed ($\alpha_f = 0$) and $F_w = 0$, so no mass crosses the boundary.

```
function assemble_rhs_eb(model, U, aux, ls, geom, R, kappa_min):
    # passe 1 : flux de face ponderes par l'ouverture alpha_f (MultiFab Fx, Fy temporaires)
    for each x-face i between cells (i-1,j) and (i,j):
        lL = ls(x_cell(i-1), y_cell(j)) ; lR = ls(x_cell(i), y_cell(j))
        alpha = face_aperture(lL, lR)            # voisin inactif -> 0 ; sinon cut_distance/dx
        if alpha < 1e-6:  Fx(i,j,:) = 0          # face fermee = paroi immergee, flux normal nul
        else:
            L  = reconstruct(U, i-1, j, dir=0, +1)   # reconstruction reutilisee du cartesien
            Rr = reconstruct(U, i,   j, dir=0, -1)
            Fx(i,j,:) = alpha * numerical_flux(L, Rr, dir=0)   # on stocke alpha * F
    ... idem pour les y-faces -> Fy ...

    # passe 2 : divergence EB / kappa_eff + source
    for each cell (i, j):
        if ls(x_cell(i), y_cell(j)) >= 0:        # hors disque : residu nul, cellule non avancee
            R(i,j,:) = 0 ; continue
        kappa     = cut_fraction(ls, x_cell(i), y_cell(j), dx, dy).kappa
        kappa_eff = max(kappa, kappa_min)        # clamp petite cellule (defaut 1e-2)
        S = model.source(U(i,j), aux(i,j))
        for each component c:
            div_x = (Fx(i+1,j,c) - Fx(i,j,c)) / dx     # Fx contient deja alpha*F
            div_y = (Fy(i,j+1,c) - Fy(i,j,c)) / dy
            # accumulation terme A terme : avec kappa_eff=1 et alpha=1, bit-identique au cartesien
            R(i,j,c) = S[c] - div_x/kappa_eff - div_y/kappa_eff - 0   # F_wall = 0

function face_aperture(lc, ln):
    if ln >= 0:  return 0                         # voisin inactif : face fermee (no-penetration)
    return cut_distance(lc, ln, h) / h            # voisin actif : ouverture lineaire (== mur elliptique)
```

**Code.** `System::set_disc_domain(cx, cy, R, mode)` (#216,
[`include/adc/runtime/system.hpp`](../include/adc/runtime/system.hpp), defined in
[`python/system.cpp`](../python/system.cpp)) sets a `DiscDomain`
([`include/adc/runtime/wall_predicate.hpp`](../include/adc/runtime/wall_predicate.hpp), `level_set`) and
the transport mode; `set_geometry_mode(mode)` switches the mode alone; `disc_mask()` materializes the
mask (all-active if no disc). The stepper routes each block: `assemble_rhs` (full),
`assemble_rhs_masked`
([`include/adc/numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp), 0/1
gate) or `assemble_rhs_eb`
([`include/adc/numerics/spatial_operator_eb.hpp`](../include/adc/numerics/spatial_operator_eb.hpp),
EB). The device kernels are named functors (`detail::EbFaceFluxXKernel`,
`EbFaceFluxYKernel`, `EbAssembleRhsKernel`, and the adapter `detail::DiscLevelSet` which forwards
`DiscDomain::level_set`) for cross-TU emission under nvcc; `eb_face_aperture` closes the face toward an
inactive neighbor. The apertures and `kappa` come from
[`include/adc/numerics/elliptic/cut_fraction.hpp`](../include/adc/numerics/elliptic/cut_fraction.hpp)
(same geometry as the elliptic cut-cell of section 14); the reconstruction (`reconstruct<>`) and
the numerical flux (`RusanovFlux`) are reused verbatim from the Cartesian operator.

**Constraints / remarks.** Small-cell problem: the factor $1/\kappa$ amplifies the residual
when $\kappa \to 0$ on the cut layer, which would make a fixed explicit step explode. Two stacked
guards: (i) the floor $\theta \ge 10^{-3}$ of `cut_distance` inherited from the elliptic wall (bound
$\kappa \gtrsim 2.5\times 10^{-7}$, insufficient alone); (ii) the clamp on the retained volume
$\kappa_{\mathrm{eff}} = \max(\kappa, \kappa_{\min})$, $\kappa_{\min} = 10^{-2}$ by default, which bounds
the amplification to $1/\kappa_{\min} = 100$ (implicit volume merging, calibrated for a stable fixed step
whatever the degree of cut). The clamp acts only on the denominator (volume), not on the fluxes:
the global mass stays exact, at the cost of a slight local non-conservation on the most
cut cells. Documented alternative (out of scope): the flux redistribution of AMReX-EB (exact local
conservation, non-local stencil). The path is purely additive and opt-in: a run without an EB disc is
bit-identical to the Cartesian. Validation: `test_disc_domain_mask` (mask bounds, all-active without a
disc), `test_eb_transport` (EB cut-cell transport: conservation and smooth boundary). On the Python side:
`test_disc_domain_mask`.

## 16. Polar geometry: transport and Poisson on a ring (r, theta)

**Intuition.** For a diocotron ring, the Cartesian grid pays a structural over-rate at the edges
of the ring (the "Cartesian ring edges" lock). An annular polar grid
$r \in [r_{\min}, r_{\max}] \times \theta \in [0, 2\pi)$ aligns the geometry on the problem: theta is
periodic, r is physical (walls), and the ring excludes $r = 0$ ($r_{\min} > 0$) so no
coordinate singularity. Polar transport and Poisson reuse the same reconstruction, flux and source
bricks as the Cartesian; only the metrics change.

**Formula / discretization (transport, conservative FV).** The polar divergence
$\mathrm{div}\,F = \tfrac{1}{r}\partial_r(r F_r) + \tfrac{1}{r}\partial_\theta(F_\theta)$ is discretized
by storing the radial flux weighted by the face radius, $r_{i\pm1/2} F_r$, and the direct azimuthal flux,
then differencing:

$$R = S + S_g - \frac{1}{r_i}\frac{r_{i+1/2} F_r^{i+1} - r_{i-1/2} F_r^{i}}{dr}
 - \frac{1}{r_i}\frac{F_\theta^{j+1} - F_\theta^{j}}{d\theta}.$$

$S_g$ is the geometric curvature source ($-\rho v_\theta^2/r$ etc.), not captured by the conservative
divergence in a rotating local basis; it is carried per cell (null for a scalar ExB brick
-> bit-identical to the historical polar ExB transport). The weight $r_{i+1/2}$ of an interior face is
shared by the two neighboring cells, so the radial term telescopes; the azimuthal term telescopes
exactly (periodic). With `wall_radial`, the radial flux is forced to zero at the two physical boundary
faces -> mass $\sum n_{ij}\, r_i\, dr\, d\theta$ conserved to the machine whatever $v_r$.

**Formula / discretization (Poisson, FFT-in-theta + tridiag-in-r).** We solve
$\tfrac{1}{r}\partial_r(r\,\partial_r\phi) + \tfrac{1}{r^2}\partial_\theta^2\phi = f$ directly
(no multigrid, which stagnates on the $1/r^2$ operator). Theta being periodic with constant
coefficient, an FFT in theta diagonalizes $\partial_\theta^2$ exactly: the DFT mode $m$ has the signed
wavenumber $k(m) = m$ if $m \le n_\theta/2$, otherwise $m - n_\theta$, and the spectral eigenvalue
$-k(m)^2$ (and not the 2-point stencil $(2\cos - 2)/d\theta^2$, which is only an
$O(d\theta^2)$ approximation). The azimuthal term per cell becomes $(-k(m)^2/r_i^2)\,\hat\phi(i,m)$, diagonal in
$m$. The radial term is FV order 2,

$$\frac{1}{r_i}\left[\frac{r_{i+1/2}(\phi_{i+1}-\phi_i)}{dr} - \frac{r_{i-1/2}(\phi_i-\phi_{i-1})}{dr}\right]\frac{1}{dr},$$

so, per mode $m$, a tridiagonal system in $r$ solved by Thomas with

$$a_i = \frac{r_{i-1/2}}{r_i\, dr^2},\qquad c_i = \frac{r_{i+1/2}}{r_i\, dr^2},\qquad
b_i = -(a_i + c_i) - \frac{k(m)^2}{r_i^2}.$$

```
function polar_poisson_solve(geom, bc, rhs f, out phi):
    nr, nth = geom.nr, geom.ntheta
    # 1) FFT en theta, ligne radiale par ligne radiale : f(i, .) -> fhat(i, m)
    for i in 0..nr-1:  fhat[i] = fft( f(i, .) )
    # 2) coefficients radiaux independants du mode (geometrie pure)
    for i in 0..nr-1:
        a[i] = r_face(i)   / (r_cell(i) * dr^2)        # sous-diag
        c[i] = r_face(i+1) / (r_cell(i) * dr^2)        # sur-diag
        d_rad[i] = -(a[i] + c[i]) ;  inv_r2[i] = 1 / r_cell(i)^2
    # 3) une tridiagonale (Thomas) par mode azimutal m
    for m in 0..nth-1:
        k = (m <= nth/2) ? m : m - nth                 # nombre d'onde signe (repliement DFT)
        for i: b[i] = d_rad[i] - k*k * inv_r2[i]        # diag = radiale + azimutale spectrale
        rhs_m = fhat[., m]
        apply_radial_bc(b, rhs_m, m)                   # Dirichlet: b-=a/c, rhs-=2 a v (m=0) ; Neumann: b+=a/c
        pin0 = (deux bords Neumann) and (m == 0)        # operateur radial singulier -> jauge phi_hat(0,0)=0
        phat[., m] = thomas(a, b, c, rhs_m, pin0)
    # 4) FFT inverse en theta : phat(i, m) -> phi(i, theta) (partie reelle)
    for i in 0..nr-1:  phi(i, .) = real( ifft( phat[i] ) )
```

Boundary conditions in $r$ (via `BCRec.xlo/.xhi`): Dirichlet (value $v$ at the face, reflection ghost
$\phi_{-1} = 2v - \phi_0$ -> $b_0 \mathrel{-}= a_0$, and $2 a_0 v$ to the right-hand side of the mode
$m=0$ alone) or homogeneous Neumann (Foextrap, $\phi_{-1} = \phi_0$ -> $b_0 \mathrel{+}= a_0$). Mode $m=0$
+ two Neumann boundaries: the radial operator has the constant in its kernel (singular tridiagonal); we fix
the gauge by pinning $\hat\phi(0,0) = 0$ (row 0 replaced by the identity in Thomas).

**Code.** [`include/adc/mesh/geometry.hpp`](../include/adc/mesh/geometry.hpp)`::PolarGeometry` (ring,
opt-in via `adc.PolarMesh`; `cfg.geometry == "polar"` on the
[`python/system.cpp`](../python/system.cpp) side). Transport:
[`include/adc/numerics/spatial_operator_polar.hpp`](../include/adc/numerics/spatial_operator_polar.hpp)`::assemble_rhs_polar<Limiter, NumericalFlux>`
(`recon_prim`, `wall_radial`), via the named functors `detail::PolarFaceFluxRKernel` (radial flux
weighted by `r_face`, optional wall at the boundary faces), `PolarFaceFluxThetaKernel`,
`PolarAssembleRhsKernel`; the physical source and the geometric source are routed by the concepts
`PolarHasSource` / `PolarHasGeomSource` (`if constexpr`: zero codegen for a scalar brick,
ExB path bit-identical). Instantiated via `runtime/block_builder_polar.hpp`, wired in
`System::step` for `geometry == "polar"`. Poisson:
[`include/adc/numerics/elliptic/polar_poisson_solver.hpp`](../include/adc/numerics/elliptic/polar_poisson_solver.hpp)`::PolarPoissonSolver`
(FFT-in-theta `fft1d` reused from `poisson_fft.hpp` + complex `thomas_solve` in r; models the
concept `PolarEllipticSolver` `rhs()/phi()/solve()/residual()/geom()`). The aux is derived in the local
basis $(e_r, e_\theta)$: `aux[1] = d phi/dr`, `aux[2] = (1/r) d phi/d theta`
(`block_builder_polar.hpp`, `System::solve_fields_polar`).

**Polar tensor operator + polar Schur.** When the coupled implicit source goes polar (diocotron at
high $\omega_c$), the Schur condenses a full tensor operator
$A = I + c\,\rho\, B^{-1}$ with cross terms $a_{rt}, a_{tr}$ and a theta-dependent coefficient: the
FFT-in-theta of `PolarPoissonSolver` no longer applies (it requires a constant theta coefficient
without cross coupling).
[`include/adc/numerics/elliptic/polar_tensor_operator.hpp`](../include/adc/numerics/elliptic/polar_tensor_operator.hpp)`::PolarTensorKrylovSolver`
then solves by matrix-free BiCGStab (handles the non-symmetric of the cross term), preconditioned
`Jacobi` or `RadialLine` (radial Thomas per theta line, default). No MG V-cycle (stagnation on
$1/r^2$). Singular operator (pure radial Neumann + periodic theta): gauge fixed by projection onto
the subspace of zero FV mean (`project_mean`, the iterative counterpart of the mode-0 pinning). The
9-point stencil reads the diagonal corners filled by `fill_ghosts` (without which the cross term would be wrong at the
box boundary). `coupling/polar_condensed_schur_source_stepper.hpp::PolarCondensedSchurSourceStepper` is
the polar counterpart of `CondensedSchurSourceStepper` (#212). Multi-rank MPI / multi-box by azimuthal
splitting only under `RadialLine` (the Thomas sweep in r must stay local to a box, safeguard
`check_radial_columns`), free 2D tiling under `Jacobi`; polar Schur raised under MPI (#227).

**Constraints / remarks.** PolarPoissonSolver: single-rank scope, single box covering the ring
(the FFT-in-theta + tridiag-in-r requires the complete theta line AND the radial column on one rank; the
distributed would impose a parallel transpose, out of Phase 2a scope) -> hard safeguard (active in Release)
if `n_ranks() > 1` or `ba.size() != 1`, raised on all ranks (no deadlock); `solve()` /
`residual()` are `local_size()==0`-safe. Theta spectral: exact for a band-limited datum
(diocotron = few azimuthal modes), `dtheta` does not enter the eigenvalue. The tridiag is
diagonally dominant (azimuthal term $\le 0$, folded BC) -> Thomas stable without pivoting. The host
residence of the RHS is synchronized (`sync_host`) before any host read (a device kernel possibly in
flight; no-op under a host Kokkos space (Serial/OpenMP), targeted `device_fence` under Kokkos Cuda). PolarTensorKrylovSolver:
RadialLine $\sim$ moderately growing iteration count (isotropic $\times 2$ per grid doubling,
tensor $\times 2.4$); Jacobi grows in $1/h^2$ (sanity check / fallback). The cross term and the azimuthal
coupling are not in the preconditioner (an honest limit, later refinement possible).
Validation: `test_polar_transport_mms` / `test_polar_mms_vr` (polar transport MMS order 2),
`test_polar_ring_advection`, `test_polar_fluid_transport`, `test_polar_lorentz_source`,
`test_polar_conservation_radial_flux` (radial wall, mass conserved), `test_polar_poisson_mms`
(PolarPoissonSolver, radial order 2), `test_polar_tensor_elliptic_mms` (polar tensor operator),
`test_polar_condensed_schur_source_stepper`, `test_mpi_polar_schur` (polar Schur multi-rank).
`test_polar_system_step` validates the full polar `System::step` path (field-solve + aux in the local
basis + SSPRK3 transport + wall). On the Python side: `test_polar_system`, `test_polar_diocotron`,
`test_polar_rejections`, `test_polar_schur_via_system`, `test_polar_conservation_radial_flux`,
`test_polar_teardown_stability`.


---

## 17. AMR: Berger-Oliger subcycling + conservative reflux

**Intuition.** Refine only where needed. A fine level (step $\Delta x_c / r$) covers a
sub-region; to respect its CFL it does $r$ substeps of $\Delta t / r$ while the coarse does
a single step of $\Delta t$. At the fine-coarse interface, the two levels compute different fluxes,
which breaks discrete conservation. Reflux corrects the bordering coarse cell by the difference
(time-integrated fine flux minus coarse flux).

**Formula / discretization.** Let a fine-coarse face in $x$ between the coarse cell $(I, J)$ and the
fine patch. During the coarse step we have already advanced the coarse with its own face flux $F_c$ (over
$\Delta t$). We accumulate in parallel the fine flux of the same face over the $r$ substeps. The
correction replaces the coarse contribution by the fine contribution:

$$U_c(I,J) \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_{s=1}^{r} \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

with $\Delta t_f = \Delta t / r$ and $\bar F_f^{(s)}$ the fine flux averaged over the fine faces covering the
coarse face. In the code the integrated fine quantity is already $\sum_s \Delta t_f \bar F_f^{(s)}$
(stored `fL/fR/fB/fT` of the per-patch register) and the coarse quantity is $F_c$ (stored `cL/cR/cB/cT`),
multiplied by $\Delta t$ at the time of deposit. The ratio is fixed at $r = 2$ (`SubcyclingSchedule`),
hence the coarse footprint $I_0 = \mathrm{lo}/2$, $I_1 = (\mathrm{hi}-1)/2$ of an aligned fine patch
(`PatchRange`). The interpolation of the fine ghosts from the coarse is linear in time,
$U^\star = (1-\alpha)\,U_c^{\mathrm{old}} + \alpha\,U_c^{\mathrm{new}}$ with $\alpha = s/r$
(`SubcyclingSchedule::frac`), constant in space (piecewise injection).

```
function subcycle_level(coarse_level, fine_level, dt, r=2):
    # 1. avancer le grossier d'un pas dt, en memorisant son flux de face
    #    le long de l'interface fin-grossier avant de le mettre a jour
    Fc = sample_coarse_face_flux(coarse_level)        # cL,cR,cB,cT par patch
    advance_one_step(coarse_level, dt)

    # 2. r sous-pas fins de dt/r
    reg.f{L,R,B,T} = 0                                 # accumulateur fin time-integre
    for s in 1..r:
        alpha = s / r                                  # position temporelle (frac)
        fill_fine_ghosts(fine_level, Uc_old, Uc_new, alpha)   # interp espace + temps
        fill_boundary(fine_level)                      # fin-fin ecrase les ghosts couverts
        Ff = compute_face_fluxes(fine_level)
        advance_one_step(fine_level, dt/r)
        reg.f{L,R,B,T} += (dt/r) * mean_over_fine_faces(Ff)   # sum_s dt_f * Ff^(s)

    # 3. average_down : la zone couverte du grossier = moyenne du fin (conservatif)
    average_down(fine_level, coarse_level, r)

    # 4. reflux : corriger les cellules grossieres bordantes (non couvertes)
    cfi = CoarseFineInterface(coarse_region, fine_boxarray_global)
    for patch g in fine_level:
        cfi.route_reflux(reg[g], dx, dy, dt, flux_register, nc)
    flux_register.gather()                             # all_reduce_sum (identite en serie)
    for cell (I,J) bordante:
        Uc(I,J) -= flux_register.at(I,J,k)
```

**Code.** [`numerics/time/amr_reflux_mf.hpp`](../include/adc/numerics/time/amr_reflux_mf.hpp) is the
umbrella that aggregates the sub-headers. The unified production entry is `advance_amr` in
[`numerics/time/amr_advance.hpp`](../include/adc/numerics/time/amr_advance.hpp) (a faithful facade of the
N-level multi-patch engine `detail::amr_step_multilevel_multipatch`). The roles are promoted to named types
in [`numerics/time/amr_patch_range.hpp`](../include/adc/numerics/time/amr_patch_range.hpp):
`SubcyclingSchedule` (cadence $r$, $\Delta t/r$, $\mathrm{frac}(s)=s/r$), `PatchRange` (coarse footprint
$[I_0..I_1]\times[J_0..J_1]$ of a fine patch), `FluxRegister` (a global-index buffer, accumulation
`add`/`set` then `gather`), `CoverageMask` (shadowed cells), `CoarseFineInterface::route_reflux`
(bordering deposit). The inter-level transfers `average_down` (conservative average over $r\times r$ blocks),
`interpolate` (piecewise-constant injection) and `parallel_copy` are in
[`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp). The per-cell fine ghost goes through
`fill_cf_ghost_cell` (space + time interpolation), shared by the three variants `mf_fill_fine_ghosts_*`.

**Constraints / remarks.** The temporal ratio is fixed at $r = 2$: `PatchRange` uses the
historical arithmetic $(hi-1)/2$ for the upper bound, which is not `Box2D::coarsen` (floor of both bounds), and which
assumes aligned patches (even lo, odd hi). The order of operations is critical: the coarse flux
must be sampled before advancing the coarse; the average_down must precede any mass measurement,
otherwise the covered zone is counted twice. Validation: `test_refinement` (conservative average_down +
interpolate), `test_amr_hierarchy` (coarse + nested fine + ghost interpolation), `test_flux_register`
(register indexing), `test_coverage_mask` (covered-cell marking), `test_advance_amr`
(unified engine 2 AND 3 levels, maxdiff = 0), `test_amr_diagnostics` (mass and drift velocity via the
seam reducer).

## 18. Multi-patch AMR: coverage-aware reflux, MPI-distributed

**Intuition.** A fine level is not a single box but a set of patches. Two subtleties
follow: (a) at the joint between two neighboring patches (fine-fine interface) one must not reflux, because the
two sides are fine and the balance is already conservative; (b) the correction must go into the right
parent box when the coarse is itself multi-box or distributed across several MPI ranks.

**Formula / discretization.** The multi-patch reflux is the same operator as section 17, but filtered
by a coverage mask. For a bordering coarse cell $(I,J)$ adjacent to the face of a patch $g$,
the correction is poured only if $(I,J)$ is not shadowed by any fine patch:

$$U_c(I,J) \mathrel{-}= \mathbb{1}\big[\lnot\,\mathrm{covered}(I,J)\big]\cdot\frac{\bar F_f - F_c\,\Delta t}{\Delta x}$$

where $\mathrm{covered}(I,J)$ tests membership in the coarse footprint `PatchRange` of any fine
patch. The mask is built on the global BoxArray (all patches, known to all ranks), so independent
of the MPI distribution. The flux register has a global indexing: each rank fills its local
contributions (zero elsewhere), then $\mathrm{buf} \leftarrow \sum_{\text{rangs}} \mathrm{buf}$ by
`all_reduce_sum_inplace`; in serial the all_reduce is the identity, so bit-for-bit identical to the mono-rank.

```
function reflux_multipatch(coarse_level, fine_boxarray_global, registers, distribution):
    # masque sur le box_array global -> correct sous n'importe quelle distribution MPI
    cfi = CoarseFineInterface(coarse_region, fine_boxarray_global)
        for g in fine_boxarray_global:
            cfi.cmask.mark(PatchRange(g).box())        # empreinte grossiere de chaque patch

    flux_register = FluxRegister(coarse_region, nc)    # index global, buf=0
    for patch g OWNED-LOCALLY by this rank:
        # route gauche/droite (x) puis bas/haut (y), uniquement sur cellules non couvertes
        for J in g.J0..g.J1, k in 0..nc:
            if not cfi.covered(g.I0-1, J): ref.add(g.I0-1, J, -(fL - cL*dt)/dx)
            if not cfi.covered(g.I1+1, J): ref.add(g.I1+1, J, +(fR - cR*dt)/dx)
        for I in g.I0..g.I1, k in 0..nc:
            if not cfi.covered(I, g.J0-1): ref.add(I, g.J0-1, -(fB - cB*dt)/dy)
            if not cfi.covered(I, g.J1+1): ref.add(I, g.J1+1, +(fT - cT*dt)/dy)

    flux_register.gather()                             # all_reduce_sum sur tous les rangs

    if coarse REPLICATED (default):
        apply correction locally (chaque rang a la copie complete)
    else: # coarse DE-replique (multi-box reparti)
        parallel_copy correction into the owning coarse box
        average_down zone couverte via mf_average_down_mb / parallel_copy
```

**Code.** The coverage-aware types live in
[`numerics/time/amr_patch_range.hpp`](../include/adc/numerics/time/amr_patch_range.hpp):
`CoverageMask` (built on the coarse region, `mark` marks the intersected footprint, `covered` is
bounded outside the region), `CoarseFineInterface` (assembles the mask on `fine_ba.size()` global patches and
exposes `route_reflux`, a named function templated on the register type `Reg`/`RegMP` hence safe under nvcc),
`FluxRegister::gather` (inter-rank sum by `all_reduce_sum_inplace`). The MPI routing of the distributed
coarse goes through `parallel_copy` in
[`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp) (general redistribution between two MultiFab
on the same domain with different decompositions: local copies via `BoxHash::query`, then
`MPI_Isend`/`MPI_Irecv` jobs enumerated deterministically, tag 1). The replicated coarse fills its
periodic ghosts by `fill_periodic_local` (a purely local self-fold, without an MPI plan). The
`coarse_replicated` flag of `LevelHierarchy` (default `true`) is passed to the engine by `advance_amr` in
[`numerics/time/amr_advance.hpp`](../include/adc/numerics/time/amr_advance.hpp); without this passing, a
de-replicated coarse would revert to replicated mode (`mf_find_box` instead of `parallel_copy`).

**Constraints / remarks.** Without a coverage mask, the fine-fine joint would be refluxed twice, hence
non-conservation: the mask is the central invariant of the correction. The register must be filled locally
(zero elsewhere) before `gather`, otherwise the all_reduce double-counts. Bit-for-bit reproducibility requires a
deterministic enumeration order of the `parallel_copy` jobs (spatial hash on the source, sorted candidates).
Validation: `test_amr_spatial_parity` (the spatial core of the AMR path is identical to that of `System`:
same primitive reconstruction, same HLLC/Roe flux), `test_mpi_mbox_parity` (residual invariant to the box
splitting AND to the number of ranks np = 1/2/4, dmax = 0), `test_mpi_amr_distributed_coarse` (distributed coarse
identical to the replicated coarse bit-for-bit, np = 1/2/4).

## 19. Berger-Rigoutsos clustering and regrid

**Intuition.** Given the tagged cells (strong gradient, or any physical predicate), find a
small number of rectangular boxes that cover them without too much waste. The algorithm cuts
recursively a region where the signature (histogram of tags projected onto an axis) presents a hole
(empty column), otherwise an inflection (extremum of the change of Laplacian of the signature), otherwise at the middle.

**Formula / discretization.** For a region $R$, we define the efficiency $\eta = N_{\mathrm{tag}}(R) / |R|$
(fraction of tagged cells). We accept $R$ as a box if $\eta \ge \eta_{\min}$
(`min_efficiency`, default $0.7$) or if $R$ is not splittable. Otherwise we cut. The signature on axis
$a$ is the projection $s_a[k] = \sum_{\text{ligne/col } k} \mathrm{tag}$. A hole is an interior index
$k \in [\mathrm{mb}, \mathrm{len}-\mathrm{mb}]$ with $s_a[k] = 0$, the closest to the center. Failing that, we
take the inflection: discrete Laplacian $D[k] = s[k+1] - 2 s[k] + s[k-1]$, and we cut at the index that
maximizes $|D[k] - D[k-1]|$. Failing that again, we cut at the middle of the largest splittable dimension.
The splittability criterion is $n_a \ge 2\,\mathrm{mb}$ with $\mathrm{mb} = \max(1, b)$ with $b$ = `min_box_size`.
After acceptance, each raw box is chopped into sub-boxes of side $\le$ `max_box_size`. At the regrid,
the coarse boxes are refined by `refine(ref_ratio)` in the index space of the fine level.

```
function berger_rigoutsos(tags, params):
    raw = []
    cluster_rec(tags, tags.box, params, raw)
    result = []
    for b in raw:
        result += chop(b, params.max_box_size)         # BoxArray::from_domain
    return result

function cluster_rec(tags, region, p, out):
    region = bounding_box_of_tags(region)               # trim
    if region empty: return
    eff = count_tags(region) / num_cells(region)
    mb  = max(1, p.min_box_size)
    sx  = region.nx >= 2*mb ;  sy = region.ny >= 2*mb
    if eff >= p.min_efficiency or (not sx and not sy):
        out.push(region) ; return                       # accepte
    Sx = signature(region, axis=0) ; Sy = signature(region, axis=1)
    hx = sx ? best_hole(Sx, mb) : -1                    # trou interieur le plus central
    hy = sy ? best_hole(Sy, mb) : -1
    choose (axis, kcut):
        both holes  -> couper l'axe le plus long
        one hole    -> cet axe
        no hole     -> best_inflection (max |Laplacien'|), score le plus fort
        no infl.    -> milieu de la plus grande dim splittable
    (left, right) = split(region, axis, kcut)
    cluster_rec(tags, left, p, out)
    cluster_rec(tags, right, p, out)

function regrid_level(hierarchy, coarse_lev, crit, params):
    tags  = tag_cells(data(coarse_lev), domain, crit)   # predicat (Array4, i, j) -> bool
    grown = grow_tags(tags, n_buffer, domain)           # dilatation (nesting + buffer)
    if grown.count() == 0:
        clear_above(coarse_lev) ; return                # plus rien a raffiner
    # MPI : OU global des tags avant clustering, sinon BoxArray fin diverge par rang
    all_reduce_or(grown)                                 # (grossier reparti)
    cboxes = berger_rigoutsos(grown, params.cluster)
    fboxes = [ b.refine(ref_ratio) for b in cboxes ]
    newfine = MultiFab(fboxes, DistributionMapping, ncomp, n_grow)
    interpolate(data(coarse_lev), newfine, ref_ratio)   # injection grossier -> fin
    if niveau fin existant: parallel_copy(newfine, data(coarse_lev+1))  # preserver l'ancien fin
    install_level(coarse_lev+1, fba, newfine)
```

**Code.** The clustering is `berger_rigoutsos` in
[`amr/cluster.hpp`](../include/adc/amr/cluster.hpp), with the helpers `detail::tag_bbox` (trim),
`detail::signature`, `detail::best_hole`, `detail::best_inflection` (max $|D[k]-D[k-1]|$),
`detail::cluster_rec` (recursion), and the final chop by `BoxArray::from_domain(b, max_box_size)`. The
parameters are `ClusterParams` (`min_efficiency`, `min_box_size`, `max_box_size`). The regrid is
`regrid_level` in [`amr/regrid.hpp`](../include/adc/amr/regrid.hpp): `tag_cells` (generic predicate
on `ConstArray4`), `grow_tags` (square dilation bounded to the domain), `berger_rigoutsos`, then
`Box2D::refine(ref_ratio)`, `interpolate` and `parallel_copy` (to preserve the values of the old fine)
of [`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp), finally `AmrHierarchy::install_level`. Without
a tag, `clear_above` removes the fine level and the finer ones. Under MPI, the global OR of the tags
(`all_reduce_or_inplace`) must precede the clustering, otherwise the fine BoxArray would differ per rank.

**Constraints / remarks.** The clustering is pure, sequential, without physics nor MPI: it consumes a
`TagBox` already gathered. The proper nesting (each fine patch strictly interior to the parent
coverage) relies on the dilation `grow_tags` (radius `n_buffer`) and must be guaranteed after the clustering,
otherwise the inter-level ghost-fill reads outside the parent coverage. The tagging predicate is agnostic
of the physics; for a gradient criterion the caller fills the ghosts beforehand. The signature pushes the
cuts toward the real geometric holes: a full block gives 1 box, two blocks separated by an empty
band give 2 boxes. Validation: `test_cluster` (full block -> 1 box, two separated blocks -> 2 boxes,
large block chopped by `max_box_size`), `test_regrid` (a fine level is created around the tagged region,
fine data interpolated from the coarse).

**Export of the patch geometry.** The fine `BoxArray` resulting from the clustering is exposed to Python by
`AmrSystem.patch_boxes()` (a list of `(level, ilo, jlo, ihi, jhi)`, inclusive corners in the index space
of the level, ratio 2) and the facade `AmrSystem.patch_rectangles()` (conversion into physical rectangles
`(x0, y0, w, h)` on `[0, L]^2`). Same source as `n_patches()` (the same global `box_array()`, so
rank-independent and MPI-safe); it is a read between the steps, with no cost on the hot path. Wired
on both engines (mono-block `AmrCouplerMP` and multi-block `AmrRuntime`). Allows tracing the real
patches (for example a GIF of the refinement) without rebuilding a proxy. Validation:
`test_amr_patch_boxes` (cardinality equal to `n_patches`, corners consistent in index and in physics, mono and
multi-block).


---

## 20. Distributed mesh: global BoxArray, halos, load balancing

**Intuition.** Distributed AMR requires that all ranks know all the boxes (global
BoxArray) to compute the multi-patch coverage and enumerate the halo exchanges in a
deterministic way, but that each rank allocates only its local fabs (via the `DistributionMapping`).
The halo exchange fills the parallel ghosts; the load balancing distributes the boxes across the ranks.

**Formula / discretization.** The tiling `from_domain(domain, m)` splits each axis `[lo, hi]`
(length `len = hi - lo + 1`) into `n = ceil(len / m)` segments distributed as evenly as possible:
the first `len mod n` segments have `floor(len/n) + 1` cells, the others `floor(len/n)` (no
greedy tail). The global box index is its position in the `y` outer / `x` inner order,
identical on all ranks.

The load balancing minimizes the imbalance `max_r charge(r) / moyenne(charge)`, the load of a box
being its number of cells (cost proxy). Two strategies: Z-order (Morton curve, contiguous
segments of target load `total / nranks` -> spatial locality) and knapsack/lpt (the heaviest box
to the least loaded rank -> minimal maximal imbalance, without locality). The Morton key interleaves
the bits of `(x, y)` brought back to the origin of the bounding box:

$$\mathrm{morton}(x, y) = \sum_{b\ge 0} \big(x_b\,4^b + y_b\,2\cdot 4^b\big),\qquad
  \mathrm{cible}_r = \frac{\text{total cellules}}{\text{nranks}}$$

The halo exchange fills the ghost `D(i,j)` of a fab from the valid cell shifted
`S(i - s_x, j - s_y)` of the neighbor, where `(s_x, s_y)` ranges over
`{0, +-L_x} x {0, +-L_y}` (the periodic shifts are active only if the direction is periodic).

```
function from_domain(domain, m):                      # BoxArray::from_domain
    sx = split_range(domain.lo[0], domain.hi[0], m)   # segments en x
    sy = split_range(domain.lo[1], domain.hi[1], m)   # segments en y
    boxes = []
    for (ylo, yhi) in sy:        # y externe, x interne -> ordre deterministe (= indice global)
        for (xlo, xhi) in sx:
            boxes.push( Box2D{{xlo, ylo}, {xhi, yhi}} )
    return BoxArray(boxes)

function split_range(lo, hi, m):
    len = hi - lo + 1 ; n = ceil(len / m)
    base = len div n ; rem = len mod n ; cur = lo
    for k in 0..n-1:
        l = base + (1 if k < rem else 0)              # rem premiers segments +1
        emit (cur, cur + l - 1) ; cur += l

function make_sfc_distribution(ba, nranks):           # load_balance.hpp (Z-order)
    order = argsort boxes by morton_key(lo - bounding_box.lo)
    target = ba.num_cells() / nranks ; acc = 0 ; r = 0 ; rank[*] = 0
    for k, b in enumerate(order):
        rank[b] = r ; acc += ba[b].num_cells()
        if r < nranks-1 and acc >= target*(r+1) and boxes_left >= ranks_left:
            r += 1                                    # garantit >=1 box/rang
    return DistributionMapping(rank)

function fill_boundary_begin(mf, domain, per):        # halos, 2 phases
    shifts = product({0,+-Lx if per.x}, {0,+-Ly if per.y})
    hash = BoxHash(mf.box_array())                    # accelere la recherche de voisins
    for li in local fabs of mf:                       # --- copies locales ---
        gbox = grow(fab(li).box, ng)
        for (sx, sy) in shifts:
            for gB in hash.query( gbox shifted by (-sx,-sy) ):
                if gB local: copy_shifted(fab(li), fab(gB), region, sx, sy)
    if n_ranks() <= 1: return
    for gF in 0..ba.size()-1:                          # --- enum globale deterministe ---
        for (sx, sy) in shifts:
            for gB in hash.query(...):
                if owner(gF)==me xor owner(gB)==me:    # une extremite locale
                    classer en send[owner(gF)] ou recv[owner(gB)]
    pack(send buffers via for_each PackKernel) ; device_fence()
    post MPI_Isend / MPI_Irecv (tag 0, pointeurs unifies -> GPUDirect si CUDA-aware)

function fill_boundary_end(mf, h):
    MPI_Waitall(h.reqs)
    unpack(recv buffers via for_each UnpackKernel) -> ghosts
```

**Code.** [`mesh/box_array.hpp`](../include/adc/mesh/box_array.hpp) (`BoxArray::from_domain`,
`split_range`, the vector order is the box identity);
[`mesh/distribution_mapping.hpp`](../include/adc/mesh/distribution_mapping.hpp)
(`DistributionMapping`, round-robin `i % nranks` by default, replicated metadata);
[`mesh/multifab.hpp`](../include/adc/mesh/multifab.hpp) (`MultiFab` allocates only the fabs where
`dm_[i] == my_rank()`, iterates over `local_size()`, `global_index` / `local_index_of` bridge);
[`mesh/fill_boundary.hpp`](../include/adc/mesh/fill_boundary.hpp) (`fill_boundary_begin` /
`fill_boundary_end` non-blocking + `fill_boundary` blocking, `HaloExchange` owns the buffers and
`MPI_Request`, kernels `CopyShiftedKernel` / `PackKernel` / `UnpackKernel` device-clean);
[`parallel/load_balance.hpp`](../include/adc/parallel/load_balance.hpp) (`morton_key`,
`make_sfc_distribution`, `make_knapsack_distribution`, `load_imbalance`);
[`parallel/comm.hpp`](../include/adc/parallel/comm.hpp) wraps the collectives
(`all_reduce_sum`, `all_reduce_sum_inplace` for the reflux, `all_reduce_or_inplace` for the union of the
regrid tags) and degenerates cleanly in serial.

**Constraints / remarks.** The metadata (BoxArray + DistributionMapping) are replicated on
all ranks: this is what makes the enumeration of the halo jobs deterministic, so the buffers
`sbuf[A->B]` and `rbuf[B<-A]` align without negotiating sizes. If MPI is not initialized,
`my_rank()` returns 0 and `n_ranks()` returns 1 (a serial test linked against the MPI lib does not break), hence
`comm_init()` required at the beginning of `main()` for a real distributed run. The out-of-domain ghosts
without periodicity are not touched by `fill_boundary` (they are the physical BCs,
`physical_bc.hpp`). The buffers live in unified memory and are passed as-is to MPI (host
bounce avoided if the MPI stack is CUDA-aware); a `device_fence()` separates the pack from the `Isend`.
**Validation.** `test_box_array`, `test_multifab`, `test_load_balance`; under MPI:
`test_mpi_fillboundary` (halo exchange), `test_mpi_poisson` (distributed Poisson),
`test_mpi_fft_distributed` (FFT by bands), `test_mpi_redistribute`, `test_mpi_array_reduce`,
`test_mpi_coupler_inject` (np=4, results bit-identical to np=1/2/4).

## 21. Extensible aux channel

**Intuition.** The hyperbolic-elliptic coupling goes through an `aux` channel with base contract
`(phi, grad_x, grad_y)`, that is `kAuxBaseComps = 3` components. Some models need additional
fields: an out-of-plane magnetic field `B_z` (Lorentz force), an electronic temperature
`T_e` derived from `p/rho`. The channel is widened on demand, and stays bit-identical to the
history when `n_aux = 3` (the extra components equal 0, never read).

**Formula / discretization.** A model declares a static member `n_aux > 3`; the effective
width read by the spatial operator and allocated by the runtime is

$$\mathrm{naux}(M) =
  \begin{cases} M\text{::n}_\text{aux} & \text{si } M \text{ le declare}\\ 3 & \text{sinon}\end{cases}$$

evaluated at compile time by `aux_comps<M>()`. A magnetized brick sets `n_aux = 4`: `B_z` occupies
component 3, a pure function of position sampled per level; `T_e` (component 4) is
derived from a fluid block. A `CompositeModel<Hyp, Src, Ell>` propagates the maximal aux width of its
three bricks to the system, which allocates and marshals the right number of components.

```
function aux_comps<M>():                       # physical_model.hpp, constexpr
    if requires { M::n_aux }: return M::n_aux   # ex. 4 si une brique lit B_z
    else: return kAuxBaseComps                  # 3 : phi, grad_x, grad_y

# cote bloc compile / runtime : on elargit le canal avant de capturer son adresse
function add_compiled_model(sys, name, model, ...):  # dsl_block.hpp
    sys.ensure_aux_width( aux_comps<Model>() )       # MultiFab aux >= naux comp (no-op si 3)
    ctx = sys.grid_context()                         # ctx.aux pointe l'aux reel, large
    install_block(...)                               # assemble_rhs lit load_aux<naux>

# cote ABI plate (marshaling) : le tableau aux_in porte exactement naux composantes
function make_grid(n, dx, dy, periodic, aux_in, naux):  # compiled_block_abi.hpp
    aux = MultiFab(ba, dm, naux, 1) ; aux.set_val(0)
    for c in 0..naux-1, j, i: aux(i,j,c) = aux_in[c*n*n + j*n + i]
    fill ghosts (memes CL que le System) -> load_aux<naux> lit B_z / T_e
```

**Code.** [`core/physical_model.hpp`](../include/adc/core/physical_model.hpp):
`aux_comps<M>()` (detects `M::n_aux` via `requires`, falls back to `kAuxBaseComps = 3`), lives in the
contract header so that `CompositeModel` propagates `n_aux` without pulling in the numerics; the concept
`PhysicalModel` enforces `M::Aux == adc::Aux`. On the virtual dispatch side,
[`runtime/dynamic_model.hpp`](../include/adc/runtime/dynamic_model.hpp):
`IModel<NV>::n_aux()` (default `kAuxBaseComps`), `ModelAdapter<M>::n_aux()` returns `aux_comps<M>()`.
The widening is anchored in `System::ensure_aux_width` (called by
[`runtime/dsl_block.hpp`](../include/adc/runtime/dsl_block.hpp) before `grid_context()`), and the
flat marshaling in [`runtime/compiled_block_abi.hpp`](../include/adc/runtime/compiled_block_abi.hpp)
(`make_grid(..., naux)`, symbol `adc_compiled_naux()` = `aux_comps<MODEL>()`).

**Constraints / remarks.** The widening must precede the capture of the aux address (otherwise the
closure would read a too-short aux). `B_z` being a function of position, it is sampled
per level (exact at the coarse like `eps`, order 2 preserved). A base model (`n_aux` not declared)
falls back to 3 -> allocation and results bit-identical to the history. **Validation.**
`test_aux_extra` (a model declares `n_aux > 3`), `test_aux_composite` (a `CompositeModel` propagates
the aux width of its bricks), `test_aux_coupler_bz` / `test_aux_system_bz` / `test_amr_aux_bz` /
`test_amr_system_bz_pop` / `test_amr_system_bz_multibox` (B_z read and populated along the
coupler, system, AMR, multi-box paths), `test_aux_te` (T_e derived from `p/rho`), `test_aux_single_source`
(a single source generates `load_aux` + marshaling, all fields covered). Validated bit-identical on
GH200 (B_z device single + multi-box, cf. GPU_RUNTIME_PORT.md).

## 22. Runtime composition and multi-species system

**Intuition.** Python composes what (one block per species: composite model + spatial scheme + temporal
policy), C++ computes per cell. N species interact in the elliptic right-hand side
(`f = sum_s q_s n_s`) and in the inter-species source, never in the flux: a block's flux only
sees its own state and the shared aux.

**Formula / discretization.** At each macro-step, the system solves the shared Poisson whose
right-hand side is the co-located sum of the elliptic bricks of all the blocks, populates the aux
`(phi, grad phi)`, then advances each block according to its policy (explicit / IMEX, substeps, stride):

$$f_{ij} = \sum_{b} \mathrm{elliptic\_rhs}_b(U^b_{ij}),\qquad
  \frac{dU^b}{dt} = -\,\mathrm{div}\,F_b(U^b, \mathrm{aux}) + S_b(U^b, \mathrm{aux})$$

A model is assembled by `dispatch_model(spec, visitor)`: it builds the transport brick
(`exb` / `compressible` / `isothermal`), the source brick (`none` / `potential` / `gravity` /
`magnetic` / `potential_magnetic`, the fluid sources requiring `NV >= 3`), the elliptic brick
(`charge` / `background` / `gravity`), combines them into `CompositeModel<TR, Src, Ell>` and calls
`visitor(model)`. The core names no scenario; a scenario is this composition, named on the
`adc_cases` side.

```
function dispatch_model(spec, visitor):              # model_factory.hpp
    dispatch_transport(spec, TR ->                    # exb | compressible | isothermal
      dispatch_source<TR::n_vars>(spec, Src ->        # none | potential | gravity | (potential_)magnetic
        dispatch_elliptic(spec, Ell ->                # charge | background | gravity
          visitor( CompositeModel<TR, Src, Ell>{TR, Src, Ell} ))))
    # combinaison invalide (source fluide sur transport scalaire) -> throw

function System.step(dt):                            # runtime/system.hpp
    solve_fields()                                    # Poisson partage : f = sum_b elliptic_rhs_b
    fill aux (phi, grad phi[, B_z, T_e])
    for b in blocks:
        if not b.evolve: continue                     # espece gelee : vue par Poisson, non avancee
        if b held by stride this macro-step: continue # multirate hold-then-catch-up
        advance b (explicit SSPRK / IMEX, b.substeps sous-pas)

function System.step_cfl(cfl):
    dt = cfl * h * min_b( substeps_b / (stride_b * w_b) )   # honore substeps + stride
    step(dt) ; return dt
```

**Code.** [`runtime/system.hpp`](../include/adc/runtime/system.hpp): `System` (multi-block
single-level, shared Poisson, `add_block(name, ModelSpec, limiter, riemann, recon, time, substeps,
evolve, stride, implicit_vars, implicit_roles)`, `step` / `step_cfl`, `evolve=false` for a frozen
species seen by Poisson); [`runtime/amr_system.hpp`](../include/adc/runtime/amr_system.hpp):
`AmrSystem` (1 block -> historical mono-model `AmrCouplerMP<Model>`; `>= 2 add_block` -> engine
`AmrRuntime` multi-block on a shared hierarchy, same BoxArray + DistributionMapping + dx/dy per
level via `same_layout_or_throw`, coarse Poisson co-located sum, conservation per block,
`add_coupled_source` for the inter-species sources, `n_blocks()`). On the coupling side:
`coupling/system_coupler.hpp` (`SystemAssembler` assembles, `SystemDriver` advances),
`coupling/amr_system_coupler.hpp` (the system carried over AMR).
[`runtime/model_factory.hpp`](../include/adc/runtime/model_factory.hpp):
`dispatch_model` / `dispatch_transport` / `dispatch_source` / `dispatch_elliptic` assemble a
`CompositeModel` from a `ModelSpec` (the core names no scenario).

**Constraints / remarks.** The blocks share an aux and a Poisson; the coupling between species
goes through the elliptic right-hand side (sum) and the coupled sources, not through the flux. `substeps`
and `stride` are orthogonal (a slow block on `stride=M` is held M-1 steps then catches up by an effective
step `M*dt`); between two catch-ups the held block enters the Poisson sum with its stale state.
In multi-block AMR, `regrid_every > 0` is refused (the tag-union regrid is a later PR)
and `set_conservative_state` is mono-block only. Without an explicit IMEX mask
(`implicit_vars` / `implicit_roles` empty) the model default applies -> bit-identical.
**Validation.** `test_system_abstraction`, `test_system_coupler`, `test_two_species_minimal`,
`test_coupled_source` (inter-species source), `test_system_two_explicit`, `test_assembler_driver`
(the assembler assembles, the driver advances), `test_amr_system_coupler`, `test_system_hardening`,
`test_variable_role` (addressing a component by its physical role rather than by index).

## 23. Symbolic DSL: codegen, JIT, AOT

**Intuition.** A mini-DSL on the Python side describes a model in formulas and emits it as a C++ brick: flux,
source, elliptic right-hand side, with common subexpression elimination (CSE). Three implementation
paths, from the most dynamic to the most native, with a growing dispatch / performance / GPU
trade-off.

**Formula / discretization.** The DSL emits a `CompositeModel<Hyp, Src, Ell>` (the same type as the native
composition). The three wirings differ by WHERE the boundary lives and WHAT transits:

- JIT type-erased: the `.so` exposes `IModel<NV>*` (virtual dispatch per cell). Host path,
  Rusanov order 1, to prototype without recompiling the core. `ModelAdapter<M>` wraps the static
  model and forwards `source` / `elliptic_rhs` / conversions when `M` exposes them (default otherwise).
- AOT marshaled: `extern "C"` ABI (`compiled_block_abi.hpp`). The `.so` runs the production path
  (`assemble_rhs<Limiter, Flux>`, SSPRK2/IMEX) on the `CompositeModel` known at ITS
  compilation; only flat component-major arrays `c*n*n + j*n + i` cross the `dlopen`
  (no shared C++ symbol). No AMR nor MPI. The runtime params arrive through the symbols
  suffixed `_p` (`pvals`, `npar`), seeded by the declaration defaults.
- AOT native: `add_compiled_model` wires the `CompositeModel` known at compile time as a native
  block of the `System`, on the real grid context (`grid_context`), without marshaling: the residual
  does `fill_boundary` (MPI halos) + `assemble_rhs` (Kokkos) on the real MultiFab. To stay
  device-clean, the transport and the mesh go through named functors (no extended
  lambdas, which nvcc does not emit reliably cross-TU).

```
function add_compiled_model(sys, name, model, lim, riem, recon, time, gamma, substeps, ...):  # dsl_block.hpp
    imex = (time == "imex") ; recon_prim = (recon == "primitive")
    method = "ssprk3" if time=="ssprk3" else "ssprk2"     # ignore en imex
    sys.ensure_aux_width( aux_comps<Model>() )            # canal aux assez large (B_z...)
    ctx = sys.grid_context()                              # vrais dom/CL/aux (zero-copie)
    clo = make_block(model, lim, riem, ctx, imex, recon_prim, method)   # foncteurs nommes
    ms  = make_max_speed(model, ctx)
    pr  = make_poisson_rhs(model)
    sys.install_block(name, Model::n_vars, conservative_vars, primitive_vars, gamma, clo, ms, pr,
                      substeps, evolve, stride)
    sys.set_block_conversion(name, to_primitive, to_conservative)   # cons<->prim du modele
    sys.set_block_ghosts(name, block_n_ghost(lim))         # weno5 -> 3 ghosts (sinon lecture hors bornes)

# AOT marshale : un .so genere se reduit a
#   using Model = adc::CompositeModel<...>;  ADC_DEFINE_COMPILED_BLOCK(Model)
# qui emet adc_compiled_residual/_advance/_max_speed/_poisson_rhs (+ variantes _p params runtime)
function residual<Model>(U, R, aux_in, n, dx, dy, periodic, lim, riem, recon_prim):  # compiled_block_abi.hpp
    lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>())
    Umf = MultiFab(ncomp=n_vars, ghosts=block_n_ghost(lim)) ; fill_interior(Umf, U)
    clo = make_block(model, lim, riem, ctx, imex=false, recon_prim)
    clo.rhs_into(Umf, Rmf) ; extract(Rmf, R)               # device_fence() avant la lecture hote
```

**Code.**
- JIT: [`runtime/dynamic_model.hpp`](../include/adc/runtime/dynamic_model.hpp) (`IModel<NV>`
  virtual, `ModelAdapter<M>`, `make_dynamic`); `System.add_dynamic_block` wires a virtual-dispatch
  model (host path, Rusanov, prototyping).
- AOT marshaled: [`runtime/compiled_block_abi.hpp`](../include/adc/runtime/compiled_block_abi.hpp)
  (`make_grid`, `fill_interior` / `extract`, `residual` / `advance` / `max_speed` / `poisson_rhs`,
  macro `ADC_DEFINE_COMPILED_BLOCK`, runtime params via `make_model_with_params` and the symbols
  `_p`); `System.add_compiled_block` (`extern "C"` ABI, without AMR nor MPI).
- AOT native: [`runtime/dsl_block.hpp`](../include/adc/runtime/dsl_block.hpp) (`add_compiled_model`,
  wires a `CompositeModel` known at compile time as a native block, `ensure_aux_width` +
  `grid_context` + `make_block` + `install_block` + `set_block_ghosts`). The
  device-clean machinery is `runtime/block_builder.hpp` (named functors `BlockRhsEval`, `AdvanceExplicit`,
  `AdvanceImex`, `MaxSpeed`; see the seam section 24).

**Constraints / remarks.** The type-erased JIT costs an indirect jump per cell (out of the
high-performance hot path); the AOT marshaled recopies the arrays at each call but stays mono-rank; the AOT
native is the only zero-copy / GPU / MPI / AMR path. The native path loads a `.so` via a loader
([`runtime/native_loader.hpp`](../include/adc/runtime/native_loader.hpp)) which compares an ABI key
(`abi_key`: header signature, compiler, C++ standard) between the model's `.so` and the module
already loaded; a divergence is refused cleanly (no loading of an incompatible `.so`). The
parity is locked at each level.
**Validation.** On the C++ side: `test_dynamic_model` (type-erased model == static Euler),
`test_block_builder` (block closures instantiable outside System), `test_compiled_model_parity`
(AOT native == native block on CPU/Serial), `test_amr_compiled_model` (AOT native on an AMR hierarchy).
On the Python side: the `test_dsl*` suite (flux/source/elliptic codegen, CSE, JIT `.so`, type-erased
dispatch, recon, roles, aux). On GH200, the named-functor path is validated bit-identical
(GPU_RUNTIME_PORT.md, phase 9); `add_compiled_model` with extended lambdas still hits an nvcc
limit (phase 8).

## 24. The dispatch seam (Kokkos: Serial / OpenMP / Cuda / MPI)

**Intuition.** Not a numerical algorithm but the switch point that makes them all portable.
`for_each_cell(box, f)` dispatches the loop over the cells of a `Box2D` to Kokkos, the only on-node
backend; the execution space (Serial sequential, OpenMP multi-thread, Cuda/HIP GPU) is chosen AT
THE INSTALLATION OF KOKKOS, not by an adc flag. The operators (assemble_rhs, V-cycle, couplers)
never see the execution space and no CUDA kernel is hand-written. Detail in
[ARCHITECTURE.md](ARCHITECTURE.md) section 4 (execution layer).

**Formula / discretization.** The functor `f(i, j)` is taken by value and captures only
`Array4` handles (POD), never the `Fab` nor anything virtual: exactly the constraint of a device
kernel. It always becomes `Kokkos::parallel_for(MDRangePolicy<Rank<2>, IndexType<int>>)` (signed
indices for the ghost boxes with negative bounds), instantiated for the Kokkos execution space
chosen at install (Serial, OpenMP or Cuda/HIP); no `#pragma omp` nor hand-written double loop
is a production path. Bit-identity: `for_each_cell` has no
inter-iteration dependency (each `f(i,j)` writes the single cell `(i,j)` and reads cells it does
not write in the same call: red-black GS smoother, residual/restriction/prolongation write
a distinct destination), so the result is independent of the order. The reductions carry a
FP choice: the `Kokkos::Sum` sum re-associates the addition per tile (non-associative in IEEE754), so
`sum` is deterministic/idempotent (same data, same Kokkos space -> same bits) but is NOT
bit-identical to a hand-written lexicographic sum, and this holds for ALL spaces (Serial,
OpenMP, Cuda) since there is only a single Kokkos path. The max is exact everywhere (associative/commutative,
without roundoff). A threshold `ADC_FOREACH_SERIAL_THRESHOLD` (default 4096 cells) switches to a small
sequential host loop (an optimization INTERNAL to the Kokkos path, not a separate backend) for the
small boxes (coarse V-cycle levels ~2x2..32x32) where the fork/join would crush the computation, but
only if the default Kokkos execution space is the host space (`if constexpr`: on device,
parallel_for whatever the size, otherwise a data race).

```
function for_each_cell(box b, f):                    # for_each.hpp  (#error sans ADC_HAS_KOKKOS)
    if DefaultExecSpace == DefaultHostExecSpace:     # if constexpr (espace Kokkos hote)
        if (b.nx * b.ny) < foreach_serial_threshold():
            for j in b: for i in b: f(i, j)          # petite boucle hote, INTERNE au chemin Kokkos
            return
    Kokkos::parallel_for( MDRangePolicy<Rank<2>, IndexType<int>>(lo, hi+1), f )  # Serial/OpenMP/Cuda

function for_each_cell_reduce_sum(b, f):             # Kokkos::Sum deterministe par tuile
    Kokkos::parallel_reduce(..., acc += f(i,j), Sum<Real>)   # reassocie : non bit-id a une somme lexicographique

function sync_host():  device_fence()                # avant un acces hote (memoire unifiee)
function sync_device(): pass                          # no-op sous SharedSpace (scaffolding)
```

**Code.** [`mesh/for_each.hpp`](../include/adc/mesh/for_each.hpp): `for_each_cell` (`Kokkos::parallel_for`
on the execution space chosen at install, `#error` without `ADC_HAS_KOKKOS`, guard `if constexpr` device,
threshold `foreach_serial_threshold` for the internal small host loop),
`for_each_cell_reduce_sum` / `_max` (reducers `Kokkos::Sum` / `Max` deterministic), the variants
with a reducer functor `reduce_sum_cell` / `reduce_max_cell` (passed directly to `parallel_reduce`
without a wrapper lambda, a device-clean cross-TU path for a Model-template kernel), and the
coherence seam `sync_host()` (= targeted `device_fence()`) / `sync_device()` (no-op under unified memory).
The fabs and the reduction `sum(MultiFab)` (all-reduce on all ranks) live in
[`mesh/multifab.hpp`](../include/adc/mesh/multifab.hpp). The MPI collectives are wrapped in
[`parallel/comm.hpp`](../include/adc/parallel/comm.hpp) (`all_reduce_sum`, `all_reduce_max`,
`all_reduce_sum_inplace`, `all_reduce_or_inplace`, `barrier`, `comm_init` / `comm_finalize`), which
degenerate into the serial identity.

**Constraints / remarks.** The CPU -> GPU switch changes no call site: one changes the
Kokkos execution space at install, the physics stays unchanged. The functor must be device-callable
under Kokkos (annotated `ADC_HD`, POD captured by value); capturing an object with a vtable or a `Fab`
breaks the device. The switch to the small host loop of the threshold is safe only under a host Kokkos space
(the `if constexpr` evaporates on device, zero overhead, the GPU path strictly unchanged). GPU
discipline: `device_fence()` (via `sync_host`) between a device kernel and a host loop on the same unified
memory, otherwise a host-write / kernel race (cf. CHOICES.md). **Validation.** The seam is exercised
transversally by the whole suite; specifically the MPI tests of section 20
(`test_mpi_fillboundary`, `test_mpi_poisson`, `test_mpi_array_reduce`, np=1/2/4 bit-identical) and the
GH200 device validations (GPU_RUNTIME_PORT.md) which confirm that the Kokkos Serial, OpenMP and
Cuda spaces give the same results (up to the FP choice of the Kokkos sum, documented).


---

## 25. Capabilities to qualify (present but limited, or off master)

What exists with a restricted scope, or what is written/designed without being on `master` as of the date
of this page. The goal is not to present a partial capability as complete.

- Runtime HLL flux. The `HLLFlux` brick exists in the core (section 2). The runtime exposure
  `riemann="hll"` for a 3-variable model (wave speeds without pressure) is on a branch
  (PR #239), not yet on `master`.
- GaussPolicy restart/evolve. An experimental policy (re-imposing Gauss at each step, or keeping the
  `phi` evolved by Schur) on a branch (PR #237); the associated experiment is discarded. Not on
  `master`.
- Global Schur on AMR. The Schur-condensed source step (section 13) has no AMR counterpart in
  production: a design exists (PR #232), the implementation does not. On AMR, the Poisson is solved at the coarse
  level then injected toward the fine, without a composite multi-level elliptic solve.
- Distributed FFT under System. `DistributedFFTSolver` (section 10) exists and is tested separately, but
  `System` under MPI np > 1 refuses the FFT cleanly (no automatic routing); use the
  geometric multigrid.
- Polar Poisson. `PolarPoissonSolver` (FFT in theta, Thomas in r, section 16) is mono-rank and
  mono-box. The polar tensor/Krylov path (polar Schur) lifts this limit on its perimeter.
- Cut-cell and Hoffart fidelity. The cut-cell (sections 14, 15) is a numerical capability of the core; it
  is not presented as a proven correction of the growth rates of the Hoffart benchmark (cf.
  [HOFFART_FIDELITY.md](HOFFART_FIDELITY.md)).
- Energy under Schur. The Schur step (section 13) adjusts the kinetic energy if an `Energy` role is
  declared; the isothermal case does not use the energy equation.

---

## Which scheme or solver when

| Probleme | Choix | Pourquoi |
|---|---|---|
| general hyperbolic transport | Godunov finite volumes + Rusanov flux | robust, works for any equation (section 1, 2) |
| compressible Euler with shocks | primitive reconstruction + HLLC or Roe | resolves the contact, less diffusive than Rusanov (section 2, 3) |
| smooth zones, high precision | WENO5-Z + SSPRK3 | order 5, low dissipation (section 3, 4) |
| stiff source (Lorentz, relaxation) | local IMEX, or global Schur condensation | implicit, no exploding time step (section 5, 13) |
| periodic Poisson, $n = 2^k$ | `poisson_fft_solver` | direct, $O(N \log N)$ (section 10) |
| Poisson with wall, Dirichlet, or $\varepsilon(x)$ | `geometric_mg` | multigrid, arbitrary geometry (section 9, 11) |
| full-tensor operator (anisotropic, polar) | `krylov_solver` (matrix-free BiCGStab) | no matrix assembly (section 12, 16) |
| localized feature (front, ring) | `AmrSystem` + `set_refinement` | adaptive refinement, conservative reflux (section 17 to 19) |
| inter-species sources | `CoupledSource` (bytecode) | conservative by construction (section 22) |
| non-rectangular domain | EB cut-cell (disc) or polar ring | curved boundary without staircase (section 14 to 16) |

## References

- Finite volumes and Riemann fluxes: LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge, 2002. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009
  (HLLC, Roe).
- WENO reconstruction: Jiang & Shu, *Efficient Implementation of Weighted ENO Schemes*, JCP 126
  (1996). WENO-Z: Borges et al., JCP 227 (2008).
- SSP integration: Gottlieb, Shu, Tadmor, *Strong Stability-Preserving Time Discretization Methods*,
  SIAM Review 43 (2001).
- IMEX: Ascher, Ruuth, Spiteri, *Implicit-explicit Runge-Kutta methods*, Appl. Numer. Math. 25 (1997).
- Multigrid: Briggs, Henson, McCormick, *A Multigrid Tutorial*, SIAM, 2000.
- Krylov: Saad, *Iterative Methods for Sparse Linear Systems*, SIAM, 2003 (BiCGStab, section 7).
- AMR: Berger & Oliger, *Adaptive mesh refinement for hyperbolic partial differential equations*, JCP
  53 (1984). Berger & Colella, *Local adaptive mesh refinement for shock hydrodynamics*, JCP 82 (1989).
  Berger & Rigoutsos, *An algorithm for point clustering and grid generation*, IEEE Trans. SMC 21 (1991).
- Cut-cell: Shortley & Weller, *The numerical solution of Laplace's equation*, J. Appl. Phys. 9 (1938).
- Condensed Schur (magnetized Euler-Poisson): see [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) and
  [HOFFART_FIDELITY.md](HOFFART_FIDELITY.md).

