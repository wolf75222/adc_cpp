"""Scheme bricks : inter-species couplings, spatial scheme, explicit time (Spec-4 PR-F).

Inter-species couplings (Ionization / Collision / ThermalExchange), the spatial discretization
(Spatial / FiniteVolume) and the plain ``Explicit`` time treatment. The implicit / split time
policies (IMEX / SourceImplicit / IMEXRK / Implicit / Role / CondensedSchur / Split / Strang)
live in ``_bricks_time`` (split out for the 500-line cap). ``pops.runtime.bricks`` re-exports
these together with the model bricks in ``_bricks_model`` and the time policies in ``_bricks_time``.
"""


# --- Inter-species couplings (operator-split): objects passed to sim.add_coupling ---
class Ionization:
    """Ionization n_g -> n_i + n_e (rate k n_e n_g). Mass transferred from the neutral to the ion."""

    def __init__(self, electron, ion, neutral, rate):
        self.electron = electron
        self.ion = ion
        self.neutral = neutral
        self.rate = rate


class Collision:
    """Inter-species friction: force k (u_a - u_b), momentum conserved. Fluid blocks (>= 3 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


class ThermalExchange:
    """Thermal exchange k (T_a - T_b), energy conserved. Euler blocks (4 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


# --- Spatial scheme + time treatment (per block) ------------------------
class Spatial:
    """Spatial discretization: reconstruction (limiter) + numerical Riemann flux.

    - ``limiter``: "none" | "minmod" | "vanleer" | "weno5" (shortcuts none=/minmod=/vanleer=/weno5=).
      weno5 = WENO5-Z, order 5 in smooth regions, 5-point stencil (3 ghosts), oscillation-free
      capture near a front; only the native ``add_block`` path exposes it (the compiled .so paths
      allocate 2 ghosts -> explicit rejection).
    - ``flux``: "rusanov" | "hll" | "hllc" | "roe".
      rusanov = minimal generic (requires only max_wave_speed, any model).
      hll = generic with signed waves (requires model.wave_speeds: native isothermal/compressible model,
      or a DSL model declaring a primitive 'p'); less diffusive than rusanov, without requiring a
      pressure or n_vars == 4. This is the recommended path for a NON Euler model with signed waves
      (moment system, isothermal): ``hll`` + ``minmod``.
      hllc / roe = contact-resolving (HLLC) and Roe-linearized solvers. Canonical path is 2D Euler
      (4 variables rho/rho_u/rho_v/E + ideal-gas pressure); they are also generic when the model
      supplies the hooks HasHLLCStructure / HasRoeDissipation (DSL m.enable_hllc()/m.enable_roe()),
      with EulerHLLCFlux2D / EulerRoeFlux2D naming the Euler fallback on the C++ side.
    - ``recon``: "conservative" | "primitive" (reconstructed variables; primitive more robust
      for Euler: positivity of rho and p; shortcut primitive=).
    - ``positivity_floor``: DENSITY floor of the reconstructed face states (positivity limiter
      Zhang-Shu, ADC-76): conservative scaling of the face state toward the cell mean
      so that rho_face >= floor. 0/None (default) = inactive, bit-identical path.
      Motivated by the top-hat jump of contrast 1e6 in the Hoffart diocotron, where WENO5 reconstructs a
      negative density -> NaN. Requires a model exposing the Density role.
    - ``wave_speed_cache``: flux='hll' + explicit time ONLY. Pre-computes model.wave_speeds ONCE per
      cell and direction (instead of per face) then bounds each face by min/max of the two neighbor
      cells. Net gain when wave_speeds is expensive (moment hierarchy). With limiter='none' +
      recon='conservative' it is BIT-IDENTICAL to the per-face path; with a 2nd-order+ limiter it is a
      Davis bound on the cell values (different result, opt-in assumed). False (default) = per-face path
      unchanged. Wired on the FULL cartesian advance only: refused if flux != 'hll', IMEX time, polar
      geometry, or a staircase/cutcell disc transport mode is active (set_disc_domain / set_geometry_mode).
    """

    def __init__(self, limiter="minmod", flux="rusanov", recon="conservative", *, none=False,
                 minmod=False, vanleer=False, weno5=False, primitive=False,
                 positivity_floor=None, wave_speed_cache=False):
        if none:
            limiter = "none"
        elif minmod:
            limiter = "minmod"
        elif vanleer:
            limiter = "vanleer"
        elif weno5:
            limiter = "weno5"
        if primitive:
            recon = "primitive"
        self.limiter = limiter
        self.flux = flux
        self.recon = recon
        pf = 0.0 if positivity_floor is None else float(positivity_floor)
        if not (pf >= 0.0):
            raise ValueError("Spatial: positivity_floor >= 0 (0/None = inactive; received %r)"
                             % (positivity_floor,))
        self.positivity_floor = pf
        self.wave_speed_cache = bool(wave_speed_cache)


def FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative",
                 positivity_floor=None, wave_speed_cache=False):
    """Finite-volume scheme (stable surface Phase A): remaps onto the existing Spatial object.

    The NUMERICAL Riemann flux is named ``riemann`` (NOT ``flux``, reserved for the PHYSICAL flux of the
    DSL model m.flux) so the two meanings do not collide. Argument mapping:

    - ``limiter`` -> Spatial.limiter ("none" | "minmod" | "vanleer" | "weno5")
    - ``riemann`` -> Spatial.flux ("rusanov" | "hll" | "hllc" | "roe"); "hll" is the generic
      signed-wave path (requires model.wave_speeds), "hllc"/"roe" run on the canonical Euler 2D
      layout or generically via the model hooks HasHLLCStructure / HasRoeDissipation
    - ``variables`` -> Spatial.recon ("conservative" | "primitive")

    cf. docs/DSL_MODEL_DESIGN.md section 6. Returns a Spatial (consumed as-is by add_block /
    add_equation). pops.Spatial stays available identically. ``positivity_floor`` (ADC-76):
    density floor of the face states (Zhang-Shu limiter), None/0 = inactive.
    ``wave_speed_cache``: HLL wave speed cache (riemann='hll' + explicit), cf. Spatial."""
    return Spatial(limiter=limiter, flux=riemann, recon=variables,
                   positivity_floor=positivity_floor, wave_speed_cache=wave_speed_cache)


class Explicit:
    """Explicit time treatment.

    substeps=N: the block advances N times per macro-step, each substep of length dt/N
                 (fast electrons: substeps=10). Default 1 = historical behavior.
    stride=M   : block cadence, HOLD-THEN-CATCH-UP semantics (catch-up at the END of the window).
                 The block is HELD (not advanced) while (macro_step + 1) % M != 0, then advances by an
                 effective step M*dt at the macro-step where (macro_step + 1) % M == 0, i.e. at the end of each
                 window of M macro-steps (slow block, e.g. neutrals: stride=20). It thus stays
                 temporally CONSISTENT with the fast blocks (never advanced "into the future"). Default
                 1 = every macro-step, bit-identical to the historical behavior. substeps and stride are ORTHOGONAL:
                 stride=M, substeps=N -> N substeps of M*dt/N once at the end of the window.
                 POISSON COUPLING: between two catch-ups, the held block contributes to the right-hand side of the
                 system Poisson (and to the coupled sources) with its STALE state -- its last advanced
                 density/charge, frozen until the next catch-up. step_cfl honors the cadence: the stable
                 step includes the stride factor (dt <= cfl*h*substeps / (stride*w)).
                 NB: the 'aot' backend (System.add_equation on a CompiledModel backend='aot') does NOT
                 carry the cadence and REJECTS stride > 1 (explicit path, no silent ignore);
                 add_block (native) and backend='production' support the stride.
    method     : "ssprk2" (default, Shu-Osher 2-stage order 2) | "ssprk3" (3-stage order 3,
                 less dissipative, to pair with weno5) | "euler" (ForwardEuler, order 1: fidelity
                 to first-order references, validation only). Shortcut ssprk3=True.
    """

    def __init__(self, substeps=1, method="ssprk2", stride=1, *, ssprk3=False):
        if ssprk3:
            method = "ssprk3"
        if method not in ("ssprk2", "ssprk3", "euler"):
            raise ValueError("Explicit: method 'ssprk2' | 'ssprk3' | 'euler' (received %r)" % (method,))
        if int(substeps) < 1:
            raise ValueError("Explicit: substeps >= 1 (received %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("Explicit: stride >= 1 (received %r)" % (stride,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.method = method
        # kind passed to the compiled facade: "explicit" (SSPRK2, bit-identical default), "ssprk3"
        # or "euler" (order 1, fidelity to first-order references -- validation, never default).
        self.kind = method if method in ("ssprk3", "euler") else "explicit"
