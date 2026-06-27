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
# Spec 5 sec.7: the spatial scheme is chosen with TYPED descriptors, never bare strings. The
# tables below map each accepted descriptor scheme to the canonical token the C++ ABI consumes,
# and name the typed alternative a rejected string should point at (reject_string_selector).
# The descriptor category gates which slot a descriptor may fill (a riemann flux in the limiter
# slot is a clear error, not a silent swap).
_LIMITER_SCHEMES = {  # reconstruction / limiter descriptor scheme -> Spatial.limiter token
    "none": "none", "firstorder": "none",
    "minmod": "minmod", "vanleer": "vanleer",
    "weno5": "weno5", "weno5z": "weno5",
}
_FLUX_SCHEMES = {  # riemann descriptor scheme -> Spatial.flux token
    "rusanov": "rusanov", "hll": "hll", "hllc": "hllc", "roe": "roe", "user": "user",
}
_RECON_SCHEMES = {  # variables descriptor scheme -> Spatial.recon token
    "conservative": "conservative", "primitive": "primitive",
}
_LIMITER_SUGGEST = ("pops.numerics.reconstruction.limiters.Minmod() / .VanLeer(), "
                    "pops.numerics.reconstruction.FirstOrder() / WENO5() / MUSCL(...)")
_FLUX_SUGGEST = "pops.numerics.riemann.Rusanov() / HLL() / HLLC() / Roe()"
_RECON_SUGGEST = "pops.numerics.variables.Conservative() / Primitive()"


def _lower_selector(value, *, param, schemes, suggestion, categories):
    """Lower a typed spatial-scheme descriptor to its canonical C++ token (Spec 5 sec.7).

    @p value is a typed descriptor (``BrickDescriptor`` / ``Descriptor``) carrying ``.scheme`` and
    ``.category``. A bare ``str`` is REJECTED via :func:`reject_string_selector` -- Spec 5 forbids
    naming a scheme with a string; the message points at the typed @p suggestion. A descriptor of
    the wrong category (a Riemann flux passed for the limiter slot) is a clear ``TypeError``. An
    unknown scheme is rejected rather than silently passed to the C++ boundary.
    """
    from pops.descriptors import reject_string_selector
    if value is None:
        return None
    if isinstance(value, str):
        reject_string_selector(value, param, suggestion)  # always raises
    category = getattr(value, "category", None)
    scheme = getattr(value, "scheme", None)
    if category is None or scheme is None:
        raise TypeError(
            "Spatial: %s must be a typed pops.numerics descriptor (got %r). Use %s."
            % (param, type(value).__name__, suggestion))
    if category not in categories:
        raise TypeError(
            "Spatial: %s expects a %s descriptor, got a %r descriptor (%s). Use %s."
            % (param, " / ".join(categories), category, scheme, suggestion))
    token = schemes.get(scheme)
    if token is None:
        raise ValueError(
            "Spatial: %s descriptor scheme %r is not a known %s scheme (%s). Use %s."
            % (param, scheme, param, ", ".join(sorted(schemes)), suggestion))
    return token


class Spatial:
    """Spatial discretization: reconstruction (limiter) + numerical Riemann flux.

    Spec 5 sec.7: every scheme is chosen with a TYPED ``pops.numerics`` descriptor; a bare string is
    rejected with a message naming the typed object. The boolean shortcuts (none=/minmod=/vanleer=/
    weno5=/primitive=) stay as typed-flag sugar.

    - ``limiter`` (Spec 5 sec.14.1 alias: ``reconstruction``): a reconstruction / limiter descriptor
      lowering to "none" | "minmod" | "vanleer" | "weno5".
      ``pops.numerics.reconstruction.FirstOrder()`` -> none, ``.limiters.Minmod()`` /
      ``.VanLeer()``, ``.WENO5()`` / ``.WENO5Z()`` -> weno5, ``.MUSCL(limiter=...)`` -> its limiter.
      weno5 = WENO5-Z, order 5 in smooth regions, 5-point stencil (3 ghosts), oscillation-free
      capture near a front; only the native ``add_block`` path exposes it (the compiled .so paths
      allocate 2 ghosts -> explicit rejection).
    - ``flux``: a ``pops.numerics.riemann`` descriptor lowering to "rusanov" | "hll" | "hllc" | "roe".
      Rusanov() = minimal generic (requires only max_wave_speed, any model).
      HLL() = generic with signed waves (requires model.wave_speeds: native isothermal/compressible
      model, or a DSL model declaring a primitive 'p'); less diffusive than rusanov, without
      requiring a pressure or n_vars == 4. This is the recommended path for a NON Euler model with
      signed waves (moment system, isothermal): HLL() + Minmod().
      HLLC() / Roe() = contact-resolving (HLLC) and Roe-linearized solvers. Canonical path is 2D
      Euler (4 variables rho/rho_u/rho_v/E + ideal-gas pressure); they are also generic when the
      model supplies the hooks HasHLLCStructure / HasRoeDissipation (DSL m.enable_hllc()/
      m.enable_roe()), with EulerHLLCFlux2D / EulerRoeFlux2D naming the Euler fallback on C++.
    - ``recon``: a ``pops.numerics.variables`` descriptor lowering to "conservative" | "primitive"
      (reconstructed variables; primitive more robust for Euler: positivity of rho and p; shortcut
      primitive=).
    - ``positivity_floor``: DENSITY floor of the reconstructed face states (positivity limiter
      Zhang-Shu, ADC-76): conservative scaling of the face state toward the cell mean
      so that rho_face >= floor. 0/None (default) = inactive, bit-identical path.
      Motivated by the top-hat jump of contrast 1e6 in the Hoffart diocotron, where WENO5 reconstructs a
      negative density -> NaN. Requires a model exposing the Density role.
    - ``wave_speed_cache``: flux=HLL() + explicit time ONLY. Pre-computes model.wave_speeds ONCE per
      cell and direction (instead of per face) then bounds each face by min/max of the two neighbor
      cells. Net gain when wave_speeds is expensive (moment hierarchy). With limiter=FirstOrder() +
      recon=Conservative() it is BIT-IDENTICAL to the per-face path; with a 2nd-order+ limiter it is a
      Davis bound on the cell values (different result, opt-in assumed). False (default) = per-face path
      unchanged. Wired on the FULL cartesian advance only: refused if flux != HLL(), IMEX time, polar
      geometry, or a staircase/cutcell disc transport mode is active (set_disc_domain / set_geometry_mode).
    """

    def __init__(self, limiter=None, flux=None, recon=None, *, none=False,
                 minmod=False, vanleer=False, weno5=False, primitive=False,
                 positivity_floor=None, wave_speed_cache=False, reconstruction=None,
                 _tokens=None):
        # Spec 5 sec.14.1 names the reconstruction/limiter slot ``reconstruction=``; keep ``limiter=``
        # working and accept ``reconstruction=`` as an alias (only one of the two at a time).
        if reconstruction is not None:
            if limiter is not None:
                raise TypeError("Spatial: pass limiter= or reconstruction= (the alias), not both")
            limiter = reconstruction
        # Private fast path: _tokens = (limiter, flux, recon) already-lowered canonical strings.
        # Used by Spatial._from_tokens (the lib-descriptor lowering, whose options are strings).
        if _tokens is not None:
            lim_tok, flux_tok, recon_tok = _tokens
        else:
            lim_tok = _lower_selector(
                limiter, param="limiter", schemes=_LIMITER_SCHEMES,
                suggestion=_LIMITER_SUGGEST, categories=("reconstruction", "limiter"))
            flux_tok = _lower_selector(
                flux, param="flux", schemes=_FLUX_SCHEMES,
                suggestion=_FLUX_SUGGEST, categories=("riemann",))
            recon_tok = _lower_selector(
                recon, param="recon", schemes=_RECON_SCHEMES,
                suggestion=_RECON_SUGGEST, categories=("variables",))
        # Boolean shortcuts (typed flags, not strings): override the limiter / recon slot. They
        # stay as convenience sugar -- only the bare-string selectors are forbidden (Spec 5 sec.7).
        if none:
            lim_tok = "none"
        elif minmod:
            lim_tok = "minmod"
        elif vanleer:
            lim_tok = "vanleer"
        elif weno5:
            lim_tok = "weno5"
        if primitive:
            recon_tok = "primitive"
        # Canonical defaults (mirror the historical minmod + rusanov + conservative).
        self.limiter = lim_tok if lim_tok is not None else "minmod"
        self.flux = flux_tok if flux_tok is not None else "rusanov"
        self.recon = recon_tok if recon_tok is not None else "conservative"
        pf = 0.0 if positivity_floor is None else float(positivity_floor)
        if not (pf >= 0.0):
            raise ValueError("Spatial: positivity_floor >= 0 (0/None = inactive; received %r)"
                             % (positivity_floor,))
        self.positivity_floor = pf
        self.wave_speed_cache = bool(wave_speed_cache)

    def __str__(self):
        # Spec 5 sec.12.1: a SHORT, deterministic one-line summary of the chosen scheme (the
        # default object repr leaks a memory address, so print() was unreadable). Only the
        # non-default knobs (positivity floor, wave-speed cache) are appended, to keep the line
        # tight on the common path. __repr__ is intentionally left as the default for debug.
        body = "limiter=%s, flux=%s, recon=%s" % (self.limiter, self.flux, self.recon)
        if self.positivity_floor:
            body += ", positivity_floor=%g" % self.positivity_floor
        if self.wave_speed_cache:
            body += ", wave_speed_cache=True"
        return "Spatial(%s)" % body

    @classmethod
    def _from_tokens(cls, limiter, flux, recon, *, positivity_floor=None, wave_speed_cache=False):
        """Build a Spatial from ALREADY-canonical string tokens (internal lowering only).

        The lib-spatial descriptor (``pops.lib.spatial.FiniteVolume``) carries its scheme choice as
        string options; ``System._lower_spatial`` resolves those to the canonical tokens and calls
        this to bypass the typed-descriptor guard. Not part of the public API -- public callers pass
        typed descriptors to ``Spatial`` / ``FiniteVolume``.
        """
        return cls(_tokens=(limiter, flux, recon),
                   positivity_floor=positivity_floor, wave_speed_cache=wave_speed_cache)

    def validate(self, ghost_depth=None, block=None):
        """Reject a reconstruction whose ghost depth exceeds an EXPLICIT block halo (Spec 5 sec.7).

        The fifth-order WENO5 stencil needs a 3-cell halo; reading past a too-thin halo is a
        correctness bug (criterion 11). This checks the chosen reconstruction's DECLARED
        requirement and raises a clear, actionable error when an EXPLICIT @p ghost_depth
        constrains the block below it.

        The discipline is NO FALSE POSITIVE. The native runtime GROWS each block's halo to match
        its reconstruction (``block_n_ghost(lim)``: 3 for weno5), so WENO5 on a default block is
        a VALID problem -- and ``ghost_depth=None`` (the default) means exactly that and never
        rejects. A MUSCL / minmod / vanleer scheme (requirement <= 2) passes at any depth >= 2,
        and an undeclared reconstruction is never rejected.

        Args:
            ghost_depth: An EXPLICIT block ghost (halo) depth to check against, or ``None`` to
                defer to the scheme-matched runtime allocation (no rejection).
            block: Optional block name woven into the error message.

        Returns:
            bool: ``True`` when the reconstruction fits the (explicit or scheme-matched) depth.

        Raises:
            ValueError: When an explicit @p ghost_depth is below the reconstruction's requirement.
        """
        from pops.numerics.reconstruction import validate_ghost_depth

        available = None if ghost_depth is None else int(ghost_depth)
        return validate_ghost_depth(self.limiter, available=available, block=block)


def FiniteVolume(limiter=None, riemann=None, variables=None,
                 positivity_floor=None, wave_speed_cache=False, *, reconstruction=None,
                 none=False, minmod=False, vanleer=False, weno5=False, primitive=False):
    """Finite-volume scheme (stable surface Phase A): remaps onto the existing Spatial object.

    The NUMERICAL Riemann flux is named ``riemann`` (NOT ``flux``, reserved for the PHYSICAL flux of the
    DSL model m.flux) so the two meanings do not collide. Spec 5 sec.7: each argument is a TYPED
    ``pops.numerics`` descriptor (a bare string raises, pointing at the typed object). Argument mapping:

    - ``limiter`` (Spec 5 sec.14.1 alias: ``reconstruction``; a reconstruction / limiter descriptor) ->
      Spatial.limiter (``pops.numerics.reconstruction.FirstOrder()`` -> none, ``.limiters.Minmod()`` /
      ``.VanLeer()``, ``.WENO5()``, ``.MUSCL(limiter=...)``)
    - ``riemann`` (``pops.numerics.riemann`` descriptor) -> Spatial.flux (Rusanov()/HLL()/HLLC()/Roe());
      HLL() is the generic signed-wave path (requires model.wave_speeds), HLLC()/Roe() run on the
      canonical Euler 2D layout or generically via the model hooks HasHLLCStructure / HasRoeDissipation
    - ``variables`` (``pops.numerics.variables`` descriptor) -> Spatial.recon (Conservative()/Primitive())

    The boolean-flag shortcuts of ``Spatial`` are forwarded identically:
    ``none=/minmod=/vanleer=/weno5=`` select the limiter and ``primitive=`` selects the variable set, so
    ``FiniteVolume(minmod=True)`` and ``FiniteVolume(primitive=True)`` mean the same as on ``Spatial``.

    cf. docs/DSL_MODEL_DESIGN.md section 6. Returns a Spatial (consumed as-is by add_block /
    add_equation). pops.Spatial stays available identically. ``positivity_floor`` (ADC-76):
    density floor of the face states (Zhang-Shu limiter), None/0 = inactive.
    ``wave_speed_cache``: HLL wave speed cache (riemann=HLL() + explicit), cf. Spatial."""
    # Reject a bare string at THIS boundary so the message names the FiniteVolume parameter
    # (``riemann`` / ``variables``), not the internal Spatial slot (``flux`` / ``recon``).
    from pops.descriptors import reject_string_selector
    if isinstance(riemann, str):
        reject_string_selector(riemann, "riemann", _FLUX_SUGGEST)
    if isinstance(variables, str):
        reject_string_selector(variables, "variables", _RECON_SUGGEST)
    return Spatial(limiter=limiter, flux=riemann, recon=variables, reconstruction=reconstruction,
                   none=none, minmod=minmod, vanleer=vanleer, weno5=weno5, primitive=primitive,
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
