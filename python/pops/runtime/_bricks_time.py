"""Time policies bricks : implicit / split temporal treatments (Spec-4 PR-F).

The per-block time treatments beyond plain ``Explicit``: ``IMEX`` / ``SourceImplicit`` /
``SourceImplicitBE`` / ``IMEXRK`` / the deprecated ``Implicit`` shim, the physical ``Role`` enum,
and the Schur-condensed source stage + splitting policies (``CondensedSchur`` / ``Split`` /
``Strang``), plus the mask normalization helpers. Split out of ``_bricks_scheme`` to keep each
module under the 500-line cap ; ``pops.runtime.bricks`` re-exports all of them. ``Explicit`` (the
``Split`` transport stage) is imported from ``_bricks_scheme``.
"""

from pops.runtime._bricks_scheme import Explicit


def _role_to_stable(name):
    """Normalize a role name to the STABLE key expected by the C++ (role_from_name): lowercase
    snake_case ("momentum_x", "energy"). Tolerates the PascalCase variants of the C++ enum exposed in
    the target API (e.g. "MomentumX" -> "momentum_x", "Energy" -> "energy") by inserting a '_' before each
    internal uppercase letter before lowercasing. A name already in snake_case ("momentum_x") is unchanged."""
    s = str(name).strip()
    if not s:
        return s
    if s == s.lower():  # already snake_case / lowercase: unchanged
        return s
    out = [s[0].lower()]
    for ch in s[1:]:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _norm_implicit(label, implicit_vars, implicit_roles):
    """Normalize the implicit-mask lists (names / physical roles) into lists of strings.

    None -> [] (default: inactive mask, model default, bit-identical). A bare string is tolerated
    (e.g. implicit_vars="rho_u" -> ["rho_u"]). The roles are reduced to the STABLE C++ key (snake_case)
    via _role_to_stable -> "MomentumX" and "momentum_x" are equivalent. The mask lives on the TEMPORAL
    POLICY / block side (and NOT the model): the SAME model is reused with distinct implicit treatments.
    The RESOLUTION of names/roles -> indices and the validation (name/role absent from the block) lives
    on the C++ side (System::add_block), the only source of truth for the block names/roles."""
    def as_list(x, what):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        try:
            out = [str(v) for v in x]
        except TypeError:
            raise ValueError("%s: %s must be a list of strings (received %r)" % (label, what, x))
        return out
    names = as_list(implicit_vars, "implicit_vars")
    roles = [_role_to_stable(r) for r in as_list(implicit_roles, "implicit_roles")]
    return names, roles


class IMEX:
    """IMEX: explicit transport (SSPRK) + stiff implicit source (backward-Euler, local Newton).

    PARTIAL treatment: only the SOURCE is implicit (backward-Euler, local cell Newton,
    via backward_euler_source / ImplicitSourceStepper on the C++ side). The TRANSPORT stays explicit
    (advanced by the SSPRK core). This is NOT a global implicit solver (flux + source + Poisson
    solved implicitly / Newton-Krylov) -- that work is a distinct future phase.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit): the block is held
      while (macro_step + 1) % M != 0, then advances by an effective step M*dt at the end of the window. Between
      two catch-ups, its STALE state contributes to the system Poisson. Default 1 = every macro-step,
      bit-identical. Backend 'aot': stride > 1 rejected (cf. Explicit).
    - ``implicit_vars``: names of the conserved variables to treat IMPLICITLY in the source step;
      the others stay explicit (forward Euler). The mask is CARRIED BY THIS POLICY / the block,
      NOT by the model -> the SAME model is reused with different implicit treatments.
      Default [] (+ implicit_roles []) = model default (Model::is_implicit, or all implicit by
      default), BIT-IDENTICAL. Resolved on the C++ side against the block names (an absent name raises an error).
      E.g. pops.IMEX(implicit_vars=["rho_u", "rho_v"]).
    - ``implicit_roles``: same mask but by physical ROLE ("density", "momentum_x", "energy", ...)
      instead of the name (cf. System.variable_roles). Union with implicit_vars. E.g.
      pops.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"]).
    - ``newton_max_iters``: iteration budget of the local Newton (default 2 = historical constant).
    - ``newton_rel_tol`` / ``newton_abs_tol``: per-cell stopping criterion
      ||F||_inf <= abs_tol + rel_tol*||F0||_inf (0/0 = disabled, bit-identical historical loop).
    - ``newton_fd_eps``: step of the finite-difference Jacobian (default 1e-7 = historical).
    - ``newton_diagnostics``: enables the Newton report (sim.newton_report(name) -> dict
      {enabled, converged, max_residual, max_iters_used, n_failed}), aggregated over the last advance
      of the block. OPT-IN: default False = zero extra cost.

    NOMENCLATURE (audit 2026-06): the wired scheme is exactly ForwardEuler(transport without
    source) + local backward-Euler on the source ("SourceImplicitBE"). It is NOT an
    IMEX-RK / ARK family (no choice of Butcher tableau, ``method=`` of the explicit does not apply
    to the IMEX half-step); a true IMEXRK family would be a distinct future work.
    """

    kind = "imex"
    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("IMEX: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEX: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEX: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEX: newton_damping in (0, 1] (got %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEX: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit("IMEX", implicit_vars, implicit_roles)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


class SourceImplicit:
    """Implicit treatment of the STIFF SOURCE (backward-Euler, local Newton), explicit transport.

    Clear name for the source-only IMEX scheme: only the SOURCE is treated implicitly
    (backward-Euler solved by local per-cell Newton, via backward_euler_source /
    ImplicitSourceStepper on the C++ side). TRANSPORT stays EXPLICIT (advanced by the SSPRK core).

    IMPORTANT -- this is NOT a global implicit PDE solver. A global implicit solver
    (flux + source + Poisson all implicit, Newton-Krylov or global Schur) is a distinct
    future effort. SourceImplicit = source-only IMEX (strictly equivalent to IMEX/pops.Implicit,
    bit-identical numerics).

    WHEN TO USE IT (SourceImplicit LOCAL vs pops.CondensedSchur GLOBAL) -- both mechanisms
    treat a stiff source implicitly, but at different scales:

    - SourceImplicit is LOCAL: the implicit part couples only the components of A SINGLE CELL
      (backward-Euler solved by per-cell Newton), there is NO spatial coupling between
      cells. Suited to purely local stiff terms (relaxation, reactions, friction).
    - pops.CondensedSchur (via pops.Split) is GLOBAL: it assembles and solves a tensor
      elliptic operator by Schur (Krylov BiCGStab) that COUPLES the whole domain. Suited to
      non-local stiff Lorentz / electrostatic coupling (e.g. magnetized Euler-Poisson from the
      Hoffart paper, arXiv:2510.11808). A local stiff source does NOT need Schur.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit). Default 1.
    - ``implicit_vars`` / ``implicit_roles``: implicit mask by NAME or by physical ROLE of the
      conserved variables to treat implicitly in the source step (cf. IMEX). Mask CARRIED BY
      THIS POLICY / the block, not by the model. Defaults [] = model default, bit-identical.
    """

    kind = "imex"  # same C++ path as IMEX (ImplicitSourceStepper)

    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("SourceImplicit: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("SourceImplicit: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("SourceImplicit: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("SourceImplicit: newton_damping in (0, 1] (got %r)"
                             % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("SourceImplicit: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit(
            "SourceImplicit", implicit_vars, implicit_roles)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


# PRECISE name of the scheme wired by IMEX / SourceImplicit (audit 2026-06): ForwardEuler transport
# without source + LOCAL backward-Euler on the source (per-cell Newton). STRICT alias of
# SourceImplicit (same object): to use when you want to name the hypothesis in a script.
SourceImplicitBE = SourceImplicit


class IMEXRK:
    """IMEX-RK family (Implicit-Explicit Runge-Kutta), ARS(2,2,2) scheme, ORDER 2.

    Ascher-Ruuth-Spiteri scheme (1997): the hyperbolic transport L = -div F is treated by the
    EXPLICIT tableau, the stiff source S by the IMPLICIT tableau (LOCAL per-cell backward-Euler,
    Newton, like pops.IMEX) -- but with coupled stages that raise the GLOBAL ORDER TO 2 (transport
    AND source), whereas pops.IMEX stays a ForwardEuler(transport) + backward-Euler(source) of order 1.

    Coefficients: gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma). Tableaus (stiffly accurate):
    explicit A_E = [[0,0,0],[gamma,0,0],[delta,1-delta,0]], b_E = [delta,1-delta,0];
    implicit A_I = [[0,0,0],[0,gamma,0],[0,1-gamma,gamma]], b_I = [0,1-gamma,gamma].

    DISTINCT FAMILY from pops.IMEX (kind="imexrk_ars222" != "imex"): the pops.IMEX default (local
    backward-Euler, order 1) is UNCHANGED / bit-identical. SCOPE: CARTESIAN System only -- AMR, the
    polar grid, compiled models (.so: prototype/aot/production) and the Strang/Schur splittings
    REJECT it explicitly (use pops.IMEX / pops.Explicit on those paths).

    - ``scheme``: "ars222" (only wired scheme; another name raises an explicit error).
    - ``substeps=N``: substeps per macro-step (cf. pops.Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. pops.Explicit). Default 1.
    - ``newton_*``: SAME options as pops.IMEX (max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy/
      diagnostics) -- they parametrize BOTH implicit stage solves of the scheme. Defaults =
      historical constants (max_iters=2, fd_eps=1e-7), without extra cost.

    FULLY IMPLICIT SOURCE: unlike pops.IMEX, IMEXRK does NOT expose implicit_vars /
    implicit_roles (the ARS(2,2,2) stage-consistency relation assumes a homogeneous solve). A partial
    mask is rejected on the C++ side; for a partial per-component IMEX, use pops.IMEX.
    """

    kind = "imexrk_ars222"

    def __init__(self, scheme="ars222", substeps=1, stride=1,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if scheme != "ars222":
            raise ValueError("IMEXRK: scheme 'ars222' (only wired IMEX-RK scheme; got %r)"
                             % (scheme,))
        if int(substeps) < 1:
            raise ValueError("IMEXRK: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEXRK: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEXRK: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEXRK: newton_damping in (0, 1] (got %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEXRK: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.scheme = str(scheme)
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


def Implicit(dt_ratio=1, substeps=None, stride=1):
    """DEPRECATED -- use pops.SourceImplicit(...) or pops.IMEX(...) instead.

    pops.Implicit was an alias of IMEX (implicit stiff source via backward-Euler, explicit
    transport). The name "Implicit" is MISLEADING: it suggests a global implicit PDE solver
    (flux + source + Poisson all implicit / Newton-Krylov), which is NOT the case.
    pops.SourceImplicit is the clear name of the same scheme (bit-identical numerics).

    Kept for backward compatibility; emits a DeprecationWarning. Use:
      pops.SourceImplicit(substeps=k, stride=s)  -- new clear name
      pops.IMEX(substeps=k, stride=s)            -- official acronym
    """
    import warnings
    warnings.warn(
        "pops.Implicit is deprecated: the name is misleading (it is NOT a global implicit "
        "PDE solver). Use pops.SourceImplicit(...) (implicit backward-Euler source, "
        "explicit transport) or pops.IMEX(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return IMEX(substeps=substeps if substeps is not None else dt_ratio, stride=stride)


class Role:
    """PHYSICAL roles of a model's components (cf. VariableRole on the C++ side / variable_roles).

    Lets you address a component by its MEANING in pops.CondensedSchur(density=pops.Role.Density,
    momentum=(pops.Role.MomentumX, pops.Role.MomentumY), energy=pops.Role.Energy) rather than by a literal
    name. The values are the STABLE keys expected by the C++ (role_from_name: snake_case). The
    role -> component RESOLUTION is done on the C++ side (the block reads its own VariableRole): these
    constants serve to EXPRESS the intent in the formula and to validate that a required role is requested.
    """

    Density = "density"
    MomentumX = "momentum_x"
    MomentumY = "momentum_y"
    MomentumZ = "momentum_z"
    Energy = "energy"
    VelocityX = "velocity_x"
    VelocityY = "velocity_y"
    VelocityZ = "velocity_z"
    Pressure = "pressure"
    Temperature = "temperature"
    Scalar = "scalar"


class CondensedSchur:
    """SOURCE stage condensed by Schur (Hoffart et al., arXiv:2510.11808; cf.
    docs/SCHUR_CONDENSATION_DESIGN.md). NAMES the algorithm of the implicit source coupling potential /
    velocity / Lorentz and MAPS the fields onto the block's physical roles. This is the `source=` of an
    pops.Split temporal policy (EXPLICIT / IMPLICIT splitting).

    kind="electrostatic_lorentz" (only one for now) selects ElectrostaticLorentzCondensation:
    the stage assembles the condensed elliptic operator A = I + theta^2 dt^2 alpha rho B^{-1}, solves it
    (MG-preconditioned BiCGStab), reconstructs the velocity v = B^{-1}(v^n - theta dt grad phi) and extrapolates
    to the full step. Everything is in C++ (CondensedSchurSourceStepper, #126): NO per-cell Python callback.

    The block must expose the Density / MomentumX / MomentumY roles (Energy optional) and a B_z field
    (set_magnetic_field) -- a missing role / B_z raises an EXPLICIT error at add_equation. Works for
    a native-brick model as well as for a compiled DSL model that declares these roles (electrons).

    GEOMETRY: wired in CARTESIAN (System(mesh=pops.CartesianMesh(...))) AND in POLAR
    (System(mesh=pops.PolarMesh(...)), ring (r, theta), Track A step 2c). The choice of the condensed stepper
    (cartesian CondensedSchurSourceStepper / polar PolarCondensedSchurSourceStepper) is made on the C++ side
    according to the System geometry: the SAME pops.CondensedSchur(...) is used in both cases. The
    polar counterpart is MULTI-RANK-SAFE (correct collectives under MPI) but the facade still builds
    ONE global box (on the owner rank): correct and bit-identical to single-rank, without
    effective parallelism at this level -- the facade theta decomposition is a dedicated follow-up (update
    audit 2026-06; the old mention "n_ranks>1 raises" was stale).

    WHEN TO USE IT (CondensedSchur GLOBAL vs pops.SourceImplicit LOCAL). CondensedSchur is a
    GLOBAL implicit: it COUPLES the whole domain via the condensed tensor elliptic operator
    (solved by Krylov BiCGStab), for non-local stiff Lorentz / electrostatic coupling. If the
    stiff source is purely LOCAL (couples only the components of a single cell, without spatial
    coupling: relaxation, reactions, friction), prefer pops.SourceImplicit instead: it is cheaper
    and there is then NO elliptic solve to do.

    - ``theta``: theta-scheme in (0, 1] (0.5 = Crank-Nicolson, 1 = backward Euler).
    - ``alpha``: electrostatic coupling constant of the source subsystem
      (d_t(-Lap phi) = -alpha div(rho v)).
    - ``density`` / ``momentum`` / ``energy`` / ``magnetic_field`` / ``potential``: role / field
      descriptors. They EXPRESS the intent; the role -> component resolution is done on the C++ side
      (the block reads its own VariableRole). They accept pops.Role.* (recommended), a stable role name,
      or a variable name of the block. momentum is a pair (x, y).
    - ``krylov_tol`` / ``krylov_max_iters``: tolerance and budget of the stage's Krylov (BiCGStab)
      solve. None (defaults) = historical constants (1e-10; 400 in cartesian, 600 in polar),
      made configurable by the 2026-06 audit (explicit numerical constants).
    """

    def __init__(self, kind="electrostatic_lorentz", theta=0.5, alpha=1.0,
                 density=Role.Density, momentum=(Role.MomentumX, Role.MomentumY),
                 energy=None, magnetic_field="B_z", potential="phi",
                 krylov_tol=None, krylov_max_iters=None):
        self.krylov_tol = float(krylov_tol) if krylov_tol is not None else 0.0
        self.krylov_max_iters = int(krylov_max_iters) if krylov_max_iters is not None else 0
        if krylov_tol is not None and not (0.0 < self.krylov_tol < 1.0):
            raise ValueError("CondensedSchur: krylov_tol must be in (0, 1) (got %r)" % (krylov_tol,))
        if krylov_max_iters is not None and self.krylov_max_iters < 1:
            raise ValueError("CondensedSchur: krylov_max_iters >= 1 (got %r)" % (krylov_max_iters,))
        if kind != "electrostatic_lorentz":
            raise ValueError(
                "CondensedSchur: kind 'electrostatic_lorentz' (only one supported); got %r" % (kind,))
        if not (0.0 < float(theta) <= 1.0):
            raise ValueError("CondensedSchur: theta must be in (0, 1] (got %r)" % (theta,))
        # momentum must be a pair (role_x, role_y); a bare string (iterable of characters)
        # is rejected explicitly (otherwise tuple("xy") would give two components by accident).
        if isinstance(momentum, str):
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y), not a string (got %r)"
                % (momentum,))
        try:
            mom = tuple(momentum)
        except TypeError:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        if len(mom) != 2:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        # Role / field descriptors CARRIED in the C++ ABI (audit wave 2): density /
        # momentum / energy accept an pops.Role.* (stable role name) OR a variable name of the
        # block; the role-or-name -> component resolution is done on the C++ side (set_source_stage,
        # explicit error if not found). The DEFAULTS (canonical roles) keep the bit-identical
        # historical behavior. magnetic_field accepts a canonical aux field name
        # (AUX_CANONICAL: "B_z", "T_e", ...) -> carried aux component. potential stays fixed
        # to "phi" (the stage uses the system Poisson potential; another field would have
        # no solver behind it -> explicit rejection, no silent ignore).
        def _spec(v):
            return "" if v is None else str(v)
        # Canonical defaults -> EMPTY strings on the ABI side (the C++ then resolves the canonical
        # roles, historical path strictly unchanged).
        self.density_spec = "" if density == Role.Density else _spec(density)
        self.momentum_x_spec = "" if mom[0] == Role.MomentumX else _spec(mom[0])
        self.momentum_y_spec = "" if mom[1] == Role.MomentumY else _spec(mom[1])
        if energy is None:
            self.energy_spec = ""
        elif energy == Role.Energy:
            self.energy_spec = ""
        else:
            self.energy_spec = _spec(energy)
        if magnetic_field == "B_z":
            self.bz_aux_component = -1  # canonical channel (default, bit-identical)
        else:
            from pops.physics.aux import AUX_CANONICAL
            if magnetic_field not in AUX_CANONICAL:
                raise ValueError(
                    "CondensedSchur: magnetic_field=%r unknown (canonical aux fields: %s)"
                    % (magnetic_field, sorted(AUX_CANONICAL)))
            self.bz_aux_component = int(AUX_CANONICAL[magnetic_field])
        if potential != "phi":
            raise ValueError(
                "CondensedSchur: potential=%r not configurable (the source stage solves the "
                "system Poisson potential phi; another field would have no solver "
                "behind it); leave potential='phi' (default)." % (potential,))
        self.kind = kind
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.density = density
        self.momentum = mom
        self.energy = energy
        self.magnetic_field = magnetic_field
        self.potential = potential
    def _has_field_overrides(self):
        """True if a non-canonical descriptor is requested (AMR: explicit rejection, not wired)."""
        return bool(self.density_spec or self.momentum_x_spec or self.momentum_y_spec
                    or self.energy_spec or self.bz_aux_component >= 0)


# The typed constructor ElectrostaticLorentzSchur(...) for the (currently unique) CondensedSchur
# kind lives in _bricks_typed (Spec 5 sec.14.2.5 typed native-brick constructors), beside the typed
# native boundary bricks, to keep this module under the 500-line cap. pops.runtime.bricks re-exports
# it next to CondensedSchur.


class Split:
    """Temporal policy EXPLICIT / IMPLICIT SPLITTING: an EXPLICIT hyperbolic transport stage
    (pops.Explicit, SSPRK) followed by a separate SOURCE stage (cf. docs/SCHUR_CONDENSATION_DESIGN.md
    section 6). This is the OPT-IN of the Schur work: a block that does NOT use pops.Split keeps the
    default path (Explicit / IMEX / SourceImplicit), BIT-IDENTICAL.

    ::

        time=pops.Split(
            hyperbolic=pops.Explicit(ssprk3=True),
            source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, ...),
        )

    - ``hyperbolic`` : pops.Explicit (the transport; SSPRK2/3, substeps, stride inherit from it).
    - ``source`` : pops.CondensedSchur (the condensed source stage, runs AFTER the transport). Only
      source backend wired for now.
    """

    # kind="explicit": the transport is run by the core explicit path (SSPRK), the condensed source
    # is plugged IN ADDITION via set_source_stage (cf. System.add_equation). The block is therefore
    # NOT IMEX (the local stiff source backward-Euler): its source is the condensed stage, apart.
    def __init__(self, hyperbolic=None, source=None):
        hyperbolic = hyperbolic if hyperbolic is not None else Explicit()
        if not isinstance(hyperbolic, Explicit):
            raise TypeError(
                "Split: hyperbolic must be an pops.Explicit (explicit SSPRK transport); got %r"
                % type(hyperbolic).__name__)
        if source is None:
            raise ValueError(
                "Split: source= is required (the separate source stage); e.g. "
                "pops.Split(hyperbolic=pops.Explicit(), source=pops.CondensedSchur(...))")
        if not isinstance(source, CondensedSchur):
            raise TypeError(
                "Split: source must be an pops.CondensedSchur(...) (only wired source stage); got %r"
                % type(source).__name__)
        self.hyperbolic = hyperbolic
        self.source = source
        # The transport takes the core explicit path: we relay the kind / substeps / stride of
        # the hyperbolic stage (SSPRK2/3). The condensed source is plugged separately (add_equation).
        self.kind = hyperbolic.kind
        self.method = hyperbolic.method
        self.substeps = hyperbolic.substeps
        self.stride = hyperbolic.stride
        # Splitting policy WIRED to the system stepper (set_time_scheme). pops.Split = "lie"
        # (Godunov, 1st order): H(dt) then S(dt) once per macro-step, BIT-IDENTICAL to the history.
        # pops.Strang overrides this attribute to "strang" (cf. below).
        self.scheme = "lie"


class Strang(Split):
    """Temporal policy STRANG SPLITTING (symmetric, 2nd order): one macro-step runs
    H(dt/2); S(dt); H(dt/2), where H is the EXPLICIT hyperbolic transport (pops.Explicit, SSPRK)
    and S the separate SOURCE stage (pops.CondensedSchur). This is the 2nd-order extension of pops.Split
    (Lie / Godunov, 1st order): same bricks (SSPRK transport + condensed source stage), only the ORDER
    and the cadence of field solves change.

    ::

        time=pops.Strang(
            hyperbolic=pops.Explicit(ssprk3=True),
            source=pops.CondensedSchur(theta=0.5, alpha=alpha),
        )

    The system stepper RE-SOLVES solve_fields BETWEEN stages (before each half-advance and before
    the source) so that the transport always reads a phi consistent with the current density (the
    SINGLE leading solve_fields, sufficient for Lie or a single transport advance to follow, does not
    suffice for the 2nd Strang half-advance). cf. docs/HOFFART_STEP_SEQUENCE.md and SystemStepper::step_strang.

    ``hyperbolic`` / ``source`` : identical to pops.Split. Wired by add_equation (which plugs the source
    stage AND calls set_time_scheme('strang') on the System)."""

    def __init__(self, hyperbolic=None, source=None):
        super().__init__(hyperbolic=hyperbolic, source=source)
        self.scheme = "strang"
