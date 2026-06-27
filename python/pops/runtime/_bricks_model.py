"""Model bricks : state / transport / source / elliptic value objects (Spec-4 PR-F).

The composable bricks a MODEL is built from, plus the ``Model`` composer (ModelSpec), the
HYBRID composition path (``CompositeModel`` + ``_native_to_brick``) and the elliptic physical
model (EPM) bricks/helpers. ``pops.runtime.bricks`` re-exports everything here together with the
scheme/time policies in ``_bricks_scheme``. ``ModelSpec`` comes from the loaded extension via
``pops._bootstrap``.
"""

from pops._bootstrap import ModelSpec


# --- State bricks ---------------------------------------------------------
class Scalar:
    """Scalar state (1 variable, e.g. a transported density)."""


class FluidState:
    """Fluid state. kind = "compressible" (gamma) or "isothermal" (cs2).

    vacuum_floor (isothermal only, ADC-77): quasi-vacuum density floor. When > 0 the model computes
    the velocity as u = m/max(rho, vacuum_floor), bounding the wave speed and the advective flux where
    the flow evacuates the background (rho -> ~0). It does NOT modify the conserved state (only the
    velocity estimate). 0 (default) = inactive (bit-identical). This is independent of the spatial
    positivity_floor (the Zhang-Shu reconstruction limiter): the two address different failure modes
    and must be enabled separately. Honored on the native pops.Model(...) / System / AmrSystem path;
    the compiled/DSL path (pops.CompositeModel / JIT / AOT) does not carry it yet, so set it on the
    native path.
    """

    def __init__(self, kind="compressible", gamma=1.4, cs2=0.5, vacuum_floor=0.0):
        self.kind = kind
        self.gamma = float(gamma)
        self.cs2 = float(cs2)
        if not (float(vacuum_floor) >= 0.0):
            raise ValueError("FluidState: vacuum_floor >= 0 (0 = inactive)")
        self.vacuum_floor = float(vacuum_floor)


# --- Transport bricks ---------------------------------------------------
class ExB:
    """Scalar advection by the E x B drift (magnetic field B0)."""

    def __init__(self, B0=1.0):
        self.B0 = float(B0)


class CompressibleFlux:
    """Compressible Euler flux (gamma comes from the FluidState state)."""


class IsothermalFlux:
    """Isothermal Euler flux (cs2 comes from the FluidState state)."""


# --- Source bricks ------------------------------------------------------
class NoSource:
    """No source."""


class PotentialForce:
    """Potential force (q/m) rho E on the momentum (+ work if 4 vars)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class GravityForce:
    """Gravitational force rho g (+ work if 4 vars)."""


class MagneticLorentzForce:
    """MAGNETIC Lorentz force q (v x B_z) on the momentum (native C++ brick
    pops::MagneticLorentzForce, exposed to the Python API by the 2026-06 audit).

    EXPLICIT regime (moderate omega_c): pointwise algebraic term, no work (F . v = 0, energy
    unchanged). Reads B_z from the aux channel (canonical component 3): call
    ``sim.set_magnetic_field(Bz)`` to populate it. Requires a fluid transport >= 3 variables (momentum
    on 2 axes); rejected on a scalar. The STIFF regime (large omega_c) goes through the condensed stage
    pops.CondensedSchur (Schur), NOT through this explicit brick.

    ``charge`` = q/m, sign included (same convention as PotentialForce)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class PotentialMagneticForce:
    """Electrostatic force + magnetic Lorentz SUMMED: (q/m) rho E + q (v x B_z) (native C++
    brick CompositeSource<PotentialForce, MagneticLorentzForce>, the full magnetized diocotron
    force). Same q/m for both forces (same species). Reads B_z (set_magnetic_field); requires a
    fluid transport >= 3 variables. ``charge`` = q/m, sign included."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


# --- Elliptic right-hand-side bricks ------------------------------------
class ChargeDensity:
    """Charge density f = q n."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class BackgroundDensity:
    """Neutralizing background f = alpha (n - n0)."""

    def __init__(self, alpha=1.0, n0=0.0):
        self.alpha = float(alpha)
        self.n0 = float(n0)


class GravityCoupling:
    """Self-consistent coupling f = sign 4piG (rho - rho0). sign = +1 gravity, -1 plasma."""

    def __init__(self, sign=1.0, four_pi_G=1.0, rho0=1.0):
        self.sign = float(sign)
        self.four_pi_G = float(four_pi_G)
        self.rho0 = float(rho0)


def Model(state, transport, source, elliptic):
    """Compose a model (ModelSpec) from state, transport, source, elliptic bricks.

    Validates the state <-> transport consistency (Scalar with ExB; compressible FluidState with
    CompressibleFlux; isothermal with IsothermalFlux) and carries the parameters into the spec.
    """
    spec = ModelSpec()

    if isinstance(state, Scalar):
        if not isinstance(transport, ExB):
            raise ValueError("Scalar requires transport=ExB(...)")
    elif isinstance(state, FluidState):
        if state.kind == "compressible":
            spec.gamma = state.gamma
            if not isinstance(transport, CompressibleFlux):
                raise ValueError("FluidState(compressible) requires transport=CompressibleFlux()")
        elif state.kind == "isothermal":
            spec.cs2 = state.cs2
            spec.vacuum_floor = getattr(state, "vacuum_floor", 0.0)  # ADC-77 quasi-vacuum velocity bound
            if not isinstance(transport, IsothermalFlux):
                raise ValueError("FluidState(isothermal) requires transport=IsothermalFlux()")
        else:
            raise ValueError("FluidState.kind: 'compressible' | 'isothermal'")
    else:
        raise ValueError("state: pops.Scalar() | pops.FluidState(...)")

    if isinstance(transport, ExB):
        spec.transport = "exb"; spec.B0 = transport.B0
    elif isinstance(transport, CompressibleFlux):
        spec.transport = "compressible"
    elif isinstance(transport, IsothermalFlux):
        spec.transport = "isothermal"
    else:
        raise ValueError("transport: ExB | CompressibleFlux | IsothermalFlux")

    if isinstance(source, NoSource):
        spec.source = "none"
    elif isinstance(source, PotentialForce):
        spec.source = "potential"; spec.qom = source.charge
    elif isinstance(source, GravityForce):
        spec.source = "gravity"
    elif isinstance(source, MagneticLorentzForce):
        spec.source = "magnetic"; spec.qom = source.charge
    elif isinstance(source, PotentialMagneticForce):
        spec.source = "potential_magnetic"; spec.qom = source.charge
    else:
        raise ValueError("source: NoSource | PotentialForce | GravityForce | MagneticLorentzForce "
                         "| PotentialMagneticForce")

    if isinstance(elliptic, ChargeDensity):
        spec.elliptic = "charge"; spec.q = elliptic.charge
    elif isinstance(elliptic, BackgroundDensity):
        spec.elliptic = "background"; spec.alpha = elliptic.alpha; spec.n0 = elliptic.n0
    elif isinstance(elliptic, GravityCoupling):
        spec.elliptic = "gravity"; spec.sign = elliptic.sign
        spec.four_pi_G = elliptic.four_pi_G; spec.rho0 = elliptic.rho0
    else:
        raise ValueError("elliptic: ChargeDensity | BackgroundDensity | GravityCoupling")

    return spec


# --- HYBRID composition: native brick + DSL brick IN A model --------
# pops.Model(...) composes 100% native bricks into a ModelSpec (C++ tags); pops.dsl.Model(...)
# generates a 100% DSL model. pops.CompositeModel(...) fills the in-between: MIX, in ONE SINGLE
# model, NATIVE bricks (pops.ExB / PotentialForce / ChargeDensity ...) and PARTIAL compiled DSL
# bricks (pops.dsl.HyperbolicBrick(...).compile() / SourceBrick / EllipticBrick). The
# mix is compiled into ONE composite .so (prototype: backend 'aot'). cf. pops/dsl.py (Phase B).
def _native_to_brick(obj, role):
    """Translate a NATIVE brick (pops.* object) into a NativeBrick descriptor for the @p role slot.
    An already-compiled DSL brick (CompiledBrick) passes through unchanged (after slot check)."""
    from pops.physics.bricks import CompiledBrick, NativeBrick
    if isinstance(obj, CompiledBrick):
        if obj.kind != role:
            raise ValueError("pops.CompositeModel: DSL brick of type %r placed in the %r slot"
                             % (obj.kind, role))
        return obj
    if role == "hyperbolic":
        if isinstance(obj, ExB):
            return NativeBrick("pops::ExBVelocity", "hyperbolic", fields={"B0": obj.B0},
                                   var_names=["n"], n_vars=1, prim_names=["n"])
        if isinstance(obj, CompressibleFlux):
            g = float(getattr(obj, "gamma", 1.4))
            return NativeBrick("pops::CompressibleFlux", "hyperbolic", fields={"gamma": g},
                                   var_names=["rho", "rho_u", "rho_v", "E"], n_vars=4,
                                   prim_names=["rho", "u", "v", "p"], gamma=g)
        if isinstance(obj, IsothermalFlux):
            cs2 = float(getattr(obj, "cs2", 1.0))
            return NativeBrick("pops::IsothermalFlux", "hyperbolic", fields={"cs2": cs2},
                                   var_names=["rho", "rho_u", "rho_v"], n_vars=3,
                                   prim_names=["rho", "u", "v"])
        raise ValueError("pops.CompositeModel transport: ExB | CompressibleFlux | IsothermalFlux "
                         "(native) or dsl.HyperbolicBrick(...).compile()")
    if role == "source":
        if isinstance(obj, NoSource):
            return NativeBrick("pops::NoSource", "source", min_vars=1)
        if isinstance(obj, PotentialForce):
            return NativeBrick("pops::PotentialForce", "source", fields={"qom": obj.charge},
                                   min_vars=3)
        if isinstance(obj, GravityForce):
            return NativeBrick("pops::GravityForce", "source", min_vars=3)
        if isinstance(obj, MagneticLorentzForce):
            # n_aux=4: the brick reads B_z (canonical aux channel 3) -> the composite sizes the aux.
            return NativeBrick("pops::MagneticLorentzForce", "source",
                                   fields={"qom": obj.charge}, min_vars=3, n_aux=4)
        if isinstance(obj, PotentialMagneticForce):
            # NESTED fields of CompositeSource (public members a / b): the NativeBrick emit
            # writes `a.qom = ...; b.qom = ...;` in the constructor of the derived struct.
            return NativeBrick(
                "pops::CompositeSource<pops::PotentialForce, pops::MagneticLorentzForce>", "source",
                fields={"a.qom": obj.charge, "b.qom": obj.charge}, min_vars=3, n_aux=4)
        raise ValueError("pops.CompositeModel source: NoSource | PotentialForce | GravityForce | "
                         "MagneticLorentzForce | PotentialMagneticForce (native) or "
                         "dsl.SourceBrick(...).compile()")
    if role == "elliptic":
        if isinstance(obj, ChargeDensity):
            return NativeBrick("pops::ChargeDensity", "elliptic", fields={"q": obj.charge},
                                   min_vars=1)
        if isinstance(obj, BackgroundDensity):
            return NativeBrick("pops::BackgroundDensity", "elliptic",
                                   fields={"alpha": obj.alpha, "n0": obj.n0}, min_vars=1)
        if isinstance(obj, GravityCoupling):
            return NativeBrick("pops::GravityCoupling", "elliptic",
                                   fields={"sign": obj.sign, "four_pi_G": obj.four_pi_G,
                                           "rho0": obj.rho0}, min_vars=1)
        raise ValueError("pops.CompositeModel elliptic: ChargeDensity | BackgroundDensity | "
                         "GravityCoupling (native) or dsl.EllipticBrick(...).compile()")
    raise ValueError("pops.CompositeModel: unknown slot %r" % (role,))


def CompositeModel(transport, source, elliptic, name="hybrid"):
    """Compose a HYBRID model mixing NATIVE bricks and PARTIAL DSL bricks in ONE model.

    Each slot (transport / source / elliptic) is EITHER a native brick (pops.ExB(...),
    pops.PotentialForce(...), pops.ChargeDensity(...) ...), OR a compiled partial DSL brick
    (pops.dsl.HyperbolicBrick(...).compile(), pops.dsl.SourceBrick(...).compile(),
    pops.dsl.EllipticBrick(...).compile()). AT LEAST one slot must be a DSL brick: a
    100% native composition is written with pops.Model(...) (ModelSpec).

        tr = pops.dsl.HyperbolicBrick("iso") ...        # DSL transport
        m  = pops.CompositeModel(transport=tr.compile(),
                                source=pops.PotentialForce(charge=-1.0),   # native source
                                elliptic=pops.ChargeDensity(charge=-1.0))  # native elliptic
        co = m.compile(backend="aot")                  # -> CompiledModel
        sim.add_equation("ions", co, spatial=pops.FiniteVolume(), names=[...])

    Returns an pops.dsl.HybridModel; call .compile(backend="aot") for a CompiledModel pluggable
    via System.add_equation. (Prototype: only the 'aot' backend is wired.)"""
    from pops.physics.bricks import CompiledBrick
    from pops.physics.hybrid import HybridModel
    tr = _native_to_brick(transport, "hyperbolic")
    sr = _native_to_brick(source, "source")
    el = _native_to_brick(elliptic, "elliptic")
    if not any(isinstance(b, CompiledBrick) for b in (tr, sr, el)):
        raise ValueError(
            "pops.CompositeModel: all-native composition; use pops.Model(...) (ModelSpec) for "
            "a 100% native model. CompositeModel is for MIXING native + DSL in a single model.")
    return HybridModel(tr, sr, el, name=name)


# --- Elliptic model (EPM): Poisson = a composable instance ------------
# The elliptic model is not a hard-coded special case; it is an EllipticPhysicalModel
# composed of bricks (operator + right-hand side + output). Poisson is its current instance.
class DivEpsGrad:
    """Elliptic operator D = div(eps grad .). eps constant (1.0 = Poisson). Variable eps(x) and
    other operators (diffusion, projection) are refinements (they would touch the solver)."""

    def __init__(self, epsilon=1.0):
        self.epsilon = float(epsilon)


class CompositeRhs:
    """System right-hand side f = sum_s elliptic_rhs_s(u_s): the SUM of the elliptic bricks
    carried by the blocks. Each block chooses its brick (charge q n, background alpha (n-n0), gravity
    coupling sign 4piG (rho-rho0)) via Model(elliptic=...); this right-hand side assembles them. It is the
    GENERIC right-hand side of the EPM: it assumes NO particular form for the contributions."""


class ChargeDensitySource(CompositeRhs):
    """Usual case of the composite right-hand side: all blocks carry a charge density, so
    f = sum_s q_s n_s. Historical alias of CompositeRhs (the computation stays the sum of the bricks)."""


class ElectricFieldFromPotential:
    """Post-processing: E = -grad phi, reinjected into aux of the hyperbolic models."""


class EllipticModel:
    """EllipticPhysicalModel: unknown + operator + right-hand side + output."""

    def __init__(self, unknown, operator, rhs, output):
        self.unknown = unknown
        self.operator = operator
        self.rhs = rhs
        self.output = output


def div_eps_grad(epsilon=1.0):
    return DivEpsGrad(epsilon)


def charge_density():
    return ChargeDensitySource()


def composite_rhs():
    """Generic right-hand side f = sum_s elliptic_rhs_s(u_s) (sum of the per-block bricks)."""
    return CompositeRhs()


def electric_field_from_potential():
    return ElectricFieldFromPotential()


def elliptic(unknown="phi", operator=None, rhs=None, output=None):
    """Compose an EPM. Poisson = elliptic(operator=div_eps_grad(), rhs=charge_density(),
    output=electric_field_from_potential()). The right-hand side can be composite_rhs() (GENERIC
    sum of the per-block elliptic bricks: charge, background, gravity); charge_density() is
    the usual case (alias)."""
    return EllipticModel(unknown, operator or DivEpsGrad(), rhs or CompositeRhs(),
                         output or ElectricFieldFromPotential())


class EllipticSolver:
    """Elliptic solver: 'geometric_mg' (any case, wall) | 'fft' (periodic, n = 2^k, discrete
    stencil) | 'fft_spectral' (periodic, continuous symbol -(kx^2+ky^2): fidelity to spectral
    references such as poisson_fft.m, exact on sinusoids)."""

    def __init__(self, kind="geometric_mg"):
        self.kind = kind
