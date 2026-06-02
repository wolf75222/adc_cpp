"""adc : bindings Python de la lib adc_cpp.

Le coeur expose des BRIQUES generiques compilees (transport, source, second membre
elliptique) ; un MODELE est une composition de briques, nommee cote application. Python
compose les briques (objets), le calcul cellule par cellule reste C++ compile (pas de
numpy, GPU/MPI conserves).

    import adc
    sim = adc.System(n=192, periodic=False)
    sim.add_block(
        "ne",
        model=adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                        source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
    sim.set_density("ne", ne_numpy)
    sim.step_cfl(0.4)

Les noms de scenarios (diocotron, electron_euler...) sont des compositions cote
application (cf. adc_cases). Aucun nom de scenario ici.
"""

from ._adc import (SystemConfig, ModelSpec, System as _System,
                   AmrSystemConfig, AmrSystem as _AmrSystem,
                   TwoFluidAP, TwoFluidAPConfig)

__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "Explicit", "IMEX", "Implicit", "integrate",
    "TwoFluidAP", "TwoFluidAPConfig",
]


# --- Briques d'etat ---------------------------------------------------------
class Scalar:
    """Etat scalaire (1 variable, p.ex. une densite transportee)."""


class FluidState:
    """Etat fluide. kind = "compressible" (gamma) ou "isothermal" (cs2)."""

    def __init__(self, kind="compressible", gamma=1.4, cs2=0.5):
        self.kind = kind
        self.gamma = float(gamma)
        self.cs2 = float(cs2)


# --- Briques de transport ---------------------------------------------------
class ExB:
    """Advection scalaire par la derive E x B (champ magnetique B0)."""

    def __init__(self, B0=1.0):
        self.B0 = float(B0)


class CompressibleFlux:
    """Flux d'Euler compressible (gamma vient de l'etat FluidState)."""


class IsothermalFlux:
    """Flux d'Euler isotherme (cs2 vient de l'etat FluidState)."""


# --- Briques de source ------------------------------------------------------
class NoSource:
    """Pas de source."""


class PotentialForce:
    """Force du potentiel (q/m) rho E sur la quantite de mouvement (+ travail si 4 var)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class GravityForce:
    """Force gravitationnelle rho g (+ travail si 4 var)."""


# --- Briques de second membre elliptique ------------------------------------
class ChargeDensity:
    """Densite de charge f = q n."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class BackgroundDensity:
    """Fond neutralisant f = alpha (n - n0)."""

    def __init__(self, alpha=1.0, n0=0.0):
        self.alpha = float(alpha)
        self.n0 = float(n0)


class GravityCoupling:
    """Couplage self-consistant f = sign 4piG (rho - rho0). sign = +1 gravite, -1 plasma."""

    def __init__(self, sign=1.0, four_pi_G=1.0, rho0=1.0):
        self.sign = float(sign)
        self.four_pi_G = float(four_pi_G)
        self.rho0 = float(rho0)


def Model(state, transport, source, elliptic):
    """Compose un modele (ModelSpec) a partir de briques d'etat, transport, source, elliptique.

    Valide la coherence etat <-> transport (Scalar avec ExB ; FluidState compressible avec
    CompressibleFlux ; isotherme avec IsothermalFlux) et reporte les parametres dans la spec.
    """
    spec = ModelSpec()

    if isinstance(state, Scalar):
        if not isinstance(transport, ExB):
            raise ValueError("Scalar exige transport=ExB(...)")
    elif isinstance(state, FluidState):
        if state.kind == "compressible":
            spec.gamma = state.gamma
            if not isinstance(transport, CompressibleFlux):
                raise ValueError("FluidState(compressible) exige transport=CompressibleFlux()")
        elif state.kind == "isothermal":
            spec.cs2 = state.cs2
            if not isinstance(transport, IsothermalFlux):
                raise ValueError("FluidState(isothermal) exige transport=IsothermalFlux()")
        else:
            raise ValueError("FluidState.kind : 'compressible' | 'isothermal'")
    else:
        raise ValueError("state : adc.Scalar() | adc.FluidState(...)")

    if isinstance(transport, ExB):
        spec.transport = "exb"; spec.B0 = transport.B0
    elif isinstance(transport, CompressibleFlux):
        spec.transport = "compressible"
    elif isinstance(transport, IsothermalFlux):
        spec.transport = "isothermal"
    else:
        raise ValueError("transport : ExB | CompressibleFlux | IsothermalFlux")

    if isinstance(source, NoSource):
        spec.source = "none"
    elif isinstance(source, PotentialForce):
        spec.source = "potential"; spec.qom = source.charge
    elif isinstance(source, GravityForce):
        spec.source = "gravity"
    else:
        raise ValueError("source : NoSource | PotentialForce | GravityForce")

    if isinstance(elliptic, ChargeDensity):
        spec.elliptic = "charge"; spec.q = elliptic.charge
    elif isinstance(elliptic, BackgroundDensity):
        spec.elliptic = "background"; spec.alpha = elliptic.alpha; spec.n0 = elliptic.n0
    elif isinstance(elliptic, GravityCoupling):
        spec.elliptic = "gravity"; spec.sign = elliptic.sign
        spec.four_pi_G = elliptic.four_pi_G; spec.rho0 = elliptic.rho0
    else:
        raise ValueError("elliptic : ChargeDensity | BackgroundDensity | GravityCoupling")

    return spec


# --- Schema spatial + traitement temporel (par bloc) ------------------------
class Spatial:
    """Discretisation spatiale : reconstruction (limiteur) + flux numerique de Riemann.

    limiter : "none" | "minmod" | "vanleer"  (raccourcis none=/minmod=/vanleer=)
    flux    : "rusanov" | "hllc"  (hllc exige un transport compressible)
    """

    def __init__(self, limiter="minmod", flux="rusanov", *, none=False, minmod=False,
                 vanleer=False):
        if none:
            limiter = "none"
        elif minmod:
            limiter = "minmod"
        elif vanleer:
            limiter = "vanleer"
        self.limiter = limiter
        self.flux = flux


class Explicit:
    """Traitement temporel explicite (SSPRK2). substeps : sous-cyclage du bloc."""

    kind = "explicit"

    def __init__(self, substeps=1):
        self.substeps = int(substeps)


class IMEX:
    """IMEX : transport explicite + source raide implicite (Newton local), implicite PARTIEL."""

    kind = "imex"

    def __init__(self, substeps=1):
        self.substeps = int(substeps)


def Implicit(dt_ratio=1, substeps=None):
    """Alias d'IMEX (implicite PARTIEL : source raide implicite, transport explicite).

    Pas d'implicite TOTAL expose (le cas couteux que l'on evite). `Implicit(dt_ratio=k)`
    vaut `IMEX(substeps=k)`.
    """
    return IMEX(substeps=substeps if substeps is not None else dt_ratio)


class System:
    """Le systeme/coupleur : compose des blocs, partage un Poisson, avance le tout.

    add_block prend un modele compose (adc.Model(...)) + des objets Spatial / Explicit / IMEX.
    Tout le reste (set_poisson, set_density, step, step_cfl, step_adaptive, diagnostics,
    primitives eval_rhs/get_state/set_state) est transmis a la facade compilee.
    """

    def __init__(self, config=None, **cfg_kw):
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        self._s = _System(config)
        self._names = []

    def add_block(self, name, model, spatial=None, time=None):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        self._s.add_block(name, model, spatial.limiter, spatial.flux, time.kind,
                          getattr(time, "substeps", 1))
        self._names.append(name)

    def block_names(self):
        """Noms des blocs ajoutes, dans l'ordre (utile a un integrateur Python)."""
        return list(self._names)

    def __getattr__(self, attr):
        return getattr(self._s, attr)


class AmrSystem:
    """Pendant raffine de System : un bloc porte sur une hierarchie AMR (+ set_refinement)."""

    def __init__(self, config=None, **cfg_kw):
        if config is None:
            config = AmrSystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        self._s = _AmrSystem(config)

    def add_block(self, name, model, spatial=None, time=None):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        self._s.add_block(name, model, spatial.limiter, spatial.flux, time.kind,
                          getattr(time, "substeps", 1))

    def __getattr__(self, attr):
        return getattr(self._s, attr)


from . import integrate  # noqa: E402  (apres la definition de System)
