"""adc : bindings Python de la lib adc_cpp.

Composition multi-especes a l'execution : on compose un systeme bloc par bloc, et
pour chaque bloc on choisit independamment son modele, son schema spatial (limiteur
+ flux), son traitement temporel (explicite / IMEX, ou un integrateur ecrit en
Python) et son nombre de sous-pas. Python dit QUOI assembler, le calcul cellule par
cellule reste dans la lib C++ compilee.

API lisible (objets) :

    import adc
    sim = adc.System(n=192, L=1.0, periodic=False)
    sim.add_block("electrons", model="electron_euler", charge=-1.0,
                  spatial=adc.Spatial(vanleer=True, flux="hllc"),
                  time=adc.IMEX(substeps=10))          # raide -> sous-cycle, source implicite
    sim.add_block("ions", model="ion_isothermal", charge=+1.0,
                  spatial=adc.Spatial(minmod=True, flux="rusanov"),
                  time=adc.Explicit())
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                    wall="circle", wall_radius=0.40)
    sim.set_density("electrons", ne_numpy)             # CI ecrite en numpy
    sim.step_cfl(0.4)

Le bloc compile sa fermeture d'avancee a l'ajout, aucun callback Python dans le hot
path, sauf si on fournit soi-meme un integrateur temporel en Python via les primitives
sim.solve_fields()/sim.eval_rhs(name)/get_state/set_state (Python par pas, le C++ reste
par cellule). Voir adc.integrate.ssprk2_step.
"""

from ._adc import (SystemConfig, System as _System,
                   TwoFluidAP, TwoFluidAPConfig, DiocotronAmr, DiocotronAmrConfig)

__all__ = ["System", "SystemConfig", "Spatial", "Explicit", "IMEX", "Implicit", "integrate",
           "TwoFluidAP", "TwoFluidAPConfig", "DiocotronAmr", "DiocotronAmrConfig"]


class Spatial:
    """Discretisation spatiale d'un bloc : reconstruction (limiteur) + flux numerique.

    limiter : "none" | "minmod" | "vanleer"   (raccourcis booleens minmod=/vanleer=)
    flux    : "rusanov" (robuste) | "hllc" (onde de contact, modeles Euler complets)
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
    """IMEX : transport EXPLICITE + source raide IMPLICITE (Newton local).

    C'est l'« implicite partiel » : seule la partie raide (source) est implicite,
    le transport reste explicite, beaucoup moins cher qu'un implicite total.
    substeps : sous-cyclage du bloc.
    """

    kind = "imex"

    def __init__(self, substeps=1):
        self.substeps = int(substeps)


def Implicit(dt_ratio=1, substeps=None):
    """Alias d'IMEX (implicite PARTIEL : source raide implicite, transport explicite).

    Il n'y a pas d'implicite TOTAL expose (backward-Euler sur le transport aussi) : c'est
    le cas couteux que l'on evite. `Implicit(dt_ratio=k)` vaut exactement `IMEX(substeps=k)`.
    """
    return IMEX(substeps=substeps if substeps is not None else dt_ratio)


class System:
    """Le systeme/coupleur : compose des blocs, partage un Poisson, avance le tout.

    Enveloppe lisible autour de _adc.System : add_block accepte des objets Spatial
    et Explicit/IMEX. Tout le reste (set_poisson, set_density, step, step_cfl,
    diagnostics, primitives eval_rhs/get_state/set_state) est transmis tel quel.
    """

    def __init__(self, config=None, **cfg_kw):
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        self._s = _System(config)
        self._names = []

    def add_block(self, name, model, charge=0.0, spatial=None, time=None):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        self._s.add_block(name, model, float(charge), spatial.limiter, spatial.flux,
                          time.kind, getattr(time, "substeps", 1))
        self._names.append(name)

    def add_species(self, name, model, charge):
        self._s.add_species(name, model, float(charge))
        self._names.append(name)

    def block_names(self):
        """Noms des blocs ajoutes, dans l'ordre (utile a un integrateur Python)."""
        return list(self._names)

    # Tout le reste de l'API (set_poisson, set_density, solve_fields, step, step_cfl,
    # eval_rhs/get_state/set_state, diagnostics) est transmis a la facade compilee.
    def __getattr__(self, attr):
        return getattr(self._s, attr)


from . import integrate  # noqa: E402  (apres la definition de System)
