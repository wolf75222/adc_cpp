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
                   abi_key)  # cle d'ABI du module (chemin DSL "production" / diagnostic)

# L'API PUBLIQUE n'expose QUE des briques composables (System, AmrSystem, Model...) : aucun
# scenario physique nomme. L'integrateur AP deux-fluides (schema asymptotic-preserving, non
# composable bloc a bloc) a quitte le coeur : ce n'est pas une brique generique mais un SCENARIO,
# qui vit desormais dans adc_cases (cf. adc_cases/two_fluid_ap/), compile a la volee contre les
# en-tetes generiques d'adc_cpp. Il n'est donc ni reexporte ici ni present dans le module _adc.
__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "Implicit", "integrate",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "PythonFlux", "dsl", "abi_key",
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


# --- Modele elliptique (EPM) : Poisson = une instance composable ------------
# Le modele elliptique n'est pas un cas special hard-code ; c'est un EllipticPhysicalModel
# compose de briques (operateur + second membre + sortie). Poisson en est l'instance courante.
class DivEpsGrad:
    """Operateur elliptique D = div(eps grad .). eps constant (1.0 = Poisson). eps(x) variable et
    d'autres operateurs (diffusion, projection) sont des raffinements (ils toucheraient le solveur)."""

    def __init__(self, epsilon=1.0):
        self.epsilon = float(epsilon)


class CompositeRhs:
    """Second membre de systeme f = somme_s elliptic_rhs_s(u_s) : la SOMME des briques elliptiques
    portees par les blocs. Chaque bloc choisit sa brique (charge q n, fond alpha (n-n0), couplage
    gravite sign 4piG (rho-rho0)) via Model(elliptic=...) ; ce second membre les assemble. C'est le
    second membre GENERIQUE de l'EPM : il ne suppose AUCUNE forme particuliere des contributions."""


class ChargeDensitySource(CompositeRhs):
    """Cas usuel du second membre composite : tous les blocs portent une densite de charge, donc
    f = somme_s q_s n_s. Alias historique de CompositeRhs (le calcul reste la somme des briques)."""


class ElectricFieldFromPotential:
    """Post-traitement : E = -grad phi, reinjecte dans aux des modeles hyperboliques."""


class EllipticModel:
    """EllipticPhysicalModel : inconnue + operateur + second membre + sortie."""

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
    """Second membre generique f = somme_s elliptic_rhs_s(u_s) (somme des briques par bloc)."""
    return CompositeRhs()


def electric_field_from_potential():
    return ElectricFieldFromPotential()


def elliptic(unknown="phi", operator=None, rhs=None, output=None):
    """Compose un EPM. Poisson = elliptic(operator=div_eps_grad(), rhs=charge_density(),
    output=electric_field_from_potential()). Le second membre peut etre composite_rhs() (somme
    GENERIQUE des briques elliptiques par bloc : charge, fond, gravite) ; charge_density() en est
    le cas usuel (alias)."""
    return EllipticModel(unknown, operator or DivEpsGrad(), rhs or CompositeRhs(),
                         output or ElectricFieldFromPotential())


class EllipticSolver:
    """Solveur elliptique : 'geometric_mg' (tout cas, paroi) | 'fft' (periodique, n = 2^k)."""

    def __init__(self, kind="geometric_mg"):
        self.kind = kind


# --- Couplages inter-especes (operator-split) : objets passes a sim.add_coupling ---
class Ionization:
    """Ionisation n_g -> n_i + n_e (taux k n_e n_g). Masse transferee du neutre vers l'ion."""

    def __init__(self, electron, ion, neutral, rate):
        self.electron = electron
        self.ion = ion
        self.neutral = neutral
        self.rate = rate


class Collision:
    """Friction inter-especes : force k (u_a - u_b), qte de mvt conservee. Blocs fluides (>= 3 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


class ThermalExchange:
    """Echange thermique k (T_a - T_b), energie conservee. Blocs Euler (4 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


# --- Schema spatial + traitement temporel (par bloc) ------------------------
class Spatial:
    """Discretisation spatiale : reconstruction (limiteur) + flux numerique de Riemann.

    limiter : "none" | "minmod" | "vanleer" | "weno5"  (raccourcis none=/minmod=/vanleer=/weno5=)
              weno5 = WENO5-Z, ordre 5 en zone lisse, stencil 5 points (3 ghosts) ; capture sans
              oscillation pres d'un front. Seul le chemin natif add_block l'expose (les chemins
              compiles .so allouent 2 ghosts -> rejet explicite).
    flux    : "rusanov" | "hllc" | "roe"  (hllc/roe exigent un transport compressible)
    recon   : "conservative" | "primitive"  (variables reconstruites ; primitif plus robuste
              pour Euler : positivite de rho et p ; raccourci primitive=)
    """

    def __init__(self, limiter="minmod", flux="rusanov", recon="conservative", *, none=False,
                 minmod=False, vanleer=False, weno5=False, primitive=False):
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


def FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative"):
    """Schema volumes finis (surface stable Phase A) : remappe sur l'objet Spatial existant.

    Le flux NUMERIQUE de Riemann s'appelle `riemann` (NON `flux`, reserve au flux PHYSIQUE du modele
    DSL m.flux) pour ne pas collisionner les deux sens. Mapping des arguments :
      limiter   -> Spatial.limiter  ("none" | "minmod" | "vanleer" | "weno5")
      riemann   -> Spatial.flux     ("rusanov" | "hllc" | "roe")
      variables -> Spatial.recon    ("conservative" | "primitive")

    cf. docs/DSL_MODEL_DESIGN.md section 6. Renvoie un Spatial (consomme tel quel par add_block /
    add_equation). adc.Spatial reste disponible a l'identique."""
    return Spatial(limiter=limiter, flux=riemann, recon=variables)


class Explicit:
    """Traitement temporel explicite. substeps : sous-cyclage du bloc.

    method : "ssprk2" (defaut, Shu-Osher 2 etages ordre 2) | "ssprk3" (3 etages ordre 3, moins
             dissipatif, a apparier a une reconstruction d'ordre eleve comme weno5). Raccourci
             ssprk3=True. Le defaut reste SSPRK2 (comportement historique inchange).
    """

    def __init__(self, substeps=1, method="ssprk2", *, ssprk3=False):
        if ssprk3:
            method = "ssprk3"
        if method not in ("ssprk2", "ssprk3"):
            raise ValueError("Explicit : method 'ssprk2' | 'ssprk3' (recu %r)" % (method,))
        self.substeps = int(substeps)
        self.method = method
        # kind transmis a la facade compilee : "explicit" (SSPRK2, defaut bit-identique) ou "ssprk3".
        self.kind = "ssprk3" if method == "ssprk3" else "explicit"


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

    def add_block(self, name, model, spatial=None, time=None, evolve=True):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), evolve)

    def add_equation(self, name, model, spatial=None, time=None, substeps=None, names=None,
                     evolve=True):
        """Ajoute une equation/bloc en aiguillant sur le TYPE de @p model (DSL Phase A) :

        - un ModelSpec (adc.Model(...)) -> add_block (briques natives composees) ;
        - un CompiledModel (m.compile(...)) -> l'adder du backend (add_dynamic_block pour prototype,
          add_compiled_block pour aot, add_native_block pour production), avec les noms/roles/gamma
          transportes par le .so.

        Centralise le couplage backend <-> adder (un .so AOT ne doit pas etre branche sur
        add_dynamic_block, et inversement). cf. docs/DSL_MODEL_DESIGN.md section 3.

        @p spatial : adc.FiniteVolume(...) / adc.Spatial(...) (defaut minmod+rusanov+conservatif).
        @p time : adc.Explicit / IMEX (defaut Explicit). @p substeps : surcharge time.substeps.
        @p names : noms des composantes (longueur = n_vars du modele compile). @p evolve : bloc avance.
        """
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)

        # --- ModelSpec : briques natives composees -> add_block (chemin existant) ---
        # NB : on appelle _s.add_block DIRECTEMENT avec nsub (pas self.add_block, dont la signature
        # n'a pas de substeps -> elle utiliserait time.substeps et IGNORERAIT l'override substeps=).
        if isinstance(model, ModelSpec):
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, evolve)
            return

        if not isinstance(model, dsl.CompiledModel):
            raise TypeError("add_equation : model doit etre un adc.Model(...) (ModelSpec) ou un "
                            "CompiledModel (m.compile(...)) ; recu %r" % type(model).__name__)

        compiled = model
        # Garde-fou noms : longueur explicite verifiee tot (le C++ leve aussi, mais on diagnostique ici).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError("add_equation : names= a %d noms mais le bloc '%s' a %d variables"
                             % (len(names), name, compiled.n_vars))
        names_arg = list(names) if names is not None else []

        backend = compiled.backend
        # Garde-fou WENO5 : tout CompiledModel passe par un .so qui alloue 2 ghosts ; WENO5 en exige 3
        # (rejete cote C++, system.cpp pour aot/production). On le rejette ICI, AVANT le C++, sur TOUS
        # les chemins .so : WENO5 n'est valide que via le chemin natif add_block (adc.Model ->
        # ModelSpec). Differe pour les .so (Phase A : les 3 ghosts ne sont pas regles).
        if spatial.limiter == "weno5":
            raise ValueError(
                "add_equation : limiter 'weno5' non supporte sur un CompiledModel (.so a 2 ghosts ; "
                "WENO5 en exige 3) ; WENO5 n'est valide que via add_block (modele compose "
                "adc.Model(...)). Differe pour les .so (Phase A).")

        # Garde-fou flux numerique : HLLC/Roe exigent une pression -> la brique generee n'emet
        # pressure()/wave_speeds() que si une primitive 'p' est declaree. Sans 'p', make_block ne
        # compile pas le flux : on le diagnostique ici avant la frontiere C++.
        if spatial.flux in ("hllc", "roe") and "p" not in compiled.prim_names:
            raise ValueError(
                "add_equation : riemann '%s' exige une pression : declarer une primitive 'p' "
                "(m.primitive('p', ...)) dans le modele ; sinon utiliser riemann='rusanov'"
                % spatial.flux)

        # Aiguillage AUTORITAIRE par l'adder du CompiledModel (fixe par le backend, cf. dsl._BACKENDS) :
        # prototype -> add_dynamic_block, aot -> add_compiled_block, production -> add_native_block (#85).
        adder = compiled.adder
        if adder == "add_dynamic_block":
            # JIT, residu HOTE Rusanov ordre 1 : ne prend que le LIMITER MUSCL (none/minmod/vanleer)
            # + substeps ; pas de flux HLLC/Roe, pas de recon primitif.
            if spatial.flux != "rusanov":
                raise ValueError(
                    "add_equation : backend 'prototype' (JIT, residu hote Rusanov ordre 1) n'expose "
                    "que riemann='rusanov' (recu '%s') ; utiliser backend='aot'/'production' pour "
                    "HLLC/Roe" % spatial.flux)
            self._s.add_dynamic_block(name, compiled.so_path, nsub, names_arg, spatial.limiter)
            return
        if adder == "add_compiled_block":
            # AOT host-marshale : limiter x riemann x recon, mono-rang (sans MPI/AMR).
            self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                       spatial.recon, time.kind, nsub, names_arg)
            return
        if adder == "add_native_block":
            # NATIF zero-copie (#85) : bloc installe sur le CONTEXTE REEL du System (meme chemin que
            # add_block). Prend un gamma, PAS de names= (les noms/roles viennent des metadonnees du .so).
            # La validation device/MPI end-to-end depuis Python est une PR dediee ulterieure.
            if names is not None:
                raise ValueError(
                    "add_equation : names= non supporte sur le chemin natif (production) ; les noms et "
                    "roles sont portes par les metadonnees du modele compile (.so)")
            gamma = compiled.gamma if compiled.gamma is not None else 1.4
            self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                     spatial.recon, time.kind, gamma, nsub, evolve)
            return
        raise ValueError("add_equation : adder %r inconnu (backend %r)" % (adder, backend))

    def run(self, t_end, cfl=0.4, max_steps=1_000_000):
        """Avance jusqu'a t_end par pas CFL (sucre : `while time() < t_end: step_cfl(cfl)`).

        @p cfl : nombre de Courant passe a step_cfl. @p max_steps : garde-fou (evite une boucle
        infinie si dt -> 0). Renvoie le nombre de pas effectues. cf. DSL_MODEL_DESIGN.md section 6."""
        steps = 0
        while self.time() < t_end and steps < max_steps:
            self.step_cfl(cfl)
            steps += 1
        return steps

    def add_background(self, name, model, density, spatial=None):
        """Espece GELEE (non avancee) : un fond fixe qui contribue au Poisson de systeme (et, a
        venir, aux sources couplees). density : tableau n*n. Equivaut a add_block(evolve=False)
        suivi de set_density."""
        self.add_block(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def add_elliptic_model(self, name, model, solver=None, bc="auto", wall="none",
                           wall_radius=0.0):
        """EPM : configure le modele elliptique de systeme (Poisson en est l'instance courante).
        model = adc.elliptic(operator=adc.div_eps_grad(eps), rhs=adc.composite_rhs(),
        output=adc.electric_field_from_potential()). set_poisson(...) reste le raccourci equivalent.

        Operateur : div(eps grad) a eps CONSTANT (eps != 1 supporte : eps lap phi = f) ; eps(x)
        variable se branche via set_epsilon_field. Second membre : composite_rhs() = somme GENERIQUE
        des briques elliptiques portees par les blocs (charge q n, fond alpha (n-n0), couplage gravite
        sign 4piG (rho-rho0)) ; charge_density() en est le cas usuel. Diffusion / projection (autre
        operateur) demanderaient un solveur a coefficients variables (raffinement non disponible)."""
        if not isinstance(model.operator, DivEpsGrad):
            raise NotImplementedError("add_elliptic_model : seul l'operateur div_eps_grad (Poisson) "
                                      "est supporte ; diffusion / projection -> raffinement (solveur)")
        if not isinstance(model.rhs, CompositeRhs):
            raise NotImplementedError("add_elliptic_model : rhs doit etre composite_rhs() (somme des "
                                      "briques par bloc) ou charge_density() (son cas usuel)")
        kind = solver.kind if solver is not None else "geometric_mg"
        # Token honnete : "composite" pour un second membre generique, "charge_density" (alias,
        # bit-identique) quand tous les blocs portent une densite de charge. Les deux empruntent le
        # MEME chemin numerique cote C++ (somme des briques elliptiques de chaque bloc).
        rhs_tok = "charge_density" if type(model.rhs) is ChargeDensitySource else "composite"
        self.set_poisson(rhs=rhs_tok, solver=kind, bc=bc, wall=wall, wall_radius=wall_radius,
                         epsilon=model.operator.epsilon)

    def add_coupling(self, coupling):
        """Ajoute un couplage inter-especes : objet adc.Ionization / Collision / ThermalExchange.
        Equivaut a add_ionization / add_collision / add_thermal_exchange."""
        if isinstance(coupling, Ionization):
            self.add_ionization(electron=coupling.electron, ion=coupling.ion,
                                neutral=coupling.neutral, rate=coupling.rate)
        elif isinstance(coupling, Collision):
            self.add_collision(coupling.a, coupling.b, coupling.rate)
        elif isinstance(coupling, ThermalExchange):
            self.add_thermal_exchange(coupling.a, coupling.b, coupling.rate)
        else:
            raise TypeError("add_coupling attend adc.Ionization / Collision / ThermalExchange")

    def block_names(self):
        """Noms des blocs ajoutes, dans l'ordre (utile a un integrateur Python).

        Delegue au registre de blocs C++ (source unique), donc inclut les blocs charges via
        add_dynamic_block (.so JIT) et add_compiled_block (.so AOT), pas seulement add_block.
        """
        return list(self._s.block_names())

    @staticmethod
    def abi_key():
        """Cle d'ABI du module (compilateur, standard C++, signature des en-tetes adc). Comparee a
        celle d'un loader natif par add_native_block. Exposee aussi en tant qu'attribut de classe (le
        delegue __getattr__ ne couvre que les instances), donc adc.System.abi_key() fonctionne."""
        return _System.abi_key()

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
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1))

    def add_equation(self, name, model, spatial=None, time=None, substeps=None):
        """Ajoute l'UNIQUE equation/bloc AMR en aiguillant sur le TYPE de @p model (DSL Phase D) :

        - un ModelSpec (adc.Model(...)) -> add_block (briques natives composees sur la hierarchie) ;
        - un CompiledModel(backend='production', target='amr_system') (m.compile(...)) -> chemin NATIF
          add_native_block : le loader .so inline add_compiled_model(AmrSystem&), donc le bloc tourne
          la MEME hierarchie AMR que add_block (reflux conservatif, regrid), ZERO-COPIE.

        LIMITES AMR (AmrSystem n'est PAS a parite avec System : mono-bloc, EXPLICITE, sans
        reconstruction primitive ni flux de Riemann complet). La facade REJETTE clairement, AVANT la
        frontiere C++, les chemins non cables sur AMR pour un CompiledModel :
          - variables="primitive" (spatial.recon) : le chemin .so AMR reste conservatif ;
          - riemann="roe"/"hllc" (spatial.flux) : seul Rusanov est cable sur AMR via le .so ;
          - limiter="weno5" : le .so alloue 2 ghosts, WENO5 en exige 3.
        Ces limites valent pour le chemin COMPILE (.so) ; un modele natif (adc.Model -> add_block)
        garde l'API add_block existante (recon primitive accepte cote C++). cf. DSL_MODEL_DESIGN.md
        Phase D (point 10).

        @p spatial : adc.FiniteVolume(...) / adc.Spatial(...) (defaut minmod+rusanov+conservatif).
        @p time : adc.Explicit (defaut ; AMR EXPLICITE uniquement). @p substeps : surcharge time.substeps.
        """
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)

        # --- ModelSpec : briques natives composees -> add_block (chemin existant) ---
        if isinstance(model, ModelSpec):
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub)
            return

        if not isinstance(model, dsl.CompiledModel):
            raise TypeError("AmrSystem.add_equation : model doit etre un adc.Model(...) (ModelSpec) "
                            "ou un CompiledModel (m.compile(...)) ; recu %r" % type(model).__name__)

        compiled = model
        # Seul le chemin NATIF "production" cible AmrSystem : il inline add_compiled_model(AmrSystem&).
        # Les .so prototype (JIT) / aot n'ont pas de pendant AMR (add_dynamic_block/add_compiled_block
        # sont mono-niveau). On exige donc backend='production' + target='amr_system'.
        if compiled.adder != "add_native_block":
            raise ValueError(
                "AmrSystem.add_equation : seul un CompiledModel backend='production' (chemin natif) "
                "est branchable sur AMR ; recu backend=%r (les .so prototype/aot sont mono-niveau, "
                "sans pendant AMR)" % compiled.backend)
        if getattr(compiled, "target", "system") != "amr_system":
            raise ValueError(
                "AmrSystem.add_equation : le CompiledModel a ete compile pour target='system' ; "
                "recompiler avec m.compile(..., backend='production', target='amr_system') pour que "
                "le loader inline add_compiled_model(AmrSystem&) (symbole adc_install_native_amr)")

        # LIMITES AMR (non-parite avec System) : rejet CLAIR avant le C++ pour le chemin .so AMR.
        if spatial.recon == "primitive":
            raise ValueError(
                "AmrSystem.add_equation : variables='primitive' non supporte sur le chemin compile "
                "AMR (le .so reste conservatif) ; utiliser variables='conservative' (AmrSystem n'est "
                "pas a parite avec System : pas de reconstruction primitive sur le .so AMR)")
        if spatial.flux in ("roe", "hllc"):
            raise ValueError(
                "AmrSystem.add_equation : riemann='%s' non supporte sur le chemin compile AMR ; seul "
                "riemann='rusanov' est cable sur la hierarchie via le .so (AmrSystem n'est pas a "
                "parite avec System : pas de flux de Riemann complet sur le .so AMR)" % spatial.flux)
        if spatial.limiter == "weno5":
            raise ValueError(
                "AmrSystem.add_equation : limiter 'weno5' non supporte sur un CompiledModel (.so a 2 "
                "ghosts ; WENO5 en exige 3) ; utiliser none/minmod/vanleer")

        gamma = compiled.gamma if compiled.gamma is not None else 1.4
        self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                 spatial.recon, time.kind, gamma, nsub)

    def __getattr__(self, attr):
        return getattr(self._s, attr)


class PythonFlux:
    """Backend de PROTOTYPAGE (hote, numpy) pour l'interface Flux : l'utilisateur fournit le flux
    physique et la vitesse d'onde en Python, et PythonFlux assemble le residu -div(F*) par volumes
    finis (Rusanov, ordre 1, domaine periodique) sur tout le tableau d'un coup.

    HORS hot path GPU/MPI : c'est un chemin HOTE pur (numpy), il ne passe JAMAIS par un kernel
    Kokkos. Pour la production (GPU/MPI), composer un flux COMPILE (brique adc.CompressibleFlux,
    adc.ExB...). PythonFlux formalise le pattern du cas custom_scheme : iterer vite sur un flux
    inedit sans recompiler (adc.System servant d'oracle de Poisson si besoin).

    flux(U, dir) -> F : U et F sont des numpy (ncomp, n, n) ; dir = 0 (x) ou 1 (y).
    max_wave_speed(U) -> float : borne pour le flux de Rusanov et le CFL.
    """

    def __init__(self, flux, max_wave_speed):
        self.flux = flux
        self.max_wave_speed = max_wave_speed

    def residual(self, U, dx, dy=None):
        """-div(F*) par flux de Rusanov (ordre 1, periodique). U numpy (ncomp, n, n) ; rend dU/dt."""
        import numpy as np
        dy = dx if dy is None else dy
        a = float(self.max_wave_speed(U))
        res = np.zeros_like(U)
        for axis, h, d in ((2, dx, 0), (1, dy, 1)):  # x = axe 2, y = axe 1
            F = self.flux(U, d)
            UR = np.roll(U, -1, axis=axis)
            FR = np.roll(F, -1, axis=axis)
            face = 0.5 * (F + FR) - 0.5 * a * (UR - U)       # flux a la face +d de chaque cellule
            res -= (face - np.roll(face, 1, axis=axis)) / h  # -div : (F_{i+1/2} - F_{i-1/2}) / h
        return res

    def cfl_dt(self, U, h, cfl=0.4):
        """Pas de temps stable : dt = cfl * h / max_wave_speed(U)."""
        return cfl * h / max(float(self.max_wave_speed(U)), 1e-30)


from . import integrate  # noqa: E402  (apres la definition de System)
from . import dsl  # noqa: E402  mini-DSL symbolique (prototype, interprete CPU ; apres PythonFlux)
