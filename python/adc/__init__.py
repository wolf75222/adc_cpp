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

import os as _os
import sys as _sys

# Le backend DSL "production" charge ensuite un loader .so par dlopen. Ce loader
# resout des symboles C++ exportes par l'extension _adc (System::install_block,
# grid_context, ensure_aux_width, etc.). CPython charge normalement les extensions
# en RTLD_LOCAL sur Unix/macOS ; les symboles restent alors invisibles au loader et
# add_native_block echoue au dlopen ("symbol not found in flat namespace"). On
# charge donc _adc en RTLD_GLOBAL, puis on restaure les flags pour les imports
# suivants. Le module deja charge conserve sa portee globale.
def _explain_missing_extension(exc):
    """Transforme le ModuleNotFoundError brut sur adc._adc en message ACTIONNABLE (bug recurrent :
    l'extension est epinglee a l'ABI cpython-3XY de l'interpreteur qui l'a construite ; sous un
    autre python, l'import echoue sans dire pourquoi). On liste les .so presents a cote du paquet
    et on compare leur tag a l'interpreteur courant."""
    import glob
    here = _os.path.dirname(__file__)
    sos = sorted(_os.path.basename(p) for p in glob.glob(_os.path.join(here, "_adc.*")))
    cur = "cpython-%d%d" % (_sys.version_info[0], _sys.version_info[1])
    if not sos:
        hint = ("aucune extension _adc.*.so dans %s : le module n'est pas construit. Construire avec "
                "`cmake --preset python && cmake --build --preset python`, puis PYTHONPATH=<build>/python."
                % here)
    elif not any(cur in s for s in sos):
        hint = ("extension(s) presente(s) : %s, mais l'interpreteur courant est %s (%s). Utiliser le "
                "python qui a construit le module (env conda `adc`), ou rebatir avec cet interpreteur "
                "(-DPython_EXECUTABLE=%s)." % (", ".join(sos), cur, _sys.executable, _sys.executable))
    else:
        hint = ("l'extension %s correspond a l'interpreteur (%s) mais son import echoue : dependance "
                "manquante ou .so corrompu ; relancer le build du module." % (", ".join(sos), cur))
    return ImportError("import adc._adc impossible : %s\n(cause d'origine : %s)" % (hint, exc))


if hasattr(_sys, "setdlopenflags") and hasattr(_sys, "getdlopenflags"):
    _adc_old_dlopenflags = _sys.getdlopenflags()
    _adc_global_dlopenflags = _adc_old_dlopenflags
    if hasattr(_os, "RTLD_NOW"):
        _adc_global_dlopenflags |= _os.RTLD_NOW
    if hasattr(_os, "RTLD_GLOBAL"):
        _adc_global_dlopenflags |= _os.RTLD_GLOBAL
    _sys.setdlopenflags(_adc_global_dlopenflags)
    try:
        from ._adc import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # cle d'ABI du module (chemin DSL "production" / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e
    finally:
        _sys.setdlopenflags(_adc_old_dlopenflags)
    del _adc_old_dlopenflags, _adc_global_dlopenflags
else:
    try:
        from ._adc import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # cle d'ABI du module (chemin DSL "production" / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e

del _os, _sys, _explain_missing_extension

# Version du paquet = celle bakee dans l'extension (source unique : project(VERSION) CMake).
# Vieux module sans l'attribut -> on degrade en "unknown" plutot que de casser l'import.
try:
    from ._adc import __version__
except ImportError:
    __version__ = "unknown"


# --- Parallelisme : un seul knob runtime --------------------------------------------------------
# Le backend de calcul est COMPILE dans _adc. Le multi-thread (et le GPU) ne sont possibles QUE si
# _adc a ete construit avec -DADC_USE_KOKKOS=ON (device OpenMP). A l'execution, Kokkos s'initialise
# PARESSEUSEMENT a la creation du 1er System/AmrSystem et lit OMP_NUM_THREADS a cet instant precis.
# adc.set_threads(n) ecrit OMP_NUM_THREADS AVANT cette init : un seul appel remplace le rituel
# `OMP_NUM_THREADS=n python ...`. A appeler juste apres `import adc`, avant de creer le 1er systeme.
_first_system_built = False


def has_kokkos():
    """True si _adc a ete compile avec Kokkos (multi-thread/GPU possible), False si SERIE.

    None si le module est trop ancien pour exposer l'info (attribut __has_kokkos__ absent)."""
    from . import _adc
    return getattr(_adc, "__has_kokkos__", None)


def set_threads(n=None):
    """Fixe le nombre de threads de calcul (backend Kokkos OpenMP) en UNE ligne.

    Equivaut a exporter OMP_NUM_THREADS=n avant de lancer Python, mais sans toucher au shell. N'a
    d'effet que si _adc a ete compile avec -DADC_USE_KOKKOS=ON (preset 'python-parallel'), et DOIT
    etre appele AVANT le 1er System/AmrSystem (Kokkos s'initialise paresseusement a ce moment-la et
    lit OMP_NUM_THREADS une seule fois) :

        import adc
        adc.set_threads(8)     # 8 threads
        adc.set_threads()      # tous les coeurs (os.cpu_count())
        sim = adc.System(n=256)

    Un module SERIE ou un appel tardif sont signales par un avertissement (sans lever d'exception)."""
    import os
    import warnings
    if n is None:                       # defaut : tous les coeurs logiques disponibles
        n = os.cpu_count() or 1
    n = int(n)
    if n < 1:
        raise ValueError("adc.set_threads : n doit etre >= 1")
    # Source de verite : l'etat REEL du runtime Kokkos (couvre TOUTES les voies d'init lazy --
    # System, AmrSystem, .so DSL, usage direct de _adc). Le drapeau Python reste le repli pour
    # un vieux module sans le binding.
    from . import _adc
    _kokkos_started = getattr(_adc, "kokkos_is_initialized", lambda: _first_system_built)()
    if _kokkos_started or _first_system_built:
        warnings.warn(
            "adc.set_threads : appele APRES l'initialisation du runtime (1er System/AmrSystem ou "
            "1re allocation) -> SANS EFFET. Appeler set_threads juste apres `import adc`.",
            RuntimeWarning, stacklevel=2)
        return
    if has_kokkos() is False:
        warnings.warn(
            "adc.set_threads : _adc est SERIE (compile sans -DADC_USE_KOKKOS=ON) -> le reglage de "
            "threads est ignore au calcul. Reconstruire avec -DADC_USE_KOKKOS=ON "
            "-DKokkos_ROOT=$CONDA_PREFIX pour le multi-thread.", RuntimeWarning, stacklevel=2)
    # On ecrit l'env meme en cas de doute (inoffensif) : un .so DSL backend='production' compile avec
    # Kokkos lira lui aussi OMP_NUM_THREADS a son initialisation.
    # On positionne DEUX variables pour etre agnostique au backend que Kokkos a compile :
    #   - OMP_NUM_THREADS  : lu par le device OpenMP (cas usuel) ;
    #   - KOKKOS_NUM_THREADS : lu par Kokkos::initialize quel que soit le device (OpenMP OU Threads),
    #     utile si le Kokkos installe (p.ex. conda-forge) utilise le backend Threads et non OpenMP.
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["KOKKOS_NUM_THREADS"] = str(n)
    # OMP_PROC_BIND=false UNIQUEMENT sur macOS (evite les warnings/oversubscription de libomp sur
    # les Mac de dev). Sur Linux/cluster on n'impose RIEN : desactiver l'affinite y degraderait le
    # scaling NUMA, et un job SLURM qui exporte OMP_PROC_BIND=close/spread reste maitre (setdefault
    # ne l'ecraserait pas de toute facon).
    import sys as _s
    if _s.platform == "darwin":
        os.environ.setdefault("OMP_PROC_BIND", "false")


def parallel_info():
    """Etat du parallelisme : backend compile, OMP_NUM_THREADS courant, init Kokkos deja faite."""
    import os
    return {
        "has_kokkos": has_kokkos(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "first_system_built": _first_system_built,
    }


def doctor(verbose=True):
    """Diagnostic de l'environnement adc en UNE commande : python -c "import adc; adc.doctor()".

    Verifie chaque maillon dont dependent le module ET la compilation runtime du DSL (la classe de
    bugs "environnement du build != environnement d'execution", p.ex. le `which c++` d'un env conda
    qui rejette -std=c++23). Renvoie un dict {check: (ok, detail)} ; verbose=True l'affiche."""
    import os
    import sys
    checks = {}

    # 1. interpreteur + extension (piege ABI cpython-3XY)
    from . import _adc
    so = getattr(_adc, "__file__", "?")
    checks["interpreteur"] = (True, "%s (%d.%d) ; extension %s"
                              % (sys.executable, sys.version_info[0], sys.version_info[1], so))

    # 2. numpy (requis a l'import de adc.dsl)
    try:
        import numpy
        checks["numpy"] = (True, numpy.__version__)
    except Exception as e:
        checks["numpy"] = (False, "ABSENT de cet interpreteur (%s) -> `import adc.dsl` echouera. "
                                  "Installer numpy dans CE python." % e)

    # 3. backend de calcul compile
    hk = has_kokkos()
    checks["kokkos"] = (hk is not False,
                        {True: "module Kokkos (multi-thread possible ; adc.set_threads actif)",
                         False: "module SERIE (set_threads sans effet ; rebatir preset python-parallel)",
                         None: "indetermine (vieux module sans __has_kokkos__)"}[hk])

    # 4. compilateur du DSL runtime (le maillon du bug -std=c++23)
    try:
        from . import dsl as _dsl
    except Exception as e:
        checks["dsl"] = (False, "import adc.dsl impossible (%s)" % e)
        _dsl = None
    if _dsl is not None:
        baked = _dsl.loader_cxx_compiler()
        cc = _dsl._default_cxx(None)
        if not cc:
            checks["compilateur"] = (False, "AUCUN compilateur C++ trouve (ADC_CXX, module, PATH). "
                                            "Installer Xcode CLT (macOS) ou `conda install cxx-compiler`.")
        else:
            origin = ("$ADC_CXX" if os.environ.get("ADC_CXX") == cc
                      else "bake par le build de _adc" if cc == baked else "PATH (which)")
            try:
                std = _dsl._probe_cxx_std(cc, _dsl.loader_cxx_std())
                checks["compilateur"] = (True, "%s [%s] ; -std=%s accepte" % (cc, origin, std))
            except RuntimeError as e:
                checks["compilateur"] = (False, str(e).splitlines()[0])
            if baked and cc != baked:
                checks["compilateur_abi"] = (False, "compilateur runtime (%s) != build (%s) -> risque "
                                                    "de rejet 'ABI incompatible' sur backend "
                                                    "production. export ADC_CXX=%r pour forcer celui "
                                                    "du build." % (cc, baked, baked))

        # 5. en-tetes adc (DSL production : la signature doit matcher celle bakee dans _adc)
        try:
            inc = _dsl.adc_include()
            checks["include"] = (True, inc)
            # 5b. SYNCHRONISATION en-tetes <-> module (bug reel : module bati AVANT un git pull ->
            # le loader DSL reference des signatures C++ absentes du vieux .so -> dlopen 'symbol
            # not found' cryptique). On compare la signature bakee a celle de l'arbre actuel.
            baked_sig = _dsl.module_header_signature()
            if baked_sig is not None:
                cur_sig = _dsl.adc_header_signature(inc)
                if cur_sig == baked_sig:
                    checks["headers_sync"] = (True, "en-tetes == build du module (sig %s...)"
                                              % baked_sig[:12])
                else:
                    checks["headers_sync"] = (False, "en-tetes MODIFIES depuis le build de _adc "
                                                     "(module perime) -> rebatir : cmake --build "
                                                     "build-py --target _adc (sinon : dlopen "
                                                     "'symbol not found' sur backend production)")
        except RuntimeError as e:
            checks["include"] = (False, "en-tetes adc introuvables (definir ADC_INCLUDE) : %s" % e)

    # 6. threads courants
    checks["threads"] = (True, "OMP_NUM_THREADS=%s ; premier System cree=%s"
                         % (os.environ.get("OMP_NUM_THREADS", "(defaut)"), _first_system_built))

    if verbose:
        for cname, (ok, detail) in checks.items():
            print("[%s] %-16s %s" % ("OK " if ok else "FAIL", cname, detail))
        if all(ok for ok, _ in checks.values()):
            print("=> environnement sain : module importable, DSL compilable, ABI coherente.")
        else:
            print("=> corriger les FAIL ci-dessus avant d'utiliser le DSL backend='production'.")
    return checks


# L'API PUBLIQUE n'expose QUE des briques composables (System, AmrSystem, Model...) : aucun
# scenario physique nomme. L'integrateur AP deux-fluides (schema asymptotic-preserving, non
# composable bloc a bloc) a quitte le coeur : ce n'est pas une brique generique mais un SCENARIO,
# qui vit desormais dans adc_cases (cf. adc_cases/two_fluid_ap/), compile a la volee contre les
# en-tetes generiques d'adc_cpp. Il n'est donc ni reexporte ici ni present dans le module _adc.
__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model", "CompositeModel",
    "CartesianMesh", "PolarMesh",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE",
    "Implicit", "Split", "Strang", "CondensedSchur", "Role", "integrate",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "PythonFlux", "dsl", "abi_key", "capabilities",
    "set_threads", "has_kokkos", "parallel_info", "doctor",
]


# --- Maillage / geometrie (chantier "grille polaire", Phase 1) --------------
# Le CHOIX de la geometrie vit dans un objet MAILLAGE, pas dans le schema : adc.FiniteVolume reste
# reconstruction + flux de Riemann + variables (aucun argument de geometrie). On passe le maillage au
# systeme via adc.System(mesh=...). adc.CartesianMesh est le defaut implicite (domaine carre, numerique
# STRICTEMENT inchangee, bit-identique). adc.PolarMesh decrit un anneau global (r, theta).
class CartesianMesh:
    """Maillage CARTESIEN (defaut implicite) : domaine carre [0, L]^2, n x n cellules.

    C'est la geometrie historique : adc.System(mesh=adc.CartesianMesh(n, L, periodic)) est STRICTEMENT
    equivalent (bit-identique) a adc.System(n=n, L=L, periodic=periodic). Fourni pour la symetrie avec
    adc.PolarMesh (le choix de geometrie est explicite des deux cotes)."""

    def __init__(self, n=64, L=1.0, periodic=True):
        self.n = int(n)
        self.L = float(L)
        self.periodic = bool(periodic)

    def _apply(self, config):
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic


class PolarMesh:
    """Maillage POLAIRE ANNULAIRE GLOBAL (chantier "grille polaire diocotron", Phase 1) : domaine
    r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta cellules. theta est PERIODIQUE, r porte une
    condition aux limites PHYSIQUE (paroi / sortie). Convention d'axes : direction 0 = radiale,
    direction 1 = azimutale (cf. PolarGeometry / assemble_rhs_polar cote C++).

    Le proto Phase-0 a quantifie que la grille cartesienne diffuse le gradient RADIAL d'un anneau en
    rotation azimutale (rapport 73 vs polaire) : porter la direction radiale sur un axe de grille leve
    ce verrou structural du diocotron.

    PORTEE (mise a jour audit 2026-06) : le chemin polaire est BRANCHE dans System.step (transport
    polaire assemble_rhs_polar + Poisson polaire + aux derive en base locale (e_r, e_theta)).
    adc.System(mesh=adc.PolarMesh(...)) construit un anneau global et avance dessus. TROIS niveaux a
    ne pas confondre :
    - transport polaire : ExB scalaire ET fluide isotherme (IsothermalFluxPolar) ; flux Riemann
      'rusanov' (defaut, tout transport) ET 'hll' (fluide isotherme seulement -- gate model.wave_speeds,
      identique au cartesien ; l'ExB scalaire ne fournit pas de wave_speeds -> 'hll' leve un rejet
      clair). 'hllc'/'roe' restent leves cote C++ (Euler 4 var, sans brique polaire) ;
    - Poisson polaire DIRECT (PolarPoissonSolver) : mono-rang, une box couvrant l'anneau ;
    - etage Schur polaire TENSORIEL (PolarCondensedSchurSourceStepper, via adc.Split/CondensedSchur) :
      le solveur C++ est multi-rang/multi-box (decoupage theta).

    DECOUPAGE THETA DU TRANSPORT (theta_boxes, ADC-67). theta_boxes=1 (defaut) = mono-box,
    STRICTEMENT bit-identique a l'historique. theta_boxes>1 = l'anneau est decoupe en BANDES theta
    (chaque boite couvre tout le rayon et une bande azimutale ; theta_boxes doit DIVISER ntheta et
    rester <= ntheta) et le TRANSPORT polaire (assemble_rhs_polar + fill_ghosts collectif) tourne
    multi-box. MATRICE des capacites multi-box :
    - TRANSPORT polaire (System transport, get/set state, eval_rhs, density) : multi-box OK
      (assemblage par boite + halos collectifs ; l'etat global est reconstruit a la lecture) ;
    - Poisson polaire DIRECT (PolarPoissonSolver) : MONO-BOX ONLY. Un System a theta_boxes>1 qui
      resout le champ direct (solve_fields / step / potential, p.ex. un bloc ExB scalaire couple)
      leve une erreur AMONT claire (le solveur direct exige lignes theta + colonnes r completes sur
      une box) : utiliser theta_boxes=1 OU l'etage Schur tensoriel ;
    - etage Schur tensoriel polaire (adc.Split + adc.CondensedSchur) : multi-box (solveur C++
      multi-box ; le decoupage theta est desormais pilotable par theta_boxes).
    Mono-rang (le Poisson polaire direct refuse MPI). Pas de couplage cartesien<->polaire (anneau
    global). Bornes de pas optionnelles (stability_speed/stability_dt/source_frequency) NON cablees
    sur le chemin polaire (transport max_wave_speed seulement). Cf. docs/GENERICITY_2026-06.md
    section 3 et adc.capabilities()['geometry']."""

    def __init__(self, r_min, r_max, nr, ntheta, theta_boxes=1):
        if not (r_max > r_min >= 0.0):
            raise ValueError("PolarMesh : exige r_max > r_min >= 0 (anneau)")
        # nr >= 3 : la derive radiale de l'aux (System.solve_fields_polar) utilise un stencil DECENTRE
        # d'ordre 2 aux deux parois sur phi (sans ghost) ; nr < 3 lirait phi hors bornes. Un anneau
        # global a toujours nr >= 3. ntheta >= 1 (la derive azimutale enroule l'indice periodique).
        if nr < 3:
            raise ValueError("PolarMesh : nr >= 3 (stencil radial decentre d'ordre 2 aux parois)")
        if ntheta < 1:
            raise ValueError("PolarMesh : ntheta >= 1")
        # theta_boxes : decoupage du transport en bandes theta (1 = mono-box, defaut). On valide ICI
        # (cote Python, message clair) ET cote C++ (check_geometry, pour un SystemConfig construit a la
        # main) : 1 <= theta_boxes <= ntheta ET theta_boxes DIVISE ntheta (bandes azimutales egales).
        tb = int(theta_boxes)
        if tb < 1:
            raise ValueError("PolarMesh : theta_boxes >= 1 (1 = mono-box)")
        if tb > int(ntheta):
            raise ValueError("PolarMesh : theta_boxes <= ntheta (au moins une cellule azimutale par bande)")
        if int(ntheta) % tb != 0:
            raise ValueError("PolarMesh : theta_boxes doit DIVISER ntheta (bandes azimutales egales)")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.theta_boxes = tb

    def _apply(self, config):
        config.geometry = "polar"
        config.nr = self.nr
        config.ntheta = self.ntheta
        config.r_min = self.r_min
        config.r_max = self.r_max
        config.theta_boxes = self.theta_boxes
        config.n = self.nr  # n sert de taille par defaut au reste de la config (diagnostics)


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


class MagneticLorentzForce:
    """Force de Lorentz MAGNETIQUE q (v x B_z) sur la quantite de mouvement (brique C++ native
    adc::MagneticLorentzForce, exposee a l'API Python par l'audit 2026-06).

    Regime EXPLICITE (omega_c modere) : terme ponctuel algebrique, sans travail (F . v = 0, energie
    inchangee). Lit B_z dans le canal aux (composante canonique 3) : appeler
    ``sim.set_magnetic_field(Bz)`` pour le peupler. Exige un transport fluide >= 3 variables (qdm
    sur 2 axes) ; rejete sur un scalaire. Le regime RAIDE (omega_c grand) passe par l'etage condense
    adc.CondensedSchur (Schur), PAS par cette brique explicite.

    ``charge`` = q/m, signe inclus (meme convention que PotentialForce)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class PotentialMagneticForce:
    """Force electrostatique + Lorentz magnetique SOMMEES : (q/m) rho E + q (v x B_z) (brique C++
    native CompositeSource<PotentialForce, MagneticLorentzForce>, la force complete du diocotron
    magnetise). Meme q/m pour les deux forces (meme espece). Lit B_z (set_magnetic_field) ; exige un
    transport fluide >= 3 variables. ``charge`` = q/m, signe inclus."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


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
    elif isinstance(source, MagneticLorentzForce):
        spec.source = "magnetic"; spec.qom = source.charge
    elif isinstance(source, PotentialMagneticForce):
        spec.source = "potential_magnetic"; spec.qom = source.charge
    else:
        raise ValueError("source : NoSource | PotentialForce | GravityForce | MagneticLorentzForce "
                         "| PotentialMagneticForce")

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


# --- Composition HYBRIDE : brique native + brique DSL DANS UN modele --------
# adc.Model(...) compose des briques 100% natives en une ModelSpec (tags C++) ; adc.dsl.Model(...)
# genere un modele 100% DSL. adc.CompositeModel(...) comble l'entre-deux : MELANGER, dans UN SEUL
# modele, des briques NATIVES (adc.ExB / PotentialForce / ChargeDensity ...) et des briques DSL
# PARTIELLES compilees (adc.dsl.HyperbolicBrick(...).compile() / SourceBrick / EllipticBrick). Le
# melange est compile en UN .so composite (prototype : backend 'aot'). cf. adc/dsl.py (Phase B).
def _native_to_brick(obj, role):
    """Traduit une brique NATIVE (objet adc.*) en descripteur dsl.NativeBrick pour le slot @p role.
    Une brique DSL deja compilee (dsl.CompiledBrick) passe telle quelle (apres verification du slot)."""
    from . import dsl
    if isinstance(obj, dsl.CompiledBrick):
        if obj.kind != role:
            raise ValueError("adc.CompositeModel : brique DSL de type %r placee dans le slot %r"
                             % (obj.kind, role))
        return obj
    if role == "hyperbolic":
        if isinstance(obj, ExB):
            return dsl.NativeBrick("adc::ExBVelocity", "hyperbolic", fields={"B0": obj.B0},
                                   var_names=["n"], n_vars=1, prim_names=["n"])
        if isinstance(obj, CompressibleFlux):
            g = float(getattr(obj, "gamma", 1.4))
            return dsl.NativeBrick("adc::CompressibleFlux", "hyperbolic", fields={"gamma": g},
                                   var_names=["rho", "rho_u", "rho_v", "E"], n_vars=4,
                                   prim_names=["rho", "u", "v", "p"], gamma=g)
        if isinstance(obj, IsothermalFlux):
            cs2 = float(getattr(obj, "cs2", 1.0))
            return dsl.NativeBrick("adc::IsothermalFlux", "hyperbolic", fields={"cs2": cs2},
                                   var_names=["rho", "rho_u", "rho_v"], n_vars=3,
                                   prim_names=["rho", "u", "v"])
        raise ValueError("adc.CompositeModel transport : ExB | CompressibleFlux | IsothermalFlux "
                         "(natif) ou dsl.HyperbolicBrick(...).compile()")
    if role == "source":
        if isinstance(obj, NoSource):
            return dsl.NativeBrick("adc::NoSource", "source", min_vars=1)
        if isinstance(obj, PotentialForce):
            return dsl.NativeBrick("adc::PotentialForce", "source", fields={"qom": obj.charge},
                                   min_vars=3)
        if isinstance(obj, GravityForce):
            return dsl.NativeBrick("adc::GravityForce", "source", min_vars=3)
        if isinstance(obj, MagneticLorentzForce):
            # n_aux=4 : la brique lit B_z (canal aux canonique 3) -> le composite dimensionne l'aux.
            return dsl.NativeBrick("adc::MagneticLorentzForce", "source",
                                   fields={"qom": obj.charge}, min_vars=3, n_aux=4)
        if isinstance(obj, PotentialMagneticForce):
            # Champs IMBRIQUES de CompositeSource (membres publics a / b) : l'emit du NativeBrick
            # ecrit `a.qom = ...; b.qom = ...;` dans le constructeur du struct derive.
            return dsl.NativeBrick(
                "adc::CompositeSource<adc::PotentialForce, adc::MagneticLorentzForce>", "source",
                fields={"a.qom": obj.charge, "b.qom": obj.charge}, min_vars=3, n_aux=4)
        raise ValueError("adc.CompositeModel source : NoSource | PotentialForce | GravityForce | "
                         "MagneticLorentzForce | PotentialMagneticForce (natif) ou "
                         "dsl.SourceBrick(...).compile()")
    if role == "elliptic":
        if isinstance(obj, ChargeDensity):
            return dsl.NativeBrick("adc::ChargeDensity", "elliptic", fields={"q": obj.charge},
                                   min_vars=1)
        if isinstance(obj, BackgroundDensity):
            return dsl.NativeBrick("adc::BackgroundDensity", "elliptic",
                                   fields={"alpha": obj.alpha, "n0": obj.n0}, min_vars=1)
        if isinstance(obj, GravityCoupling):
            return dsl.NativeBrick("adc::GravityCoupling", "elliptic",
                                   fields={"sign": obj.sign, "four_pi_G": obj.four_pi_G,
                                           "rho0": obj.rho0}, min_vars=1)
        raise ValueError("adc.CompositeModel elliptic : ChargeDensity | BackgroundDensity | "
                         "GravityCoupling (natif) ou dsl.EllipticBrick(...).compile()")
    raise ValueError("adc.CompositeModel : slot %r inconnu" % (role,))


def CompositeModel(transport, source, elliptic, name="hybrid"):
    """Compose un modele HYBRIDE melant briques NATIVES et briques DSL PARTIELLES dans UN modele.

    Chaque slot (transport / source / elliptic) est SOIT une brique native (adc.ExB(...),
    adc.PotentialForce(...), adc.ChargeDensity(...) ...), SOIT une brique DSL partielle compilee
    (adc.dsl.HyperbolicBrick(...).compile(), adc.dsl.SourceBrick(...).compile(),
    adc.dsl.EllipticBrick(...).compile()). AU MOINS un slot doit etre une brique DSL : une composition
    100% native s'ecrit avec adc.Model(...) (ModelSpec).

        tr = adc.dsl.HyperbolicBrick("iso") ...        # transport DSL
        m  = adc.CompositeModel(transport=tr.compile(),
                                source=adc.PotentialForce(charge=-1.0),   # source native
                                elliptic=adc.ChargeDensity(charge=-1.0))  # elliptique native
        co = m.compile(backend="aot")                  # -> CompiledModel
        sim.add_equation("ions", co, spatial=adc.FiniteVolume(), names=[...])

    Renvoie un adc.dsl.HybridModel ; appeler .compile(backend="aot") pour un CompiledModel branchable
    via System.add_equation. (Prototype : seul le backend 'aot' est cable.)"""
    from . import dsl
    tr = _native_to_brick(transport, "hyperbolic")
    sr = _native_to_brick(source, "source")
    el = _native_to_brick(elliptic, "elliptic")
    if not any(isinstance(b, dsl.CompiledBrick) for b in (tr, sr, el)):
        raise ValueError(
            "adc.CompositeModel : composition tout-native ; utiliser adc.Model(...) (ModelSpec) pour "
            "un modele 100% natif. CompositeModel sert au MELANGE natif + DSL dans un seul modele.")
    return dsl.HybridModel(tr, sr, el, name=name)


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

    - ``limiter`` : "none" | "minmod" | "vanleer" | "weno5" (raccourcis none=/minmod=/vanleer=/weno5=).
      weno5 = WENO5-Z, ordre 5 en zone lisse, stencil 5 points (3 ghosts), capture sans oscillation
      pres d'un front ; seul le chemin natif ``add_block`` l'expose (les chemins compiles .so
      allouent 2 ghosts -> rejet explicite).
    - ``flux`` : "rusanov" | "hll" | "hllc" | "roe".
      rusanov = generique minimal (ne demande que max_wave_speed, tout modele).
      hll = generique a ondes signees (exige model.wave_speeds : modele natif isotherme/compressible,
      ou modele DSL declarant une primitive 'p') ; moins diffusif que rusanov, sans exiger de
      pression ni n_vars == 4. C'est le chemin recommande pour un modele NON Euler a ondes signees
      (systeme de moments, isotherme) : ``hll`` + ``minmod``.
      hllc / roe = EULER 2D SEULEMENT (4 variables rho/rho_u/rho_v/E + pression gaz parfait) ;
      ce ne sont PAS des solveurs generiques (cf. EulerHLLCFlux2D / EulerRoeFlux2D cote C++).
    - ``recon`` : "conservative" | "primitive" (variables reconstruites ; primitif plus robuste
      pour Euler : positivite de rho et p ; raccourci primitive=).
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

    Le flux NUMERIQUE de Riemann s'appelle ``riemann`` (NON ``flux``, reserve au flux PHYSIQUE du modele
    DSL m.flux) pour ne pas collisionner les deux sens. Mapping des arguments :

    - ``limiter`` -> Spatial.limiter ("none" | "minmod" | "vanleer" | "weno5")
    - ``riemann`` -> Spatial.flux ("rusanov" | "hll" | "hllc" | "roe") ; "hll" est le chemin
      generique a ondes signees (exige model.wave_speeds), "hllc"/"roe" sont Euler 2D seulement
    - ``variables`` -> Spatial.recon ("conservative" | "primitive")

    cf. docs/DSL_MODEL_DESIGN.md section 6. Renvoie un Spatial (consomme tel quel par add_block /
    add_equation). adc.Spatial reste disponible a l'identique."""
    return Spatial(limiter=limiter, flux=riemann, recon=variables)


class Explicit:
    """Traitement temporel explicite.

    substeps=N : le bloc avance N fois par macro-pas, chaque sous-pas de longueur dt/N
                 (electrons rapides : substeps=10). Defaut 1 = comportement historique.
    stride=M   : cadence du bloc, semantique HOLD-THEN-CATCH-UP (rattrapage en FIN de fenetre).
                 Le bloc est TENU (non avance) tant que (macro_step + 1) % M != 0, puis avance d'un
                 pas effectif M*dt au macro-pas ou (macro_step + 1) % M == 0, i.e. a la fin de chaque
                 fenetre de M macro-pas (bloc lent, p.ex. neutres : stride=20). Il reste ainsi
                 temporellement COHERENT avec les blocs rapides (jamais avance "dans le futur"). Defaut
                 1 = chaque macro-pas, bit-identique a l'historique. substeps et stride sont ORTHOGONAUX :
                 stride=M, substeps=N -> N sous-pas de M*dt/N une fois en fin de fenetre.
                 COUPLAGE POISSON : entre deux rattrapages, le bloc tenu contribue au second membre du
                 Poisson de systeme (et aux sources couplees) avec son etat PERIME -- sa derniere densite
                 /charge avancee, figee jusqu'au prochain rattrapage. step_cfl honore la cadence : le pas
                 stable inclut le facteur stride (dt <= cfl*h*substeps / (stride*w)).
                 NB : le backend 'aot' (System.add_equation sur un CompiledModel backend='aot') ne
                 transporte PAS la cadence et REJETTE stride > 1 (route explicite, pas d'ignore silencieux) ;
                 add_block (natif) et backend='production' supportent le stride.
    method     : "ssprk2" (defaut, Shu-Osher 2 etages ordre 2) | "ssprk3" (3 etages ordre 3,
                 moins dissipatif, a apparier a weno5). Raccourci ssprk3=True.
    """

    def __init__(self, substeps=1, method="ssprk2", stride=1, *, ssprk3=False):
        if ssprk3:
            method = "ssprk3"
        if method not in ("ssprk2", "ssprk3"):
            raise ValueError("Explicit : method 'ssprk2' | 'ssprk3' (recu %r)" % (method,))
        if int(substeps) < 1:
            raise ValueError("Explicit : substeps >= 1 (recu %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("Explicit : stride >= 1 (recu %r)" % (stride,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.method = method
        # kind transmis a la facade compilee : "explicit" (SSPRK2, defaut bit-identique) ou "ssprk3".
        self.kind = "ssprk3" if method == "ssprk3" else "explicit"


def _role_to_stable(name):
    """Normalise un nom de role vers la cle STABLE attendue par le C++ (role_from_name) : minuscules
    snake_case ("momentum_x", "energy"). Tolere les variantes PascalCase de l'enum C++ exposees dans
    l'API cible (ex. "MomentumX" -> "momentum_x", "Energy" -> "energy") en inserant un '_' avant chaque
    majuscule interne avant la mise en minuscules. Un nom deja en snake_case ("momentum_x") est inchange."""
    s = str(name).strip()
    if not s:
        return s
    if s == s.lower():  # deja snake_case / minuscules : inchange
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
    """Normalise les listes du masque implicite (noms / roles physiques) en listes de chaines.

    None -> [] (defaut : masque inactif, defaut modele, bit-identique). Une chaine seule est tolerree
    (ex. implicit_vars="rho_u" -> ["rho_u"]). Les roles sont ramenes a la cle STABLE du C++ (snake_case)
    via _role_to_stable -> "MomentumX" et "momentum_x" sont equivalents. Le masque vit cote POLITIQUE
    TEMPORELLE / bloc (et NON le modele) : le MEME modele se reutilise avec des traitements implicites
    distincts. La RESOLUTION des noms/roles -> indices et la validation (nom/role absent du bloc) vit
    cote C++ (System::add_block), seule source de verite des noms/roles du bloc."""
    def as_list(x, what):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        try:
            out = [str(v) for v in x]
        except TypeError:
            raise ValueError("%s : %s doit etre une liste de chaines (recu %r)" % (label, what, x))
        return out
    names = as_list(implicit_vars, "implicit_vars")
    roles = [_role_to_stable(r) for r in as_list(implicit_roles, "implicit_roles")]
    return names, roles


class IMEX:
    """IMEX : transport explicite (SSPRK) + source raide implicite (backward-Euler, Newton local).

    Traitement PARTIEL : seule la SOURCE est implicite (backward-Euler, Newton local a la cellule,
    via backward_euler_source / ImplicitSourceStepper cote C++). Le TRANSPORT reste explicite
    (avance par le coeur SSPRK). Ce n'est PAS un solveur implicite global (flux + source + Poisson
    resolus implicitement / Newton-Krylov) -- ce chantier est une phase future distincte.

    - ``substeps=N`` : sous-pas par macro-pas (cf. Explicit). Defaut 1.
    - ``stride=M`` : cadence du bloc, semantique hold-then-catch-up (cf. Explicit) : le bloc est tenu
      tant que (macro_step + 1) % M != 0, puis avance d'un pas effectif M*dt en fin de fenetre. Entre
      deux rattrapages, son etat PERIME contribue au Poisson de systeme. Defaut 1 = chaque macro-pas,
      bit-identique. Backend 'aot' : stride > 1 rejete (cf. Explicit).
    - ``implicit_vars`` : noms des variables conservees a traiter en IMPLICITE dans le pas de source ;
      les autres restent explicites (Euler avant). Le masque est PORTE PAR CETTE POLITIQUE / le bloc,
      PAS par le modele -> le MEME modele se reutilise avec des traitements implicites differents.
      Defaut [] (+ implicit_roles []) = defaut du modele (Model::is_implicit, ou tout implicite a
      defaut), BIT-IDENTIQUE. Resolu cote C++ contre les noms du bloc (un nom absent leve une erreur).
      Ex. adc.IMEX(implicit_vars=["rho_u", "rho_v"]).
    - ``implicit_roles`` : meme masque mais par ROLE physique ("density", "momentum_x", "energy", ...)
      au lieu du nom (cf. System.variable_roles). Union avec implicit_vars. Ex.
      adc.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"]).
    - ``newton_max_iters`` : budget d'iterations du Newton local (defaut 2 = constante historique).
    - ``newton_rel_tol`` / ``newton_abs_tol`` : critere d'arret par cellule
      ||F||_inf <= abs_tol + rel_tol*||F0||_inf (0/0 = desactive, boucle historique bit-identique).
    - ``newton_fd_eps`` : pas de la jacobienne par differences finies (defaut 1e-7 = historique).
    - ``newton_diagnostics`` : active le rapport Newton (sim.newton_report(name) -> dict
      {enabled, converged, max_residual, max_iters_used, n_failed}), agrege sur la derniere avance
      du bloc. OPT-IN : defaut False = zero cout supplementaire.

    NOMENCLATURE (audit 2026-06) : le schema cable est exactement ForwardEuler(transport sans
    source) + backward-Euler local sur la source ("SourceImplicitBE"). Ce n'est PAS une famille
    IMEX-RK / ARK (pas de choix de tableau de Butcher, ``method=`` de l'explicite ne s'applique pas
    au demi-pas IMEX) ; une vraie famille IMEXRK serait un chantier futur distinct.
    """

    kind = "imex"

    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("IMEX : substeps >= 1 (recu %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEX : stride >= 1 (recu %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEX : newton_max_iters >= 1 (recu %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEX : newton_damping dans (0, 1] (recu %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEX : newton_fail_policy 'none'|'warn'|'throw' (recu %r)"
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
    """Traitement implicite de la SOURCE raide (backward-Euler, Newton local), transport explicite.

    Nom clair pour le schema IMEX source-only : seule la SOURCE est traitee en implicite
    (backward-Euler resolu par Newton local a la cellule, via backward_euler_source /
    ImplicitSourceStepper cote C++). Le TRANSPORT reste EXPLICITE (avance par le coeur SSPRK).

    IMPORTANT -- ce n'est PAS un solveur implicite global PDE. Un solveur implicite global
    (flux + source + Poisson tous implicites, Newton-Krylov ou Schur global) est un chantier
    futur distinct. SourceImplicit = IMEX source-only (strictement equivalent a IMEX/adc.Implicit,
    numerique bit-identique).

    QUAND L'UTILISER (SourceImplicit LOCAL vs adc.CondensedSchur GLOBAL) -- ces deux mecanismes
    traitent une source raide implicitement, mais a des echelles differentes :

    - SourceImplicit est LOCAL : l'implicite ne couple que les composantes d'UNE MEME cellule
      (backward-Euler resolu par Newton a la cellule), il n'y a AUCUN couplage spatial entre
      cellules. Adapte aux termes raides purement locaux (relaxation, reactions, friction).
    - adc.CondensedSchur (via adc.Split) est GLOBAL : il assemble et resout un operateur
      elliptique tensoriel par Schur (Krylov BiCGStab) qui COUPLE tout le domaine. Adapte au
      couplage Lorentz / electrostatique raide non local (ex. Euler-Poisson magnetise du papier
      Hoffart, arXiv:2510.11808). Une source raide locale n'a PAS besoin de Schur.

    - ``substeps=N`` : sous-pas par macro-pas (cf. Explicit). Defaut 1.
    - ``stride=M`` : cadence du bloc, semantique hold-then-catch-up (cf. Explicit). Defaut 1.
    - ``implicit_vars`` / ``implicit_roles`` : masque implicite par NOM ou par ROLE physique des
      variables conservees a traiter en implicite dans le pas de source (cf. IMEX). Masque PORTE PAR
      CETTE POLITIQUE / le bloc, pas par le modele. Defauts [] = defaut modele, bit-identique.
    """

    kind = "imex"  # meme chemin C++ que IMEX (ImplicitSourceStepper)

    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("SourceImplicit : substeps >= 1 (recu %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("SourceImplicit : stride >= 1 (recu %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("SourceImplicit : newton_max_iters >= 1 (recu %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("SourceImplicit : newton_damping dans (0, 1] (recu %r)"
                             % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("SourceImplicit : newton_fail_policy 'none'|'warn'|'throw' (recu %r)"
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


# Nom PRECIS du schema cable par IMEX / SourceImplicit (audit 2026-06) : transport ForwardEuler
# sans source + backward-Euler LOCAL sur la source (Newton par cellule). Alias STRICT de
# SourceImplicit (meme objet) : a employer quand on veut nommer l'hypothese dans un script.
SourceImplicitBE = SourceImplicit


class IMEXRK:
    """Famille IMEX-RK (Implicit-Explicit Runge-Kutta), schema ARS(2,2,2), ORDRE 2.

    Schema d'Ascher-Ruuth-Spiteri (1997) : le transport hyperbolique L = -div F est traite par le
    tableau EXPLICITE, la source raide S par le tableau IMPLICITE (backward-Euler LOCAL par cellule,
    Newton, comme adc.IMEX) -- mais avec des etages couples qui montent l'ORDRE GLOBAL A 2 (transport
    ET source), la ou adc.IMEX reste un ForwardEuler(transport) + backward-Euler(source) d'ordre 1.

    Coefficients : gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma). Tableaux (stiffly accurate) :
    explicite A_E = [[0,0,0],[gamma,0,0],[delta,1-delta,0]], b_E = [delta,1-delta,0] ;
    implicite A_I = [[0,0,0],[0,gamma,0],[0,1-gamma,gamma]], b_I = [0,1-gamma,gamma].

    FAMILLE DISTINCTE de adc.IMEX (kind="imexrk_ars222" != "imex") : le defaut adc.IMEX (backward-Euler
    local, ordre 1) est INCHANGE / bit-identique. PERIMETRE : System CARTESIEN seulement -- l'AMR, le
    polaire, les modeles compiles (.so : prototype/aot/production) et les splittings Strang/Schur la
    REJETTENT explicitement (utiliser adc.IMEX / adc.Explicit sur ces chemins).

    - ``scheme`` : "ars222" (seul schema cable ; un autre nom leve une erreur explicite).
    - ``substeps=N`` : sous-pas par macro-pas (cf. adc.Explicit). Defaut 1.
    - ``stride=M`` : cadence du bloc, semantique hold-then-catch-up (cf. adc.Explicit). Defaut 1.
    - ``newton_*`` : MEMES options que adc.IMEX (max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy/
      diagnostics) -- elles parametrent les DEUX solves implicites d'etage du schema. Defauts =
      constantes historiques (max_iters=2, fd_eps=1e-7), sans cout supplementaire.

    SOURCE PLEINEMENT IMPLICITE : contrairement a adc.IMEX, IMEXRK n'expose PAS implicit_vars /
    implicit_roles (la relation de coherence d'etage ARS(2,2,2) suppose un solve homogene). Un masque
    partiel est rejete cote C++ ; pour un IMEX partiel par composante, utiliser adc.IMEX.
    """

    kind = "imexrk_ars222"

    def __init__(self, scheme="ars222", substeps=1, stride=1,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if scheme != "ars222":
            raise ValueError("IMEXRK : scheme 'ars222' (seul schema IMEX-RK cable ; recu %r)"
                             % (scheme,))
        if int(substeps) < 1:
            raise ValueError("IMEXRK : substeps >= 1 (recu %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEXRK : stride >= 1 (recu %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEXRK : newton_max_iters >= 1 (recu %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEXRK : newton_damping dans (0, 1] (recu %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEXRK : newton_fail_policy 'none'|'warn'|'throw' (recu %r)"
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
    """OBSOLETE -- utiliser adc.SourceImplicit(...) ou adc.IMEX(...) a la place.

    adc.Implicit etait un alias d'IMEX (source raide implicite via backward-Euler, transport
    explicite). Le nom "Implicit" est TROMPEUR : il suggere un solveur implicite global PDE
    (flux + source + Poisson tous implicites / Newton-Krylov), ce qui n'est PAS le cas.
    adc.SourceImplicit est le nom clair du meme schema (numerique bit-identique).

    Conserve pour la retrocompatibilite ; emet un DeprecationWarning. Utiliser :
      adc.SourceImplicit(substeps=k, stride=s)  -- nouveau nom clair
      adc.IMEX(substeps=k, stride=s)            -- acronyme officiel
    """
    import warnings
    warnings.warn(
        "adc.Implicit est obsolete : le nom est trompeur (ce n'est PAS un solveur implicite "
        "global PDE). Utiliser adc.SourceImplicit(...) (source implicite backward-Euler, "
        "transport explicite) ou adc.IMEX(...) a la place.",
        DeprecationWarning,
        stacklevel=2,
    )
    return IMEX(substeps=substeps if substeps is not None else dt_ratio, stride=stride)


class Role:
    """Roles PHYSIQUES des composantes d'un modele (cf. VariableRole cote C++ / variable_roles).

    Permet d'adresser une composante par son SENS dans adc.CondensedSchur(density=adc.Role.Density,
    momentum=(adc.Role.MomentumX, adc.Role.MomentumY), energy=adc.Role.Energy) plutot que par un nom
    litteral. Les valeurs sont les cles STABLES attendues par le C++ (role_from_name : snake_case). La
    RESOLUTION role -> composante est faite cote C++ (le bloc lit ses propres VariableRole) : ces
    constantes servent a EXPRIMER l'intention dans la formule et a valider qu'un role requis est demande.
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
    """Etage SOURCE condense par Schur (Hoffart et al., arXiv:2510.11808 ; cf.
    docs/SCHUR_CONDENSATION_DESIGN.md). NOMME l'algorithme de la source implicite couplee potentiel /
    vitesse / Lorentz et MAPPE les champs sur les roles physiques du bloc. C'est le `source=` d'une
    politique temporelle adc.Split (splitting EXPLICITE / IMPLICITE).

    kind="electrostatic_lorentz" (seul pour l'instant) selectionne ElectrostaticLorentzCondensation :
    l'etage assemble l'operateur elliptique condense A = I + theta^2 dt^2 alpha rho B^{-1}, le resout
    (BiCGStab preconditionne MG), reconstruit la vitesse v = B^{-1}(v^n - theta dt grad phi) et extrapole
    au pas plein. Tout est en C++ (CondensedSchurSourceStepper, #126) : AUCUN callback Python par cellule.

    Le bloc doit exposer les roles Density / MomentumX / MomentumY (Energy optionnel) et un champ B_z
    (set_magnetic_field) -- un role / B_z manquant leve une erreur EXPLICITE a add_equation. Marche pour
    un modele en briques natives comme pour un modele DSL compile qui declare ces roles (electrons).

    GEOMETRIE : cable en CARTESIEN (System(mesh=adc.CartesianMesh(...))) ET en POLAIRE
    (System(mesh=adc.PolarMesh(...)), anneau (r, theta), Voie A etape 2c). Le choix du stepper condense
    (cartesien CondensedSchurSourceStepper / polaire PolarCondensedSchurSourceStepper) est fait cote C++
    selon la geometrie du System : la MEME adc.CondensedSchur(...) s'utilise dans les deux cas. Le
    pendant polaire est MULTI-RANG-SUR (collectifs corrects sous MPI) mais la facade construit encore
    UNE box globale (sur le rang proprietaire) : correct et bit-identique au mono-rang, sans
    parallelisme effectif a ce niveau -- le decoupage theta facade est un suivi dedie (mise a jour
    audit 2026-06 ; l'ancienne mention "n_ranks>1 leve" etait perimee).

    QUAND L'UTILISER (CondensedSchur GLOBAL vs adc.SourceImplicit LOCAL). CondensedSchur est un
    implicite GLOBAL : il COUPLE tout le domaine via l'operateur elliptique tensoriel condense
    (resolu par Krylov BiCGStab), pour le couplage Lorentz / electrostatique raide non local. Si la
    source raide est purement LOCALE (ne couple que les composantes d'une meme cellule, sans couplage
    spatial : relaxation, reactions, friction), prendre plutot adc.SourceImplicit : c'est moins cher
    et il n'y a alors AUCUN solve elliptique a faire.

    - ``theta`` : theta-schema dans (0, 1] (0.5 = Crank-Nicolson, 1 = Euler retrograde).
    - ``alpha`` : constante de couplage electrostatique du sous-systeme source
      (d_t(-Lap phi) = -alpha div(rho v)).
    - ``density`` / ``momentum`` / ``energy`` / ``magnetic_field`` / ``potential`` : descripteurs de
      roles / champs. Ils EXPRIMENT l'intention ; la resolution role -> composante est faite cote C++
      (le bloc lit ses propres VariableRole). Tolerent adc.Role.* (recommande), un nom de role stable,
      ou un nom de variable du bloc. momentum est un couple (x, y).
    - ``krylov_tol`` / ``krylov_max_iters`` : tolerance et budget du solve Krylov (BiCGStab) de
      l'etage. None (defauts) = constantes historiques (1e-10 ; 400 en cartesien, 600 en polaire),
      rendues configurables par l'audit 2026-06 (constantes numeriques explicites).
    """

    def __init__(self, kind="electrostatic_lorentz", theta=0.5, alpha=1.0,
                 density=Role.Density, momentum=(Role.MomentumX, Role.MomentumY),
                 energy=None, magnetic_field="B_z", potential="phi",
                 krylov_tol=None, krylov_max_iters=None):
        self.krylov_tol = float(krylov_tol) if krylov_tol is not None else 0.0
        self.krylov_max_iters = int(krylov_max_iters) if krylov_max_iters is not None else 0
        if krylov_tol is not None and not (0.0 < self.krylov_tol < 1.0):
            raise ValueError("CondensedSchur : krylov_tol doit etre dans (0, 1) (recu %r)" % (krylov_tol,))
        if krylov_max_iters is not None and self.krylov_max_iters < 1:
            raise ValueError("CondensedSchur : krylov_max_iters >= 1 (recu %r)" % (krylov_max_iters,))
        if kind != "electrostatic_lorentz":
            raise ValueError(
                "CondensedSchur : kind 'electrostatic_lorentz' (seul supporte) ; recu %r" % (kind,))
        if not (0.0 < float(theta) <= 1.0):
            raise ValueError("CondensedSchur : theta doit etre dans (0, 1] (recu %r)" % (theta,))
        # momentum doit etre un couple (role_x, role_y) ; une chaine seule (iterable de caracteres)
        # est rejetee explicitement (sinon tuple("xy") donnerait deux composantes par accident).
        if isinstance(momentum, str):
            raise ValueError(
                "CondensedSchur : momentum doit etre un couple (role_x, role_y), pas une chaine (recu %r)"
                % (momentum,))
        try:
            mom = tuple(momentum)
        except TypeError:
            raise ValueError(
                "CondensedSchur : momentum doit etre un couple (role_x, role_y) (recu %r)" % (momentum,))
        if len(mom) != 2:
            raise ValueError(
                "CondensedSchur : momentum doit etre un couple (role_x, role_y) (recu %r)" % (momentum,))
        # Descripteurs roles / champs TRANSPORTES dans l'ABI C++ (audit vague 2) : density /
        # momentum / energy acceptent un adc.Role.* (nom de role stable) OU un nom de variable du
        # bloc ; la resolution role-ou-nom -> composante est faite cote C++ (set_source_stage,
        # erreur explicite si introuvable). Les DEFAUTS (roles canoniques) gardent le comportement
        # historique bit-identique. magnetic_field accepte un nom de champ aux canonique
        # (AUX_CANONICAL : "B_z", "T_e", ...) -> composante aux transportee. potential reste fige
        # a "phi" (l'etage utilise le potentiel du Poisson de systeme ; un autre champ n'aurait
        # pas de solveur derriere -> rejet explicite, pas d'ignore silencieux).
        def _spec(v):
            return "" if v is None else str(v)
        # Defauts canoniques -> chaines VIDES cote ABI (le C++ resout alors les roles canoniques,
        # chemin historique strictement inchange).
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
            self.bz_aux_component = -1  # canal canonique (defaut, bit-identique)
        else:
            from . import dsl as _dsl
            if magnetic_field not in _dsl.AUX_CANONICAL:
                raise ValueError(
                    "CondensedSchur : magnetic_field=%r inconnu (champs aux canoniques : %s)"
                    % (magnetic_field, sorted(_dsl.AUX_CANONICAL)))
            self.bz_aux_component = int(_dsl.AUX_CANONICAL[magnetic_field])
        if potential != "phi":
            raise ValueError(
                "CondensedSchur : potential=%r non configurable (l'etage source resout le "
                "potentiel phi du Poisson de systeme ; un autre champ n'aurait pas de solveur "
                "derriere) ; laisser potential='phi' (defaut)." % (potential,))
        self.kind = kind
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.density = density
        self.momentum = mom
        self.energy = energy
        self.magnetic_field = magnetic_field
        self.potential = potential

    def _has_field_overrides(self):
        """True si un descripteur non canonique est demande (AMR : rejet explicite, non cable)."""
        return bool(self.density_spec or self.momentum_x_spec or self.momentum_y_spec
                    or self.energy_spec or self.bz_aux_component >= 0)


class Split:
    """Politique temporelle SPLITTING EXPLICITE / IMPLICITE : un etage de transport hyperbolique
    EXPLICITE (adc.Explicit, SSPRK) suivi d'un etage SOURCE separe (cf. docs/SCHUR_CONDENSATION_DESIGN.md
    section 6). C'est l'OPT-IN du chantier Schur : un bloc qui n'emploie PAS adc.Split garde le chemin
    par defaut (Explicit / IMEX / SourceImplicit), BIT-IDENTIQUE.

    ::

        time=adc.Split(
            hyperbolic=adc.Explicit(ssprk3=True),
            source=adc.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, ...),
        )

    - ``hyperbolic`` : adc.Explicit (le transport ; SSPRK2/3, substeps, stride heritent de lui).
    - ``source`` : adc.CondensedSchur (l'etage source condense, joue APRES le transport). Seul backend
      source cable pour l'instant.
    """

    # kind="explicit" : le transport est joue par le chemin explicite du coeur (SSPRK), la source
    # condensee est branchee EN PLUS via set_source_stage (cf. System.add_equation). Le bloc n'est donc
    # PAS IMEX (la source raide locale backward-Euler) : sa source est l'etage condense, a part.
    def __init__(self, hyperbolic=None, source=None):
        hyperbolic = hyperbolic if hyperbolic is not None else Explicit()
        if not isinstance(hyperbolic, Explicit):
            raise TypeError(
                "Split : hyperbolic doit etre un adc.Explicit (transport explicite SSPRK) ; recu %r"
                % type(hyperbolic).__name__)
        if source is None:
            raise ValueError(
                "Split : source= est requis (l'etage source separe) ; ex. "
                "adc.Split(hyperbolic=adc.Explicit(), source=adc.CondensedSchur(...))")
        if not isinstance(source, CondensedSchur):
            raise TypeError(
                "Split : source doit etre un adc.CondensedSchur(...) (seul etage source cable) ; recu %r"
                % type(source).__name__)
        self.hyperbolic = hyperbolic
        self.source = source
        # Le transport emprunte le chemin explicite du coeur : on relaie le kind / substeps / stride de
        # l'etage hyperbolique (SSPRK2/3). La source condensee est branchee separement (add_equation).
        self.kind = hyperbolic.kind
        self.method = hyperbolic.method
        self.substeps = hyperbolic.substeps
        self.stride = hyperbolic.stride
        # Politique de splitting CABLEE au stepper de systeme (set_time_scheme). adc.Split = "lie"
        # (Godunov, 1er ordre) : H(dt) puis S(dt) une fois par macro-pas, BIT-IDENTIQUE a l'historique.
        # adc.Strang surcharge cet attribut a "strang" (cf. ci-dessous).
        self.scheme = "lie"


class Strang(Split):
    """Politique temporelle SPLITTING DE STRANG (symetrique, 2e ordre) : un macro-pas joue
    H(dt/2) ; S(dt) ; H(dt/2), ou H est le transport hyperbolique EXPLICITE (adc.Explicit, SSPRK)
    et S l'etage SOURCE separe (adc.CondensedSchur). C'est l'extension 2e ordre de adc.Split (Lie /
    Godunov, 1er ordre) : memes briques (transport SSPRK + etage source condense), seul l'ORDRE et la
    cadence des resolutions de champ changent.

    ::

        time=adc.Strang(
            hyperbolic=adc.Explicit(ssprk3=True),
            source=adc.CondensedSchur(theta=0.5, alpha=alpha),
        )

    Le stepper de systeme RE-RESOUT solve_fields ENTRE les etages (avant chaque demi-avance et avant
    la source) pour que le transport lise toujours un phi coherent avec la densite courante (le
    solve_fields UNIQUE de tete, suffisant pour Lie ou une seule avance de transport suit, ne suffit
    pas a la 2nde demi-avance Strang). cf. docs/HOFFART_STEP_SEQUENCE.md et SystemStepper::step_strang.

    ``hyperbolic`` / ``source`` : identiques a adc.Split. Cable par add_equation (qui branche l'etage
    source ET appelle set_time_scheme('strang') sur le System)."""

    def __init__(self, hyperbolic=None, source=None):
        super().__init__(hyperbolic=hyperbolic, source=source)
        self.scheme = "strang"


class System:
    """Le systeme/coupleur : compose des blocs, partage un Poisson, avance le tout.

    add_block prend un modele compose (adc.Model(...)) + des objets Spatial / Explicit / IMEX.
    Tout le reste (set_poisson, set_density, step, step_cfl, step_adaptive, diagnostics,
    primitives eval_rhs/get_state/set_state) est transmis a la facade compilee.

    GEOMETRIE : le choix vit dans un objet MAILLAGE passe en mesh= (adc.CartesianMesh / adc.PolarMesh),
    PAS dans le schema (adc.FiniteVolume reste reconstruction + Riemann + variables). Defaut (mesh=None
    ou adc.CartesianMesh) = domaine carre, bit-identique a l'historique. adc.PolarMesh (anneau global)
    est BRANCHE dans System.step (Phase 2b) : transport ExB polaire + Poisson polaire + aux en base
    locale (e_r, e_theta). Limites : transport ExB scalaire, mono-rang, pas de couplage cart<->polaire."""

    def __init__(self, config=None, mesh=None, **cfg_kw):
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # Le maillage (s'il est fourni) porte le CHOIX de geometrie et ecrase les champs correspondants
        # de la config. Applique APRES cfg_kw : mesh= prevaut sur les n=/L= passes en mots-cles.
        if mesh is not None:
            if not hasattr(mesh, "_apply"):
                raise TypeError("System : mesh doit etre un adc.CartesianMesh / adc.PolarMesh (recu %r)"
                                % type(mesh).__name__)
            mesh._apply(config)
        # Marque l'init Kokkos comme imminente : _System(config) alloue des Fabs -> Kokkos s'initialise
        # (lazy) ici. Apres ce point, adc.set_threads n'a plus d'effet (averti par set_threads).
        global _first_system_built
        _first_system_built = True
        self._s = _System(config)  # geometry == 'polar' construit un anneau global (Phase 2b, cf. PolarMesh)
        # Table des champs aux NOMMES par bloc (ADC-70 phase 1) : bloc -> {nom: composante canonique}.
        # Remplie par add_equation depuis CompiledModel.aux_extra_names (la composante du k-ieme nom =
        # dsl.AUX_NAMED_BASE + k). C'est la FACADE qui detient les noms : le C++ ne manipule que des
        # indices de composante (set_aux_field_component / aux_field_component). Vide pour un bloc sans
        # champ aux nomme. cf. set_aux_field / aux_field.
        self._aux_field_index = {}

    def add_block(self, name, model, spatial=None, time=None, evolve=True):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # adc.Split (etage source condense) n'est cable que par add_equation (qui branche
        # set_source_stage apres l'ajout du bloc) : le rejeter ICI plutot que de jouer le seul transport
        # en silence (la source condensee serait perdue).
        if isinstance(time, Split):
            raise TypeError(
                "System.add_block : adc.Split (etage source condense par Schur) n'est supporte que par "
                "add_equation (qui branche l'etage source) ; utiliser add_equation(..., time=adc.Split(...)).")
        # Masque implicite + options Newton portes par la politique temporelle (IMEX/SourceImplicit) ;
        # defauts neutres sur les autres politiques (Explicit). Resolus/valides cote C++
        # (System::add_block) contre les noms/roles du bloc.
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), evolve, getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", 2),
                          getattr(time, "newton_rel_tol", 0.0),
                          getattr(time, "newton_abs_tol", 0.0),
                          getattr(time, "newton_fd_eps", 1e-7),
                          getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"))

    def add_equation(self, name, model, spatial=None, time=None, substeps=None, names=None,
                     evolve=True, stride=None):
        """Ajoute une equation/bloc en aiguillant sur le TYPE de @p model (DSL Phase A) :

        - un ModelSpec (adc.Model(...)) -> add_block (briques natives composees) ;
        - un CompiledModel (m.compile(...)) -> l'adder du backend (add_dynamic_block pour prototype,
          add_compiled_block pour aot, add_native_block pour production), avec les noms/roles/gamma
          transportes par le .so.

        Centralise le couplage backend <-> adder (un .so AOT ne doit pas etre branche sur
        add_dynamic_block, et inversement). cf. docs/DSL_MODEL_DESIGN.md section 3.

        @p spatial : adc.FiniteVolume(...) / adc.Spatial(...) (defaut minmod+rusanov+conservatif).
        @p time : adc.Explicit / IMEX (defaut Explicit). @p substeps : surcharge time.substeps.
        @p stride : surcharge time.stride (1 = chaque macro-pas, defaut bit-identique).
        @p names : noms des composantes (longueur = n_vars du modele compile). @p evolve : bloc avance ;
        evolve=False (champ gele) n'est cable que sur le chemin natif (ModelSpec -> add_block, backend
        'production' -> add_native_block). Sur backend 'prototype'/'aot' (l'ABI .so ne transporte pas
        evolve) un evolve=False est REJETE explicitement -> utiliser un bloc natif (add_background).
        """
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # --- adc.Split (Lie) / adc.Strang (2e ordre) : splitting EXPLICITE / IMPLICITE, OPT-IN Schur --
        # Le bloc est d'abord ajoute avec l'etage HYPERBOLIQUE explicite (chemin de production existant,
        # aucune duplication d'aiguillage), PUIS on branche l'etage SOURCE condense (set_source_stage,
        # C++). La source est jouee APRES le transport a chaque pas. Le defaut (sans Split) est inchange.
        # La POLITIQUE de splitting (Lie / Strang) est CABLEE au stepper de systeme via set_time_scheme :
        # adc.Split -> "lie" (defaut, bit-identique), adc.Strang -> "strang" (H(dt/2) S(dt) H(dt/2)).
        if isinstance(time, Split):
            self.add_equation(name, model, spatial=spatial, time=time.hyperbolic,
                              substeps=substeps, names=names, evolve=evolve, stride=stride)
            src = time.source
            self._s.set_source_stage(name, src.kind, src.theta, src.alpha,
                                     getattr(src, "krylov_tol", 0.0),
                                     getattr(src, "krylov_max_iters", 0),
                                     getattr(src, "density_spec", ""),
                                     getattr(src, "momentum_x_spec", ""),
                                     getattr(src, "momentum_y_spec", ""),
                                     getattr(src, "energy_spec", ""),
                                     getattr(src, "bz_aux_component", -1))
            self._s.set_time_scheme(time.scheme)  # "lie" (Split) ou "strang" (Strang)
            return

        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)
        nstride = stride if stride is not None else getattr(time, "stride", 1)

        # --- ModelSpec : briques natives composees -> add_block (chemin existant) ---
        # NB : on appelle _s.add_block DIRECTEMENT avec nsub/nstride (pas self.add_block, dont la
        # signature n'a pas de substeps -> elle utiliserait time.substeps et IGNORERAIT les overrides).
        if isinstance(model, ModelSpec):
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, evolve, nstride,
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", 2),
                              getattr(time, "newton_rel_tol", 0.0),
                              getattr(time, "newton_abs_tol", 0.0),
                              getattr(time, "newton_fd_eps", 1e-7),
                              getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"))
            return

        # Masque implicite (IMEX) : seul le chemin natif compose (ModelSpec -> add_block) le cable. Les
        # backends compiles (.so : dynamic/aot/production) n'exposent pas l'argument -> on REJETTE un
        # masque non vide plutot que l'ignorer en silence (cf. le rejet du stride sur backend 'aot').
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "add_equation : implicit_vars / implicit_roles (masque IMEX par bloc) ne sont supportes "
                "que sur un modele compose adc.Model(...) (-> add_block). Le modele compile (.so) ne "
                "transporte pas le masque ; utiliser un adc.Model(...) natif.")
        # Memes regles pour les options/diagnostics Newton (IMEX) : non transportees par l'ABI des .so.
        # Des valeurs non-defaut seraient ignorees EN SILENCE -> rejet explicite.
        if (getattr(time, "newton_max_iters", 2) != 2
                or getattr(time, "newton_rel_tol", 0.0) != 0.0
                or getattr(time, "newton_abs_tol", 0.0) != 0.0
                or getattr(time, "newton_fd_eps", 1e-7) != 1e-7
                or getattr(time, "newton_diagnostics", False)
                or getattr(time, "newton_damping", 1.0) != 1.0
                or getattr(time, "newton_fail_policy", "none") != "none"):
            raise ValueError(
                "add_equation : les options Newton (newton_max_iters/rel_tol/abs_tol/fd_eps/"
                "diagnostics/damping/fail_policy) ne sont supportees que sur un modele compose "
                "adc.Model(...) (-> add_block). L'ABI du modele compile (.so) ne les transporte "
                "pas ; utiliser un adc.Model(...) natif.")

        if not isinstance(model, dsl.CompiledModel):
            raise TypeError("add_equation : model doit etre un adc.Model(...) (ModelSpec) ou un "
                            "CompiledModel (m.compile(...)) ; recu %r" % type(model).__name__)

        compiled = model
        # Garde-fou noms : longueur explicite verifiee tot (le C++ leve aussi, mais on diagnostique ici).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError("add_equation : names= a %d noms mais le bloc '%s' a %d variables"
                             % (len(names), name, compiled.n_vars))
        names_arg = list(names) if names is not None else []

        # Champs aux NOMMES (ADC-70 phase 1) : table nom -> composante du bloc, depuis les noms ORDONNES
        # du modele compile (le k-ieme nom = composante dsl.AUX_NAMED_BASE + k, miroir de l'emission C++).
        # Consommee par set_aux_field / aux_field. add_compiled_block / add_native_block / add_dynamic_block
        # ont deja elargi le canal aux (adc_compiled_naux -> ensure_aux_width), donc la composante existe.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        self._aux_field_index[name] = {nm: dsl.AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

        backend = compiled.backend
        # Garde-fou flux numerique : HLLC/Roe exigent une pression -> la brique generee n'emet
        # pressure()/wave_speeds() que si une primitive 'p' est declaree. Sans 'p', make_block ne
        # compile pas le flux : on le diagnostique ici avant la frontiere C++.
        # hllc / roe : la capability emise (m.enable_hllc -> has_hllc, m.enable_roe -> has_roe)
        # OUVRE le flux meme hors Euler 4-var (la garde C++ requires l'accepte) ; sinon la voie
        # canonique exige 'p' dans les primitives.
        if (spatial.flux in ("hllc", "roe") and "p" not in compiled.prim_names
                and not (spatial.flux == "hllc" and getattr(compiled, "has_hllc", False))
                and not (spatial.flux == "roe" and getattr(compiled, "has_roe", False))):
            raise ValueError(
                "add_equation : riemann '%s' exige une pression : declarer une primitive 'p' "
                "(m.primitive('p', ...)) dans le modele, ou emettre la capability "
                "(m.enable_hllc() / m.enable_roe()) ; sinon utiliser riemann='rusanov'"
                % spatial.flux)
        # HLL : PAS de garde Python sur prim_names ici -- la brique generee emet wave_speeds des
        # qu'une primitive 'p' est DECLAREE (m.primitive('p', ...)), meme HORS layout primitive_vars
        # (cas isotherme 3-var Hoffart : prim_names = rho/u/v sans 'p', HLL pourtant disponible).
        # Le requires-gate C++ de make_block rejette deja avec le remede exact quand wave_speeds
        # manque ("declarer une primitive 'p' / des eigenvalues").

        # Aiguillage AUTORITAIRE par l'adder du CompiledModel (fixe par le backend, cf. dsl._BACKENDS) :
        # prototype -> add_dynamic_block, aot -> add_compiled_block, production -> add_native_block (#85).
        adder = compiled.adder
        if adder == "add_dynamic_block":
            # JIT, residu HOTE Rusanov ordre 1 : ne prend que le LIMITER MUSCL (none/minmod/vanleer)
            # + substeps ; pas de flux HLLC/Roe, pas de recon primitif. WENO5 (stencil 5 points) n'est
            # PAS un limiteur MUSCL et ce chemin n'execute pas assemble_rhs : on le rejette ICI (les
            # chemins aot/production, eux, acceptent weno5 -- la grille .so / le bloc natif allouent
            # block_n_ghost(limiter) = 3 ghosts).
            if spatial.limiter == "weno5":
                raise ValueError(
                    "add_equation : limiter 'weno5' non supporte sur backend 'prototype' (JIT, residu "
                    "hote Rusanov ordre 1, sans assemble_rhs) ; utiliser backend='aot'/'production' "
                    "(WENO5 cable de bout en bout) ou add_block (modele compose adc.Model(...)).")
            if spatial.flux != "rusanov":
                raise ValueError(
                    "add_equation : backend 'prototype' (JIT, residu hote Rusanov ordre 1) n'expose "
                    "que riemann='rusanov' (recu '%s') ; utiliser backend='aot'/'production' pour "
                    "HLLC/Roe" % spatial.flux)
            # evolve=False (bloc GELE / fond fixe) n'est PAS cable : l'ABI add_dynamic_block ne
            # transporte pas evolve (push_dynamic le force a true cote C++) -> le bloc serait avance
            # en SILENCE. On le REJETTE (rejet plutot qu'ignore silencieux). Pour un champ gele,
            # utiliser un bloc natif/production (add_background -> add_block(..., evolve=False)).
            if not evolve:
                raise ValueError(
                    "add_equation : evolve=False non supporte sur backend 'prototype' (l'ABI du .so JIT "
                    "ne transporte pas evolve ; le bloc serait avance en silence). Utiliser un modele "
                    "natif compose adc.Model(...) -> add_block(..., evolve=False) (ou add_background) "
                    "pour un champ gele.")
            self._s.add_dynamic_block(name, compiled.so_path, nsub, names_arg, spatial.limiter)
            return
        if adder == "add_compiled_block":
            # AOT host-marshale : limiter x riemann x recon, mono-rang (sans MPI/AMR). L'ABI extern "C"
            # du .so AOT (add_compiled_block) NE transporte PAS de cadence : le bloc tournerait a stride=1
            # en SILENCE. On REJETTE donc stride > 1 sur ce backend (route explicite) plutot que de
            # l'ignorer. Le stride par bloc est cable sur add_block (natif compose) et add_native_block
            # (backend='production'). On lit time.stride ET l'override stride= (nstride couvre les deux).
            if nstride != 1:
                raise ValueError(
                    "add_equation : stride=%d non supporte sur backend 'aot' (l'ABI du .so AOT ne "
                    "transporte pas la cadence ; le bloc tournerait a stride=1 en silence). Utiliser "
                    "backend='production' (chemin natif, cadence cablee) ou un modele natif compose "
                    "adc.Model(...) -> add_block." % nstride)
            # evolve=False (bloc GELE / fond fixe) n'est PAS cable : l'ABI add_compiled_block ne
            # transporte pas evolve (add_compiled_block le force a true cote C++) -> le bloc serait
            # avance en SILENCE. On le REJETTE (rejet plutot qu'ignore silencieux). Pour un champ gele,
            # utiliser backend='production' (add_native_block transporte evolve) ou un modele natif
            # compose adc.Model(...) -> add_block(..., evolve=False) (ou add_background).
            if not evolve:
                raise ValueError(
                    "add_equation : evolve=False non supporte sur backend 'aot' (l'ABI du .so AOT ne "
                    "transporte pas evolve ; le bloc serait avance en silence). Utiliser "
                    "backend='production' (chemin natif, evolve cable) ou un modele natif compose "
                    "adc.Model(...) -> add_block(..., evolve=False) (ou add_background) pour un champ gele.")
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
            # Garde PRE-DLOPEN au branchement : couvre AUSSI le cache HIT (ou compile_native ne tourne
            # pas) -- un module _adc perime donnerait sinon un dlopen 'symbol not found' cryptique.
            dsl.check_compiled_matches_module(getattr(compiled, "abi_key", ""))
            gamma = compiled.gamma if compiled.gamma is not None else 1.4
            self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                     spatial.recon, time.kind, gamma, nsub, evolve, nstride)
            return
        raise ValueError("add_equation : adder %r inconnu (backend %r)" % (adder, backend))

    def _resolve_aux_field(self, block, name):
        """Resout (bloc, nom de champ aux NOMME) -> composante canonique du canal aux (ADC-70 phase 1).
        Regle de resolution : un nom CANONIQUE (phi/grad/B_z/T_e) est REJETE ici -- ces champs ont
        leurs chemins dedies (B_z -> set_magnetic_field, T_e -> set_electron_temperature_from, phi/grad
        derives par solve_fields). Sinon on cherche dans la table du bloc (remplie a add_equation depuis
        le modele compile). Leve ValueError avec un message actionnable sur bloc/nom inconnu."""
        from . import dsl  # import tardif (cycle dsl <-> __init__)
        if name == "B_z":
            raise ValueError(
                "set_aux_field : 'B_z' (champ magnetique) se fixe via sim.set_magnetic_field(Bz), "
                "PAS via set_aux_field (B_z est un champ aux canonique, pas un champ nomme).")
        if name == "T_e":
            raise ValueError(
                "set_aux_field : 'T_e' (temperature electronique) est DERIVE d'un bloc fluide via "
                "sim.set_electron_temperature_from(bloc), PAS fixe via set_aux_field.")
        if name in dsl.AUX_CANONICAL:
            raise ValueError(
                "set_aux_field : '%s' est un champ aux CANONIQUE (derive par le solveur, non fixable) ; "
                "set_aux_field ne porte que les champs NOMMES declares par m.aux_field(...)." % name)
        table = self._aux_field_index.get(block)
        if table is None:
            raise ValueError(
                "set_aux_field : bloc '%s' inconnu (ou ajoute sans champ aux nomme) ; ajouter le bloc "
                "via add_equation(model=...) avec un modele declarant m.aux_field('%s')." % (block, name))
        if name not in table:
            known = sorted(table) if table else "(aucun)"
            raise ValueError(
                "set_aux_field : champ aux '%s' non declare par le bloc '%s' ; champs nommes connus : %s"
                % (name, block, known))
        return table[name]

    def set_aux_field(self, block, name, field):
        """Fixe un champ aux NOMME (ADC-70 phase 1) d'un bloc : @p name doit avoir ete declare par le
        modele via m.aux_field(name) (et le bloc ajoute par add_equation). @p field : tableau 2D (ny, nx)
        ou plat (n*n), row-major. Le champ est STATIQUE (fourni par l'utilisateur, comme B_z) et PERSISTE
        d'un pas a l'autre (solve_fields ne reecrit jamais les composantes nommees). Pour B_z / T_e,
        utiliser leurs chemins dedies (set_magnetic_field / set_electron_temperature_from)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr.reshape(-1))

    def aux_field(self, block, name):
        """Lit un champ aux NOMME (ADC-70 phase 1) d'un bloc -> tableau 2D (ny, nx). Vaut 0 partout tant
        qu'aucun set_aux_field ne l'a ecrit (canal aux initialise a zero, jamais reecrit par solve_fields
        au-dela des composantes derivees). @p name : declare par m.aux_field(name)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        return np.asarray(self._s.aux_field_component(comp), dtype=float)

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

    def set_disc_domain(self, cx, cy, R, mode="none"):
        """Fixe le DOMAINE DE TRANSPORT comme un DISQUE de centre (cx, cy) et de rayon R, et CABLE le
        transport selon mode= (chantiers T2 / T5-PR3). Materialise un masque 0/1 cellule-centre (cellule
        active quand son centre est dans le disque, level set hypot(x-cx, y-cy) - R < 0, MEME convention
        que le mur conducteur du Poisson). C'est le pendant volumes-finis du mur elliptique : le papier
        (Hoffart et al., arXiv:2510.11808) transporte sur un VRAI disque alors qu'ADC transporte sur le
        carre cartesien plein, le cercle n'agissant que dans la paroi de Poisson (verrou des bords
        d'anneau cartesiens, cf. docs/HOFFART_FIDELITY.md).

        Le parametre ``mode`` cable le transport :

        - 'none' (defaut) : le masque est materialise (consultable via disc_mask()) mais le transport
          reste PLEIN cartesien (assemble_rhs) -> step() BIT-IDENTIQUE meme avec le disque pose ;
        - 'staircase' : transport masque conservatif (assemble_rhs_masked, porte de face 0/1) ;
        - 'cutcell' : transport cut-cell / embedded-boundary (assemble_rhs_eb, apertures alpha_f +
          fraction de volume kappa, frontiere lisse, ordre 2 a l'interieur du disque).

        Le mode est honore sous Lie ET Strang (cf. Split / Strang). R > 0 ; cartesien seulement (le
        polaire borne deja l'anneau par ses parois radiales -> erreur explicite)."""
        self._s.set_disc_domain(cx, cy, R, mode)

    def set_geometry_mode(self, mode):
        """Bascule SEULE le mode de transport disque ('none'|'staircase'|'cutcell') sans (re)definir le
        disque. Un mode != 'none' exige un disque deja fixe (set_disc_domain) -> erreur sinon. Remettre
        a 'none' restaure le transport plein cartesien (bit-identique)."""
        self._s.set_geometry_mode(mode)

    def disc_mask(self):
        """Masque de domaine 0/1 cellule-centre, tableau (ny, nx) (diagnostic / verification du
        contrat). Tout 1.0 tant que set_disc_domain n'a pas ete appele (sous-domaine = domaine
        entier, chemin par defaut)."""
        return self._s.disc_mask()

    def add_coupling(self, coupling):
        """Ajoute un couplage inter-especes (operator-split, applique apres le transport) :

        - objet NOMME adc.Ionization / Collision / ThermalExchange -> formule figee
          (add_ionization / add_collision / add_thermal_exchange) ;
        - CompiledCoupledSource (adc.dsl.CoupledSource(...).compile(...)) -> source GENERIQUE decrite en
          formules, transportee en bytecode et interpretee cote C++ (System.add_coupled_source ; aucun
          callback Python par cellule, MPI-safe)."""
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        if isinstance(coupling, dsl.CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       # Frequence PAR CELLULE mu(U) (vides = constante seule, cf.
                                       # CoupledSource.frequency(Expr)). Forwardes a la frontiere C++.
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        elif isinstance(coupling, Ionization):
            self.add_ionization(electron=coupling.electron, ion=coupling.ion,
                                neutral=coupling.neutral, rate=coupling.rate)
        elif isinstance(coupling, Collision):
            self.add_collision(coupling.a, coupling.b, coupling.rate)
        elif isinstance(coupling, ThermalExchange):
            self.add_thermal_exchange(coupling.a, coupling.b, coupling.rate)
        else:
            raise TypeError("add_coupling attend adc.Ionization / Collision / ThermalExchange ou un "
                            "CompiledCoupledSource (adc.dsl.CoupledSource(...).compile(...))")

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

    def set_primitive_state(self, name, **prims):
        """Initialise un bloc depuis ses variables PRIMITIVES, nommees (rho/u/v/p ...) :

            sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)

        Chaque primitive est un tableau (n, n). Les noms attendus sont ceux de
        variable_names(name, "primitive") (l'ordre du modele du bloc). Le tableau (ncomp, n, n) est
        assemble dans cet ordre, puis CONVERTI en variables conservatives par le modele du bloc (cote
        C++ : compressible E = p/(g-1) + 1/2 rho|v|^2 ; isotherme rho u ; scalaire identite) et ecrit
        dans l'etat. Pendant ergonomique de set_density (qui ne pose que la densite, reste au repos).

        Leve une erreur claire si un nom de primitive est inconnu pour le bloc, ou s'il en manque."""
        import numpy as np  # local : numpy n'est requis que pour cet assemblage host

        names = list(self._s.variable_names(name, "primitive"))
        n = self.nx()
        unknown = [k for k in prims if k not in names]
        if unknown:
            raise ValueError(
                "set_primitive_state : primitive(s) inconnue(s) %r pour le bloc '%s' ; "
                "primitives attendues : %r" % (unknown, name, names))
        missing = [k for k in names if k not in prims]
        if missing:
            raise ValueError(
                "set_primitive_state : primitive(s) manquante(s) %r pour le bloc '%s' ; "
                "fournir toutes les primitives : %r" % (missing, name, names))
        # Assemble (ncomp, n, n) dans l'ORDRE du modele (primitive_vars), pas l'ordre des kwargs.
        prim = np.empty((len(names), n, n), dtype=np.float64)
        for c, nm in enumerate(names):
            arr = np.asarray(prims[nm], dtype=np.float64)
            if arr.shape != (n, n):
                raise ValueError(
                    "set_primitive_state : primitive '%s' de forme %r, attendu (%d, %d)"
                    % (nm, tuple(arr.shape), n, n))
            prim[c] = arr
        self._s.set_primitive_state(name, prim)

    def get_primitive_state(self, name):
        """Lit l'etat conservatif d'un bloc et le rend en variables PRIMITIVES (diagnostic) :

            P = sim.get_primitive_state("electrons")   # {"rho": ..., "u": ..., "v": ..., "p": ...}

        Renvoie un dict {nom_primitive: tableau (n, n)} dans l'ordre de variable_names(name,
        "primitive"). Inverse de set_primitive_state (round-trip exact a la precision machine, la
        conversion cons <-> prim du modele etant consistante)."""
        names = list(self._s.variable_names(name, "primitive"))
        prim = self._s.get_primitive_state(name)  # (ncomp, n, n)
        return {nm: prim[c] for c, nm in enumerate(names)}

    def check_model(self, block, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """Verification RUNTIME generique d'un bloc installe (audit 2026-06, chantier 6) : controle
        sur l'ETAT COURANT du bloc (quel que soit le backend : natif compose, .so JIT/AOT/production) :

        - etat U fini ;
        - residu -div F + S fini (exerce flux + source + reconstruction de bout en bout) ;
        - positivite des composantes a role Density (via variable_roles) ;
        - positivite de la primitive de role Pressure / nommee 'p' (via get_primitive_state) ;
        - round-trip cons -> prim -> cons ~= identite (conversions du modele coherentes ;
          l'etat est SAUVE puis RESTAURE, le bloc n'est pas modifie).

        Pendant RUNTIME de dsl.Model.check_model (qui verifie les FORMULES avant compilation).
        @return dict {"ok", "failures", "block"} ; raise_on_error=True (defaut) leve ValueError."""
        import numpy as np
        failures = []
        nv = self._s.n_vars(block)
        U = np.asarray(self._s.get_state(block), dtype=float)
        if not np.all(np.isfinite(U)):
            failures.append("etat U non fini")
        self._s.solve_fields()  # aux a jour : le residu lit phi / grad phi
        R = np.asarray(self._s.eval_rhs(block), dtype=float)
        if not np.all(np.isfinite(R)):
            failures.append("residu -div F + S non fini (flux/source/reconstruction)")
        ncell = U.size // max(nv, 1)
        Uc = U.reshape(nv, ncell)
        roles = [r.lower() for r in self._s.variable_roles(block, "conservative")]
        names = list(self._s.variable_names(block, "conservative"))
        for i, r in enumerate(roles):
            if r == "density" and not bool(np.all(Uc[i] > 0)):
                failures.append("composante '%s' (role Density) non strictement positive" % names[i])
        prim_roles = [r.lower() for r in self._s.variable_roles(block, "primitive")]
        prim_names = list(self._s.variable_names(block, "primitive"))
        try:
            P = np.asarray(self._s.get_primitive_state(block), dtype=float)
            if not np.all(np.isfinite(P)):
                failures.append("etat primitif non fini (to_primitive)")
            else:
                for i, (r, nm) in enumerate(zip(prim_roles, prim_names)):
                    if (r == "pressure" or nm == "p") and not bool(np.all(P[i] > 0)):
                        failures.append("primitive '%s' (pression) non strictement positive" % nm)
                # round-trip cons -> prim -> cons : etat sauve puis restaure (aucune mutation nette).
                U0 = U.copy()
                self._s.set_primitive_state(block, P)
                U1 = np.asarray(self._s.get_state(block), dtype=float)
                self._s.set_state(block, U0)
                if not np.allclose(U1, U0, rtol=rtol, atol=atol):
                    err = float(np.max(np.abs(U1 - U0)))
                    failures.append("round-trip to_conservative(to_primitive(U)) != U "
                                    "(ecart max %g : conversions du modele incoherentes)" % err)
        except RuntimeError as ex:  # bloc sans conversions (chemins .so anterieurs) : on le signale
            failures.append("conversions cons<->prim indisponibles sur ce bloc (%s)" % ex)
        report = {"ok": not failures, "failures": failures, "block": block}
        if failures and raise_on_error:
            raise ValueError("System.check_model('%s') : %d echec(s) :\n  - %s"
                             % (block, len(failures), "\n  - ".join(failures)))
        return report

    # ------------------------------------------------------------------
    # SORTIES / CHECKPOINT / RESTART v1 (audit 2026-06, IO ; cf. docs/IO_CHECKPOINT_PLAN.md).
    # Python pur (zero changement du chemin chaud C++), mono-rang ; HDF5 agrege/parallele et AMR =
    # PR-IO-3. Ecriture ATOMIQUE (fichier .tmp puis os.replace : un crash en cours d'ecriture ne
    # corrompt jamais un checkpoint precedent).
    # ------------------------------------------------------------------
    def write(self, path, format="vtk", step=None, fields=None, parallel=False):
        """SORTIE DE VISUALISATION : ecrit l'etat courant dans un fichier ouvert (ParaView/numpy).

        - ``format="vtk"`` : ImageData .vti ASCII (cartesien ; ouvert par ParaView / VisIt) -- une
          CellData par variable conservative de chaque bloc + le potentiel phi.
        - ``format="npz"`` : np.savez compresse (tout backend / toute geometrie) -- etats par bloc,
          noms/roles, phi, t, macro_step, grille.
        @p step : suffixe numerote (path_000123.vti) ; None = path brut + extension.
        @p fields : sous-ensemble de blocs a ecrire (None = tous).
        @p parallel : ecriture HDF5 PARALLELE par hyperslabs (opt-in, format='hdf5' SEULEMENT). Defaut
          False = chemin gather rang-0 ci-dessous, STRICTEMENT inchange. True = chaque rang ecrit SES
          boites dans un fichier unique via h5py(mpio) -- exige h5py compile MPI + mpi4py (sinon erreur
          CLAIRE avec remede, jamais d'ecriture silencieuse degradee). Cf. _write_hdf5_parallel.
        @return le chemin ecrit.

        MULTI-RANGS (MPI np>1) : les champs sont rassembles via les accesseurs GLOBAUX collectifs
        (state_global / potential_global -- chaque rang DOIT donc appeler write), puis SEUL le rang 0
        ecrit le fichier (un fichier unique, identique au mono-rang). Le System etant mono-box (une
        box couvrant tout le domaine, sur le rang 0), le gather est exact. Les autres rangs rendent le
        chemin sans I/O. HDF5 PARALLELE (hyperslabs par rang) : parallel=True (cf. _write_hdf5_parallel ;
        vrai parallelisme seulement en MULTI-BOX, le System cartesien etant mono-box)."""
        import os
        import numpy as np
        from . import _adc
        if parallel and format != "hdf5":
            raise ValueError(
                "write : parallel=True n'est supporte que pour format='hdf5' (ecriture par "
                "hyperslabs) ; format=%r passe par le chemin gather rang-0 (parallel=False)."
                % (format,))
        rank0 = (_adc.my_rank() == 0)
        blocks = [b for b in self._s.block_names() if fields is None or b in fields]
        suffix = ("_%06d" % int(step)) if step is not None else ""
        nxv, nyv = self._s.nx(), self._s.ny()
        if format == "npz":
            # Gather COLLECTIF (tous les rangs) AVANT la garde rang-0 : state_global / potential_global
            # font un all_reduce interne et doivent etre appeles par chaque rang.
            out = {"t": self._s.time(), "macro_step": self._s.macro_step(),
                   "nx": nxv, "ny": nyv, "blocks": np.array(blocks)}
            for b in blocks:
                nv = self._s.n_vars(b)
                out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                    nv, nyv, nxv)
                out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
                out["roles_" + b] = np.array(list(self._s.variable_roles(b, "conservative")))
            out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".npz"
            if not rank0:
                return target  # seul le rang 0 ecrit le fichier (gather deja fait collectivement)
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, names = [], []
            for b in blocks:
                nv = self._s.n_vars(b)
                st = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(nv, nyv, nxv)
                for c, nm in enumerate(self._s.variable_names(b, "conservative")):
                    arrays.append(st[c]); names.append("%s_%s" % (b, nm))
            arrays.append(np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv))
            names.append("phi")
            if not rank0:
                return target  # gather collectif fait ci-dessus ; seul le rang 0 ecrit
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (nxv, nyv, 1.0 / nxv, 1.0 / nyv),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (nxv, nyv),
                     '      <CellData>']
            for nm, arr in zip(names, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        if format == "hdf5":
            if parallel:
                # HDF5 PARALLELE par hyperslabs (PR-IO-3, opt-in) : chaque rang ecrit SES boites dans
                # un fichier unique (h5py mpio), pas de gather global. Chemin SEPARE -- le chemin serie
                # ci-dessous reste STRICTEMENT inchange.
                return self._write_hdf5_parallel(path + suffix + ".h5", blocks, nxv, nyv)
            # HDF5 AGREGE v1 (vague 3, PR-IO-2 du plan) : un fichier unique, un groupe par bloc,
            # attributs pour l'horloge/grille. Multi-rangs : gather collectif (state_global /
            # potential_global) puis ecriture rang 0 (un fichier unique). HDF5 PARALLELE (hyperslabs
            # par rang) = parallel=True (branche ci-dessus). h5py optionnel : absent -> erreur claire.
            # Gather COLLECTIF (tous rangs) AVANT la garde rang-0.
            states = {b: np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                self._s.n_vars(b), nyv, nxv) for b in blocks}
            phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".h5"
            if not rank0:
                return target  # seul le rang 0 ecrit le fichier
            try:
                import h5py
            except ImportError:
                raise RuntimeError(
                    "write(format='hdf5') : h5py absent (pip/conda install h5py) ; "
                    "utiliser format='npz' (equivalent, sans dependance) en attendant.")
            tmp = target + ".tmp"
            with h5py.File(tmp, "w") as f:
                f.attrs["t"] = self._s.time()
                f.attrs["macro_step"] = self._s.macro_step()
                f.attrs["nx"] = nxv
                f.attrs["ny"] = nyv
                f.attrs["abi_key"] = abi_key()
                for b in blocks:
                    g = f.create_group(b)
                    g.create_dataset("state", data=states[b], compression="gzip")
                    g.attrs["names"] = [s.encode() for s in
                                        self._s.variable_names(b, "conservative")]
                    g.attrs["roles"] = [s.encode() for s in
                                        self._s.variable_roles(b, "conservative")]
                f.create_dataset("phi", data=phi_g, compression="gzip")
            os.replace(tmp, target)
            return target
        raise ValueError("write : format 'vtk' | 'npz' | 'hdf5' (recu %r)" % (format,))

    def _write_hdf5_parallel(self, target, blocks, nxv, nyv):
        """ECRITURE HDF5 PARALLELE par hyperslabs (write(format='hdf5', parallel=True)) -- PR-IO-3.

        CHEMIN OPT-IN, separe du chemin serie (gather rang-0) qui reste intouche. Au lieu de rassembler
        tout le champ sur le rang 0, chaque rang ECRIT SES BOITES dans un fichier UNIQUE ouvert en
        collectif (h5py driver='mpio'). Les datasets globaux (ncomp, ny, nx) par bloc + phi (ny, nx)
        sont crees COLLECTIVEMENT, les metadonnees (t, macro_step, nx, ny, abi_key, noms/roles) ecrites
        collectivement, puis chaque rang ecrit ses hyperslabs dset[:, jlo:jhi+1, ilo:ihi+1] en I/O
        INDEPENDANTE (boites disjointes ; un rang sans box n'ecrit rien).

        VRAI PARALLELISME = MULTI-BOX seulement. Le System cartesien est MONO-BOX (une box couvrant le
        domaine, sur le rang 0) : sous np>1 le rang 0 ecrit l'unique box et les autres rangs ne portent
        aucune box -- le gain hyperslab apparait sur une geometrie multi-box (cf. AMR, ADC-65). La
        mecanique reste CORRECTE dans le cas general (iteration sur tous les fabs locaux). phi est
        resolu/rassemble COLLECTIVEMENT (potential_global, all_reduce) puis ecrit par le seul rang 0
        (champ scalaire complet, dataset contigu).

        DATASETS CONTIGUS (pas de gzip) : le HDF5 parallele n'autorise pas l'ecriture independante de
        datasets chunk-filtres. Le chemin serie garde gzip ; les VALEURS relues sont identiques champ a
        champ (parallel=True sous np=1 == parallel=False, verifie par test_hdf5_parallel).

        JAMAIS SILENCIEUX : h5py absent, h5py sans MPI, ou mpi4py absent -> RuntimeError avec remede
        (installer h5py compile MPI + mpi4py, ou parallel=False)."""
        import os
        import numpy as np
        from . import _adc
        # h5py D'ABORD, PUIS le test du support MPI : un h5py present mais SANS MPI doit donner
        # l'erreur ciblee (remede), independamment de la presence de mpi4py.
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py absent. Remede : installer h5py compile "
                "MPI (HDF5 parallele), ou parallel=False (gather global + ecriture rang-0).")
        if not h5py.get_config().mpi:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py present mais SANS support MPI "
                "(h5py.get_config().mpi == False). Remede : installer h5py compile MPI (HDF5 "
                "parallele), ou parallel=False (gather global + ecriture rang-0).")
        try:
            from mpi4py import MPI
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : mpi4py absent (requis pour l'ouverture mpio). "
                "Remede : installer mpi4py, ou parallel=False (gather global + ecriture rang-0).")
        # Garde-fou : module _adc construit AVANT les accesseurs locaux (build anterieur a ADC-66).
        if not hasattr(self._s, "local_boxes"):
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : le module _adc charge n'expose pas "
                "local_boxes/local_state (build anterieur a l'ecriture par hyperslabs). Remede : "
                "reconstruire adc_cpp, ou parallel=False.")
        comm = MPI.COMM_WORLD
        rank0 = (_adc.my_rank() == 0)
        # phi : resolu + rassemble COLLECTIVEMENT (tous rangs ; potential_global fait l'all_reduce),
        # ecrit ensuite par le seul rang 0 (champ scalaire global, dataset contigu).
        phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
        # Descripteurs identiques sur tous les rangs (composition partagee) : pre-calcules pour des
        # operations collectives coherentes (create_dataset / attrs).
        ncomp = {b: self._s.n_vars(b) for b in blocks}
        names = {b: [s.encode() for s in self._s.variable_names(b, "conservative")] for b in blocks}
        roles = {b: [s.encode() for s in self._s.variable_roles(b, "conservative")] for b in blocks}
        tmp = target + ".tmp"
        # Ouverture COLLECTIVE (tous les rangs ouvrent le meme fichier via mpio).
        f = h5py.File(tmp, "w", driver="mpio", comm=comm)
        try:
            # Metadonnees collectives -- identiques au chemin serie.
            f.attrs["t"] = self._s.time()
            f.attrs["macro_step"] = self._s.macro_step()
            f.attrs["nx"] = nxv
            f.attrs["ny"] = nyv
            f.attrs["abi_key"] = abi_key()
            for b in blocks:
                g = f.create_group(b)  # collectif
                # Dataset GLOBAL (ncomp, ny, nx) CONTIGU (pas de gzip : ecriture independante interdite
                # sur dataset chunk-filtre en parallele).
                dset = g.create_dataset("state", shape=(ncomp[b], nyv, nxv), dtype="f8")  # collectif
                g.attrs["names"] = names[b]
                g.attrs["roles"] = roles[b]
                # Chaque rang ecrit SES boites locales en hyperslabs (I/O independante : boites
                # disjointes, un rang sans box -> boucle vide). local_state rend deja (ncomp, bny, bnx).
                for li, (ilo, jlo, ihi, jhi) in enumerate(self._s.local_boxes(b)):
                    dset[:, jlo:jhi + 1, ilo:ihi + 1] = np.asarray(
                        self._s.local_state(b, li), dtype=np.float64)
            phi_d = f.create_dataset("phi", shape=(nyv, nxv), dtype="f8")  # collectif
            if rank0:
                phi_d[...] = phi_g  # champ global deja rassemble : ecrit par le seul rang 0
        finally:
            f.close()  # collectif
        comm.Barrier()  # tous les rangs ont ferme AVANT le rename atomique
        if rank0:
            os.replace(tmp, target)
        comm.Barrier()  # le rename est visible (FS partage) avant tout retour
        return target

    def checkpoint(self, path, parallel=False):
        """CHECKPOINT REDEMARRABLE v1 (npz) : etat COMPLET des blocs + horloge (t, macro_step --
        OBLIGATOIRE pour la cadence stride) + grille + provenance (abi_key). CONTRAT (cf.
        docs/IO_CHECKPOINT_PLAN.md) : restart NE reconstruit PAS la composition -- le script
        utilisateur rejoue ses add_block/set_poisson/couplages puis appelle sim.restart(path), qui
        VERIFIE la coherence (blocs, tailles) et leve une erreur explicite sinon. @return le chemin.

        MULTI-RANGS (MPI np>1) : les etats sont rassembles par les accesseurs GLOBAUX collectifs
        (state_global / potential_global -- tous les rangs DOIVENT appeler checkpoint), puis SEUL le
        rang 0 ecrit le fichier UNIQUE (identique au mono-rang). Le couple checkpoint/restart reste
        bit-identique sous np>1 (System mono-box : tout l'etat vit sur le rang 0, gather exact).

        @p parallel : le checkpoint v1 reste TOUJOURS gather-rang-0 (format npz, pas HDF5). L'ecriture
        par hyperslabs (parallel=True) ne s'applique qu'a la SORTIE de visualisation
        write(format='hdf5') : un checkpoint npz n'a ni datasets HDF5 ni decoupage par boites. Passer
        parallel=True leve donc une erreur EXPLICITE (jamais d'ecriture silencieuse degradee) : pour une
        sortie parallele, utiliser write(format='hdf5', parallel=True) ; un checkpoint HDF5 parallele
        redemarrable est un chantier ulterieur (PR-IO-3, cf. docs/IO_CHECKPOINT_PLAN.md)."""
        import os
        import numpy as np
        from . import _adc
        if parallel:
            raise NotImplementedError(
                "checkpoint(parallel=True) : le checkpoint v1 est un npz gather-rang-0 (format non "
                "HDF5, pas de decoupage par boites). L'ecriture par hyperslabs ne concerne que "
                "write(format='hdf5', parallel=True) (sortie de visualisation). Un checkpoint HDF5 "
                "parallele redemarrable reste a faire (PR-IO-3, docs/IO_CHECKPOINT_PLAN.md) ; pour "
                "l'instant : checkpoint(parallel=False).")
        blocks = list(self._s.block_names())
        out = {"adc_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "nx": self._s.nx(), "ny": self._s.ny(),
               "abi_key": abi_key(), "blocks": np.array(blocks)}
        # Gather COLLECTIF (tous rangs) AVANT la garde rang-0.
        for b in blocks:
            nv = self._s.n_vars(b)
            out["ncomp_" + b] = nv
            out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64)
            out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
        # phi : warm start du multigrille (reprise BIT-IDENTIQUE) ; ETAT physique si
        # gauss_policy="evolve" (phi n'y est plus re-derive de rho).
        out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        if _adc.my_rank() != 0:
            return target  # seul le rang 0 ecrit le checkpoint (gather deja fait)
        tmp = target + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """REPREND un checkpoint v1 : VERIFIE la composition (memes blocs, memes tailles -- erreur
        explicite sinon, jamais de reprise silencieusement fausse), restaure l'etat de chaque bloc
        puis l'horloge (t, macro_step : la cadence stride reprend exactement). La COMPOSITION
        (add_block / set_poisson / set_magnetic_field / couplages) doit avoir ete rejouee par le
        script AVANT l'appel (contrat v1, cf. checkpoint).

        MULTI-RANGS (MPI np>1) : tous les rangs lisent le fichier (systeme de fichiers partage) et
        appellent set_state / set_potential / set_clock. set_state / set_potential sont MPI-safe (le
        rang proprietaire -- rang 0, mono-box -- ecrit, les autres font no-op) ; set_clock pose
        l'horloge sur chaque rang. La reprise est donc bit-identique sous np>1."""
        import numpy as np
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["adc_checkpoint_version"]) != 1:
            raise ValueError("restart : version de checkpoint %r non supportee (attendu 1)"
                             % (d["adc_checkpoint_version"],))
        if int(d["nx"]) != self._s.nx() or int(d["ny"]) != self._s.ny():
            raise ValueError("restart : grille du checkpoint (%d x %d) != systeme (%d x %d)"
                             % (int(d["nx"]), int(d["ny"]), self._s.nx(), self._s.ny()))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : blocs du checkpoint %r != composition courante %r "
                             "(rejouer la MEME composition avant restart)" % (chk_blocks, cur_blocks))
        for b in chk_blocks:
            if int(d["ncomp_" + b]) != self._s.n_vars(b):
                raise ValueError("restart : bloc '%s' a %d composantes dans le checkpoint, %d ici"
                                 % (b, int(d["ncomp_" + b]), self._s.n_vars(b)))
            self._s.set_state(b, np.asarray(d["state_" + b], dtype=np.float64))
        # phi AVANT l'horloge : warm start du solveur restaure (reprise bit-identique ; etat
        # physique en gauss_policy="evolve").
        if "phi" in d:
            self._s.set_potential(np.asarray(d["phi"], dtype=np.float64).ravel())
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))

    def __getattr__(self, attr):
        return getattr(self._s, attr)


def capabilities():
    """MATRICE OFFICIELLE des capacites par facade / geometrie / backend (audit 2026-06, vague 2).

    Source de verite UNIQUE consultable par les scripts et la doc (les audits ont montre que System,
    AMR, polaire et les backends DSL divergeaient silencieusement). Les entrees refletent les GATES
    reellement codees (make_block / dispatch_amr_* / block_builder_polar / dsl._BACKENDS) ; les
    combinaisons hors matrice levent une erreur explicite cote C++ (jamais d'ignore silencieux).
    """
    return {
        "riemann": {
            "system_cartesian": ["rusanov", "hll", "hllc", "roe"],
            "system_polar": ["rusanov", "hll"],
            "amr": ["rusanov", "hll", "hllc", "roe"],
            "notes": {
                "rusanov": "generique minimal (max_wave_speed seul)",
                "hll": "generique a ondes signees (model.wave_speeds ; DSL : primitive 'p') ; "
                       "polaire : eligible au fluide isotherme (IsothermalFluxPolar), pas a l'ExB "
                       "scalaire (pas de wave_speeds) -- meme gate que le cartesien",
                "hllc": "Euler 2D canonique (4 var + pression) OU capability modele "
                        "HasHLLCStructure -- emise par le DSL via m.enable_hllc() (roles + 'p', "
                        "y compris 3-var non Euler, scalaires passifs advectes)",
                "roe": "Euler 2D gaz parfait canonique OU capability modele HasRoeDissipation "
                       "-- DEUX voies DSL : (a) m.enable_roe() genere depuis les roles (roles + "
                       "'p' : avec Energy = algebre canonique transcrite, sans Energy = "
                       "c=sqrt(p/rho) moyenne Roe, scalaires passifs sur l'onde entropique) ; (b) "
                       "m.roe_dissipation(x=, y=) FOURNIE par l'utilisateur (eigenstructure propre, "
                       "left()/right() des deux etats, helper m.flux_jacobian auto-derive). Voies "
                       "exclusives (un seul fournisseur du hook). has_roe couvre les deux",
            },
        },
        "time": {
            "system": ["explicit (ssprk2|ssprk3)", "imex (= SourceImplicitBE)",
                       "imexrk_ars222 (famille IMEX-RK, schema ARS(2,2,2), ordre 2 ; cartesien seul ; "
                       "source pleinement implicite)",
                       "split lie|strang + CondensedSchur"],
            "amr": ["explicit (Euler avant par sous-pas)", "ssprk3 (ordre 3 + reflux par etage)",
                    "imex (= SourceImplicitBE)",
                    "split lie|strang + CondensedSchur (mono-bloc, grossier)"],
            "system_polar": ["explicit (ssprk2|ssprk3)", "split + CondensedSchur polaire"],
            "newton_options": "options (max_iters/tol/fd_eps/damping/fail_policy) : System + AMR "
                              "mono-bloc ET multi-blocs natif (loaders .so : rejet explicite) ; "
                              "jacobien analytique via m.source_jacobian ; newton_diagnostics/"
                              "newton_report : System + AMR multi-blocs natif (mono-bloc AMR et "
                              "loaders .so : rejet explicite)",
        },
        "stability_policy": {
            "system": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                       "stability_dt", "coupled_source.frequency", "add_dt_bound (global, "
                       "all_reduce_min)", "last_dt_bound"],
            "amr": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                    "stability_dt", "coupled_source.frequency (multi-blocs)", "add_dt_bound",
                    "last_dt_bound"],
            "system_polar": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                             "stability_dt", "coupled_source.frequency", "add_dt_bound",
                             "last_dt_bound"],
        },
        "poisson": {
            "system_cartesian": ["geometric_mg (paroi, eps(x), aniso, ecrante)",
                                 "fft (periodique, n = 2^k, eps constant, mono-box)"],
            "system_polar": ["polar direct (mono-rang, une box) -- REJET AMONT clair si theta_boxes>1"],
            "amr": ["geometric_mg seulement ; rhs charge_density|composite"],
        },
        "geometry": {
            "system_cartesian": "carre n x n ; mono-box (multi-box = AmrSystem ou MPI mono-box)",
            "system_polar": "anneau (r, theta) global ; theta_boxes=1 mono-box (defaut) OU "
                            "theta_boxes>1 decoupage en bandes theta (divise ntheta). MATRICE "
                            "multi-box (ADC-67) : TRANSPORT (assemble_rhs_polar + fill_ghosts "
                            "collectif) multi-box OK ; Poisson polaire DIRECT mono-box only (rejet "
                            "amont si theta_boxes>1) ; etage Schur tensoriel polaire multi-box. "
                            "get/set state (et eval_rhs/density) reconstruisent l'anneau global "
                            "multi-box ; mono-rang (le Poisson direct refuse MPI).",
            "amr": "hierarchie de niveaux (BoxArray par niveau, regrid dynamique)",
        },
        "schur": {
            "system_cartesian": "complet ; roles/champs configurables (density=/momentum=/energy=/"
                                "magnetic_field=), krylov_tol/max_iters configurables",
            "system_polar": "roles configurables (density=/momentum=/energy=, vague 3) ; "
                            "magnetic_field fige B_z ; solveur multi-box C++, facade une box globale",
            "amr": "mono-bloc ; roles + krylov_tol/max_iters configurables (vague 3, "
                   "magnetic_field fige B_z grossier) ; mono-niveau complet + composite Phase 3c "
                   "(2 niveaux, 1 patch fin mono-box, mono-rang) ; Phase 4 (multi-patch/"
                   ">2 niveaux/MPI/multi-blocs) a faire",
        },
        "backends_dsl": {
            "default": "auto (ADC-63) : production si parite toolchain etablie (module charge + "
                       "compilateur bake + en-tetes concordants), aot sinon ; raison posee sur "
                       "CompiledModel.backend_auto_reason ; backend explicite = court-circuit",
            "prototype": {"adder": "add_dynamic_block", "riemann": ["rusanov"],
                          "limiter": ["none", "minmod", "vanleer"], "stride": False,
                          "evolve_false": False, "mpi": False, "amr": False},
            "aot": {"adder": "add_compiled_block", "riemann": ["rusanov", "hll", "hllc", "roe"],
                    "limiter": ["none", "minmod", "vanleer", "weno5"], "stride": False,
                    "evolve_false": False, "mpi": False, "amr": False,
                    "runtime_params": True},
            "production": {"adder": "add_native_block",
                           "riemann": ["rusanov", "hll", "hllc", "roe"],
                           "limiter": ["none", "minmod", "vanleer", "weno5"], "stride": True,
                           "evolve_false": True, "mpi": True, "amr": "target='amr_system'",
                           "stability_hooks": True},
        },
        "io": {
            "write": ["vtk (.vti cartesien)", "npz",
                      "hdf5 (h5py optionnel, agrege gather rang-0 par defaut)",
                      "hdf5 parallele (write(parallel=True) : hyperslabs par rang via h5py mpio + "
                      "mpi4py ; opt-in, erreur claire si h5py sans MPI ; vrai parallelisme en "
                      "MULTI-BOX, System cartesien mono-box)",
                      "AmrSystem.write npz/vtk (grossier + rectangles des patchs)"],
            "checkpoint_restart": "v1 npz mono-rang/gather-rang-0 (System ; etats + phi + t/macro_step ; "
                                  "composition rejouee par le script ; reprise bit-identique ; "
                                  "checkpoint(parallel=True) leve, reste npz gather-rang-0) ; "
                                  "AMR mono-bloc mono-rang regrid_every=0 (ADC-65 : etat conservatif "
                                  "complet par niveau + phi warm-start + hierarchie imposee ; reprise "
                                  "bit-identique) ; AMR multi-blocs / np>1 et CHECKPOINT HDF5 "
                                  "parallele = suite (docs/IO_CHECKPOINT_PLAN.md ; rejets explicites)",
        },
        "amr_layout": {
            "set_conservative_state": "mono-bloc ET multi-blocs natifs (vague 3 ; loaders .so : "
                                      "rejet explicite)",
        },
        "aux": {
            "canonical": "phi/grad_x/grad_y (base) + B_z (set_magnetic_field) + T_e "
                         "(set_electron_temperature_from), liste fermee ADC_AUX_FIELDS/AUX_CANONICAL",
            "named": "champs NOMMES par modele (ADC-70 phase 1) : m.aux_field('nom') -> composantes "
                     ">= 5 (kAuxNamedBase) ; set_aux_field(bloc, nom, array) / aux_field(bloc, nom) sur "
                     "System CARTESIEN ; au plus kAuxMaxExtra=4 par modele ; statiques, persistants",
            "named_followups": "AMR (canal aux par niveau / regrid), polaire (validation), halos "
                               "custom par champ, table nom->comp cote C++ Impl (resolution sans "
                               "Python) = SUIVI ; phase 1 = System cartesien seulement",
        },
    }


def _reject_newton_amr_compiled(label, time):
    """REJETTE les options/diagnostics Newton sur le chemin AMR COMPILE (.so loader, ABI plate
    add_native_block / adc_install_native_amr) -- vague 3, solde. Cote NATIF (adc.Model(...)), les
    OPTIONS Newton sont desormais cablees en mono-bloc (coupleur) ET multi-blocs (moteur), et le
    RAPPORT newton_diagnostics en multi-blocs natif ; mais l'ABI plate du loader .so ne transporte NI
    les options (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy) NI le rapport. Passees
    via le loader, elles seraient prises a leurs defauts EN SILENCE (iters=2, pas de rapport). On les
    REJETTE explicitement (meme esprit que le rejet stride/masque du chemin production AMR). Pour ces
    parametres : AmrSystem.add_block (modele natif) ou add_compiled_model(AmrSystem&) en direct (C++)."""
    if (getattr(time, "newton_max_iters", 2) != 2
            or getattr(time, "newton_rel_tol", 0.0) != 0.0
            or getattr(time, "newton_abs_tol", 0.0) != 0.0
            or getattr(time, "newton_fd_eps", 1e-7) != 1e-7
            or getattr(time, "newton_damping", 1.0) != 1.0
            or getattr(time, "newton_fail_policy", "none") != "none"
            or getattr(time, "newton_diagnostics", False)):
        raise ValueError(
            "%s : les options/diagnostics Newton (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/"
            "fail_policy/diagnostics) ne sont pas transportes par le chemin production AMR (loader "
            ".so, ABI plate add_native_block : ils seraient pris a leurs defauts en silence). "
            "Utiliser AmrSystem.add_block (modele natif adc.Model(...)) ou add_compiled_model("
            "AmrSystem&) en direct (C++)." % label)


class AmrSystem:
    """Pendant raffine de System : un ou PLUSIEURS blocs portes sur une hierarchie AMR.

    MONO-BLOC (1 add_block) : chemin AmrCouplerMP historique (regrid dynamique, reflux). MULTI-BLOCS
    (>= 2 add_block) : N blocs co-localises sur UNE hierarchie AMR PARTAGEE (moteur AmrRuntime),
    Poisson de SYSTEME a second membre SOMME co-localise (Sum_b q_b n_b), conservation PAR BLOC. Les
    blocs peuvent avoir des SCHEMAS SPATIAUX DIFFERENTS, un TRAITEMENT TEMPOREL par bloc (explicit /
    imex), du MULTIRATE (substeps / stride), des SOURCES COUPLEES inter-especes et le DSL production
    multi-bloc. En multi-blocs le NOM du bloc indexe set_density(name) / mass(name) / density(name).

    REGRID D'UNION DES TAGS (multi-blocs + regrid_every > 0) : la hierarchie partagee est re-grillee a
    partir de l'UNION des tags de tous les blocs. Deux criteres se composent (OU cellule a cellule) :

    - DENSITE PAR BLOC (set_refinement(threshold)) : raffine la ou la densite (composante 0) d'un bloc
      depasse threshold ;
    - ``grad phi`` (set_phi_refinement(grad_threshold)) : raffine la ou la norme du gradient du potentiel
      electrostatique depasse grad_threshold (bord d'anneau du diocotron). Desactive par defaut
      (grad_threshold <= 0). MULTI-BLOCS uniquement.

    regrid_every == 0 -> hierarchie FIGEE (regrid jamais appele, bit-identique).
    """

    def __init__(self, config=None, **cfg_kw):
        if config is None:
            config = AmrSystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # cf. System.__init__ : _AmrSystem(config) declenche l'init Kokkos (lazy). set_threads
        # n'a plus d'effet apres ce point.
        global _first_system_built
        _first_system_built = True
        self._s = _AmrSystem(config)
        self._L = float(config.L)  # cote de [0, L]^2 (pour patch_rectangles : index -> physique)
        # Cadence regrid (checkpoint/restart ADC-65) : une reprise BIT-IDENTIQUE exige regrid_every == 0
        # (sinon le regrid post-restart re-divergerait la hierarchie). Memorise pour la garde de restart.
        self._regrid_every = int(config.regrid_every)

    def patch_rectangles(self):
        """Rectangles physiques (x0, y0, largeur, hauteur) des patchs fins courants, dans [0, L]^2.

        Convertit patch_boxes() (espace d'indices, coins inclusifs) en coordonnees physiques. Le pas
        du niveau est dx = L / (n << level) (ratio 2 par niveau) ; un patch [ilo..ihi] x [jlo..jhi]
        couvre (ihi - ilo + 1) cellules en x depuis x0 = ilo * dx (et de meme en y). Convention de
        grille ne[j, i] -> indice 0 = x (i), indice 1 = y (j), coherent avec density() et un imshow
        d'extent [0, L, 0, L]. Pratique pour tracer les VRAIS patchs (ex. matplotlib Rectangle) sans
        reconstruire un proxy de densite. Renvoie une liste de (x0, y0, w, h), un par patch fin (tous
        niveaux fins confondus). Query (entre les pas) : declenche le build paresseux comme
        n_patches(), aucun cout sur le chemin chaud.
        """
        n, L = self._s.nx(), self._L
        rects = []
        for level, ilo, jlo, ihi, jhi in self._s.patch_boxes():
            dx = L / (n << level)
            rects.append((ilo * dx, jlo * dx, (ihi - ilo + 1) * dx, (jhi - jlo + 1) * dx))
        return rects

    def add_block(self, name, model, spatial=None, time=None):
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # adc.Split / adc.Strang (etage source condense par Schur) n'est cable que par add_equation (qui
        # branche set_source_stage + set_time_scheme APRES l'ajout du bloc) : on le rejette ICI plutot que
        # de jouer le seul transport et de PERDRE la source en silence (meme garde que System.add_block).
        if isinstance(time, Split):
            raise TypeError(
                "AmrSystem.add_block : adc.Split / adc.Strang (etage source condense par Schur) n'est "
                "supporte que par add_equation (qui branche l'etage source) ; utiliser "
                "add_equation(..., time=adc.Strang(hyperbolic=adc.Explicit(...), "
                "source=adc.CondensedSchur(...))).")
        # On thread substeps/stride (multirate, capstone iv), le masque IMEX partiel, les OPTIONS Newton
        # ET newton_diagnostics (vague 3, solde). Resolus / valides cote C++ (AmrSystem::add_block) contre
        # les noms/roles du bloc : vides -> backward-Euler plein. Les options sont cablees en mono-bloc
        # (coupleur) ET multi-blocs ; newton_diagnostics est cable en MULTI-BLOCS natif et REJETE au build
        # C++ en mono-bloc (le coupleur n'agrege pas de rapport) -- pas de filtrage de facade ici (la
        # facade ne connait pas encore le nombre total de blocs : la decision mono/multi est au build).
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", 2),
                          getattr(time, "newton_rel_tol", 0.0),
                          getattr(time, "newton_abs_tol", 0.0),
                          getattr(time, "newton_fd_eps", 1e-7),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"),
                          getattr(time, "newton_diagnostics", False))

    def write(self, path, format="npz", step=None):
        """SORTIE DE VISUALISATION AMR (vague 3) : champs GROSSIERS par bloc + phi + empreintes des
        patchs fins. format='npz' (densites par bloc, phi, patch_rectangles, t) ou 'vtk' (.vti du
        GROSSIER : densite par bloc + phi -- les patchs fins sont fournis en npz via leurs
        rectangles, le multi-resolution VTK = PR-IO-3). @p step : suffixe numerote. @return chemin."""
        import os
        import numpy as np
        n = self._s.nx()
        suffix = ("_%06d" % int(step)) if step is not None else ""
        # CHAQUE bloc, par son nom (binding AmrSystem::block_names, parite System) : en multi-blocs,
        # density() sans nom ne lirait QUE le bloc 0 et perdrait les autres EN SILENCE.
        names = list(self._s.block_names())
        if not names:
            names = [""]
        if format == "npz":
            out = {"t": self._s.time(), "n": n,
                   "patch_rectangles": np.array(self.patch_rectangles(), dtype=np.float64)
                   if self.patch_rectangles() else np.zeros((0, 4))}
            for b in names:
                key = b if b else "block"
                out["density_" + key] = np.asarray(self.density(b) if b else self.density(),
                                                   dtype=np.float64)
            out["phi"] = np.asarray(self.potential(), dtype=np.float64)
            target = path + suffix + ".npz"
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, labels = [], []
            for b in names:
                key = b if b else "block"
                arrays.append(np.asarray(self.density(b) if b else self.density(),
                                         dtype=np.float64).reshape(n, n))
                labels.append("%s_density" % key)
            arrays.append(np.asarray(self.potential(), dtype=np.float64).reshape(n, n))
            labels.append("phi")
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (n, n, self._L / n, self._L / n),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (n, n),
                     '      <CellData>']
            for nm, arr in zip(labels, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        raise ValueError("AmrSystem.write : format 'npz' | 'vtk' (recu %r)" % (format,))

    def checkpoint(self, path):
        """CHECKPOINT AMR REDEMARRABLE BIT-IDENTIQUE v1 (npz), MONO-BLOC MONO-RANG (ADC-65). Ecrit
        l'ETAT CONSERVATIF COMPLET de CHAQUE niveau (toutes composantes ; le grossier ET les patchs
        fins, cellules valides), le phi de chaque niveau (le niveau 0 = WARM-START du multigrille,
        load-bearing pour la reprise bit-identique), la HIERARCHIE (patch_boxes), l'horloge (t,
        macro_step) et la cadence regrid. CONTRAT (parite System.checkpoint) : restart NE reconstruit
        PAS la composition -- le script rejoue ses add_block/set_poisson/set_refinement/set_density
        puis appelle sim.restart(path), qui VERIFIE la coherence et leve sinon. @return le chemin.

        PERIMETRE (rejets EXPLICITES, jamais un checkpoint silencieusement faux/partiel) :
          - MONO-BLOC seulement : le multi-blocs (moteur AmrRuntime) partage layout ET aux entre blocs
            et n'expose pas l'etat par niveau/bloc -> suite (les accesseurs C++ rejettent aussi).
          - MONO-RANG (np == 1) : les accesseurs de niveau lisent les fabs LOCAUX sans gather MPI ;
            un gather par niveau (BoxArray + DistributionMapping) est une suite.
          - regrid_every == 0 : une reprise bit-identique exige une hierarchie FIGEE (sinon le regrid
            re-divergerait apres le restart). On rejette des le checkpoint (echec tot, message clair).

        Repli hors perimetre : AmrSystem.write (visualisation) ou un System mono-niveau."""
        import os
        import numpy as np
        from . import _adc
        if _adc.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : MPI np>1 non cable (ADC-65 mono-rang : les etats par niveau "
                "sont lus sur les fabs LOCAUX, le gather par niveau = suite). Lancer en mono-rang, ou "
                "utiliser un System mono-niveau (checkpoint/restart bit-identique y compris sous MPI).")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : multi-blocs non cable (ADC-65 mono-bloc : le moteur AmrRuntime "
                "partage layout ET aux entre blocs et n'expose pas l'etat par niveau/bloc = suite). "
                "Utiliser un seul add_block, ou un System mono-niveau (checkpoint/restart bit-identique).")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.checkpoint : reprise bit-identique cablee pour regrid_every == 0 seulement "
                "(hierarchie figee) ; ce systeme a regrid_every=%d (le regrid post-restart re-divergerait "
                "la hierarchie). Reconstruire le systeme avec regrid_every=0." % self._regrid_every)
        nlev = int(self._s.n_levels())
        pb = self._s.patch_boxes()  # (level, ilo, jlo, ihi, jhi) inclusifs, espace d'indices du niveau
        out = {"adc_amr_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "n": self._s.nx(), "L": self._L, "regrid_every": self._regrid_every,
               "abi_key": abi_key(), "blocks": np.array(list(self._s.block_names())),
               "n_vars": int(self._s.n_vars()), "n_levels": nlev,
               "patch_boxes": (np.asarray(pb, dtype=np.int64) if pb
                               else np.zeros((0, 5), dtype=np.int64))}
        for k in range(nlev):
            # Etat conservatif COMPLET du niveau k (c*nf*nf + j*nf + i) + phi (nf*nf). Niveau fin : seules
            # les cellules des patchs sont definies (0 ailleurs) ; le restart ne reecrit que ces cellules.
            out["state_%d" % k] = np.asarray(self._s.level_state(k), dtype=np.float64)
            out["phi_%d" % k] = np.asarray(self._s.level_potential(k), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        tmp = target + ".tmp"  # ecriture ATOMIQUE (.tmp + os.replace : un crash ne corrompt rien)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """REPREND un checkpoint AMR v1 (BIT-IDENTIQUE, MONO-BLOC MONO-RANG, ADC-65). VERIFIE la
        coherence (version, grille, blocs, composantes, regrid_every == 0) puis : (1) IMPOSE la
        hierarchie fine sauvee (set_hierarchy, au lieu du clustering Berger-Rigoutsos) ; (2) restaure
        l'etat conservatif COMPLET de chaque niveau TEL QUEL (pas de re-prolongation) ; (3) restaure le
        phi de chaque niveau (le niveau 0 = warm-start du multigrille -> le 1er solve post-restart
        repart du meme guess) ; (4) restaure l'horloge (t, macro_step). La COMPOSITION (add_block /
        set_poisson / set_refinement / set_density) doit avoir ete REJOUEE par le script AVANT l'appel.

        ORDRE : set_hierarchy AVANT set_level_state (l'imposition du layout precede la restauration des
        cellules valides) ; phi et horloge ensuite. Le 1er step rejoue update() (sync_down + solve
        warm-start) puis advance -- les ghosts (grossier ET fins) sont refaits par le step, exactement
        comme apres un regrid, d'ou la reprise bit-identique sans restaurer de ghosts."""
        import numpy as np
        from . import _adc
        if _adc.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : MPI np>1 non cable (ADC-65 mono-rang ; cf. checkpoint). Lancer en "
                "mono-rang, ou utiliser un System mono-niveau.")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : multi-blocs non cable (ADC-65 mono-bloc ; cf. checkpoint). Utiliser "
                "un seul add_block, ou un System mono-niveau.")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.restart : exige regrid_every == 0 (hierarchie figee ; sinon le regrid "
                "post-restart re-divergerait la hierarchie restauree). Reconstruire le systeme avec "
                "regrid_every=0 avant restart. (regrid_every courant = %d)" % self._regrid_every)
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["adc_amr_checkpoint_version"]) != 1:
            raise ValueError("restart : version de checkpoint AMR %r non supportee (attendu 1)"
                             % (d["adc_amr_checkpoint_version"],))
        if int(d["n"]) != self._s.nx():
            raise ValueError("restart : grille du checkpoint (n=%d) != systeme (n=%d)"
                             % (int(d["n"]), self._s.nx()))
        if float(d["L"]) != self._L:
            raise ValueError("restart : domaine du checkpoint (L=%r) != systeme (L=%r) -- dx different"
                             % (float(d["L"]), self._L))
        if int(d["regrid_every"]) != 0:
            raise ValueError("restart : checkpoint pris avec regrid_every=%d != 0 (reprise "
                             "bit-identique impossible)" % int(d["regrid_every"]))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : blocs du checkpoint %r != composition courante %r "
                             "(rejouer la MEME composition avant restart)" % (chk_blocks, cur_blocks))
        if int(d["n_vars"]) != int(self._s.n_vars()):
            raise ValueError("restart : %d composantes dans le checkpoint, %d ici"
                             % (int(d["n_vars"]), int(self._s.n_vars())))
        nlev = int(d["n_levels"])
        if nlev != int(self._s.n_levels()):
            raise ValueError("restart : %d niveaux dans le checkpoint, %d ici (la composition / le "
                             "raffinement different ?)" % (nlev, int(self._s.n_levels())))
        # (1) IMPOSER la hierarchie fine sauvee (le coupleur filtre le niveau 1), sauf hierarchie
        # MONO-NIVEAU (n_levels == 1, p.ex. chemin amr-schur sans patch fin) : rien a imposer alors.
        boxes = [tuple(int(x) for x in row) for row in np.asarray(d["patch_boxes"], dtype=np.int64)]
        if nlev >= 2:
            if not any(b[0] == 1 for b in boxes):
                raise ValueError("restart : hierarchie a %d niveaux mais aucun patch fin (niveau 1) "
                                 "dans le checkpoint (incoherent)." % nlev)
            self._s.set_hierarchy(boxes)
        # (2) restaurer l'etat conservatif COMPLET de chaque niveau TEL QUEL (pas de re-prolongation) ;
        # set_level_state applatit le tableau et n'ecrit que les cellules valides (les patchs).
        for k in range(nlev):
            self._s.set_level_state(k, np.asarray(d["state_%d" % k], dtype=np.float64))
        # (3) restaurer le phi (niveau 0 = warm-start du multigrille : reprise bit-identique).
        for k in range(nlev):
            self._s.set_level_potential(k, np.asarray(d["phi_%d" % k], dtype=np.float64).ravel())
        # (4) restaurer l'horloge APRES l'etat (parite System ; macro_step pousse la phase de cadence).
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))

    def add_equation(self, name, model, spatial=None, time=None, substeps=None):
        """Ajoute l'UNIQUE equation/bloc AMR en aiguillant sur le TYPE de @p model (DSL Phase D) :

        - un ModelSpec (adc.Model(...)) -> add_block (briques natives composees sur la hierarchie) ;
        - un CompiledModel(backend='production', target='amr_system') (m.compile(...)) -> chemin NATIF
          add_native_block : le loader .so inline add_compiled_model(AmrSystem&), donc le bloc tourne
          la MEME hierarchie AMR que add_block (reflux conservatif, regrid), ZERO-COPIE.

        Le traitement temporel est cable a {explicit, imex} : imex traite la source raide en IMPLICITE
        (backward_euler_source), le transport restant explicite et porte par le reflux conservatif
        (parite avec l'IMEX du System ; la source etant cellule-locale, le split implicite ne touche pas
        la conservation aux interfaces grossier-fin). recon "primitive" et flux "roe"/"hllc"/weno5 sont
        CABLES sur AMR (parite avec add_block ; cf. dispatch_amr_compiled). limiter="weno5" (WENO5-Z,
        3 ghosts) : le coupleur alloue ses niveaux a Limiter::n_ghost et le regrid herite n_grow(), donc
        le stencil 5 points ne lit pas hors bornes. cf. DSL_MODEL_DESIGN.md Phase D (point 10).

        CADENCE MULTIRATE (stride) et MASQUE IMEX PARTIEL (implicit_vars / implicit_roles) :

        - chemin ModelSpec (adc.Model(...)) : FORWARDES a AmrSystem::add_block, qui les SUPPORTE et les
          valide (parite avec le wrapper add_block) ;
        - chemin CompiledModel production (.so) : REJETES explicitement (ValueError). L'ABI plate du
          loader (add_native_block / adc_install_native_amr) ne les transporte pas ; ils seraient pris a
          leurs defauts EN SILENCE (stride=1, backward-Euler plein). Pour un .so multirate ou a masque
          IMEX partiel, utiliser AmrSystem.add_block (natif) ou add_compiled_model(AmrSystem&) en direct
          (C++), qui exposent stride et le masque.

        @p spatial : adc.FiniteVolume(...) / adc.Spatial(...) (defaut minmod+rusanov+conservatif).
        @p time : adc.Explicit (defaut) ou adc.IMEX (source raide implicite). @p substeps : surcharge
        time.substeps.
        """
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # --- adc.Split (Lie) / adc.Strang (2e ordre) : CHEMIN amr-schur (etage source condense GLOBAL) --
        # Pendant AMR de System.add_equation (cf. ~ligne 925) : on ajoute d'abord le bloc avec son seul
        # etage HYPERBOLIQUE explicite (transport SOURCE-FREE ; le modele doit porter une brique source
        # NoSource), PUIS on branche l'etage SOURCE condense (set_source_stage, C++) et la politique de
        # splitting (set_time_scheme : "lie" pour Split, "strang" pour Strang). L'etage condense est
        # GLOBAL (assemble/resout l'operateur electrostatique/Lorentz sur le grossier, en composant
        # l'etage uniforme), par opposition a la source IMEX LOCALE cellule par cellule de time=adc.IMEX.
        # PREREQUIS : appeler sim.set_magnetic_field(B_z) AVANT le 1er step (le terme de Lorentz lit
        # Omega = B_z) ; sinon erreur claire au build. MONO-BLOC uniquement (set_source_stage leve sinon).
        if isinstance(time, Split):
            self.add_equation(name, model, spatial=spatial, time=time.hyperbolic, substeps=substeps)
            src = time.source
            # Reglages TRANSPORTES par le chemin amr-schur depuis la vague 3 (parite System) :
            # tolerances Krylov du solve grossier + descripteurs de champs (role stable ou nom de
            # variable, resolus au build contre le Model concret). magnetic_field reste fige sur
            # le tampon B_z grossier dedie (amr_write_coarse_bz) : un autre champ aux n'a pas de
            # pendant AMR -> rejet explicite (pas d'ignore silencieux).
            if getattr(src, "bz_aux_component", -1) >= 0:
                raise ValueError(
                    "AmrSystem.add_equation : magnetic_field != 'B_z' n'est pas transporte par le "
                    "chemin amr-schur (l'etage AMR lit le tampon B_z grossier dedie). Laisser "
                    "magnetic_field='B_z', ou utiliser System (mono-niveau).")
            self._s.set_source_stage(name, src.kind, src.theta, src.alpha,
                                     getattr(src, "krylov_tol", 0.0),
                                     getattr(src, "krylov_max_iters", 0),
                                     getattr(src, "density_spec", ""),
                                     getattr(src, "momentum_x_spec", ""),
                                     getattr(src, "momentum_y_spec", ""),
                                     getattr(src, "energy_spec", ""))
            self._s.set_time_scheme(time.scheme)  # "lie" (Split) ou "strang" (Strang)
            return

        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)

        # --- ModelSpec : briques natives composees -> add_block (chemin existant) ---
        # On FORWARDE stride (multirate, capstone iv) ET le masque IMEX partiel implicit_vars /
        # implicit_roles (capstone vii), exactement comme le wrapper AmrSystem.add_block ci-dessus :
        # le C++ AmrSystem::add_block les SUPPORTE et les valide (vides -> backward-Euler plein ; un
        # masque demande en explicite ou en mono-bloc leve une erreur claire cote C++,
        # amr_system.cpp:325-328 / :283-287). Ne PAS dupliquer ces gardes ici.
        if isinstance(model, ModelSpec):
            # Modele NATIF : OPTIONS Newton cablees (mono + multi) + newton_diagnostics (multi-blocs natif,
            # rejete au build C++ en mono-bloc). Pas de filtrage de facade : C++ AmrSystem::add_block valide.
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, getattr(time, "stride", 1),
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", 2),
                              getattr(time, "newton_rel_tol", 0.0),
                              getattr(time, "newton_abs_tol", 0.0),
                              getattr(time, "newton_fd_eps", 1e-7),
                              getattr(time, "newton_damping", 1.0),
                              getattr(time, "newton_fail_policy", "none"),
                              getattr(time, "newton_diagnostics", False))
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

        # recon "primitive" et flux "roe"/"hllc" sont CABLES sur AMR via dispatch_amr_compiled : le
        # chemin .so passe recon_prim a AmrBuildParams (consomme par advance_amr/compute_face_fluxes)
        # et hllc/roe sont instancies sous la MEME garde requires que System::make_block (transport
        # compressible a 4 variables + pression). Plus de rejet de facade (parite stricte avec
        # add_block, cf. test_amr_riemann_native).
        # Garde-fou flux numerique (porte de System.add_equation) : HLLC/Roe exigent une pression ; la
        # brique generee n'emet pressure()/wave_speeds() que si une primitive 'p' est declaree. Sans
        # 'p', dispatch_amr_compiled retombe sur la branche else (requires non satisfait) et leve une
        # erreur C++ generique : on diagnostique ICI, clairement, avant la frontiere C++.
        if (spatial.flux in ("roe", "hllc") and "p" not in compiled.prim_names
                and not (spatial.flux == "hllc" and getattr(compiled, "has_hllc", False))
                and not (spatial.flux == "roe" and getattr(compiled, "has_roe", False))):
            raise ValueError(
                "AmrSystem.add_equation : riemann '%s' exige une pression : declarer une primitive 'p' "
                "(m.primitive('p', ...)) dans le modele, ou emettre la capability "
                "(m.enable_hllc() / m.enable_roe()) ; sinon utiliser riemann='rusanov'"
                % spatial.flux)

        # L'ABI plate du loader .so (adc_install_native_amr / add_native_block) ne transporte NI la
        # cadence multirate (stride) NI le masque IMEX partiel (implicit_vars / implicit_roles) :
        # add_compiled_model(AmrSystem&) les expose seulement en DIRECT (chemin C++). Passes via le
        # loader, ils prendraient leurs defauts (stride=1, masque vide = backward-Euler plein) EN
        # SILENCE. On les REJETTE plutot que de les ignorer (route explicite, meme esprit que le rejet
        # du stride/masque sur les backends compiles de System.add_equation, cf. ~lignes 886-955).
        nstride = getattr(time, "stride", 1)
        if nstride != 1:
            raise ValueError(
                "AmrSystem.add_equation : stride=%d non transporte par le chemin production AMR "
                "(loader .so, ABI plate add_native_block : le bloc tournerait a stride=1 en silence). "
                "Utiliser AmrSystem.add_block (modele natif adc.Model(...), cadence cablee) ou "
                "add_compiled_model(AmrSystem&) en direct (C++) qui expose stride." % nstride)
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "AmrSystem.add_equation : implicit_vars / implicit_roles (masque IMEX partiel) non "
                "transportes par le chemin production AMR (loader .so, ABI plate add_native_block : le "
                "masque serait vide = backward-Euler plein en silence). Utiliser AmrSystem.add_block "
                "(modele natif adc.Model(...), masque cable) ou add_compiled_model(AmrSystem&) en "
                "direct (C++) qui expose le masque IMEX.")
        # Options / diagnostics Newton : meme ABI plate -> ni les options ni le rapport ne transitent
        # par le loader .so. Rejet explicite (sinon iters=2 / pas de rapport en silence), parite avec le
        # rejet stride/masque ci-dessus et avec System.add_equation (backend compile).
        _reject_newton_amr_compiled("AmrSystem.add_equation", time)

        # Garde PRE-DLOPEN au branchement (couvre le cache HIT, cf. System.add_equation) : module
        # _adc perime vs .so compile sur les en-tetes a jour -> erreur actionnable, pas un dlopen
        # 'symbol not found' cryptique.
        from . import dsl as _dsl_guard
        _dsl_guard.check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        gamma = compiled.gamma if compiled.gamma is not None else 1.4
        self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                 spatial.recon, time.kind, gamma, nsub)

    def add_coupling(self, coupling):
        """Ajoute une SOURCE COUPLEE inter-especes generique (adc.dsl.CoupledSource(...).compile(...))
        sur la hierarchie AMR PARTAGEE (MULTI-BLOCS), pendant raffine de System.add_coupling. La source
        est transportee en bytecode et interpretee cote C++ (AmrSystem.add_coupled_source ; aucun
        callback Python par cellule). La frequence du couplage (CoupledSource.frequency) est honoree :
        constante -> borne dt <= cfl/mu ; Expr -> frequence PAR CELLULE mu(U) evaluee sur le GROSSIER a
        chaque step_cfl (les vecteurs freq_prog_* sont forwardes). Doit etre appele AVANT le premier
        step (la source est figee puis injectee au build paresseux du moteur runtime)."""
        from . import dsl  # import tardif (dsl importe ce module : eviter le cycle a l'import)

        if isinstance(coupling, dsl.CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        else:
            raise TypeError("AmrSystem.add_coupling attend un CompiledCoupledSource "
                            "(adc.dsl.CoupledSource(...).compile(...)) : la source couplee AMR est "
                            "MULTI-BLOCS et decrite en formules")

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


from . import integrate  # noqa: E402  (apres la definition de System ; sans dependance numpy)


# adc.dsl PARESSEUX (PEP 562) : dsl.py fait `import numpy` au niveau module (evaluateur hote du
# prototype). L'import eager rendait numpy obligatoire pour `import adc` ENTIER, alors que le
# chemin natif (System/add_block) et le backend production n'en ont pas besoin. Avec ce
# __getattr__, `adc.dsl.Model(...)` et `from adc import dsl` marchent a l'identique, mais numpy
# n'est requis QU'AU premier usage du DSL -- et son absence donne un message cible (doctor aussi).
def __getattr__(name):
    if name == "dsl":
        import importlib
        try:
            return importlib.import_module(".dsl", __name__)
        except ImportError as exc:
            raise ImportError(
                "adc.dsl requiert numpy dans cet interpreteur (evaluateur hote du DSL) : "
                "`pip install numpy` / `conda install numpy`. Le reste d'adc (System, add_block) "
                "fonctionne sans. Cause : %s" % exc) from exc
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
