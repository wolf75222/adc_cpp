"""adc.dsl : mini-DSL SYMBOLIQUE de modeles physiques.

Python ECRIT les formules (variables nommees, expressions), pas une fonction appelee par cellule.
Les operations (+, -, *, /, **, adc.dsl.sqrt) construisent un ARBRE d'expressions. Un
HyperbolicModel declare ses variables conservatives, ses primitives (definies par des formules),
son flux, ses valeurs propres, sa source et sa contribution elliptique.

    e = adc.dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    p = e.primitive("p", (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    e.set_flux(x=[rhou, rhou*u + p, ...], y=[...])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[...])

Backends disponibles via m.compile(backend=..., target=...) :
  - "prototype" : evaluateur JIT NumPy/hote, Rusanov ordre 1, TEST uniquement (hote seul) ;
  - "aot"       : genere un .so ABI plate (debug ABI), compilation ahead-of-time ;
  - "production": loader natif zero-copie (add_native_block), RECOMMANDE par defaut ; chemin
                  device-clean (foncteurs nommes, valide GH200, #97/#93) et MPI/AMR-ready.

Cible (target=) :
  - "System"    : systeme mono-niveau ;
  - "AmrSystem" : systeme AMR cable (#92/#105), uniquement avec backend "production".

Ergonomie :
  - m.compile() auto-detecte les includes adc et cache le .so par (model_hash, abi_key) (#103).
  - Les roles physiques (gamma, n_aux, B_z, T_e) sont preserves et transmis au C++.

Couplage inter-especes (#131, #167) :
  - CoupledSource decrit un echange entre blocs par formules DSL (source croisee arbitraire).
  - CoupledSource.add_pair(block_a, block_b, role, expr) garantit la conservation par
    construction (+expr sur A, -expr sur B, MEME sous-arbre) ; equivalent DSL de
    add_collision / add_thermal_exchange cote C++. compile(verify_conservation=True) controle
    la propriete symboliquement sur l'ensemble des termes (add et add_pair).
"""
import os
import shutil
import sys

import numpy as np


# --- Signature de l'arbre d'en-tetes du coeur (cle d'ABI du chemin "production") -------------
# Le backend "production" (compile_native) emet un loader .so qui inline le gabarit en-tete
# adc::add_compiled_model et appelle des methodes hors-ligne du module _adc deja charge. Loader et
# module DOIVENT partager la meme ABI C++ (memes en-tetes, compilateur, standard). On materialise la
# "signature des en-tetes" dans la cle d'ABI (adc/runtime/abi_key.hpp, jeton ADC_HEADER_SIG) ; le
# build du module la bake (CMake) et compile_native la rebake (flag -D) en la calculant a l'IDENTIQUE.
# Le calcul DOIT etre bit-pour-bit identique cote CMake (python/CMakeLists.txt) et ici : sha256 du
# concatene trie "<relpath>\n<sha256(contenu)>\n" de chaque .hpp/.h sous include/. cf. abi_key.hpp.
def adc_header_signature(include):
    """Signature stable de l'arbre d'en-tetes adc sous @p include : sha256 du concatene trie
    "<chemin relatif>\\n<sha256 du contenu>\\n" de chaque .hpp/.h. MIROIR EXACT du calcul CMake
    (python/CMakeLists.txt) : si un en-tete change, la signature change des deux cotes, donc la cle
    d'ABI diverge et add_native_block leve une erreur explicite (jamais d'UB silencieux)."""
    import hashlib
    import os
    entries = []
    for root, _dirs, files in os.walk(include):
        for fn in files:
            if fn.endswith((".hpp", ".h")):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, include).replace(os.sep, "/")  # CMake ecrit les chemins en '/' (meme Windows)
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                entries.append("%s\n%s\n" % (rel, digest))
    blob = "".join(sorted(entries)).encode()
    return hashlib.sha256(blob).hexdigest()


# --- Auto-detection du dossier include adc -----------------------------------
# Pour rendre m.compile(...) ergonomique, le dossier des en-tetes adc est deduit automatiquement
# quand l'appelant ne le passe pas. MIROIR de adc_cases/common/native.py::adc_include : on essaie
# $ADC_INCLUDE (override explicite), puis on remonte depuis le paquet `adc` installe (build-py/python/
# adc/ -> ../../../include), puis le depot voisin ../adc_cpp/include. Critere de validite : le fichier
# canonique adc/mesh/multifab.hpp existe. Pas d'import dur de adc ici (le module dsl peut etre charge
# hors paquet) : on resout `adc.__file__` paresseusement.
def adc_include():
    """Dossier include/ d'adc_cpp (en-tetes header-only du coeur), auto-detecte.

    Priorite : $ADC_INCLUDE (override), sinon depuis le paquet `adc` installe
    (.../adc -> ../../../include), sinon le depot voisin (.../adc_cpp/include depuis ce module).
    Exige que adc/mesh/multifab.hpp existe. Leve RuntimeError si introuvable (diagnostic listant les
    candidats), pour ne JAMAIS compiler contre un include silencieusement faux."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))           # .../python/adc
    candidates = []
    env = os.environ.get("ADC_INCLUDE")
    if env:
        candidates.append(env)
    try:
        import adc as _adc_pkg
        pkg = os.path.dirname(os.path.abspath(_adc_pkg.__file__))   # .../adc
        candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    except Exception:
        pass
    # depuis ce fichier (python/adc/dsl.py) : python/adc -> python -> racine depot -> include
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "include")))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "adc", "mesh", "multifab.hpp")):
            return c
    raise RuntimeError(
        "en-tetes adc introuvables (cherche adc/mesh/multifab.hpp). "
        "Passer include=<adc_cpp>/include ou definir ADC_INCLUDE. Candidats essayes : "
        + ", ".join(repr(c) for c in candidates))


# --- Norme C++ du loader natif (frontiere ABI du chemin "production") ----------
# Le backend "production" genere un loader .so qui inline add_compiled_model<> et appelle des methodes
# hors-ligne du module _adc DEJA charge. La cle d'ABI (adc/runtime/abi_key.hpp) encode __cplusplus :
# le loader et le module doivent donc partager la MEME norme C++, sinon add_native_block rejette
# ("ABI incompatible"). Le module bake sa norme reelle (ADC_CXX_STD : 20 sous Kokkos car CUDA 12.x
# n'a pas -std=c++23, 23 sinon) et l'expose en _adc.__cxx_std__. On en derive le flag -std attendu du
# modele natif AU LIEU de figer c++23 (qui cassait le chemin natif sous Kokkos/GH200, ou le module est
# en c++20). MIROIR direct du build, donc jamais d'ecart silencieux entre loader et modele.
def loader_cxx_std():
    """Flag '-std=c++NN' que le modele natif (backend="production") DOIT utiliser pour partager l'ABI
    du module _adc charge. Source de verite : _adc.__cxx_std__ (entier 20/23 baked par le build, =
    ADC_CXX_STD : 20 sous Kokkos, 23 sinon). Fallbacks gracieux si l'attribut manque (vieux module) :
    on parse __cplusplus depuis _adc.abi_key() (>202002L -> c++23, sinon c++20) ; faute de tout cela,
    on retombe sur le defaut historique c++23 (cas hote non-Kokkos, inchange)."""
    try:
        import _adc
    except Exception:
        try:
            from . import _adc  # paquet adc : l'extension est un sous-module
        except Exception:
            _adc = None
    std = _adc_cxx_std_from_module(_adc) if _adc is not None else None
    return std or "c++23"


def _adc_cxx_std_from_module(mod):
    """Norme C++ du module @p mod sous forme 'c++NN', ou None si indeterminable. Priorite a l'entier
    __cxx_std__ (baked par le build) ; sinon on extrait std=<__cplusplus> de la cle d'ABI."""
    n = getattr(mod, "__cxx_std__", None)
    if isinstance(n, int) and n in (20, 23):
        return "c++%d" % n
    # Fallback : parser "...;std=<__cplusplus>;..." de la cle d'ABI (vieux module sans __cxx_std__).
    abi_key = getattr(mod, "abi_key", None)
    if callable(abi_key):
        try:
            key = abi_key()
        except Exception:
            return None
        for tok in str(key).split(";"):
            if tok.startswith("std="):
                val = tok[len("std="):].rstrip("Ll")
                if val.isdigit():
                    return "c++23" if int(val) > 202002 else "c++20"
    return None


# --- Compilateur des .so DSL (frontiere ABI, pendant de loader_cxx_std) -------------------------
# BUG REEL corrige ici : dans un env conda active, `which c++` designe souvent un AUTRE compilateur
# que celui qui a construit _adc (vieux gcc/clang du PATH conda). Symptome : la compilation runtime
# du loader DSL production echoue avec l'erreur brute du compilateur ("error: invalid value 'c++23'
# in '-std=c++23'") ; et meme si elle passait, la cle d'ABI (qui encode __VERSION__ du compilateur,
# cf. abi_key.hpp) rejetterait le .so ("ABI incompatible"). Le SEUL compilateur garanti compatible
# est celui du build de _adc : CMake le bake (ADC_CXX_COMPILER -> _adc.__cxx_compiler__) et on le
# prefere ici au PATH. $ADC_CXX reste l'override conscient (toolchain conda choisi, wrapper...).
def _adc_module():
    """Le module d'extension _adc s'il est chargeable, sinon None (dsl.py reste utilisable seul)."""
    try:
        import _adc
        return _adc
    except Exception:
        try:
            from . import _adc
            return _adc
        except Exception:
            return None


def loader_cxx_compiler():
    """Chemin du compilateur qui a CONSTRUIT le module _adc (bake par CMake en __cxx_compiler__),
    ou None s'il est inconnu (vieux module, build manuel) ou absent de cette machine.

    macOS : CMake bake souvent le c++ INTERNE de la toolchain Xcode / CommandLineTools
    (.../XcodeDefault.xctoolchain/usr/bin/c++), qui invoque clang SANS sysroot SDK -> tout .so
    DSL echoue sur \"'string' file not found\". Le shim /usr/bin/c++ (xcrun) execute LE MEME
    clang en resolvant le SDK : meme __VERSION__, donc meme cle d'ABI -- on prefere donc le shim
    (piege et remede identiques a compile_loader des tests C++ natifs)."""
    import sys
    mod = _adc_module()
    cc = getattr(mod, "__cxx_compiler__", "") if mod is not None else ""
    if not (cc and os.path.isfile(cc) and os.access(cc, os.X_OK)):
        return None
    if sys.platform == "darwin" and (".xctoolchain/" in cc or "/CommandLineTools/" in cc) \
            and os.path.isfile("/usr/bin/c++"):
        return "/usr/bin/c++"
    return cc


def module_header_signature():
    """Signature d'en-tetes BAKEE dans le module _adc charge (token headers= de abi_key()), ou None
    si le module n'est pas chargeable / la cle absente ("unknown" d'un build manuel -> None aussi)."""
    mod = _adc_module()
    abi = getattr(mod, "abi_key", None) if mod is not None else None
    if not callable(abi):
        return None
    try:
        key = abi()
    except Exception:
        return None
    for tok in str(key).split(";"):
        if tok.startswith("headers="):
            sig = tok[len("headers="):]
            return sig if sig and sig != "unknown" else None
    return None


def resolve_auto_backend(include=None):
    """Politique du backend PAR DEFAUT (backend='auto', decision actee -- ADC-63).

    'production' (loader natif zero-copie, parite stricte add_block) DES QUE la parite
    toolchain avec le module _adc est etablie : module chargeable + compilateur bake connu +
    signature d'en-tetes de @p include == celle bakee dans le module. SINON 'aot' (defaut
    historique : host-marshale, fonctionne sans module ni parite). Jamais silencieux : renvoie
    (backend, raison) et les facades posent la raison sur CompiledModel.backend_auto_reason.
    Un backend EXPLICITE passe par l'appelant court-circuite cette politique (inchange)."""
    mod = _adc_module()
    if mod is None:
        return "aot", "module _adc non chargeable (le chemin production exige le module)"
    if not loader_cxx_compiler():
        return "aot", "compilateur du module inconnu (vieux module ou build manuel)"
    baked = module_header_signature()
    if not baked:
        return "aot", "signature d'en-tetes absente du module (build manuel)"
    try:
        inc = include if include is not None else adc_include()
        sig = adc_header_signature(inc)
    except Exception as e:  # en-tetes introuvables / illisibles -> defaut sur
        return "aot", "en-tetes adc introuvables pour la parite (%s)" % e
    if sig != baked:
        return "aot", ("en-tetes != module (rebatir le module ou pointer les en-tetes du build ; "
                       "production refuserait, cf. _check_headers_match_module)")
    return "production", "parite toolchain etablie (module + compilateur bake + en-tetes concordants)"


def _check_headers_match_module(include):
    """GARDE PRE-DLOPEN du chemin natif (bug reel) : si les en-tetes sous @p include ont change depuis
    le build de _adc (pull recent, autre clone...), le loader compile contre eux reference des
    signatures C++ que le VIEUX module n'exporte pas -> le dlopen d'add_native_block echoue AVANT le
    garde ABI, avec une erreur cryptique ("symbol not found in flat namespace '__ZN3adc6System13
    install_block...'"). On compare donc ICI, avant toute compilation, la signature d'en-tetes bakee
    dans le module a celle de l'arbre @p include, et on echoue avec un remede clair. No-op si le
    module n'est pas chargeable ou sans signature (build manuel : degradation historique)."""
    baked = module_header_signature()
    current = adc_header_signature(include)
    if baked is not None and current != baked:
        mod = _adc_module()
        so = getattr(mod, "__file__", "(inconnu)")
        raise RuntimeError(
            "adc.dsl : les en-tetes adc de %r NE CORRESPONDENT PAS a ceux avec lesquels le module "
            "_adc a ete construit (%s).\n"
            "  signature en-tetes actuelle : %s\n"
            "  signature bakee dans _adc   : %s\n"
            "Cause typique : `git pull` / en-tetes edites APRES le build du module -> le loader DSL "
            "referencerait des signatures C++ absentes du module (dlopen : 'symbol not found').\n"
            "Remede : REBATIR le module avec ces en-tetes :\n"
            "  cmake --preset python && cmake --build --preset python   (ou le build-py habituel)\n"
            "ou pointer ADC_INCLUDE vers les en-tetes du build qui a produit ce module."
            % (include, so, current[:16], baked[:16]))
    return current  # signature de l'arbre @p include, reutilisable (evite un 2e walk+sha256)


def check_compiled_matches_module(abi_key):
    """Garde PRE-DLOPEN au BRANCHEMENT d'un CompiledModel (add_equation -> add_native_block).

    COMPLEMENT INDISPENSABLE de _check_headers_match_module : sur un cache HIT, compile_native ne
    tourne pas (le .so sort du cache) et la garde de compilation ne protege donc pas ce chemin --
    un module _adc perime dlopen-erait le .so et echouerait sur le 'symbol not found' cryptique.
    Ici on compare la signature d'en-tetes EMBARQUEE dans la cle Python du .so ('<sig>|<cxx>|<std>',
    cf. _abi_key_python, recalculee sur l'arbre include ACTUEL a chaque compile()) a celle bakee
    dans le module. Pure comparaison de CHAINES (aucun re-hash). No-op si l'une des deux manque
    (vieux module, CompiledModel construit a la main)."""
    baked = module_header_signature()
    if baked is None or not abi_key:
        return
    so_sig = str(abi_key).split("|", 1)[0]
    if so_sig and so_sig != baked:
        raise RuntimeError(
            "adc : le modele compile (.so) a ete produit contre des en-tetes adc DIFFERENTS de ceux "
            "du module _adc charge (signature %s... vs %s... bakee dans le module).\n"
            "Cause typique : module _adc perime (bati avant un `git pull`) alors que le .so vient "
            "d'etre (re)compile sur les en-tetes a jour -- le dlopen echouerait avec un 'symbol not "
            "found' cryptique.\n"
            "Remede : REBATIR le module puis relancer :\n"
            "  cmake --preset python && cmake --build --preset python\n"
            "Diagnostic complet : python -c \"import adc; adc.doctor()\"."
            % (so_sig[:12], baked[:12]))


def _default_cxx(cxx=None):
    """Resolution CENTRALISEE du compilateur des .so DSL (tous backends). Priorite :
      1. cxx explicite (argument de l'appelant) ;
      2. $ADC_CXX (override d'environnement conscient) ;
      3. le compilateur qui a construit _adc (seul garanti ABI-compatible, cf. ci-dessus) ;
      4. c++ / g++ / clang++ du PATH (comportement historique, dernier recours)."""
    return (cxx or os.environ.get("ADC_CXX") or loader_cxx_compiler()
            or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))


# Orthographes historiques des memes niveaux de langue : les clang < 17 / gcc < 11 ne connaissent
# que 'c++2b'/'c++2a'. Meme niveau demande ; sur un VIEUX compilateur __cplusplus peut differer du
# module -> le cas echeant, rejet ABI explicite en aval (jamais d'UB silencieux).
_STD_ALIAS = {"c++23": "c++2b", "c++20": "c++2a"}


def _run_compile(cmd, what):
    """Lance la commande de compilation @p cmd en CAPTURANT stderr : en cas d'echec, leve une
    RuntimeError AUTONOME (commande + sortie compilateur + remedes) au lieu du CalledProcessError
    brut dont le message ne contient que la ligne de commande (bug reel : l'utilisateur ne voit
    qu'un 'returned non-zero exit status 1' noye dans le traceback)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip()
        out = (r.stdout or b"").decode(errors="replace").strip()  # MSVC cl ecrit les erreurs sur STDOUT
        err = (err + "\n" + out).strip() if out else err
        raise RuntimeError(
            "adc.dsl : la compilation du .so (%s) a echoue (exit %d).\n"
            "Commande : %s\n"
            "Sortie du compilateur :\n%s\n"
            "Pistes : `python -c \"import adc; adc.doctor()\"` diagnostique l'environnement "
            "(compilateur/norme/en-tetes) ; ADC_CXX force un compilateur precis."
            % (what, r.returncode, " ".join(cmd), err[:4000] or "(vide)"))


_probe_cache = {}  # (cc, std) -> std effectif : evite de re-prober en boucle (N modeles compiles)


def _probe_cxx_std(cc, std):
    """Verifie AVANT compilation que @p cc accepte -std=@p std (probe -fsyntax-only sur source vide).

    Renvoie le std EFFECTIF : @p std s'il passe, sinon son alias historique (c++23 -> c++2b) s'il
    passe, sinon leve une RuntimeError ACTIONNABLE (compilateur utilise, compilateur du build,
    solutions) au lieu de l'erreur brute du compilateur. Saute pour nvcc_wrapper (semantique -x
    differente ; chemin GPU explicite, deja gate par ADC_KOKKOS_CXX/ADC_KOKKOS_USE_NVCC_WRAPPER).
    Resultat memoise par (cc, std) : un seul probe meme si N modeles sont compiles a la suite."""
    import subprocess
    if "nvcc" in os.path.basename(cc or ""):
        return std
    if sys.platform == "win32":
        return std  # cl/clang-cl : sonde -fsyntax-only inapplicable ; std traduit en /std: au compile
    cached = _probe_cache.get((cc, std))
    if cached is not None:
        return cached

    def accepts(s):
        try:
            r = subprocess.run([cc, "-x", "c++", "-std=" + s, "-fsyntax-only", "-"],
                               input=b"", capture_output=True, timeout=60)
            return r.returncode == 0, (r.stderr or b"").decode(errors="replace")
        except Exception as exc:  # introuvable, non executable, timeout : meme diagnostic actionnable
            return False, str(exc)

    good, err = accepts(std)
    if good:
        _probe_cache[(cc, std)] = std
        return std
    alias = _STD_ALIAS.get(std)
    if alias:
        good_alias, _ = accepts(alias)
        if good_alias:
            _probe_cache[(cc, std)] = alias
            return alias
    baked = loader_cxx_compiler()
    raise RuntimeError(
        "adc.dsl : le compilateur %r ne supporte pas -std=%s (norme requise pour partager l'ABI du "
        "module _adc).\nSortie du compilateur :\n%s\n"
        "Compilateur du build de _adc : %s\n"
        "Solutions :\n"
        "  - utiliser le compilateur du build : export ADC_CXX=%r (ou cxx=... dans m.compile) ;\n"
        "  - macOS : mettre a jour Xcode / les Command Line Tools (AppleClang recent) ;\n"
        "  - conda : `conda install -c conda-forge cxx-compiler` (gcc>=13 / clang>=17) puis "
        "export ADC_CXX=$CONDA_PREFIX/bin/clang++ (macOS) ou $CONDA_PREFIX/bin/g++ (Linux).\n"
        "NB : un compilateur DIFFERENT de celui du build peut compiler mais etre ensuite rejete "
        "('ABI incompatible' : la cle d'ABI encode la version du compilateur) ; preferer celui du build."
        % (cc, std, (err or "").strip()[:800],
           baked or "(inconnu : module sans __cxx_compiler__, rebatir _adc pour le baker)",
           baked or "<chemin/du/compilateur/du/build>"))


# --- Cache de build hors source ----------------------------------------------
# Quand l'appelant ne fournit pas so_path, m.compile(...) ecrit la .so dans un cache PARTAGE hors
# source (jamais a cote du .cpp temporaire), indexe par une cle stable du modele : model_hash (formules
# + roles + n_aux + params) ET abi_key (signature des en-tetes + compilateur + std). Deux compilations
# du MEME modele (meme cle) reutilisent la .so en cache (cache HIT, pas de recompilation) ; changer le
# modele OU un parametre OU la toolchain change la cle -> nouvelle .so (cache MISS, recompilation). Le
# nom de fichier porte la cle, donc plusieurs variantes coexistent sans collision. cf. la meme idee
# dans adc_cases/common/native.py (cle d'ABI = compilateur + flags + signature des en-tetes).
def adc_cache_dir():
    """Dossier de cache des .so generees par m.compile() sans so_path explicite.

    $ADC_CACHE_DIR (override), sinon $XDG_CACHE_HOME/adc/dsl, sinon ~/.cache/adc/dsl. Cree au besoin.
    Hors source par construction (jamais dans l'arbre du depot), donc rien a ignorer cote git."""
    import os
    base = os.environ.get("ADC_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "adc", "dsl")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_so_path(model_hash, abi_key, backend, target, name):
    """Chemin .so en cache pour ce (model_hash, abi_key, backend, target, name).

    La cle de cache combine model_hash (le QUOI : formules/roles/params) et abi_key (le COMMENT :
    en-tetes + compilateur + std), plus backend/target/name qui changent le code emis (loader natif
    vs AOT vs JIT, System vs AmrSystem). Le nom de fichier est <model_hash[:16]>-<sha(reste)[:16]>.so :
    lisible (prefixe = identite du modele) et sans collision (suffixe = reste de la cle)."""
    import hashlib
    import os
    # _platform_cache_key : l'arch CPU + les optflags entrent dans la cle (un .so x86_64 ou
    # -march=native reutilise sur une autre machine via un cache partage = SIGILL silencieux).
    rest = "|".join((abi_key or "", backend or "", target or "", name or "",
                     _platform_cache_key())).encode()
    tag = hashlib.sha256(rest).hexdigest()[:16]
    fname = "%s-%s.so" % ((model_hash or "nohash")[:16], tag)
    return os.path.join(adc_cache_dir(), fname)


def _native_kokkos_root():
    """Racine Kokkos pour compiler les loaders DSL avec le MEME backend que le module _adc.

    adc_cpp est KOKKOS-ONLY : tout .so DSL qui inclut les en-tetes adc (aot, native) DOIT etre compile
    avec Kokkos (for_each.hpp #error sinon). Le root est lu depuis ADC_KOKKOS_ROOT / Kokkos_ROOT /
    KOKKOS_ROOT ; None s'il n'est pas trouve (l'appelant leve alors une erreur explicite)."""
    for key in ("ADC_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        root = os.environ.get(key)
        if root and os.path.isfile(os.path.join(root, "include", "Kokkos_Core.hpp")):
            return root
    return None


def _libomp_prefix():
    """Prefixe Homebrew libomp sur macOS (pour -Xpreprocessor -fopenmp), ou None. AppleClang ne gere
    pas `-fopenmp` seul : il faut -Xpreprocessor -fopenmp + l'include/lib de libomp (cf. CMakeLists)."""
    if sys.platform != "darwin":
        return None
    import subprocess
    try:
        p = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        prefix = p.stdout.strip()
        if prefix and os.path.isdir(os.path.join(prefix, "lib")):
            return prefix
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _native_feature_key():
    """Traits qui changent le code inline du loader natif et doivent donc entrer dans le cache (sinon
    un .so SERIE en cache serait reutilise sur un module Kokkos -> fallback serie silencieux).

    Au-dela du on/off, la cle empreinte KokkosCore_config.h de l'install visee : ce header genere
    encode la VERSION de Kokkos ET ses backends actifs. Sans lui, changer de Kokkos entre deux runs
    (mise a jour, bascule Serial->OpenMP au meme prefix) n'invalidait pas le cache -> reutilisation
    d'un .so compile contre l'ancien Kokkos."""
    root = _native_kokkos_root()
    if not root:
        return "kokkos=off"
    import hashlib
    cfg = os.path.join(root, "include", "KokkosCore_config.h")
    try:
        with open(cfg, "rb") as f:
            tag = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        tag = "unknown"
    return "kokkos=on;kcfg=%s" % tag


def _platform_cache_key():
    """Traits MACHINE qui changent le code binaire du .so sans changer la cle d'ABI C++ (__VERSION__
    est identique cross-arch) : architecture CPU + flags d'optimisation. Sans eux dans la cle de
    cache, un .so x86_64 (Rosetta) ou -march=native serait reutilise sur une autre machine/arch via
    un cache partage (NFS, home synchronise) -> SIGILL (illegal instruction) ou dlopen cryptique."""
    import platform
    return "arch=%s;optflags=%s" % (platform.machine(),
                                    os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG"))


def _warn_kokkos_parity():
    """Avertit (sans bloquer) quand le BACKEND du loader natif divergerait de celui du module _adc :
    - module Kokkos + loader serie (ADC_KOKKOS_ROOT absent) -> le bloc DSL tombe sur le fallback
      SERIE en silence : zero-copie mais NE SCALE PAS threads/GPU (mesure ROMEO : DSL warm invariant
      threads=1/4/8). C'est une degradation de PERF muette, pas un crash -> warning explicite.
    - module serie + ADC_KOKKOS_ROOT defini -> le loader instancierait -DADC_HAS_KOKKOS contre un
      module qui ne l'est pas (layouts allocator/types divergents, non couverts par la cle d'ABI).
    Source de verite : _adc.__has_kokkos__ (bake par le build) vs _native_kokkos_root() (env)."""
    import warnings
    mod = _adc_module()
    has = getattr(mod, "__has_kokkos__", None) if mod is not None else None
    root = _native_kokkos_root()
    if has is True and root is None:
        warnings.warn(
            "adc.dsl : le module _adc est compile AVEC Kokkos mais ADC_KOKKOS_ROOT n'est pas defini "
            "-> le bloc DSL 'production' sera compile en SERIE (il tournera, mais ne scalera pas "
            "avec les threads/GPU). Definir ADC_KOKKOS_ROOT=<install Kokkos du build> pour la parite.",
            RuntimeWarning, stacklevel=3)
    elif has is False and root is not None:
        warnings.warn(
            "adc.dsl : ADC_KOKKOS_ROOT est defini mais le module _adc est SERIE (compile sans "
            "-DADC_USE_KOKKOS=ON) -> le loader serait compile avec Kokkos contre un module qui ne "
            "l'est pas (layouts memoire divergents, non couverts par la cle d'ABI). Retirer "
            "ADC_KOKKOS_ROOT ou rebatir _adc avec Kokkos (preset python-parallel).",
            RuntimeWarning, stacklevel=3)


def _env_truthy(value):
    return str(value or "").strip().lower() in ("1", "on", "true", "yes", "y")


def _native_kokkos_compiler(cxx):
    """Compilateur effectif du loader natif. CUDA doit etre EXPLICITE (ADC_KOKKOS_CXX=<nvcc_wrapper>
    ou ADC_KOKKOS_USE_NVCC_WRAPPER=1) : beaucoup d'installs Kokkos OpenMP fournissent aussi un
    nvcc_wrapper, le choisir d'office casserait les jobs CPU sans nvcc. Pour Kokkos OpenMP, l'hote suffit."""
    if cxx:
        return cxx
    env = os.environ.get("ADC_KOKKOS_CXX")
    if env:
        return env
    root = _native_kokkos_root()
    wrapper = os.path.join(root, "bin", "nvcc_wrapper") if root else ""
    if wrapper and os.path.exists(wrapper) and _env_truthy(os.environ.get("ADC_KOKKOS_USE_NVCC_WRAPPER")):
        return wrapper
    # Fallback centralise : $ADC_CXX, puis le compilateur du build de _adc (seul ABI-compatible
    # garanti pour un loader natif), puis le PATH (historique). cf. _default_cxx.
    return _default_cxx(None)


def _adc_import_lib():
    """(Windows, ADC-100) Chemin de l'import library _adc.lib (symboles System ADC_EXPORT) contre
    laquelle linker la .dll DSL. Cherchee a cote du module _adc. None si absente."""
    mod = _adc_module()
    if mod is None:
        return None
    d = os.path.dirname(getattr(mod, "__file__", "") or "")
    cand = os.path.join(d, "_adc.lib")
    return cand if os.path.exists(cand) else None


def _native_kokkos_flags():
    """Flags compile/link pour que le loader DSL production instancie add_compiled_model AVEC Kokkos.

    Le loader contient les templates header-only (make_block / assemble_rhs / for_each_cell). Compile
    sans ADC_HAS_KOKKOS alors que _adc l'est AVEC Kokkos, le bloc DSL reste zero-copie mais ses kernels
    sont instancies sur le fallback SERIE (ne scale pas threads/GPU). ADC_KOKKOS_ROOT force la parite."""
    root = _native_kokkos_root()
    if not root:
        return [], []
    inc = os.path.join(root, "include")
    if sys.platform == "win32":
        # MSVC/clang-cl : Kokkos en DLL PARTAGEE -> linker l'import lib kokkoscore.lib (UN seul runtime ;
        # _adc charge la meme kokkoscore.dll). cl accepte -D/-I. Pas de -fopenmp/-ldl/-pthread (POSIX).
        return (["-DADC_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc],
                [os.path.join(root, "lib", "kokkoscore.lib")])
    compile_flags = ["-DADC_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc]
    # NE PAS linker libkokkos* DANS le .so : le module _adc a deja charge le runtime Kokkos, un
    # SINGLETON (registre global des execution spaces), et add_native_block le promeut en portee
    # globale (RTLD_GLOBAL). Linker une 2e copie de Kokkos dans le loader donne deux runtimes : le
    # calcul tourne, mais a la sortie Kokkos::finalize() abort "Execution space instance to be removed
    # couldn't be found!" (SIGABRT, atexit). On laisse donc les symboles Kokkos INDEFINIS dans le .so
    # (resolus au chargement contre le module, comme install_block/grid_context). Seuls -fopenmp (le
    # backend OpenMP du noyau genere) + -ldl/-pthread restent. Validation ROMEO : DSL warm scale et
    # sortie propre (exit 0), ratio ~0.96x les briques.
    link_flags = ["-ldl", "-pthread"]
    if "nvcc_wrapper" not in os.path.basename(_native_kokkos_compiler(None) or ""):
        # OpenMP requis pour l'espace d'exec Kokkos OpenMP (inoffensif/ignore sous Kokkos Serial).
        if sys.platform == "darwin":
            # macOS / AppleClang : `-fopenmp` seul est rejete -> -Xpreprocessor -fopenmp (+ include
            # libomp Homebrew si present). On NE LIE PAS libomp (-lomp) dans le .so : une 2e copie de
            # libomp donne DEUX runtimes OpenMP ("mutex lock failed: Invalid argument" a l'execution).
            # Les symboles omp_*/__kmpc_* se resolvent au chargement (flat namespace, -undefined
            # dynamic_lookup pose par compile_aot/compile_native) contre le libomp deja charge par _adc.
            libomp = _libomp_prefix()
            compile_flags += ["-Xpreprocessor", "-fopenmp"]
            if libomp is not None:
                compile_flags += ["-I", os.path.join(libomp, "include")]
        else:
            # ELF/Linux : libgomp est partagee par soname (pas de double-runtime), -fopenmp des deux cotes.
            compile_flags.append("-fopenmp")
            link_flags.append("-fopenmp")
    return compile_flags, link_flags


def adc_loader_build_flags(cxx=None):
    """Drapeaux pour compiler HORS CMake un .so qui INCLUT les en-tetes adc et sera charge dans le
    module _adc (loaders DSL, tests d'ABI). adc_cpp etant Kokkos-only, le .so DOIT etre compile avec
    Kokkos (for_each.hpp #error sinon). Renvoie (compilateur, flags_compile, flags_lien) : Kokkos +
    (macOS) -undefined dynamic_lookup. Les symboles Kokkos restent INDEFINIS, resolus au chargement
    contre le runtime Kokkos deja charge par _adc (pas de 2e copie). Leve si aucun Kokkos installe n'est
    visible via ADC_KOKKOS_ROOT / Kokkos_ROOT (Serial suffit sur CPU)."""
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "adc_loader_build_flags : adc_cpp est Kokkos-only -- pointe un Kokkos installe via "
            "ADC_KOKKOS_ROOT (ou Kokkos_ROOT), p.ex. `export ADC_KOKKOS_ROOT=/chemin/vers/kokkos`.")
    cc = _native_kokkos_compiler(cxx)
    cflags, lflags = _native_kokkos_flags()
    if sys.platform == "darwin":
        cflags = list(cflags) + ["-undefined", "dynamic_lookup"]
    return cc, cflags, lflags


# --- Canal aux : disposition canonique --------------------------------------
# Les champs auxiliaires nommes (aux('...')) sont des COMPOSANTES a indice FIXE du canal aux
# (cf. adc::Aux / kAuxBaseComps cote C++). phi/grad_x/grad_y = contrat de BASE (3 composantes) ;
# les suivants (B_z, ...) ELARGISSENT le canal -> la brique generee declare alors n_aux pour que
# le systeme dimensionne et peuple le canal partage (cf. CompositeModel::n_aux, ensure_aux_width).
#
# DUPLICATION INHERENTE C++ <-> Python : la table ci-dessous DOIT rester le MIROIR de la source
# unique C++ ADC_AUX_FIELDS (include/adc/core/state.hpp), d'ou sont generes load_aux (lecture
# device) et le marshaling hote (python/system.cpp). Python ne lit pas les en-tetes C++, donc on
# ne peut pas la generer : ajouter un champ aux extra = 1 ligne ici ET 1 ligne dans ADC_AUX_FIELDS,
# avec le MEME {nom, indice}. C'est le seul recopie restant ; les 3 sites C++ sont desormais unifies.
AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
AUX_BASE_COMPS = 3

# Champs aux NOMMES par le modele (ADC-70 phase 1) : m.aux_field("nom"). Composantes a partir de
# AUX_NAMED_BASE (= 5, juste apres T_e=4) -- le k-ieme nom declare est la composante AUX_NAMED_BASE + k,
# lue en C++ via aux.extra_field(k). MIROIRS de kAuxNamedBase / kAuxMaxExtra (include/adc/core/state.hpp,
# source unique C++). Decouple les noms utilisateur du canal canonique : B_z / T_e gardent leurs indices
# 3 / 4 et leurs chemins dedies (set_magnetic_field / set_electron_temperature_from).
AUX_NAMED_BASE = 5
AUX_NAMED_MAX = 4  # nombre maximal de champs aux nommes par modele (= kAuxMaxExtra cote C++)

# Borne du nombre de parametres RUNTIME par bloc (P7-b). MIROIR de kMaxRuntimeParams
# (include/adc/runtime/runtime_params.hpp) : le porteur C++ RuntimeParams a un tableau de cette taille
# FIXE (device-copiable sans allocation), donc un modele depassant la borne est rejete au codegen.
_K_MAX_RUNTIME_PARAMS = 32


def aux_n_aux(aux_names):
    """Largeur du canal aux requise par ces champs CANONIQUES : max(3, plus grand indice + 1).
    Leve ValueError sur un nom inconnu (un champ aux canonique DOIT etre une composante de adc::Aux)."""
    w = AUX_BASE_COMPS
    for nm in aux_names:
        if nm not in AUX_CANONICAL:
            raise ValueError("champ aux '%s' inconnu : attendus %s (composantes de adc::Aux)"
                             % (nm, sorted(AUX_CANONICAL)))
        w = max(w, AUX_CANONICAL[nm] + 1)
    return w


def aux_total_n_aux(aux_names, aux_extra_names):
    """Largeur TOTALE du canal aux : max de la largeur canonique (aux_n_aux) et, si des champs NOMMES
    (aux_field) sont declares, AUX_NAMED_BASE + nombre de noms (le dernier nomme = composante
    AUX_NAMED_BASE + len-1). Sans champ nomme -> aux_n_aux (chemin historique, bit-identique)."""
    w = aux_n_aux(aux_names)
    if aux_extra_names:
        w = max(w, AUX_NAMED_BASE + len(aux_extra_names))
    return w


# --- Roles physiques : nom de variable -> VariableRole ----------------------
# Mapping CANONIQUE nom -> role physique (cf. adc::VariableRole / role_name cote C++). Permet a une
# brique generee de DECLARER le SENS de ses composantes (densite, qte de mvt, energie...) au lieu de
# roles vides, pour que les couplages inter-especes (System::add_collision / add_thermal_exchange)
# resolvent par index_of(role) plutot que par un indice litteral. Les noms usuels des modeles fluides
# (rho, rho_u, u, p, E, n...) sont reconnus ; un nom inconnu reste 'Custom'. Un modele peut imposer
# ses roles explicitement (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# pour un layout non standard. Cle = nom EXACT de la variable, valeur = membre de adc::VariableRole.
CANONICAL_ROLES = {
    "rho": "Density", "n": "Density", "density": "Density",
    "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
    "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
    "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
    "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
    "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
    "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
    "p": "Pressure", "pressure": "Pressure",
    "T": "Temperature", "temperature": "Temperature",
}


def role_of(name):
    """Role physique CANONIQUE du nom @p name (membre de adc::VariableRole), 'Custom' si inconnu."""
    return CANONICAL_ROLES.get(name, "Custom")


def roles_for(names, override=None):
    """Liste des roles (membres adc::VariableRole) paralleles a @p names. @p override (optionnel) :
    liste de meme longueur fixant explicitement les roles (chaine 'Density'... ou None pour retomber
    sur le mapping canonique du nom). Sert aux layouts non standard ou les noms ne suffisent pas."""
    if override is None:
        return [role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles : %d roles pour %d variables" % (len(override), len(names)))
    return [(r if r is not None else role_of(nm)) for nm, r in zip(names, override)]


# --- Arbre d'expressions ----------------------------------------------------
class Expr:
    """Noeud d'expression symbolique. Les operateurs construisent l'arbre ; eval(env) l'applique a
    des tableaux numpy (env : nom -> tableau ou scalaire)."""

    def __add__(self, o): return Add(self, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self)
    def __sub__(self, o): return Sub(self, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self)
    def __mul__(self, o): return Mul(self, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self)
    def __truediv__(self, o): return Div(self, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self)
    def __neg__(self): return Neg(self)
    def __pos__(self): return self  # +expr = identite (l'API CoupledSource ecrit +k*ne*ng)
    def __abs__(self): return Abs(self)  # abs(expr) -> |expr| (valeur absolue, p.ex. |lambda| de Roe)
    def __pow__(self, o): return Pow(self, _wrap(o))

    def eval(self, env): raise NotImplementedError
    def deps(self): return set()
    def __repr__(self): return self._str()
    def _str(self): return "?"


def _wrap(o):
    if isinstance(o, Expr):
        return o
    # Un Param expose son NOEUD d'arbre interne (_node : Const pour 'const', RuntimeParamRef pour
    # 'runtime'). On le promeut par ce noeud, PAS par float(o) : sinon sqrt(param_runtime) /
    # dsl.sqrt(param) inlinerait la valeur de declaration (Const) au lieu d'emettre params.get(...).
    node = getattr(o, "_node", None)
    if isinstance(node, Expr):
        return node
    return Const(float(o))


class Const(Expr):
    def __init__(self, value): self.value = float(value)
    def eval(self, env): return self.value
    def to_cpp(self): return repr(self.value)
    def _str(self): return repr(self.value)


class Var(Expr):
    """Variable nommee : conservative, primitive, auxiliaire (champ) ou constante."""

    def __init__(self, name, kind): self.name = name; self.kind = kind
    def eval(self, env):
        if self.name not in env:
            raise KeyError("variable '%s' (%s) absente de l'environnement" % (self.name, self.kind))
        return env[self.name]
    def deps(self): return {self.name}
    def to_cpp(self): return self.name
    def _str(self): return self.name


class _Bin(Expr):
    op = "?"
    def __init__(self, a, b): self.a = a; self.b = b
    def deps(self): return self.a.deps() | self.b.deps()
    def to_cpp(self): return "(%s %s %s)" % (self.a.to_cpp(), self.op, self.b.to_cpp())
    def _str(self): return "(%s %s %s)" % (self.a, self.op, self.b)


class Add(_Bin):
    op = "+"
    def eval(self, env): return self.a.eval(env) + self.b.eval(env)


class Sub(_Bin):
    op = "-"
    def eval(self, env): return self.a.eval(env) - self.b.eval(env)


class Mul(_Bin):
    op = "*"
    def eval(self, env): return self.a.eval(env) * self.b.eval(env)


class Div(_Bin):
    op = "/"
    def eval(self, env): return self.a.eval(env) / self.b.eval(env)


class Pow(_Bin):
    op = "**"
    def eval(self, env): return self.a.eval(env) ** self.b.eval(env)
    def to_cpp(self): return "std::pow(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())


class Neg(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env): return -self.a.eval(env)
    def deps(self): return self.a.deps()
    def to_cpp(self): return "(-%s)" % self.a.to_cpp()
    def _str(self): return "(-%s)" % self.a


class Sqrt(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env): return np.sqrt(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::sqrt(%s)" % self.a.to_cpp()
    def _str(self): return "sqrt(%s)" % self.a


def sqrt(x):
    """Racine carree symbolique."""
    return Sqrt(_wrap(x))


class Abs(Expr):
    """Valeur absolue |a| (p.ex. |lambda_k| d'une dissipation de Roe). Emise std::fabs au codegen
    (egale au ternaire a<0?-a:a hors -0.0). Non derivable par dsl.diff (pas de noeud signe)."""
    def __init__(self, a): self.a = a
    def eval(self, env): return np.abs(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::fabs(%s)" % self.a.to_cpp()
    def _str(self): return "abs(%s)" % self.a


def abs_(x):
    """Valeur absolue symbolique (equivalent de abs(expr) ; nom suffixe pour ne pas masquer abs)."""
    return Abs(_wrap(x))


class StateRef(Expr):
    """Marqueur d'ETAT pour la dissipation de Roe a DEUX etats (m.roe_dissipation) : la
    sous-expression encadree s'evalue sur l'etat GAUCHE UL (side='L', dsl.left) ou DROIT UR
    (side='R', dsl.right). A la codegen du hook roe_dissipation(UL, AL, UR, AR, dir), left(e) emet e
    avec les locales calculees depuis UL, right(e) depuis UR. N'a de sens QUE dans les lignes
    fournies a m.roe_dissipation : l'interprete numpy ne le traite pas (la dissipation deux etats est
    compilee en C++, pas evaluee a l'hote). deps() = deps de la sous-expression (controle des
    dependances)."""

    def __init__(self, side, expr):
        if side not in ("L", "R"):
            raise ValueError("StateRef : side doit etre 'L' (UL) ou 'R' (UR), recu %r" % (side,))
        self.side = side
        self.expr = _wrap(expr)

    def deps(self):
        return self.expr.deps()

    def eval(self, env):
        raise NotImplementedError(
            "StateRef (dsl.left / dsl.right) ne s'evalue pas par l'interprete numpy : la dissipation "
            "de Roe a deux etats est EMISE en C++ (m.roe_dissipation), pas interpretee a l'hote.")

    def _str(self):
        return "%s(%s)" % ("left" if self.side == "L" else "right", self.expr)


def left(expr):
    """Marque @p expr comme evaluee sur l'etat GAUCHE UL (dissipation de Roe, m.roe_dissipation)."""
    return StateRef("L", expr)


def right(expr):
    """Marque @p expr comme evaluee sur l'etat DROIT UR (dissipation de Roe, m.roe_dissipation)."""
    return StateRef("R", expr)


class RuntimeParamRef(Expr):
    """Reference a un parametre RUNTIME (P7-b) dans l'arbre d'expressions. A la difference d'un Const
    (param const inline EN DUR), ce noeud emet `params.get(<indice>)` au codegen : la brique generee
    LIT la valeur dans son membre adc::RuntimeParams au lieu de l'avoir cuite. La valeur peut donc etre
    CHANGEE au runtime sans recompiler le .so (cf. include/adc/runtime/runtime_params.hpp).

    @c index : indice STABLE du parametre dans le bloc RuntimeParams (attribue par le modele a la
    compilation, ordre trie des noms) ; -1 tant qu'il n'est pas assigne. @c value : valeur de
    DECLARATION (utilisee par l'interprete numpy eval et comme defaut du membre genere -> sans appel
    set au runtime, le bloc se comporte comme avec un param const de cette valeur).

    Cle structurelle de CSE (cf. _key) : le NOM (deux refs au meme param runtime partagent la meme
    locale CSE) ; la valeur de declaration n'entre pas dans la cle (elle est runtime, pas structurelle)."""

    def __init__(self, name, value, index=-1):
        self.name = name
        self.value = float(value)
        self.index = index

    def eval(self, env):
        # Interprete numpy (proto hote / debug) : la valeur de declaration tient lieu de valeur courante
        # (le chemin numpy ne passe pas par RuntimeParams ; il sert au prototypage, pas a la production).
        return self.value

    def deps(self):
        # Un parametre runtime n'est PAS une variable d'environnement (cons/prim/aux) : il vient du canal
        # RuntimeParams, donc rien a verifier dans check() (comme un Const).
        return set()

    def to_cpp(self):
        if self.index < 0:
            raise RuntimeError(
                "RuntimeParamRef('%s') : indice non assigne au codegen (appeler la compilation via "
                "dsl.Model qui attribue les indices runtime)" % self.name)
        return "params.get(%d)" % self.index

    def _str(self):
        return "rparam(%s)" % self.name


# --- Elimination des sous-expressions communes (CSE) -------------------------
# Le codegen inline chaque sous-expression a chaque apparition (H, c... recalcules). La CSE detecte
# les sous-expressions COMPOUND (non feuilles) apparaissant plusieurs fois et les sort en variables
# locales 'cseK_', en ordre de dependance (les plus petites d'abord). Repose sur une cle STRUCTURELLE
# par noeud : deux sous-arbres identiques ont la meme cle, donc la meme locale.
def _children(e):
    if isinstance(e, _Bin):
        return (e.a, e.b)
    if isinstance(e, (Neg, Sqrt, Abs)):
        return (e.a,)
    if isinstance(e, StateRef):
        return (e.expr,)  # marqueur left/right : un seul enfant (decouverte de params runtime, etc.)
    return ()


def _key(e):
    if isinstance(e, Const):
        return ("const", e.value)
    if isinstance(e, RuntimeParamRef):
        return ("rparam", e.name)  # cle = nom : deux refs au meme param runtime partagent la locale CSE
    if isinstance(e, Var):
        return ("var", e.name)
    if isinstance(e, Neg):
        return ("neg", _key(e.a))
    if isinstance(e, Sqrt):
        return ("sqrt", _key(e.a))
    if isinstance(e, Abs):
        return ("abs", _key(e.a))
    if isinstance(e, StateRef):
        return ("state", e.side, _key(e.expr))  # defensif : les lignes Roe ne passent pas par la CSE
    return (e.op, tuple(_key(c) for c in _children(e)))  # _Bin (Add/Sub/Mul/Div/Pow)


def _cpp_expand(e, cse_map):
    """C++ du noeud e en developpant SON niveau ; les enfants passent par _cpp_cse (-> locales CSE)."""
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()  # params.get(<indice>) : lit le membre RuntimeParams de la brique
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_cse(e.a, cse_map), _cpp_cse(e.b, cse_map))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_cse(e.a, cse_map), e.op, _cpp_cse(e.b, cse_map))
    raise TypeError("expression non geree par le codegen : %r" % (e,))


def _cpp_cse(e, cse_map):
    """C++ de e ; si e correspond a une locale CSE deja definie, renvoie son nom."""
    k = _key(e)
    if k in cse_map:
        return cse_map[k]
    return _cpp_expand(e, cse_map)


def _cse_emit(roots, real, indent):
    """Retourne (lignes_de_locales, [C++ par racine]). Les sous-expressions compound vues >= 2 fois
    deviennent des locales ``cseK_``. roots : liste d'Expr.

    Memo par id() : un OBJET Expr partage (DAG, p.ex. intermediaires reutilises d'un gros modele)
    n'est parcouru qu'une fois ; ses occurrences suivantes re-creditent le compteur de son
    sous-arbre sans re-marche. Sans memo le parcours coute O(nombre de CHEMINS racine-feuille),
    exponentiel sur un DAG profond (flux polynomial de polynomes). Equivalence stricte avec la
    re-marche historique : memes comptages, memes tailles, meme ORDRE d'insertion des cles
    (post-ordre de la premiere visite) -> C++ emis bit-identique."""
    counts, rep, size = {}, {}, {}
    memo = {}  # id(e) -> (taille, {cle: occurrences} du sous-arbre, en post-ordre d'insertion)

    def visit(e):
        if isinstance(e, (Const, Var)):
            return 1, None
        sub = memo.get(id(e))
        if sub is None:
            k = _key(e)
            s, cnt = 1, {}
            for c in _children(e):
                cs, ccnt = visit(c)
                s += cs
                if ccnt:
                    for ck, cc in ccnt.items():
                        cnt[ck] = cnt.get(ck, 0) + cc
            cnt[k] = cnt.get(k, 0) + 1  # post-ordre : enfants d'abord, comme la re-marche historique
            rep.setdefault(k, e)
            size[k] = s
            sub = (s, cnt)
            memo[id(e)] = sub
        return sub

    for r in roots:
        _, cnt = visit(r)
        if cnt:
            for k, c in cnt.items():
                counts[k] = counts.get(k, 0) + c
    cand = sorted((k for k, c in counts.items() if c >= 2), key=lambda k: size[k])
    cse_map, lines = {}, []
    for i, k in enumerate(cand):
        name = "cse%d_" % i
        lines.append("%sconst %s %s = %s;" % (indent, real, name, _cpp_expand(rep[k], cse_map)))
        cse_map[k] = name
    return lines, [_cpp_cse(r, cse_map) for r in roots]


# --- Differentiation symbolique (autodiff de l'arbre Expr) -------------------
# dsl.diff(expr, var) derive l'arbre noeud par noeud : linearite (+, -), produit (a*b)' = a'b + ab',
# quotient (a/b)' = (a'b - ab')/b^2, puissance (a^n)' = n a^(n-1) a' (exposant constant), racine
# sqrt(a)' = a'/(2 sqrt(a)), oppose. Sert a construire le Jacobien de flux A = dF/dU (flux_jacobian)
# que l'utilisateur emploie pour ecrire sa dissipation de Roe (m.roe_dissipation). Une primitive
# DEFINIE est derivee PAR SA DEFINITION (regle de chaine) ; une occurrence NON derivee reste un
# symbole (emission lisible), seule la DERIVEE descend jusqu'aux conservatives. Noeud inconnu ->
# NotImplementedError (jamais de zero silencieux). Simplifications minimales (0*x, 1*x, x+0).
def _is_const(e, val=None):
    """Vrai si e est une constante numerique (Const) ; si @p val est donne, egale a val."""
    return isinstance(e, Const) and (val is None or e.value == val)


def _s_add(a, b):
    if _is_const(a, 0.0):
        return b
    if _is_const(b, 0.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value + b.value)
    return Add(a, b)


def _s_neg(a):
    if _is_const(a, 0.0):
        return Const(0.0)
    if isinstance(a, Const):
        return Const(-a.value)
    if isinstance(a, Neg):
        return a.a
    return Neg(a)


def _s_sub(a, b):
    if _is_const(b, 0.0):
        return a
    if _is_const(a, 0.0):
        return _s_neg(b)
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value - b.value)
    return Sub(a, b)


def _s_mul(a, b):
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return Const(0.0)
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value * b.value)
    return Mul(a, b)


def _s_div(a, b):
    if _is_const(a, 0.0):
        return Const(0.0)
    if _is_const(b, 1.0):
        return a
    return Div(a, b)


def _s_pow(a, b):
    # b : exposant (Expr), ici suppose INDEPENDANT de la variable de derivation.
    if _is_const(b, 0.0):
        return Const(1.0)
    if _is_const(b, 1.0):
        return a
    return Pow(a, b)


def diff(expr, var, defs=None):
    """Derivee symbolique de @p expr par rapport a @p var (nom de variable ou Var).

    @p defs (optionnel) : dictionnaire {nom de primitive: Expr de definition}. Quand la derivation
    rencontre une primitive DEFINIE, elle derive sa DEFINITION (regle de chaine) -- les primitives
    sont developpees jusqu'aux conservatives sans substitution manuelle. Une primitive de meme nom
    que @p var est traitee comme la variable independante (derivee 1). Sans defs, toute variable
    autre que @p var est independante (derivee 0).

    @return un Expr simplifie minimalement (0*x, 1*x, x+0, ... elimines pour une emission lisible).
    Leve NotImplementedError sur un noeud non derivable (en nommant son type) ou une puissance dont
    l'exposant depend de @p var (necessiterait un logarithme, noeud absent du DSL)."""
    target = var.name if isinstance(var, Var) else str(var)
    d = defs or {}

    def go(e):
        if isinstance(e, Const):
            return Const(0.0)
        if isinstance(e, RuntimeParamRef):
            return Const(0.0)  # parametre runtime : constant vis-a-vis de l'etat conservatif
        if isinstance(e, Var):
            if e.name == target:
                return Const(1.0)
            if e.name in d:
                return go(d[e.name])  # primitive definie -> derivee de sa definition (chaine)
            return Const(0.0)         # autre variable, independante de var
        if isinstance(e, Add):
            return _s_add(go(e.a), go(e.b))
        if isinstance(e, Sub):
            return _s_sub(go(e.a), go(e.b))
        if isinstance(e, Mul):
            return _s_add(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
        if isinstance(e, Div):
            num = _s_sub(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
            return _s_div(num, _s_mul(e.b, e.b))
        if isinstance(e, Neg):
            return _s_neg(go(e.a))
        if isinstance(e, Sqrt):
            return _s_div(go(e.a), _s_mul(Const(2.0), Sqrt(e.a)))
        if isinstance(e, Abs):
            # d|u| = (u / |u|) u' -- derivee exacte hors du pli u = 0 (les planchers lisses
            # max(x, eps) = ((x+eps) + |x-eps|)/2 des modeles 'robust' y donnent exactement
            # l'indicatrice attendue) ; AU pli, u/|u| vaut NaN : singularite de mesure nulle,
            # documentee (comme la division des quotients).
            return _s_mul(_s_div(e.a, Abs(e.a)), go(e.a))
        if isinstance(e, Pow):
            if not _is_const(go(e.b), 0.0):
                raise NotImplementedError(
                    "dsl.diff : derivee de a**b avec exposant dependant de '%s' (necessite un "
                    "logarithme, noeud absent du DSL)" % target)
            # exposant constant vis-a-vis de var : (a^b)' = b a^(b-1) a'
            return _s_mul(_s_mul(e.b, _s_pow(e.a, _s_sub(e.b, Const(1.0)))), go(e.a))
        raise NotImplementedError("dsl.diff : noeud non derivable %s (%r)" % (type(e).__name__, e))

    return go(_wrap(expr))


def _dir_key(direction):
    """Normalise une direction en 'x' / 'y' (accepte 0/'x'/'X' et 1/'y'/'Y'). Leve sinon."""
    if direction in (0, "x", "X"):
        return "x"
    if direction in (1, "y", "Y"):
        return "y"
    raise ValueError("direction invalide %r (attendu 0/'x' ou 1/'y')" % (direction,))


# --- Dissipation de Roe FOURNIE par l'utilisateur (m.roe_dissipation) ---------
# Les lignes d_i sont ecrites en termes de left(...)/right(...) de variables/primitives + constantes.
# _roe_validate controle qu'aucune variable n'apparait HORS marqueur (etat indetermine) et qu'aucun
# marqueur n'est imbrique ; _cpp_roe rend le C++ en resolvant left/right par un prefixe de locale
# (L_ pour UL, R_ pour UR).
def _roe_validate(e, in_marker):
    """Controle structurel d'une ligne de roe_dissipation. Leve ValueError si une variable est hors
    marqueur left()/right() (etat indetermine) ou si un marqueur est imbrique. Const / parametre
    runtime : autorises partout (sans etat). N'evalue rien (utilisable avant l'attribution des
    indices runtime)."""
    if isinstance(e, StateRef):
        if in_marker:
            raise ValueError("m.roe_dissipation : marqueur left()/right() imbrique interdit "
                             "(une sous-expression appartient a un seul etat)")
        _roe_validate(e.expr, True)
        return
    if isinstance(e, Var):
        if not in_marker:
            raise ValueError(
                "m.roe_dissipation : variable '%s' hors marqueur ; encadrer chaque variable ou "
                "primitive par dsl.left(...) (etat UL) ou dsl.right(...) (etat UR)" % e.name)
        return
    if isinstance(e, (Const, RuntimeParamRef)):
        return
    for c in _children(e):
        _roe_validate(c, in_marker)


def _cpp_roe(e, prefix):
    """C++ d'une expression de ligne roe_dissipation. @p prefix : None au niveau racine (aucun etat
    actif -> une variable nue est une erreur), 'L_' / 'R_' dans un marqueur (les variables prennent
    ce prefixe de locale). Sert AUSSI a rendre une definition de primitive avec un prefixe d'etat
    (e ne contient alors pas de StateRef). Suppose _roe_validate deja passe (erreurs defensives)."""
    if isinstance(e, StateRef):
        if prefix is not None:
            raise ValueError("m.roe_dissipation : marqueur left()/right() imbrique interdit")
        return _cpp_roe(e.expr, e.side + "_")
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()
    if isinstance(e, Var):
        if prefix is None:
            raise ValueError("m.roe_dissipation : variable '%s' hors marqueur left()/right()"
                             % e.name)
        return prefix + e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_roe(e.a, prefix), _cpp_roe(e.b, prefix))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_roe(e.a, prefix), e.op, _cpp_roe(e.b, prefix))
    raise TypeError("m.roe_dissipation : expression non geree par le codegen : %r" % (e,))


# --- Modele hyperbolique declaratif -----------------------------------------
class HyperbolicModel:
    """Modele hyperbolique ecrit en FORMULES : variables conservatives, primitives (definies par
    des expressions), flux, valeurs propres, source, contribution elliptique. cf. module docstring."""

    def __init__(self, name):
        self.name = name
        self.cons_names = []
        self.prim_defs = {}     # nom -> Expr (en fonction des cons / prims precedents / aux)
        self.aux_names = []      # champs aux CANONIQUES lus (phi/grad/B_z/T_e), cf. AUX_CANONICAL
        self.aux_extra_names = []  # champs aux NOMMES (aux_field) : ordre = indice AUX_NAMED_BASE + k
        self._flux = {}         # "x" / "y" -> liste d'Expr (une par composante conservative)
        self._eig = {}          # "x" / "y" -> liste d'Expr (valeurs propres)
        self._wave_speeds = None  # {"x"/"y": (smin Expr, smax Expr)} : vitesses SIGNEES explicites
                                  # (set_wave_speeds) ; None = derive des eigenvalues si 'p' (historique)
        self._ws_jacobian = None  # {"x"/"y": [[Expr]]} + meta (eig, blocks) : vitesses signees EXACTES
                                  # par valeurs propres du jacobien de flux (set_wave_speeds_from_jacobian)
        self._source = None     # liste d'Expr (une par composante) ou None
        self._elliptic = None   # Expr (contribution au second membre elliptique) ou None
        self._stab_speed = None  # Expr : vitesse de STABILITE lambda* (None = fallback eigenvalues)
        self._stab_dt = None     # Expr : pas ADMISSIBLE direct dt(U, aux) (None = pas de borne)
        self._src_freq = None    # Expr : frequence mu(U, aux) de la SOURCE (None = pas de borne)
        self._src_jac = None     # [[Expr]] n x n : Jacobien ANALYTIQUE dS/dU (None = diff. finies)
        self._hllc = False       # True : emettre la capability HLLC (contact_speed + star state)
        self._roe = False        # True : emettre la capability ROE (roe_dissipation depuis les roles)
        self._roe_rows = None    # {"x": [Expr], "y": [Expr]} : roe_dissipation FOURNIE (hors roles)
        self.prim_state = []    # noms ordonnes de l'etat primitif (layout de Prim) ; pour le codegen
        self.cons_from = None   # liste d'Expr : conservatif en fonction des primitives (to_conservative)
        self.cons_roles = None  # override explicite des roles conservatifs (sinon mapping canonique)
        self.prim_roles = None  # override explicite des roles primitifs (sinon mapping canonique)
        self.gamma = None       # indice adiabatique du bloc (EOS), lu par les couplages inter-especes
                                # cote System. None -> symbole adc_compiled_gamma non emis (le System
                                # retombe alors sur son defaut historique 1.4, retro-compat stricte).

    def cons(self, name):
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names, roles=None):
        """Declare les variables conservatives. @p roles (optionnel) : liste de meme longueur fixant
        explicitement le role physique de chaque composante (chaine 'Density'/'MomentumX'... ou None
        pour retomber sur le mapping canonique du nom) ; sert a un layout non standard ou les noms ne
        suffisent pas a deduire le sens. Sans roles, le mapping canonique nom -> role s'applique."""
        if roles is not None and len(roles) != len(names):
            raise ValueError("conservative_vars : %d roles pour %d variables" % (len(roles), len(names)))
        self.cons_roles = list(roles) if roles is not None else None
        return tuple(self.cons(n) for n in names)

    def primitive(self, name, expr):
        """Definit une primitive par sa formule (en fonction des cons / primitives precedentes)."""
        self.prim_defs[name] = _wrap(expr)
        return Var(name, "prim")

    def aux(self, name):
        """Champ auxiliaire CANONIQUE (p.ex. grad_x, grad_y, B_z, T_e) fourni a l'execution. Le nom
        DOIT etre une clef de AUX_CANONICAL. Pour un champ NOMME arbitraire, voir aux_field."""
        self.aux_names.append(name)
        return Var(name, "aux")

    def aux_field(self, name):
        """Champ auxiliaire NOMME (ADC-70 phase 1) fourni a l'execution par bloc via
        System.set_aux_field(bloc, name, array). Contrairement a aux(...) (composantes CANONIQUES
        phi/grad/B_z/T_e), name est ARBITRAIRE : le k-ieme appel reserve la composante
        AUX_NAMED_BASE + k du canal aux (lue en C++ via aux.extra_field(k)). Renvoie une Var
        utilisable dans flux / source / eigenvalues / elliptic_rhs comme toute autre variable aux.

        Au plus AUX_NAMED_MAX champs nommes par modele (borne FIXE cote C++, Aux POD). Un nom deja
        canonique (B_z, T_e, phi...) est REJETE : ces champs ont leurs chemins dedies (aux('B_z') +
        set_magnetic_field, etc.) ; un doublon de nom nomme est aussi rejete."""
        # Le nom devient une LOCALE C++ dans la formule generee (cf. _aux_locals_lines) ET la cle de la
        # table de la facade : il doit etre un identifiant C++ valide (lettres/chiffres/_, pas un
        # chiffre en tete). Rejet explicite plutot qu'un .so qui ne compile pas.
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("aux_field(%r) : nom invalide (identifiant C++ attendu : "
                             "lettres/chiffres/_, sans chiffre initial)" % (name,))
        if name in AUX_CANONICAL:
            raise ValueError(
                "aux_field('%s') : '%s' est un champ aux CANONIQUE ; utiliser aux('%s') (et le "
                "chemin dedie, p.ex. set_magnetic_field pour B_z, set_electron_temperature_from "
                "pour T_e)" % (name, name, name))
        if name in self.aux_extra_names:
            raise ValueError("aux_field('%s') : champ deja declare" % name)
        if len(self.aux_extra_names) >= AUX_NAMED_MAX:
            raise ValueError("aux_field('%s') : au plus %d champs aux nommes par modele "
                             "(borne kAuxMaxExtra cote C++)" % (name, AUX_NAMED_MAX))
        self.aux_extra_names.append(name)
        return Var(name, "aux")

    def _aux_locals_lines(self):
        """Locales C++ pour les champs aux lus dans une formule : canonique '<n>' <- a.<n> ;
        nomme '<n>' <- a.extra_field(k) (k = position dans aux_extra_names). Le nom de la locale est
        IDENTIQUE a celui que l'Expr emet (Var.to_cpp), donc la formule la reference directement."""
        lines = ["    const adc::Real %s = a.%s;" % (n, n) for n in self.aux_names]
        lines += ["    const adc::Real %s = a.extra_field(%d);" % (n, k)
                  for k, n in enumerate(self.aux_extra_names)]
        return lines

    def _reads_aux(self):
        """Vrai si une formule lit un champ aux (canonique ou nomme) : pilote le nommage du parametre
        Aux ('a' vs anonyme) pour ne pas declencher d'avertissement de parametre inutilise."""
        return bool(self.aux_names) or bool(self.aux_extra_names)

    def _total_n_aux(self):
        """Largeur TOTALE du canal aux du modele (canonique + champs nommes)."""
        return aux_total_n_aux(self.aux_names, self.aux_extra_names)

    def set_flux(self, x, y): self._flux = {"x": list(x), "y": list(y)}
    def set_eigenvalues(self, x, y): self._eig = {"x": list(x), "y": list(y)}

    def set_wave_speeds(self, x, y):
        """Vitesses d'onde SIGNEES explicites par direction : x = (smin_x, smax_x), y = (smin_y,
        smax_y), expressions des cons / prims / aux. Emet ``wave_speeds(U, aux, dir, smin, smax)``
        sur la brique generee SANS exiger de primitive 'p' : riemann='hll' devient disponible pour
        un modele sans pression (systeme de moments, isotherme...). Le coeur ne gate HLL que sur
        requires { m.wave_speeds(...) } (block_builder.hpp) : aucun changement C++.

        Prioritaire sur le chemin historique (primitive 'p' -> wave_speeds = min/max des
        eigenvalues) quand les deux existent. SANS appel : emission strictement historique.
        Si set_eigenvalues n'est PAS appele, max_wave_speed (Rusanov / CFL) est derive de
        max(|smin|, |smax|) sur les deux expressions de la direction."""
        x, y = tuple(x), tuple(y)
        if len(x) != 2 or len(y) != 2:
            raise ValueError("set_wave_speeds : attendu x=(smin, smax) et y=(smin, smax) "
                             "(recu x=%d expression(s), y=%d)" % (len(x), len(y)))
        if self._ws_jacobian is not None:
            raise ValueError("set_wave_speeds : set_wave_speeds_from_jacobian deja declare -- un "
                             "seul fournisseur de wave_speeds")
        self._wave_speeds = {"x": (_wrap(x[0]), _wrap(x[1])),
                             "y": (_wrap(y[0]), _wrap(y[1]))}

    def set_wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """Vitesses d'onde signees EXACTES : smin/smax = extremes des valeurs propres du jacobien
        de flux A = dF/dU, calcules NUMERIQUEMENT par cellule (adc::real_eig_minmax, QR de Francis
        sur tampon pile, repli Gershgorin sur non-convergence = borne externe sure). Emet
        ``wave_speeds(U, aux, dir, smin, smax)`` (gate HLL du coeur) et, sans set_eigenvalues,
        ``max_wave_speed`` = max(|smin|, |smax|) sur les memes blocs.

        @p x, @p y : matrices n_vars x n_vars d'expressions dA[i][j] = dF_dir[i]/dU[j]. None
        (defaut) = AUTODIFF du flux declare via flux_jacobian(dir) (dsl.diff, primitives
        developpees par la regle de chaine) -- le jacobien ne peut alors pas se desynchroniser du
        flux. Fournir x/y explicites n'a de sens que pour court-circuiter l'autodiff (formes
        simplifiees a la main) ; check_model les confronte alors aux differences finies du flux.

        @p eig : "numeric" (defaut) = entrees du jacobien emises en formules, valeurs propres par
        bloc a l'execution ; "fd" = jacobien construit PAR COLONNES aux differences finies du flux
        COMPILE ((flux(U + eps e_k) - flux(U))/eps, eps = 1e-6 |U[0]| + 1e-30, miroir de la
        branche flagsym != 1 du MATLAB de reference) -- bring-up/debug generique, jamais
        production (troncature O(eps)).

        @p blocks : None (defaut) = UN bloc plein n_vars x n_vars, seul mode inconditionnellement
        correct. Sinon, liste de LISTES D'INDICES (eventuellement non contigus, p.ex.
        [[0, 1, 4], [2, 3]]) appliquee aux DEUX directions, ou dict {"x": [...], "y": [...]}
        (les structures bloc-triangulaires de dFx/dU et dFy/dU different en general : pour un
        systeme de moments, les chaines en x sont contigues et celles en y ne le sont pas).
        Les extremes sont pris sur l'union des spectres des sous-blocs diagonaux A[idx][idx].
        CONTRAT : l'appelant AFFIRME que A est bloc-(inferieure-)triangulaire selon cette
        partition (a permutation pres) -- sur une matrice quelconque les extremes de sous-blocs
        NE BORNENT PAS le spectre (contre-exemple [[0, k], [k, 0]] : spectre +-k, sous-blocs 1x1
        nuls). Des indices peuvent etre omis (lignes/colonnes porteuses d'aucune valeur propre
        extreme, cf. le bloc saute du MATLAB de reference).

        Diagnostics : la non-convergence QR retombe en silence sur la borne de Gershgorin du bloc
        (PLUS LARGE, jamais fausse -- HLL reste stable, seulement plus diffusif) ; une perte
        d'hyperbolicite (valeurs propres complexes) n'est pas signalee par cellule -- la verifier
        hors ligne (check_model, golden type eigenvalues15_2D)."""
        if self._wave_speeds is not None:
            raise ValueError("set_wave_speeds_from_jacobian : set_wave_speeds deja declare -- un "
                             "seul fournisseur de wave_speeds")
        if eig not in ("numeric", "fd"):
            raise ValueError("set_wave_speeds_from_jacobian : eig 'numeric' | 'fd' (recu %r)" % (eig,))
        nv = self.n_vars
        if (x is None) != (y is None):
            raise ValueError("set_wave_speeds_from_jacobian : fournir x ET y, ou aucun (autodiff)")
        if eig == "fd" and x is not None:
            raise ValueError("set_wave_speeds_from_jacobian : eig='fd' construit le jacobien aux "
                             "differences finies du flux compile -- x/y n'ont pas de sens ici")
        rows = {}
        if eig == "numeric":
            if x is None:
                if not self._flux:
                    raise ValueError("set_wave_speeds_from_jacobian : appeler set_flux(...) d'abord "
                                     "(autodiff du jacobien)")
                rows = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}
            else:
                for key, mat in (("x", x), ("y", y)):
                    if len(mat) != nv or any(len(r) != nv for r in mat):
                        raise ValueError("set_wave_speeds_from_jacobian : jacobien %s attendu "
                                         "%d x %d" % (key, nv, nv))
                    rows[key] = [[_wrap(e) for e in r] for r in mat]
        def norm_blocks(blk, label):
            blk = [list(int(i) for i in b) for b in blk]
            seen = set()
            for b in blk:
                if not b:
                    raise ValueError("set_wave_speeds_from_jacobian : bloc vide (%s)" % label)
                local = set()
                for i in b:
                    if not (0 <= i < nv):
                        raise ValueError("set_wave_speeds_from_jacobian : indice %d hors [0, %d) "
                                         "(%s)" % (i, nv, label))
                    if i in local:
                        raise ValueError("set_wave_speeds_from_jacobian : indice %d present deux "
                                         "fois dans le meme bloc (%s)" % (i, label))
                    if i in seen:
                        raise ValueError("set_wave_speeds_from_jacobian : indice %d present dans "
                                         "deux blocs (%s)" % (i, label))
                    local.add(i)
                    seen.add(i)
            return blk

        if blocks is None:
            per_dir = {"x": [list(range(nv))], "y": [list(range(nv))]}
        elif isinstance(blocks, dict):
            if set(blocks) != {"x", "y"}:
                raise ValueError("set_wave_speeds_from_jacobian : blocks dict attendu avec les "
                                 "cles 'x' et 'y' (recu %r)" % sorted(blocks))
            per_dir = {k: norm_blocks(blocks[k], k) for k in ("x", "y")}
        else:
            shared = norm_blocks(blocks, "x et y")
            per_dir = {"x": shared, "y": [list(b) for b in shared]}
        self._ws_jacobian = {"rows": rows or None, "eig": eig, "blocks": per_dir,
                             "explicit": x is not None}
    def set_source(self, s): self._source = [_wrap(e) for e in s]
    def set_elliptic_rhs(self, e): self._elliptic = _wrap(e)

    def stability_speed(self, expr):
        """Vitesse de STABILITE lambda* (expression des cons / prims / aux) : pilote la CFL du bloc
        a la place de max(|eigenvalues|). Emise comme ``stability_speed(U, aux, dir)`` (trait C++
        ``HasStabilitySpeed``) : System::step_cfl l'utilise alors pour la borne de transport
        dt <= cfl*h/lambda*, tandis que les solveurs de Riemann continuent de lire max_wave_speed
        (stabilite != precision). SANS appel, le FALLBACK est strictement l'historique :
        max(abs(eigenvalues)) via max_wave_speed. Compilee comme flux/source (pas de callback Python
        par cellule : compatible production GPU/MPI). Cablee sur System ET AmrSystem (mono et
        multi-blocs ; cote AMR la reduction est evaluee sur le niveau GROSSIER, la ou vit la CFL)."""
        self._stab_speed = _wrap(expr)

    def stability_dt(self, expr_dt):
        """Pas ADMISSIBLE direct dt(U, aux) (expression > 0, en unites de temps) : borne locale du
        pas, emise comme ``stability_dt(U, aux)`` (trait C++ ``HasStabilityDt``). System::step_cfl
        impose dt <= min_cellules(stability_dt) * substeps / stride (le cfl n'est PAS applique : le
        modele declare deja un pas admissible). Forme la plus generale (source raide, couplage local,
        formule transport+source non reductible). SANS appel, aucune borne supplementaire (politique
        de pas historique). Compilee comme flux/source (production GPU/MPI). Cablee sur System ET
        AmrSystem (mono et multi-blocs ; cote AMR evaluee sur le niveau GROSSIER)."""
        self._stab_dt = _wrap(expr_dt)

    def source_frequency(self, expr_mu):
        """FREQUENCE locale mu(U, aux) [1/temps] de la SOURCE (relaxation, collision, reaction) :
        la 'deuxieme CFL' du meeting -- borne dt <= cfl * substeps / (stride * max_cellules(mu)),
        SANS pas d'espace (une source borne en 1/temps). Emise comme ``frequency(U, aux)`` sur la
        BRIQUE DE SOURCE generee (contrat C++ des briques source, cf. physics/source.hpp) ;
        CompositeModel la forwarde (trait HasSourceFrequency) et System/AmrSystem::step_cfl
        l'agregent. EXIGE set_source/m.source (la frequence est une propriete de la source).
        SANS appel, la source ne contraint pas le pas (historique). Compilee (production GPU/MPI)."""
        self._src_freq = _wrap(expr_mu)

    def source_jacobian(self, rows):
        """JACOBIEN ANALYTIQUE de la source : dS/dU, matrice n_vars x n_vars d'expressions
        (rows[r][c] = dS_r/dU_c, en fonction des cons / prims / aux). Emis comme
        ``jacobian(U, aux, J)`` sur la brique de SOURCE generee, forwarde par CompositeModel
        (trait C++ ``HasSourceJacobian``) : le Newton de la source implicite (IMEX /
        SourceImplicitBE) l'utilise A LA PLACE des differences finies -- exactitude (plus de bruit
        fd_eps) et evaluations de source economisees. EXIGE m.source. SANS appel : differences
        finies historiques, bit-identique."""
        self._src_jac = [[_wrap(e) for e in row] for row in rows]

    def enable_hllc(self):
        """Emet la CAPABILITY HLLC (audit vague 3) : ``contact_speed`` (Toro) + ``hllc_star_state``
        GENERES depuis les ROLES du bloc (Density / MomentumX / MomentumY, Energy optionnel) et la
        primitive 'p' -- le solveur HLLC contact-resolving du coeur (trait C++ HasHLLCStructure)
        devient alors disponible pour CE modele, MEME hors Euler 4 variables (isotherme 3-var,
        moments avec scalaires passifs : toute composante sans role particulier est advectee
        passivement dans l'etat etoile, Us[c] = fac*U[c]/rho). EXIGE : roles Density/MomentumX/
        MomentumY declares + primitive 'p' (erreur explicite a l'emission sinon)."""
        self._hllc = True
        return self

    def enable_roe(self):
        """Emet la CAPABILITY ROE (solde de l'audit, GENERICITY_2026-06.md point 11) :
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = |A_roe| (UR - UL) GENEREE depuis les ROLES du
        bloc -- le solveur Roe-like du coeur (trait C++ HasRoeDissipation, F = 1/2(FL+FR) - 1/2 d)
        devient disponible pour CE modele, MEME hors Euler 4 variables :

        - roles Density/MomentumX/MomentumY + Energy : algebre de Roe du gaz parfait, TRANSCRIPTION
          exacte du chemin canonique C++ (moyennes ponderees sqrt(rho), gamma-1 deduit de
          p/(E - 1/2 rho |v|^2), correction d'entropie de Harten sur les ondes acoustiques) ;
        - roles Density/MomentumX/MomentumY SANS Energy (isotherme / pseudo-pression) : meme
          decomposition sans la ligne d'energie, vitesse du son LOCALE c = sqrt(p/rho) moyennee
          a la Roe (generalisation standard hors gaz parfait) ;
        - toute composante HORS roles fluides est traitee en SCALAIRE PASSIF porte par l'onde
          entropique (ligne identique a la quantite de mouvement tangentielle, phi = q/rho).

        EXIGE : roles Density/MomentumX/MomentumY declares + primitive 'p' (erreur explicite a
        l'emission sinon). Sans appel : rien d'emis, riemann='roe' reste Euler-4-var-only.

        EXCLUSIF de m.roe_dissipation : la capability depuis les roles et la dissipation FOURNIE par
        l'utilisateur sont deux fournisseurs du MEME hook roe_dissipation -- les declarer toutes deux
        leve (un seul fournisseur)."""
        if self._roe_rows is not None:
            raise ValueError("enable_roe : roe_dissipation(...) deja fournie -- un seul fournisseur "
                             "du hook roe_dissipation (capability depuis les roles OU fournie)")
        self._roe = True

    def roe_dissipation(self, x, y):
        """Dissipation de Roe FOURNIE par l'utilisateur (hors familles a roles fluides) : n_vars
        expressions par direction (lignes d_i), emises comme le hook C++
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = d (trait HasRoeDissipation ; le coeur fait
        F = 1/2(FL+FR) - 1/2 d, cf. RoeFlux). C'est le pendant 'fourni' de m.enable_roe (genere
        depuis les ROLES) : ici l'utilisateur ecrit SON eigenstructure -- meme esprit que
        m.source_jacobian (fourni, pas invente). Le helper m.flux_jacobian(dir) (A = dF/dU
        auto-derive par dsl.diff) assiste cette ecriture.

        VOCABULAIRE DEUX ETATS : chaque variable/primitive doit etre encadree par dsl.left(...) (etat
        UL) ou dsl.right(...) (etat UR) ; une moyenne de Roe se note donc explicitement
        (left(sqrt(rho))*left(u) + right(sqrt(rho))*right(u)) / (left(sqrt(rho)) + right(sqrt(rho))).
        Une variable NUE (sans marqueur) leve a la declaration (etat indetermine).

        @p x, @p y : listes de n_vars expressions (lignes pour dir=0 et dir=1). DEUX jeux EXPLICITES
        (pas de mapping de roles ici) : a dir=0 la composante normale est l'axe x, a dir=1 l'axe y.

        Garde-fous : longueur n_vars par direction ; chaque variable sous left/right ; conflit avec
        enable_roe (un seul fournisseur du hook) -> erreur. SANS appel : rien d'emis (bit-identique).
        Requiert le backend 'aot' ou 'production' (le hook est emis dans la brique generee)."""
        if self._roe:
            raise ValueError("roe_dissipation : enable_roe() deja appele -- un seul fournisseur du "
                             "hook roe_dissipation (capability depuis les roles OU fournie)")
        rx, ry = list(x), list(y)
        if len(rx) != self.n_vars or len(ry) != self.n_vars:
            raise ValueError("roe_dissipation : %d expressions attendues par direction (recu x=%d, "
                             "y=%d)" % (self.n_vars, len(rx), len(ry)))
        rows = {"x": [_wrap(e) for e in rx], "y": [_wrap(e) for e in ry]}
        for key in ("x", "y"):
            for e in rows[key]:
                _roe_validate(e, False)  # rejette toute variable hors marqueur left()/right()
        self._roe_rows = rows

    def flux_jacobian(self, dir):
        """Jacobien de flux A = dF_dir/dU : matrice n_vars x n_vars d'expressions, A[i][j] =
        d(flux_dir[i])/d(cons[j]), auto-derivee des flux declares (via dsl.diff avec substitution des
        primitives). HELPER de CONSTRUCTION (l'utilisateur s'en sert pour ecrire m.roe_dissipation) :
        N'EMET RIEN par lui-meme. @p dir : 0/'x' (axe x) ou 1/'y' (axe y). EXIGE set_flux(...).

        Les primitives sont developpees par leur definition (chaine) ; une primitive non derivee
        reste un symbole dans le resultat (l'evaluer numeriquement exige un env contenant ses valeurs,
        p.ex. HyperbolicModel._env)."""
        if not self._flux:
            raise ValueError("flux_jacobian : appeler set_flux(...) d'abord")
        key = _dir_key(dir)
        comps = self._flux.get(key, [])
        if len(comps) != self.n_vars:
            raise ValueError("flux_jacobian : flux %s attendu avec %d composantes (recu %d)"
                             % (key, self.n_vars, len(comps)))
        defs = self.prim_defs
        return [[diff(comps[i], self.cons_names[j], defs) for j in range(self.n_vars)]
                for i in range(self.n_vars)]

    def left(self, expr):
        """Marque @p expr comme evaluee sur l'etat GAUCHE UL (sucre pour dsl.left, m.roe_dissipation)."""
        return left(expr)

    def right(self, expr):
        """Marque @p expr comme evaluee sur l'etat DROIT UR (sucre pour dsl.right, m.roe_dissipation)."""
        return right(expr)

    def set_gamma(self, gamma):
        """Indice adiabatique du bloc (EOS compressible). Transporte par le .so genere via le symbole
        optionnel adc_compiled_gamma, pour que les couplages inter-especes du System (collision,
        echange thermique, T_e) utilisent le BON gamma au lieu du defaut historique 1.4. Sans appel,
        aucun symbole gamma n'est emis (retro-compat : le System garde son defaut)."""
        self.gamma = float(gamma)

    def set_primitive_state(self, *vars_or_names, roles=None):
        """Declare le layout ORDONNE de l'etat primitif (Prim) : noms des composantes, dans l'ordre.
        Necessaire au codegen brique (to_primitive remplit Prim dans cet ordre). Chaque nom doit
        etre une variable conservative ou une primitive deja definie. @p roles (optionnel) : meme
        convention que conservative_vars (override explicite par composante, None = mapping canonique)."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]
        if roles is not None and len(roles) != len(self.prim_state):
            raise ValueError("set_primitive_state : %d roles pour %d variables"
                             % (len(roles), len(self.prim_state)))
        self.prim_roles = list(roles) if roles is not None else None

    def set_conservative_from(self, exprs):
        """Formules du conservatif en fonction des primitives (une par variable conservative, dans
        l'ordre de conservative_vars). Sert a generer to_conservative : le DSL ne sait pas inverser
        symboliquement les primitives, l'utilisateur fournit donc l'inverse explicitement."""
        self.cons_from = [_wrap(e) for e in exprs]

    @property
    def n_vars(self): return len(self.cons_names)

    # --- evaluation (interprete CPU, numpy) ---
    def _env(self, U, aux):
        """Environnement : cons (depuis U), aux (fournis), puis primitives derivees (ordre
        d'insertion = ordre de dependance)."""
        env = {self.cons_names[i]: U[i] for i in range(len(self.cons_names))}
        if aux:
            env.update(aux)
        for pname, pexpr in self.prim_defs.items():
            env[pname] = pexpr.eval(env)
        return env

    def flux(self, U, aux, dir):
        """Flux physique dans la direction dir (0=x, 1=y). U : numpy (n_vars, ...)."""
        env = self._env(U, aux)
        comps = self._flux["x" if dir == 0 else "y"]
        return np.stack([np.broadcast_to(c.eval(env), U[0].shape) for c in comps], axis=0)

    def max_wave_speed(self, U, aux, dir):
        """max_k max_cellules ``|lambda_k|`` : borne de Rusanov / CFL. Source : eigenvalues
        (historique) ; SANS set_eigenvalues, max(|smin|, |smax|) des vitesses signees explicites
        (set_wave_speeds), miroir exact de l'emission C++."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if not self._eig.get(key) and self._ws_jacobian is not None:
            lo, hi = self._ws_jacobian_value(U, env, key)
            return max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
        exprs = self._eig.get(key) or (list(self._wave_speeds[key])
                                       if self._wave_speeds is not None else None)
        if not exprs:
            raise ValueError("max_wave_speed : ni set_eigenvalues(...) ni set_wave_speeds(...) ni "
                             "set_wave_speeds_from_jacobian(...) declares sur le modele '%s'"
                             % self.name)
        return max(float(np.max(np.abs(np.asarray(e.eval(env))))) for e in exprs)

    def _ws_jacobian_value(self, U, env, key):
        """Evaluateur numpy du chemin jacobien : extremes des parties reelles des valeurs propres
        des sous-blocs, par echantillon (miroir du wave_speeds emis ; np.linalg.eigvals)."""
        ws = self._ws_jacobian
        nv = self.n_vars
        nsmp = int(np.asarray(U[0]).reshape(-1).shape[0])
        if ws["eig"] == "fd":
            base = np.stack([np.broadcast_to(np.asarray(c.eval(env), dtype=float), (nsmp,))
                             if hasattr(c, "eval") else np.full((nsmp,), float(c))
                             for c in (self._flux[key])], axis=0)
            J = np.empty((nsmp, nv, nv))
            Uflat = np.stack([np.broadcast_to(np.asarray(env[c], dtype=float), (nsmp,))
                              for c in self.cons_names], axis=0)
            for k in range(nv):
                eps = 1e-6 * np.abs(Uflat[0]) + 1e-30
                Up = Uflat.copy()
                Up[k] += eps
                envp = self._env(Up, {n: env[n] for n in self.aux_names} if self.aux_names else None)
                Fp = np.stack([np.broadcast_to(np.asarray(c.eval(envp), dtype=float), (nsmp,))
                               for c in self._flux[key]], axis=0)
                J[:, :, k] = ((Fp - base) / eps).T
        else:
            rows = ws["rows"][key]
            J = np.empty((nsmp, nv, nv))
            for i in range(nv):
                for j in range(nv):
                    J[:, i, j] = np.broadcast_to(
                        np.asarray(rows[i][j].eval(env), dtype=float), (nsmp,))
        lo = np.full((nsmp,), np.inf)
        hi = np.full((nsmp,), -np.inf)
        for b in ws["blocks"][key]:
            idx = np.asarray(b)
            lam = np.linalg.eigvals(J[:, idx[:, None], idx[None, :]])
            lo = np.minimum(lo, lam.real.min(axis=1))
            hi = np.maximum(hi, lam.real.max(axis=1))
        return lo, hi

    def _flux_jacobian_spectral_radius(self, U, aux, dir):
        """Rayon spectral max_cellules max_k |Re(lambda_k)| du jacobien dense COMPLET A = dF_dir/dU,
        evalue aux differences finies CENTRALES sur le flux interprete. Independant de toute
        partition declaree (set_wave_speeds_from_jacobian blocks=...) : sert de borne de reference
        non circulaire face a max_wave_speed. Rend None si un etat perturbe sort du domaine (flux
        non fini) -- on ne peut alors pas conclure."""
        nv = self.n_vars
        U = np.asarray(U, dtype=float)
        nsmp = U.shape[1]
        J = np.empty((nsmp, nv, nv))
        for j in range(nv):
            eps = 1e-6 * np.abs(U[j]) + 1e-7
            Up = U.copy(); Up[j] = Up[j] + eps
            Um = U.copy(); Um[j] = Um[j] - eps
            Fp = self.flux(Up, aux, dir)
            Fm = self.flux(Um, aux, dir)
            if not (bool(np.all(np.isfinite(Fp))) and bool(np.all(np.isfinite(Fm)))):
                return None
            for i in range(nv):
                J[:, i, j] = (np.broadcast_to(Fp[i], (nsmp,))
                              - np.broadcast_to(Fm[i], (nsmp,))) / (2.0 * eps)
        lam = np.linalg.eigvals(J)
        return float(np.max(np.abs(lam.real)))

    def wave_speeds_value(self, U, aux, dir):
        """Evaluateur numpy des vitesses signees (smin, smax) -- miroir du wave_speeds emis :
        paire explicite (set_wave_speeds) si declaree, sinon min/max des eigenvalues (chemin
        historique, qui exige 'p' pour etre EMIS mais reste evaluable ici)."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if self._wave_speeds is not None:
            lo, hi = self._wave_speeds[key]
            return (np.asarray(lo.eval(env), dtype=float),
                    np.asarray(hi.eval(env), dtype=float))
        if self._ws_jacobian is not None:
            return self._ws_jacobian_value(U, env, key)
        eigs = [np.asarray(e.eval(env), dtype=float) for e in self._eig.get(key, [])]
        if not eigs:
            raise ValueError("wave_speeds_value : ni set_wave_speeds(...) ni set_eigenvalues(...) "
                             "declares sur le modele '%s'" % self.name)
        eigs = list(np.broadcast_arrays(*eigs)) if len(eigs) > 1 else eigs  # formes mixtes (lambda constante)
        return (np.min(np.stack(eigs), axis=0), np.max(np.stack(eigs), axis=0))

    def source_value(self, U, aux):
        """Terme source (numpy (n_vars, ...)), ou zeros si non defini."""
        if self._source is None:
            return np.zeros_like(U)
        env = self._env(U, aux)
        return np.stack([np.broadcast_to(s.eval(env), U[0].shape) for s in self._source], axis=0)

    def to_python_flux(self, aux=None):
        """Produit un adc.PythonFlux (backend hote) a partir des formules : le modele TOURNE
        (interprete CPU). aux : dict nom -> tableau (champs auxiliaires), fige pour ce flux."""
        import adc
        a = aux or {}
        return adc.PythonFlux(
            lambda U, d: self.flux(U, a, d),
            lambda U: max(self.max_wave_speed(U, a, 0), self.max_wave_speed(U, a, 1)))

    def check(self):
        """Verifie que toute variable referencee (primitives, flux, valeurs propres, source) est
        bien declaree (cons / prim / aux). Leve ValueError sinon (verification de dependances)."""
        known = (set(self.cons_names) | set(self.prim_defs) | set(self.aux_names)
                 | set(self.aux_extra_names))  # champs aux nommes (aux_field) : ADC-70
        used = set()
        groups = [self._flux.get("x", []), self._flux.get("y", []),
                  self._eig.get("x", []), self._eig.get("y", []), self._source or [],
                  [e for e in (self._stab_speed, self._stab_dt, self._src_freq)
                   if e is not None],
                  [e for row in (self._src_jac or []) for e in row]]
        if self._wave_speeds is not None:
            groups.append(list(self._wave_speeds["x"]) + list(self._wave_speeds["y"]))
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):
                groups.append([e for row in self._ws_jacobian["rows"][d] for e in row])
        if self._roe_rows is not None:
            groups.append(self._roe_rows["x"])
            groups.append(self._roe_rows["y"])
        for e in self.prim_defs.values():
            used |= e.deps()
        for grp in groups:
            for e in grp:
                used |= e.deps()
        if self._elliptic is not None:
            used |= self._elliptic.deps()
        missing = used - known
        if missing:
            raise ValueError("modele '%s' : variables non definies %s" % (self.name, sorted(missing)))
        # source_frequency est une propriete de la SOURCE (emise sur la brique source generee) :
        # la declarer sans source serait perdue EN SILENCE -> erreur explicite.
        if self._src_freq is not None and self._source is None:
            raise ValueError("modele '%s' : source_frequency(...) declare sans source "
                             "(appeler m.source([...]) -- la frequence est emise sur la brique de "
                             "source generee)" % self.name)
        if self._src_jac is not None and self._source is None:
            raise ValueError("modele '%s' : source_jacobian(...) declare sans source "
                             "(appeler m.source([...]) -- le Jacobien est emis sur la brique de "
                             "source generee)" % self.name)
        # roe_dissipation et enable_roe sont deux fournisseurs du MEME hook : exclusifs (defensif ;
        # deja rejete a la declaration). Re-controle structurel des lignes (left/right) au passage.
        if self._roe_rows is not None:
            if self._roe:
                raise ValueError("modele '%s' : enable_roe() et roe_dissipation(...) declares "
                                 "ensemble -- un seul fournisseur du hook roe_dissipation" % self.name)
            for key in ("x", "y"):
                for e in self._roe_rows[key]:
                    _roe_validate(e, False)
        return True

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Verification NUMERIQUE generique du modele symbolique (audit 2026-06, chantier 6) :
        evalue les formules sur des etats echantillons et controle, quand la piece existe :

        - flux fini (les deux directions) ;
        - source finie ;
        - elliptic_rhs fini ;
        - valeurs propres finies et reelles ; max_wave_speed fini et >= 0 ;
        - coherence wave_speeds <-> max_wave_speed : max(|lambda_min|, |lambda_max|) <= mws ;
        - encadrement NON CIRCULAIRE du spectre : le rayon spectral du jacobien dense complet du
          flux (differences finies centrales, independant de toute partition blocks=) ne depasse
          pas max_wave_speed -- attrape une partition set_wave_speeds_from_jacobian qui ne borne
          pas les valeurs propres (mws sous-estime, CFL non sure) la ou la coherence ci-dessus,
          derivee de la MEME partition, reste vraie ;
        - round-trip to_conservative(to_primitive(U)) ~= U (si prim_state + cons_from declares) ;
        - positivite des composantes a role Density (et de la primitive 'p' si declaree) sur les
          echantillons (qui sont generes positifs pour ces roles).

        @p jac_rtol, @p jac_atol : tolerances de l'encadrement spectral (radius <= mws*(1+jac_rtol)
        + jac_atol) ; relachees pour absorber le bruit des differences finies du jacobien.

        @p samples : tableau (n_vars, N) d'etats conservatifs a tester ; None -> N = n_samples etats
        aleatoires (seed fixe, reproductible) : composantes a role Density dans [0.1, 2], les autres
        dans [-1, 1] ; une composante a role Energy recoit 1 + |cinetique| pour rester physique.
        @p aux : dict nom -> valeur(s) des champs auxiliaires (defaut : zeros).
        @return dict {"ok": bool, "failures": [str], "n_samples": N}. raise_on_error=True (defaut)
        leve ValueError listant les echecs. PRE-COMPILATION : verifie les FORMULES (le .so compile
        emet exactement ces formules) ; le pendant RUNTIME sur un bloc installe est
        System.check_model(block)."""
        self.check()  # dependances declarees (leve si une variable n'existe pas)
        rng = np.random.default_rng(seed)
        nv = self.n_vars
        roles = roles_for(self.cons_names, self.cons_roles)
        if samples is None:
            U = rng.uniform(-1.0, 1.0, size=(nv, int(n_samples)))
            kinetic = np.zeros(int(n_samples))
            for i, r in enumerate(roles):
                if r == "Density":
                    U[i] = rng.uniform(0.1, 2.0, size=int(n_samples))
            for i, r in enumerate(roles):
                if r in ("MomentumX", "MomentumY"):
                    kinetic += U[i] ** 2
            for i, r in enumerate(roles):
                if r == "Energy":
                    U[i] = 1.0 + kinetic  # au-dessus du cinetique : pression > 0 pour un gaz parfait
        else:
            U = np.asarray(samples, dtype=float)
            if U.ndim != 2 or U.shape[0] != nv:
                raise ValueError("check_model : samples doit etre (n_vars=%d, N)" % nv)
        a = {n: np.zeros(U.shape[1]) for n in (self.aux_names + self.aux_extra_names)}
        if aux:
            for k, v in aux.items():
                a[k] = np.broadcast_to(np.asarray(v, dtype=float), (U.shape[1],)).copy()
        failures = []

        def finite(x):
            return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))

        for d, dn in ((0, "x"), (1, "y")):
            if not finite(self.flux(U, a, d)):
                failures.append("flux %s non fini sur les echantillons" % dn)
        if self._source is not None and not finite(self.source_value(U, a)):
            failures.append("source non finie sur les echantillons")
        if self._elliptic is not None:
            env = self._env(U, a)
            if not finite(self._elliptic.eval(env)):
                failures.append("elliptic_rhs non fini sur les echantillons")
        env = self._env(U, a)
        for d in ("x", "y"):
            for k, e in enumerate(self._eig.get(d, [])):
                lam = np.asarray(e.eval(env), dtype=float)
                if np.iscomplexobj(lam):
                    failures.append("valeur propre %s[%d] complexe (systeme non hyperbolique ?)" % (d, k))
                elif not finite(lam):
                    failures.append("valeur propre %s[%d] non finie" % (d, k))
        if self._wave_speeds is not None:
            for d in ("x", "y"):
                lo = np.asarray(self._wave_speeds[d][0].eval(env), dtype=float)
                hi = np.asarray(self._wave_speeds[d][1].eval(env), dtype=float)
                if not (finite(lo) and finite(hi)):
                    failures.append("wave_speeds %s (explicites) non finies" % d)
                elif bool(np.any(lo > hi)):
                    failures.append("wave_speeds %s (explicites) : smin > smax sur des echantillons" % d)
        for d, dn in ((0, "x"), (1, "y")):
            mws = self.max_wave_speed(U, a, d)
            if not np.isfinite(mws) or mws < 0:
                failures.append("max_wave_speed %s non fini ou negatif (%r)" % (dn, mws))
            else:
                # coherence wave_speeds <-> max_wave_speed : les extremes SIGNES effectivement emis
                # (paire explicite si declaree, sinon eigenvalues) doivent etre couverts par la
                # borne de Rusanov / CFL.
                lo, hi = self.wave_speeds_value(U, a, d)
                ext = max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
                if ext > mws * (1.0 + rtol) + atol:
                    failures.append("wave_speeds %s incoherentes avec max_wave_speed (%g > %g)"
                                    % (dn, ext, mws))
                # encadrement NON CIRCULAIRE : le rayon spectral du jacobien dense du flux (FD
                # centrales, sans partition) doit etre majore par max_wave_speed. Une partition
                # blocks= qui n'est pas reellement bloc-triangulaire rend des extremes de
                # sous-blocs qui NE bornent PAS le spectre -> mws trop petit, detecte ici.
                if self._flux:
                    radius = self._flux_jacobian_spectral_radius(U, a, d)
                    if radius is not None and radius > mws * (1.0 + jac_rtol) + jac_atol:
                        failures.append(
                            "partition %s : max_wave_speed (%g) n'encadre pas le spectre du "
                            "jacobien de flux (rayon spectral %g) -- la partition blocks= de "
                            "set_wave_speeds_from_jacobian ne borne pas les valeurs propres, la "
                            "borne de Rusanov/CFL est sous-estimee" % (dn, mws, radius))
        # round-trip cons -> prim -> cons (quand les deux sens sont declares)
        if self.prim_state and self.cons_from is not None:
            penv = {nm: np.broadcast_to(np.asarray(env[nm], dtype=float), (U.shape[1],))
                    for nm in self.prim_state}
            U2 = np.stack([np.broadcast_to(np.asarray(e.eval(penv), dtype=float), (U.shape[1],))
                           for e in self.cons_from], axis=0)
            if not finite(U2):
                failures.append("to_conservative(to_primitive(U)) non fini")
            elif not np.allclose(U2, U, rtol=rtol, atol=atol):
                err = float(np.max(np.abs(U2 - U)))
                failures.append("round-trip to_conservative(to_primitive(U)) != U (ecart max %g : "
                                "conversions incoherentes)" % err)
        # positivite : roles Density (conservatif) et primitive 'p' (pression) si declares
        for i, r in enumerate(roles):
            if r == "Density" and not bool(np.all(U[i] > 0)):
                failures.append("composante '%s' (role Density) non strictement positive sur les "
                                "echantillons" % self.cons_names[i])
        if "p" in self.prim_defs:
            p = np.asarray(env["p"], dtype=float)
            if not finite(p):
                failures.append("primitive 'p' (pression) non finie")
            elif not bool(np.all(p > 0)):
                failures.append("primitive 'p' (pression) non strictement positive sur des etats "
                                "physiques (EOS suspecte)")
        report = {"ok": not failures, "failures": failures, "n_samples": int(U.shape[1])}
        if failures and raise_on_error:
            raise ValueError("check_model('%s') : %d echec(s) :\n  - %s"
                             % (self.name, len(failures), "\n  - ".join(failures)))
        return report

    # --- parametres RUNTIME (P7-b) : collecte + attribution d'indices + membre genere -------------
    def _all_exprs(self):
        """Toutes les Expr du modele (primitives, flux, valeurs propres, source, elliptique,
        cons_from). Sert a decouvrir les noeuds RuntimeParamRef caches dans l'arbre."""
        out = list(self.prim_defs.values())
        for d in ("x", "y"):
            out += self._flux.get(d, [])
            out += self._eig.get(d, [])
        if self._wave_speeds is not None:  # vitesses signees explicites : params runtime inclus
            for d in ("x", "y"):
                out += list(self._wave_speeds[d])
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):  # entrees du jacobien : params runtime inclus
                out += [e for row in self._ws_jacobian["rows"][d] for e in row]
        if self._source is not None:
            out += [_wrap(e) for e in self._source]
        if self.cons_from is not None:
            out += list(self.cons_from)
        if self._elliptic is not None:
            out.append(self._elliptic)
        if self._roe_rows is not None:  # lignes Roe fournies : decouvrir leurs params runtime (via StateRef)
            out += self._roe_rows["x"] + self._roe_rows["y"]
        return out

    def runtime_param_nodes(self):
        """Noeuds RuntimeParamRef PRESENTS dans les formules, dedupliques par nom (un meme param peut
        apparaitre plusieurs fois mais partage le MEME objet noeud). Ordre TRIE par nom (indice stable
        = position dans cette liste, miroir de RuntimeParams cote C++)."""
        seen = {}

        def walk(e):
            if isinstance(e, RuntimeParamRef):
                seen.setdefault(e.name, e)
                return
            for c in _children(e):
                walk(c)

        for e in self._all_exprs():
            walk(e)
        return [seen[k] for k in sorted(seen)]

    def assign_runtime_indices(self):
        """Attribue a chaque RuntimeParamRef son indice STABLE (ordre trie des noms) et renvoie la liste
        ordonnee des noeuds. APPELE avant tout codegen de brique : sans cet appel, to_cpp() leverait
        (indice -1). Idempotent (reaffecte les memes indices). Rejette un modele depassant la borne C++
        kMaxRuntimeParams (sinon le tableau de taille fixe deborderait)."""
        nodes = self.runtime_param_nodes()
        if len(nodes) > _K_MAX_RUNTIME_PARAMS:
            raise ValueError(
                "modele '%s' : %d parametres runtime > borne kMaxRuntimeParams=%d "
                "(include/adc/runtime/runtime_params.hpp) ; reduire le nombre de params runtime"
                % (self.name, len(nodes), _K_MAX_RUNTIME_PARAMS))
        for k, node in enumerate(nodes):
            node.index = k
        return nodes

    def _runtime_params_member(self):
        """Ligne C++ declarant le membre RuntimeParams d'une brique generee, initialise aux valeurs de
        DECLARATION (defaut sans appel set au runtime). Chaine vide si le modele n'a aucun param runtime
        (brique strictement identique a l'historique -> bit-identite des params const preservee)."""
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        vals = ", ".join(repr(node.value) for node in nodes)
        return ("  adc::RuntimeParams params{%d, {%s}};  // params RUNTIME (P7-b) : ecrasables a "
                "l'execution\n" % (len(nodes), vals))

    def has_runtime_params(self):
        """Vrai si au moins une formule lit un parametre runtime (kind='runtime')."""
        return bool(self.runtime_param_nodes())

    # --- codegen (etape 2 : arbre symbolique -> C++ compilable) ---
    def _codegen_exprs(self, exprs, cse, real="adc::Real", indent="    "):
        """(lignes de locales CSE, [C++ par expr]). Si cse, factorise les sous-expressions communes
        (H, c...) en locales ``cseK_`` ; sinon inline chaque expression via to_cpp."""
        if cse:
            return _cse_emit(list(exprs), real, indent)
        return [], [e.to_cpp() for e in exprs]

    def emit_cpp(self, func=None, cse=True):
        """Genere une fonction C++ compilable calculant le flux physique a partir de l'arbre
        symbolique (chaque noeud Expr sait s'ecrire en C++ via to_cpp).

        Signature produite : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constantes inlinees ; chaque primitive devient une variable locale. cse=True (defaut) factorise
        les sous-expressions communes (H, c...) en locales ``cseK_`` ; cse=False les recalcule inline.

        Etape (2) du DSL (cf. docs/ARCHITECTURE_CIBLE.md sect. 3) : C++ HOTE (templatable sur Real)."""
        name = func or self.name
        if not self._flux:
            raise ValueError("emit_cpp : appeler set_flux(...) d'abord")
        if len(self._flux.get("x", [])) != self.n_vars or len(self._flux.get("y", [])) != self.n_vars:
            raise ValueError("emit_cpp : flux attendu avec %d composantes par direction" % self.n_vars)
        nc = self.n_vars
        out = [
            "// genere depuis le modele symbolique '%s' (adc.dsl.emit_cpp)" % self.name,
            "// flux physique F = flux(U, dir) ; dir 0=x, 1=y ; U et F de taille %d." % nc,
            "#include <cmath>",
            "template <class Real>",
            "inline void %s_flux(const Real* U, Real* F, int dir) {" % name,
        ]
        out += ["  const Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]
        out += ["  const Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]
        tl, cpps = self._codegen_exprs(self._flux["x"] + self._flux["y"], cse, real="Real", indent="  ")
        out += tl
        out.append("  if (dir == 0) {")
        out += ["    F[%d] = %s;" % (i, cpps[i]) for i in range(nc)]
        out.append("  } else {")
        out += ["    F[%d] = %s;" % (i, cpps[nc + i]) for i in range(nc)]
        out += ["  }", "}"]
        return "\n".join(out) + "\n"

    def emit_cpp_brick(self, name=None, namespace="adc_generated", cse=True):
        """Genere une BRIQUE C++ satisfaisant le concept adc::HyperbolicModel (emballage : etape
        2bis). Le struct produit utilise StateVec / Aux / ADC_HD / Variables et expose flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : il peut
        donc entrer dans un CompositeModel et tourner dans le solveur compile.

        Exige set_primitive_state(...) (layout de Prim) et set_conservative_from([...]) (to_conservative,
        que le DSL ne sait pas inverser tout seul). cse=True (defaut) factorise les sous-expressions
        communes (H, c...) en locales ``cseK_``. Reste a faire (cf. ARCHITECTURE_CIBLE.md sect. 3) :
        codegen Kokkos/CUDA, JIT."""
        if not self.prim_state:
            raise ValueError("emit_cpp_brick : appeler set_primitive_state(...) d'abord")
        if self.cons_from is None or len(self.cons_from) != self.n_vars:
            raise ValueError("emit_cpp_brick : set_conservative_from([...]) attendu (%d expressions)"
                             % self.n_vars)
        if not self._flux:
            raise ValueError("emit_cpp_brick : appeler set_flux(...) d'abord")
        if len(self._flux.get("x", [])) != self.n_vars or len(self._flux.get("y", [])) != self.n_vars:
            raise ValueError("emit_cpp_brick : flux attendu avec %d composantes par direction"
                             % self.n_vars)
        if not self._eig and self._wave_speeds is None and self._ws_jacobian is None:
            raise ValueError("emit_cpp_brick : appeler set_eigenvalues(...), set_wave_speeds(...) "
                             "ou set_wave_speeds_from_jacobian(...) d'abord (source de "
                             "max_wave_speed / CFL)")
        nm = name or (self.name.capitalize() + "Gen")
        nc, npr = self.n_vars, len(self.prim_state)

        def cons_locals():
            return ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]

        def prim_locals():
            return ["    const adc::Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]

        def aux_locals():
            return self._aux_locals_lines()  # canonique (a.<n>) + nommes (a.extra_field(k)), ADC-70

        # parametre Aux nomme 'a' uniquement si une formule lit un champ auxiliaire (canonique OU
        # nomme ; sinon anonyme, pour ne pas declencher d'avertissement de parametre inutilise).
        aux_param = "const Aux& a" if self._reads_aux() else "const Aux&"

        def eig_reduce(cpps, ind):
            # cpps : C++ deja genere (eventuellement CSE) des valeurs propres. Noms internes a suffixe
            # '_' : ne masquent ni une variable utilisateur ni le parametre Aux 'a' (cf. revue adverse).
            lines = ["%sconst adc::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
            lines.append("%sadc::Real mws_ = lam0_ < 0 ? -lam0_ : lam0_;" % ind)
            for k in range(1, len(cpps)):
                lines.append("%s{ const adc::Real cand_ = lam%d_ < 0 ? -lam%d_ : lam%d_;"
                             " if (cand_ > mws_) mws_ = cand_; }" % (ind, k, k, k))
            lines.append("%sreturn mws_;" % ind)
            return lines

        def eig_minmax(cpps, ind):
            # vitesses d'onde signees : smin = plus petite, smax = plus grande valeur propre (pour
            # HLLC / Roe). Memes noms internes a suffixe '_' que eig_reduce.
            lines = ["%sconst adc::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
            lines.append("%ssmin = lam0_; smax = lam0_;" % ind)
            for k in range(1, len(cpps)):
                lines.append("%sif (lam%d_ < smin) smin = lam%d_;" % (ind, k, k))
                lines.append("%sif (lam%d_ > smax) smax = lam%d_;" % (ind, k, k))
            return lines

        def ws_jac_pieces(key):
            # chemin jacobien 'numeric' : CSE des entrees NON NULLES des sous-blocs de la
            # direction @p key ; les zeros structurels (10 lignes unite d'un systeme de moments,
            # creux quelconque) sont emis en litteraux sans passer par le CSE.
            ws = self._ws_jacobian
            rows = ws["rows"][key]
            entries, zeros = [], []
            for bi, b in enumerate(ws["blocks"][key]):
                for r, gi in enumerate(b):
                    for c, gj in enumerate(b):
                        e = rows[gi][gj]
                        if isinstance(e, Const) and e.value == 0.0:
                            zeros.append((bi, r, c))
                        else:
                            entries.append((bi, r, c, e))
            tl, cpps = self._codegen_exprs([e for (_, _, _, e) in entries], cse)
            fill = {}
            for (bi, r, c, _), cpp in zip(entries, cpps):
                fill.setdefault(bi, []).append((r, c, cpp))
            for (bi, r, c) in zeros:
                fill.setdefault(bi, []).append((r, c, "adc::Real(0)"))
            return tl, fill

        def ws_jac_body(ind, lo, hi, key="x", fill=None):
            # corps du calcul jacobien -> extremes (@p lo/@p hi : noms des destinations).
            # eig='fd' : jacobien par colonnes aux differences finies du flux COMPILE ;
            # eig='numeric' : remplissage des sous-blocs depuis @p fill. @p key : direction
            # (choisit la partition de blocs).
            ws = self._ws_jacobian
            nv = self.n_vars
            L = []
            if ws["eig"] == "fd":
                L.append("%sconst State F0_ = flux(U, a, dir);" % ind)
                L.append("%sadc::Real Jf_[%d][%d];" % (ind, nv, nv))
                L.append("%sconst adc::Real eps_ = adc::Real(1e-6) * (U[0] < 0 ? -U[0] : U[0])"
                         " + adc::Real(1e-30);" % ind)
                L.append("%sfor (int k_ = 0; k_ < %d; ++k_) {" % (ind, nv))
                L.append("%s  State Up_ = U;" % ind)
                L.append("%s  Up_[k_] += eps_;" % ind)
                L.append("%s  const State Fk_ = flux(Up_, a, dir);" % ind)
                L.append("%s  for (int i_ = 0; i_ < %d; ++i_) Jf_[i_][k_] = (Fk_[i_] - F0_[i_])"
                         " / eps_;" % (ind, nv))
                L.append("%s}" % ind)
            for bi, b in enumerate(ws["blocks"][key]):
                nb = len(b)
                L.append("%s{" % ind)
                L.append("%s  adc::Real Jb_[%d][%d];" % (ind, nb, nb))
                if ws["eig"] == "fd":
                    for r, gi in enumerate(b):
                        for c, gj in enumerate(b):
                            L.append("%s  Jb_[%d][%d] = Jf_[%d][%d];" % (ind, r, c, gi, gj))
                else:
                    for (r, c, cpp) in sorted(fill.get(bi, [])):
                        L.append("%s  Jb_[%d][%d] = %s;" % (ind, r, c, cpp))
                L.append("%s  const adc::EigBounds eb_ = adc::real_eig_minmax(Jb_);" % ind)
                if bi == 0:
                    L.append("%s  %s = eb_.lmin; %s = eb_.lmax;" % (ind, lo, hi))
                else:
                    L.append("%s  if (eb_.lmin < %s) %s = eb_.lmin;" % (ind, lo, lo))
                    L.append("%s  if (eb_.lmax > %s) %s = eb_.lmax;" % (ind, hi, hi))
                L.append("%s}" % ind)
            return L

        cnames = ", ".join('"%s"' % c for c in self.cons_names)
        pnames = ", ".join('"%s"' % p for p in self.prim_state)
        # Roles physiques paralleles aux noms : initialiseur C++ d'adc::VariableSet::roles. Emis SI au
        # moins une composante a un role reconnu (sinon roles vides -> brique identique a l'historique,
        # les couplages retombent sur les indices de fallback). Les roles permettent a System de
        # resoudre les couplages inter-especes par index_of(role) au lieu d'un indice litteral.
        def roles_init(roles):
            if all(r == "Custom" for r in roles):
                return None  # aucun role utile : on n'emet pas le 4e champ (retro-compat stricte)
            return ", ".join("adc::VariableRole::%s" % r for r in roles)

        croles = roles_init(roles_for(self.cons_names, self.cons_roles))
        proles = roles_init(roles_for(self.prim_state, self.prim_roles))
        # P7-b : attribuer les indices runtime AVANT tout to_cpp() (un RuntimeParamRef leve sinon).
        rt_member = self._runtime_params_member()
        S = [
            "#include <cmath>",  # std::sqrt / std::pow : brique autosuffisante (g++ ne tire pas cmath)
            "// brique HYPERBOLIQUE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_brick)."
            % self.name,
            "// Satisfait adc::HyperbolicModel : flux + max_wave_speed + conversions + descripteurs.",
        ]
        if rt_member:  # en-tete RuntimeParams uniquement si une formule lit un param runtime
            S.append("#include <adc/runtime/runtime_params.hpp>")
        if self._ws_jacobian is not None:  # valeurs propres de blocs denses (wave_speeds exacts)
            S.append("#include <adc/numerics/dense_eig.hpp>")
        S += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
            "  using State = adc::StateVec<%d>;" % nc,
            "  using Prim  = adc::StateVec<%d>;" % npr,
            "  using Aux   = adc::Aux;",
            "  static constexpr int n_vars = %d;" % nc,
        ]
        if rt_member:  # membre adc::RuntimeParams params{count, {defauts}} (P7-b)
            S.append(rt_member.rstrip("\n"))
        # n_aux si une formule (flux / valeurs propres) lit un champ aux supplementaire : canonique
        # (B_z...) OU nomme (aux_field -> kAuxNamedBase + k). Sans champ extra -> pas de n_aux emis,
        # brique strictement bit-identique a l'historique.
        if self._total_n_aux() > AUX_BASE_COMPS:
            S.append("  static constexpr int n_aux = %d;" % self._total_n_aux())
        S += [
            "",
            "  ADC_HD State flux(const State& U, %s, int dir) const {" % aux_param,
        ]
        S += cons_locals() + prim_locals() + aux_locals()
        ftl, fcpps = self._codegen_exprs(self._flux["x"] + self._flux["y"], cse)
        S += ftl
        S.append("    State F{};")
        S.append("    if (dir == 0) {")
        S += ["      F[%d] = %s;" % (i, fcpps[i]) for i in range(nc)]
        S.append("    } else {")
        S += ["      F[%d] = %s;" % (i, fcpps[nc + i]) for i in range(nc)]
        S += ["    }", "    return F;", "  }", ""]

        # en mode jacobien 'fd' SANS eigenvalues, max_wave_speed appelle flux(U, a, dir) : le
        # parametre Aux doit etre nomme meme si aucune formule ne lit d'aux.
        jac_fd = self._ws_jacobian is not None and self._ws_jacobian["eig"] == "fd"
        mws_aux_param = "const Aux& a" if (jac_fd and not self._eig) else aux_param
        S.append("  ADC_HD adc::Real max_wave_speed(const State& U, %s, int dir) const {"
                 % mws_aux_param)
        S += cons_locals() + prim_locals() + aux_locals()
        if self._eig:
            # source historique : max(|eigenvalues|), bit-identique.
            nx = len(self._eig["x"])
            etl, ecpps = self._codegen_exprs(self._eig["x"] + self._eig["y"], cse)
            S += etl
            S.append("    if (dir == 0) {")
            S += eig_reduce(ecpps[:nx], "      ")
            S.append("    } else {")
            S += eig_reduce(ecpps[nx:], "      ")
            S += ["    }", "  }", ""]
        elif self._wave_speeds is not None:
            # SANS eigenvalues : borne de Rusanov / CFL derivee des vitesses SIGNEES explicites,
            # max(|smin|, |smax|) -- la paire borne le spectre par contrat de set_wave_speeds.
            ws = self._wave_speeds
            wtl, wcpps = self._codegen_exprs(list(ws["x"]) + list(ws["y"]), cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S += eig_reduce(wcpps[:2], "      ")
            S.append("    } else {")
            S += eig_reduce(wcpps[2:], "      ")
            S += ["    }", "  }", ""]
        else:
            # SANS eigenvalues : borne de Rusanov / CFL = max(|smin|, |smax|) des extremes du
            # spectre du jacobien (memes blocs que wave_speeds : Rusanov et HLL partagent la
            # meme verite).
            S.append("    adc::Real lo_ = adc::Real(0), hi_ = adc::Real(0);")
            jac_same_blocks = self._ws_jacobian["blocks"]["x"] == self._ws_jacobian["blocks"]["y"]
            if self._ws_jacobian["eig"] == "fd" and jac_same_blocks:
                S += ws_jac_body("    ", "lo_", "hi_")
            elif self._ws_jacobian["eig"] == "fd":
                S.append("    if (dir == 0) {")
                S += ws_jac_body("      ", "lo_", "hi_", "x")
                S.append("    } else {")
                S += ws_jac_body("      ", "lo_", "hi_", "y")
                S.append("    }")
            else:
                ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
                S.append("    if (dir == 0) {")
                S += ptx[0]
                S += ws_jac_body("      ", "lo_", "hi_", "x", ptx[1])
                S.append("    } else {")
                S += pty[0]
                S += ws_jac_body("      ", "lo_", "hi_", "y", pty[1])
                S.append("    }")
            S.append("    const adc::Real alo_ = lo_ < 0 ? -lo_ : lo_;")
            S.append("    const adc::Real ahi_ = hi_ < 0 ? -hi_ : hi_;")
            S += ["    return alo_ > ahi_ ? alo_ : ahi_;", "  }", ""]

        # pression : emise SI une primitive 'p' (pression) est declaree (convention compressible) ;
        # requise par les flux HLLC / Roe canoniques (make_block : requires { m.pressure(s); }).
        if "p" in self.prim_defs:
            S.append("  ADC_HD adc::Real pressure(const State& U) const {")
            S += cons_locals() + prim_locals()
            S += ["    return p;", "  }", ""]

        # vitesses d'onde SIGNEES wave_speeds(U, aux, dir, smin, smax) : gate HLL du coeur
        # (block_builder.hpp requires { m.wave_speeds(...) }). Deux sources, par priorite :
        #   1. paire EXPLICITE set_wave_speeds (smin, smax par direction) -- INDEPENDANTE de 'p' :
        #      un modele sans pression (moments, isotherme...) accede a riemann='hll' ;
        #   2. historique : min/max des eigenvalues, emise SEULEMENT si 'p' est declaree
        #      (convention compressible HLLC / Roe, bit-identique a l'existant).
        # Sans aucune des deux (p.ex. transport scalaire ExB) : rien d'emis, Rusanov seul, inchange.
        if self._wave_speeds is not None:
            ws = self._wave_speeds
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % aux_param)
            S += cons_locals() + prim_locals() + aux_locals()
            wtl, wcpps = self._codegen_exprs(list(ws["x"]) + list(ws["y"]), cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S.append("      smin = %s; smax = %s;" % (wcpps[0], wcpps[1]))
            S.append("    } else {")
            S.append("      smin = %s; smax = %s;" % (wcpps[2], wcpps[3]))
            S += ["    }", "  }", ""]
        elif self._ws_jacobian is not None:
            # vitesses EXACTES par valeurs propres du jacobien (cf. set_wave_speeds_from_jacobian :
            # 'numeric' = entrees en formules, 'fd' = colonnes aux differences finies du flux
            # compile ; extremes par sous-blocs via adc::real_eig_minmax, repli Gershgorin sur).
            ws_aux = aux_param if self._ws_jacobian["eig"] != "fd" else "const Aux& a"
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % ws_aux)
            S += cons_locals() + prim_locals() + aux_locals()
            ws_same_blocks = self._ws_jacobian["blocks"]["x"] == self._ws_jacobian["blocks"]["y"]
            if self._ws_jacobian["eig"] == "fd" and ws_same_blocks:
                S += ws_jac_body("    ", "smin", "smax")
            elif self._ws_jacobian["eig"] == "fd":
                S.append("    if (dir == 0) {")
                S += ws_jac_body("      ", "smin", "smax", "x")
                S.append("    } else {")
                S += ws_jac_body("      ", "smin", "smax", "y")
                S.append("    }")
            else:
                ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
                S.append("    if (dir == 0) {")
                S += ptx[0]
                S += ws_jac_body("      ", "smin", "smax", "x", ptx[1])
                S.append("    } else {")
                S += pty[0]
                S += ws_jac_body("      ", "smin", "smax", "y", pty[1])
                S.append("    }")
            S += ["  }", ""]
        elif "p" in self.prim_defs:
            nx = len(self._eig["x"])
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % aux_param)
            S += cons_locals() + prim_locals() + aux_locals()
            wtl, wcpps = self._codegen_exprs(self._eig["x"] + self._eig["y"], cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S += eig_minmax(wcpps[:nx], "      ")
            S.append("    } else {")
            S += eig_minmax(wcpps[nx:], "      ")
            S += ["    }", "  }", ""]

        # CAPABILITY HLLC (m.enable_hllc, audit vague 3) : contact_speed (Toro) + hllc_star_state
        # GENERES depuis les ROLES du bloc (aucun indice litteral : Density/MomentumX/MomentumY
        # resolus, Energy optionnelle, toute autre composante advectee passivement Us[c]=fac*U[c]/r).
        # Le coeur (HasHLLCStructure) applique alors l'algorithme HLLC generique -- y compris sur un
        # modele NON Euler (isotherme 3-var, moments + scalaires passifs). Sans appel : rien d'emis.
        if self._hllc:
            roles_l = roles_for(self.cons_names, self.cons_roles)
            if "p" not in self.prim_defs:
                raise ValueError("enable_hllc : la primitive 'p' (pression) doit etre declaree "
                                 "(m.primitive('p', ...)) -- contact_speed/star state en dependent")
            try:
                iD = roles_l.index("Density")
                iX = roles_l.index("MomentumX")
                iY = roles_l.index("MomentumY")
            except ValueError:
                raise ValueError("enable_hllc : roles Density / MomentumX / MomentumY requis "
                                 "(declarer conservative_vars(..., roles=[...])) ; roles actuels %r"
                                 % (roles_l,))
            iE = roles_l.index("Energy") if "Energy" in roles_l else -1
            S.append("  // CAPABILITY HLLC generee depuis les ROLES (enable_hllc) : algorithme")
            S.append("  // contact-resolving generique du coeur (HasHLLCStructure), aucun layout fige.")
            S.append("  ADC_HD adc::Real contact_speed(const State& UL, const State& UR, "
                     "adc::Real pL, adc::Real pR, adc::Real sL, adc::Real sR, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const adc::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
            S.append("    const adc::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
            S.append("    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /")
            S.append("           (rL * (sL - unL) - rR * (sR - unR));")
            S += ["  }", ""]
            S.append("  ADC_HD State hllc_star_state(const State& U, adc::Real p, adc::Real s, "
                     "adc::Real sStar, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const adc::Real r = U[%d];" % iD)
            S.append("    const adc::Real un = U[in_] / r;")
            S.append("    const adc::Real fac = r * (s - un) / (s - sStar);")
            S.append("    State Us{};")
            S.append("    for (int c = 0; c < %d; ++c) Us[c] = fac * (U[c] / r);  "
                     "// defaut : advection passive" % nc)
            S.append("    Us[%d] = fac;" % iD)
            S.append("    Us[in_] = fac * sStar;")
            if iE >= 0:
                S.append("    Us[%d] = fac * (U[%d] / r + (sStar - un) * (sStar + p / "
                         "(r * (s - un))));" % (iE, iE))
            S += ["    return Us;", "  }", ""]

        # CAPABILITY ROE (m.enable_roe, solde de l'audit) : roe_dissipation = |A_roe| (UR - UL)
        # GENEREE depuis les ROLES. Avec Energy : TRANSCRIPTION exacte de l'algebre canonique
        # Euler du coeur (numerical_flux.hpp), gamma-1 deduit de p/(E - 1/2 rho |v|^2) -- parite
        # numerique attendue avec le chemin historique. Sans Energy : meme decomposition sans la
        # ligne E, c = sqrt(p/rho) par cote puis moyenne de Roe (generalisation standard). Les
        # composantes HORS roles fluides sont des scalaires passifs portes par l'onde entropique
        # (ligne tangentielle, phi = q/rho). Le coeur (HasRoeDissipation) fait F = 1/2(FL+FR) - d/2.
        if self._roe:
            roles_l = roles_for(self.cons_names, self.cons_roles)
            if "p" not in self.prim_defs:
                raise ValueError("enable_roe : la primitive 'p' (pression) doit etre declaree "
                                 "(m.primitive('p', ...)) -- la linearisation de Roe en depend")
            try:
                iD = roles_l.index("Density")
                iX = roles_l.index("MomentumX")
                iY = roles_l.index("MomentumY")
            except ValueError:
                raise ValueError("enable_roe : roles Density / MomentumX / MomentumY requis "
                                 "(declarer conservative_vars(..., roles=[...])) ; roles actuels %r"
                                 % (roles_l,))
            iE = roles_l.index("Energy") if "Energy" in roles_l else -1
            passives = [c for c in range(nc) if c not in (iD, iX, iY, iE)]
            S.append("  // CAPABILITY ROE generee depuis les ROLES (enable_roe) : dissipation")
            S.append("  // |A_roe| dU du coeur generique (HasRoeDissipation), aucun layout fige.")
            S.append("  ADC_HD State roe_dissipation(const State& UL, const adc::Aux&, "
                     "const State& UR, const adc::Aux&, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const int it_ = dir == 0 ? %d : %d;" % (iY, iX))
            S.append("    const adc::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
            S.append("    const adc::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
            S.append("    const adc::Real utL = UL[it_] / rL, utR = UR[it_] / rR;")
            S.append("    const adc::Real pL = pressure(UL), pR = pressure(UR);")
            S.append("    const adc::Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), "
                     "den = sqL + sqR;")
            S.append("    const adc::Real un = (sqL * unL + sqR * unR) / den;")
            S.append("    const adc::Real ut = (sqL * utL + sqR * utR) / den;")
            S.append("    const adc::Real rho = sqL * sqR;")
            if iE >= 0:
                S.append("    // gaz parfait : H de Roe + gamma-1 deduit (algebre canonique du coeur)")
                S.append("    const adc::Real HL = (UL[%d] + pL) / rL, HR = (UR[%d] + pR) / rR;"
                         % (iE, iE))
                S.append("    const adc::Real H = (sqL * HL + sqR * HR) / den;")
                S.append("    const adc::Real q2 = un * un + ut * ut;")
                S.append("    const adc::Real gm1 = pL / (UL[%d] - adc::Real(0.5) * rL * "
                         "(unL * unL + utL * utL));" % iE)
                S.append("    const adc::Real c2 = gm1 * (H - adc::Real(0.5) * q2);")
                S.append("    const adc::Real c = std::sqrt(c2);")
            else:
                S.append("    // sans Energy : c LOCAL = sqrt(p/rho) par cote, moyenne de Roe")
                S.append("    const adc::Real c = (sqL * std::sqrt(pL / rL) + sqR * "
                         "std::sqrt(pR / rR)) / den;")
                S.append("    const adc::Real c2 = c * c;")
            S.append("    const adc::Real dr = rR - rL, dp = pR - pL, dun = unR - unL, "
                     "dut = utR - utL;")
            S.append("    const adc::Real a1 = (dp - rho * c * dun) / (adc::Real(2) * c2);")
            S.append("    const adc::Real a2 = dr - dp / c2;")
            S.append("    const adc::Real a3 = rho * dut;")
            S.append("    const adc::Real a5 = (dp + rho * c * dun) / (adc::Real(2) * c2);")
            S.append("    // correction d'entropie de Harten, MEME politique que le chemin")
            S.append("    // canonique : eps = adc::kRoeEntropyFixFraction * c (0.1, Euler/Roe).")
            S.append("    const adc::Real eps = adc::Real(0.1) * c;")
            S.append("    const adc::Real l1r = un - c, l5r = un + c;")
            S.append("    const adc::Real al1 = (l1r < 0 ? -l1r : l1r) < eps ? "
                     "adc::Real(0.5) * (l1r * l1r / eps + eps) : (l1r < 0 ? -l1r : l1r);")
            S.append("    const adc::Real al2 = un < 0 ? -un : un;")
            S.append("    const adc::Real al5 = (l5r < 0 ? -l5r : l5r) < eps ? "
                     "adc::Real(0.5) * (l5r * l5r / eps + eps) : (l5r < 0 ? -l5r : l5r);")
            S.append("    State d{};")
            S.append("    d[%d] = al1 * a1 + al2 * a2 + al5 * a5;" % iD)
            S.append("    d[in_] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);")
            S.append("    d[it_] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;")
            if iE >= 0:
                S.append("    d[%d] = al1 * a1 * (H - un * c) + al2 * (a2 * adc::Real(0.5) * q2 "
                         "+ a3 * ut) + al5 * a5 * (H + un * c);" % iE)
            for cpa in passives:
                S.append("    {  // scalaire passif [%d] : porte par l'onde entropique (phi = q/rho)"
                         % cpa)
                S.append("      const adc::Real fL = UL[%d] / rL, fR = UR[%d] / rR;" % (cpa, cpa))
                S.append("      const adc::Real ft = (sqL * fL + sqR * fR) / den;")
                S.append("      d[%d] = al1 * a1 * ft + al2 * (a2 * ft + rho * (fR - fL)) "
                         "+ al5 * a5 * ft;" % cpa)
                S.append("    }")
            S += ["    return d;", "  }", ""]

        # CAPABILITY ROE FOURNIE (m.roe_dissipation) : pendant 'utilisateur' de enable_roe. Les lignes
        # d_i viennent de l'utilisateur (son eigenstructure), ecrites en left()/right() des deux etats.
        # On emet le MEME hook roe_dissipation(UL, AL, UR, AR, dir) que la voie roles (trait
        # HasRoeDissipation, le coeur fait F = 1/2(FL+FR) - 1/2 d). left(e) -> e sur les locales L_
        # (calculees depuis UL), right(e) -> locales R_ (depuis UR). _roe_rows et _roe sont exclusifs
        # (garde-fou a la declaration et dans check()).
        if self._roe_rows is not None:
            has_aux = bool(self.aux_names)  # parametres Aux nommes aL/aR uniquement si des aux existent
            aL = "const adc::Aux& aL" if has_aux else "const adc::Aux&"
            aR = "const adc::Aux& aR" if has_aux else "const adc::Aux&"
            S.append("  // CAPABILITY ROE FOURNIE (m.roe_dissipation) : dissipation d ecrite par")
            S.append("  // l'utilisateur via left()/right() des deux etats ; hook HasRoeDissipation.")
            S.append("  ADC_HD State roe_dissipation(const State& UL, %s, const State& UR, %s, "
                     "int dir) const {" % (aL, aR))
            # locales des DEUX etats : conservatives, primitives (def avec prefixe), puis aux lus.
            for side, U, av in (("L_", "UL", "aL"), ("R_", "UR", "aR")):
                S += ["    const adc::Real %s%s = %s[%d];" % (side, c, U, i)
                      for i, c in enumerate(self.cons_names)]
                S += ["    const adc::Real %s%s = %s;" % (side, p, _cpp_roe(e, side))
                      for p, e in self.prim_defs.items()]
                if has_aux:
                    S += ["    const adc::Real %s%s = %s.%s;" % (side, n, av, n)
                          for n in self.aux_names]
            S.append("    State d{};")
            S.append("    if (dir == 0) {")
            S += ["      d[%d] = %s;" % (i, _cpp_roe(self._roe_rows["x"][i], None)) for i in range(nc)]
            S.append("    } else {")
            S += ["      d[%d] = %s;" % (i, _cpp_roe(self._roe_rows["y"][i], None)) for i in range(nc)]
            S += ["    }", "    return d;", "  }", ""]

        # Bornes de pas OPTIONNELLES (m.stability_speed / m.stability_dt) : emises comme les traits
        # C++ HasStabilitySpeed / HasStabilityDt (cf. adc/core/physical_model.hpp). Une seule
        # expression (isotrope) : dir est ignore. SANS appel, rien n'est emis -> fallback strict
        # max_wave_speed (politique de pas historique).
        if self._stab_speed is not None:
            S.append("  ADC_HD adc::Real stability_speed(const State& U, %s, int dir) const {"
                     % aux_param)
            S.append("    (void)dir;  // borne isotrope : une seule expression pour les deux directions")
            S += cons_locals() + prim_locals() + aux_locals()
            stl, scpps = self._codegen_exprs([self._stab_speed], cse)
            S += stl
            S += ["    return %s;" % scpps[0], "  }", ""]
        if self._stab_dt is not None:
            S.append("  ADC_HD adc::Real stability_dt(const State& U, %s) const {" % aux_param)
            S += cons_locals() + prim_locals() + aux_locals()
            dtl, dcpps = self._codegen_exprs([self._stab_dt], cse)
            S += dtl
            S += ["    return %s;" % dcpps[0], "  }", ""]

        S.append("  ADC_HD Prim to_primitive(const State& U) const {")
        S += cons_locals() + prim_locals()
        S.append("    Prim P{};")
        S += ["    P[%d] = %s;" % (i, p) for i, p in enumerate(self.prim_state)]
        S += ["    return P;", "  }", ""]

        S.append("  ADC_HD State to_conservative(const Prim& P) const {")
        S += ["    const adc::Real %s = P[%d];" % (p, i) for i, p in enumerate(self.prim_state)]
        ctl, ccpps = self._codegen_exprs(self.cons_from, cse)
        S += ctl
        S.append("    State U{};")
        S += ["    U[%d] = %s;" % (i, c) for i, c in enumerate(ccpps)]
        S += ["    return U;", "  }", ""]

        cons_set = "{adc::VariableKind::Conservative, {%s}, %d%s}" % (
            cnames, nc, (", {%s}" % croles) if croles is not None else "")
        prim_set = "{adc::VariableKind::Primitive, {%s}, %d%s}" % (
            pnames, npr, (", {%s}" % proles) if proles is not None else "")
        S.append('  static adc::VariableSet conservative_vars() { return %s; }' % cons_set)
        S.append('  static adc::VariableSet primitive_vars() { return %s; }' % prim_set)
        S += ["};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def emit_cpp_source(self, name=None, namespace="adc_generated", cse=True):
        """Genere une BRIQUE de SOURCE C++ composable (au sens adc) depuis self._source.

        Le struct produit expose apply(U, a) renvoyant le terme source S(U, aux), avec une ligne par
        composante conservative (S[i] = self._source[i].to_cpp()). Il a la meme forme que les briques
        de source ecrites a la main (NoSource, PotentialForce dans adc/model/bricks.hpp) et peut donc
        entrer comme parametre Source d'un CompositeModel.

        CONVENTION : les noms auxiliaires (poses via aux(...)) doivent etre des CHAMPS de adc::Aux,
        car ils sont lus directement comme a.<nom> (p.ex. aux('grad_x') -> a.grad_x, aux('grad_y') ->
        a.grad_y). Cette convention est la meme que celle des briques manuelles, ou la source ne lit
        l'etat exterieur que par le canal adc::Aux (potentiel et son gradient).

        Style identique a emit_cpp_brick (constantes inlinees, cons -> locals, primitives -> locals ;
        en plus, aux -> locals) ; cse=True factorise les sous-expressions communes. Leve ValueError si
        set_source(...) n'a pas ete appele."""
        if self._source is None:
            raise ValueError("emit_cpp_source : appeler set_source([...]) d'abord")
        nm = name or (self.name.capitalize() + "Source")
        nc = self.n_vars

        def cons_locals():
            return ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]

        def prim_locals():
            return ["    const adc::Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]

        def aux_locals():
            return self._aux_locals_lines()  # canonique (a.<n>) + nommes (a.extra_field(k)), ADC-70

        na = self._total_n_aux()  # largeur aux requise (B_z / T_e / champs nommes -> > 3)
        rt_member = self._runtime_params_member()  # P7-b : indices runtime AVANT tout to_cpp()
        S = [
            "#include <cmath>",  # autosuffisant pour std::sqrt / std::pow
            "// brique de SOURCE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_source)."
            % self.name,
            "// apply(U, a) -> terme source S(U, aux) ; noms aux = champs de adc::Aux (grad_x, grad_y).",
        ]
        if rt_member:  # en-tete RuntimeParams uniquement si une formule lit un param runtime
            S.append("#include <adc/runtime/runtime_params.hpp>")
        if self._ws_jacobian is not None:  # valeurs propres de blocs denses (wave_speeds exacts)
            S.append("#include <adc/numerics/dense_eig.hpp>")
        S += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
        ]
        if rt_member:  # membre adc::RuntimeParams params{count, {defauts}} (P7-b)
            S.append(rt_member.rstrip("\n"))
        # Si une formule lit un champ aux SUPPLEMENTAIRE (B_z...), declarer n_aux : CompositeModel le
        # propage (max sur les briques) et le systeme dimensionne/peuple le canal aux partage. Sans
        # champ extra -> pas de n_aux emis -> brique strictement identique a l'historique.
        if na > AUX_BASE_COMPS:
            S.append("  static constexpr int n_aux = %d;" % na)
        S.append("  ADC_HD adc::StateVec<%d> apply(const adc::StateVec<%d>& U, const adc::Aux& a) const {"
                 % (nc, nc))
        S += cons_locals() + prim_locals() + aux_locals()
        # _wrap : une composante peut etre un litteral Python (p.ex. 0.0), promu en Const.
        stl, scpps = self._codegen_exprs([_wrap(e) for e in self._source], cse)
        S += stl
        S.append("    adc::StateVec<%d> S{};" % nc)
        S += ["    S[%d] = %s;" % (i, c) for i, c in enumerate(scpps)]
        S += ["    return S;", "  }"]
        # FREQUENCE de la source (m.source_frequency, audit vague 2) : emise comme frequency(U, a)
        # -- le contrat OPTIONNEL des briques source (cf. physics/source.hpp), forwarde par
        # CompositeModel (HasSourceFrequency) et agrege par step_cfl. Sans appel : rien d'emis.
        if self._src_freq is not None:
            S.append("")
            S.append("  ADC_HD adc::Real frequency(const adc::StateVec<%d>& U, const adc::Aux& a) "
                     "const {" % nc)
            S += cons_locals() + prim_locals() + aux_locals()
            ftl, fcpps = self._codegen_exprs([self._src_freq], cse)
            S += ftl
            S += ["    return %s;" % fcpps[0], "  }"]
        # JACOBIEN ANALYTIQUE (m.source_jacobian, audit vague 3) : emis comme jacobian(U, a, J),
        # forwarde par CompositeModel (HasSourceJacobian) -> le Newton implicite remplace les
        # differences finies. Sans appel : rien d'emis (FD historiques, bit-identique).
        if self._src_jac is not None:
            if len(self._src_jac) != nc or any(len(r) != nc for r in self._src_jac):
                raise ValueError("source_jacobian : matrice %dx%d attendue (dS_r/dU_c)" % (nc, nc))
            S.append("")
            S.append("  ADC_HD void jacobian(const adc::StateVec<%d>& U, const adc::Aux& a, "
                     "adc::Real (&J)[%d][%d]) const {" % (nc, nc, nc))
            S += cons_locals() + prim_locals() + aux_locals()
            flat = [e for row in self._src_jac for e in row]
            jtl, jcpps = self._codegen_exprs(flat, cse)
            S += jtl
            for r in range(nc):
                for c in range(nc):
                    S.append("    J[%d][%d] = %s;" % (r, c, jcpps[r * nc + c]))
            S += ["  }"]
        S += ["};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def _emit_bricks(self, name=None):
        """Genere les briques (hyperbolique + source + elliptique) et le type CompositeModel<...>
        partages par les DEUX backends (JIT IModel et AOT). Source / elliptique OPTIONNELS : sans
        set_source -> adc::NoSource ; sans set_elliptic_rhs -> rhs nul (pas de couplage Poisson).
        Renvoie (nv, code_des_briques, type_composite)."""
        nm = name or (self.name.capitalize() + "Gen")
        nv = self.n_vars
        # Garde-fou du CODEGEN (pas seulement de check(), que compile() n'appelle pas) : une
        # frequence ou un jacobien de source sans m.source(...) serait PURGE en silence par la
        # branche NoSource ci-dessous -- rejet explicite (regle : jamais d'option ignoree).
        if self._source is None:
            if self._src_freq is not None:
                raise ValueError("source_frequency(...) declare sans source : appelez "
                                 "m.source([...]) (la frequence est emise sur la brique de source)")
            if self._src_jac is not None:
                raise ValueError("source_jacobian(...) declare sans source : appelez "
                                 "m.source([...]) (le jacobien est emis sur la brique de source)")
        parts = [self.emit_cpp_brick(name=nm + "Hyp")]
        if self._source is not None:  # brique de source generee, sinon NoSource
            parts.append(self.emit_cpp_source(name=nm + "Src"))
            src_type = "adc_generated::%sSrc" % nm
        else:
            src_type = "adc::NoSource"
        if self._elliptic is not None:  # brique elliptique generee, sinon rhs nul (pas de couplage)
            parts.append(self.emit_cpp_elliptic(name=nm + "Ell"))
        else:
            parts.append(
                "namespace adc_generated { struct %sEll {\n"
                "  template <class State> ADC_HD adc::Real rhs(const State&) const { return adc::Real(0); }\n"
                "}; }\n" % nm)
        composite = ("adc::CompositeModel<adc_generated::%sHyp, %s, adc_generated::%sEll>"
                     % (nm, src_type, nm))
        return nv, "".join(parts), composite

    def _emit_metadata(self, model_alias):
        """Symboles OPTIONNELS de metadonnees du bloc .so, lus par dlsym cote System. PARTAGES par les
        deux backends (JIT et AOT). Les NOMS + ROLES sont toujours emis (ADC_EXPORT_BLOCK_METADATA) :
        ils viennent du VariableSet du modele (source unique de verite), le System les lit au lieu du
        fallback u0.. / pas de roles. Le GAMMA n'est emis (ADC_EXPORT_BLOCK_GAMMA) que si set_gamma(...)
        a ete appele ; sinon aucun symbole gamma -> le System garde son defaut 1.4 (retro-compat).

        @p model_alias doit etre un alias SANS virgule de niveau superieur (le preprocesseur decoupe
        les arguments de macro sur les virgules) : les callers passent un `using ... = CompositeModel<...>`."""
        out = "\nADC_EXPORT_BLOCK_METADATA(%s)\n" % model_alias
        if self.gamma is not None:
            out += "ADC_EXPORT_BLOCK_GAMMA(%r)\n" % self.gamma
        # Table des noms d'aux NOMMES (aux_field, ADC-70), CSV ordonne (ordre = indice AUX_NAMED_BASE +
        # k). Symbole OPTIONNEL, motif des noms/roles : rend le .so AUTO-DESCRIPTIF (un loader C++
        # pourrait resoudre nom -> composante ; cote Python la table vit deja dans CompiledModel). Emis
        # SEULEMENT si le modele declare des champs nommes -> retro-compatible (.so sans champ nomme
        # inchange, symbole absent).
        if self.aux_extra_names:
            # Noms = identifiants C++ valides (valides dans aux_field) -> CSV sans guillemet, litteral
            # C sur (uniquement [A-Za-z0-9_,]).
            out += ('extern "C" const char* adc_compiled_aux_extra_names() { return "%s"; }\n'
                    % ",".join(self.aux_extra_names))
        return out

    def emit_cpp_so_source(self, name=None):
        """Source de la bibliotheque JIT (backend "jit") : le MODELE COMPLET en CompositeModel<GenHyp,
        GenSrc, GenEll> derriere une fabrique extern "C" (adc_model_nvars / adc_make_model /
        adc_destroy_model via adc::ModelAdapter). C'est ce que compile_so compile et que
        System.add_dynamic_block charge comme bloc couple a DISPATCH VIRTUEL (prototypage hote)."""
        nv, bricks, composite = self._emit_bricks(name)
        return ('#include <adc/runtime/dynamic_model.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'  # CompositeModel + NoSource + briques
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using JitModel = %s; }\n' % composite  # alias sans virgule (macro metadata)
                + 'extern "C" int adc_model_nvars() { return %d; }\n' % nv
                + 'extern "C" void* adc_make_model() { return new adc::ModelAdapter<adc_generated::JitModel>(); }\n'
                + 'extern "C" void adc_destroy_model(void* p) { delete static_cast<adc::IModel<%d>*>(p); }\n' % nv
                + self._emit_metadata("adc_generated::JitModel"))

    def compile_so(self, so_path, include=None, name=None, cxx=None, std="c++20"):
        """JIT : genere le MODELE COMPLET (emit_cpp_so_source) et compile une bibliotheque partagee
        chargeable par System.add_dynamic_block (dlopen). Le .so expose un CompositeModel<hyperbolique,
        source, elliptique> : le bloc dynamique applique le flux ET la source, et contribue au Poisson
        de systeme via elliptic_rhs (vrai bloc couple, plus seulement transport). include = dossier des
        en-tetes adc (None -> auto-detecte via adc_include()) ; cxx = compilateur (defaut
        c++/g++/clang++). Renvoie so_path. Exige set_primitive_state(...) et
        set_conservative_from([...]) (comme emit_cpp_brick)."""
        import os
        import shutil
        import subprocess
        import tempfile

        if include is None:
            include = adc_include()
        src = self.emit_cpp_so_source(name=name)
        cc = _default_cxx(cxx)
        if not cc:
            raise RuntimeError("compile_so : aucun compilateur C++ trouve")
        std = _probe_cxx_std(cc, std)  # erreur ACTIONNABLE si le std n'est pas supporte (vs erreur brute)
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                          "-o", so_path], "backend jit, compile_so")
        return so_path

    def emit_cpp_aot_source(self, name=None):
        """Source de la bibliotheque AOT (backend "compile") : le MODELE COMPLET en CompositeModel<...>
        derriere l'ABI extern "C" de compiled_block_abi.hpp. Le .so EXECUTE le chemin de PRODUCTION
        (assemble_rhs<Limiter, Flux>, SSPRK2/IMEX du coeur) sur le modele genere : numerique inlinee,
        identique a un bloc natif add_block. Oppose au backend "jit" (IModel, dispatch virtuel)."""
        nv, bricks, composite = self._emit_bricks(name)
        return ('#include <adc/runtime/compiled_block_abi.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'  # CompositeModel + NoSource + briques
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using AotModel = %s; }\n' % composite
                + 'ADC_DEFINE_COMPILED_BLOCK(adc_generated::AotModel)\n'
                + self._emit_metadata("adc_generated::AotModel"))  # alias sans virgule (macro metadata)

    def compile_aot(self, so_path, include=None, name=None, cxx=None, std="c++20"):
        """Backend "compile" (AOT) : genere le MODELE COMPLET (emit_cpp_aot_source) et compile une .so
        chargeable par System.add_compiled_block. Contrairement au backend "jit" (compile_so : IModel,
        dispatch virtuel, Rusanov hote), le bloc tourne ici le chemin de PRODUCTION (flux HLLC/Roe au
        choix, ordre 2, SSPRK2/IMEX) sur le modele genere -- numerique identique a un bloc natif.
        include = dossier des en-tetes adc (None -> auto-detecte via adc_include()) ; cxx = compilateur.
        Renvoie so_path.

        KOKKOS-ONLY : le modele AOT inclut les en-tetes adc (multifab/for_each), qui ne compilent PAS
        sans ADC_HAS_KOKKOS. On compile donc le .so AVEC Kokkos (memes flags que le loader natif), ce qui
        aligne en plus son ABI sur le module _adc (lui aussi Kokkos). Un Kokkos installe doit etre visible
        via ADC_KOKKOS_ROOT / Kokkos_ROOT (Serial suffit sur CPU)."""
        import os
        import subprocess
        import tempfile

        if include is None:
            include = adc_include()
        src = self.emit_cpp_aot_source(name=name)
        if _native_kokkos_root() is None:
            raise RuntimeError(
                "compile_aot : adc_cpp est Kokkos-only -- le modele AOT inclut les en-tetes adc qui "
                "exigent Kokkos. Pointe un Kokkos installe via ADC_KOKKOS_ROOT (ou Kokkos_ROOT), p.ex. "
                "`export ADC_KOKKOS_ROOT=/chemin/vers/kokkos` (Serial suffit sur CPU).")
        cc = _native_kokkos_compiler(cxx)
        if not cc:
            raise RuntimeError("compile_aot : aucun compilateur C++ trouve")
        std = _probe_cxx_std(cc, std)  # erreur ACTIONNABLE si le std n'est pas supporte (vs erreur brute)
        kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
        # Comme le loader natif, le .so AOT laisse les symboles Kokkos INDEFINIS (resolus au chargement
        # contre le runtime Kokkos deja charge par _adc -- pas de 2e copie). macOS/Apple-ld exige alors
        # -undefined dynamic_lookup (sur ELF/Linux -shared l'autorise deja ; l'option n'est PAS du ld GNU).
        link_extra = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_aot.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include]
                         + kokkos_compile_flags + link_extra + [cpp, "-o", so_path] + kokkos_link_flags,
                         "backend aot, compile_aot")
        return so_path

    def emit_cpp_native_loader(self, name=None, target="system"):
        """Source du LOADER NATIF (backend "production") : le MODELE COMPLET en CompositeModel<...>
        derriere une ABI extern "C" MINCE.

        A la difference du backend "aot" (emit_cpp_aot_source : ABI plate de tableaux, le .so
        recalcule tout sur une grille locale et marshale les tableaux), le loader natif NE porte PAS
        la numerique : il se contente d'INSTALLER le modele genere comme bloc NATIF de la facade deja
        construite, via le gabarit en-tete adc::add_compiled_model<ProdModel>. Ce gabarit fabrique les
        fermetures sur le CONTEXTE REEL de la facade -> le bloc tourne ensuite le MEME chemin que
        add_block, ZERO-COPIE, device-clean (foncteurs nommes).

        @p target : "system" (defaut) | "amr_system". Choisit la facade visee et donc la SURCHARGE
        add_compiled_model appelee :

        - "system" : adc::System -> add_compiled_model(System&, ..., evolve) ; bloc plat
          mono-niveau (fermetures sur grid_context, chemin de production de add_block).
        - "amr_system" : adc::AmrSystem -> add_compiled_model(AmrSystem&, ...) ; bloc unique porte
          sur la hierarchie AMR (reflux conservatif, regrid). PAS de parametre evolve (AMR mono-bloc).

        Symboles extern "C" emis :

        - adc_native_abi_key() : cle d'ABI figee a la compilation DU LOADER, emise comme LITTERAL
          preprocesseur (ADC_ABI_KEY_LITERAL) et PAS via la fonction inline abi_key_string() : sous
          ELF/RTLD_GLOBAL, une inline (liaison faible) serait interposee vers la copie du module et
          le loader renverrait la cle DU MODULE (garde tautologique, jamais de rejet -- bug CI reel
          quand gcc cesse d'inliner). add_native_block la compare a abi_key() du module -> erreur
          explicite si en-tetes / compilateur / standard divergent (pas d'UB silencieux).
          Commune aux deux cibles.
        - adc_install_native (target="system") OU adc_install_native_amr (target="amr_system") :
          reinterpret_cast<adc::System*|adc::AmrSystem*>(sys) puis add_compiled_model<ProdModel>(...).
          Le schema transite en arguments plats (chaines + double + int) ; aucun objet C++ ne
          traverse l'ABI dans CE sens (seul le facade* est repris par reference cote loader, d'ou
          l'exigence d'ABI identique verifiee par la cle). Symbole DISTINCT par cible : un loader
          System ne peut pas etre branche sur AmrSystem.add_native_block, et inversement."""
        if target not in ("system", "amr_system"):
            raise ValueError("emit_cpp_native_loader : target 'system' | 'amr_system' (recu %r)"
                             % (target,))
        nv, bricks, composite = self._emit_bricks(name)
        # En-tetes std EN TETE (avant tout namespace). MSVC : un #include <std> tandis qu'un namespace
        # adc est ouvert fait voir std comme adc::std (erreurs <vector>) ; g++ tolere car deja inclus via
        # guard. Les hisser ici rend les #include std internes aux briques inoffensifs (no-op guard).
        head = ('#include <cmath>\n'
                '#include <vector>\n'
                '#include <array>\n'
                '#include <cstddef>\n'
                '#include <string>\n'
                '#include <adc/runtime/abi_key.hpp>\n'         # ADC_ABI_KEY_LITERAL (cle figee a la compil)
                '#include <adc/physics/bricks.hpp>\n'          # CompositeModel + NoSource + briques
                '#include <adc/core/variables.hpp>\n')
        # Gabarit en-tete de la cible : dsl_block.hpp (System) ou amr_dsl_block.hpp (AmrSystem). Inclus
        # selectivement pour ne pas tirer la machinerie AMR dans un loader System (et inversement).
        head += ('#include <adc/runtime/dsl_block.hpp>\n' if target == "system"
                 else '#include <adc/runtime/amr_dsl_block.hpp>\n')
        # LITTERAL preprocesseur, pas d'appel a abi_key_string() : une inline serait interposee
        # (ELF/RTLD_GLOBAL) vers la copie du module -> cle du module renvoyee -> garde tautologique.
        key = ('#if defined(_WIN32)\n'
               '#define ADC_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define ADC_LOADER_API extern "C"\n'
               '#endif\n'
               'ADC_LOADER_API const char* adc_native_abi_key() {\n'
               '  return ADC_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, limiteur de positivite Zhang-Shu) : argument plat final, marshale
            # jusqu'au make_block du loader via add_compiled_model. Vieille signature = vieux .so =
            # rejete par la cle d'ABI (les en-tetes ont change), jamais un layout d'arguments errone.
            install = ('ADC_LOADER_API void adc_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  adc::System* s = reinterpret_cast<adc::System*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system : surcharge AmrSystem (pas de parametre evolve, AMR mono-bloc)
            install = ('ADC_LOADER_API void adc_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head
                + bricks
                + '\nnamespace adc_generated { using ProdModel = %s; }\n' % composite  # alias sans virgule
                + key
                + install
                + self._emit_metadata("adc_generated::ProdModel"))  # noms/roles/gamma (diagnostic, comme AOT/JIT)

    def compile_native(self, so_path, include=None, name=None, cxx=None, std="c++23", target="system"):
        """Backend "production" : genere le LOADER NATIF (emit_cpp_native_loader) et le compile en une
        .so chargeable par System.add_native_block (target="system") ou AmrSystem.add_native_block
        (target="amr_system"). Le .so inline add_compiled_model<ProdModel> : le bloc tourne le chemin
        NATIF zero-copie (parite stricte avec add_block / add_compiled_model<>).

        @p target : "system" (defaut) | "amr_system" (cf. emit_cpp_native_loader). Selectionne la
        facade visee et donc le gabarit en-tete + le symbole d'installation emis.

        Le loader appelle des methodes hors-ligne du module _adc (install_block / grid_context /
        ensure_aux_width cote System ; set_compiled_block cote AmrSystem) DEFINIES ailleurs : on compile
        donc avec '-undefined dynamic_lookup' (macOS) pour autoriser ces indefinis (resolus a
        l'execution contre le module deja charge ; cf. add_native_block). On bake aussi
        -DADC_HEADER_SIG=<signature> a l'IDENTIQUE du module pour que les cles d'ABI concordent quand
        les en-tetes concordent. std : un std different du module changerait __cplusplus donc la cle ->
        rejet explicite ; les appelants (Model.compile/HybridModel.compile) defaultent donc sur la
        norme du loader (loader_cxx_std : c++20 sous Kokkos, c++23 sinon) et non sur c++23 en dur.
        include = dossier des en-tetes adc (None -> auto-detecte via adc_include()) ; cxx = compilateur.
        Renvoie so_path."""
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        if include is None:
            include = adc_include()
        # GARDE PRE-DLOPEN : en-tetes != ceux du build de _adc -> erreur claire ICI ("rebatir le
        # module") au lieu d'un dlopen 'symbol not found' cryptique dans add_native_block. Renvoie la
        # signature calculee (reutilisee pour -DADC_HEADER_SIG : un seul walk+sha256, pas deux).
        sig = _check_headers_match_module(include)
        _warn_kokkos_parity()  # module Kokkos + loader serie (ou l'inverse) -> avertir, ne pas bloquer
        src = self.emit_cpp_native_loader(name=name, target=target)
        cc = _native_kokkos_compiler(cxx)
        if not cc:
            raise RuntimeError("compile_native : aucun compilateur C++ trouve")
        # Probe AVANT compilation : si le compilateur ne supporte pas la norme (cas reel : vieux
        # gcc/clang d'un env conda pris sur le PATH), erreur actionnable au lieu de l'erreur brute
        # "invalid value 'c++23'". Peut retrograder vers l'orthographe c++2b (meme niveau).
        std = _probe_cxx_std(cc, std)
        # -DADC_HEADER_SIG : MEME signature que le build du module (concordance des cles d'ABI).
        #
        # (1) PARITE DE BACKEND (le plus important pour le scaling) : si _adc est compile avec Kokkos
        # (OpenMP/CUDA), le loader DOIT l'etre aussi. Les templates header-only (assemble_rhs /
        # for_each_cell) compiles SANS -DADC_HAS_KOKKOS s'instancient sur le fallback SERIE : le bloc
        # DSL reste zero-copie mais ne scale PAS avec les threads/GPU (mesure ROMEO : DSL warm ~341 ms
        # invariant threads=1/4/8 alors que le natif scale 292->239->177). _native_kokkos_flags() ajoute
        # -DADC_HAS_KOKKOS + includes/libs Kokkos + -fopenmp quand ADC_KOKKOS_ROOT (ou Kokkos_ROOT)
        # pointe une install ; sinon comportement historique serie. Le compilateur suit le backend :
        # g++ (OpenMP) par defaut, nvcc_wrapper SEULEMENT si explicite (CUDA).
        #
        # (2) PARITE D'OPTIMISATION : a -O2 sans -DNDEBUG le noyau genere est ~1.48x le natif (asserts
        # hot-loop + vecto faible) ; -O3 -DNDEBUG => parite (mesure ROMEO CV<1%, ratio 1.04x). Ces
        # flags n'affectent NI l'ABI NI la portabilite. $ADC_DSL_OPTFLAGS surcharge (ex. ajouter
        # -march=native : le .so etant JIT-compile sur la machine -> ~0.88x du natif generique ; non
        # defaut car cache .so partage reutilise sur une micro-archi differente = risque illegal-instr).
        kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_native.cpp")
            # Windows : baker ADC_HEADER_SIG via #define EN TETE de la source (le quoting d'un macro
            # string en ligne de commande cl est ingerable) ; POSIX garde le -D historique ci-dessous.
            src_eff = ('#define ADC_HEADER_SIG "%s"\n' % sig + src) if sys.platform == "win32" else src
            with open(cpp, "w") as f:
                f.write(src_eff)
            if sys.platform == "win32":
                # MSVC/clang-cl (ADC-100) : .dll liee contre kokkoscore.lib (Kokkos PARTAGE) + _adc.lib
                # (symboles System ADC_EXPORT). cl accepte -D/-I ; sortie /Fe ; libs apres /link. Pas de
                # RTLD_GLOBAL : les indefinis sont resolus au LINK de la .dll (import libraries).
                adc_lib = _adc_import_lib()
                if not adc_lib:
                    raise RuntimeError(
                        "compile_native : _adc.lib introuvable a cote du module _adc (necessaire pour "
                        "lier la .dll DSL ; rebatir _adc avec ADC_EXPORT_BUILDING_MODULE).")
                # /DNOMINMAX : windows.h (tire par dynlib.hpp) ne doit pas definir min/max (casse la STL).
                # /bigobj : gros TU template. PAS de /Zc:__cplusplus : garder __cplusplus aligne sur le
                # build du module (sinon la cle d'ABI diverge).
                cl_flags = ["/nologo", "/LD", "/std:" + std, "/O2", "/DNDEBUG", "/EHsc",
                            "/permissive-", "/Zc:preprocessor", "/DNOMINMAX", "/bigobj"] + kokkos_compile_flags
                cmd = ([cc] + cl_flags + ["-I", include, cpp,
                        "/Fe:" + so_path, "/Fo" + tmp + os.sep,
                        "/link"] + kokkos_link_flags + [adc_lib])
            else:
                optflags = os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG").split()
                flags = ["-shared", "-fPIC", "-std=" + std, *optflags,
                         "-DADC_HEADER_SIG=\"%s\"" % sig, *kokkos_compile_flags]
                # macOS/Apple-ld : autoriser explicitement les indefinis (resolus a l'execution).
                if sys.platform == "darwin":
                    flags += ["-undefined", "dynamic_lookup"]
                cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags]
            _run_compile(cmd, "backend production, compile_native")
        return so_path

    def compile_or_jit(self, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                       target="system"):
        """API unifiee (facade de l'ideal m.compile_or_jit()) choisissant le backend :

        - mode="jit" -> compile_so (IModel, dispatch virtuel : prototypage hote, a brancher via
          System.add_dynamic_block) ;
        - mode="compile" -> compile_aot (chemin de production AOT, numerique identique au natif : a
          brancher via System.add_compiled_block) ;
        - mode="native" -> compile_native (loader natif zero-copie : add_compiled_model<> via
          System.add_native_block ou AmrSystem.add_native_block ; chemin "production").

        @p target : "system" (defaut) | "amr_system". UNIQUEMENT consomme par mode="native" (choix de
        la facade visee, cf. compile_native). Les autres modes (jit/compile) ne ciblent que System ;
        un target="amr_system" y est rejete (le chemin .so AMR n'existe que pour le backend natif)."""
        if mode == "jit":
            if target != "system":
                raise ValueError("compile_or_jit : target='amr_system' non supporte en mode 'jit' "
                                 "(le chemin AMR n'existe que pour mode='native')")
            return self.compile_so(so_path, include, name=name, cxx=cxx, std=std)
        if mode == "compile":
            if target != "system":
                raise ValueError("compile_or_jit : target='amr_system' non supporte en mode 'compile' "
                                 "(le chemin AMR n'existe que pour mode='native')")
            return self.compile_aot(so_path, include, name=name, cxx=cxx, std=std)
        if mode == "native":
            return self.compile_native(so_path, include, name=name, cxx=cxx, std=std, target=target)
        raise ValueError("compile_or_jit : mode 'jit' | 'compile' | 'native' (recu %r)" % mode)

    # --- facade de production : un point d'entree unique par INTENTION (backend) -----------------
    # Aiguillage du backend de compilation par INTENTION plutot que par detail d'implementation. Chaque
    # entree designe l'un des moteurs existants (compile_so / compile_aot) ET l'adder System a employer
    # cote execution -- couple ici pour qu'un caller ne branche pas un .so AOT sur add_dynamic_block (ou
    # l'inverse), ce qui chargerait mais avec une ABI/numerique incoherente.
    #   "prototype"  -> compile_so  (JIT, IModel, dispatch virtuel, Rusanov hote ordre 1 ; iteration
    #                   rapide, a brancher via System.add_dynamic_block) ;
    #   "aot"        -> compile_aot (AOT, chemin de PRODUCTION host-marshale : assemble_rhs<Limiter,
    #                   Flux>, HLLC/Roe, ordre 2, SSPRK2/IMEX sur une grille LOCALE du .so ; numerique
    #                   identique au natif mais tableaux marshales, via add_compiled_block) ;
    #   "production" -> compile_native (LOADER NATIF) : le .so inline add_compiled_model<ProdModel>, qui
    #                   installe le modele genere comme bloc NATIF du System (fermetures sur le contexte
    #                   REEL grid_context). Le bloc tourne ZERO-COPIE le MEME chemin qu'add_block (pas de
    #                   marshaling) ; device-clean par construction (foncteurs nommes de block_builder).
    #                   A brancher via System.add_native_block (cle d'ABI verifiee). C'est le chemin
    #                   prepare pour un vrai backend de production (codegen Kokkos/CUDA = PR ulterieure).
    _BACKENDS = {
        "prototype": ("jit", "add_dynamic_block"),
        "aot": ("compile", "add_compiled_block"),
        "production": ("native", "add_native_block"),
    }

    def _model_hash(self, params=None):
        """Hash stable du modele : formules (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + gamma (+ params NOMMES eventuels). Source unique du hash, reutilisee par Model._model_hash
        (qui passe ses Param). Sert a identifier/reutiliser un .so deja compile (cle de cache) et a tracer
        le run. Repose sur repr(Expr) (stable, structurel) ; insensible a l'ordre des dict (tries)."""
        import hashlib
        m = self
        parts = []
        parts.append("name=%s" % m.name)
        parts.append("cons=%s" % ",".join(m.cons_names))
        parts.append("croles=%s" % ",".join(roles_for(m.cons_names, m.cons_roles)))
        parts.append("prim_state=%s" % ",".join(m.prim_state))
        parts.append("proles=%s" % ",".join(roles_for(m.prim_state, m.prim_roles)))
        parts.append("prim=%s" % ";".join("%s=%r" % (k, m.prim_defs[k]) for k in m.prim_defs))
        for d in ("x", "y"):
            parts.append("flux_%s=%s" % (d, ";".join(repr(e) for e in m._flux.get(d, []))))
            parts.append("eig_%s=%s" % (d, ";".join(repr(e) for e in m._eig.get(d, []))))
        parts.append("source=%s" % (";".join(repr(e) for e in m._source) if m._source else ""))
        parts.append("cons_from=%s" % (";".join(repr(e) for e in m.cons_from) if m.cons_from else ""))
        parts.append("elliptic=%s" % (repr(m._elliptic) if m._elliptic is not None else ""))
        parts.append("stab_speed=%s" % (repr(m._stab_speed) if m._stab_speed is not None else ""))
        parts.append("stab_dt=%s" % (repr(m._stab_dt) if m._stab_dt is not None else ""))
        parts.append("src_freq=%s" % (repr(m._src_freq) if m._src_freq is not None else ""))
        parts.append("src_jac=%s" % (";".join(repr(e) for row in m._src_jac for e in row)
                                     if m._src_jac is not None else ""))
        parts.append("hllc=%d" % (1 if m._hllc else 0))
        parts.append("roe=%d" % (1 if getattr(m, "_roe", False) else 0))
        # roe_dissipation FOURNIE : ajoute au hash UNIQUEMENT si presente (sans appel, hash inchange
        # -> bit-identite de la cle de cache des modeles existants preservee).
        if getattr(m, "_roe_rows", None) is not None:
            parts.append("roe_rows=%s" % ";".join(repr(e) for k in ("x", "y")
                                                  for e in m._roe_rows[k]))
        # vitesses signees EXPLICITES (set_wave_speeds) : meme politique conditionnelle (sans appel,
        # hash strictement identique a l'historique -> cache .so des modeles existants preserve).
        if getattr(m, "_wave_speeds", None) is not None:
            parts.append("wave_speeds=%s" % ";".join(repr(e) for k in ("x", "y")
                                                     for e in m._wave_speeds[k]))
        if getattr(m, "_ws_jacobian", None) is not None:
            ws = m._ws_jacobian
            parts.append("ws_jac=%s|%s|%s" % (
                ws["eig"],
                "//".join(";".join(",".join(str(i) for i in b) for b in ws["blocks"][k])
                          for k in ("x", "y")),
                ";".join(repr(e) for k in ("x", "y") for row in ws["rows"][k] for e in row)
                if ws["rows"] is not None else ""))
        
        parts.append("n_aux=%d" % aux_total_n_aux(m.aux_names, m.aux_extra_names))
        # Champs aux NOMMES (aux_field, ADC-70) : leur ORDRE fixe l'indice (AUX_NAMED_BASE + k) -> ils
        # entrent dans le hash (deux modeles ne differant que par un nom/ordre d'aux nomme sont
        # distincts). N'ajoute la cle QUE si des champs nommes existent : un modele sans aux_field garde
        # ainsi un hash STRICTEMENT identique a l'historique (cache .so + tracabilite preserves).
        if m.aux_extra_names:
            parts.append("aux_extra=%s" % ",".join(m.aux_extra_names))
        parts.append("gamma=%r" % m.gamma)
        # Les params entrent dans le hash par (nom, valeur de DECLARATION, kind). La valeur d'un param
        # RUNTIME (P7-b) y figure car elle SEEDE le defaut du membre RuntimeParams genere (donc deux .so
        # a defaut different sont distincts) ; le "sans recompilation" de P7-b porte sur set_block_params
        # a l'EXECUTION, pas sur un nouvel appel compile() a valeur de declaration changee.
        params = params or {}
        parts.append("params=%s" % ";".join("%s=%r:%s" % (k, params[k].value, params[k].kind)
                                             for k in sorted(params)))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def _check_require_metadata(self, require_metadata, backend):
        """Garde-fous require_metadata (pur-Python, deterministes sur le modele + backend). Factorises
        pour etre appeles AVANT le cache (cote HyperbolicModel ET Model) : un cache HIT ne doit jamais
        masquer une exigence de metadonnees. Sans require_metadata, no-op."""
        if not require_metadata:
            return
        # backend "prototype" (add_dynamic_block, dispatch VIRTUEL, Rusanov hote ordre 1) : PAS un chemin
        # de production device-clean -> demander les metadonnees dessus est incoherent (erreur claire).
        if backend == "prototype":
            raise ValueError(
                "compile : backend 'prototype' (JIT, dispatch virtuel hote) incompatible avec "
                "require_metadata=True ; utiliser backend='aot' ou 'production' pour le chemin "
                "device-clean a metadonnees garanties")
        missing = []
        roles = roles_for(self.cons_names, self.cons_roles)
        if all(r == "Custom" for r in roles):
            missing.append("roles physiques (conservative_vars(..., roles=[...]) ou noms canoniques)")
        if self.gamma is None:
            missing.append("gamma (set_gamma(...))")
        if missing:
            raise ValueError(
                "compile(require_metadata=True) : le modele '%s' ne fournit pas %s ; le .so "
                "retomberait sur le fallback du System (roles 'custom' / gamma 1.4)"
                % (self.name, " ni ".join(missing)))

    def compile(self, so_path=None, include=None, backend="auto", name=None, cxx=None, std=None,
                require_metadata=False, target="system"):
        """Facade de compilation par INTENTION : compile le modele en une .so via le moteur designe
        par @p backend et renvoie son chemin. Wrappe les moteurs existants (compile_so / compile_aot /
        compile_native) SANS changer la numerique ; preserve de bout en bout noms, VariableRole, gamma,
        n_aux, B_z et T_e (les memes briques + metadonnees ABI que compile_or_jit).

        ERGONOMIE (ne change pas la numerique) :
          - @p include None -> auto-detecte (adc_include() : $ADC_INCLUDE, paquet adc installe, depot
            voisin) ; passer include= reste possible (retro-compat) ;
          - @p so_path None -> compile dans un cache hors source (adc_cache_dir()), avec un nom de
            fichier keye sur model_hash + abi_key (+ backend/target/name). Sur un cache HIT (.so deja
            presente pour cette cle), AUCUNE recompilation : la .so en cache est reutilisee telle quelle.
            Sur un cache MISS (modele/parametre/toolchain change -> cle differente), recompilation puis
            stockage. Passer so_path= force ce chemin et compile toujours (retro-compat stricte).

        @p backend :
          "prototype"  -> JIT (compile_so) : iteration rapide, dispatch virtuel hote (Rusanov ordre 1),
                          a brancher cote System via add_dynamic_block ;
          "aot"        -> AOT (compile_aot) : chemin de production host-marshale, numerique identique au
                          bloc natif, a brancher via add_compiled_block ;
          "production" -> NATIF (compile_native) : loader .so inline add_compiled_model<ProdModel>, bloc
                          natif zero-copie (parite stricte add_block / add_compiled_model<>), a brancher
                          via add_native_block (cle d'ABI verifiee). Chemin device-clean prepare.

        @p target : "system" (defaut) | "amr_system". Seul le backend "production" cible AmrSystem
        (System.add_native_block vs AmrSystem.add_native_block) ; un target="amr_system" sur les autres
        backends est rejete (pas de chemin .so AMR hors natif, cf. compile_or_jit).

        @p std : standard C++. Defaut None -> norme DU LOADER pour "production" (loader_cxx_std :
        c++20 sous Kokkos car CUDA 12.x n'a pas -std=c++23, c++23 sinon ; le loader natif partage l'ABI
        du module, un std different changerait __cplusplus donc la cle d'ABI -> rejet explicite par
        add_native_block), "c++20" pour les autres (inchange).

        @p require_metadata (defaut False) : si True, exige que le .so transporte des roles physiques
        utiles ET un gamma explicite (set_gamma), faute de quoi le System retomberait sur le fallback
        (roles 'custom' / gamma 1.4) -- regression silencieuse des couplages inter-especes. Sert a un
        pipeline de production qui veut une erreur EXPLICITE plutot qu'un fallback muet.

        Leve ValueError sur un backend inconnu ou une feature incompatible avec le backend demande
        (plutot qu'un echec obscur a l'execution). Renvoie so_path.

        Pour connaitre l'adder System a employer : voir adder_for(backend)."""
        import os
        import shutil
        # DEFAUT 'auto' (ADC-63) : production si la parite toolchain avec le module est etablie,
        # aot sinon (defaut historique). Un backend explicite court-circuite (inchange).
        if backend == "auto":
            backend, _auto_reason = resolve_auto_backend(include)
        if backend not in self._BACKENDS:
            raise ValueError("compile : backend %r inconnu (attendus %s + 'auto')"
                             % (backend, sorted(self._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile : target 'system' | 'amr_system' (recu %r)" % (target,))
        mode, adder = self._BACKENDS[backend]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile : target='amr_system' n'existe que pour backend='production' "
                             "(chemin natif AMR) ; recu backend=%r" % (backend,))
        if std is None:  # defaut par backend : le natif partage l'ABI du module (c++20 sous Kokkos,
            # c++23 sinon -- derive du loader, cf. loader_cxx_std), les autres restent en c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        if include is None:  # ergonomie : auto-detection du dossier d'en-tetes adc
            include = adc_include()

        # Garde-fous metadonnees (avant tout cache : ils ne dependent que du modele + backend, et un
        # cache HIT ne doit pas les masquer).
        self._check_require_metadata(require_metadata, backend)

        # CACHE hors source quand so_path est omis : nom de fichier keye sur model_hash + abi_key
        # (+ backend/target/name). Cache HIT (.so deja presente pour cette cle) -> reutilisation sans
        # recompilation. Cache MISS -> compilation dans le chemin keye (donc stockee pour la prochaine
        # fois). so_path explicite -> chemin force, toujours recompile (retro-compat stricte).
        if so_path is None:
            # Les backends qui compilent les en-tetes adc (production NATIF et aot) suivent Kokkos reel
            # (compilateur + feature-key kokkos dans la cle de cache) : sous Kokkos-only, leur .so est
            # toujours compile AVEC Kokkos (cf. compile_aot / compile_native), la cle doit le refleter.
            kokkos_like = backend in ("production", "aot")
            eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
            abi_key = _abi_key_python(include, eff_cxx, std)
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            so_path = _cache_so_path(self._model_hash(), abi_key, cache_backend, target, name)
            if os.path.exists(so_path):
                return so_path  # cache HIT : .so deja compilee pour cette cle, on la reutilise telle quelle

        return self.compile_or_jit(so_path, include, mode=mode, name=name, cxx=cxx, std=std,
                                  target=target)

    @classmethod
    def adder_for(cls, backend):
        """Nom de la methode System a employer pour brancher la .so produite par compile(backend=...) :
        'add_dynamic_block' (prototype/JIT), 'add_compiled_block' (aot) ou 'add_native_block'
        (production/natif). Couple le backend de compilation a son adder pour eviter une frontiere
        ABI incoherente. ValueError si inconnu."""
        if backend not in cls._BACKENDS:
            raise ValueError("adder_for : backend %r inconnu (attendus %s)"
                             % (backend, sorted(cls._BACKENDS)))
        return cls._BACKENDS[backend][1]

    def emit_cpp_elliptic(self, name=None, namespace="adc_generated", cse=True):
        """Genere une BRIQUE de SECOND MEMBRE elliptique composable depuis self._elliptic.

        Le struct produit expose rhs(U) -> Real (densite de charge, fond, gravite...), meme forme que
        les briques manuelles (ChargeDensity, BackgroundDensity dans adc/model/bricks.hpp) : il entre
        comme parametre Elliptic d'un CompositeModel. Constantes inlinees, cons/primitives -> locals,
        cse=True factorise les sous-expressions communes. ValueError si set_elliptic_rhs(...) absent."""
        if self._elliptic is None:
            raise ValueError("emit_cpp_elliptic : appeler set_elliptic_rhs(...) d'abord")
        nm = name or (self.name.capitalize() + "Elliptic")
        rt_member = self._runtime_params_member()  # P7-b : indices runtime AVANT tout to_cpp()
        out = [
            "#include <cmath>",  # autosuffisant pour std::sqrt / std::pow
            "// brique de SECOND MEMBRE elliptique generee depuis '%s' (adc.dsl.emit_cpp_elliptic)."
            % self.name,
            "// rhs(U) -> Real : second membre f(U) de l'operateur elliptique (p.ex. densite de charge).",
        ]
        if rt_member:  # en-tete RuntimeParams uniquement si une formule lit un param runtime
            out.append("#include <adc/runtime/runtime_params.hpp>")
        out += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
        ]
        if rt_member:  # membre adc::RuntimeParams params{count, {defauts}} (P7-b)
            out.append(rt_member.rstrip("\n"))
        out += [
            "  template <class State>",
            "  ADC_HD adc::Real rhs(const State& U) const {",
        ]
        out += ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]
        out += ["    const adc::Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]
        tl, cpps = self._codegen_exprs([self._elliptic], cse)
        out += tl
        out += ["    return %s;" % cpps[0], "  }", "};", "}  // namespace %s" % namespace]
        return "\n".join(out) + "\n"


# === Phase A : facade utilisateur pure-Python =================================
# La surface STABLE que l'utilisateur ecrit (dsl.Model / Param / CompiledModel). Pur sucre :
# aucune numerique nouvelle, aucun changement de moteur. dsl.Model COMPOSE un HyperbolicModel prive
# (_m) et delegue chaque appel a une methode existante ; Param est une constante NOMMEE qui s'inline
# au codegen ; CompiledModel empaquette le .so + les metadonnees deja connues cote Python (pas de
# relecture du .so). cf. docs/DSL_MODEL_DESIGN.md (Phase A).

# Caracteristiques HONNETES par backend (cf. DSL_MODEL_DESIGN.md section 5). Sert au diagnostic et
# aux garde-fous device/MPI/AMR (verifies au branchement/execution, pas figes a la compilation).
_BACKEND_CAPS = {
    # backend : (cpu, mpi, amr, gpu)  -- True/False selon ce que le chemin SUPPORTE aujourd'hui
    "prototype": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    "aot": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    # production = chemin NATIF (add_native_block, #85) : meme moteur que add_block, donc MPI-capable
    # par construction (halos fill_boundary). amr=True : le loader natif a desormais un pendant AMR
    # (m.compile(backend='production', target='amr_system') -> AmrSystem.add_native_block, DSL Phase D)
    # qui inline add_compiled_model(AmrSystem&) -> MEME hierarchie AMR que AmrSystem.add_block (reflux,
    # regrid). gpu=False par PRUDENCE : le chemin natif est device-clean en C++ (GH200) mais la
    # validation end-to-end depuis Python (add_native_block sur device) est une PR dediee (DSL sect. 5).
    "production": {"cpu": True, "mpi": True, "amr": True, "gpu": False},
}


class Param:
    """Parametre NOMME d'un modele DSL, utilisable comme une Expr dans les formules.

    Mode (a), constante figee a la compilation : `kind="const"` (defaut). Le codegen INLINE toute
    constante (Const.to_cpp -> repr(value)), donc le param s'inline en Const(value) au codegen (valeur
    ecrite EN DUR dans le .so) tout en gardant son IDENTITE (name/value/kind) pour l'introspection
    (m.params), les diagnostics et la reproductibilite. INCHANGE par P7-b : bit-identique a l'historique.

    Mode (b), parametre RUNTIME (modifiable SANS recompiler) : `kind="runtime"` (P7-b). Au codegen, le
    param emet `params.get(<indice>)` (lecture d'un membre adc::RuntimeParams de la brique) au lieu
    d'une constante ; sa valeur est transportee au runtime par l'ABI du .so AOT et peut etre CHANGEE
    (block.set_param / System.set_block_params) sans recompiler. La valeur passee a la declaration sert
    de DEFAUT (sans appel set, le bloc se comporte comme avec un param const de cette valeur).
    SUPPORTE par le backend "aot" (add_compiled_block). Les backends "prototype" (JIT) et "production"
    (natif) compilent un param runtime comme sa valeur de declaration (figee) : un set_param y est sans
    effet, l'API le signale (cf. CompiledModel.runtime_param_names / System.set_block_params).

    Param N'HERITE PAS d'Expr (voir NB ci-dessous) : il EXPOSE les memes hooks d'arbre
    (`eval`/`to_cpp`/`deps`) et les operateurs en DELEGANT a un NOEUD interne (Const pour 'const',
    RuntimeParamRef pour 'runtime'), donc `g * (E - ...)` construit directement l'arbre attendu. La
    valeur n'est pas une variable d'environnement -> aucune dependance a verifier dans check()."""

    # NB : Param N'HERITE PAS d'Expr pour eviter d'embarquer son etat (name/kind) dans la cle
    # structurelle de CSE ; il EXPOSE plutot les hooks d'arbre en deleguant a un noeud interne.
    def __init__(self, name, value, kind="const"):
        if kind not in ("const", "runtime"):
            raise ValueError("Param : kind 'const' | 'runtime' (recu %r)" % (kind,))
        self.name = name
        self.value = float(value)
        self.kind = kind
        if kind == "runtime":
            # Noeud RUNTIME PARTAGE : toutes les apparitions du param dans les formules pointent ce meme
            # objet, donc fixer son .index a la compilation (Model._assign_runtime_indices) suffit a
            # router toutes ses lectures vers params.get(<indice>). index=-1 tant que non assigne.
            self._node = RuntimeParamRef(self.name, self.value)
        else:
            self._node = Const(self.value)  # s'inline au codegen : valeur ecrite EN DUR dans le .so

    # --- hooks d'arbre (delegues au noeud interne) : Param utilisable comme une Expr ---
    def eval(self, env): return self._node.eval(env)
    def to_cpp(self): return self._node.to_cpp()
    def deps(self): return set()  # ni const ni runtime n'ont de dependance (rien a verifier dans check())

    # --- operateurs : Param se combine comme une Expr (promotion via _wrap du noeud interne) ---
    def __add__(self, o): return Add(self._node, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self._node)
    def __sub__(self, o): return Sub(self._node, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self._node)
    def __mul__(self, o): return Mul(self._node, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self._node)
    def __truediv__(self, o): return Div(self._node, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self._node)
    def __neg__(self): return Neg(self._node)
    def __pos__(self): return self._node  # +param = identite (Expr), pour +k*ne*ng
    def __pow__(self, o): return Pow(self._node, _wrap(o))

    def __float__(self): return self.value
    def __repr__(self): return "Param(%r, %r, kind=%r)" % (self.name, self.value, self.kind)


def RuntimeParam(name, value):
    """Sucre : un parametre RUNTIME (modifiable sans recompiler). Equivaut a Param(name, value,
    kind='runtime'). cf. Param mode (b) et include/adc/runtime/runtime_params.hpp (P7-b)."""
    return Param(name, value, kind="runtime")


class CompiledModel:
    """Resultat de `m.compile(...)` : empaquette le `.so` produit + TOUT ce qu'il faut pour le
    brancher correctement (dispatch adder, diagnostic ABI, reproductibilite). Remplace le couple
    historique (str so_path, adder_for(backend)) par un objet unique.

    Les metadonnees ne sont PAS relues du `.so` : Python detient deja noms/roles/gamma/n_aux/params
    (le HyperbolicModel les porte) ; CompiledModel les expose juste pour le dispatch (add_equation)
    et les diagnostics. cf. DSL_MODEL_DESIGN.md section 3."""

    def __init__(self, so_path, backend, adder, cons_names, cons_roles, prim_names, n_vars,
                 gamma, n_aux, params, caps, abi_key, model_hash, cxx, std, target="system",
                 hllc=False, roe=False, aux_extra_names=None, wave_speeds=False):
        self.has_hllc = bool(hllc)   # capability HLLC emise (enable_hllc) : hllc dispo hors Euler 4-var
        self.has_roe = bool(roe)     # hook ROE emis (enable_roe roles OU m.roe_dissipation fournie) : roe dispo hors Euler 4-var
        self.has_wave_speeds = bool(wave_speeds)  # wave_speeds emis (paire explicite OU 'p') : hll dispo
        self.so_path = so_path
        self.backend = backend       # "prototype" | "aot" | "production"
        self.target = target         # "system" | "amr_system" : facade visee (loader natif AMR si amr_system)
        self.adder = adder           # nom de methode (Amr)System : add_dynamic_block / add_compiled_block / add_native_block
        self.cons_names = list(cons_names)
        self.cons_roles = list(cons_roles)
        self.prim_names = list(prim_names)
        self.n_vars = int(n_vars)
        self.gamma = gamma           # None = defaut historique 1.4 cote System
        self.n_aux = int(n_aux)
        # Noms des champs aux NOMMES (aux_field, ADC-70), ORDONNES : indice de la composante = position
        # AUX_NAMED_BASE + k. La facade System.add_equation en construit la table nom -> composante par
        # bloc, consommee par System.set_aux_field / aux_field. Vide pour un modele sans champ nomme.
        self.aux_extra_names = list(aux_extra_names) if aux_extra_names else []
        self.params = dict(params)   # {name: Param}
        self.caps = dict(caps)       # {cpu/mpi/amr/gpu: bool}
        self.abi_key = abi_key       # cle ABI miroir d'adc_header_signature + compilateur/std
        self.model_hash = model_hash  # hash stable formules+roles+n_aux+params
        self.cxx = cxx
        self.std = std

    @property
    def runtime_param_names(self):
        """Noms des parametres RUNTIME du modele (kind='runtime'), TRIES : c'est l'ORDRE des indices
        cote C++ (RuntimeParams) ET l'ordre attendu par System.set_block_params(name, values) (P7-b).
        Vide si le modele n'a que des params const."""
        return sorted(k for k, p in self.params.items() if getattr(p, "kind", "const") == "runtime")

    def runtime_param_values(self):
        """Valeurs de DECLARATION des params runtime, paralleles a runtime_param_names (defaut tant
        qu'aucun set_block_params n'a ete appele)."""
        return [self.params[k].value for k in self.runtime_param_names]

    def check_runtime(self, n=16, state=None, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """Re-verification RUNTIME d'un CompiledModel SEUL (solde audit, GENERICITY pt 9) : sans le
        dsl.Model d'origine, les FORMULES ne sont plus re-verifiables (check_model symbolique), mais
        le .so, lui, l'est -- on l'installe dans un System EPHEMERE (n x n periodique, Poisson
        neutre, minmod+rusanov) et on delegue a System.check_model (etat fini, residu -div F + S
        fini, positivite par roles, round-trip des conversions DU MODELE).

        @p state : dict {nom de variable conservative: ndarray (n, n)} pour controler l'etat teste.
        None -> etat de FUMEE par ROLES (Density = 1 + bosse gaussienne, Momentum* = 0,
        Energy = 2.5, autres composantes = 0.5) -- suffisant pour exercer flux/source/conversions ;
        fournir state= pour un regime physique precis. @return le dict de System.check_model."""
        import numpy as np
        if getattr(self, "target", "system") != "system":
            raise ValueError(
                "CompiledModel.check_runtime : seul target='system' est re-verifiable dans un "
                "System ephemere ; un loader target='amr_system' se controle installe dans son "
                "AmrSystem (invariants des tests AMR), pas isolement.")
        from . import System, FiniteVolume, Explicit
        sim = System(n=int(n), L=1.0, periodic=True)
        sim.set_poisson()
        sim.add_equation("check", model=self,
                         spatial=FiniteVolume(limiter="minmod", riemann="rusanov"),
                         time=Explicit())
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        bump = 1.0 + 0.3 * np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        comps = []
        for name, role in zip(self.cons_names, self.cons_roles):
            if state is not None and name in state:
                comps.append(np.asarray(state[name], dtype=float).reshape(n, n))
            elif role == "Density":
                comps.append(bump)
            elif role in ("MomentumX", "MomentumY"):
                comps.append(np.zeros((n, n)))
            elif role == "Energy":
                comps.append(2.5 + 0.0 * bump)
            else:
                comps.append(0.5 + 0.0 * bump)
        sim._s.set_state("check", np.stack(comps).ravel())
        return sim.check_model("check", raise_on_error=raise_on_error, rtol=rtol, atol=atol)

    def __repr__(self):
        return ("CompiledModel(backend=%r, target=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "adder=%r, runtime_params=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.target, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.adder, self.runtime_param_names, self.abi_key or "", self.model_hash or ""))


def _abi_key_python(include, cxx, std):
    """Cle d'ABI cote Python, MIROIR de adc::detail::abi_key_string (compilateur + standard +
    signature des en-tetes). Rend la verification + le diagnostic disponibles cote Python AVANT le
    chargement du .so (le chemin natif compare la sienne cote C++). Forme stable et lisible :
    "<sig en-tetes>|<cxx>|<std>". include absent -> signature vide (diagnostic degrade, pas d'UB)."""
    import os
    sig = adc_header_signature(include) if include and os.path.isdir(include) else ""
    return "%s|%s|%s" % (sig, cxx or "", std or "")


class Model:
    """Facade STABLE de modele DSL (Phase A). COMPOSE un HyperbolicModel prive (_m, composition et
    NON heritage) et delegue chaque appel a une methode existante : aucune numerique nouvelle.

        m = adc.dsl.Model("euler")
        rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
        g = m.param("gamma", 1.4)                 # constante NOMMEE, inlinee au codegen
        u = m.primitive("u", rhou / rho)
        p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + ...)))
        m.flux(x=[...], y=[...])                   # DECLARATEUR symbolique du flux physique
        m.eval_flux(U, aux, dir)                   # EVALUATEUR numpy (debug), nom DISTINCT
        m.primitive_vars(rho=rho, u=u, v=v, p=p)   # layout Prim ordonne (ordre des kwargs)
        compiled = m.compile(so_path, include, backend="aot")  # -> CompiledModel

    cf. docs/DSL_MODEL_DESIGN.md sections 1-3."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)
        self.params = {}   # name -> Param (introspection / reproductibilite)

    @property
    def name(self): return self._m.name

    # --- declaration des variables (delegation directe a HyperbolicModel) ---
    def conservative_vars(self, *names, roles=None):
        """Declare les variables conservatives. @p roles : meme convention que HyperbolicModel."""
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        """Definit une primitive par sa formule (en fonction des cons / primitives precedentes)."""
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None, **named):
        """Declare les primitives ET le layout ORDONNE de Prim. Deux formes :

        - KWARGS (style cible) : `primitive_vars(rho=expr, u=expr, v=expr, p=expr)` : chaque kwarg
          DEFINIT une primitive (m.primitive(name, expr)) ET fixe le layout de Prim dans l'ordre
          d'insertion des kwargs (Python 3.7+ : ordre garanti). @p roles (liste) optionnel.
        - POSITIONNELLE : `primitive_vars(rho, u, v, p, roles=...)` : noms/Var deja definis, fixe
          juste le layout (delegue a set_primitive_state, comme HyperbolicModel).

        Les deux formes sont exclusives (melanger kwargs nommes et positionnels leve)."""
        if named and vars:
            raise ValueError("primitive_vars : melanger forme positionnelle et kwargs nommes "
                             "(choisir l'une ; les kwargs definissent ET ordonnent les primitives)")
        if named:
            # kwargs : definir chaque primitive, puis fixer le layout dans l'ordre d'insertion.
            # Une primitive n'est PAS (re)definie si le kwarg est la Var de MEME nom -- sinon le codegen
            # emettrait `const Real x = x;` (auto-init -> NaN). Deux cas de self-reference, tous deux
            # laisses REJOINDRE le layout sans redefinition (style cible primitive_vars(rho=rho, u=u, ...)) :
            #  - nom DEJA CONSERVATIF (ex. rho=rho : la densite, primitive == conservative) ;
            #  - Var PRIMITIVE DEJA DEFINIE de meme nom (ex. u=u quand u vient de m.primitive('u', ...)).
            # Sinon (kwarg = expression, ex. p=cs2*rho ou u=mx/rho) la primitive est definie normalement.
            ordered = list(named.keys())
            for nm in ordered:
                val = named[nm]
                self_ref = isinstance(val, Var) and getattr(val, "name", None) == nm
                if nm in self._m.cons_names or self_ref:
                    continue
                self._m.primitive(nm, val)
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # forme positionnelle : fixe le layout a partir de noms/Var deja definis.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name):
        """Champ auxiliaire CANONIQUE (doit etre une clef de AUX_CANONICAL : phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

    def aux_field(self, name):
        """Champ auxiliaire NOMME (ADC-70 phase 1) fourni par bloc via System.set_aux_field(bloc, name,
        array). name est ARBITRAIRE (identifiant) ; le k-ieme appel reserve la composante du canal aux
        AUX_NAMED_BASE + k (lue en C++ via aux.extra_field(k)). Au plus AUX_NAMED_MAX par modele.
        Renvoie une Var utilisable dans flux / source / eigenvalues. Delegue a HyperbolicModel.aux_field."""
        return self._m.aux_field(name)

    def conservative_from(self, exprs):
        """Inverse prim -> cons (le DSL ne sait pas inverser symboliquement)."""
        self._m.set_conservative_from(exprs)

    # --- flux : DECLARATEUR symbolique vs EVALUATEUR numpy (noms DISTINCTS, decision tranchee) ---
    def flux(self, x, y):
        """DECLARATEUR symbolique du flux physique (delegue a set_flux). x/y : listes d'Expr, une
        par composante conservative. NE PAS confondre avec l'evaluateur numpy eval_flux."""
        self._m.set_flux(x, y)

    def eval_flux(self, U, aux, dir):
        """EVALUATEUR numpy du flux physique (debug / proto hote ; delegue a HyperbolicModel.flux).
        U : numpy (n_vars, ...) ; aux : dict nom -> tableau ; dir : 0=x, 1=y."""
        return self._m.flux(U, aux, dir)

    def eigenvalues(self, x, y):
        """Valeurs propres (vitesses caracteristiques) par direction (delegue a set_eigenvalues)."""
        self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Vitesses d'onde SIGNEES explicites par direction : x = (smin_x, smax_x), y = (smin_y,
        smax_y). Emet ``wave_speeds(U, aux, dir, smin, smax)`` sur la brique SANS exiger de
        primitive 'p' : riemann='hll' devient disponible pour un modele sans pression (systeme de
        moments, isotherme...). Prioritaire sur le chemin historique (eigenvalues + 'p') ; si
        eigenvalues n'est pas declare, max_wave_speed (Rusanov / CFL) derive de max(|smin|, |smax|).
        Delegue a set_wave_speeds ; cf. HyperbolicModel.set_wave_speeds."""
        self._m.set_wave_speeds(x, y)

    def wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """Vitesses d'onde signees EXACTES par valeurs propres du jacobien de flux (delegue a
        set_wave_speeds_from_jacobian, voir son contrat complet) : x/y = dF/dU en Expr (None =
        AUTODIFF du flux declare via flux_jacobian) ; eig = 'numeric' | 'fd' (differences finies
        du flux compile, debug) ; blocks = listes d'indices des sous-blocs diagonaux (None = bloc
        plein, seul mode inconditionnellement correct -- les blocs AFFIRMENT une structure
        bloc-triangulaire)."""
        self._m.set_wave_speeds_from_jacobian(x=x, y=y, eig=eig, blocks=blocks)

    def eval_wave_speeds(self, U, aux, dir):
        """EVALUATEUR numpy des vitesses signees (smin, smax) emises (delegue a
        HyperbolicModel.wave_speeds_value) : paire explicite, jacobien (eig numpy par blocs) ou
        min/max des eigenvalues."""
        return self._m.wave_speeds_value(U, aux, dir)

    def stability_speed(self, expr):
        """Vitesse de STABILITE lambda* pilotant la CFL du bloc (OPTIONNEL ; delegue a
        HyperbolicModel.stability_speed). Fallback sans appel : max(abs(eigenvalues)), strictement
        l'historique. Compilee comme flux/source (production GPU/MPI, pas de callback par cellule)."""
        self._m.stability_speed(expr)

    def stability_dt(self, expr_dt):
        """Pas ADMISSIBLE direct dt(U, aux) borne locale du pas (OPTIONNEL ; delegue a
        HyperbolicModel.stability_dt). Le cfl n'est pas applique a cette borne. Fallback sans appel :
        aucune borne supplementaire (politique de pas historique)."""
        self._m.stability_dt(expr_dt)

    def source(self, s):
        """Terme source S(U, aux), une expression par composante (optionnel ; delegue a set_source)."""
        self._m.set_source(s)

    def source_frequency(self, expr_mu):
        """Frequence locale mu(U, aux) [1/s] de la source -- la borne de pas 'source' du meeting
        (dt <= cfl*substeps/(stride*max mu), sans pas d'espace). Emise sur la brique de SOURCE
        generee (frequency(U, aux)), forwarde par CompositeModel, agregee par System/AmrSystem
        step_cfl. EXIGE m.source([...]). Delegue a HyperbolicModel.source_frequency."""
        self._m.source_frequency(expr_mu)

    def source_jacobian(self, rows):
        """Jacobien ANALYTIQUE dS/dU de la source (rows[r][c] = dS_r/dU_c, matrice n_vars x n_vars
        d'expressions) : le Newton implicite (IMEX/SourceImplicitBE) l'utilise a la place des
        differences finies. EXIGE m.source. Delegue a HyperbolicModel.source_jacobian."""
        self._m.source_jacobian(rows)

    def implicit_source(self, jacobian=None):
        """Declaration GROUPEE de l'implicite local (audit vague 3, sucre) : le RESIDU est deja
        implique par m.source (backward-Euler : F = W - U^n - dt*S(W)) ; @p jacobian (optionnel) =
        matrice dS/dU analytique (cf. source_jacobian). Sans jacobian : differences finies."""
        if jacobian is not None:
            self._m.source_jacobian(jacobian)

    def enable_hllc(self):
        """Emet la capability HLLC (contact_speed + hllc_star_state generes depuis les ROLES +
        primitive 'p') : riemann='hllc' devient disponible pour ce modele MEME hors Euler 4
        variables. Delegue a HyperbolicModel.enable_hllc."""
        self._m.enable_hllc()

    def enable_roe(self):
        """Emet la capability ROE (roe_dissipation = |A_roe| dU generee depuis les ROLES +
        primitive 'p') : riemann='roe' devient disponible pour ce modele MEME hors Euler 4
        variables (sans Energy : c = sqrt(p/rho) moyennee a la Roe ; composantes hors roles
        fluides = scalaires passifs sur l'onde entropique). Delegue a HyperbolicModel.enable_roe."""
        self._m.enable_roe()

    def roe_dissipation(self, x, y):
        """Dissipation de Roe FOURNIE par l'utilisateur (hors roles fluides) : n_vars expressions par
        direction (x=, y=), ecrites en m.left(...)/m.right(...) (ou dsl.left/right) des deux etats,
        emises comme le hook C++ roe_dissipation(UL, AL, UR, AR, dir). Pendant 'fourni' de enable_roe
        (un seul fournisseur : les deux ensemble levent). Le helper m.flux_jacobian assiste l'ecriture.
        Delegue a HyperbolicModel.roe_dissipation (cf. sa doc)."""
        self._m.roe_dissipation(x, y)

    def flux_jacobian(self, dir):
        """Jacobien de flux A = dF_dir/dU (matrice n_vars x n_vars d'Expr, A[i][j]=d(F_i)/d(U_j)),
        auto-derive des flux declares par dsl.diff (primitives developpees). HELPER de construction
        de m.roe_dissipation, n'emet rien. @p dir : 0/'x' ou 1/'y'. Delegue a HyperbolicModel."""
        return self._m.flux_jacobian(dir)

    def left(self, expr):
        """Marque @p expr comme evaluee sur l'etat GAUCHE UL (m.roe_dissipation). Sucre pour dsl.left."""
        return left(expr)

    def right(self, expr):
        """Marque @p expr comme evaluee sur l'etat DROIT UR (m.roe_dissipation). Sucre pour dsl.right."""
        return right(expr)

    def elliptic_rhs(self, e):
        """Contribution au second membre elliptique (couplage Poisson ; delegue a set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def gamma(self, value):
        """Indice adiabatique (EOS), porte par ADC_EXPORT_BLOCK_GAMMA (delegue a set_gamma)."""
        self._m.set_gamma(value)

    def param(self, name, value, kind="const"):
        """Parametre NOMME utilisable dans les formules. Mode (a) (`kind="const"`, defaut) : constante
        figee a la compilation, inlinee au codegen ; stockee dans m.params (introspection /
        reproductibilite). Mode (b) (`kind="runtime"`, P7-b) : SUPPORTE sur le backend "aot" -- le param
        emet `params.get(<indice>)` (membre adc::RuntimeParams) et sa valeur peut etre CHANGEE au runtime
        via System.set_block_params(name, values) SANS recompiler (la valeur de declaration sert de
        defaut) ; cf. CompiledModel.runtime_param_names. Les backends "prototype"/"production" figent un
        param runtime a sa valeur de declaration.

        CAS gamma : si name == "gamma", appelle AUSSI set_gamma(value) pour que la metadonnee ABI
        reste coherente (sinon le System retombe sur 1.4)."""
        p = Param(name, value, kind=kind)  # 'runtime' -> RuntimeParamRef (P7-b), 'const' -> inline
        self.params[name] = p
        if name == "gamma":
            self._m.set_gamma(p.value)
        return p

    def check(self):
        """Verifie les dependances (variables referencees declarees). Leve ValueError sinon."""
        return self._m.check()

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Verification NUMERIQUE generique du modele (flux/source/elliptic finis, valeurs propres
        reelles et finies, coherence wave_speeds/max_wave_speed, encadrement non circulaire du
        spectre par le jacobien dense, round-trip cons<->prim, positivite Density/'p') sur des
        etats echantillons. Delegue a HyperbolicModel.check_model (cf. sa doc). A appeler AVANT
        compile() ; le pendant runtime d'un bloc installe est System.check_model."""
        return self._m.check_model(samples=samples, n_samples=n_samples, seed=seed, aux=aux,
                                   rtol=rtol, atol=atol, raise_on_error=raise_on_error,
                                   jac_rtol=jac_rtol, jac_atol=jac_atol)

    # --- introspection (lecture seule, deleguee au modele backing) ---
    @property
    def cons_names(self): return self._m.cons_names

    @property
    def prim_state(self): return self._m.prim_state

    @property
    def n_vars(self): return self._m.n_vars

    def _model_hash(self):
        """Hash stable du modele : formules (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + params NOMMES (m.params). Sert a identifier/reutiliser un .so deja compile (cle de
        cache) et a tracer le run. Delegue au calcul partage HyperbolicModel._model_hash en lui passant
        les Param de la facade (sinon deux modeles ne differant que par un param auraient le meme hash)."""
        return self._m._model_hash(params=self.params)

    def compile(self, so_path=None, include=None, backend="auto", target="system", name=None,
                cxx=None, std=None, require_metadata=False):
        """Compile le modele en un CompiledModel (Phase A). Delegue la GENERATION + compilation a
        HyperbolicModel.compile (moteurs inchanges : compile_so / compile_aot / compile_native), puis
        empaquette le .so avec les metadonnees deja connues (pas de relecture du .so).

        - ``backend`` : "prototype" | "aot" | "production" (cf. HyperbolicModel.compile).
        - ``target`` : "system" (defaut) | "amr_system" (DSL Phase D). "amr_system" exige
          backend="production" (le loader natif inline add_compiled_model(AmrSystem&), seul chemin
          .so AMR ; cf. compile_or_jit) -> a brancher via AmrSystem.add_equation. Un autre backend
          avec target="amr_system" leve ValueError (pas de chemin AMR hors natif).

        PAS d'argument ``device`` : les capacites GPU/MPI/AMR sont verifiees au branchement
        (add_equation) / a l'execution, pas figees a la compilation (DSL_MODEL_DESIGN.md point 7).

        ERGONOMIE (ne change pas la numerique) :

        - ``include`` None -> auto-detecte (adc_include()) ; passer include= reste possible ;
        - ``so_path`` None -> .so dans un cache hors source (adc_cache_dir()), nom de fichier keye sur
          model_hash (PARAMS COMPRIS) + abi_key (+ backend/target/name). Cache HIT (.so deja presente)
          -> reutilisation sans recompilation ; cache MISS (modele/param/toolchain change) ->
          recompilation + stockage. Passer so_path= force ce chemin et recompile (retro-compat).

        Renvoie un CompiledModel portant so_path, backend, target, adder, noms/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        import shutil
        # DEFAUT 'auto' (ADC-63) : production si parite toolchain etablie, aot sinon. La raison
        # est posee sur le CompiledModel (backend_auto_reason) -- jamais un choix muet.
        auto_reason = None
        if backend == "auto":
            backend, auto_reason = resolve_auto_backend(include)
        if backend not in HyperbolicModel._BACKENDS:
            raise ValueError("compile : backend %r inconnu (attendus %s + 'auto')"
                             % (backend, sorted(HyperbolicModel._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile : target 'system' | 'amr_system' (recu %r)" % (target,))

        m = self._m
        # std effectif : meme defaut par backend que HyperbolicModel.compile. Le natif suit la norme
        # du loader (c++20 sous Kokkos, c++23 sinon, cf. loader_cxx_std) ; les autres restent c++20.
        mode = HyperbolicModel._BACKENDS[backend][0]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile : target='amr_system' n'existe que pour backend='production' "
                             "(chemin natif AMR) ; recu backend=%r" % (backend,))
        eff_std = std if std is not None else (loader_cxx_std() if mode == "native" else "c++20")
        # native ET aot (mode "compile") compilent les en-tetes adc -> Kokkos reel (compilateur +
        # feature-key kokkos) pour que la cle de cache CONCORDE avec le .so produit (cf. compile_aot).
        kokkos_like = mode in ("native", "compile")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if include is None:  # ergonomie : auto-detection du dossier d'en-tetes adc
            include = adc_include()

        # Garde-fous metadonnees AVANT le cache (un HIT ne doit pas les masquer ; cf.
        # HyperbolicModel._check_require_metadata).
        m._check_require_metadata(require_metadata, backend)

        # model_hash PARAMS-COMPRIS (celui porte par le CompiledModel) ET cle d'ABI : tous deux servent
        # aussi de cle de cache, donc on les calcule ici pour les reutiliser (coherence cle/metadonnees).
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, eff_std)

        # CACHE hors source quand so_path est omis : on RESOUT le chemin keye ici (avec le hash
        # params-compris) et on le passe explicitement au moteur -- le cache de HyperbolicModel.compile
        # utiliserait sinon le hash SANS params (la facade Model ajoute les Param). HIT -> on saute la
        # compilation. so_path explicite -> chemin force, toujours recompile (retro-compat stricte).
        cache_hit = False
        if so_path is None:
            # feature-key kokkos dans la cle (cf. compile_native) : un .so SERIE n'est pas reutilise
            # sur un module Kokkos. DOIT matcher la cle du moteur, sinon recompilations a repetition.
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            cache_hit = os.path.exists(so_path)

        if cache_hit:
            out_path = so_path  # .so deja compilee pour cette cle : pas de recompilation
        else:
            # Compilation (moteurs inchanges, garde-fous require_metadata/backend/target de
            # HyperbolicModel.compile : le loader emet adc_install_native_amr pour target="amr_system").
            out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target)

        adder = HyperbolicModel.adder_for(backend)
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm = CompiledModel(
            so_path=out_path, backend=backend, adder=adder, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=abi_key, model_hash=model_hash,
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs))
        # Trace de la politique 'auto' (ADC-63) : None si le backend etait explicite. Diagnostic,
        # jamais un choix muet -- cm.backend dit ce qui a ete construit, ceci dit POURQUOI.
        cm.backend_auto_reason = auto_reason
        return cm


# === Phase B (prototype) : composition HYBRIDE brique native + brique DSL dans UN modele ========
# Jusqu'ici, melanger natif et DSL se faisait au niveau du SYSTEME (un bloc natif add_block + un bloc
# DSL add_equation). On ne pouvait pas melanger une brique native et une brique DSL DANS UN SEUL
# modele. Ce prototype l'autorise, dans les DEUX sens (transport DSL + source/elliptic natifs, ET
# transport natif + source/elliptic DSL).
#
# ARCHITECTURE (option B) : le coeur C++ COMPOSE deja sans effort des types de briques heterogenes --
# adc::CompositeModel<Hyperbolic, Source, Elliptic> accepte n'importe quel type conforme a son slot,
# natif (include/adc/physics/) ou genere par le DSL, et physics/bricks.hpp est deja inclus par tous
# les backends. Le melange est donc genere a la compilation du COMPOSITE final : un seul .so, sur le
# MEME chemin que les modeles DSL complets (numerique inlinee, identique a un bloc natif). Pas de .so
# par brique ni d'ABI virtuelle partielle (ce serait l'option A : dispatch virtuel hote, sans inline
# GPU -- ecartee).
#
# La seule subtilite : les backends ne transportent que le TYPE de modele (default-construit), donc
# les PARAMETRES d'une brique native (qom, q, cs2...) doivent etre CUITS dans le type. On emet pour
# cela un petit struct derive qui fixe les champs publics dans son constructeur (hote) :
#   namespace adc_generated { struct NatSrc : adc::PotentialForce { NatSrc() { qom = adc::Real(-1.0); } }; }
# Il herite EXACTEMENT la numerique native (vrai chemin natif, zero re-derivation) et satisfait le
# contrat du slot. Sans parametre -> simple alias `using`.
#
# Etat PROTOTYPE : backend "aot" (add_compiled_block, .so autosuffisant, chemin de production
# host-marshale). Les backends "production" (natif zero-copie) et "prototype" (JIT) hybrides, la
# propagation fine des roles/gamma/n_aux et la cible amr_system arrivent dans les PRs suivantes.


class NativeBrick:
    """Descripteur d'une brique NATIVE (include/adc/physics/) pour la composition hybride.

    Porte le type C++ de la brique, les PARAMETRES a cuire dans le type (champ public -> valeur) et,
    pour une brique hyperbolique, le layout de variables (noms conservatifs, n_vars, primitives,
    gamma). emit(struct_name) rend le texte C++ a coudre dans le .so composite : un struct derive qui
    fixe les parametres (ou un simple alias `using` si la brique n'a aucun parametre).

    - ``kind`` : 'hyperbolic' | 'source' | 'elliptic' (slot vise).
    - ``fields`` : dict {nom de champ C++ public -> valeur} ; ORDRE preserve (insertion).
    - ``var_names`` / ``n_vars`` / ``prim_names`` / ``gamma`` : metadonnees du layout (slot
      hyperbolique seulement).
    - ``min_vars`` : nombre minimal de variables qu'exige une brique TEMPLATEE (source/elliptic) ;
      p.ex. PotentialForce indexe s[1]/s[2] donc exige >= 3 variables. Verifie par HybridModel.
    - ``n_aux`` : largeur du canal aux que la brique LIT (>= 3 si elle lit B_z/T_e)."""

    def __init__(self, cpp_type, kind, fields=None, var_names=None, n_vars=None, prim_names=None,
                 gamma=None, min_vars=1, n_aux=AUX_BASE_COMPS):
        if kind not in ("hyperbolic", "source", "elliptic"):
            raise ValueError("NativeBrick : kind 'hyperbolic' | 'source' | 'elliptic' (recu %r)" % (kind,))
        self.cpp_type = cpp_type
        self.kind = kind
        self.fields = dict(fields or {})
        self.var_names = list(var_names) if var_names else None
        self.n_vars = n_vars
        self.prim_names = list(prim_names) if prim_names else (list(var_names) if var_names else None)
        self.gamma = gamma
        self.min_vars = min_vars
        self.n_aux = n_aux

    def emit(self, struct_name, namespace="adc_generated"):
        """Texte C++ de la brique cousue dans le .so composite. Sans parametre -> alias `using`
        (zero cout) ; avec parametres -> struct derive qui les fixe dans son constructeur hote
        (les valeurs sont ECRITES EN DUR, comme une constante DSL inlinee)."""
        if not self.fields:
            return "namespace %s { using %s = %s; }\n" % (namespace, struct_name, self.cpp_type)
        sets = " ".join("%s = adc::Real(%s);" % (k, repr(float(v))) for k, v in self.fields.items())
        return ("namespace %s { struct %s : %s { %s() { %s } }; }\n"
                % (namespace, struct_name, self.cpp_type, struct_name, sets))


class CompiledBrick:
    """Resultat de <brique DSL partielle>.compile() : le C++ d'UNE brique (le struct genere) + ses
    metadonnees, pret a etre cousu dans un CompositeModel hybride. La compilation MACHINE se fait au
    niveau du composite (un seul .so) ; cet objet porte la brique deja GENEREE et figee."""

    def __init__(self, kind, struct_src, type_name, n_vars=None, n_aux=AUX_BASE_COMPS,
                 cons_names=None, cons_roles=None, prim_names=None, gamma=None, hash_part="",
                 wave_speeds=True):
        self.kind = kind                 # 'hyperbolic' | 'source' | 'elliptic'
        self.struct_src = struct_src     # texte C++ du struct (namespace adc_generated { struct ... })
        self.type_name = type_name       # type qualifie a placer dans CompositeModel<...>
        self.n_vars = n_vars             # layout (hyperbolique) ou nombre de variables declare (src/ell)
        self.n_aux = n_aux
        self.cons_names = list(cons_names) if cons_names else []
        self.cons_roles = list(cons_roles) if cons_roles else []
        self.prim_names = list(prim_names) if prim_names else []
        self.gamma = gamma
        self.hash_part = hash_part       # tranche de hash stable (formules) pour la cle de cache composite
        # wave_speeds emis par le struct (brique hyperbolique DSL : 'p' OU paire explicite) ; True par
        # defaut = inconnu (brique native) : on laisse le requires-gate C++ trancher (historique).
        self.has_wave_speeds = bool(wave_speeds)

    def __repr__(self):
        return "CompiledBrick(kind=%r, type=%r, n_vars=%r)" % (self.kind, self.type_name, self.n_vars)


class CompiledHyperbolicBrick(CompiledBrick):
    """Brique hyperbolique DSL compilee (vars/flux/eigenvalues/conversions)."""
    def __init__(self, **kw): super().__init__("hyperbolic", **kw)


class CompiledSourceBrick(CompiledBrick):
    """Brique de source DSL compilee (apply(U, aux))."""
    def __init__(self, **kw): super().__init__("source", **kw)


class CompiledEllipticBrick(CompiledBrick):
    """Brique de second membre elliptique DSL compilee (rhs(U))."""
    def __init__(self, **kw): super().__init__("elliptic", **kw)


class HyperbolicBrick:
    """Brique DSL PARTIELLE hyperbolique (variables/flux/valeurs propres/conversions), composable avec
    des briques natives ou DSL pour la source et l'elliptique. Meme surface que dsl.Model mais limitee
    au slot hyperbolique. compile() -> CompiledHyperbolicBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    @property
    def name(self): return self._m.name

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None):
        """Layout ORDONNE de Prim (forme positionnelle, noms/Var deja definis)."""
        self._m.set_primitive_state(*vars, roles=roles)

    def aux(self, name): return self._m.aux(name)
    def flux(self, x, y): self._m.set_flux(x, y)
    def eigenvalues(self, x, y): self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Vitesses d'onde SIGNEES explicites (smin, smax) par direction, SANS exiger 'p' --
        meme contrat que Model.wave_speeds (le struct de brique passe par emit_cpp_brick, qui
        emet wave_speeds depuis la paire ; le CompositeModel hybride la forwarde au gate HLL)."""
        self._m.set_wave_speeds(x, y)

    def conservative_from(self, exprs): self._m.set_conservative_from(exprs)
    def gamma(self, value): self._m.set_gamma(value)
    def check(self): return self._m.check()

    def compile(self):
        """Valide + emet le struct C++ hyperbolique (emit_cpp_brick) -> CompiledHyperbolicBrick."""
        self._m.check()
        struct_name = "Hyp" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_brick(name=struct_name)
        return CompiledHyperbolicBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, cons_names=list(self._m.cons_names),
            cons_roles=roles_for(self._m.cons_names, self._m.cons_roles),
            prim_names=list(self._m.prim_state), gamma=self._m.gamma,
            n_aux=aux_n_aux(self._m.aux_names), hash_part=self._m._model_hash(),
            wave_speeds=("p" in self._m.prim_defs or self._m._wave_speeds is not None))


class SourceBrick:
    """Brique DSL PARTIELLE de source S(U, aux), composable avec un transport et une elliptique
    natifs ou DSL. Declare ses conservatives (le layout doit coincider avec le transport) + ses champs
    aux + la formule de source. compile() -> CompiledSourceBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def aux(self, name): return self._m.aux(name)
    def source(self, s): self._m.set_source(s)

    def compile(self):
        """Valide + emet le struct C++ de source (emit_cpp_source) -> CompiledSourceBrick."""
        if self._m._source is None:
            raise ValueError("SourceBrick.compile : appeler source([...]) d'abord")
        struct_name = "Src" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_source(name=struct_name)
        return CompiledSourceBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=aux_n_aux(self._m.aux_names),
            hash_part=self._m._model_hash())


class EllipticBrick:
    """Brique DSL PARTIELLE de second membre elliptique rhs(U), composable avec un transport et une
    source natifs ou DSL. Declare ses conservatives (layout) + la formule du second membre.
    compile() -> CompiledEllipticBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def elliptic_rhs(self, e): self._m.set_elliptic_rhs(e)

    def compile(self):
        """Valide + emet le struct C++ elliptique (emit_cpp_elliptic) -> CompiledEllipticBrick."""
        if self._m._elliptic is None:
            raise ValueError("EllipticBrick.compile : appeler elliptic_rhs(...) d'abord")
        struct_name = "Ell" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_elliptic(name=struct_name)
        return CompiledEllipticBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=AUX_BASE_COMPS, hash_part=self._m._model_hash())


class HybridModel:
    """Composeur d'un modele HYBRIDE : trois slots (transport, source, elliptic), chacun fourni par une
    brique NATIVE (NativeBrick) ou DSL (CompiledBrick). Assemble un adc::CompositeModel<...> melange et
    le compile en UN .so (prototype : backend 'aot'). Renvoie un CompiledModel branchable via
    System.add_equation (adder add_compiled_block).

    Le slot transport (hyperbolique) FIXE le layout : n_vars, noms conservatifs, primitives, gamma. Une
    brique DSL de source/elliptic doit declarer le MEME n_vars ; une brique native templatee (source/
    elliptic) doit seulement satisfaire son min_vars (p.ex. PotentialForce exige >= 3 variables)."""

    def __init__(self, transport, source, elliptic, name="hybrid"):
        self.name = name
        hyp = self._norm(transport, "hyperbolic", "NatHyp")
        src = self._norm(source, "source", "NatSrc")
        ell = self._norm(elliptic, "elliptic", "NatEll")

        nv = hyp["n_vars"]
        if nv is None:
            raise ValueError("HybridModel : le slot transport doit fixer n_vars (brique hyperbolique)")
        for role, slot in (("source", src), ("elliptic", ell)):
            if slot["provider"] == "dsl":
                if slot["n_vars"] != nv:
                    raise ValueError(
                        "HybridModel : la brique DSL %s declare %d variables mais le transport en a %d ; "
                        "aligner conservative_vars(...)" % (role, slot["n_vars"], nv))
            elif slot["min_vars"] > nv:
                raise ValueError(
                    "HybridModel : la brique native %s exige >= %d variables (transport=%d) ; p.ex. une "
                    "force fluide n'a pas de sens sur un transport scalaire"
                    % (role, slot["min_vars"], nv))

        self.n_vars = nv
        self.cons_names = list(hyp["cons_names"])
        self.cons_roles = list(hyp["cons_roles"])
        self.prim_names = list(hyp["prim_names"])
        self.gamma = hyp["gamma"]
        self.n_aux = max(hyp["n_aux"], src["n_aux"], ell["n_aux"])
        self._has_wave_speeds = bool(hyp.get("wave_speeds", True))
        self._slots = (hyp, src, ell)

    @staticmethod
    def _norm(prov, role, native_struct_name):
        """Normalise un slot (CompiledBrick DSL ou NativeBrick) en dict commun."""
        if isinstance(prov, CompiledBrick):
            if prov.kind != role:
                raise ValueError("HybridModel : brique DSL de type %r placee dans le slot %r"
                                 % (prov.kind, role))
            d = dict(provider="dsl", struct_text=prov.struct_src, type_name=prov.type_name,
                     n_vars=prov.n_vars, min_vars=prov.n_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                d.update(cons_names=prov.cons_names, cons_roles=prov.cons_roles,
                         prim_names=prov.prim_names, gamma=prov.gamma,
                         wave_speeds=getattr(prov, "has_wave_speeds", True))
            return d
        if isinstance(prov, NativeBrick):
            if prov.kind != role:
                raise ValueError("HybridModel : brique native de type %r placee dans le slot %r"
                                 % (prov.kind, role))
            d = dict(provider="native", struct_text=prov.emit(native_struct_name),
                     type_name="adc_generated::" + native_struct_name,
                     n_vars=prov.n_vars, min_vars=prov.min_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                names = prov.var_names or []
                d.update(cons_names=list(names), cons_roles=roles_for(names),
                         prim_names=list(prov.prim_names or names), gamma=prov.gamma,
                         wave_speeds=True)  # natif : inconnu, le requires-gate C++ tranche (historique)
            return d
        raise TypeError("HybridModel : slot %r doit etre une brique native (adc.* / NativeBrick) ou une "
                        "brique DSL compilee (CompiledBrick) ; recu %r" % (role, type(prov).__name__))

    def _emit_aot_source(self):
        """Source C++ du .so composite hybride, derriere l'ABI extern \"C\" de compiled_block_abi.hpp
        (backend aot : meme ABI plate que emit_cpp_aot_source). Les briques (DSL generees ou structs de
        liaison natifs) sont cousues, puis assemblees en adc::CompositeModel<...>."""
        hyp, src, ell = self._slots
        parts = ['#include <adc/runtime/compiled_block_abi.hpp>\n',
                 '#include <adc/physics/bricks.hpp>\n',   # CompositeModel + briques natives
                 '#include <adc/core/variables.hpp>\n']   # ADC_EXPORT_BLOCK_METADATA / _GAMMA
        for slot in self._slots:
            if slot["struct_text"]:
                parts.append(slot["struct_text"])
        parts.append('\nnamespace adc_generated { using AotModel = adc::CompositeModel<%s, %s, %s>; }\n'
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        parts.append('ADC_DEFINE_COMPILED_BLOCK(adc_generated::AotModel)\n')
        parts.append('ADC_EXPORT_BLOCK_METADATA(adc_generated::AotModel)\n')
        if self.gamma is not None:
            parts.append('ADC_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma)
        return "".join(parts)

    def _model_hash(self):
        """Hash stable du composite : provider + type + texte genere de chaque slot (le texte encode
        les formules DSL et les parametres natifs cuits). Sert de cle de cache."""
        import hashlib
        parts = ["hybrid", self.name]
        for slot in self._slots:
            parts.append("%s|%s|%s" % (slot["provider"], slot["type_name"], slot.get("struct_text", "")))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def _bricks_and_composite(self):
        """Texte C++ des briques cousues (DSL generees + structs de liaison natifs) + type composite."""
        hyp, src, ell = self._slots
        bricks = "".join(s["struct_text"] for s in self._slots if s["struct_text"])
        composite = ("adc::CompositeModel<%s, %s, %s>"
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        return bricks, composite

    def _emit_metadata(self, alias):
        """Symboles de metadonnees ABI (noms/roles depuis conservative_vars, gamma optionnel), PARTAGES
        par les backends. @p alias : un alias SANS virgule de niveau superieur (le preprocesseur decoupe
        les arguments de macro sur les virgules)."""
        out = '\nADC_EXPORT_BLOCK_METADATA(%s)\n' % alias
        if self.gamma is not None:
            out += 'ADC_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma
        return out

    def _emit_jit_source(self):
        """Source de la bibliotheque JIT (backend 'prototype') : le composite hybride derriere une
        fabrique extern \"C\" (adc_make_model via adc::ModelAdapter). Dispatch VIRTUEL hote (residu
        Rusanov ordre 1) : iteration rapide, a brancher via System.add_dynamic_block. Pendant hybride
        d'emit_cpp_so_source."""
        bricks, composite = self._bricks_and_composite()
        return ('#include <adc/runtime/dynamic_model.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using JitModel = %s; }\n' % composite
                + 'extern "C" int adc_model_nvars() { return %d; }\n' % self.n_vars
                + 'extern "C" void* adc_make_model() { return new adc::ModelAdapter<adc_generated::JitModel>(); }\n'
                + 'extern "C" void adc_destroy_model(void* p) { delete static_cast<adc::IModel<%d>*>(p); }\n'
                % self.n_vars
                + self._emit_metadata("adc_generated::JitModel"))

    def _emit_native_source(self, target="system"):
        """Source C++ du LOADER NATIF (backend 'production') : le composite hybride en CompositeModel<...>
        derriere une ABI extern \"C\" MINCE. Comme emit_cpp_native_loader, le .so ne porte PAS la
        numerique : il INSTALLE le modele genere comme bloc natif de la facade via add_compiled_model<>,
        qui fabrique les fermetures sur le CONTEXTE REEL de la facade -> meme chemin que add_block,
        ZERO-COPIE (MPI par construction, device-clean). adc_native_abi_key() fige la cle d'ABI a la
        compilation, comparee a abi_key() du module par add_native_block (rejet explicite si
        en-tetes/compilateur/std divergent).

        @p target : 'system' -> adc_install_native (System&, evolve) ; 'amr_system' ->
        adc_install_native_amr (AmrSystem&, sans evolve) inline add_compiled_model(AmrSystem&) : le bloc
        tourne la MEME hierarchie AMR que AmrSystem.add_block (reflux, regrid). Symboles DISTINCTS par
        cible (un loader System n'est pas branchable sur AmrSystem.add_native_block, et inversement)."""
        if target not in ("system", "amr_system"):
            raise ValueError("_emit_native_source : target 'system' | 'amr_system' (recu %r)" % (target,))
        bricks, composite = self._bricks_and_composite()
        head = ('#include <adc/runtime/abi_key.hpp>\n'        # ADC_ABI_KEY_LITERAL (cle figee a la compil)
                '#include <adc/physics/bricks.hpp>\n'         # CompositeModel + briques natives
                '#include <adc/core/variables.hpp>\n'
                '#include <string>\n')
        # Gabarit en-tete de la cible (selectif : ne pas tirer la machinerie AMR dans un loader System).
        head += ('#include <adc/runtime/dsl_block.hpp>\n' if target == "system"
                 else '#include <adc/runtime/amr_dsl_block.hpp>\n')
        # LITTERAL preprocesseur, pas d'appel a abi_key_string() : une inline serait interposee
        # (ELF/RTLD_GLOBAL) vers la copie du module -> cle du module renvoyee -> garde tautologique.
        key = ('#if defined(_WIN32)\n'
               '#define ADC_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define ADC_LOADER_API extern "C"\n'
               '#endif\n'
               'ADC_LOADER_API const char* adc_native_abi_key() {\n'
               '  return ADC_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, limiteur de positivite Zhang-Shu) : argument plat final, marshale
            # jusqu'au make_block du loader via add_compiled_model. Vieille signature = vieux .so =
            # rejete par la cle d'ABI (les en-tetes ont change), jamais un layout d'arguments errone.
            install = ('ADC_LOADER_API void adc_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  adc::System* s = reinterpret_cast<adc::System*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system : surcharge AmrSystem (pas de parametre evolve, AMR mono-bloc)
            install = ('ADC_LOADER_API void adc_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head + bricks
                + '\nnamespace adc_generated { using ProdModel = %s; }\n' % composite
                + key + install + self._emit_metadata("adc_generated::ProdModel"))

    def compile(self, backend="aot", so_path=None, include=None, name=None, cxx=None, std=None,
                target="system"):
        """Compile le composite hybride en un CompiledModel.

        ``backend`` :

        - 'prototype' -> add_dynamic_block : JIT, dispatch VIRTUEL hote (Rusanov ordre 1), iteration
          rapide ; pas de MPI/AMR, pas de flux HLLC/Roe ni recon primitif ;
        - 'aot' -> add_compiled_block : .so autosuffisant (ABI plate, host-marshale), chemin de
          production mono-rang ; sans MPI/AMR ;
        - 'production' -> add_native_block : loader natif zero-copie qui inline add_compiled_model<>, MEME
          chemin qu'add_block (fermetures sur le contexte reel de la facade), MPI par construction.
          Les noms/roles/gamma viennent des metadonnees du .so (pas de names=).

        ``target`` : 'system' (defaut) | 'amr_system'. 'amr_system' EXIGE backend='production' : le loader
        inline add_compiled_model(AmrSystem&) (symbole adc_install_native_amr), seul chemin .so AMR ; a
        brancher via AmrSystem.add_equation. Les autres backends n'ont pas de pendant AMR.

        so_path None -> cache hors source (cle = model_hash + abi_key + backend + target)."""
        import os
        import shutil
        import subprocess
        import sys
        import tempfile
        if backend not in ("prototype", "aot", "production"):
            raise ValueError("HybridModel.compile : backend 'prototype' | 'aot' | 'production' (recu %r)"
                             % (backend,))
        if target not in ("system", "amr_system"):
            raise ValueError("HybridModel.compile : target 'system' | 'amr_system' (recu %r)" % (target,))
        if target == "amr_system" and backend != "production":
            raise ValueError("HybridModel.compile : target='amr_system' n'existe que pour "
                             "backend='production' (chemin natif AMR) ; recu backend=%r" % (backend,))
        mode = {"prototype": "jit", "aot": "aot", "production": "native"}[backend]
        if include is None:
            include = adc_include()
        if std is None:  # le loader natif partage l'ABI du module (norme derivee du loader : c++20 sous
            # Kokkos, c++23 sinon, cf. loader_cxx_std/compile_native) ; jit/aot restent en c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        # NATIF (production) ET AOT : compilateur suivant le backend Kokkos (g++ par defaut,
        # nvcc_wrapper si explicite), flags Kokkos sans linker libkokkos (runtime unique), feature-key
        # kokkos dans le cache. KOKKOS-ONLY : l'aot hybride inclut les en-tetes adc
        # (compiled_block_abi.hpp -> multifab/for_each) qui exigent ADC_HAS_KOKKOS, memes flags que
        # compile_aot ; seul le jit (prototype) reste hote pur (-O2, dynamic_model/bricks sans
        # multifab). kokkos_like sert aussi la cle de cache.
        native = (mode == "native")
        kokkos_like = native or mode == "aot"
        if mode == "aot" and _native_kokkos_root() is None:
            raise RuntimeError(
                "HybridModel.compile : adc_cpp est Kokkos-only -- le modele AOT inclut les en-tetes "
                "adc qui exigent Kokkos. Pointe un Kokkos installe via ADC_KOKKOS_ROOT (ou "
                "Kokkos_ROOT), p.ex. `export ADC_KOKKOS_ROOT=/chemin/vers/kokkos` (Serial suffit "
                "sur CPU).")
        if native:  # garde pre-dlopen : en-tetes != build de _adc -> remede clair (cf. compile_native)
            _check_headers_match_module(include)
            _warn_kokkos_parity()
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if not eff_cxx:
            raise RuntimeError("HybridModel.compile : aucun compilateur C++ trouve")
        std = _probe_cxx_std(eff_cxx, std)  # erreur ACTIONNABLE si le std n'est pas supporte
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, std)
        if so_path is None:
            cache_backend = (("hybrid-" + backend + ";" + _native_feature_key()) if kokkos_like
                             else "hybrid-" + backend)
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            if os.path.exists(so_path):
                return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

        optflags = os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG").split() if native else ["-O2"]
        flags = ["-shared", "-fPIC", "-std=" + std, *optflags]
        kokkos_link_flags = []
        if mode == "jit":
            source = self._emit_jit_source()
        elif mode == "aot":
            # Comme compile_aot : flags Kokkos sans linker libkokkos (le module _adc a deja charge le
            # runtime, singleton), symboles indefinis resolus au chargement ; Apple-ld exige alors
            # -undefined dynamic_lookup (sur ELF/Linux -shared les autorise deja).
            source = self._emit_aot_source()
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
        else:  # native : signature en-tetes + parite backend Kokkos (cf. compile_native / _native_kokkos_flags)
            source = self._emit_native_source(target=target)  # indefinis resolus au chargement (module _adc)
            flags.append('-DADC_HEADER_SIG="%s"' % adc_header_signature(include))
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":  # Apple-ld : autoriser explicitement les indefinis (cf. compile_native)
                flags += ["-undefined", "dynamic_lookup"]
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_hybrid.cpp")
            with open(cpp, "w") as f:
                f.write(source)
            _run_compile([eff_cxx, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags],
                         "HybridModel, backend " + backend)
        return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

    def _compiled_model(self, so_path, backend, target, abi_key, model_hash, cxx, std):
        return CompiledModel(
            so_path=so_path, backend=backend, adder=HyperbolicModel.adder_for(backend),
            target=target, cons_names=self.cons_names, cons_roles=self.cons_roles,
            prim_names=self.prim_names, n_vars=self.n_vars, gamma=self.gamma, n_aux=self.n_aux,
            params={}, caps=_BACKEND_CAPS[backend], abi_key=abi_key, model_hash=model_hash,
            cxx=cxx, std=std, hllc=getattr(self, "_hllc", False),
            roe=getattr(self, "_roe", False),
            wave_speeds=getattr(self, "_has_wave_speeds", True))


# --- Source COUPLEE generique inter-especes (P5 phase 1, splitting EXPLICITE) -----------------------
#
# adc.dsl.CoupledSource decrit un couplage ARBITRAIRE entre especes en FORMULES, par-dela les couplages
# nommes (Ionization / Collision / ThermalExchange) qui figent une formule. On lit des champs (bloc,
# role) en ENTREE et on ECRIT des termes de source (bloc, role) donnes par des expressions symboliques
# (memes Expr que le DSL des modeles : +, *, -, /, **, sqrt, params). compile(backend) compile chaque
# expr en BYTECODE postfixe (machine a pile) que C++ evalue dans le meme for_each_cell device que les
# couplages nommes -- AUCUN .so ni callback Python par cellule. Applique APRES le transport (split).

# Inverse de adc::role_name : nom de role DSL (CamelCase, cf. CANONICAL_ROLES) -> nom canonique
# lowercase attendu par adc::role_from_name (frontiere C++). Source unique de la correspondance.
_ROLE_TO_CANONICAL = {
    "Density": "density",
    "MomentumX": "momentum_x", "MomentumY": "momentum_y", "MomentumZ": "momentum_z",
    "Energy": "energy",
    "VelocityX": "velocity_x", "VelocityY": "velocity_y", "VelocityZ": "velocity_z",
    "Pressure": "pressure", "Temperature": "temperature", "Scalar": "scalar",
}


def _role_canonical(role):
    """Nom de role canonique (lowercase, frontiere C++) pour un role DSL. Accepte deja-canonique."""
    if role in _ROLE_TO_CANONICAL:
        return _ROLE_TO_CANONICAL[role]
    if role in _ROLE_TO_CANONICAL.values():
        return role
    raise ValueError("CoupledSource : role %r inconnu (roles : %s)"
                     % (role, ", ".join(sorted(_ROLE_TO_CANONICAL))))


# Opcodes de la machine a pile : MIROIR de adc::CsOp (coupled_source_program.hpp). Valeurs FIGEES
# (transportees telles quelles par l'ABI Python -> C++).
_CS_PUSHREG = 0
_CS_ADD = 1
_CS_SUB = 2
_CS_MUL = 3
_CS_DIV = 4
_CS_NEG = 5
_CS_POW = 6
_CS_SQRT = 7

# Capacites FIGEES, miroir de coupled_source_program.hpp (kCsMaxReg / kCsMaxTerms / kCsMaxProg). On
# diagnostique cote Python (erreur claire) avant d'atteindre la frontiere C++.
_CS_MAX_REG = 32
_CS_MAX_TERMS = 16
_CS_MAX_PROG = 256


class _CsField(Var):
    """Handle symbolique d'un (bloc, role) : c'est une Var (donc une Expr a part entiere) dont le NOM
    d'environnement '<bloc>::<role>' indexe a la fois l'eval numpy (env) et le registre d'entree au
    codegen bytecode. Sous-classer Var donne directement operateurs / _wrap / to_cpp / eval / deps, donc
    `+k * ne * ng` construit l'arbre Expr attendu sans delegation."""

    def __init__(self, block, role):
        super().__init__("%s::%s" % (block, role), "coupled_field")
        self.block = block
        self.role = role

    def __repr__(self): return "_CsField(%r, %r)" % (self.block, self.role)


class _CsBlock:
    """Aide a la construction : `src.block("electrons").role("density")` -> _CsField. Memorise les
    (bloc, role) demandes sur la source pour fixer l'ordre des registres d'entree."""

    def __init__(self, src, name):
        self._src = src
        self.name = name

    def role(self, role):
        return self._src._field(self.name, role)


class CompiledCoupledSource:
    """Resultat de CoupledSource.compile(...) : empaquette l'ABI PLATE (bytecode) prete pour
    System.add_coupled_source, + l'evaluateur numpy de REFERENCE (memes Expr) pour les tests / un
    integrateur Python. Aucun .so : le couplage est interprete cote C++ (machine a pile device)."""

    def __init__(self, name, backend, in_blocks, in_roles, consts, out_blocks, out_roles,
                 prog_ops, prog_args, prog_lens, terms, reg_order, frequency=0.0,
                 freq_prog_ops=None, freq_prog_args=None, frequency_expr=None):
        self.name = name
        self.backend = backend
        self.frequency = float(frequency)  # mu [1/s] CONSTANTE declaree (0 = pas de borne constante)
        self.in_blocks = list(in_blocks)
        self.in_roles = list(in_roles)        # canoniques (lowercase, frontiere C++)
        self.consts = list(consts)
        self.out_blocks = list(out_blocks)
        self.out_roles = list(out_roles)      # canoniques
        self.prog_ops = list(prog_ops)
        self.prog_args = list(prog_args)
        self.prog_lens = list(prog_lens)
        # Frequence PAR CELLULE mu(U) : bytecode (memes entrees/constantes que la source). VIDES =
        # frequence constante seule. Transportes a System/AmrSystem.add_coupled_source.
        self.freq_prog_ops = list(freq_prog_ops) if freq_prog_ops else []
        self.freq_prog_args = list(freq_prog_args) if freq_prog_args else []
        self._frequency_expr = frequency_expr  # Expr de reference (eval numpy pour les tests) ; None si constante
        self._terms = list(terms)             # [(block, role_canonical, Expr)] : reference numpy
        self._reg_order = list(reg_order)     # noms d'env '<bloc>::<role>' dans l'ordre des registres

    def __repr__(self):
        return ("CompiledCoupledSource(name=%r, backend=%r, n_in=%d, n_const=%d, n_terms=%d)"
                % (self.name, self.backend, len(self.in_blocks), len(self.consts),
                   len(self.out_blocks)))

    def reference_terms(self, fields):
        """Evalue les termes de source sur des tableaux numpy (REFERENCE pour les tests). @p fields :
        dict (bloc, role_canonical) -> tableau ; renvoie [(bloc, role_canonical, dS)] avec dS = S
        (le terme symbolique evalue), AVANT multiplication par dt. Memes Expr que le codegen C++."""
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return [(b, r, e.eval(env)) for (b, r, e) in self._terms]

    def reference_frequency(self, fields):
        """Evalue la frequence PAR CELLULE mu(U) sur des tableaux numpy (REFERENCE pour les tests) :
        memes Expr / table de registres que le bytecode C++. @p fields : dict (bloc, role_canonical) ->
        tableau ; renvoie le tableau mu (memes formules). Renvoie None si la frequence est CONSTANTE
        (pas d'Expr) -- utiliser .frequency dans ce cas."""
        if self._frequency_expr is None:
            return None
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return self._frequency_expr.eval(env)


class CoupledSource:
    """Source COUPLEE generique inter-especes (adc.dsl), Phase 1 (splitting EXPLICITE). Reutilise les
    Expr du DSL des modeles pour les formules de source ; aucun couplage n'est code en dur.

        src = adc.dsl.CoupledSource("ionization")
        ne = src.block("electrons").role("density")
        ng = src.block("neutrals").role("density")
        k  = src.param("Kiz", 1.0)
        src.add("electrons", role="density", expr=+k * ne * ng)
        src.add("neutrals",  role="density", expr=-k * ne * ng)
        sim.add_coupling(src.compile(backend="production"))

    compile(backend) -> CompiledCoupledSource : ABI plate (bytecode) consommee par
    System.add_coupled_source (cote C++ : machine a pile evaluee dans un for_each_cell device, MPI-safe,
    foncteur nomme). Le backend (production / prototype) ne change PAS la numerique : le bytecode est
    interprete cote C++ dans les deux cas (pas de .so par couplage) ; il est conserve pour l'introspection
    et la parite d'API avec le DSL des modeles."""

    def __init__(self, name="coupled_source"):
        self.name = name
        self._fields = {}    # '<bloc>::<role>' -> _CsField (registre d'entree unique par (bloc, role))
        self._reg_order = []  # ordre d'apparition des champs d'entree (-> ordre des registres)
        self._params = {}    # nom -> Param
        self._terms = []     # [(block, role_canonical, Expr)]
        # Indices (dans self._terms) des termes EMIS PAR add_pair, par paire : [(idx_gain, idx_perte)].
        # add_pair garantit que les deux termes portent EXACTEMENT la meme Expr evaluee, l'un +expr,
        # l'autre -expr (Neg) -> echange conservatif par construction. verify_conservation=True les
        # revisite a la compilation pour CONTROLER la propriete (et detecter une rupture cote add manuel).
        self._pairs = []
        self._frequency = 0.0  # mu [1/s] CONSTANTE declaree (borne de pas du couplage ; 0 = pas de borne)
        # mu(U) PAR CELLULE optionnelle : une Expr (meme vocabulaire que les termes : block().role() +
        # param()) emise en bytecode au compile() contre la MEME table de registres. None = constante.
        self._frequency_expr = None

    def frequency(self, mu):
        """FREQUENCE declaree mu [1/s] du couplage (audit vague 3) : borne de pas dt <= cfl / mu
        agregee par System/AmrSystem::step_cfl (raison 'coupled_source:<nom>'). Les couplages sont
        appliques UNE fois par MACRO-pas (splitting, apply_couplings(dt)) : la borne porte sur le
        macro-dt, SANS facteur substeps/stride. mu <= 0 = pas de borne (historique).

        @p mu accepte DEUX formes :
          - un nombre (float / int) -> frequence CONSTANTE (chemin historique, bit-identique) ;
          - une Expr du MEME vocabulaire que les termes (champs block().role() + param()) ->
            frequence PAR CELLULE mu(U), emise en bytecode au compile() et evaluee par cellule cote
            C++ (MAX + all_reduce_max -> dt <= cfl / max(mu)). Les champs references DOIVENT etre
            declares via .block(...).role(...) (comme pour les termes) ; sinon compile() leve une
            ValueError EXPLICITE (champ utilise sans .block(...).role(...)).

        Renvoie self (chainable)."""
        # Un Expr/_CsField/Param -> frequence par cellule (bytecode) ; un scalaire -> constante.
        if isinstance(mu, (int, float)) and not isinstance(mu, bool):
            self._frequency = float(mu)
            self._frequency_expr = None
        else:
            self._frequency_expr = _wrap(mu)  # Expr / _CsField / Param -> noeud d'arbre (cf. _wrap)
            self._frequency = 0.0             # le bytecode porte la borne ; pas de constante en doublon
        return self

    # --- construction symbolique ----------------------------------------------------------------
    def block(self, name):
        """Handle d'un bloc : .role(role) en tire un champ symbolique (bloc, role)."""
        return _CsBlock(self, name)

    def _field(self, block, role):
        canon = _role_canonical(role)
        key = "%s::%s" % (block, canon)
        if key not in self._fields:
            f = _CsField(block, canon)
            self._fields[key] = f
            self._reg_order.append(key)
        return self._fields[key]

    def param(self, name, value):
        """Parametre NOMME constant, utilisable comme une Expr (s'inline comme reel dans le bytecode)."""
        p = Param(name, value, kind="const")
        self._params[name] = p
        return p

    def add(self, block, role=None, expr=None):
        """Ajoute un TERME de source : d_t (bloc.role) += expr. @p expr est une Expr / _CsField / Param /
        nombre. Plusieurs add sur le meme (bloc, role) s'ADDITIONNENT (somme de termes sources)."""
        if role is None:
            raise ValueError("CoupledSource.add : role= requis")
        if expr is None:
            raise ValueError("CoupledSource.add : expr= requis")
        e = expr if isinstance(expr, Expr) else _wrap(expr)  # _CsField / Var sont deja des Expr ; Param/scalaire -> _wrap
        self._terms.append((block, _role_canonical(role), e))
        return self

    def add_pair(self, block_a, block_b, role=None, expr=None):
        """Ajoute un ECHANGE CONSERVATIF de la quantite @p role entre @p block_a et @p block_b, decrit
        par UNE seule expression @p expr (Expr / _CsField / Param / nombre).

        Convention de signe (a retenir) : @p block_a GAGNE +expr, @p block_b PERD -expr, sur la MEME
        valeur evaluee de @p expr. Autrement dit :

            d_t (block_a.role) += +expr
            d_t (block_b.role) += -expr

        C'est l'equivalent DSL des couplages NOMMES du C++ (add_collision / add_thermal_exchange), qui
        calculent UNE valeur et l'appliquent avec deux signes opposes : la somme sur les deux blocs du
        terme echange est nulle a chaque cellule et a chaque pas, donc la quantite totale sum(role) sur
        (block_a, block_b) est CONSERVEE par construction -- independamment de la formule choisie, du dt
        et de l'etat. Choisir @p expr >= 0 pour un transfert de B vers A (A gagne, B perd) ; un signe
        de expr negatif inverse simplement le sens du transfert (la conservation tient dans tous les cas).

        Contraste avec deux .add(...) ecrits a la main (+expr sur A, -expr sur B) : add_pair garantit que
        les DEUX legs portent la MEME Expr (le second est exactement Neg du premier), alors qu'a la main
        rien n'empeche d'ecrire par megarde deux formules legerement differentes -> conservation rompue
        silencieusement. add_pair retire ce risque ; compile(verify_conservation=True) le controle aussi
        pour les couplages ecrits a la main.

        @p block_a et @p block_b doivent etre distincts. add_pair est purement ADDITIF par-dessus .add :
        l'API manuelle reste disponible et inchangee. Renvoie self (chainable)."""
        if role is None:
            raise ValueError("CoupledSource.add_pair : role= requis")
        if expr is None:
            raise ValueError("CoupledSource.add_pair : expr= requis")
        if block_a == block_b:
            raise ValueError("CoupledSource.add_pair : block_a et block_b doivent etre distincts "
                             "(recu %r pour les deux)" % (block_a,))
        canon = _role_canonical(role)
        gain = expr if isinstance(expr, Expr) else _wrap(expr)  # +expr (leg gagnant)
        loss = Neg(gain)                                        # -expr : MEME sous-arbre, signe oppose
        idx_gain = len(self._terms)
        self._terms.append((block_a, canon, gain))
        idx_loss = len(self._terms)
        self._terms.append((block_b, canon, loss))
        self._pairs.append((idx_gain, idx_loss))
        return self

    # --- codegen bytecode -----------------------------------------------------------------------
    def _emit_program(self, expr, reg_index):
        """Compile @p expr (arbre Expr) en bytecode postfixe (listes ops/args paralleles) contre la table
        de registres @p reg_index (nom d'env '<bloc>::<role>' OU valeur constante -> indice de registre).
        Les constantes (Const / Param inline) deviennent un registre constant dedie. Parcours postfixe
        (recursion sur la structure de l'arbre) : une Var pousse son registre, un binaire emet ses deux
        sous-arbres puis l'opcode, etc. -- exactement la semantique de CsProgram::eval cote C++."""
        ops, args = [], []

        def emit(node):
            if isinstance(node, Var):
                if node.name not in reg_index:
                    raise ValueError("CoupledSource : champ %r utilise sans .block(...).role(...)"
                                     % node.name)
                ops.append(_CS_PUSHREG); args.append(reg_index[node.name])
            elif isinstance(node, Const):
                ops.append(_CS_PUSHREG); args.append(self._const_reg(node.value, reg_index))
            elif isinstance(node, Neg):
                emit(node.a); ops.append(_CS_NEG); args.append(0)
            elif isinstance(node, Sqrt):
                emit(node.a); ops.append(_CS_SQRT); args.append(0)
            elif isinstance(node, Add):
                emit(node.a); emit(node.b); ops.append(_CS_ADD); args.append(0)
            elif isinstance(node, Sub):
                emit(node.a); emit(node.b); ops.append(_CS_SUB); args.append(0)
            elif isinstance(node, Mul):
                emit(node.a); emit(node.b); ops.append(_CS_MUL); args.append(0)
            elif isinstance(node, Div):
                emit(node.a); emit(node.b); ops.append(_CS_DIV); args.append(0)
            elif isinstance(node, Pow):
                emit(node.a); emit(node.b); ops.append(_CS_POW); args.append(0)
            else:
                raise TypeError("CoupledSource : noeud d'expression non supporte en Phase 1 : %r "
                                "(supporte : +, -, *, /, **, -unaire, sqrt, champ, constante)"
                                % type(node).__name__)

        emit(expr)
        return ops, args

    def _const_reg(self, value, reg_index):
        """Indice de registre d'une constante @p value (deduplique). Les constantes occupent les
        registres APRES les champs d'entree (cf. CoupledSourceKernel : r[n_in + c] = consts[c])."""
        key = ("const", float(value))
        if key not in reg_index:
            reg_index[key] = len(self._reg_order) + len(self._consts)
            self._consts.append(float(value))
        return reg_index[key]

    @staticmethod
    def _signed_key(expr):
        """Cle STRUCTURELLE SIGNEE de @p expr : (signe, cle_du_corps), ou signe vaut +1 / -1 et
        cle_du_corps est la cle structurelle (_key) de l'expression debarrassee de TOUS ses Neg de tete
        (le signe replie chaque Neg pele). Deux expressions sont structurellement OPPOSEES ssi elles ont
        la MEME cle_du_corps et des signes contraires (p.ex. E et Neg(E), ou -E et Neg(-E)=E). Peler
        TOUS les Neg de tete rend la cle robuste a la paire +expr / -expr d'add_pair MEME quand expr est
        deja un Neg (add_pair pose loss = Neg(gain), donc un Neg de plus). On ne normalise PAS l'algebre
        interne (k*ne vs ne*k) : au pire un faux 'non conservatif' sur des formes ecrites differemment,
        JAMAIS un faux 'conservatif' (le controle reste conservateur, donc sain)."""
        sign = 1
        while isinstance(expr, Neg):
            sign = -sign
            expr = expr.a
        return (sign, _key(expr))

    def _verify_conservation(self):
        """Verifie que, role par role, la somme des termes de source s'ANNULE structurellement : chaque
        contribution +E sur un bloc est compensee par une contribution -E (meme corps structurel) sur un
        autre bloc. Leve ValueError EXPLICITE sinon. C'est exactement la propriete que garantit add_pair
        par construction ; ce controle l'etend aux couplages ecrits a la main (deux .add) et detecte une
        rupture (formules legerement differentes, signe oublie, terme orphelin). Purement symbolique
        (aucune evaluation numerique) : meme cle structurelle que la CSE du codegen."""
        from collections import Counter
        per_role = {}
        for (_block, role, expr) in self._terms:
            sign, body = self._signed_key(expr)
            # Compteur signe par corps structurel : +1 pour +E, -1 pour -E. Tout s'annule => conservatif.
            c = per_role.setdefault(role, Counter())
            c[body] += sign
        offenders = []
        for role in sorted(per_role):
            for body, net in per_role[role].items():
                if net != 0:
                    offenders.append((role, body, net))
        if offenders:
            details = "; ".join(
                "role '%s' : terme %r non compense (net=%+d)" % (role, body, net)
                for (role, body, net) in offenders)
            raise ValueError(
                "CoupledSource.compile(verify_conservation=True) : couplage NON conservatif. "
                "Chaque contribution +E sur un bloc doit etre compensee par -E (meme expression) "
                "sur un autre bloc (utiliser add_pair pour le garantir). Termes non compenses : "
                + details)

    def compile(self, backend="production", verify_conservation=False):
        """Compile la source en CompiledCoupledSource (ABI plate bytecode). @p backend documente
        l'intention (parite d'API avec le DSL des modeles) ; la numerique est identique (interprete C++).

        @p verify_conservation (opt-in, defaut False) : controle SYMBOLIQUE que le couplage conserve
        chaque quantite (role) -- la somme des termes de source d'un meme role s'annule structurellement
        (chaque +E compense par un -E sur un autre bloc). add_pair satisfait cette propriete par
        construction ; ce mode l'etend aux couplages ecrits a la main (deux .add) et leve une ValueError
        EXPLICITE si un terme n'est pas compense (formule divergente, signe oublie, terme orphelin).
        Off par defaut : un couplage volontairement NON conservatif (creation/destruction nette, p.ex.
        ionisation creant une paire e/i) reste licite sans passer le flag."""
        if not self._terms:
            raise ValueError("CoupledSource.compile : aucun terme (.add(...) requis)")
        if verify_conservation:
            self._verify_conservation()
        # Table de registres : champs d'entree d'abord (ordre d'apparition), constantes ensuite.
        reg_index = {key: i for i, key in enumerate(self._reg_order)}
        self._consts = []
        prog_ops, prog_args, prog_lens = [], [], []
        out_blocks, out_roles = [], []
        for (block, role, expr) in self._terms:
            ops, args = self._emit_program(expr, reg_index)
            if len(ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource : programme du terme (%s.%s) trop long (%d > %d)"
                                 % (block, role, len(ops), _CS_MAX_PROG))
            prog_ops += ops
            prog_args += args
            prog_lens.append(len(ops))
            out_blocks.append(block)
            out_roles.append(role)
        # FREQUENCE PAR CELLULE optionnelle : on emet son programme APRES les termes, contre la MEME
        # table de registres (reg_index) et la MEME liste de constantes (self._consts) -- les champs
        # references doivent etre declares via .block().role() (sinon _emit_program leve : champ utilise
        # sans .block(...).role(...)). Les constantes propres a la frequence s'ajoutent apres celles des
        # termes ; cote C++ elles occupent les memes registres r[n_in ..] (CoupledFreqKernel les charge
        # comme la source). Frequence constante (ou aucune) -> programme vide (chemin historique).
        freq_prog_ops, freq_prog_args = [], []
        if self._frequency_expr is not None:
            freq_prog_ops, freq_prog_args = self._emit_program(self._frequency_expr, reg_index)
            if len(freq_prog_ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource : programme de frequence trop long (%d > %d)"
                                 % (len(freq_prog_ops), _CS_MAX_PROG))
        n_reg = len(self._reg_order) + len(self._consts)
        if n_reg > _CS_MAX_REG:
            raise ValueError("CoupledSource : trop de registres (entrees + constantes = %d > %d)"
                             % (n_reg, _CS_MAX_REG))
        if len(out_blocks) > _CS_MAX_TERMS:
            raise ValueError("CoupledSource : trop de termes de source (%d > %d)"
                             % (len(out_blocks), _CS_MAX_TERMS))
        in_blocks = [self._fields[key].block for key in self._reg_order]
        in_roles = [self._fields[key].role for key in self._reg_order]
        return CompiledCoupledSource(
            name=self.name, backend=backend, in_blocks=in_blocks, in_roles=in_roles,
            consts=list(self._consts), out_blocks=out_blocks, out_roles=out_roles,
            prog_ops=prog_ops, prog_args=prog_args, prog_lens=prog_lens,
            terms=self._terms, reg_order=self._reg_order, frequency=self._frequency,
            freq_prog_ops=freq_prog_ops, freq_prog_args=freq_prog_args,
            frequency_expr=self._frequency_expr)
