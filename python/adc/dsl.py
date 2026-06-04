"""adc.dsl : mini-DSL SYMBOLIQUE de modeles physiques (prototype, interprete CPU).

Python ECRIT les formules (variables nommees, expressions), pas une fonction appelee par cellule.
Les operations (+, -, *, /, **, adc.dsl.sqrt) construisent un ARBRE d'expressions ; l'evaluateur
applique cet arbre a des tableaux numpy (tout le domaine d'un coup). Un HyperbolicModel declare ses
variables conservatives, ses primitives (definies par des formules), son flux, ses valeurs propres,
sa source et sa contribution elliptique.

    e = adc.dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    p = e.primitive("p", (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    e.set_flux(x=[rhou, rhou*u + p, ...], y=[...])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[...])

Etat : INTERPRETE CPU (numpy) -> le modele TOURNE pour prototyper (via adc.PythonFlux, hote). Les
etapes suivantes (codegen C++ / Kokkos / CUDA, JIT) reutiliseraient le MEME arbre (cf.
docs/ARCHITECTURE_CIBLE.md sect. 3). On ne genere PAS encore de code compile : ce n'est pas le
chemin de production (qui reste les briques C++ compilees, GPU/MPI).
"""
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
                rel = os.path.relpath(p, include)
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                entries.append("%s\n%s\n" % (rel, digest))
    blob = "".join(sorted(entries)).encode()
    return hashlib.sha256(blob).hexdigest()


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


def aux_n_aux(aux_names):
    """Largeur du canal aux requise par ces champs : max(3, plus grand indice canonique + 1).
    Leve ValueError sur un nom inconnu (un champ aux DOIT etre une composante de adc::Aux)."""
    w = AUX_BASE_COMPS
    for nm in aux_names:
        if nm not in AUX_CANONICAL:
            raise ValueError("champ aux '%s' inconnu : attendus %s (composantes de adc::Aux)"
                             % (nm, sorted(AUX_CANONICAL)))
        w = max(w, AUX_CANONICAL[nm] + 1)
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
    def __pow__(self, o): return Pow(self, _wrap(o))

    def eval(self, env): raise NotImplementedError
    def deps(self): return set()
    def __repr__(self): return self._str()
    def _str(self): return "?"


def _wrap(o):
    return o if isinstance(o, Expr) else Const(float(o))


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


# --- Elimination des sous-expressions communes (CSE) -------------------------
# Le codegen inline chaque sous-expression a chaque apparition (H, c... recalcules). La CSE detecte
# les sous-expressions COMPOUND (non feuilles) apparaissant plusieurs fois et les sort en variables
# locales 'cseK_', en ordre de dependance (les plus petites d'abord). Repose sur une cle STRUCTURELLE
# par noeud : deux sous-arbres identiques ont la meme cle, donc la meme locale.
def _children(e):
    if isinstance(e, _Bin):
        return (e.a, e.b)
    if isinstance(e, (Neg, Sqrt)):
        return (e.a,)
    return ()


def _key(e):
    if isinstance(e, Const):
        return ("const", e.value)
    if isinstance(e, Var):
        return ("var", e.name)
    if isinstance(e, Neg):
        return ("neg", _key(e.a))
    if isinstance(e, Sqrt):
        return ("sqrt", _key(e.a))
    return (e.op, tuple(_key(c) for c in _children(e)))  # _Bin (Add/Sub/Mul/Div/Pow)


def _cpp_expand(e, cse_map):
    """C++ du noeud e en developpant SON niveau ; les enfants passent par _cpp_cse (-> locales CSE)."""
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_cse(e.a, cse_map)
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
    deviennent des locales 'cseK_'. roots : liste d'Expr."""
    counts, rep, size = {}, {}, {}

    def visit(e):
        if isinstance(e, (Const, Var)):
            return 1
        k = _key(e)
        s = 1 + sum(visit(c) for c in _children(e))
        counts[k] = counts.get(k, 0) + 1
        rep.setdefault(k, e)
        size[k] = s
        return s

    for r in roots:
        visit(r)
    cand = sorted((k for k, c in counts.items() if c >= 2), key=lambda k: size[k])
    cse_map, lines = {}, []
    for i, k in enumerate(cand):
        name = "cse%d_" % i
        lines.append("%sconst %s %s = %s;" % (indent, real, name, _cpp_expand(rep[k], cse_map)))
        cse_map[k] = name
    return lines, [_cpp_cse(r, cse_map) for r in roots]


# --- Modele hyperbolique declaratif -----------------------------------------
class HyperbolicModel:
    """Modele hyperbolique ecrit en FORMULES : variables conservatives, primitives (definies par
    des expressions), flux, valeurs propres, source, contribution elliptique. cf. module docstring."""

    def __init__(self, name):
        self.name = name
        self.cons_names = []
        self.prim_defs = {}     # nom -> Expr (en fonction des cons / prims precedents / aux)
        self.aux_names = []
        self._flux = {}         # "x" / "y" -> liste d'Expr (une par composante conservative)
        self._eig = {}          # "x" / "y" -> liste d'Expr (valeurs propres)
        self._source = None     # liste d'Expr (une par composante) ou None
        self._elliptic = None   # Expr (contribution au second membre elliptique) ou None
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
        """Champ auxiliaire (p.ex. E_x, E_y, B_z) fourni a l'execution."""
        self.aux_names.append(name)
        return Var(name, "aux")

    def set_flux(self, x, y): self._flux = {"x": list(x), "y": list(y)}
    def set_eigenvalues(self, x, y): self._eig = {"x": list(x), "y": list(y)}
    def set_source(self, s): self._source = [_wrap(e) for e in s]
    def set_elliptic_rhs(self, e): self._elliptic = _wrap(e)

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
        """max_k max_cellules |lambda_k| : borne de Rusanov / CFL."""
        env = self._env(U, aux)
        eigs = self._eig["x" if dir == 0 else "y"]
        return max(float(np.max(np.abs(np.asarray(e.eval(env))))) for e in eigs)

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
        known = set(self.cons_names) | set(self.prim_defs) | set(self.aux_names)
        used = set()
        groups = [self._flux.get("x", []), self._flux.get("y", []),
                  self._eig.get("x", []), self._eig.get("y", []), self._source or []]
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
        return True

    # --- codegen (etape 2 : arbre symbolique -> C++ compilable) ---
    def _codegen_exprs(self, exprs, cse, real="adc::Real", indent="    "):
        """(lignes de locales CSE, [C++ par expr]). Si cse, factorise les sous-expressions communes
        (H, c...) en locales 'cseK_' ; sinon inline chaque expression via to_cpp."""
        if cse:
            return _cse_emit(list(exprs), real, indent)
        return [], [e.to_cpp() for e in exprs]

    def emit_cpp(self, func=None, cse=True):
        """Genere une fonction C++ compilable calculant le flux physique a partir de l'arbre
        symbolique (chaque noeud Expr sait s'ecrire en C++ via to_cpp).

        Signature produite : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constantes inlinees ; chaque primitive devient une variable locale. cse=True (defaut) factorise
        les sous-expressions communes (H, c...) en locales 'cseK_' ; cse=False les recalcule inline.

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
        communes (H, c...) en locales 'cseK_'. Reste a faire (cf. ARCHITECTURE_CIBLE.md sect. 3) :
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
        if not self._eig:
            raise ValueError("emit_cpp_brick : appeler set_eigenvalues(...) d'abord")
        nm = name or (self.name.capitalize() + "Gen")
        nc, npr = self.n_vars, len(self.prim_state)

        def cons_locals():
            return ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]

        def prim_locals():
            return ["    const adc::Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]

        def aux_locals():
            return ["    const adc::Real %s = a.%s;" % (n, n) for n in self.aux_names]

        # parametre Aux nomme 'a' uniquement si une formule lit un champ auxiliaire (sinon anonyme,
        # pour ne pas declencher d'avertissement de parametre inutilise).
        aux_param = "const Aux& a" if self.aux_names else "const Aux&"

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
        S = [
            "#include <cmath>",  # std::sqrt / std::pow : brique autosuffisante (g++ ne tire pas cmath)
            "// brique HYPERBOLIQUE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_brick)."
            % self.name,
            "// Satisfait adc::HyperbolicModel : flux + max_wave_speed + conversions + descripteurs.",
            "namespace %s {" % namespace,
            "struct %s {" % nm,
            "  using State = adc::StateVec<%d>;" % nc,
            "  using Prim  = adc::StateVec<%d>;" % npr,
            "  using Aux   = adc::Aux;",
            "  static constexpr int n_vars = %d;" % nc,
        ]
        # n_aux si une formule (flux / valeurs propres) lit un champ aux supplementaire (B_z...).
        if aux_n_aux(self.aux_names) > AUX_BASE_COMPS:
            S.append("  static constexpr int n_aux = %d;" % aux_n_aux(self.aux_names))
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

        S.append("  ADC_HD adc::Real max_wave_speed(const State& U, %s, int dir) const {" % aux_param)
        S += cons_locals() + prim_locals() + aux_locals()
        nx = len(self._eig["x"])
        etl, ecpps = self._codegen_exprs(self._eig["x"] + self._eig["y"], cse)
        S += etl
        S.append("    if (dir == 0) {")
        S += eig_reduce(ecpps[:nx], "      ")
        S.append("    } else {")
        S += eig_reduce(ecpps[nx:], "      ")
        S += ["    }", "  }", ""]

        # pression + vitesses d'onde signees : emises SI une primitive 'p' (pression) est declaree
        # (convention compressible). Elles rendent la brique generee compatible avec les flux HLLC /
        # Roe (make_block les exige via requires { m.pressure(s); } + m.wave_speeds). Sans 'p' (p.ex.
        # transport scalaire ExB) elles ne sont pas emises : le modele reste limite a Rusanov, inchange.
        if "p" in self.prim_defs:
            S.append("  ADC_HD adc::Real pressure(const State& U) const {")
            S += cons_locals() + prim_locals()
            S += ["    return p;", "  }", ""]
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
            return ["    const adc::Real %s = a.%s;" % (nm_, nm_) for nm_ in self.aux_names]

        na = aux_n_aux(self.aux_names)  # largeur aux requise (B_z... -> > 3)
        S = [
            "#include <cmath>",  # autosuffisant pour std::sqrt / std::pow
            "// brique de SOURCE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_source)."
            % self.name,
            "// apply(U, a) -> terme source S(U, aux) ; noms aux = champs de adc::Aux (grad_x, grad_y).",
            "namespace %s {" % namespace,
            "struct %s {" % nm,
        ]
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
        S += ["    return S;", "  }", "};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def _emit_bricks(self, name=None):
        """Genere les briques (hyperbolique + source + elliptique) et le type CompositeModel<...>
        partages par les DEUX backends (JIT IModel et AOT). Source / elliptique OPTIONNELS : sans
        set_source -> adc::NoSource ; sans set_elliptic_rhs -> rhs nul (pas de couplage Poisson).
        Renvoie (nv, code_des_briques, type_composite)."""
        nm = name or (self.name.capitalize() + "Gen")
        nv = self.n_vars
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

    def compile_so(self, so_path, include, name=None, cxx=None, std="c++20"):
        """JIT : genere le MODELE COMPLET (emit_cpp_so_source) et compile une bibliotheque partagee
        chargeable par System.add_dynamic_block (dlopen). Le .so expose un CompositeModel<hyperbolique,
        source, elliptique> : le bloc dynamique applique le flux ET la source, et contribue au Poisson
        de systeme via elliptic_rhs (vrai bloc couple, plus seulement transport). include = dossier des
        en-tetes adc ; cxx = compilateur (defaut c++/g++/clang++). Renvoie so_path. Exige
        set_primitive_state(...) et set_conservative_from([...]) (comme emit_cpp_brick)."""
        import os
        import shutil
        import subprocess
        import tempfile

        src = self.emit_cpp_so_source(name=name)
        cc = cxx or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cc:
            raise RuntimeError("compile_so : aucun compilateur C++ trouve")
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            subprocess.run([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                            "-o", so_path], check=True)
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

    def compile_aot(self, so_path, include, name=None, cxx=None, std="c++20"):
        """Backend "compile" (AOT) : genere le MODELE COMPLET (emit_cpp_aot_source) et compile une .so
        chargeable par System.add_compiled_block. Contrairement au backend "jit" (compile_so : IModel,
        dispatch virtuel, Rusanov hote), le bloc tourne ici le chemin de PRODUCTION (flux HLLC/Roe au
        choix, ordre 2, SSPRK2/IMEX) sur le modele genere -- numerique identique a un bloc natif.
        include = dossier des en-tetes adc ; cxx = compilateur. Renvoie so_path."""
        import os
        import shutil
        import subprocess
        import tempfile

        src = self.emit_cpp_aot_source(name=name)
        cc = cxx or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cc:
            raise RuntimeError("compile_aot : aucun compilateur C++ trouve")
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_aot.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            subprocess.run([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                            "-o", so_path], check=True)
        return so_path

    def emit_cpp_native_loader(self, name=None):
        """Source du LOADER NATIF (backend "production") : le MODELE COMPLET en CompositeModel<...>
        derriere une ABI extern "C" MINCE de DEUX symboles.

        A la difference du backend "aot" (emit_cpp_aot_source : ABI plate de tableaux, le .so
        recalcule tout sur une grille locale et marshale les tableaux), le loader natif NE porte PAS
        la numerique : il se contente d'INSTALLER le modele genere comme bloc NATIF du System deja
        construit, via le gabarit en-tete adc::add_compiled_model<ProdModel>. Ce gabarit fabrique les
        fermetures sur le CONTEXTE REEL du System (grid_context) -> le bloc tourne ensuite le MEME
        chemin que add_block (assemble_rhs, fill_boundary), ZERO-COPIE, device-clean (foncteurs nommes).

        Deux symboles extern "C" :
          - adc_native_abi_key() : cle d'ABI figee a la compilation DU LOADER (adc::detail::
            abi_key_string()). add_native_block la compare a abi_key() du module -> erreur explicite
            si en-tetes / compilateur / standard divergent (pas d'UB silencieux a la frontiere C++).
          - adc_install_native(sys, name, limiter, riemann, recon, time, gamma, substeps, evolve) :
            reinterpret_cast<adc::System*>(sys) puis add_compiled_model<ProdModel>(*s, name, ...).
            Le schema transite en arguments plats (chaines + double + int) ; aucun objet C++ ne
            traverse l'ABI dans CE sens (seul le System* est repris par reference cote loader, d'ou
            l'exigence d'ABI identique verifiee par la cle)."""
        nv, bricks, composite = self._emit_bricks(name)
        return ('#include <adc/runtime/dsl_block.hpp>\n'      # add_compiled_model<Model> (gabarit natif)
                '#include <adc/runtime/abi_key.hpp>\n'         # detail::abi_key_string (cle figee a la compil)
                '#include <adc/physics/bricks.hpp>\n'          # CompositeModel + NoSource + briques
                '#include <adc/core/variables.hpp>\n'
                '#include <string>\n'
                + bricks
                + '\nnamespace adc_generated { using ProdModel = %s; }\n' % composite  # alias sans virgule
                + 'extern "C" const char* adc_native_abi_key() {\n'
                + '  static const std::string k = adc::detail::abi_key_string();\n'
                + '  return k.c_str();\n'
                + '}\n'
                + 'extern "C" void adc_install_native(void* sys, const char* name, const char* limiter,\n'
                + '                                    const char* riemann, const char* recon,\n'
                + '                                    const char* time, double gamma, int substeps,\n'
                + '                                    int evolve) {\n'
                + '  adc::System* s = reinterpret_cast<adc::System*>(sys);\n'
                + '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                + '                                                    limiter, riemann, recon, time, gamma,\n'
                + '                                                    substeps, evolve != 0);\n'
                + '}\n'
                + self._emit_metadata("adc_generated::ProdModel"))  # noms/roles/gamma (diagnostic, comme AOT/JIT)

    def compile_native(self, so_path, include, name=None, cxx=None, std="c++23"):
        """Backend "production" : genere le LOADER NATIF (emit_cpp_native_loader) et le compile en une
        .so chargeable par System.add_native_block. Le .so inline add_compiled_model<ProdModel> : le
        bloc tourne le chemin NATIF zero-copie (parite stricte avec add_block / add_compiled_model<>).

        Le loader appelle des methodes hors-ligne du module _adc (install_block / grid_context /
        ensure_aux_width) DEFINIES ailleurs : on compile donc avec '-undefined dynamic_lookup' (macOS)
        pour autoriser ces indefinis (resolus a l'execution contre le module deja charge ; cf.
        add_native_block). On bake aussi -DADC_HEADER_SIG=<signature> a l'IDENTIQUE du module pour que
        les cles d'ABI concordent quand les en-tetes concordent. std defaut c++23 (le module se compile
        en C++23 hors Kokkos) : un std different changerait __cplusplus donc la cle -> rejet explicite.
        include = dossier des en-tetes adc ; cxx = compilateur. Renvoie so_path."""
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        src = self.emit_cpp_native_loader(name=name)
        cc = cxx or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cc:
            raise RuntimeError("compile_native : aucun compilateur C++ trouve")
        sig = adc_header_signature(include)
        # Les symboles System hors-ligne (install_block/grid_context/ensure_aux_width) sont resolus au
        # CHARGEMENT contre le module _adc deja charge (add_native_block le promeut en portee globale).
        # -DADC_HEADER_SIG : MEME signature que le build du module (concordance des cles d'ABI).
        flags = ["-shared", "-fPIC", "-std=" + std, "-O2",
                 "-DADC_HEADER_SIG=\"%s\"" % sig]
        # macOS/Apple-ld : il FAUT autoriser explicitement les indefinis (resolus a l'execution). Sur
        # ELF/Linux, -shared autorise deja les indefinis ; '-undefined dynamic_lookup' n'est PAS une
        # option ld GNU (elle creerait un symbole indefini parasite 'dynamic_lookup'), donc darwin SEUL.
        if sys.platform == "darwin":
            flags += ["-undefined", "dynamic_lookup"]
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_native.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            subprocess.run([cc, *flags, "-I", include, cpp, "-o", so_path], check=True)
        return so_path

    def compile_or_jit(self, so_path, include, mode="jit", name=None, cxx=None, std="c++20"):
        """API unifiee (facade de l'ideal m.compile_or_jit()) choisissant le backend :
        mode="jit"     -> compile_so  (IModel, dispatch virtuel : prototypage hote, a brancher via
                          System.add_dynamic_block) ;
        mode="compile" -> compile_aot (chemin de production AOT, numerique identique au natif : a
                          brancher via System.add_compiled_block) ;
        mode="native"  -> compile_native (loader natif zero-copie : add_compiled_model<> via
                          System.add_native_block ; chemin "production")."""
        if mode == "jit":
            return self.compile_so(so_path, include, name=name, cxx=cxx, std=std)
        if mode == "compile":
            return self.compile_aot(so_path, include, name=name, cxx=cxx, std=std)
        if mode == "native":
            return self.compile_native(so_path, include, name=name, cxx=cxx, std=std)
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

    def compile(self, so_path, include, backend="aot", name=None, cxx=None, std=None,
                require_metadata=False):
        """Facade de compilation par INTENTION : compile le modele en une .so via le moteur designe
        par @p backend et renvoie son chemin. Wrappe les moteurs existants (compile_so / compile_aot /
        compile_native) SANS changer la numerique ; preserve de bout en bout noms, VariableRole, gamma,
        n_aux, B_z et T_e (les memes briques + metadonnees ABI que compile_or_jit).

        @p backend :
          "prototype"  -> JIT (compile_so) : iteration rapide, dispatch virtuel hote (Rusanov ordre 1),
                          a brancher cote System via add_dynamic_block ;
          "aot"        -> AOT (compile_aot) : chemin de production host-marshale, numerique identique au
                          bloc natif, a brancher via add_compiled_block ;
          "production" -> NATIF (compile_native) : loader .so inline add_compiled_model<ProdModel>, bloc
                          natif zero-copie (parite stricte add_block / add_compiled_model<>), a brancher
                          via add_native_block (cle d'ABI verifiee). Chemin device-clean prepare.

        @p std : standard C++. Defaut None -> "c++23" pour "production" (le loader natif partage l'ABI
        du module, compile en C++23 hors Kokkos : un std different changerait __cplusplus donc la cle
        d'ABI -> rejet explicite par add_native_block), "c++20" pour les autres (inchange).

        @p require_metadata (defaut False) : si True, exige que le .so transporte des roles physiques
        utiles ET un gamma explicite (set_gamma), faute de quoi le System retomberait sur le fallback
        (roles 'custom' / gamma 1.4) -- regression silencieuse des couplages inter-especes. Sert a un
        pipeline de production qui veut une erreur EXPLICITE plutot qu'un fallback muet.

        Leve ValueError sur un backend inconnu ou une feature incompatible avec le backend demande
        (plutot qu'un echec obscur a l'execution). Renvoie so_path.

        Pour connaitre l'adder System a employer : voir adder_for(backend)."""
        if backend not in self._BACKENDS:
            raise ValueError("compile : backend %r inconnu (attendus %s)"
                             % (backend, sorted(self._BACKENDS)))
        mode, adder = self._BACKENDS[backend]
        if std is None:  # defaut par backend : le natif partage l'ABI C++23 du module, les autres c++20
            std = "c++23" if mode == "native" else "c++20"

        # Garde-fou device/production : le backend "prototype" passe par add_dynamic_block (IModel,
        # dispatch VIRTUEL, Rusanov hote ordre 1). Ce n'est PAS un chemin de production device-clean :
        # demander explicitement les metadonnees de production dessus est une incoherence -> erreur
        # claire plutot que le fallback muet d'un .so prototype.
        if require_metadata and backend == "prototype":
            raise ValueError(
                "compile : backend 'prototype' (JIT, dispatch virtuel hote) incompatible avec "
                "require_metadata=True ; utiliser backend='aot' ou 'production' pour le chemin "
                "device-clean a metadonnees garanties")

        if require_metadata:
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

        return self.compile_or_jit(so_path, include, mode=mode, name=name, cxx=cxx, std=std)

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
        out = [
            "#include <cmath>",  # autosuffisant pour std::sqrt / std::pow
            "// brique de SECOND MEMBRE elliptique generee depuis '%s' (adc.dsl.emit_cpp_elliptic)."
            % self.name,
            "// rhs(U) -> Real : second membre f(U) de l'operateur elliptique (p.ex. densite de charge).",
            "namespace %s {" % namespace,
            "struct %s {" % nm,
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
    # par construction (halos fill_boundary). amr=False (System mono-niveau ; AMR = Phase D). gpu=False
    # par PRUDENCE : le chemin natif est device-clean en C++ (GH200) mais la validation end-to-end
    # depuis Python (add_native_block sur device) est une PR dediee non encore faite (DSL section 5).
    "production": {"cpu": True, "mpi": True, "amr": False, "gpu": False},
}


class Param:
    """Parametre NOMME d'un modele DSL, utilisable comme une Expr dans les formules.

    Mode (a), constante figee a la compilation (la SEULE supportee en Phase A) : `kind="const"`.
    Le codegen INLINE deja toute constante (Const.to_cpp -> repr(value)), donc Param s'inline en
    Const(value) au codegen (zero-risque cote brique generee) tout en gardant son IDENTITE
    (name/value/kind) pour l'introspection (m.params), les diagnostics et la reproductibilite.

    Mode (b), parametre runtime (modifiable sans recompiler) : `kind="runtime"` -> NotImplementedError
    (changement d'ABI + codegen requis, phase ulterieure ; cf. DSL_MODEL_DESIGN.md section 2b).

    Comme Param se comporte comme un noeud d'arbre (il herite d'Expr), `g * (E - ...)` construit
    directement l'arbre attendu : `eval`/`to_cpp`/`deps` delegues a un Const interne (la valeur est
    une constante, pas une variable d'environnement -> pas de dependance a verifier dans check())."""

    # NB : Param N'HERITE PAS d'Expr pour eviter d'embarquer son etat (name/kind) dans la cle
    # structurelle de CSE ; il EXPOSE plutot les hooks d'arbre en deleguant a un Const interne.
    def __init__(self, name, value, kind="const"):
        if kind not in ("const", "runtime"):
            raise ValueError("Param : kind 'const' | 'runtime' (recu %r)" % (kind,))
        if kind == "runtime":
            raise NotImplementedError(
                "param '%s' runtime non supporte (changement d'ABI/codegen requis, phase "
                "ulterieure) ; utiliser un param constant (kind='const') ou un champ aux" % name)
        self.name = name
        self.value = float(value)
        self.kind = kind
        self._const = Const(self.value)  # s'inline au codegen : la valeur est ecrite EN DUR dans le .so

    # --- hooks d'arbre (delegues au Const interne) : Param utilisable comme une Expr ---
    def eval(self, env): return self._const.eval(env)
    def to_cpp(self): return self._const.to_cpp()
    def deps(self): return set()  # une constante n'a aucune dependance (rien a verifier dans check())

    # --- operateurs : Param se combine comme une Expr (promotion via _wrap du Const interne) ---
    def __add__(self, o): return Add(self._const, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self._const)
    def __sub__(self, o): return Sub(self._const, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self._const)
    def __mul__(self, o): return Mul(self._const, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self._const)
    def __truediv__(self, o): return Div(self._const, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self._const)
    def __neg__(self): return Neg(self._const)
    def __pow__(self, o): return Pow(self._const, _wrap(o))

    def __float__(self): return self.value
    def __repr__(self): return "Param(%r, %r, kind=%r)" % (self.name, self.value, self.kind)


class CompiledModel:
    """Resultat de `m.compile(...)` : empaquette le `.so` produit + TOUT ce qu'il faut pour le
    brancher correctement (dispatch adder, diagnostic ABI, reproductibilite). Remplace le couple
    historique (str so_path, adder_for(backend)) par un objet unique.

    Les metadonnees ne sont PAS relues du `.so` : Python detient deja noms/roles/gamma/n_aux/params
    (le HyperbolicModel les porte) ; CompiledModel les expose juste pour le dispatch (add_equation)
    et les diagnostics. cf. DSL_MODEL_DESIGN.md section 3."""

    def __init__(self, so_path, backend, adder, cons_names, cons_roles, prim_names, n_vars,
                 gamma, n_aux, params, caps, abi_key, model_hash, cxx, std):
        self.so_path = so_path
        self.backend = backend       # "prototype" | "aot" | "production"
        self.adder = adder           # nom de methode System : add_dynamic_block / add_compiled_block / add_native_block
        self.cons_names = list(cons_names)
        self.cons_roles = list(cons_roles)
        self.prim_names = list(prim_names)
        self.n_vars = int(n_vars)
        self.gamma = gamma           # None = defaut historique 1.4 cote System
        self.n_aux = int(n_aux)
        self.params = dict(params)   # {name: Param}
        self.caps = dict(caps)       # {cpu/mpi/amr/gpu: bool}
        self.abi_key = abi_key       # cle ABI miroir d'adc_header_signature + compilateur/std
        self.model_hash = model_hash  # hash stable formules+roles+n_aux+params
        self.cxx = cxx
        self.std = std

    def __repr__(self):
        return ("CompiledModel(backend=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "adder=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.adder, self.abi_key or "", self.model_hash or ""))


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
            # CAS rho=rho : un nom DEJA conservatif (la primitive est la variable conservative elle-meme,
            # ex. la densite) n'est PAS redefini comme primitive -- sinon le codegen emettrait
            # `const Real rho = rho;` (auto-init). On le laisse simplement REJOINDRE le layout.
            ordered = list(named.keys())
            for nm in ordered:
                if nm not in self._m.cons_names:
                    self._m.primitive(nm, named[nm])
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # forme positionnelle : fixe le layout a partir de noms/Var deja definis.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name):
        """Champ auxiliaire (doit etre une clef de AUX_CANONICAL : phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

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

    def source(self, s):
        """Terme source S(U, aux), une expression par composante (optionnel ; delegue a set_source)."""
        self._m.set_source(s)

    def elliptic_rhs(self, e):
        """Contribution au second membre elliptique (couplage Poisson ; delegue a set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def gamma(self, value):
        """Indice adiabatique (EOS), porte par ADC_EXPORT_BLOCK_GAMMA (delegue a set_gamma)."""
        self._m.set_gamma(value)

    def param(self, name, value, kind="const"):
        """Parametre NOMME utilisable dans les formules. Mode (a) (`kind="const"`, defaut) : constante
        figee a la compilation, inlinee au codegen ; stockee dans m.params (introspection /
        reproductibilite). Mode (b) (`kind="runtime"`) : NotImplementedError (Phase E).

        CAS gamma : si name == "gamma", appelle AUSSI set_gamma(value) pour que la metadonnee ABI
        reste coherente (sinon le System retombe sur 1.4)."""
        p = Param(name, value, kind=kind)  # leve NotImplementedError si kind == "runtime"
        self.params[name] = p
        if name == "gamma":
            self._m.set_gamma(p.value)
        return p

    def check(self):
        """Verifie les dependances (variables referencees declarees). Leve ValueError sinon."""
        return self._m.check()

    # --- introspection (lecture seule, deleguee au modele backing) ---
    @property
    def cons_names(self): return self._m.cons_names

    @property
    def prim_state(self): return self._m.prim_state

    @property
    def n_vars(self): return self._m.n_vars

    def _model_hash(self):
        """Hash stable du modele : formules (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + params. Sert a identifier/reutiliser un .so deja compile et a tracer le run. Repose sur
        repr(Expr) (stable, structurel) ; insensible a l'ordre des dict (tries)."""
        import hashlib
        m = self._m
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
        parts.append("n_aux=%d" % aux_n_aux(m.aux_names))
        parts.append("gamma=%r" % m.gamma)
        parts.append("params=%s" % ";".join("%s=%r:%s" % (k, self.params[k].value, self.params[k].kind)
                                             for k in sorted(self.params)))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def compile(self, so_path, include, backend="aot", target="system", name=None, cxx=None,
                std=None, require_metadata=False):
        """Compile le modele en un CompiledModel (Phase A). Delegue la GENERATION + compilation a
        HyperbolicModel.compile (moteurs inchanges : compile_so / compile_aot / compile_native), puis
        empaquette le .so avec les metadonnees deja connues (pas de relecture du .so).

        @p backend : "prototype" | "aot" | "production" (cf. HyperbolicModel.compile).
        @p target : "system" (defaut) | "amr_system". En Phase A seul "system" est cable
            (AmrSystem.add_equation est Phase D) -> "amr_system" leve NotImplementedError.
        PAS d'argument `device` : les capacites GPU/MPI/AMR sont verifiees au branchement
            (add_equation) / a l'execution, pas figees a la compilation (DSL_MODEL_DESIGN.md point 7).

        Renvoie un CompiledModel portant so_path, backend, adder, noms/roles/gamma/n_aux/params, caps,
        abi_key, model_hash, cxx, std."""
        import shutil
        if backend not in HyperbolicModel._BACKENDS:
            raise ValueError("compile : backend %r inconnu (attendus %s)"
                             % (backend, sorted(HyperbolicModel._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile : target 'system' | 'amr_system' (recu %r)" % (target,))
        if target == "amr_system":
            raise NotImplementedError(
                "compile : target='amr_system' (DSL Phase D) non cable ; AmrSystem.add_equation viendra "
                "avec le pendant natif add_compiled_model(AmrSystem&). Phase A : target='system'")

        m = self._m
        # std effectif : meme defaut par backend que HyperbolicModel.compile (c++23 natif, c++20 sinon).
        mode = HyperbolicModel._BACKENDS[backend][0]
        eff_std = std if std is not None else ("c++23" if mode == "native" else "c++20")
        eff_cxx = cxx or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

        # Compilation (moteurs inchanges, garde-fous require_metadata/backend de HyperbolicModel.compile).
        out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                             require_metadata=require_metadata)

        adder = HyperbolicModel.adder_for(backend)
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        return CompiledModel(
            so_path=out_path, backend=backend, adder=adder,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_n_aux(m.aux_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=_abi_key_python(include, eff_cxx, eff_std), model_hash=self._model_hash(),
            cxx=eff_cxx, std=eff_std)
