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

    def compile_or_jit(self, so_path, include, mode="jit", name=None, cxx=None, std="c++20"):
        """API unifiee (facade de l'ideal m.compile_or_jit()) choisissant le backend :
        mode="jit"     -> compile_so  (IModel, dispatch virtuel : prototypage hote, a brancher via
                          System.add_dynamic_block) ;
        mode="compile" -> compile_aot (chemin de production AOT, numerique identique au natif : a
                          brancher via System.add_compiled_block)."""
        if mode == "jit":
            return self.compile_so(so_path, include, name=name, cxx=cxx, std=std)
        if mode == "compile":
            return self.compile_aot(so_path, include, name=name, cxx=cxx, std=std)
        raise ValueError("compile_or_jit : mode 'jit' | 'compile' (recu %r)" % mode)

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
