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

    def cons(self, name):
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names):
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

    def set_primitive_state(self, *vars_or_names):
        """Declare le layout ORDONNE de l'etat primitif (Prim) : noms des composantes, dans l'ordre.
        Necessaire au codegen brique (to_primitive remplit Prim dans cet ordre). Chaque nom doit
        etre une variable conservative ou une primitive deja definie."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]

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
    def emit_cpp(self, func=None):
        """Genere une fonction C++ compilable calculant le flux physique a partir de l'arbre
        symbolique (chaque noeud Expr sait s'ecrire en C++ via to_cpp).

        Signature produite : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Les constantes sont inlinees ; chaque primitive devient une variable locale (les
        sous-expressions H, c... sont inlinees a chaque apparition, pas de CSE pour l'instant).

        C'est l'etape (2) du DSL (cf. docs/ARCHITECTURE_CIBLE.md sect. 3) : on genere du C++ HOTE
        (templatable sur Real). Le codegen Kokkos/CUDA (3) et le JIT (4) restent a faire ; ce code
        ne passe pas encore par l'interface de brique compilee adc (StateVec/Aux/ADC_HD)."""
        name = func or self.name
        if not self._flux:
            raise ValueError("emit_cpp : appeler set_flux(...) d'abord")
        if len(self._flux.get("x", [])) != self.n_vars or len(self._flux.get("y", [])) != self.n_vars:
            raise ValueError("emit_cpp : flux attendu avec %d composantes par direction" % self.n_vars)
        out = [
            "// genere depuis le modele symbolique '%s' (adc.dsl.emit_cpp)" % self.name,
            "// flux physique F = flux(U, dir) ; dir 0=x, 1=y ; U et F de taille %d." % self.n_vars,
            "#include <cmath>",
            "template <class Real>",
            "inline void %s_flux(const Real* U, Real* F, int dir) {" % name,
        ]
        for i, c in enumerate(self.cons_names):
            out.append("  const Real %s = U[%d];" % (c, i))
        for pname, pexpr in self.prim_defs.items():
            out.append("  const Real %s = %s;" % (pname, pexpr.to_cpp()))
        out.append("  if (dir == 0) {")
        for i, comp in enumerate(self._flux["x"]):
            out.append("    F[%d] = %s;" % (i, comp.to_cpp()))
        out.append("  } else {")
        for i, comp in enumerate(self._flux["y"]):
            out.append("    F[%d] = %s;" % (i, comp.to_cpp()))
        out.append("  }")
        out.append("}")
        return "\n".join(out) + "\n"

    def emit_cpp_brick(self, name=None, namespace="adc_generated"):
        """Genere une BRIQUE C++ satisfaisant le concept adc::HyperbolicModel (emballage : etape
        2bis). Le struct produit utilise StateVec / Aux / ADC_HD / Variables et expose flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : il peut
        donc entrer dans un CompositeModel et tourner dans le solveur compile.

        Exige set_primitive_state(...) (layout de Prim) et set_conservative_from([...]) (to_conservative,
        que le DSL ne sait pas inverser tout seul). Constantes inlinees, primitives -> locals, pas de
        CSE. Reste a faire (cf. ARCHITECTURE_CIBLE.md sect. 3) : CSE, codegen Kokkos/CUDA, JIT."""
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

        def eig_block(eigs, ind):
            # noms internes a suffixe '_' : ne masquent ni une variable de l'utilisateur ni le
            # parametre Aux 'a' (cf. revue adverse : collision possible avec une primitive l0/m/a).
            lines = ["%sconst adc::Real lam%d_ = %s;" % (ind, k, e.to_cpp()) for k, e in enumerate(eigs)]
            lines.append("%sadc::Real mws_ = lam0_ < 0 ? -lam0_ : lam0_;" % ind)
            for k in range(1, len(eigs)):
                lines.append("%s{ const adc::Real cand_ = lam%d_ < 0 ? -lam%d_ : lam%d_;"
                             " if (cand_ > mws_) mws_ = cand_; }" % (ind, k, k, k))
            lines.append("%sreturn mws_;" % ind)
            return lines

        cnames = ", ".join('"%s"' % c for c in self.cons_names)
        pnames = ", ".join('"%s"' % p for p in self.prim_state)
        S = [
            "// brique HYPERBOLIQUE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_brick)."
            % self.name,
            "// Satisfait adc::HyperbolicModel : flux + max_wave_speed + conversions + descripteurs.",
            "namespace %s {" % namespace,
            "struct %s {" % nm,
            "  using State = adc::StateVec<%d>;" % nc,
            "  using Prim  = adc::StateVec<%d>;" % npr,
            "  using Aux   = adc::Aux;",
            "  static constexpr int n_vars = %d;" % nc,
            "",
            "  ADC_HD State flux(const State& U, %s, int dir) const {" % aux_param,
        ]
        S += cons_locals() + prim_locals() + aux_locals()
        S.append("    State F{};")
        S.append("    if (dir == 0) {")
        S += ["      F[%d] = %s;" % (i, c.to_cpp()) for i, c in enumerate(self._flux["x"])]
        S.append("    } else {")
        S += ["      F[%d] = %s;" % (i, c.to_cpp()) for i, c in enumerate(self._flux["y"])]
        S += ["    }", "    return F;", "  }", ""]

        S.append("  ADC_HD adc::Real max_wave_speed(const State& U, %s, int dir) const {" % aux_param)
        S += cons_locals() + prim_locals() + aux_locals()
        S.append("    if (dir == 0) {")
        S += eig_block(self._eig["x"], "      ")
        S.append("    } else {")
        S += eig_block(self._eig["y"], "      ")
        S += ["    }", "  }", ""]

        S.append("  ADC_HD Prim to_primitive(const State& U) const {")
        S += cons_locals() + prim_locals()
        S.append("    Prim P{};")
        S += ["    P[%d] = %s;" % (i, p) for i, p in enumerate(self.prim_state)]
        S += ["    return P;", "  }", ""]

        S.append("  ADC_HD State to_conservative(const Prim& P) const {")
        S += ["    const adc::Real %s = P[%d];" % (p, i) for i, p in enumerate(self.prim_state)]
        S.append("    State U{};")
        S += ["    U[%d] = %s;" % (i, e.to_cpp()) for i, e in enumerate(self.cons_from)]
        S += ["    return U;", "  }", ""]

        S.append('  static adc::Variables conservative_vars() { return {adc::VariableKind::Conservative, {%s}, %d}; }'
                 % (cnames, nc))
        S.append('  static adc::Variables primitive_vars() { return {adc::VariableKind::Primitive, {%s}, %d}; }'
                 % (pnames, npr))
        S += ["};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def emit_cpp_source(self, name=None, namespace="adc_generated"):
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
        en plus, aux -> locals) ; pas de CSE. Leve ValueError si set_source(...) n'a pas ete appele."""
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

        S = [
            "// brique de SOURCE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_source)."
            % self.name,
            "// apply(U, a) -> terme source S(U, aux) ; noms aux = champs de adc::Aux (grad_x, grad_y).",
            "namespace %s {" % namespace,
            "struct %s {" % nm,
            "  ADC_HD adc::StateVec<%d> apply(const adc::StateVec<%d>& U, const adc::Aux& a) const {"
            % (nc, nc),
        ]
        S += cons_locals() + prim_locals() + aux_locals()
        S.append("    adc::StateVec<%d> S{};" % nc)
        # _wrap : une composante peut etre un litteral Python (p.ex. 0.0), promu en Const.
        S += ["    S[%d] = %s;" % (i, _wrap(e).to_cpp()) for i, e in enumerate(self._source)]
        S += ["    return S;", "  }", "};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"
