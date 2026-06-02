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
    def set_source(self, s): self._source = list(s)
    def set_elliptic_rhs(self, e): self._elliptic = _wrap(e)

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
