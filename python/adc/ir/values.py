"""Reference / witness value nodes for the ADC IR (Spec 4).

Moved verbatim from :mod:`adc.dsl`:

  - :class:`EigWitness` -- a scalar value drawn from the spectrum of a small
    dense matrix of expressions (and its ``_EIG_FIELDS`` / ``_EIG_PREDICATES``
    catalogues);
  - :class:`StateRef`   -- the LEFT/RIGHT state marker of the two-state Roe
    dissipation;
  - :class:`RuntimeParamRef` -- a reference to a runtime parameter.

These carry a ``to_cpp`` accessor (verbatim string), but the codegen walkers
and the named-functor emission (``_eig_witness_helpers``) stay in
:mod:`adc.dsl`.
"""
import numpy as np

from .expr import Expr, _wrap


# Champs scalaires d'adc::EigBounds exposables comme une valeur DSL (cf. dense_eig.hpp).
_EIG_FIELDS = {
    "max_im": "max_im",  # plus grande |Im(lambda)| -> temoin de VP complexes (0 = spectre reel)
    "lmin": "lmin",      # plus petite partie reelle du spectre
    "lmax": "lmax",      # plus grande partie reelle du spectre
}

# Predicats REEL/COMPLEXE d'adc::EigBounds (ADC-276) exposes comme une valeur DSL 1.0/0.0 (ADC-362) :
# nom DSL -> METHODE C++ const(im_tol) -> bool, abaissee en adc::Real(...). Verrouillee sur converged
# (un repli Gershgorin -> false -> 0.0, jamais reel), au contraire d'un comparatif brut sur max_im (=0
# sous repli, donc lu comme reel). Compose dans les masques branchless de m.projection.
_EIG_PREDICATES = {
    "all_real": "all_real",  # 1.0 ssi le bloc a CONVERGE et le spectre est reel (sinon 0.0)
}


class EigWitness(Expr):
    """Valeur scalaire issue du spectre d'une PETITE matrice dense construite a partir d'expressions
    (ADC-289). La matrice @c rows (liste de @c k lignes de @c k Expr, ordre ligne-major) est diagonalisee
    par ``adc::real_eig_minmax`` (dense_eig.hpp, ADC_HD, repli Gershgorin sur non-convergence, cap QR
    releve par ADC-195) ; @c field choisit le champ rendu de @c adc::EigBounds : ``max_im`` (temoin de
    VP complexes : 0 = spectre reel donc hyperbolique), ``lmin`` / ``lmax`` (extremes des parties
    reelles), ou le PREDICAT ``all_real`` (ADC-362, cf. dsl.eig_all_real) qui rend 1.0/0.0 (spectre reel
    ET convergent, sinon 0.0) abaisse sur ``adc::EigBounds::all_real(im_tol)`` (verrouille sur converged).
    Sert la logique branchless de m.projection : ``si max_im > tol alors corriger`` s'ecrit
    en masque max/min/sign sur cette valeur, sans branche dynamique.

    Codegen device-clean : l'emission est un FONCTEUR NOMME (methode statique ADC_HD de la brique
    generee) qui remplit un ``adc::Real M[k][k]`` puis appelle ``real_eig_minmax`` -- jamais de lambda
    etendue cross-TU (casse nvcc). to_cpp() rend l'APPEL de ce foncteur en passant les entrees comme
    arguments scalaires (chacune evaluee une seule fois cote appelant : compatible CSE). La brique
    declare le foncteur une fois par couple (field, k) rencontre (cf. _eig_witness_helpers).

    eval(env) : miroir hote via numpy.linalg.eigvals (reference de test) ; field 'max_im' = max des
    |Im| (0 si toutes reelles), 'lmin'/'lmax' = extremes des parties reelles. ATTENTION : le repli
    Gershgorin du chemin C++ (non-convergence d'un bloc >= 3 sous le cap QR) n'est PAS reproduit ici --
    sur des matrices saines (cas vise) les deux chemins coincident a la tolerance QR (cf. dense_eig)."""

    def __init__(self, rows, field, im_tol=None):
        if field not in _EIG_FIELDS and field not in _EIG_PREDICATES:
            raise ValueError("EigWitness : field '%s' inconnu (attendu : %s)"
                             % (field, ", ".join(sorted({**_EIG_FIELDS, **_EIG_PREDICATES}))))
        rows = [list(r) for r in rows]
        k = len(rows)
        if k < 1:
            raise ValueError("EigWitness : matrice vide (au moins 1 ligne)")
        if k > 16:
            raise ValueError("EigWitness : matrice %dx%d > 16x16 (limite de real_eig_minmax, "
                             "tampon pile O(N^2) par thread device)" % (k, k))
        for r in rows:
            if len(r) != k:
                raise ValueError("EigWitness : matrice non carree (%d lignes, ligne de %d entrees)"
                                 % (k, len(r)))
        self.rows = [[_wrap(e) for e in r] for r in rows]
        self.k = k
        self.field = field
        # im_tol : seuil RELATIF d'|Im| -- n'a de sens que pour un PREDICAT (all_real). Pour un champ
        # scalaire (max_im/lmin/lmax) il est rejete s'il est fourni, et reste None -> chemin scalaire
        # bit-identique a l'historique (cle CSE, codegen et eval inchanges).
        if field in _EIG_PREDICATES:
            tol = 1e-5 if im_tol is None else float(im_tol)
            if not (tol > 0.0) or not np.isfinite(tol):
                raise ValueError("EigWitness : im_tol doit etre fini et > 0 (recu %r)" % (im_tol,))
            self.im_tol = tol
        else:
            if im_tol is not None:
                raise ValueError("EigWitness : im_tol ne s'applique qu'aux predicats "
                                 "(field='%s' est un champ scalaire)" % field)
            self.im_tol = None

    def is_predicate(self):
        """True si @c field est un PREDICAT reel/complexe (all_real) plutot qu'un champ scalaire."""
        return self.field in _EIG_PREDICATES

    def entries(self):
        """Entrees de la matrice a plat (ordre ligne-major), une par enfant Expr."""
        return [e for row in self.rows for e in row]

    def _extra_args_cpp(self):
        """Arguments scalaires C++ apres les entrees de la matrice : le seuil relatif im_tol pour un
        predicat (all_real), aucun pour un champ scalaire (chemin scalaire bit-identique)."""
        return [repr(self.im_tol)] if self.is_predicate() else []

    def helper_name(self):
        """Nom du foncteur nomme emis dans la brique pour ce couple (field, taille)."""
        return "adc_eig_%s_%dx%d" % (self.field, self.k, self.k)

    def eval(self, env):
        # Miroir hote (reference de test / prototypage) : empile la matrice par cellule puis numpy.
        # Les entrees sont diffusees a une forme commune ; eigvals s'applique sur le dernier axe 2x2.
        vals = [e.eval(env) for e in self.entries()]
        bshape = np.broadcast(*[np.asarray(v) for v in vals]).shape if vals else ()
        k = self.k
        M = np.empty(bshape + (k, k), dtype=float)
        for idx, v in enumerate(vals):
            M[..., idx // k, idx % k] = np.broadcast_to(np.asarray(v, dtype=float), bshape)
        ev = np.linalg.eigvals(M)  # (..., k) complexe
        if self.field == "max_im":
            out = np.max(np.abs(ev.imag), axis=-1)
        elif self.field == "lmin":
            out = np.min(ev.real, axis=-1)
        elif self.field == "lmax":
            out = np.max(ev.real, axis=-1)
        else:  # predicat all_real (ADC-362) : MEME formule RELATIVE que adc::EigBounds::all_real.
            max_im = np.max(np.abs(ev.imag), axis=-1)
            lmin = np.min(ev.real, axis=-1)
            lmax = np.max(ev.real, axis=-1)
            scale = np.maximum(np.maximum(np.abs(lmin), np.abs(lmax)), 1.0)
            # numpy/LAPACK converge toujours -> PAS de kUnknown cote hote (le miroir definit le spectre
            # comme converge par construction) ; une non-convergence DEVICE rendrait 0.0 (= PAS reel),
            # jamais 1.0 : direction sure, coherente avec all_real (converged && max_im <= im_tol*scale).
            out = (max_im <= self.im_tol * scale).astype(float)
        return out if bshape else float(out)

    def deps(self):
        d = set()
        for e in self.entries():
            d |= e.deps()
        return d

    def to_cpp(self):
        args = [e.to_cpp() for e in self.entries()] + self._extra_args_cpp()
        return "%s(%s)" % (self.helper_name(), ", ".join(args))

    def _str(self):
        return "eig_%s([%s])" % (self.field, ", ".join(str(e) for e in self.entries()))


class StateRef(Expr):
    """STATE marker for the TWO-state Roe dissipation (m.roe_dissipation): the
    enclosed sub-expression evaluates on the LEFT state UL (side='L', dsl.left) or RIGHT state UR
    (side='R', dsl.right). At codegen of the hook roe_dissipation(UL, AL, UR, AR, dir), left(e) emits e
    with the locals computed from UL, right(e) from UR. Has meaning ONLY in the lines
    given to m.roe_dissipation: the numpy interpreter does not handle it (the two-state dissipation is
    compiled into C++, not evaluated on the host). deps() = deps of the sub-expression (dependency
    checking)."""

    def __init__(self, side, expr):
        if side not in ("L", "R"):
            raise ValueError("StateRef: side must be 'L' (UL) or 'R' (UR), got %r" % (side,))
        self.side = side
        self.expr = _wrap(expr)

    def deps(self):
        return self.expr.deps()

    def eval(self, env):
        raise NotImplementedError(
            "StateRef (dsl.left / dsl.right) is not evaluated by the numpy interpreter: the two-state "
            "Roe dissipation is EMITTED in C++ (m.roe_dissipation), not interpreted on the host.")

    def _str(self):
        return "%s(%s)" % ("left" if self.side == "L" else "right", self.expr)


class RuntimeParamRef(Expr):
    """Reference to a RUNTIME parameter (P7-b) in the expression tree. Unlike a Const
    (const param inlined HARD), this node emits `params.get(<index>)` at codegen: the generated brick
    READS the value from its adc::RuntimeParams member instead of having it baked in. The value can thus
    be CHANGED at runtime without recompiling the .so (cf. include/adc/runtime/runtime_params.hpp).

    @c index: STABLE index of the parameter in the RuntimeParams block (assigned by the model at
    compilation, sorted order of names); -1 as long as it is not assigned. @c value: DECLARATION
    value (used by the numpy eval interpreter and as the default of the generated member -> without a
    set call at runtime, the block behaves as with a const param of this value).

    Structural CSE key (cf. _key): the NAME (two refs to the same runtime param share the same
    CSE local); the declaration value does not enter the key (it is runtime, not structural)."""

    def __init__(self, name, value, index=-1):
        self.name = name
        self.value = float(value)
        self.index = index

    def eval(self, env):
        # Numpy interpreter (host proto / debug): the declaration value stands in for the current value
        # (the numpy path does not go through RuntimeParams; it serves prototyping, not production).
        return self.value

    def deps(self):
        # A runtime parameter is NOT an environment variable (cons/prim/aux): it comes from the
        # RuntimeParams channel, so nothing to check in check() (like a Const).
        return set()

    def to_cpp(self):
        if self.index < 0:
            raise RuntimeError(
                "RuntimeParamRef('%s'): index not assigned at codegen (call the compilation via "
                "dsl.Model which assigns the runtime indices)" % self.name)
        return "params.get(%d)" % self.index

    def _str(self):
        return "rparam(%s)" % self.name
