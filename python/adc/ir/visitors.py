"""AST traversal helpers for the ADC IR (Spec 4).

Moved verbatim from :mod:`adc.dsl`: :func:`_children` (child iteration over the
expression tree), :func:`_expr_uses_cons_or_prim` (kind check) and :func:`_key`
(structural CSE key). These are pure-symbolic walkers with NO C++ emission; the
C++-emit walkers (``_cpp_expand``, ``_cpp_cse``, ``_cse_emit``, ...) stay in
:mod:`adc.dsl` and import these from here.
"""
from .expr import Abs, Const, Neg, Sign, Sqrt, Var, _Bin
from .values import EigWitness, RuntimeParamRef, StateRef


# --- Common subexpression elimination (CSE) --------------------------------
# The codegen inlines each sub-expression at every occurrence (H, c... recomputed). CSE detects
# the COMPOUND (non-leaf) sub-expressions appearing multiple times and hoists them into local
# variables 'cseK_', in dependency order (the smallest first). Relies on a STRUCTURAL key
# per node: two identical subtrees have the same key, hence the same local.
def _children(e):
    if isinstance(e, _Bin):
        return (e.a, e.b)
    if isinstance(e, (Neg, Sqrt, Abs, Sign)):
        return (e.a,)
    if isinstance(e, EigWitness):
        return tuple(e.entries())  # entrees de la matrice : enfants pour CSE / decouverte deps
    if isinstance(e, StateRef):
        return (e.expr,)  # left/right marker: a single child (discovery of runtime params, etc.)
    return ()


def _expr_uses_cons_or_prim(e):
    """True if the expression tree references a conservative or primitive Var. Tests the Var KIND, so
    the answer does not depend on declaration order. Used to enforce that linear_source coefficients
    are linear in U: a coefficient depending on U or a primitive is not a constant matrix entry."""
    stack = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, Var) and node.kind in ("cons", "prim"):
            return True
        stack.extend(_children(node))
    return False


def _key(e):
    if isinstance(e, Const):
        return ("const", e.value)
    if isinstance(e, RuntimeParamRef):
        return ("rparam", e.name)  # key = name: two refs to the same runtime param share the CSE local
    if isinstance(e, Var):
        return ("var", e.name)
    if isinstance(e, Neg):
        return ("neg", _key(e.a))
    if isinstance(e, Sqrt):
        return ("sqrt", _key(e.a))
    if isinstance(e, Abs):
        return ("abs", _key(e.a))
    if isinstance(e, Sign):
        return ("sign", _key(e.a))
    if isinstance(e, EigWitness):
        # cle = (field, taille, cles des entrees) : deux temoins de la MEME matrice partagent une locale.
        # Un PREDICAT ajoute im_tol a la cle (verdict different a seuil different) ; le chemin scalaire
        # garde sa cle a 4 elements -> CSE et brique bit-identiques a l'historique.
        if e.is_predicate():
            return ("eig", e.field, e.k, e.im_tol, tuple(_key(c) for c in e.entries()))
        return ("eig", e.field, e.k, tuple(_key(c) for c in e.entries()))
    if isinstance(e, StateRef):
        return ("state", e.side, _key(e.expr))  # defensive: the Roe lines do not go through CSE
    return (e.op, tuple(_key(c) for c in _children(e)))  # _Bin (Add/Sub/Mul/Div/Pow)
