"""pops.codegen.cpp_writer : standalone C++-emit helpers extracted from pops.dsl.

These top-level functions translate the symbolic IR into C++ source strings.
They are the sole owners of all C++ emission logic; pops.dsl will eventually
import them from here.  No dependency on pops.dsl (no import cycle).
"""

from pops.ir.expr import Const, Var, _Bin, Neg, Sqrt, Abs, Sign, Pow, Div, Mul
from pops.ir.values import EigWitness, StateRef, RuntimeParamRef, _EIG_FIELDS, _EIG_PREDICATES
from pops.ir.visitors import _key, _children
from pops.ir.expr import _wrap


def _cpp_expand(e, cse_map):
    """C++ of node e expanding ITS level; the children go through _cpp_cse (-> CSE locals)."""
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()  # params.get(<index>): reads the brick's RuntimeParams member
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sign):
        s = _cpp_cse(e.a, cse_map)  # l'enfant peut etre une locale CSE : evalue UNE fois
        return "(pops::Real(%s > 0) - pops::Real(%s < 0))" % (s, s)
    if isinstance(e, EigWitness):
        # appel du foncteur nomme (declare dans la brique) : chaque entree passee en argument scalaire
        # (via _cpp_cse -> partage les locales CSE, evaluee une seule fois cote appelant). Un predicat
        # ajoute le seuil im_tol en dernier argument (cf. _extra_args_cpp / _eig_witness_helpers).
        args = [_cpp_cse(c, cse_map) for c in e.entries()] + e._extra_args_cpp()
        return "%s(%s)" % (e.helper_name(), ", ".join(args))
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_cse(e.a, cse_map), _cpp_cse(e.b, cse_map))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_cse(e.a, cse_map), e.op, _cpp_cse(e.b, cse_map))
    raise TypeError("expression not handled by the codegen: %r" % (e,))


def _cpp_cse(e, cse_map):
    """C++ of e; if e matches an already-defined CSE local, returns its name."""
    k = _key(e)
    if k in cse_map:
        return cse_map[k]
    return _cpp_expand(e, cse_map)


def _cse_emit(roots, real, indent):
    """Return (local_declaration_lines, [C++ per root]). Compound subexpressions seen >= 2 times
    become ``cseK_`` locals. roots: list of Expr.

    Memo by id(): a shared Expr OBJECT (DAG, e.g. intermediates reused from a large model)
    is traversed only once; its later occurrences re-credit the counter of its
    subtree without re-walking. Without the memo the traversal costs O(number of root-to-leaf PATHS),
    exponential on a deep DAG (polynomial flux of polynomials). Strictly equivalent to the
    historical re-walk: same counts, same sizes, same key INSERTION ORDER
    (post-order of the first visit) -> emitted C++ is bit-identical."""
    counts, rep, size = {}, {}, {}
    memo = {}  # id(e) -> (size, {key: occurrences} of the subtree, in post-order of insertion)

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
            cnt[k] = cnt.get(k, 0) + 1  # post-order: children first, like the historical re-walk
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


# --- Foncteurs nommes des temoins de valeurs propres (EigWitness, ADC-289) ---
# Le codegen d'EigWitness emet un APPEL (to_cpp) vers une methode statique POPS_HD de la brique qui
# remplit un pops::Real M[k][k] puis appelle pops::real_eig_minmax. Cette methode est un FONCTEUR NOMME
# (device-clean : pas de lambda etendue cross-TU qui casse nvcc) declare une fois par couple (field, k)
# rencontre dans les formules de la brique. Ces deux fonctions decouvrent les couples et emettent les
# declarations.
def _collect_eig_witnesses(exprs):
    """Couples (field, k) des EigWitness presents dans @p exprs, dedupliques, en ordre stable
    (tri par (k, field)). Vide si aucun temoin -> aucune declaration, brique bit-identique a l'histoire."""
    seen = set()
    memo = set()  # id() : DAG -> chaque objet Expr visite une fois

    def walk(e):
        if id(e) in memo:
            return
        memo.add(id(e))
        if isinstance(e, EigWitness):
            seen.add((e.field, e.k))
        for c in _children(e):
            walk(c)

    for e in exprs:
        walk(_wrap(e))
    return sorted(seen, key=lambda fk: (fk[1], fk[0]))


def _eig_witness_helpers(pairs, indent="  "):
    """Lignes C++ des foncteurs nommes (methodes statiques POPS_HD) pour les couples (field, k) de
    @p pairs (cf. _collect_eig_witnesses). Chaque foncteur prend les k*k entrees de la matrice en
    arguments scalaires (ordre ligne-major), remplit pops::Real M[k][k] et renvoie le champ @c field de
    ``pops::real_eig_minmax(M)``. Aucun argument -> aucune ligne."""
    L = []
    for field, k in pairs:
        is_pred = field in _EIG_PREDICATES
        params = ", ".join("pops::Real m%d" % i for i in range(k * k))
        if is_pred:  # un predicat prend le seuil relatif im_tol en dernier argument scalaire
            params += (", " if params else "") + "pops::Real im_tol"
        L.append("%sstatic POPS_HD pops::Real pops_eig_%s_%dx%d(%s) {"
                 % (indent, field, k, k, params))
        L.append("%s  pops::Real M[%d][%d];" % (indent, k, k))
        for r in range(k):
            sets = " ".join("M[%d][%d] = m%d;" % (r, c, r * k + c) for c in range(k))
            L.append("%s  %s" % (indent, sets))
        if is_pred:
            # predicat verrouille sur converged (un repli Gershgorin -> false -> 0.0, jamais reel) ;
            # cast bool -> pops::Real (1.0/0.0) pour composer dans les masques branchless de m.projection.
            L.append("%s  return pops::Real(pops::real_eig_minmax(M).%s(im_tol));"
                     % (indent, _EIG_PREDICATES[field]))
        else:
            L.append("%s  return pops::real_eig_minmax(M).%s;" % (indent, _EIG_FIELDS[field]))
        L.append("%s}" % indent)
    return L


# --- Reciprocal hoist (OPT-IN) -----------------------------------------------
# Without -ffast-math, the compiler cannot replace N divisions by the same denominator
# with 1 reciprocal + N multiplications (rounding would change). We do it at codegen, opt-in:
# for a recurring conservative denominator (>= 2 uses in the live primitives of a
# method), emit once inv_<name> = 1 / <name> and replace its divisions by products.
# Restricted to CONSERVATIVE VARIABLE denominators: computable right after the cons locals,
# before the primitives. OPT-IN because rounding changes (not bit-identical to the default output).
def _count_cons_denoms(e, cons_set, counts):
    """Collects the Var-conservative denominators of @p e into @p counts (name -> occurrences)."""
    if isinstance(e, Div) and isinstance(e.b, Var) and e.b.name in cons_set:
        counts[e.b.name] = counts.get(e.b.name, 0) + 1
    for c in _children(e):
        _count_cons_denoms(c, cons_set, counts)


def _recip_rewrite(e, inv_set):
    """Rewrites @p e: any division by a conservative Var of @p inv_set becomes a product by
    its hoisted reciprocal inv_<name>. Rebuilds NEW nodes (does not mutate the model)."""
    if isinstance(e, Div):
        a = _recip_rewrite(e.a, inv_set)
        if isinstance(e.b, Var) and e.b.name in inv_set:
            return Mul(a, Var("inv_" + e.b.name, "hoist"))
        return Div(a, _recip_rewrite(e.b, inv_set))
    if isinstance(e, _Bin):
        return type(e)(_recip_rewrite(e.a, inv_set), _recip_rewrite(e.b, inv_set))
    if isinstance(e, Neg):
        return Neg(_recip_rewrite(e.a, inv_set))
    if isinstance(e, Sqrt):
        return Sqrt(_recip_rewrite(e.a, inv_set))
    if isinstance(e, Abs):
        return Abs(_recip_rewrite(e.a, inv_set))
    if isinstance(e, StateRef):
        return StateRef(e.side, _recip_rewrite(e.expr, inv_set))
    return e


def _dir_key(direction):
    """Normalize a direction into 'x' / 'y' (accepts 0/'x'/'X' and 1/'y'/'Y'). Raises otherwise."""
    if direction in (0, "x", "X"):
        return "x"
    if direction in (1, "y", "Y"):
        return "y"
    raise ValueError("invalid direction %r (expected 0/'x' or 1/'y')" % (direction,))


# --- Roe dissipation PROVIDED by the user (m.roe_dissipation) ---------
# The lines d_i are written in terms of left(...)/right(...) of variables/primitives + constants.
# _roe_validate checks that no variable appears OUTSIDE a marker (undetermined state) and that no
# marker is nested; _cpp_roe renders the C++ by resolving left/right through a local prefix
# (L_ for UL, R_ for UR).
def _roe_validate(e, in_marker):
    """Structural check of a roe_dissipation line. Raises ValueError if a variable is outside a
    left()/right() marker (undetermined state) or if a marker is nested. Const / runtime
    parameter: allowed everywhere (without state). Evaluates nothing (usable before the assignment of
    the runtime indices)."""
    if isinstance(e, StateRef):
        if in_marker:
            raise ValueError("m.roe_dissipation: nested left()/right() marker forbidden "
                             "(a subexpression belongs to a single state)")
        _roe_validate(e.expr, True)
        return
    if isinstance(e, Var):
        if not in_marker:
            raise ValueError(
                "m.roe_dissipation: variable '%s' outside marker; wrap each variable or "
                "primitive with dsl.left(...) (state UL) or dsl.right(...) (state UR)" % e.name)
        return
    if isinstance(e, (Const, RuntimeParamRef)):
        return
    for c in _children(e):
        _roe_validate(c, in_marker)


def _cpp_roe(e, prefix):
    """C++ of a roe_dissipation line expression. @p prefix: None at the root level (no active
    state -> a bare variable is an error), 'L_' / 'R_' inside a marker (the variables take
    that local prefix). ALSO used to render a primitive definition with a state prefix
    (e then contains no StateRef). Assumes _roe_validate already passed (defensive errors)."""
    if isinstance(e, StateRef):
        if prefix is not None:
            raise ValueError("m.roe_dissipation: nested left()/right() marker forbidden")
        return _cpp_roe(e.expr, e.side + "_")
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()
    if isinstance(e, Var):
        if prefix is None:
            raise ValueError("m.roe_dissipation: variable '%s' outside left()/right() marker"
                             % e.name)
        return prefix + e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Sign):
        s = _cpp_roe(e.a, prefix)
        return "(pops::Real(%s > 0) - pops::Real(%s < 0))" % (s, s)
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_roe(e.a, prefix), _cpp_roe(e.b, prefix))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_roe(e.a, prefix), e.op, _cpp_roe(e.b, prefix))
    raise TypeError("m.roe_dissipation: expression not handled by the codegen: %r" % (e,))
