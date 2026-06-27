"""pops.codegen.program_codegen -- C++ emission for a pops.time.Program.

FREE FUNCTIONS taking the ``program`` (and an optional physical ``model``), mirroring
``pops.codegen.module_codegen`` for models. ``emit_cpp_program(program, model=None)`` lowers
the Program SSA IR to the C++ source of a ``problem.so`` (the stable .so ABI installed by
``System::install_program``); the ``_emit_*`` / ``_check_*`` helpers and the per-cell kernel
emitters are the lowering machinery. ``pops.time.Program.emit_cpp_program`` is a thin
delegator into this module (lazy import), keeping ``pops.time`` free of any codegen edge.

The program-class op allow-list / kernel tables this lowering keys on live on the program
instance (``program._ALLOWED_OPS`` etc., from ``pops.time.program_base._ProgramConstants``);
the emission-only tables (_MODEL_OPS / _ALLOWED_OPS / _PROFILE_SKIP_OPS / _AUX_OUTPUT_OPS) are
module-level constants here.
"""
import hashlib
import json

# The affine-combination carriers live with the value algebra; codegen may import pops.time
# (the acyclic graph allows codegen -> time). `dsl` is imported LAZILY inside the helpers that
# need it (the per-cell kernel emitters) to avoid a heavy import at module load.
from pops.time.values import Value, _to_affine  # noqa: F401

# Emission-only op tables (formerly Program class constants; the lowering owns them).
# Ops the Phase-4b codegen lowers ONLY when a physical model is supplied (they read the model's
# symbolic source_term / linear_source coefficients). Without a model they raise NotImplementedError.
_MODEL_OPS = ("source", "apply", "solve_local_linear", "solve_local_nonlinear")

_ALLOWED_OPS = frozenset({"state", "solve_fields", "solve_fields_from_blocks", "rhs",
                          "linear_combine", "linear_source",
                          "reduce", "compare", "while", "range", "if", "matrix_free_operator",
                          "scalar_field", "laplacian", "gradient", "divergence", "solve_linear",
                          "apply_in", "apply_out", "history", "store_history",
                          "fill_boundary", "project", "record_scalar",
                          "cell_compare", "where", "rhs_jacvec",
                          "schur_coeffs", "apply_laplacian_coeff", "schur_explicit_flux",
                          "schur_rhs", "schur_reconstruct", "schur_energy",
                          "coupled_rate", "coupled_rate_out"})

_PROFILE_SKIP_OPS = frozenset({"state", "history", "hmin", "cfl"})

_AUX_OUTPUT_OPS = frozenset({"solve_fields", "solve_fields_from_blocks"})


# --- module-level emission helpers (per-cell kernels, coeff rendering, the .so template) ---
def _deref(tok):
    """C++ MultiFab-lvalue argument for a top-level (step-body) field token. Every top-level token is
    already a MultiFab lvalue expression: a state / RHS scratch (``u5``, ``r5``), a history (``h5``) or
    a dereferenced scratch scalar field (``(*sf5)``). The step-body laplacian / gradient / divergence /
    schur ops take ``pops::MultiFab&`` arguments, so the token passes through unchanged (the apply-block
    counterpart is `_apply_in_arg`, which additionally const_casts the lambda's ``in`` param)."""
    return tok


def _apply_in_arg(sub, value):
    """C++ argument for the INPUT field of a laplacian / gradient inside an apply lambda. When the input
    is the lambda's ``in`` (a const&), const_cast it (ctx.laplacian / gradient take a non-const MultiFab&
    and only write the ghosts, never the valid cells -- the same contract test_generic_krylov relies on);
    a persistent scratch shared_ptr is dereferenced."""
    tok = sub[value.id]
    if tok == "in":
        return "const_cast<pops::MultiFab&>(in)"
    return "*%s" % tok


def _emit_field_combine(result, target, sub, acc):
    """Emit C++ writing the affine combination @p result into the field @p target (a C++ MultiFab token,
    e.g. ``out``). Mirrors the linear_combine commit: zero the PERSISTENT accumulator @p acc (a scratch
    shared_ptr allocated once at install time -- no per-call/per-iteration allocation), accumulate the
    non-`target` terms onto it, then ``ctx.lincomb(target, c_target, target, 1, *acc)``. A single unit
    term that already is the target is a no-op. @p sub maps IR value ids to C++ tokens (``in``/``out``/
    scratch shared_ptrs); @p acc is the install-time accumulator shared_ptr name. Zeroing via
    ``set_val(0)`` reproduces the old ``scratch_state_like`` (a zero-initialized scratch) bit-for-bit
    over the valid cells the axpy / lincomb touch."""
    aff = _to_affine(result)._merge()
    terms = [(v, c.as_dict()) for v, c in aff]
    lines = ["%s->set_val(static_cast<pops::Real>(0));" % acc]
    c_target = {0: 0.0}
    for value, coeff in terms:
        tok = sub[value.id]
        ref = "const_cast<pops::MultiFab&>(in)" if tok == "in" else (
            "*%s" % tok if tok.startswith("sf") else tok)
        if tok == target:
            c_target = coeff
        else:
            lines.append("ctx.axpy(*%s, %s, %s);" % (acc, _coeff_cpp(coeff), ref))
    lines.append("ctx.lincomb(%s, %s, %s, static_cast<pops::Real>(1), *%s);"
                 % (target, _coeff_cpp(c_target), target, acc))
    return lines


def _coeff_cpp(powers):
    """Render a dt-polynomial coefficient (``power -> float`` dict) as a C++ ``pops::Real`` expression
    in the closure's ``dt`` parameter: ``{1: 1.0}`` -> ``static_cast<pops::Real>(dt)``,
    ``{1: 0.5}`` -> ``static_cast<pops::Real>(0.5 * dt)``, ``{0: 2.0}`` ->
    ``static_cast<pops::Real>(2.0)``. Drops a unit factor and a zero polynomial collapses to 0."""
    if not powers:
        return "static_cast<pops::Real>(0)"
    terms = []
    for power, coeff in sorted(powers.items()):
        factors = ["dt"] * int(power)
        if float(coeff) != 1.0 or not factors:
            factors = [repr(float(coeff))] + factors
        terms.append(" * ".join(factors))
    return "static_cast<pops::Real>(%s)" % " + ".join(terms)


# --- Phase-4b: lower a model's split-source / local-linear ops to per-cell C++ kernels ----------
# These helpers emit the body of a for_each_cell kernel over the VALID cells of each local fab. They
# reuse the dsl Expr -> C++ machinery (Var.to_cpp returns the bare name; we bind those names to locals)
# and the existing numerics (pops::detail::mat_inverse). A device kernel must stay heap-free /
# allocation-free: only stack scalars + fixed-size arrays, no std::vector / std::function / Eigen.

def _model_impl(model):
    """The underlying HyperbolicModel carrying the symbolic coefficients: the public pops.dsl.Model
    wraps it as ``_m``; a HyperbolicModel is already itself."""
    return getattr(model, "_m", model)


def _named_fluxes(v):
    """Resolve a ``rhs`` op's ``fluxes`` attr to the list of NAMED fluxes to assemble (ADC-419), or
    ``None`` for the historical default flux path (``ctx.rhs_into`` -- byte-identical -div F). ``None``
    or ``["default"]`` -> default path; a list of named fluxes -> that list. Mixing ``"default"`` with
    named fluxes is rejected (the centered-FV named-flux stencil differs from the Riemann rhs_into
    stencil, so they cannot be summed)."""
    fluxes = v.attrs.get("fluxes")
    if not fluxes or fluxes == ["default"]:
        return None
    named = [f for f in fluxes if f != "default"]
    if len(named) != len(fluxes):
        raise ValueError(
            "rhs '%s': fluxes mixes 'default' with named fluxes %r; request either the default flux "
            "(-div F via rhs_into) or a set of named fluxes (their -div sum), not both" % (v.name, named))
    return named


def _aux_comp(impl, name):
    """Component index of an aux field @p name in the System aux channel: canonical (dsl.AUX_CANONICAL)
    or a model NAMED aux field (dsl.AUX_NAMED_BASE + position in aux_extra_names). @p impl is the
    HyperbolicModel."""
    from pops.physics.aux import AUX_CANONICAL, AUX_NAMED_BASE
    if name in AUX_CANONICAL:
        return AUX_CANONICAL[name]
    extra = list(getattr(impl, "aux_extra_names", []) or [])
    if name in extra:
        return AUX_NAMED_BASE + extra.index(name)
    raise NotImplementedError(
        "emit_cpp_program: aux field '%s' is neither canonical (%s) nor a declared named aux field "
        "(%s); cannot map it to an aux component" % (name, sorted(AUX_CANONICAL), extra))


def _check_no_runtime_param(exprs):
    """Phase-4b kernels read coefficients from the state / aux only (const params are inlined as
    literals by the dsl Expr tree). A RUNTIME parameter would emit ``params.get(idx)``, unavailable in
    a ProgramContext kernel -> raise NotImplementedError (deferred), never a .so that fails to link."""
    from pops.ir.values import RuntimeParamRef
    from pops.ir.visitors import _children
    stack = list(exprs)
    while stack:
        e = stack.pop()
        if isinstance(e, RuntimeParamRef):
            raise NotImplementedError(
                "emit_cpp_program: a Phase-4b source / linear source references a RUNTIME parameter "
                "(%s); only constants and aux fields are supported in the per-cell kernel yet "
                "(runtime params in compiled programs are a later phase)" % e.name)
        stack.extend(_children(e))


def _cell_locals(impl, exprs, state_var, *, with_cons, with_prim):
    """C++ local declarations binding the names the @p exprs reference to per-cell values:
      - aux fields -> ``const pops::Real <name> = auxA(i, j, <comp>);`` (always, by dependency);
      - conservative vars -> ``const pops::Real <name> = <state>A(i, j, <idx>);`` (when @p with_cons);
      - primitives -> their dsl formula, in declaration order, only the LIVE ones (when @p with_prim).
    @p impl is the HyperbolicModel; @p state_var the C++ MultiFab variable (its const Array4 is
    ``<state_var>A``, the aux Array4 is ``auxA``). Raises on a runtime-param dependency (deferred)."""
    _check_no_runtime_param(exprs)
    deps = set()
    for e in exprs:
        deps |= e.deps()
    lines = []
    live = impl._live_prims(exprs) if with_prim else set()
    # A live primitive's formula (e.g. u = mx / rho) references conservative variables that the top
    # expressions may not name directly: bind those TRANSITIVE cons too, else the emitted prim line
    # references an undeclared local. (Existing source/apply kernels read cons directly, so this only
    # ADDS the cons a live prim pulls in -- it never drops one that was already bound.)
    cons_needed = set(deps)
    for p in live:
        cons_needed |= {d for d in impl.prim_defs[p].deps() if d in impl.cons_names}
    if with_cons:
        for idx, c in enumerate(impl.cons_names):
            if c in cons_needed:
                lines.append("const pops::Real %s = %sA(i, j, %d);" % (c, state_var, idx))
    if with_prim:
        for p, expr in impl.prim_defs.items():  # declaration order (a prim may use an earlier prim)
            if p in live:
                lines.append("const pops::Real %s = %s;" % (p, expr.to_cpp()))
    aux_deps = set(impl.aux_names) | set(getattr(impl, "aux_extra_names", []) or [])
    for name in sorted(deps & aux_deps):
        lines.append("const pops::Real %s = auxA(i, j, %d);" % (name, _aux_comp(impl, name)))
    return lines


def _kernel_open(out_var, state_var):
    """Open the per-fab loop + per-cell for_each_cell over the VALID cells of @p out_var, binding the
    write handle ``outA``, the read state handle ``<state_var>A`` and the aux read handle ``auxA``.

    Pairing by local fab index ``li`` is sound: the System aux is built with the SAME box array AND
    distribution map as the blocks (``aux(ba, dm, ...)`` in System::Impl), and a scratch state comes
    from ``scratch_state_like(state(0))`` which copies that ``(ba, dm)`` -- so ``out``, the input
    state and ``aux`` share one ``(ba, dm)`` and ``fab(li)`` is the same box on every rank. This is the
    same co-distribution the existing aux-reading kernels (compiled_block_abi / source bricks) rely on."""
    return [
        "pops::MultiFab& %s_aux = ctx.aux();" % out_var,
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const pops::Array4 outA = %s.fab(li).array();" % out_var,
        "  const pops::ConstArray4 %sA = %s.fab(li).const_array();" % (state_var, state_var),
        "  const pops::ConstArray4 auxA = %s_aux.fab(li).const_array();" % out_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % out_var,
    ]


def _kernel_close():
    return ["  });", "}"]


# --- per-cell conditional select (spec op 17, ADC-418): model-free for_each_cell kernels --------------
# `cell_compare` and `where` are pure layout ops over co-distributed MultiFabs (no aux / no model
# coefficients): they reuse the SAME for_each_cell + Array4 per-fab pattern as the source kernels, but
# bind several read handles (no auxA) and loop over the runtime component count `<out>.ncomp()`. Pairing
# by local fab index li is sound: a cell_compare mask is alloc_scalar_field (the System (ba, dm)), a
# where scratch is scratch_state_like(a) (a's (ba, dm)) and the inputs are the same co-distributed
# states / scalar_fields, so fab(li) is the same box on every rank.


def _emit_cell_compare_kernel(field_var, mask_var, cmp, value):
    """Lower ``cell_compare``: maskA(i,j,0) = fieldA(i,j,0) <cmp> value ? 1 : 0 over the valid cells of
    the 1-component mask. Reads component 0 of @p field_var; writes the 0/1 mask into @p mask_var."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % mask_var,
        "  const pops::Array4 maskA = %s.fab(li).array();" % mask_var,
        "  const pops::ConstArray4 fieldA = %s.fab(li).const_array();" % field_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % mask_var,
        "    maskA(i, j, 0) = (fieldA(i, j, 0) %s static_cast<pops::Real>(%s)) "
        "? static_cast<pops::Real>(1) : static_cast<pops::Real>(0);" % (cmp, repr(float(value))),
        "  });",
        "}",
    ]


def _emit_where_kernel(mask_var, a_var, b_var, out_var):
    """Lower ``where``: outA(i,j,c) = maskA(i,j,mc) != 0 ? aA(i,j,c) : bA(i,j,c) COMPONENT-WISE over the
    valid cells of @p out_var (out's runtime ncomp). The mask component mc is 0 when the mask is
    1-component (a shared mask) and c when the mask has the SAME ncomp as a/b (a per-component mask) --
    decided per cell from the mask's own ncomp, so both layouts lower with ONE kernel."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const pops::Array4 outA = %s.fab(li).array();" % out_var,
        "  const pops::ConstArray4 maskA = %s.fab(li).const_array();" % mask_var,
        "  const pops::ConstArray4 aA = %s.fab(li).const_array();" % a_var,
        "  const pops::ConstArray4 bA = %s.fab(li).const_array();" % b_var,
        "  const int ncomp_ = %s.ncomp();" % out_var,
        "  const int mask_ncomp_ = %s.ncomp();" % mask_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % out_var,
        "    for (int c = 0; c < ncomp_; ++c) {",
        "      const int mc = (mask_ncomp_ == 1) ? 0 : c;",
        "      outA(i, j, c) = (maskA(i, j, mc) != static_cast<pops::Real>(0)) ? aA(i, j, c) "
        ": bA(i, j, c);",
        "    }",
        "  });",
        "}",
    ]


def _emit_source_kernel(model, name, state_var, out_var):
    """Lower ``source`` (a named ``m.source_term``): outA(i,j,c) = S_c(U, prims, aux) per cell."""
    impl = _model_impl(model)
    if name not in impl._source_terms:
        raise NotImplementedError(
            "emit_cpp_program: source '%s' is not declared on the model (m.source_term); declared: %s"
            % (name, sorted(impl._source_terms)))
    exprs = impl._source_terms[name]
    body = _kernel_open(out_var, state_var)
    body += ["    " + ln for ln in _cell_locals(impl, exprs, state_var, with_cons=True,
                                                 with_prim=True)]
    body += ["    outA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(exprs)]
    body += _kernel_close()
    return body


def _emit_coupled_rate_kernel(components, by_block, var, scratch):
    """Lower a ``coupled_rate`` (Spec 3 criterion 27, ADC-457) to ONE multi-state for_each_cell kernel
    filling every participating block's rate scratch at once.

    @p components: ``{block: [Expr, ...]}`` -- the per-block component formulas (cons-only MVP).
    @p by_block:   ``{block: state Value}`` -- each block's input state (its StateSpace gives the cons
                   names + their component indices; its C++ token gives the read Array4).
    @p var:        the id -> C++ token map (the input states are already bound to ``ctx.state(idx)``).
    @p scratch:    ``{block: scratch var name}`` -- the per-block rate scratch (alloc'd by the caller).

    The component formulas reference cons vars from MULTIPLE input states, so the blocks share ONE loop
    (they cannot be independent single-block rates). The first block drives the loop; all inputs and
    scratches are co-located (same ba/dm as the System aux), so ``fab(li)`` is the same box on every
    rank -- the co-distribution every aux-reading kernel relies on (see _kernel_open). Each input
    state binds its OWN read handle (``<state token>A``); a referenced cons var binds from its state's
    Array4 at its component index. A cons NAME shared by two states' components AND referenced by a
    formula is ambiguous (no single source) -- rejected loud, never silently bound to one state."""
    blocks = list(components)
    driver = scratch[blocks[0]]                  # the block whose box / local_size drives the loop
    # Which cons vars does any formula reference, and from which state does each come?
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    cons_source = {}                             # cons name -> (state token, component index)
    for st in by_block.values():                 # ALL input states, incl. read-only catalysts
        if getattr(st, "space", None) is None:
            continue
        for idx, c in enumerate(st.space.components):
            if c not in referenced:
                continue
            if c in cons_source and cons_source[c] != (var[st.id], idx):
                raise NotImplementedError(
                    "coupled_rate kernel codegen (ADC-457): cons var %r is a component of more than "
                    "one input state; the cons-only MVP needs disjoint component names across the "
                    "coupled states (rename one of them)" % (c,))
            cons_source[c] = (var[st.id], idx)

    def state_handle(token):
        return "%sA" % token                     # read handle for an input state token (u0A / u1A)

    lines = ["for (int li = 0; li < %s.local_size(); ++li) {" % driver]
    # Bind a write handle per OUTPUT block scratch, then a read handle per DISTINCT input state that a
    # formula actually reads (incl. a read-only catalyst input that is not an output block), all inside
    # the per-fab loop and BEFORE for_each_cell so the device lambda captures them by value.
    for blk in blocks:
        lines.append("  const pops::Array4 %sA = %s.fab(li).array();" % (scratch[blk], scratch[blk]))
    read_tokens = {src[0] for src in cons_source.values()}
    seen_states = []
    for st in by_block.values():                 # input order (v.inputs); deterministic
        tok = var[st.id]
        if tok in read_tokens and tok not in seen_states:
            seen_states.append(tok)
            lines.append("  const pops::ConstArray4 %s = %s.fab(li).const_array();"
                         % (state_handle(tok), tok))
    lines.append("  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % driver)
    for c in sorted(cons_source):                # bind only the referenced cons (no unused locals)
        tok, idx = cons_source[c]
        lines.append("    const pops::Real %s = %s(i, j, %d);" % (c, state_handle(tok), idx))
    for blk in blocks:
        for comp, e in enumerate(components[blk]):
            lines.append("    %sA(i, j, %d) = %s;" % (scratch[blk], comp, e.to_cpp()))
    lines += ["  });", "}"]
    return lines


def _emit_flux_kernel(model, names, state_var, fx_var, fy_var):
    """Lower NAMED fluxes (ADC-419): fxA(i,j,c) = sum_k F^k_x[c](U, prims, aux),
    fyA(i,j,c) = sum_k F^k_y[c](U, prims, aux) over the selected named fluxes @p names. ONE kernel
    evaluates the SUM per direction into the two n_cons flux fields (the subsequent neg_div_flux_into
    takes -div). Reuses the same per-cell local machinery as the source kernel (cons/prim/aux locals)."""
    impl = _model_impl(model)
    flux_terms = impl._flux_terms
    for name in names:
        if name not in flux_terms:
            raise NotImplementedError(
                "emit_cpp_program: flux '%s' is not declared on the model (m.flux_term); declared: %s"
                % (name, sorted(flux_terms)))
    n = len(impl.cons_names)
    x_exprs = [flux_terms[names[0]]["x"][c] for c in range(n)]
    y_exprs = [flux_terms[names[0]]["y"][c] for c in range(n)]
    for name in names[1:]:  # accumulate the additional named fluxes (their SUM is one -div)
        x_exprs = [x_exprs[c] + flux_terms[name]["x"][c] for c in range(n)]
        y_exprs = [y_exprs[c] + flux_terms[name]["y"][c] for c in range(n)]
    body = _kernel_open(fx_var, state_var)
    # fx and fy share the (ba, dm) of the scratch state, so the SAME loop / handles write both: bind a
    # second write handle to fy's local fab right after _kernel_open's outA (= fxA), still INSIDE the
    # per-fab loop and BEFORE for_each_cell so the device lambda captures it.
    body.insert(3, "  const pops::Array4 fyA = %s.fab(li).array();" % fy_var)
    body += ["    " + ln for ln in _cell_locals(impl, x_exprs + y_exprs, state_var, with_cons=True,
                                                 with_prim=True)]
    body += ["    outA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(x_exprs)]
    body += ["    fyA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(y_exprs)]
    body += _kernel_close()
    return body


def _emit_apply_kernel(model, name, state_var, out_var):
    """Lower ``apply`` (a named ``m.linear_source`` L): outA(i,j,r) = sum_c L[r][c](aux) * U(i,j,c)."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    body = _kernel_open(out_var, state_var)
    # L coefficients depend on aux / const only (linear_source invariant): cons/prim locals not needed.
    body += ["    " + ln for ln in _cell_locals(impl, flat, state_var, with_cons=False,
                                                 with_prim=False)]
    for r in range(n):
        terms = ["(%s) * %sA(i, j, %d)" % (rows[r][c].to_cpp(), state_var, c) for c in range(n)]
        body.append("    outA(i, j, %d) = %s;" % (r, " + ".join(terms)))
    body += _kernel_close()
    return body


def _emit_solve_local_linear_kernel(model, name, a_coeff, rhs_var, out_var):
    """Lower ``solve_local_linear``: per cell M = I - a*L (a = a_coeff(dt)), invert M (dense N x N
    via pops::detail::mat_inverse) and set outA(i,j,r) = sum_c Minv[r][c] * q(i,j,c), q = the rhs state.
    L's coefficients depend on aux / const only, so M is assembled from the aux locals + the literal a."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    a_cpp = _coeff_cpp(a_coeff)
    body = _kernel_open(out_var, rhs_var)
    body += ["    " + ln for ln in _cell_locals(impl, flat, rhs_var, with_cons=False,
                                                 with_prim=False)]
    body.append("    const pops::Real a_ = %s;" % a_cpp)
    body.append("    pops::Real M_[%d][%d];" % (n, n))
    for r in range(n):
        for c in range(n):
            ident = "pops::Real(1)" if r == c else "pops::Real(0)"
            body.append("    M_[%d][%d] = %s - a_ * (%s);" % (r, c, ident, rows[r][c].to_cpp()))
    body.append("    pops::Real Minv_[%d][%d];" % (n, n))
    # mat_inverse returns false on a singular M; we do not branch in the device kernel (no throw on
    # device). I - a*L is invertible for a well-posed local source (e.g. Lorentz: det = 1 + (a*B)^2 > 0);
    # a singular user operator yields a non-finite result that surfaces downstream, not a plausible wrong one.
    body.append("    pops::detail::mat_inverse<%d>(M_, Minv_);" % n)
    for r in range(n):
        terms = ["Minv_[%d][%d] * %sA(i, j, %d)" % (r, c, rhs_var, c) for c in range(n)]
        body.append("    outA(i, j, %d) = %s;" % (r, " + ".join(terms)))
    body += _kernel_close()
    return body


def _residual_term_exprs(impl, w):
    """The per-component Expr list of one LOCAL residual sub-block op @p w, as a function of the bare
    conservative-variable names (which the Newton kernel binds to the iterate stack ``Ueval[c]``):

      - ``source`` (a named ``m.source_term``): S_c(U) -- the declared source expressions;
      - ``apply`` (a named ``m.linear_source`` L): (L U)_c = sum_k L[c][k] * <cons_k>.

    The iterate / guess State placeholders and ``linear_combine`` are handled by the affine walk in
    `_emit_residual_eval`, not here (they are not standalone-evaluable Exprs)."""
    from pops.ir.expr import Const, Var
    if w.op == "source":
        name = w.attrs["source"]
        if name not in impl._source_terms:
            raise NotImplementedError(
                "emit_cpp_program: residual source '%s' is not declared on the model (m.source_term); "
                "declared: %s" % (name, sorted(impl._source_terms)))
        return list(impl._source_terms[name])
    if w.op == "apply":
        rows = _linear_source_rows(impl, w.attrs["linear_source"])
        n = len(rows)
        # (L U)_r = sum_c L[r][c] * cons_c -- a per-component Expr in the cons names + aux.
        return [sum((rows[r][c] * Var(impl.cons_names[c], "cons") for c in range(n)),
                    Const(0.0)) for r in range(n)]
    raise NotImplementedError(
        "emit_cpp_program: residual op '%s' is not a per-cell Expr term (source / apply only)" % w.op)


def _emit_residual_eval(impl, v, n):
    """Build the device residual-evaluation lambda body for ``solve_local_nonlinear``: lines computing
    ``rout[0..n-1] = r(Ueval)`` from the iterate stack ``Ueval`` (bound to the conservative names), the
    frozen guess stack ``Gval`` (the initial-guess State, read as a per-cell constant) and the captured
    aux locals. Mirrors the affine walk: each residual sub-block op is one of the iterate / guess State
    placeholders, a ``source`` / ``apply`` per-cell Expr term, or a ``linear_combine`` (an affine over
    earlier terms). The result is the affine the residual returned.

    @p v is the solve_local_nonlinear op; @p n the conservative count. Returns the lambda BODY lines
    (indented two spaces past the lambda header). The lambda captures the aux locals + ``Gval`` by ref."""
    block = v.attrs["residual_block"]
    iterate_id = v.attrs["iterate"].id
    guess_id = v.attrs["guess"].id
    # term id -> a list of n C++ expression strings (one per conservative component). The iterate is the
    # stack Ueval; the guess is the frozen Gval; source / apply lower to Exprs over the cons names.
    comps = {iterate_id: ["Ueval[%d]" % c for c in range(n)],
             guess_id: ["Gval[%d]" % c for c in range(n)]}
    lines = []
    for w in block:
        if w.op == "state":
            continue  # the iterate / guess placeholders: bound in `comps` above, nothing to emit
        if w.op in ("source", "apply"):
            exprs = _residual_term_exprs(impl, w)
            comps[w.id] = ["(%s)" % e.to_cpp() for e in exprs]
        elif w.op == "linear_combine":
            # An affine sum over earlier terms: comps[w] = sum_k coeff_k(dt) * comps[input_k].
            coeffs = w.attrs["coeffs"]  # aligned with w.inputs; each a dt-polynomial power->float dict
            for inp in w.inputs:
                if inp.id not in comps:  # an input outside the residual sub-block (validate() guards this)
                    raise NotImplementedError(
                        "emit_cpp_program: residual combine reads value '%s' which is not produced "
                        "inside the residual (only the iterate / guess and earlier residual ops are "
                        "available to a per-cell Newton kernel)" % inp.name)
            row = []
            for c in range(n):
                parts = []
                for inp, coeff in zip(w.inputs, coeffs, strict=True):
                    parts.append("%s * (%s)" % (_coeff_cpp(coeff), comps[inp.id][c]))
                row.append(" + ".join(parts) if parts else "static_cast<pops::Real>(0)")
            comps[w.id] = row
        else:  # builder guards _RESIDUAL_LOCAL_OPS; this is belt-and-suspenders
            raise NotImplementedError(
                "emit_cpp_program: residual op '%s' is not lowerable in a local Newton kernel" % w.op)
    result = comps[v.attrs["residual"].id]
    for c in range(n):
        lines.append("rout[%d] = %s;" % (c, result[c]))
    return lines


def _emit_solve_local_nonlinear_kernel(model, v, guess_var, out_var):
    """Lower ``solve_local_nonlinear`` (spec op 10) to a per-cell Newton kernel: from the initial guess
    U0 (the @p guess_var state), iterate ``J dU = -r``, ``U -= dU`` until ``max_c |r_c| < tol`` or the
    fixed budget, then write the converged U into @p out_var. Reuses ``pops::for_each_cell`` + the SAME
    stack dense inverse ``pops::detail::mat_inverse<N>`` as `solve_local_linear` -- no heap / std::function
    / Eigen in the device kernel (only stack scalars + fixed ``[N]`` / ``[N][N]`` arrays).

    The residual is evaluated by an inlined device lambda built from the residual sub-block (the
    iterate stack, the frozen guess, named ``source`` / ``apply`` per-cell Exprs, affine combines). The
    Jacobian is finite-difference: column j perturbs ``U[j] += eps`` and forms ``(r(U+eps e_j)-r(U))/eps``
    with a relative ``eps`` so it scales with the iterate magnitude."""
    impl = _model_impl(model)
    n = len(impl.cons_names)
    tol = repr(float(v.attrs["tol"]))
    max_iter = int(v.attrs["max_iter"])
    # The aux fields the residual reads: bind them once per cell (constant across the Newton iterates),
    # so the residual lambda captures them by reference. Gather the dependency set over every term Expr.
    term_exprs = []
    for w in v.attrs["residual_block"]:
        if w.op in ("source", "apply"):
            term_exprs += _residual_term_exprs(impl, w)
    body = _kernel_open(out_var, guess_var)
    # Per-cell aux + the live primitives are NOT pre-bound here: the prims depend on the ITERATE (they
    # are recomputed inside the residual lambda from Ueval). Only the aux locals are cell constants.
    body += ["    " + ln for ln in _cell_locals(impl, term_exprs, guess_var, with_cons=False,
                                                with_prim=False)]
    # The frozen initial guess as a stack vector (the residual reads it as a per-cell constant).
    body.append("    pops::Real Gval[%d];" % n)
    for c in range(n):
        body.append("    Gval[%d] = %sA(i, j, %d);" % (c, guess_var, c))
    # The residual-eval lambda r(Ueval) -> rout (device, stack-only, no std::function): captures the
    # cell-constant aux locals + the frozen guess by reference; recomputes the iterate-dependent
    # primitives inside from Ueval (bound to the conservative names).
    body.append("    auto residual_eval = [&](const pops::Real (&Ueval)[%d], pops::Real (&rout)[%d]) {"
                % (n, n))
    for c, cn in enumerate(impl.cons_names):
        body.append("      const pops::Real %s = Ueval[%d];" % (cn, c))
    # Live primitives of the residual terms, in declaration order (a prim may use an earlier prim).
    live = impl._live_prims(term_exprs) if term_exprs else set()
    for p, expr in impl.prim_defs.items():
        if p in live:
            body.append("      const pops::Real %s = %s;" % (p, expr.to_cpp()))
    body += ["      " + ln for ln in _emit_residual_eval(impl, v, n)]
    body.append("    };")
    # Newton state: the iterate U_ (seeded to the guess), the residual r_, the FD Jacobian J_ and step.
    body.append("    pops::Real U_[%d];" % n)
    for c in range(n):
        body.append("    U_[%d] = Gval[%d];" % (c, c))
    body.append("    pops::Real r_[%d];" % n)
    body.append("    for (int it_ = 0; it_ < %d; ++it_) {" % max_iter)
    body.append("      residual_eval(U_, r_);")
    # Convergence on max_c |r_c| (the per-cell residual infinity norm).
    body.append("      pops::Real rmax_ = pops::Real(0);")
    body.append("      for (int c_ = 0; c_ < %d; ++c_) rmax_ = std::fmax(rmax_, std::fabs(r_[c_]));" % n)
    body.append("      if (rmax_ < static_cast<pops::Real>(%s)) break;" % tol)
    # FD Jacobian J_[i][j] = (r_i(U + eps e_j) - r_i(U)) / eps, eps relative to |U_j| (floored).
    body.append("      pops::Real J_[%d][%d];" % (n, n))
    body.append("      pops::Real Up_[%d];" % n)
    body.append("      pops::Real rp_[%d];" % n)
    body.append("      for (int j_ = 0; j_ < %d; ++j_) {" % n)
    body.append("        for (int k_ = 0; k_ < %d; ++k_) Up_[k_] = U_[k_];" % n)
    body.append("        const pops::Real eps_ = static_cast<pops::Real>(1e-7) "
                "* std::fmax(std::fabs(U_[j_]), static_cast<pops::Real>(1));")
    body.append("        Up_[j_] += eps_;")
    body.append("        residual_eval(Up_, rp_);")
    body.append("        for (int i_ = 0; i_ < %d; ++i_) J_[i_][j_] = (rp_[i_] - r_[i_]) / eps_;" % n)
    body.append("      }")
    # Newton step J dU = -r via the SAME stack dense inverse solve_local_linear uses; U -= dU.
    body.append("      pops::Real Jinv_[%d][%d];" % (n, n))
    body.append("      pops::detail::mat_inverse<%d>(J_, Jinv_);" % n)
    body.append("      for (int i_ = 0; i_ < %d; ++i_) {" % n)
    body.append("        pops::Real du_ = pops::Real(0);")
    body.append("        for (int k_ = 0; k_ < %d; ++k_) du_ += Jinv_[i_][k_] * r_[k_];" % n)
    body.append("        U_[i_] -= du_;")
    body.append("      }")
    body.append("    }")
    for c in range(n):
        body.append("    outA(i, j, %d) = U_[%d];" % (c, c))
    body += _kernel_close()
    return body


def _linear_source_rows(impl, name):
    """The n_cons x n_cons matrix of Expr of a model linear source @p name (m.linear_source).
    @p impl is the HyperbolicModel."""
    if name not in impl._linear_sources:
        raise NotImplementedError(
            "emit_cpp_program: linear source '%s' is not declared on the model (m.linear_source); "
            "declared: %s" % (name, sorted(impl._linear_sources)))
    return impl._linear_sources[name]


# Source of a generated problem.so. The includes + pops_install_program closure match the shape
# tests/test_program_loader compiles+runs in CI; pops_program_hash is added per the spec .so ABI (a
# cache/restart key) and is not yet consumed by System::install_program. {name} is a JSON-escaped C
# string literal, {hash} the IR hash, {prelude} the INSTALL-TIME C++ (persistent scratch + matrix-free
# apply lambdas, captured into the step closure by [=]), {body} the step-closure body (both already
# indented); the literal braces of the C++ scaffold are doubled for str.format.
_PROGRAM_CPP_TEMPLATE = '''\
// GENERATED by pops.time.Program.emit_cpp_program (epic ADC-399 / ADC-401). Do not edit by hand.
// A compiled time Program installed across the stable .so ABI: it drives sim.step(dt) entirely in
// C++ via ProgramContext, reusing the adc_cpp runtime (no MultiFab / flux / solver reimplementation).
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/fab2d.hpp>          // Array4 / ConstArray4 (per-cell handles)
#include <pops/mesh/execution/for_each.hpp>     // for_each_cell (Phase-4b per-cell kernels)
#include <pops/numerics/linalg/dense_eig.hpp>   // pops::detail::mat_inverse (local dense solve)
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>  // pops::cg_solve / bicgstab_solve / richardson_solve / gmres_solve (matrix-free)
#include <pops/core/foundation/types.hpp>
#include <chrono>                              // std::chrono::steady_clock (per-node profiling pair, ADC-459)
#include <cmath>                               // std::sqrt / std::fabs / std::pow in lowered formulas
#include <limits>                              // std::numeric_limits (dt_bound +inf sentinel)
#include <memory>                              // std::make_shared (persistent matrix-free scratch)
#include <vector>                              // pointer list for the coupled multi-block field-solve (ADC-457)

extern "C" const char* pops_program_abi_key() {{ return POPS_ABI_KEY_LITERAL; }}
extern "C" const char* pops_program_name() {{ return {name}; }}
extern "C" const char* pops_program_hash() {{ return "{hash}"; }}

{block_names}
{module_metadata}
extern "C" void pops_install_program(void* sys) {{
  pops::runtime::program::ProgramContext ctx(sys);
{prelude}
  ctx.install([=](double dt) {{
    (void)dt;
{body}
  }});
}}

// OPTIONAL dt bound (spec s18 / ADC-417). pops_program_has_dt_bound() is true iff the Program set one;
// pops_program_dt_bound(ctx, cfl) returns the lowered scalar bound (min'd into the native CFL by
// step_cfl). When no bound was set, has_dt_bound() is false and dt_bound returns +inf (unreached).
extern "C" bool pops_program_has_dt_bound() {{ return {has_dt_bound}; }}
extern "C" pops::Real pops_program_dt_bound(pops::runtime::program::ProgramContext* ctxp, pops::Real cfl) {{
  pops::runtime::program::ProgramContext& ctx = *ctxp;
  (void)ctx; (void)cfl;
{dt_bound_body}
}}
'''



# --- Program -> C++ lowering (free functions taking `program`) ------------------------------
# --- C++ codegen (Phase 2c-ii / Phase 4b): lower the IR to a problem.so source ---
def emit_cpp_program(program, model=None):
    """Generate the C++ source of a problem.so implementing this Program (codegen).

    Exports the stable .so ABI -- ``pops_program_abi_key`` (the ``POPS_ABI_KEY_LITERAL``
    preprocessor literal, NOT the interposable inline), ``pops_program_name``, ``pops_program_hash``,
    ``pops_install_program`` -- and installs the macro step as a closure built from `ProgramContext`
    primitives only (no MultiFab / flux / solver reimplementation). It is the source the C++ loader
    (`System::install_program`) compiles, dlopens, and runs.

    Lowers the Program by a topological walk of the SSA IR: each block's current state is its base
    (``ctx.state(idx)``); ``solve_fields()`` runs the elliptic solve; each RHS becomes a
    scratch + ``rhs_into``; each intermediate ``linear_combine`` becomes a zero scratch accumulated
    with ``axpy``; the committed combine writes the block state via ``lincomb``. Forward Euler,
    SSPRK2/SSPRK3 and RK4 all lower this way -- no per-scheme class.

    Multi-block (ADC-426): N ``P.state(\"a\")`` / ``P.state(\"b\")`` declarations + N ``P.commit``
    are lowered -- each op routes to its own block's runtime index (``_block_indices``, in the order
    the blocks are first declared via ``P.state``). The .so also exports its block NAMES in that
    order (``pops_program_block_count`` / ``pops_program_block_name``); ``System::install_program``
    binds them to the instantiated System blocks BY NAME (Spec 3 criterion 23, ADC-457), so the
    System blocks (``sim.add_equation`` / ``sim.add_block``) may be added in ANY order -- a Program
    block whose name has no instantiated System block fails loud (``Program requires block instance
    '<name>', but simulation did not instantiate it``). A block declared but never committed is a
    READ-ONLY block (allowed; e.g. a passive field whose charge couples the others through the shared
    Poisson). A commit of a block no ``P.state`` declares is rejected. A single-block Program lowers
    byte-identically (its one block is index 0; an order-matching multi-block Program too -- the
    name map is the identity).

    Phase-4b also lowers the SPLIT-SOURCE / LOCAL-LINEAR ops -- ``source`` (a named ``m.source_term``
    evaluated per cell), ``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear``
    ((I -/+ a*L) U = rhs solved cell by cell via a dense per-cell inverse) -- but ONLY when the
    physical ``model`` (the ``pops.dsl`` model whose ``source_term`` / ``linear_source`` they name)
    is provided: the codegen reads the model's symbolic coefficients to emit the per-cell kernels.
    Without ``model`` those ops raise NotImplementedError (the Program cannot be lowered in
    isolation); ``model=None`` still lowers FE / SSPRK / RK4 (no model needed). A ``rhs`` routes its
    base on its ``flux`` flag and whether ``"default"`` is among the requested ``sources`` (ADC-425 /
    ADC-430, spec criterion 17 -- flux and sources are explicit, never summed implicitly). With
    ``flux=True``: ``"default"`` present -> ``ctx.rhs_into`` (= ``-div F`` + the model's
    default/composite source, the historical path); ``"default"`` absent (incl. the empty list
    ``[]``) -> ``ctx.neg_div_flux_default_into`` (= ``-div F`` only, NO default source). With
    ``flux=False`` (SOURCE-ONLY, ADC-430): NO ``-div F`` base -- ``"default"`` present (or ``None``)
    -> ``ctx.source_default_into`` (= S only, the exact mirror); ``"default"`` absent -> the zeroed
    scratch (the named sources, if any, are the whole RHS). Each NAMED source (``sources=[...]``
    beyond ``"default"``) then lowers with a model: the same per-cell ``m.source_term`` kernel as the
    standalone ``source`` op, accumulated onto ``R`` via ``axpy``. So ``flux=True,sources=[]`` is flux
    only, ``flux=True,sources=["default"]`` is flux + default source (unchanged),
    ``flux=False,sources=["default"]`` is the default source only, ``flux=False,sources=["s"]`` is
    just ``s`` -- the named ones never double-count the default (it is folded in iff "default" was
    listed). More than one block now lowers (ADC-426): each op routes to its block's runtime index
    (``_block_indices``, in P.state declaration order) and control flow (while/range/if) inside a
    block lowers per block; a SIMULTANEOUS multi-target coupled field solve
    (``solve_fields_from_blocks([Ua, Ub])``) lowers to ``ctx.solve_fields_from_blocks`` (see below).

    Each ``solve_fields(state=...)`` op lowers to ``ctx.solve_fields_from_state(idx, <stage state>)``
    (ADC-409): the elliptic fields are re-solved -- and the shared aux re-filled -- from THAT stage's
    state, not the block's current state. So a field-coupled multi-stage scheme (Poisson feedback
    into the flux) is exact: stage k's RHS reads phi solved from stage k's own state. For the first
    stage the stage state is U^n, so this is identical to the historical ``solve_fields()``; for an
    uncoupled model the field solve is inert either way. This is already a COUPLED multi-block solve:
    the system Poisson RHS is ``Sum_s elliptic_rhs_s(U_s)`` (``assemble_poisson_rhs``), so block
    ``idx`` reads its stage state while every OTHER block contributes its LIVE state into the one
    shared phi/aux. A per-block ``P.solve_fields(state=Ub)`` therefore sees all blocks' charge. A
    SIMULTANEOUS multi-target override (several blocks at their stage states in ONE solve) lowers to
    ``ctx.solve_fields_from_blocks(<vec>)`` (Spec 3 criterion 24, ADC-457): the RHS is
    ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state at once
    (``assemble_poisson_rhs_from_blocks``), each slotted at its block index (nullptr = the block's
    live state) -- the coupled multi-species field solve."""
    program.validate()
    _check_lowerable(program, model)
    prelude, body = _emit_body(program, model)
    # Optional dt bound (spec s18 / ADC-417): emit the SECOND ABI pair -- pops_program_has_dt_bound()
    # (true iff a bound was set) and pops_program_dt_bound(ProgramContext*, cfl) (the lowered scalar
    # expression). Without a bound, has_dt_bound() returns false and the dt_bound function returns a
    # +inf sentinel (never reached: the loader stores the closure only when has_dt_bound() is true).
    has_dt_bound, dt_bound_body = _emit_dt_bound(program, model)
    return _PROGRAM_CPP_TEMPLATE.format(
        name=json.dumps(program.name), hash=program._ir_hash(), prelude=prelude, body=body,
        has_dt_bound=has_dt_bound, dt_bound_body=dt_bound_body,
        module_metadata=_emit_module_metadata(program, model),
        block_names=_emit_block_names(program))

def _emit_block_names(program):
    """C++ source of the NAME-based block-binding ABI the .so exports (Spec 3 criterion 23, ADC-457):
    ``pops_program_block_count()`` and ``pops_program_block_name(int)`` -- the Program's block names in
    ``_block_indices`` order (P.state declaration order, the order the step body's ``ctx.state(idx)``
    addresses). System::install_program reads them, matches each to the instantiated System block of
    that name, and stores the program-index -> system-index map (read by ProgramContext), so the
    System blocks may be added in ANY order vs the Program's P.state declarations -- a Program block
    whose name has no System block fails loud. The block names are also part of the IR identity (the
    block_order field of _serialize feeds the IR hash), so reordering P.state changes the hash."""
    order = program._block_indices()  # name -> index, declaration order
    names = sorted(order, key=order.get)
    cases = "".join('    case %d: return %s;\n' % (order[nm], json.dumps(nm)) for nm in names)
    return (
        "// NAME-based block binding (Spec 3 criterion 23, ADC-457): the Program's block names in\n"
        "// P.state declaration order. install_program matches each to a System block BY NAME (not\n"
        "// add-order) and builds the program-index -> system-index map ProgramContext resolves.\n"
        'extern "C" int pops_program_block_count() { return %d; }\n' % len(names) +
        'extern "C" const char* pops_program_block_name(int i) {\n'
        '  switch (i) {\n%s    default: return "";\n  }\n}\n' % cases)

def _emit_module_metadata(program, model=None):
    """C++ source of the GeneratedModule metadata the .so exports (Spec 2 / ADC-442).

    A combined model+program .so carries, alongside ``GeneratedProgram`` (the step), a
    ``GeneratedModule`` descriptor: ``extern "C"`` accessors exposing the typed operator registry
    -- a count and, per integer ``OperatorId`` (the array index), the operator name / kind /
    signature / requirements -- plus the state and field space names. These are read ONCE at
    install (introspection + requirement validation, ``module_metadata.hpp``); the step body never
    calls them, so operators stay inlined and there is no string lookup in any hot kernel.
    ``model=None`` emits an empty module (count 0). The metadata is derived from the model's typed
    registry, so it does not perturb the program IR hash.
    """
    ops, states, fields = [], [], []
    if model is not None and hasattr(model, "operator_registry"):
        reg = model.operator_registry()
        ops = [reg.get(nm) for nm in reg.names()]
        if hasattr(model, "state_space"):
            states = [model.state_space().name]
        if hasattr(model, "field_space"):
            fields = [model.field_space().name]

    def table(accessor, values):
        cases = "".join('    case %d: return %s;\n' % (i, json.dumps(v))
                        for i, v in enumerate(values))
        return ('extern "C" const char* pops_module_%s(int i) {\n'
                '  switch (i) {\n%s    default: return "";\n  }\n}\n' % (accessor, cases))

    def req_json(op):
        # The operator's own kind always wins (a requirements dict must not shadow it).
        return json.dumps({**op.requirements, "kind": op.kind})

    parts = [
        "// GeneratedModule metadata (Spec 2 / ADC-442): the typed operator registry exposed by\n"
        "// the .so for introspection + install-time validation. OperatorId = the array index.\n"
        "// NOT called from any hot kernel -- operators are inlined at codegen.\n",
        'extern "C" int pops_module_operator_count() { return %d; }\n' % len(ops),
        'extern "C" int pops_module_state_space_count() { return %d; }\n' % len(states),
        'extern "C" int pops_module_field_space_count() { return %d; }\n' % len(fields),
        table("operator_name", [op.name for op in ops]),
        table("operator_kind", [op.kind for op in ops]),
        table("operator_signature", [repr(op.signature) for op in ops]),
        table("operator_requirements", [req_json(op) for op in ops]),
        table("state_space_name", states),
        table("field_space_name", fields),
    ]
    return "".join(parts)

def _emit_dt_bound(program, model=None):
    """Lower the optional dt bound (spec s18 / ADC-417) to ``(has_dt_bound, body)``: the bool literal
    pops_program_has_dt_bound returns and the C++ body of pops_program_dt_bound. No bound -> ("false",
    a +inf return that is never reached). The bound is a READ-ONLY scalar sub-program: it reuses the
    same per-op lowering (state -> ctx.state(idx), reductions, cfl/hmin/max_wave_speed, scalar_op) and
    returns the final scalar. ADC-426: a multi-block dt bound may read several blocks' states (e.g.
    the min over blocks of cfl*hmin/max_wave_speed), so each op resolves its OWN block index / base.
    No commit lives in a dt bound (empty committed_ids)."""
    if program._dt_bound is None:
        return "false", "    return std::numeric_limits<pops::Real>::infinity();"
    sub, result = program._dt_bound
    block_idx = program._block_indices()
    bases = {}
    for v in sub:
        if v.op == "state" and v.block not in bases:
            bases[v.block] = v
    var = {}
    lines = []
    for v in sub:
        _emit_op(program, v, bases.get(v.block), frozenset(), var, model, lines, None, block_idx)
    lines.append("return %s;" % var[result.id])
    body = "\n".join("    " + ln for ln in lines)
    return "true", body



def _check_lowerable(program, model=None):
    """Raise NotImplementedError if the IR uses a construct the current codegen cannot lower yet,
    naming the offending construct (never a silent mis-lowering). @p model: the physical model that
    declares the named sources / linear sources; required for the Phase-4b ops.

    Multi-block (ADC-426): N ``P.state`` blocks + N ``P.commit`` are supported -- each op routes to
    its block's index (``_block_indices``). Validation: a block is committed AT MOST once (enforced
    at ``commit`` time); a read-only block (declared via ``P.state`` but never committed) is allowed
    (e.g. a passive field whose charge couples the others); a commit of a block that was never
    declared by ``P.state`` is rejected (an unknown-block commit cannot route to an index)."""
    blocks = program._block_indices()
    for b in program._commits:
        if b not in blocks:
            raise ValueError(
                "commit of unknown block '%s': no P.state('%s') declares it (declared blocks: %s)"
                % (b, b, sorted(blocks)))
    _check_schedules_lowerable(program)
    for v in program._values:
        _check_op_lowerable(program, v, model)
    # Per-cell dense fallback bound for the local dense solves (mat_inverse<N> uses fixed stack
    # buffers): solve_local_linear (M = I - a*L) and solve_local_nonlinear (the Newton FD Jacobian).
    dense_ops = ("solve_local_linear", "solve_local_nonlinear")
    if model is not None and any(v.op in dense_ops for v in _all_ops(program)):
        impl = _model_impl(model)
        n_cons = len(getattr(impl, "cons_names", []) or [])
        if n_cons > 8:
            raise ValueError(
                "local dense fallback currently supports n_cons <= 8 (got %d)" % n_cons)

def _check_schedules_lowerable(program):
    """Gate the unified Program scheduler lowering (ADC-458, Spec 3 sections 17-18). EVERY kind/policy
    now lowers (``_emit_schedule_wrap``) EXCEPT the two that need a runtime primitive the compiled
    .so does not have, which still fail loud (never a silent no-op):

      - ``on_end()``: a compiled ``sim.step(dt)`` loop carries no end-of-run signal, so the .so cannot
        know which step is the last. (Use an on_end host hook instead.)
      - ``when(cond)`` whose cond is a bare Python callable, not a Program Bool predicate: a callable
        is not a Program value and cannot be lowered to C++.

    The cadence RUNTIME (the cache cadence in a stepping .so) is exercised on ROMEO; the cache
    MANAGER is unit-tested by tests/test_cache_manager.cpp."""
    for v in _all_ops(program):
        sched = v.attrs.get("schedule")
        if sched is None or sched.is_always():
            continue
        if sched.kind == "on_end":
            raise NotImplementedError(
                "schedule on_end() on node %r (op '%s') is not lowerable: a compiled sim.step(dt) "
                "loop never sees an end-of-run signal, so the .so cannot know the last step. Use "
                "on_start()/every()/when()/subcycle(), or an on_end host hook (ADC-458)."
                % (v.name, v.op))
        if sched.kind == "when":
            cond = sched.params.get("cond")
            if not (isinstance(cond, Value) and cond.vtype == "bool"):
                raise NotImplementedError(
                    "schedule when(cond) on node %r lowers only a Program Bool predicate (e.g. "
                    "P.norm2(r) < tol), not a Python callable (ADC-458)." % v.name)
        if sched.kind == "subcycle" and v.op not in _AUX_OUTPUT_OPS:
            # subcycle re-runs the body COUNT times in a for-loop scope. A node whose output is a
            # step-body scratch (rhs / source / linear_combine / ...) would declare that scratch
            # INSIDE the loop, leaving it out of scope for any downstream consumer -- broken C++. Only
            # an aux-output op (a field solve, which writes the persistent System aux) is well-defined
            # under sub-cycling; a scratch sub-step has no single 'result' to consume. Fail loud.
            raise NotImplementedError(
                "schedule subcycle on node %r (op '%s') is lowerable only for a field solve (its "
                "output is the persistent System aux); a scratch-output op sub-cycled has no single "
                "result a downstream node can read (ADC-458). Sub-cycle the field solve, or express "
                "the inner steps explicitly." % (v.name, v.op))

# 'linear_source' is a pure NAME-reference SSA node (vtype 'operator'): it carries no runtime work
# (consumed by apply / solve_local_linear, which read the model coefficients), so it lowers to
# nothing -- always allowed, model or not. 'reduce' / 'compare' / 'while' are the ADC-404a control
# flow / reduction ops (lowered inline via pops::dot; no model needed). 'matrix_free_operator' /
# 'scalar_field' / 'laplacian' / 'gradient' / 'divergence' / 'solve_linear' are the ADC-405 / ADC-412
# matrix-free Krylov ops (the operator declaration carries an apply sub-block; solve_linear lowers to
# pops::*_solve; divergence is the centered FV divergence of a gradient field).

# Ops NOT wrapped in a per-node profile scope (ADC-459): they bind a reference or read a cached
# scalar and do no per-step numerical work, so timing them only adds always-zero noise to
# sim.profile_report(). Every other op that emits a statement is wrapped (rhs / solve_fields /
# linear_combine / source / apply / reductions / loops / Schur kernels / ...).

def _all_ops(program):
    """Iterate over every op of the Program, descending into control-flow + apply sub-blocks (a flat
    view used by the lowerability guards: the sub-block ops are not in program._values). Nested control
    flow is disallowed, so the sub-blocks are flat (one level)."""
    for v in program._values:
        yield v
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            blk = v.attrs.get(key)
            if isinstance(blk, list):
                yield from blk

def _check_op_lowerable(program, v, model):
    """Lowerability check for a single op (used for both the top-level walk and a while sub-block).
    Raises NotImplementedError / ValueError naming the offending construct (never a mis-lowering)."""
    if v.op in _MODEL_OPS:
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower op '%s' (value '%s') without the physical model "
                "that declares its named source / linear source; pass model= "
                "(compile_problem threads it through)" % (v.op, v.name))
        if v.op == "solve_local_nonlinear":  # recurse: the residual sub-block ops must lower too
            for w in v.attrs["residual_block"]:
                _check_op_lowerable(program, w, model)
        return  # _emit_op lowers it from the model's symbolic coefficients
    if v.op not in _ALLOWED_OPS:
        raise NotImplementedError(
            "emit_cpp_program cannot lower op '%s' (value '%s') yet; supported ops are %s "
            "(+ %s with a model; nested control flow / Krylov are later phases)"
            % (v.op, v.name, sorted(_ALLOWED_OPS), sorted(_MODEL_OPS)))
    if v.op == "coupled_rate":
        # A coupled_rate (collisions / ionization, Spec 3 criterion 27) lowers to ONE multi-state
        # for_each_cell kernel (see _emit_coupled_rate_kernel). The lowering reaches the operator
        # body (its per-block component formulas) through the BOUND registry, and binds each input
        # state's cons names from that input's StateSpace -- so the operator must be bound and the
        # formulas must be cons-only (the MVP). Validate both here so a non-lowerable coupled_rate
        # fails loud naming ADC-457, never emits an undefined reference.
        _coupled_rate_components(program, v)
        return
    if v.op == "coupled_rate_out":
        # A pure projection of one block out of the coupled bundle: it emits nothing (its var
        # aliases that block's rate scratch). Lowerable iff its producing coupled_rate is (checked
        # when that node is walked); nothing to validate here.
        return
    if v.op in ("while", "range", "if"):  # recurse: the cond / body sub-blocks must lower too
        for key in ("cond_block", "body_block"):
            for w in v.attrs.get(key, []):
                _check_op_lowerable(program, w, model)
        return
    if v.op == "matrix_free_operator":  # recurse into the apply sub-block (set by set_apply)
        if v.attrs.get("apply_block") is None:
            raise ValueError(
                "matrix_free_operator '%s' has no apply; call P.set_apply before lowering"
                % v.name)
        for w in v.attrs["apply_block"]:
            _check_op_lowerable(program, w, model)
        return
    if v.op == "solve_fields":
        # A NAMED elliptic field (ADC-419/ADC-428) drives a SECOND elliptic solve into its own aux
        # channel. The runtime now hosts it (System::solve_fields_from_state(field, ...) via
        # ProgramContext); lowering needs the model so the field name can be validated against the
        # declared m.elliptic_field set (the codegen emits the named ctx call).
        field = v.attrs.get("field")
        if field is not None:
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower solve_fields with a named elliptic field "
                    "('%s') without the physical model that declares it (m.elliptic_field); pass "
                    "model= (compile_problem threads it through)" % field)
            if field not in _model_impl(model)._elliptic_fields:
                raise ValueError(
                    "unknown elliptic_field '%s' in solve_fields '%s'; declared: %s"
                    % (field, v.name, sorted(_model_impl(model)._elliptic_fields)))
        return
    if v.op == "rhs":
        named_fluxes = _named_fluxes(v)
        # ADC-430: flux=False is SOURCE-ONLY -- no -div F base. Named fluxes (a -div of selected
        # flux_terms) contradict "no flux": reject the combination loud rather than silently picking
        # one (request flux=True for named fluxes, or flux=False for a source-only stage).
        if not v.attrs.get("flux", True) and named_fluxes is not None:
            raise ValueError(
                "rhs '%s' sets flux=False (source-only) but also requests named fluxes %r; a "
                "source-only stage has no flux divergence -- drop fluxes= or set flux=True"
                % (v.name, named_fluxes))
        if named_fluxes is not None:  # NAMED fluxes (ADC-419): need the model's flux_term coeffs
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower rhs '%s' with named fluxes %r without the "
                    "physical model that declares them (m.flux_term); pass model= "
                    "(compile_problem threads it through)" % (v.name, named_fluxes))
            impl_f = _model_impl(model)
            ft = impl_f._flux_terms
            for f in named_fluxes:
                if f not in ft:
                    raise ValueError(
                        "unknown flux_term '%s' in rhs '%s'; declared flux_terms: %s"
                        % (f, v.name, sorted(ft)))
            # The named-flux path emits -div(selected fluxes) only (no ctx.rhs_into), so the model's
            # DEFAULT source would be silently dropped -- reject it (it must be requested as a named
            # source_term instead). The named sources below are still axpy'd on top.
            if getattr(impl_f, "_source", None):
                raise NotImplementedError(
                    "rhs with named fluxes %r needs a model whose default source is empty (no "
                    "m.source); rhs '%s' has a non-empty default source that the named-flux path "
                    "would drop (declare it as a source_term instead)" % (named_fluxes, v.name))
        extra = [s for s in (v.attrs.get("sources") or []) if s != "default"]
        if not extra:
            return
        # A named source in an rhs reads the model's symbolic source_term coefficients (same as the
        # standalone 'source' op): lowering needs the model.
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower rhs '%s' with named sources %r without the "
                "physical model that declares them (m.source_term); pass model= "
                "(compile_problem threads it through)" % (v.name, extra))
        impl = _model_impl(model)
        # ADC-425: the named sources are axpy'd on top of an EXPLICIT base. With "default" requested
        # the base is ctx.rhs_into (flux + the model's default/composite source); without it the base
        # is ctx.neg_div_flux_default_into (flux only). Either way the default source is folded in iff
        # the caller listed "default", so adding distinct named source_terms cannot double-count it --
        # the old "model default source must be empty" rejection is gone (the routing is now exact).
        for s in extra:
            if s not in impl._source_terms:
                raise ValueError(
                    "unknown source_term '%s' in rhs '%s'; declared source_terms: %s"
                    % (s, v.name, sorted(impl._source_terms)))

def _coupled_rate_components(program, v):
    """Resolve a ``coupled_rate`` node @p v to its per-block component formulas (Spec 3 criterion
    27, ADC-457), validated for the cons-only MVP. Returns ``{block: [Expr, ...]}`` (one formula
    per component of that block's StateSpace).

    The component formulas live in the BOUND operator's body (``op.body`` = the ``expr=`` dict
    passed to ``Module.operator``), reachable through the registry the node's ``operator`` attr
    names; the input states' cons names come from each input value's StateSpace (set by
    ``P.state(space=...)``). Raises a clear NotImplementedError naming ADC-457 when a coupled_rate
    cannot lower in this MVP: no bound registry, no operator body, a block whose component count
    does not match its StateSpace, or a formula referencing a non-cons (prim / aux) Var."""
    from pops.ir.expr import Var
    op_name = v.attrs["operator"]
    if program._registry is None:
        raise NotImplementedError(
            "the coupled_rate kernel codegen (ADC-457) needs the bound operator registry to reach "
            "operator %r's component formulas; call P.bind_operators(module) before emitting "
            "(node %r)" % (op_name, v.name))
    op = program._registry.get(op_name)
    expr = op.body
    if not isinstance(expr, dict):
        raise NotImplementedError(
            "the coupled_rate kernel codegen (ADC-457) needs operator %r to carry its per-block "
            "component formulas as an expr={block: [Expr, ...]} dict (got %r); a decorator-body "
            "coupled_rate is a later phase (node %r)" % (op_name, type(expr).__name__, v.name))
    # Each coupled_rate_out block must own one input state (its rate scratch is shaped like that
    # block's state) whose StateSpace gives the component count + cons names.
    by_block = {s.block: s for s in v.inputs}
    components = {}
    for blk, comps in expr.items():
        state_in = by_block.get(blk)
        if state_in is None or getattr(state_in, "space", None) is None:
            raise NotImplementedError(
                "the coupled_rate kernel codegen (ADC-457) needs every output block to map to an "
                "input State declared with a StateSpace (P.state(%r, space=...)); operator %r "
                "block %r has none (node %r)" % (blk, op_name, blk, v.name))
        ncons = len(state_in.space.components)
        if len(comps) != ncons:
            raise NotImplementedError(
                "coupled_rate operator %r block %r emits %d component formulas but its StateSpace "
                "has %d components; the rate must be full-rank over the block state (ADC-457, "
                "node %r)" % (op_name, blk, len(comps), ncons, v.name))
        for e in comps:
            for node in _walk_expr(e):
                if isinstance(node, Var) and node.kind != "cons":
                    raise NotImplementedError(
                        "coupled_rate formulas referencing prim/aux vars are deferred (ADC-457): "
                        "operator %r block %r references %s var %r; the MVP per-cell binding is "
                        "cons-only (node %r)" % (op_name, blk, node.kind, node.name, v.name))
        components[blk] = list(comps)
    # Every cons var a formula references must be a component of SOME input state (an output block
    # OR a read-only catalyst input). A name in no input state -- a typo, or a name the author
    # forgot to add to a P.state(space=...) -- would emit an undefined C++ identifier that only
    # fails at the AOT compile, far from the authoring site; reject it loud here, like prim/aux.
    all_cons = {c for s in v.inputs if getattr(s, "space", None) is not None
                for c in s.space.components}
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    missing = referenced - all_cons
    if missing:
        raise NotImplementedError(
            "coupled_rate operator %r references cons var(s) %s that are a component of no input "
            "state; declare them via P.state(space=...) or fix the formula (ADC-457, node %r)"
            % (op_name, sorted(missing), v.name))
    return components

def _walk_expr(e):
    """Yield every node of a dsl Expr tree (used to scan a coupled_rate formula for non-cons Vars)."""
    from pops.ir.visitors import _children
    stack = [e]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(_children(node))

def _emit_body(program, model=None):
    """Generate the C++ of the install function in TWO phases (each list indented uniformly by the
    template). Assumes `_check_lowerable` has passed. @p model supplies the symbolic coefficients of
    the Phase-4b source / apply / solve_local_linear ops. Returns ``(prelude, body)``:

      - ``prelude``: INSTALL-TIME C++ (before ``ctx.install``) -- persistent scratch fields (held
        via ``std::shared_ptr`` so they outlive the install call and are reused across every step
        and every Krylov iteration) and the matrix-free apply lambdas. Captured by value into the
        step closure (shared_ptr / lambda / ctx all copy cheaply).
      - ``body``: the STEP closure body (one macro-step over dt).

    Multi-block (ADC-426): the SSA walk allocates a per-block base (``ctx.state(idx)`` for each
    declared block) and routes every op to ITS block's index via ``_block_indices`` / ``v.block``.
    Each committed block's final value is copied into that block's state (a scratch commit) or was
    written in place (a linear_combine commit). A single block reduces to the historical lowering."""
    block_idx = program._block_indices()
    # The first-declared state Value per block: the "base" any op of that block clones / commits into.
    bases = {}
    for v in program._values:
        if v.op == "state" and v.block not in bases:
            bases[v.block] = v
    # IR value id -> C++ token: a MultiFab variable name (states / RHS scratches), a scalar variable
    # name (reductions, ``s{id}``) or a parenthesized boolean expression (compares).
    var = {}
    prelude = []
    lines = []
    # Bool-predicate value id -> its C++ token, for a when(cond) schedule whose cond is a Program
    # compare value emitted earlier in the body (ADC-458). Reset per emit (tokens are body-local).
    program._when_tokens = {}
    committed_ids = {s.id for s in program._commits.values()}
    # Multistep histories (ADC-406a): register each declared history at its MAX lag FIRST (a
    # registration-only call, NOT a read -- a read before the first store fails loud), so the ring
    # depth is locked before any store. The first ctx.store_history then cold-start-fills every
    # (already-allocated) slot -- step 0 reads the same value at every lag and the scheme degenerates
    # to a one-step method. register_history is idempotent (no-op once registered).
    for name, lag in sorted(program._histories.items()):
        lines.append("ctx.register_history(%s, %d);" % (json.dumps(name), int(lag)))
    for v in program._values:
        base = bases.get(v.block)  # the block-state value of THIS op's block (None: a scalar op)
        _emit_op(program, v, base, committed_ids, var, model, lines, prelude, block_idx)
    # Each committed block: a scratch commit (solve_local_linear / solve_linear / a non-base
    # linear_combine wrote a scratch) is copied into the block state; a linear_combine commit already
    # wrote ctx.state(idx) in place (var == base), so its copy is a no-op (skipped).
    for block, committed in program._commits.items():
        base = bases[block]
        if var[committed.id] != var[base.id]:
            lines.append(
                "ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                % (var[base.id], var[base.id], var[committed.id]))
    # Rotate the history rings ONCE at the very end of the step (after the commit), so the next step
    # reads lag k as the value k stores ago. Only emitted when the Program uses histories.
    if program._histories:
        lines.append("ctx.rotate_histories();")
    prelude_src = "\n".join("  " + ln for ln in prelude)
    body_src = "\n".join("    " + ln for ln in lines)
    return prelude_src, body_src

def _emit_op(program, v, base, committed_ids, var, model, lines, prelude=None, block_idx=None):
    """Lower a SINGLE op to C++, appending to @p lines and recording its C++ token in @p var. Shared
    by the top-level walk and the while sub-blocks (a while body re-runs this per op each pass), so
    reductions / compares / linear_combine all lower identically inside the loop. @p base is the
    block-state value of THIS op's block (its C++ var is the loop variable inside a while sub-block);
    @p committed_ids is the set of committed value ids (empty inside a sub-block: a body combine is
    never a commit). @p prelude collects INSTALL-TIME lines (persistent scratch + apply lambdas) for
    the matrix-free Krylov ops; None inside a sub-block (those ops only appear at the top level for
    now). @p block_idx maps a block name to its runtime index (ADC-426); None inside a sub-block,
    where every op shares the single enclosing block, so a missing map resolves to index 0."""
    bidx = (block_idx or {}).get(v.block, 0)  # this op's runtime block index (0 single-block)
    # PER-NODE PROFILING (ADC-459, Spec 3 section 29): bracket the C++ this op emits with a
    # steady_clock pair recorded under "node:<v.name>", so sim.profile_report() shows each Program
    # node's wall time next to the coarse step / field_solve phases. A steady_clock now() + a
    # ctx.profile_record (NOT a RAII ProfileScope block) is used so the node's emitted C++
    # declarations stay at the step-body scope -- a surrounding { } would hide them from later
    # nodes (e.g. r2 / acc3 read across ops). The timing is additive and ~free when profiling is
    # off (ctx.profile_record early-returns inside Profiler::record); it changes no numerics. Ops
    # that emit no statement (a pure inline token: cfl / compare) are not wrapped (the len guard
    # below skips them). _start marks where this op's lines begin so the open line can be inserted.
    _profile_start = len(lines)
    if v.op == "state":
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab& %s = ctx.state(%d);" % (var[v.id], bidx))
    elif v.op == "solve_fields":
        # Per-stage field solve (ADC-409): solve from the EXPLICIT stage state recorded by
        # P.solve_fields(state=...) so a field-coupled multi-stage scheme re-solves phi from each
        # stage's own state (the shared aux is re-filled before this stage's RHS reads it). For the
        # first stage state == U^n, so this is identical to the old ctx.solve_fields(). Multi-block
        # (ADC-426): solve_fields_from_state(idx, U_stage) is a genuinely COUPLED solve -- the system
        # Poisson RHS is Sum_s elliptic_rhs_s(U_s) (assemble_poisson_rhs), so block idx reads its
        # stage state while every OTHER block contributes its live state into the shared phi/aux.
        (state_in,) = v.inputs  # solve_fields inputs = (state,)
        field = v.attrs.get("field")
        if field is not None:
            # NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve into the field's OWN aux
            # channel (distinct from the shared phi/grad). Lowers to the named overload
            # ctx.solve_fields_from_state(field, block, U) -- block from block_idx (ADC-426); the
            # default (unnamed) path keeps the 2-arg overload below, byte-identical.
            solve_stmt = ('ctx.solve_fields_from_state(%s, %d, %s);'
                          % (json.dumps(field), bidx, var[state_in.id]))
        else:
            solve_stmt = "ctx.solve_fields_from_state(%d, %s);" % (bidx, var[state_in.id])
        lines.append(solve_stmt)
    elif v.op == "solve_fields_from_blocks":
        # Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): a SIMULTANEOUS solve where
        # EVERY listed block reads its OWN stage state at once -- the system Poisson RHS is
        # Sum_s elliptic_rhs_s(U_s) over all coupled blocks (assemble_poisson_rhs_from_blocks), not a
        # single-target override. Lowers to ctx.solve_fields_from_blocks(u_stages), a vector indexed
        # BY BLOCK INDEX (size == ctx.n_blocks(); a nullptr entry uses the block's live state). The
        # listed states are slotted at their block index, so the runtime sees each coupled block at
        # its stage state and every other (unlisted) block at its live state -- the seam a multi-
        # species step uses (the IR commit_many guarantee: no operator observes a partial group).
        # Each input is routed to the slot of ITS OWN block index (not its position in the list), so
        # a reordered list still solves correctly; an input whose block was never declared via
        # P.state has no slot -> fail loud at emit rather than silently mis-route to index 0.
        bmap = block_idx or {}
        vec = "u_stages_%d" % v.id
        lines.append("std::vector<const pops::MultiFab*> %s(ctx.n_blocks(), nullptr);" % vec)
        for st in v.inputs:  # inputs = the N state values, slotted by their own block index
            if st.block not in bmap:
                raise ValueError(
                    "solve_fields_from_blocks: input node %r has block %r, which is not a "
                    "declared program block %r -- cannot route it to a coupled slot"
                    % (st.id, st.block, sorted(bmap)))
            lines.append("%s[%d] = &%s;" % (vec, bmap[st.block], var[st.id]))
        lines.append("ctx.solve_fields_from_blocks(%s);" % vec)
        # solve_fields_from_blocks returns a FieldContext (the shared aux); its var aliases the first
        # listed state so a downstream rhs(state, fields) reads the refreshed shared aux like any
        # solve_fields result (the FieldContext carries no readable buffer of its own).
        var[v.id] = var[v.inputs[0].id]
    elif v.op == "coupled_rate":
        # A coupled rate (collisions / ionization, Spec 3 criterion 27, ADC-457): ONE multi-state
        # for_each_cell kernel fills the per-block rate scratch of EVERY participating block at
        # once -- the component formulas reference cons vars from MULTIPLE input states, so the
        # blocks cannot be lowered as independent single-block rates. Allocate one rate scratch per
        # block (shaped like that block's state, via rhs_scratch_like), emit the shared kernel that
        # binds each input state's Array4 + cons names and writes all block scratches, and record
        # each block's scratch name so the coupled_rate_out for that block aliases it. All input
        # states are co-located (same ba/dm as the System aux), so a single shared loop is sound
        # (the same co-distribution every aux-reading kernel relies on; see _kernel_open).
        components = _coupled_rate_components(program, v)
        by_block = {s.block: s for s in v.inputs}
        scratch = {}
        for blk in components:                       # bundle / expr block order
            scratch[blk] = "cr%d_%s" % (v.id, blk)
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (scratch[blk], var[by_block[blk].id]))
        lines += _emit_coupled_rate_kernel(components, by_block, var, scratch)
        # Per-block scratch names keyed by (coupled node id, block) so each coupled_rate_out aliases
        # its block's scratch (the projection emits no code of its own).
        program._coupled_scratch.update({(v.id, blk): scratch[blk] for blk in scratch})
        var[v.id] = scratch[next(iter(scratch))]     # a stable alias (the bundle has no single value)
    elif v.op == "coupled_rate_out":
        # Pure projection of one block out of the coupled bundle: its var aliases that block's rate
        # scratch (filled by the coupled_rate kernel above). Emits nothing -- like the FieldContext
        # alias of solve_fields_from_blocks. The producing coupled_rate is the node's sole input.
        (coupled_in,) = v.inputs
        var[v.id] = program._coupled_scratch[(coupled_in.id, v.attrs["out_block"])]
    elif v.op == "history":
        # Read the SYSTEM-OWNED history slot (a MultiFab&, ADC-406a): lag steps back. The reference
        # is bound to a C++ name the affine combine then reads like any other state/RHS term.
        var[v.id] = "h%d" % v.id
        lines.append("pops::MultiFab& %s = ctx.history(%s, %d);"
                     % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"])))
    elif v.op == "store_history":
        # Side-effect: copy the value into the current slot of the history (the cold-start fill on
        # the first store happens System-side). store_history is a State-typed node but carries no
        # readable value -- nothing combines it. Its var maps to the stored value (a harmless alias).
        (value_in,) = v.inputs
        lines.append("ctx.store_history(%s, %s);"
                     % (json.dumps(v.attrs["history"]), var[value_in.id]))
        var[v.id] = var[value_in.id]
    elif v.op == "fill_boundary":
        # Side effect on the field's ghosts (the valid cells are untouched). The result aliases the
        # input field (any subsequent op reading it sees the same C++ MultiFab, now with filled
        # halos). Forwards to ctx.fill_boundary (the shared transport-BC ghost exchange).
        (x,) = v.inputs
        lines.append("ctx.fill_boundary(%s);" % var[x.id])
        var[v.id] = var[x.id]
    elif v.op == "project":
        # In-place positivity projection of the state (the block's own project closure). The result
        # aliases the input state. Forwards to ctx.apply_projection(idx, state) (ADC-426: the op's
        # own block, so each block runs its own projection).
        (state_in,) = v.inputs
        lines.append("ctx.apply_projection(%d, %s);" % (bidx, var[state_in.id]))
        var[v.id] = var[state_in.id]
    elif v.op == "cell_compare":
        # A PER-CELL threshold (spec op 17, ADC-418): mask(i,j,0) = field(i,j,0) <cmp> value ? 1 : 0,
        # a fresh 1-component scalar_field. Lowered to a for_each_cell select kernel (the mask the
        # `where` op selects on); no aux / model needed -- it reads component 0 of the input field.
        (field_in,) = v.inputs
        var[v.id] = "m%d" % v.id
        lines.append("pops::MultiFab %s = ctx.alloc_scalar_field(1, 1);" % var[v.id])
        lines += _emit_cell_compare_kernel(var[field_in.id], var[v.id], v.attrs["cmp"],
                                           v.attrs["value"])
    elif v.op == "where":
        # A PER-CELL conditional select (spec op 17, ADC-418): out(i,j,c) = mask ? a(i,j,c) :
        # b(i,j,c), COMPONENT-WISE. A fresh scratch the same shape as `a` (its vtype / ncomp); the
        # ternary is decided per cell inside the kernel (NOT a scalar runtime branch -- that is if_).
        mask_in, a_in, b_in = v.inputs
        var[v.id] = "w%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], var[a_in.id]))
        lines += _emit_where_kernel(var[mask_in.id], var[a_in.id], var[b_in.id], var[v.id])
    elif v.op == "record_scalar":
        # Store the (already-computed) Scalar into the System diagnostics map under its name. A
        # side-effecting op; its var maps to the recorded scalar (a harmless alias). The scalar input
        # is a 'reduce' result emitted earlier in the body (a const pops::Real local).
        (scalar_in,) = v.inputs
        lines.append("ctx.record_scalar(%s, %s);"
                     % (json.dumps(v.attrs["diagnostic"]), var[scalar_in.id]))
        var[v.id] = var[scalar_in.id]
    elif v.op == "rhs":
        state_in = v.inputs[0]  # rhs inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        named_fluxes = _named_fluxes(v)
        requested = v.attrs.get("sources")
        want_flux = v.attrs.get("flux", True)
        # ADC-425 routing (spec criterion 17): the default/composite source is folded in iff the
        # caller did NOT exclude it -- i.e. sources is None (the legacy default) OR "default" is in
        # the explicit list. An EMPTY list [] (or a list of only named sources) excludes it -> flux
        # only. None and [] are recorded distinctly in the IR, so this is unambiguous.
        want_default_source = requested is None or "default" in requested
        if not want_flux:
            # SOURCE-ONLY (ADC-430): flux=False -- NO -div F base (the rhs_scratch starts at zero).
            # The default/composite source is added iff requested (the same want_default_source
            # routing as flux=True): "default" present (or None) -> ctx.source_default_into (S only,
            # the exact mirror of neg_div_flux_default_into); excluded -> R stays the zeroed scratch.
            # The named source_terms below axpy on top either way -- so flux=False,sources=["default"]
            # is the default source only; flux=False,sources=["s"] is just s; flux=False,sources=[]
            # is the zero RHS. Named fluxes are rejected upstream (no flux base to divide). This is
            # the fix: before ADC-430 a flux=False stage still emitted the -div F base (it ignored the
            # flux attr), double-adding the flux on any non-zero-flux model in a Lie/Strang split.
            if want_default_source:
                lines.append("ctx.source_default_into(%d, %s, %s);"
                             % (bidx, var[state_in.id], var[v.id]))
        elif named_fluxes is None:
            if want_default_source:
                # R <- -div F + default/composite source (ctx.rhs_into) for THIS op's block (ADC-426
                # bidx), the historical path: sources is None (legacy) or "default" is requested.
                lines.append("ctx.rhs_into(%d, %s, %s);" % (bidx, var[state_in.id], var[v.id]))
            else:
                # FLUX-ONLY (ADC-425): "default" is NOT among the requested sources (the empty list
                # [] or a named-only list) -> R <- -div F(U) WITHOUT the model's default source
                # (ctx.neg_div_flux_default_into), for THIS op's block (bidx). The named source_terms
                # below are then axpy'd on top -- sources=[] is flux only, ["a","b"] is flux + a + b.
                lines.append("ctx.neg_div_flux_default_into(%d, %s, %s);"
                             % (bidx, var[state_in.id], var[v.id]))
        else:
            # NAMED fluxes (ADC-419): R <- -div(sum of selected named fluxes). Evaluate the SUM of
            # the flux expressions per direction into two n_cons scratch fields (fx / fy) by a
            # per-cell kernel, then take the negated centered FV divergence into R. Linear in the
            # named pieces -> splitting the physical flux into named pieces that sum to it gives the
            # SAME -div (to round-off). Distinct stencil from rhs_into (centered FV vs Riemann), so
            # this path is NEVER mixed with the default (guarded by _named_fluxes).
            fx = "%s_fx" % var[v.id]
            fy = "%s_fy" % var[v.id]
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fx, var[state_in.id]))
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fy, var[state_in.id]))
            lines += _emit_flux_kernel(model, named_fluxes, var[state_in.id], fx, fy)
            lines.append("ctx.neg_div_flux_into(%s, %s, %s);" % (var[v.id], fx, fy))
        named = [s for s in (v.attrs.get("sources") or []) if s != "default"]
        for s in named:
            # R += S_s(U, aux): assemble the named source into a scratch (same per-cell kernel as
            # the standalone 'source' op) and axpy it onto R.
            ssrc = "%s_%s" % (var[v.id], s)
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (ssrc, var[state_in.id]))
            lines += _emit_source_kernel(model, s, var[state_in.id], ssrc)
            lines.append("ctx.axpy(%s, static_cast<pops::Real>(1), %s);" % (var[v.id], ssrc))
    elif v.op == "source":
        state_in = v.inputs[0]  # source inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        lines += _emit_source_kernel(model, v.attrs["source"], var[state_in.id], var[v.id])
    elif v.op == "apply":
        state_in = v.inputs[0]  # apply inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        lines += _emit_apply_kernel(model, v.attrs["linear_source"], var[state_in.id], var[v.id])
    elif v.op == "solve_local_linear":
        rhs_in = v.inputs[0]  # solve inputs = (rhs_state, op_value[, fields]); rhs first
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);"
                     % (var[v.id], var[base.id]))
        lines += _emit_solve_local_linear_kernel(
            model, v.attrs["linear_source"], v.attrs["a_coeff"], var[rhs_in.id], var[v.id])
    elif v.op == "solve_local_nonlinear":
        # Per-cell Newton (spec op 10): solve residual(U) = 0 from the initial guess U0, cell by
        # cell, with an in-kernel FD Jacobian + the SAME stack dense inverse solve_local_linear
        # uses. The output is a fresh scratch state; the guess input seeds the iterate.
        guess_in = v.inputs[0]  # solve inputs = (initial_guess,)
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);"
                     % (var[v.id], var[base.id]))
        lines += _emit_solve_local_nonlinear_kernel(model, v, var[guess_in.id], var[v.id])
    elif v.op == "schur_coeffs":
        # Anisotropic condensed-Schur coefficient bundle (ADC-421): allocate the four 1-component
        # coefficient fields ONCE (persistent shared_ptr in the prelude, captured by the apply
        # lambda) and FILL them per step in the body from the live state + B_z aux. The bundle's var
        # is the 4 shared_ptr names; apply_laplacian_coeff dereferences them inside the apply.
        if prelude is None:
            raise NotImplementedError(
                "schur_coeffs is only lowerable at the top level / step body, not inside a "
                "control-flow (if/while/range) body")
        _emit_schur_coeffs(program, v, var, lines, prelude)
    elif v.op == "scalar_field":
        # A step-body scratch scalar field (e.g. the explicit-flux buffer the RHS assembly fills):
        # a persistent shared_ptr (prelude, alloc-once) reused every step. Inside an apply sub-block
        # the scalar_field is handled by _emit_matrix_free_operator instead (this branch is the
        # top-level / step-body path -- prelude is not None there).
        if prelude is None:
            raise NotImplementedError(
                "scalar_field is only lowerable at the top level / step body or inside a "
                "matrix_free_operator apply sub-block, not inside a control-flow (if/while/range) body")
        sp = "sf%d" % v.id
        var[v.id] = "(*%s)" % sp
        ncomp = int(v.attrs.get("ncomp", 1))
        prelude.append("auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
                       % (sp, ncomp))
    elif v.op == "schur_explicit_flux":
        # F = B^{-1} (mx, my) per cell into a 2-component scalar field (Fx comp 0, Fy comp 1): the
        # explicit condensed-Schur flux, a step-body one-shot (not a per-iteration apply).
        out_in, state_in = v.inputs
        lines.append("ctx.schur_explicit_flux(%s, %s, %s, %d, %d, %d);"
                     % (_deref(var[out_in.id]), var[state_in.id], _coeff_cpp(v.attrs["th_dt"]),
                        v.attrs["c_mx"], v.attrs["c_my"], v.attrs["c_bz"]))
        var[v.id] = var[out_in.id]
    elif v.op == "schur_rhs":
        # Fused condensed-Schur RHS = -Lap(phi^n) - g*div(F) into a 1-component scalar field: the
        # native assemble_rhs in one ctx call (no scalar-field affine combine at IR level).
        out_in, phi_in, state_in = v.inputs
        lines.append(
            "ctx.assemble_schur_rhs(%s, %s, %s, %s, %s, %d, %d, %d);"
            % (_deref(var[out_in.id]), _deref(var[phi_in.id]), var[state_in.id],
               _coeff_cpp(v.attrs["th_dt"]), _coeff_cpp(v.attrs["g"]), v.attrs["c_mx"],
               v.attrs["c_my"], v.attrs["c_bz"]))
        var[v.id] = var[out_in.id]
    elif v.op == "laplacian":
        # Step-body bare Laplacian (e.g. Lap phi^n for the condensed RHS). Inside an apply sub-block
        # this op is handled by _emit_matrix_free_operator; here it is the top-level path.
        o, i = v.inputs
        lines.append("ctx.laplacian(%s, %s);" % (_deref(var[o.id]), _deref(var[i.id])))
        var[v.id] = var[o.id]
    elif v.op == "gradient":
        o, p = v.inputs
        lines.append("ctx.gradient(%s, %s);" % (_deref(var[o.id]), _deref(var[p.id])))
        var[v.id] = var[o.id]
    elif v.op == "divergence":
        o, fx, fy = v.inputs
        lines.append("ctx.divergence(%s, %s, %s);"
                     % (_deref(var[o.id]), _deref(var[fx.id]), _deref(var[fy.id])))
        var[v.id] = var[o.id]
    elif v.op == "schur_reconstruct":
        # In-place velocity reconstruction v = B^{-1}(v^n - theta dt grad phi); mom = rho v. Result
        # aliases the input state (mx/my overwritten). phi is a scalar_field / 1-comp State token.
        state_in, phi_in = v.inputs
        lines.append("ctx.schur_reconstruct(%s, %s, %s, %d, %d, %d, %d);"
                     % (var[state_in.id], _deref(var[phi_in.id]), _coeff_cpp(v.attrs["th_dt"]),
                        v.attrs["c_rho"], v.attrs["c_mx"], v.attrs["c_my"], v.attrs["c_bz"]))
        var[v.id] = var[state_in.id]
    elif v.op == "schur_energy":
        # In-place energy increment E += (1/2) rho (|v^{n+1}|^2 - |v^n|^2) (ADC-427). Reads v^{n+1}
        # from the updated state and v^n / E^n from state_old (U^n); rho frozen (same in both).
        state_in, old_in = v.inputs
        lines.append("ctx.schur_energy(%s, %s, %d, %d, %d, %d);"
                     % (var[state_in.id], var[old_in.id], v.attrs["c_rho"], v.attrs["c_mx"],
                        v.attrs["c_my"], v.attrs["c_E"]))
        var[v.id] = var[state_in.id]
    elif v.op == "matrix_free_operator":
        # Install-time: emit the apply lambda `apply_A{id}` into the prelude. Its persistent scratch
        # (the scalar_field ops of the apply sub-block) are shared_ptr fields, captured by value so
        # they outlive the install call and are reused across every Krylov iteration (alloc-once).
        # The lambda is itself captured by the step closure ([=]) and passed to pops::*_solve. An
        # rhs_jacvec apply (ADC-431) also captures persistent jac_uk / jac_r0 scratch the lambda
        # dereferences; the step body refreshes them from the live iterate / rhs(U^k) here (@p lines).
        _emit_matrix_free_operator(program, v, var, prelude, lines)
    elif v.op in ("apply_in", "apply_out", "apply_laplacian_coeff"):
        # The lambda in/out placeholders and the coefficiented apply matvec only appear INSIDE a
        # matrix_free_operator apply sub-block (lowered by _emit_matrix_free_operator); they never
        # lower standalone at the top level.
        raise NotImplementedError(
            "emit_cpp_program: op '%s' (value '%s') is only lowerable inside a matrix_free_operator "
            "apply sub-block" % (v.op, v.name))
    elif v.op == "solve_linear":
        _emit_solve_linear(program, v, base, var, prelude, lines)
    elif v.op == "reduce":
        # A collective all_reduce -> a C++ scalar. norm2 = sqrt(dot(u, u)); dot(a, b) directly;
        # sum/max/min (over a component) via the matching pops reduction. All MUST run on every rank
        # (the reductions are collective all_reduce); they sit at the top of the loop body.
        var[v.id] = "s%d" % v.id
        kind = v.attrs["kind"]
        if kind == "norm2":
            (u,) = v.inputs
            lines.append("const pops::Real %s = std::sqrt(pops::dot(%s, %s));"
                         % (var[v.id], var[u.id], var[u.id]))
        elif kind == "norm_inf":
            (u,) = v.inputs
            lines.append("const pops::Real %s = pops::norm_inf(%s);" % (var[v.id], var[u.id]))
        elif kind in ("sum", "max", "min"):
            (u,) = v.inputs
            comp = int(v.attrs.get("comp", 0))
            lines.append("const pops::Real %s = pops::reduce_%s(%s, %d);"
                         % (var[v.id], kind, var[u.id], comp))
        else:  # dot
            a, b = v.inputs
            lines.append("const pops::Real %s = pops::dot(%s, %s);"
                         % (var[v.id], var[a.id], var[b.id]))
    elif v.op == "cfl":
        # The dt_bound's runtime cfl argument -- the C++ parameter of pops_program_dt_bound. It is
        # NOT a statement; its token is the bound parameter name (spec s18 / ADC-417).
        var[v.id] = "cfl"
    elif v.op == "hmin":
        # MIN physical cell size (ctx.hmin(), = the native CFL's hmin). A scalar local (spec s18).
        var[v.id] = "s%d" % v.id
        lines.append("const pops::Real %s = ctx.hmin();" % var[v.id])
    elif v.op == "max_wave_speed":
        # Max |wave speed| of the block on the state (ctx.max_wave_speed(idx, u)): the SAME per-block
        # reduction the native CFL reads, REUSED (spec s18). A collective reduction -> a scalar local.
        # ADC-426: the wave speed of the input state's OWN block (idx of u.block).
        (u,) = v.inputs
        var[v.id] = "s%d" % v.id
        lines.append("const pops::Real %s = ctx.max_wave_speed(%d, %s);"
                     % (var[v.id], (block_idx or {}).get(u.block, 0), var[u.id]))
    elif v.op == "scalar_op":
        # Scalar arithmetic (add/sub/mul/div) over scalar locals / literal constants -> a new scalar
        # local. Used by the dt_bound expression cfl * hmin / max_wave_speed (spec s18).
        var[v.id] = "s%d" % v.id
        toks = []
        for kind, val in v.attrs["operands"]:
            if kind == "v":
                toks.append(var[v.inputs[val].id])
            else:  # a literal constant
                toks.append("static_cast<pops::Real>(%s)" % repr(float(val)))
        cppop = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[v.attrs["fn"]]
        lines.append("const pops::Real %s = (%s %s %s);"
                     % (var[v.id], toks[0], cppop, toks[1]))
    elif v.op == "compare":
        # A predicate over scalars -> an inline boolean C++ expression (no statement of its own; the
        # while op embeds it directly in `if (!(<expr>)) break;`).
        lhs = v.inputs[0]
        if len(v.inputs) == 2:  # scalar vs scalar
            rhs_tok = var[v.inputs[1].id]
        else:  # scalar vs float tolerance
            rhs_tok = "static_cast<pops::Real>(%s)" % repr(float(v.attrs["rhs"]))
        var[v.id] = "(%s %s %s)" % (var[lhs.id], v.attrs["cmp"], rhs_tok)
        program._when_tokens[v.id] = var[v.id]  # reusable as a when(cond) due test (ADC-458)
    elif v.op == "while":
        _emit_while(program, v, base, var, model, lines, block_idx)
    elif v.op == "range":
        _emit_range(program, v, base, var, model, lines, block_idx)
    elif v.op == "if":
        _emit_if(program, v, base, var, model, lines, block_idx)
    elif v.op == "linear_combine":
        terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
        if v.id in committed_ids:
            # Commit: block state <- c_base * base + sum(non-base coeff * term), in place.
            c_base = {0: 0.0}
            acc = "acc%d" % v.id
            lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (acc, var[base.id]))
            for inp, coeff in terms:
                if inp.id == base.id:
                    c_base = coeff
                else:
                    lines.append("ctx.axpy(%s, %s, %s);" % (acc, _coeff_cpp(coeff), var[inp.id]))
            lines.append("ctx.lincomb(%s, %s, %s, static_cast<pops::Real>(1), %s);"
                         % (var[base.id], _coeff_cpp(c_base), var[base.id], acc))
            var[v.id] = var[base.id]  # the commit wrote the block state in place (no final copy)
        else:
            var[v.id] = "u%d" % v.id  # an intermediate stage state (scratch, zero-initialized)
            lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], var[base.id]))
            for inp, coeff in terms:
                lines.append("ctx.axpy(%s, %s, %s);" % (var[v.id], _coeff_cpp(coeff), var[inp.id]))
    # UNIFIED SCHEDULER (ADC-458, Spec 3 sections 17-18): if this op carries a non-always schedule,
    # wrap the statements it just emitted (lines[_profile_start:]) in the due-test guard + policy
    # branch. Done HERE, after the op lowered itself, so EVERY schedulable node (field solve, rhs,
    # source, linear_combine, where, ...) reuses the one general mechanism -- no per-op special
    # case. The wrap nests INSIDE the per-node profiling pair below (the profiler times the guarded
    # block as the node's cost). An always() schedule (or no schedule) leaves the lines untouched.
    _emit_schedule_wrap(program, v, var, lines, _profile_start)
    # PER-NODE PROFILING (ADC-459): if this op emitted at least one statement, bracket those
    # statements with the steady_clock pair (see the note at the top of _emit_op). A ProfileScope is
    # named "node:<v.name>"; profile_record(name, _pt) accumulates now() - _pt into the System
    # Profiler. Inserted only when lines grew (a pure inline-token op emits nothing and is skipped).
    # The pure reference-binding ops (state / history bind a MultiFab&; hmin reads a cached scalar)
    # do no per-step numerical work, so they are not wrapped -- the report keeps the meaningful
    # work nodes (rhs / solve_fields / linear_combine / source / apply / reductions / loops).
    if v.op not in _PROFILE_SKIP_OPS and len(lines) > _profile_start:
        node_name = json.dumps("node:%s" % v.name)
        pt = "_pt%d" % v.id  # unique per node id (no redefinition at body scope or in a loop pass)
        lines.insert(_profile_start,
                     "const auto %s = std::chrono::steady_clock::now();  // ProfileScope %s"
                     % (pt, node_name))
        lines.append("ctx.profile_record(%s, %s);" % (node_name, pt))

# Ops whose schedulable OUTPUT is the System aux (phi / grad / E), not a step-body scratch: a held
# field solve caches and restores the aux. Every other schedulable op writes a named scratch
# MultiFab (var[v.id]) that the scratch cache holds/restores instead.

def _schedule_due_test(program, v, sched):
    """The C++ boolean 'is this node due this step' for a non-subcycle schedule kind. Reused as the
    guard of the policy branch. Raises (naming ADC-458) for a kind that needs a runtime primitive the
    compiled .so does not have (on_end: no end-of-run signal reaches a sim.step(dt) loop)."""
    kind = sched.kind
    if kind == "every":
        # Cadence: due cold-start, then every N macro-steps (CacheManager::is_due via macro_step()).
        return "ctx.cache_should_update(%d, %d)" % (v.id, int(sched.params.get("n", 1)))
    if kind == "on_start":
        return "(ctx.macro_step() == 0)"
    if kind == "when":
        # A runtime predicate: a Program Bool value already lowered to a parenthesized C++ expr token
        # (a compare over reductions). A bare Python callable cannot lower (it is not a Program value).
        cond = sched.params.get("cond")
        if not (isinstance(cond, Value) and cond.vtype == "bool"):
            raise NotImplementedError(
                "when(cond) lowers only a Program Bool predicate (e.g. P.norm2(r) < tol), not a "
                "Python callable: node %r (ADC-458). Build the condition with Program compares."
                % v.name)
        if cond.id not in program._when_tokens:
            raise ValueError(
                "when(cond) on node %r references a Bool value not emitted before it; build the "
                "predicate earlier in the Program (ADC-458)" % v.name)
        return program._when_tokens[cond.id]
    raise NotImplementedError(
        "schedule kind %r on node %r is not lowerable: on_end() needs an end-of-run signal that a "
        "compiled sim.step(dt) loop never sees (the .so cannot know the last step); use on_start()/"
        "every()/when()/subcycle() or an on_end host hook (ADC-458)." % (kind, v.name))

def _emit_schedule_wrap(program, v, var, lines, start):
    """Wrap the C++ statements node @p v emitted (``lines[start:]``) in its schedule's due-test guard
    + policy branch (ADC-458, Spec 3 sections 17-18). Generic over the op: a field solve caches the
    System aux, any other node caches its named scratch (var[v.id]). An always()/absent schedule
    leaves the lines untouched (byte-identical to the unscheduled lowering)."""
    sched = v.attrs.get("schedule")
    if sched is None or sched.is_always():
        return
    body = lines[start:]
    del lines[start:]
    if sched.kind == "subcycle":
        # Structured sub-cycling of a field solve (the gate restricts subcycle to an aux-output op):
        # run the op body COUNT times over the sub-dt (macro_dt / count by default, or an explicit
        # dt), refreshing the persistent System aux each pass. A pure recompute cadence -- no cache.
        # The sub-dt is a const local exposed for a dt-scaled body; the MVP field-solve body is
        # dt-free, so it documents the cadence (the aux solve re-runs COUNT times).
        count = int(sched.params["count"])
        sub_dt = sched.params.get("dt")
        sd = "_subdt%d" % v.id
        if sub_dt is None:
            lines.append("const pops::Real %s = dt / static_cast<pops::Real>(%d);" % (sd, count))
        else:
            lines.append("const pops::Real %s = static_cast<pops::Real>(%s);"
                         % (sd, repr(float(sub_dt))))
        lines.append("(void)%s;" % sd)  # the MVP body is self-contained; sd documents the cadence
        lines.append("for (int _sub%d = 0; _sub%d < %d; ++_sub%d) {" % (v.id, v.id, count, v.id))
        lines += ["  " + ln for ln in body]
        lines.append("}")
        return
    due = _schedule_due_test(program, v, sched)
    policy = sched.policy
    is_aux = v.op in _AUX_OUTPUT_OPS
    # The scratch node's output token (the MultiFab the policy holds / zeroes). A field solve writes
    # the System aux and sets no var[v.id], so out is read only on the scratch path.
    out = None if is_aux else var.get(v.id)
    if policy == "recompute":
        # Run only when due; on a NOT-due step do nothing (the aux / scratch keeps its last content).
        # recompute off-cadence is simply 'run when due' -- no cache, no else branch. A scratch node
        # hoists its output declaration so the buffer stays in scope when the body does not run.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if policy == "skip":
        # Do not run the op off-cadence: the value keeps its previous content (the cacheable contract
        # -- downstream must tolerate a stale value). A scratch node hoists its output declaration so
        # the stale buffer stays in scope across the guard (no else branch: nothing happens off-
        # cadence); a field solve writes the persistent aux, so its whole body simply guards.
        if is_aux:
            lines.append("if (%s) {  // skip: stale aux off-cadence" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {  // skip: stale value off-cadence" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if policy == "zero":
        # Off-cadence, zero the node's output. The output must EXIST in both branches: for a scratch
        # node hoist its allocation out of the guard (the first emitted line declares var[v.id]); the
        # aux always exists (System-owned).
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  ctx.aux().set_val(static_cast<pops::Real>(0));")
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  %s.set_val(static_cast<pops::Real>(0));" % out)
            lines.append("}")
        return
    if policy == "hold":
        # Recompute + cache when due; restore the cached value off-cadence (no recompute). The aux
        # path uses cache_store_aux/restore_aux; a scratch node hoists its allocation and uses the
        # named-scratch cache. _validate_schedule already rejected hold on a non-cacheable operator.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if policy == "accumulate_dt":
        # Off-cadence: accumulate THIS step's dt (the real skipped dt, never N*dt_current) and hold the
        # cached value. When due: read eff_dt = dt + sum(skipped) (resets the accumulator), recompute,
        # cache. eff_dt is bound so a dt-dependent recompute can read it (the MVP field solve / scratch
        # fill is dt-free, but eff_dt is exposed for a dt-scaled body). Cacheable (validated upstream).
        ed = "_effdt%d" % v.id
        if is_aux:
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if policy == "error":
        # Guard that a stale value is never read off-cadence: run when due, else fail loud (the node
        # asserts it is only consumed on its cadence). Emitted as a runtime throw on the not-due path.
        # A scratch node hoists its output declaration so the buffer stays in scope (the throw never
        # returns, but the C++ must still be well-scoped); a field solve guards the aux body directly.
        err = ('ctx.scheduler_error(%s);'
               % json.dumps("node '%s' (op '%s') read off its schedule cadence (policy=error)"
                            % (v.name, v.op)))
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        return
    raise NotImplementedError(
        "schedule policy %r on node %r is not lowerable (ADC-458)" % (policy, v.name))

def _split_output_decl(program, body, out, v):
    """Split a scratch node's emitted @p body into (declaration_line, rest): the OUTPUT scratch
    ``out`` must be declared OUTSIDE the policy guard so both branches see it, while the fill stays
    inside. The op declares its output as its FIRST emitted line (``pops::MultiFab <out> = ...;``);
    hoist exactly that one line. Raises if the shape is unexpected (a node whose output is not a
    freshly-declared scratch cannot use a cache/zero policy through this path)."""
    decl_prefix = "pops::MultiFab %s = " % out
    if not body or not body[0].startswith(decl_prefix):
        raise NotImplementedError(
            "schedule policy on node %r (op '%s') needs its output scratch %r declared as its first "
            "emitted line to hoist it out of the guard; got %r (ADC-458)"
            % (v.name, v.op, out, body[0] if body else None))
    return body[0], body[1:]

def _emit_while(program, v, base, var, model, lines, block_idx=None):
    """Lower a while op to an infinite C++ loop with a break (the condition re-evaluates each pass).
    The loop variable is a single MultiFab mutated IN PLACE across iterations; the cond / body sub-
    blocks re-run the per-op lowering each pass, with the loop-variable value id seeded to the loop
    var so their references resolve to it."""
    loop_in = v.inputs[0]  # the initial loop-variable state
    x = "x%d" % v.id
    var[v.id] = x
    # Hoist + initialize the loop variable from the entry state (x <- loop_in).
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (;;) {")
    # The sub-blocks see the loop variable in place of the entry-state value id (the body / cond were
    # built reading the loop-var State; they resolve to x here). A fresh sub-var map keeps the inner
    # scratch names from leaking out, but inherits the outer bindings (the loop var, target, ...).
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["cond_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    cond_expr = sub[v.attrs["cond"].id]
    body_lines.append("if (!(%s)) break;" % cond_expr)
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    # Write the next state into the loop variable in place (x <- body result).
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_range(program, v, base, var, model, lines, block_idx=None):
    """Lower a range op to a C++ ``for`` over a fixed count. Like a while, the loop variable is one
    MultiFab mutated in place and the body sub-block is emitted ONCE inside the loop (re-run each
    pass at runtime); the loop-variable value id is seeded to the loop var for the sub-block."""
    loop_in = v.inputs[0]
    x = "x%d" % v.id
    i = "i%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (int %s = 0; %s < %d; ++%s) {" % (i, i, int(v.attrs["count"]), i))
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_if(program, v, base, var, model, lines, block_idx=None):
    """Lower an if op to a C++ branch. @p cond was emitted at the top level (its boolean expression
    is var[cond.id]); the loop variable is a copy of the input state, overwritten in place only when
    the branch is taken (so the result is the input state when the condition is false at runtime)."""
    state_in, cond = v.inputs[0], v.inputs[1]
    x = "x%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[state_in.id]))
    lines.append("if (%s) {" % var[cond.id])
    sub = dict(var)
    sub[state_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_schur_coeffs(program, v, var, lines, prelude):
    """Lower a schur_coeffs bundle (ADC-421): allocate the four 1-component coefficient fields
    (eps_x, eps_y, a_xy, a_yx) ONCE as persistent shared_ptrs (prelude, alloc-once, captured by the
    step closure and by the apply lambda that consumes them) and FILL them per step in the body from
    the live state + B_z aux via ``ctx.assemble_schur_coeffs`` (the SAME native
    detail::SchurOperatorCoeffKernel). The bundle's token is the tuple of the four shared_ptr names;
    ``apply_laplacian_coeff`` dereferences them inside the matrix-free apply."""
    (state_in,) = v.inputs
    ex = "ceps_x%d" % v.id
    ey = "ceps_y%d" % v.id
    axy = "ca_xy%d" % v.id
    ayx = "ca_yx%d" % v.id
    for sp in (ex, ey, axy, ayx):
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(1, 1));" % sp)
    var[v.id] = (ex, ey, axy, ayx)  # the bundle token: the four coefficient shared_ptr names
    lines.append(
        "ctx.assemble_schur_coeffs(*%s, *%s, *%s, *%s, %s, %s, %s, %d, %d);"
        % (ex, ey, axy, ayx, var[state_in.id], _coeff_cpp(v.attrs["c"]),
           _coeff_cpp(v.attrs["th_dt"]), v.attrs["c_rho"], v.attrs["c_bz"]))

def _emit_matrix_free_operator(program, v, var, prelude, lines=None):
    """Lower a matrix_free_operator to an INSTALL-TIME C++ apply lambda ``apply_A{id}`` (appended to
    @p prelude). The lambda has the pops::ApplyFn signature ``(pops::MultiFab& out, const pops::MultiFab&
    in)``; its body re-emits the apply sub-block:

      - each ``scalar_field`` scratch -> a PERSISTENT shared_ptr field (declared in the prelude
        BEFORE the lambda, captured by value), reused across every Krylov iteration (alloc-once);
      - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
        which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
      - ``rhs_jacvec(out, in, iterate, r0, ...)`` (ADC-431) -> a finite-difference Jacobian-vector
        product ``out = in - (c*dt/eps)(rhs(U^k + eps*in) - rhs(U^k))`` calling ``ctx.rhs_into`` (or
        ``neg_div_flux_default_into``) on PERSISTENT jac_uk / jac_r0 scratch the lambda captures; the
        step body refreshes that scratch from the live iterate / rhs(U^k) (@p lines, see below);
      - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
        ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

    The lambda captures ``[ctx, <scratch shared_ptrs>]``; the step closure captures it by value. @p
    lines is the step-body line list (for the rhs_jacvec scratch refresh); None when the operator has
    no jacvec op (the historical matrix-free path, prelude only)."""
    apply_id = v.id
    lam = "apply_A%d" % apply_id
    var[apply_id] = lam
    in_sf = v.attrs["apply_in"]
    out_sf = v.attrs["apply_out"]
    block = v.attrs["apply_block"]
    result = v.attrs["apply_result"]
    # Sub-scope token map: the lambda params + persistent scratch. `in` is the const lambda param;
    # `out` is the (non-const) lambda param the result is written into.
    sub = {in_sf.id: "in", out_sf.id: "out"}
    # 1) Persistent scratch (the scalar_field ops): one shared_ptr per scratch, declared before the
    #    lambda so it is in scope to capture. Collected first so the capture list is known.
    scratch = [w for w in block if w.op == "scalar_field"]
    captures = ["ctx"]
    for w in scratch:
        sp = "sf%d_%d" % (apply_id, w.id)
        sub[w.id] = sp
        ncomp = int(w.attrs.get("ncomp", 1))  # >1 for a gradient buffer consumed by divergence
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
            % (sp, ncomp))
        captures.append(sp)
    # The affine result-write accumulator: one PERSISTENT shared_ptr (alloc-once, like the scratch),
    # zeroed and reused every matvec instead of allocated per call -- so the apply lambda allocates
    # NOTHING per Krylov iteration (the runtime r/p/Ap scratch in generic_krylov.hpp is likewise
    # alloc-once). _emit_field_combine writes the affine into `out` through it. It carries the
    # operator's component count so the axpy / lincomb cover ALL components (a vector / state apply).
    op_ncomp = int(v.attrs["ncomp"])
    acc_sp = "acc%d" % apply_id
    prelude.append(
        "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
        % (acc_sp, op_ncomp))
    captures.append(acc_sp)
    # A coefficiented apply (apply_laplacian_coeff) reads an OUTER schur_coeffs bundle (assembled in
    # the step body, before the operator): capture its four coefficient shared_ptrs (already
    # allocated in the prelude by _emit_schur_coeffs) so the lambda can dereference them.
    for w in block:
        if w.op == "apply_laplacian_coeff":
            coeffs = w.inputs[2]
            for sp in var[coeffs.id]:
                if sp not in captures:
                    captures.append(sp)
    # An rhs_jacvec apply (ADC-431, implicit-flux BDF) needs the FROZEN Newton iterate U^k and its
    # precomputed rhs(U^k) inside the lambda. They are step-body locals that CHANGE each Newton
    # iteration, so -- like schur_coeffs -- they become PERSISTENT shared_ptr scratch (jac_uk / jac_r0)
    # captured by value (shared pointee), refreshed from the live iterate / r0 in the step body BEFORE
    # the solve. Plus a perturbed-state scratch (jac_up) and a perturbed-rhs scratch (jac_rp) the
    # lambda fills per matvec. All carry the operator's component count (= the block n_cons).
    jac_ops = [w for w in block if w.op == "rhs_jacvec"]
    if jac_ops and lines is None:
        raise NotImplementedError(
            "rhs_jacvec is only lowerable in a top-level / step-body matrix-free solve, not inside a "
            "control-flow (if/while/range) body (the Newton outer loop must be a static_range unroll)")
    jac_scratch = {}  # jacvec op id -> (uk, r0, up, rp, cdt) names
    ng_state = "ctx.state(0).n_grow()"  # the jacvec scratch needs the state's ghost count for rhs_into
    for w in jac_ops:
        uk = "jac_uk%d_%d" % (apply_id, w.id)
        r0 = "jac_r0%d_%d" % (apply_id, w.id)
        up = "jac_up%d_%d" % (apply_id, w.id)
        rp = "jac_rp%d_%d" % (apply_id, w.id)
        for sp in (uk, r0, up, rp):
            prelude.append(
                "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, %s));"
                % (sp, op_ncomp, ng_state))
            captures.append(sp)
        # The BDF coefficient c*dt depends on the step's dt (the step-closure parameter), which the
        # install-time lambda cannot see; carry it through a captured shared_ptr<Real> the step body
        # sets to its dt value before the solve (the same persistent-scratch idiom as jac_uk).
        cdt = "jac_cdt%d_%d" % (apply_id, w.id)
        prelude.append("auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % cdt)
        captures.append(cdt)
        jac_scratch[w.id] = (uk, r0, up, rp, cdt)
        # Step body: refresh the FROZEN captures from this iteration's live iterate / rhs(U^k) / dt.
        iterate_in, r0_in = w.inputs[2], w.inputs[3]
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (uk, uk, var[iterate_in.id]))
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (r0, r0, var[r0_in.id]))
        lines.append("*%s = %s;" % (cdt, _coeff_cpp(w.attrs["c_dt"])))
    # 2) The lambda body: the laplacian / gradient ops + the result write into `out`.
    body = []
    for w in block:
        if w.op in ("scalar_field", "apply_in", "apply_out"):
            continue  # scratch shared_ptr / lambda params: already bound in `sub`, nothing to emit
        if w.op == "laplacian":
            o, i = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.laplacian(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, i)))
        elif w.op == "gradient":
            o, p = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.gradient(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, p)))
        elif w.op == "divergence":
            o, fx, fy = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.divergence(*%s, %s, %s);"
                        % (sub[o.id], _apply_in_arg(sub, fx), _apply_in_arg(sub, fy)))
        elif w.op == "apply_laplacian_coeff":
            # out = div(A grad in), A the schur_coeffs tensor: forwards to the native
            # apply_laplacian coefficient path. eps_x/eps_y/a_xy/a_yx are the captured coeff fields.
            o, i, coeffs = w.inputs
            ex, ey, axy, ayx = var[coeffs.id]
            sub[w.id] = sub[o.id]
            body.append("ctx.apply_laplacian_coeff(*%s, %s, *%s, *%s, *%s, *%s);"
                        % (sub[o.id], _apply_in_arg(sub, i), ex, ey, axy, ayx))
        elif w.op == "rhs_jacvec":
            # out = J(U^k) in = in - (c*dt/h)(rhs(U^k + h*in) - rhs(U^k)), the finite-difference
            # Jacobian-vector product of the implicit-flux BDF residual (ADC-431). h is a relatively
            # scaled FD step (Brown-Saad / WP: h = eps*(1+||U^k||)/||in||, eps the relative step). The
            # captured jac_uk / jac_r0 hold U^k and rhs(U^k) (refreshed in the step body); jac_up /
            # jac_rp are per-matvec scratch; jac_cdt holds c*dt. The op writes directly into `out`.
            o, i = w.inputs[0], w.inputs[1]
            uk, r0, up, rp, cdt = jac_scratch[w.id]
            in_arg = _apply_in_arg(sub, i)        # the Krylov vector v (the lambda's const `in`)
            out_tok = sub[o.id]                   # the apply out buffer (== "out")
            eps = repr(float(w.attrs["eps"]))
            sub[w.id] = out_tok
            want_default = w.attrs.get("sources")
            want_default = want_default is None or "default" in want_default
            rhs_call = ("ctx.rhs_into(0, *%s, *%s);" if (w.attrs["flux"] and want_default)
                        else "ctx.neg_div_flux_default_into(0, *%s, *%s);") % (up, rp)
            body.append("{")
            # FD step norms via krylov_dot (all components when ncomp>1, component 0 otherwise --
            # the SAME reduction the Krylov loop uses for its residual norm).
            body.append("  const pops::Real jvn = std::sqrt(pops::detail::krylov_dot(%s, %s));"
                        % (in_arg, in_arg))
            body.append("  const pops::Real jukn = std::sqrt(pops::detail::krylov_dot(*%s, *%s));"
                        % (uk, uk))
            body.append("  const pops::Real jh = jvn > pops::Real(0) ? "
                        "static_cast<pops::Real>(%s) * (pops::Real(1) + jukn) / jvn "
                        ": static_cast<pops::Real>(%s);" % (eps, eps))
            # U^k + h*v -> jac_up; rhs(U^k + h*v) -> jac_rp (one rhs per matvec, U^k / rhs(U^k) frozen).
            body.append("  ctx.lincomb(*%s, pops::Real(1), *%s, jh, %s);" % (up, uk, in_arg))
            body.append("  %s" % rhs_call)
            # out = v - (c*dt/h)(rhs(U^k + h*v) - rhs(U^k)): lincomb then axpy back the frozen rhs(U^k).
            body.append("  const pops::Real jc = *%s / jh;" % cdt)
            body.append("  ctx.lincomb(%s, pops::Real(1), %s, -jc, *%s);" % (out_tok, in_arg, rp))
            body.append("  ctx.axpy(%s, jc, *%s);" % (out_tok, r0))
            body.append("}")
        else:
            raise NotImplementedError(
                "emit_cpp_program: op '%s' is not lowerable inside a matrix_free_operator apply "
                "(supported: scalar_field, laplacian, gradient, divergence, apply_laplacian_coeff, "
                "rhs_jacvec)" % w.op)
    body += _emit_field_combine(result, "out", sub, acc_sp)
    prelude.append("pops::ApplyFn %s = [%s](pops::MultiFab& out, const pops::MultiFab& in) {"
                   % (lam, ", ".join(captures)))
    prelude += ["  " + ln for ln in body]
    prelude.append("};")

def _emit_solve_linear(program, v, base, var, prelude, lines):
    """Lower solve_linear to a call into the runtime's matrix-free Krylov loop. The solution field
    ``sf_sol{id}`` is a PERSISTENT shared_ptr (prelude, captured by the step closure); the step body
    seeds the initial guess (zero, or a copy of the supplied guess), then calls
    ``pops::cg_solve`` / ``bicgstab_solve`` / ``richardson_solve`` with the operator's apply lambda.
    The KrylovResult is kept (diagnostics) but the trip count is decided C++-side, inside the loop --
    invisible to the IR. The result token is the solution field, dereferenced for the final copy back
    into the block state at commit."""
    op_value = v.inputs[0]
    rhs_in = v.inputs[1]
    guess_in = v.inputs[2] if v.attrs["has_guess"] else None
    lam = var[op_value.id]  # the apply lambda (already emitted into the prelude)
    sol_sp = "sf_sol%d" % v.id
    # The solution carries the operator's component count: a vector / state solve writes an ncomp
    # iterate (the Krylov scratch r/p/Ap is co-allocated from it, so the whole loop is ncomp-wide).
    op_ncomp = int(v.attrs.get("ncomp", 1))
    prelude.append(
        "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
        % (sol_sp, op_ncomp))
    var[v.id] = "(*%s)" % sol_sp  # token: the dereferenced solution MultiFab
    # Initial guess: zero (default) or a copy of the guess field.
    if guess_in is None:
        lines.append("%s->set_val(static_cast<pops::Real>(0));" % sol_sp)
    else:
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), "
                     "%s);" % (sol_sp, sol_sp, var[guess_in.id]))
    tol = "static_cast<pops::Real>(%s)" % repr(float(v.attrs["tol"]))
    max_iter = int(v.attrs["max_iter"])
    rhs_tok = var[rhs_in.id]
    method = v.attrs["method"]
    kr = "kr%d" % v.id
    if method == "cg":
        lines.append("pops::KrylovResult %s = pops::cg_solve(%s, *%s, %s, %s, %d);"
                     % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
    elif method == "bicgstab":
        # Identity preconditioner = empty ApplyFn (unpreconditioned BiCGStab).
        lines.append("pops::KrylovResult %s = pops::bicgstab_solve(%s, pops::ApplyFn{}, *%s, %s, %s, "
                     "%d);" % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
    elif method == "gmres":
        # Restarted GMRES(m): identity preconditioner = empty ApplyFn; restart = the basis size.
        restart = int(v.attrs["restart"])
        lines.append("pops::KrylovResult %s = pops::gmres_solve(%s, pops::ApplyFn{}, *%s, %s, %s, "
                     "%d, %d);" % (kr, lam, sol_sp, rhs_tok, tol, max_iter, restart))
    else:  # richardson: omega = 1 (the operator is expected to be pre-scaled / well-conditioned)
        lines.append("pops::KrylovResult %s = pops::richardson_solve(%s, *%s, %s, "
                     "static_cast<pops::Real>(1), %s, %d);"
                     % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
    lines.append("(void)%s;" % kr)


