"""pops.codegen.program_emit_model_kernels : the model-coefficient per-cell kernels.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These Phase-4b helpers emit the body of a for_each_cell
kernel over the VALID cells of each local fab from a physical model's symbolic coefficients
(source_term / flux_term / linear_source).  They reuse the shared primitives in
``program_emit_kernels`` (``_kernel_open`` / ``_cell_locals`` / ``_coeff_cpp`` / ...).
"""
from pops.codegen.program_emit_kernels import (
    _aux_comp,  # noqa: F401
    _cell_locals,
    _coeff_cpp,
    _kernel_close,
    _kernel_open,
    _model_impl,
)


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
