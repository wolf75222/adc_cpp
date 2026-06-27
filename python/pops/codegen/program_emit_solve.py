"""pops.codegen.program_emit_solve : matrix-free Krylov + condensed-Schur op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These three leaf emitters (called from
``program_emit_ops._emit_op`` for the schur_coeffs / matrix_free_operator / solve_linear
ops) build install-time apply lambdas + the Krylov solve calls; they never recurse back
into the op dispatcher.  They reuse the shared primitives in ``program_emit_kernels``.
"""
from pops.codegen.program_emit_kernels import (
    _apply_in_arg,
    _coeff_cpp,
    _emit_field_combine,
)


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
