"""pops.codegen.program_emit_ops : the per-op SSA -> C++ dispatcher.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  ``_emit_op`` lowers a SINGLE SSA op to C++ (appending
to the line list and recording its token), shared by the top-level body walk
(``program_emit_control._emit_body``) and the control-flow sub-blocks.  It dispatches to
the per-cell model kernels (``program_emit_model_kernels``), the matrix-free / Schur
emitters (``program_emit_solve``), the control-flow emitters (``program_emit_control``)
and the schedule wrap (``program_emit_schedule``).
"""
import json

from pops.codegen.program_emit_kernels import (
    _PROFILE_SKIP_OPS,
    _coeff_cpp,
    _deref,
    _emit_cell_compare_kernel,
    _emit_where_kernel,
    _named_fluxes,
)
from pops.codegen.program_emit_model_kernels import (
    _emit_apply_kernel,
    _emit_coupled_rate_kernel,
    _emit_flux_kernel,
    _emit_solve_local_linear_kernel,
    _emit_solve_local_nonlinear_kernel,
    _emit_source_kernel,
)
from pops.codegen.program_emit_control import (
    _coupled_rate_components,
    _emit_if,
    _emit_range,
    _emit_while,
)
from pops.codegen.program_emit_solve import (
    _emit_matrix_free_operator,
    _emit_schur_coeffs,
    _emit_solve_linear,
)
from pops.codegen.program_emit_schedule import _emit_schedule_wrap


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
