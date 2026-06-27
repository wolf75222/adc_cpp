"""pops.codegen.program_emit_kernels : shared codegen primitives for the program emitter.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  This is the LEAF module of the program emitter: the
emission-only op tables, the small text helpers (coeff rendering, field combine, aux
component resolution), the model-free per-cell kernels (cell_compare / where) and the
``_PROGRAM_CPP_TEMPLATE``.  ``dsl`` is imported LAZILY inside the helpers that need it.

The model-coefficient per-cell kernels (source / flux / apply / local solves) live in
``program_emit_model_kernels``; the dispatch core in ``program_emit_ops`` /
``program_emit_control``; the schedule wrap in ``program_emit_schedule``; the matrix-free
Krylov emitters in ``program_emit_solve``.  ``program_codegen`` re-imports every name so
its public surface is unchanged.
"""
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


# Source of a generated problem.so. The includes + pops_install_program closure match the shape
# tests/test_program_loader compiles+runs in CI; pops_program_hash is added per the spec .so ABI (a
# cache/restart key) and is not yet consumed by System::install_program. {name} is a JSON-escaped C
# string literal, {hash} the IR hash, {prelude} the INSTALL-TIME C++ (persistent scratch + matrix-free
# apply lambdas, captured into the step closure by [=]), {body} the step-closure body (both already
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
