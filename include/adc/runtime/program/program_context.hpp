#pragma once

#include <functional>
#include <string>
#include <utility>

#include <adc/core/foundation/types.hpp>  // Real, ADC_HD
#include <adc/coupling/schur/core/schur_condensation.hpp>  // SchurOperatorCoeffKernel / SchurExplicitFluxKernel (native coeff assembly, ADC-421)
#include <adc/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <adc/mesh/execution/for_each.hpp>  // for_each_cell (per-cell coeff / reconstruct kernels + negated divergence copy)
#include <adc/mesh/geometry/geometry.hpp>  // Geometry (mesh metric of the Laplacian / gradient)
#include <adc/mesh/storage/fab2d.hpp>      // Array4 / ConstArray4 (per-cell handles)
#include <adc/mesh/storage/mf_arith.hpp>   // saxpy (linear combine over a MultiFab)
#include <adc/mesh/storage/multifab.hpp>   // MultiFab
#include <adc/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess (centered gradient)
#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <adc/numerics/linalg/lorentz_eliminator.hpp>  // LorentzEliminator (closed B^{-1}, ADC-421 reconstruct)
#include <adc/runtime/context/grid_context.hpp>  // GridContext (System aux seam)
#include <adc/runtime/system.hpp>                // System (the runtime this facade forwards to)

/// @file
/// @brief ProgramContext -- the C++-side facade a generated problem.so calls to run a compiled time
///        Program during sim.step(dt) (epic ADC-399, ADC-401 Phase 2b).
///
/// It REIMPLEMENTS NOTHING. Each method forwards to an existing adc::System primitive:
///   install(fn)          -> System::install_program_step(fn)   (registers the macro-step body)
///   solve_fields()       -> System::solve_fields()             (elliptic solve + aux at current U)
///   solve_fields_from_state(b, U) -> System::solve_fields_from_state(b, U) (aux at a stage state)
///   n_blocks()           -> System::n_blocks()
///   state(b)             -> System::block_state(b)             (the block's live MultiFab, zero-copy)
///   rhs_into(b, U, R)    -> System::block_rhs_into(b, U, R)    (R <- -div F + S, Poisson frozen)
///   neg_div_flux_default_into(b, U, R) -> System::block_neg_div_flux_into (R <- -div F, NO source)
///   axpy(U, a, R)        -> adc::saxpy(U, a, R)                (U <- U + a R, device-dispatched)
///
/// The Program composes the chain (e.g. Forward Euler = solve_fields(); for each block:
/// rhs_into(b, U, R); axpy(U, dt, R)) and installs it via install(...). The .so NEVER touches
/// System::Impl / Array4 / fill_boundary / the elliptic solver / Kokkos / MPI / CFL / substeps.
///
/// IDIOM: ProgramContext is a plain (non-template) class holding a System*. A generated .so receives
/// the System as a flat void* across the dlopen boundary (like the native loader's `void* self`) and
/// wraps it here; it reaches per-block storage through the System's public accessors because
/// System::Impl is private to the _adc translation unit.
namespace adc {
namespace runtime {
namespace program {

namespace detail {

/// Aux-component-aware variants of the native Schur kernels (coupling/schur/core/schur_condensation.hpp
/// + condensed_schur_source_stepper.hpp). The native kernels read B_z from a DEDICATED B_z MultiFab at
/// component 0; a compiled Program reads B_z straight from the System aux channel at an arbitrary
/// component @c c_bz, so these thin wrappers carry @c c_bz and otherwise REPRODUCE the native formulas
/// verbatim (same LorentzEliminator B^{-1}, same coefficients, same centered gradient) -- the native
/// CondensedSchur path is untouched. Named functors (device-clean, nvcc cross-TU rule, like the native
/// ones). epic ADC-399 / ADC-421.

/// A_op = I + c*rho*B^{-1} per cell (eps_x/eps_y diag, a_xy/a_yx cross). Mirrors
/// detail::SchurOperatorCoeffKernel but reads B_z from the aux at c_bz.
struct SchurOperatorCoeffKernelC {
  ConstArray4 s;    ///< fluid state (rho at c_rho)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 ex, ey;    ///< output: eps_x, eps_y (diagonal of A)
  Array4 axy, ayx;  ///< output: cross terms a_xy, a_yx
  Real c;           ///< c = theta^2 dt^2 alpha
  Real th_dt;       ///< theta*dt (w = th_dt*B_z)
  int c_rho, c_bz;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    const Real cr = c * rho;  // c rho: common factor of the 4 entries of A - I
    ex(i, j, 0) = Real(1) + cr * le.binv_11();
    ey(i, j, 0) = Real(1) + cr * le.binv_22();
    axy(i, j, 0) = cr * le.binv_12();
    ayx(i, j, 0) = cr * le.binv_21();
  }
};

/// out = B^{-1} (mx, my) at the center (Fx in comp 0, Fy in comp 1): the explicit flux F = rho*B^{-1}*v.
struct SchurExplicitFluxKernelC {
  ConstArray4 s;    ///< fluid state (mx, my at c_mx / c_my)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 out;       ///< output: Fx (comp 0), Fy (comp 1)
  Real th_dt;       ///< theta*dt (w = th_dt*B_z)
  int c_mx, c_my, c_bz;
  ADC_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    Real Fx, Fy;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fx, Fy);  // B^{-1} (mx, my) = rho*B^{-1}*v
    out(i, j, 0) = Fx;
    out(i, j, 1) = Fy;
  }
};

/// rhs = -Lap phi^n - g*div(F), the centered FV divergence of the explicit flux F packed in ONE
/// 2-component buffer (Fx in comp 0, Fy in comp 1 -- the layout schur_explicit_flux writes), fused with
/// the already-negated -Lap phi^n. Mirrors detail::SchurRhsAssembleKernel verbatim except it reads both
/// flux components from the single buffer @c f instead of two separate fx/fy MultiFabs.
struct SchurRhsAssembleKernelC {
  ConstArray4 neg_lap;      ///< -Lap phi^n (already negated)
  ConstArray4 f;            ///< explicit flux F at the center (Fx comp 0, Fy comp 1; ghosts filled)
  Array4 rhs;               ///< output: condensed right-hand side
  Real g;                   ///< theta dt alpha
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy)
  ADC_HD void operator()(int i, int j) const {
    const Real divF =
        (f(i + 1, j, 0) - f(i - 1, j, 0)) * half_idx + (f(i, j + 1, 1) - f(i, j - 1, 1)) * half_idy;
    rhs(i, j, 0) = neg_lap(i, j, 0) - g * divF;
  }
};

/// Reconstruct v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi) and write mom = rho^n*v (rho frozen).
/// Mirrors detail::SchurReconstructKernel but reads B_z from the aux at c_bz (no separate vx/vy
/// buffers: v^n = (mx, my)/rho read inline from the state).
struct SchurReconstructKernelC {
  ConstArray4 phi;  ///< phi^{n+theta} (ghosts filled: centered grad reads i+-1, j+-1)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 st;        ///< fluid state (READ rho, mx, my; WRITE mx, my)
  Real th_dt;
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy) (centered gradient)
  int c_rho, c_mx, c_my, c_bz;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv_rho = rho != Real(0) ? Real(1) / rho : Real(0);
    const Real vx = st(i, j, c_mx) * inv_rho;  // v^n = (mx, my)/rho
    const Real vy = st(i, j, c_my) * inv_rho;
    const Real gx = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idx;  // d_x phi^{n+theta}
    const Real gy = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * half_idy;
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    Real nx, ny;
    le.apply_Binv(vx - th_dt * gx, vy - th_dt * gy, nx, ny);  // B^{-1}(v^n - theta dt grad phi)
    st(i, j, c_mx) = rho * nx;
    st(i, j, c_my) = rho * ny;
  }
};

}  // namespace detail

class ProgramContext {
 public:
  explicit ProgramContext(System* sys) : sys_(sys) {}
  /// Wraps a System passed as a flat void* (what adc_install_program(void* sys) receives).
  explicit ProgramContext(void* sys) : sys_(static_cast<System*>(sys)) {}

  /// Register the macro-step body. @p step advances ONE macro-step over dt (it owns solve_fields,
  /// the RHS, the linear combine and the commit). Empty std::function clears it.
  void install(std::function<void(double)> step) const {
    sys_->install_program_step(std::move(step));
  }

  void solve_fields() const { sys_->solve_fields(); }
  /// Per-stage field solve (ADC-409): re-solve the elliptic fields and re-fill the shared aux from
  /// block @p b's STAGE state @p u_stage (not its live state), so a field-coupled multi-stage
  /// Program's stage k reads phi solved from stage k's own state. Forwards to
  /// System::solve_fields_from_state. With b = 0 and u_stage = U^n (the first stage) it matches
  /// solve_fields(); the codegen lowers every solve_fields op to this, passing the stage's state var.
  void solve_fields_from_state(int b, MultiFab& u_stage) const {
    sys_->solve_fields_from_state(b, u_stage);
  }
  int n_blocks() const { return sys_->n_blocks(); }
  MultiFab& state(int b) const { return sys_->block_state(b); }
  void rhs_into(int b, MultiFab& u, MultiFab& r) const { sys_->block_rhs_into(b, u, r); }

  /// r <- -div F(u) for block @p b -- the SAME flux divergence as @ref rhs_into but WITHOUT the model's
  /// default/composite source (Poisson frozen). Forwards to System::block_neg_div_flux_into (the block's
  /// SourceFreeModel<Model> rhs path, bit-identical to rhs_into minus the source). The codegen lowers a
  /// hyperbolic stage that excludes the default source (P.rhs(flux=True, sources without "default"),
  /// incl. the empty list) to this, so a Lie/Strang split assembles "flux but no source" without the
  /// default source leaking in (epic ADC-399 / ADC-425, spec criterion 17). Header-inline forwarder,
  /// like @ref rhs_into.
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r) const {
    sys_->block_neg_div_flux_into(b, u, r);
  }

  /// The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)) -- the
  /// SAME hmin the native CFL uses. Forwards to System::cfl_min_dx. A compiled time Program's dt bound
  /// (epic ADC-399 / ADC-417, spec s18) reads it to express e.g. cfl * hmin / max_wave_speed.
  Real hmin() const { return sys_->cfl_min_dx(); }

  /// The maximum |wave speed| of block @p b on the state @p u: the SAME per-block reduction step_cfl
  /// reads (BlockState::max_speed). Forwards to System::block_max_speed -- it REUSES the block's
  /// wave-speed closure, it does not recompute the speed. @p u is the state the bound is evaluated on
  /// (the block's current state for a CFL bound). The dt_bound expression uses it as the denominator of
  /// cfl * hmin / max_wave_speed (epic ADC-399 / ADC-417, spec s18).
  Real max_wave_speed(int b, const MultiFab& u) const { return sys_->block_max_speed(b, u); }

  /// The System aux MultiFab (phi=0, grad_x=1, grad_y=2, B_z=3, T_e=4, named fields from
  /// kAuxNamedBase). NOT owned by the context: it is the live System aux (stable address), the same
  /// channel solve_fields() fills. A generated local-linear-solve kernel reads the operator
  /// coefficients (e.g. B_z) from it. Forwards to System::grid_context().aux.
  MultiFab& aux() const { return *sys_->grid_context().aux; }

  /// A fresh scalar field co-distributed with the System mesh (block 0's box array / distribution),
  /// @p n_comp components, @p n_ghost ghost layers, zero-initialized. Forwards to
  /// System::alloc_scalar_field. The scratch fields (residual, search direction, solution) a
  /// matrix-free Krylov solve allocates -- a 1-component field is distinct from the n_cons block state,
  /// but shares its (ba, dm) so laplacian / gradient pair it with the state and aux by local fab index.
  MultiFab alloc_scalar_field(int n_comp = 1, int n_ghost = 1) const {
    return sys_->alloc_scalar_field(n_comp, n_ghost);
  }

  /// The System mesh geometry (index domain + physical bounds, dx/dy). BY VALUE: grid_context()
  /// returns a temporary, so a reference to its @c geom member would dangle. The metric the matrix-free
  /// Laplacian / gradient read.
  Geometry geom() const { return sys_->grid_context().geom; }

  /// out = Lap(in): fill @p in's ghosts (transport BC, periodic by default) then apply the SHARED
  /// discrete 5-point Laplacian (adc::apply_laplacian, all optional coefficients null -> the bare
  /// bit-identical Laplacian). @p in is non-const because the ghost fill WRITES its halos (the valid
  /// cells are unchanged); this is the same matvec idiom the matrix-free Krylov test
  /// (tests/test_generic_krylov.cpp) wraps in its ApplyFn. The compiled program forms an operator
  /// A(in) = in - alpha*Lap(in) by combining this with ctx.lincomb.
  void laplacian(MultiFab& out, MultiFab& in) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(in, gc.geom.domain, gc.bc);
    apply_laplacian(in, gc.geom, out);  // all optional pointers null -> bare 5-point Laplacian
  }

  /// out = grad(@p phi) by centered differences: out(.,0) = d phi/dx, out(.,1) = d phi/dy (@p out
  /// needs >= 2 components). Fills @p phi's ghosts then forwards to adc::field_postprocess with
  /// store_phi=false (the gradient lands in components 0/1) and the centered factors cx = 1/(2 dx),
  /// cy = 1/(2 dy) -- the same derivation the elliptic aux post-process uses (+grad sign).
  void gradient(MultiFab& out, MultiFab& phi) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(phi, gc.geom.domain, gc.bc);
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  /// out = div(@p fx, @p fy) by centered differences: out = d fx/dx + d fy/dy (component 0). The x-flux
  /// is read from component 0 of @p fx and the y-flux from component 1 of @p fy, the SAME layout
  /// @ref gradient writes (d/dx in component 0, d/dy in component 1) -- so chaining ctx.gradient(g, phi)
  /// then ctx.divergence(out, g, g) recovers the 5-point Laplacian. Fills the ghosts of @p fx and @p fy
  /// (transport BC, periodic by default) then forwards to adc::apply_divergence -- the exact inverse
  /// stencil of @ref gradient and the same centered FV divergence the native Schur condensation
  /// assembles (coupling/schur/core/schur_condensation.hpp). @p fx and @p fy are non-const because the
  /// ghost fill WRITES their halos (the valid cells are unchanged). A compiled Program forms a
  /// Schur-like flux operator A(phi) = phi - alpha*div(grad phi) by chaining ctx.gradient then
  /// ctx.divergence inside a matrix-free apply.
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    if (&fy != &fx)
      fill_ghosts(fy, gc.geom.domain, gc.bc);  // skip the redundant halo fill when fy aliases fx
    apply_divergence(fx, fy, gc.geom, out, /*cx=*/0, /*cy=*/1);
  }

  /// @name Anisotropic Schur condensation (epic ADC-399 / ADC-421)
  /// The full condensed-Schur operator is L_schur(phi) = -div((I + c*rho*B^{-1}) grad phi), a tensor
  /// elliptic operator whose per-cell coefficient varies with rho and B_z. These primitives let a
  /// compiled Program ASSEMBLE that coefficient tensor (from the live state + B_z aux) and APPLY it
  /// matrix-free, REUSING the native Schur kernels (coupling/schur/core/schur_condensation.hpp) and
  /// adc::apply_laplacian's coefficient path -- no stencil / elimination reimplementation. The native
  /// adc::CondensedSchur source stepper is untouched.
  /// @{

  /// Assemble the tensor coefficient A_op = I + c*rho*B^{-1} of the condensed-Schur operator per cell:
  /// eps_x = 1 + c*rho*binv_11, eps_y = 1 + c*rho*binv_22, a_xy = c*rho*binv_12, a_yx = c*rho*binv_21,
  /// with B^{-1} the closed 2x2 LorentzEliminator(th_dt, 1, B_z). @p state carries rho at component
  /// @p c_rho; B_z is read from the System aux at component @p c_bz. The four coefficient fields are
  /// filled over the valid cells (the SAME detail::SchurOperatorCoeffKernel the native builder uses)
  /// and their ghosts extended by zero-gradient (Foextrap, periodic preserved) -- the eps_bc the
  /// GeometricMG / native assembly use, so the face mean at the boundary is consistent. @p c =
  /// theta^2 dt^2 alpha, @p th_dt = theta*dt. Assembled ONCE per step (rho / B_z frozen in the source),
  /// then reused across every Krylov iteration of the matrix-free phi solve.
  void assemble_schur_coeffs(MultiFab& eps_x, MultiFab& eps_y, MultiFab& a_xy, MultiFab& a_yx,
                             const MultiFab& state, Real c, Real th_dt, int c_rho, int c_bz) const {
    const GridContext gc = sys_->grid_context();
    const MultiFab& aux = *sys_->grid_context().aux;
    for (int li = 0; li < eps_x.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = aux.fab(li).const_array();
      for_each_cell(eps_x.box(li),
                    detail::SchurOperatorCoeffKernelC{s, b, eps_x.fab(li).array(),
                                                      eps_y.fab(li).array(), a_xy.fab(li).array(),
                                                      a_yx.fab(li).array(), c, th_dt, c_rho, c_bz});
    }
    const BCRec ebc = coeff_bc(gc.bc);
    fill_ghosts(eps_x, gc.geom.domain, ebc);
    fill_ghosts(eps_y, gc.geom.domain, ebc);
    fill_ghosts(a_xy, gc.geom.domain, ebc);
    fill_ghosts(a_yx, gc.geom.domain, ebc);
  }

  /// out = div(A grad in), A = [[eps_x, a_xy], [a_yx, eps_y]] -- the coefficiented matrix-free matvec
  /// of the condensed-Schur operator. Fills @p in's ghosts (transport BC) then forwards to the SAME
  /// adc::apply_laplacian coefficient path the native GeometricMG operator uses (eps / cross pointers),
  /// component 0 (the scalar potential). @p in is non-const because the ghost fill writes its halos.
  /// The condensed operator is L_schur(phi) = -div(A grad phi) = -out, so a matrix-free apply forms
  /// it as ``ctx.apply_laplacian_coeff(out, in, ...); out *= -1`` via the affine algebra. The
  /// coefficient fields are the ones assemble_schur_coeffs filled (1 ghost each).
  void apply_laplacian_coeff(MultiFab& out, MultiFab& in, const MultiFab& eps_x,
                             const MultiFab& eps_y, const MultiFab& a_xy,
                             const MultiFab& a_yx) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(in, gc.geom.domain, gc.bc);
    apply_laplacian(in, gc.geom, out, /*coef=*/nullptr, /*eps=*/&eps_x, /*kappa=*/nullptr,
                    /*eps_y=*/&eps_y, /*a_xy=*/&a_xy, /*a_yx=*/&a_yx);
  }

  /// out = B^{-1} (mx, my) per cell -- the EXPLICIT condensed-Schur flux F = rho*B^{-1}*v^n (= B^{-1}
  /// applied to the momentum, avoiding the divide by rho). @p out has >= 2 components (Fx in comp 0,
  /// Fy in comp 1, the layout ctx.divergence reads). @p state carries mx / my at @p c_mx / @p c_my;
  /// B_z from the aux at @p c_bz; @p th_dt = theta*dt (w = th_dt*B_z). Reuses the native
  /// detail::SchurExplicitFluxKernel. The condensed RHS is then -Lap phi^n - theta*dt*alpha*div(F),
  /// assembled with ctx.laplacian + ctx.divergence + the affine algebra.
  void schur_explicit_flux(MultiFab& out, const MultiFab& state, Real th_dt, int c_mx, int c_my,
                           int c_bz) const {
    const GridContext gc = sys_->grid_context();
    const MultiFab& aux = *sys_->grid_context().aux;
    for (int li = 0; li < out.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = aux.fab(li).const_array();
      Array4 o = out.fab(li).array();
      for_each_cell(out.box(li),
                    detail::SchurExplicitFluxKernelC{s, b, o, th_dt, c_mx, c_my, c_bz});
    }
    const BCRec ebc = coeff_bc(gc.bc);
    fill_ghosts(out, gc.geom.domain, ebc);
  }

  /// rhs = -Lap(phi_n) - g*div(F), F = B^{-1}(mx, my) -- the FUSED condensed-Schur right-hand side
  /// (the native ElectrostaticLorentzCondensation::assemble_rhs, reading B_z from the aux at @p c_bz).
  /// @p phi_n is phi^n (its ghosts are filled here for the Laplacian); @p state carries mx / my at
  /// @p c_mx / @p c_my; @p th_dt = theta*dt; @p g = theta*dt*alpha. @p rhs is a 1-component scalar field.
  /// Internal Lap / flux buffers are allocated on @p rhs's layout (transient, like the native assembler).
  /// Mirrors native assemble_rhs step-for-step (bare apply_laplacian + NegateKernel + the explicit flux
  /// + SchurRhsAssembleKernel), so the top-level RHS assembly is a SINGLE op (no scalar-field affine
  /// combine at IR level): the same fused -Lap - g*div(F) the native source stepper assembles.
  void assemble_schur_rhs(MultiFab& rhs, MultiFab& phi_n, const MultiFab& state, Real th_dt, Real g,
                          int c_mx, int c_my, int c_bz) const {
    const GridContext gc = sys_->grid_context();
    const MultiFab& aux = *sys_->grid_context().aux;
    const BoxArray& ba = rhs.box_array();
    const DistributionMapping& dm = rhs.dmap();
    // 1) -Lap phi^n (bare 5-point Laplacian of the warm-started potential, negated).
    fill_ghosts(phi_n, gc.geom.domain, gc.bc);
    MultiFab lap(ba, dm, 1, 0);
    apply_laplacian(phi_n, gc.geom, lap);
    MultiFab neg_lap(ba, dm, 1, 0);
    for (int li = 0; li < neg_lap.local_size(); ++li)
      for_each_cell(neg_lap.box(li),
                    adc::detail::NegateKernel{lap.fab(li).const_array(), neg_lap.fab(li).array()});
    // 2) explicit flux F = B^{-1}(mx, my) at the center (1 ghost for the centered divergence).
    MultiFab fx(ba, dm, 2, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = aux.fab(li).const_array();
      for_each_cell(fx.box(li), detail::SchurExplicitFluxKernelC{s, b, fx.fab(li).array(), th_dt,
                                                                 c_mx, c_my, c_bz});
    }
    const BCRec ebc = coeff_bc(gc.bc);
    fill_ghosts(fx, gc.geom.domain, ebc);
    // 3) rhs = -Lap phi^n - g*div(F) (centered FV divergence; Fx in comp 0, Fy in comp 1 of fx).
    const Real half_idx = Real(1) / (Real(2) * gc.geom.dx());
    const Real half_idy = Real(1) / (Real(2) * gc.geom.dy());
    for (int li = 0; li < rhs.local_size(); ++li)
      for_each_cell(rhs.box(li), detail::SchurRhsAssembleKernelC{
                                     neg_lap.fab(li).const_array(), fx.fab(li).const_array(),
                                     rhs.fab(li).array(), g, half_idx, half_idy});
  }

  /// Reconstruct v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi^{n+theta}) and write mom = rho^n*v into
  /// @p state in place (rho frozen). @p phi is phi^{n+theta} (its ghosts are filled here for the
  /// centered gradient); B_z from the aux at @p c_bz; @p th_dt = theta*dt. v^n is read from the state
  /// (mx/my / rho), the same closed B^{-1} (LorentzEliminator) the native reconstruction uses. The
  /// final n+1 extrapolation (factor 1/theta) is left to the caller's affine algebra.
  void schur_reconstruct(MultiFab& state, MultiFab& phi, Real th_dt, int c_rho, int c_mx, int c_my,
                         int c_bz) const {
    const GridContext gc = sys_->grid_context();
    const MultiFab& aux = *sys_->grid_context().aux;
    fill_ghosts(phi, gc.geom.domain, gc.bc);
    const Real half_idx = Real(1) / (Real(2) * gc.geom.dx());
    const Real half_idy = Real(1) / (Real(2) * gc.geom.dy());
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 ph = phi.fab(li).const_array();
      const ConstArray4 b = aux.fab(li).const_array();
      Array4 st = state.fab(li).array();
      for_each_cell(state.box(li),
                    detail::SchurReconstructKernelC{ph, b, st, th_dt, half_idx, half_idy, c_rho,
                                                    c_mx, c_my, c_bz});
    }
  }
  /// @}
  /// r <- -div(fx, fy) per conservative component (ADC-419 named fluxes): r(.,c) = -(d fx(.,c)/dx +
  /// d fy(.,c)/dy), centered FV, for every component c of @p r. @p fx and @p fy hold the n_cons x- and
  /// y-flux fields a compiled Program's named-flux kernel wrote (component c = the flux of conservative
  /// component c). REUSES adc::apply_divergence component-by-component (the SAME centered stencil as
  /// @ref divergence, the inverse of @ref gradient -- no new differencing): the ghosts are filled once
  /// per field, then each component's divergence lands in a 1-component scratch and is copied with a
  /// sign flip into @p r. @p fx / @p fy are non-const because the ghost fill writes their halos (the
  /// valid cells are unchanged). This semi-discrete -div F is LINEAR in the flux, so the -div of a SUM
  /// of named fluxes equals the sum of their -div (the named-flux parity guarantee).
  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    fill_ghosts(fy, gc.geom.domain, gc.bc);
    MultiFab divc(r.box_array(), r.dmap(), 1,
                  0);  // 1-component divergence scratch (no ghosts needed)
    for (int c = 0; c < r.ncomp(); ++c) {
      apply_divergence(fx, fy, gc.geom, divc, /*cx=*/c, /*cy=*/c);  // divc(.,0) = div(fx_c, fy_c)
      for (int li = 0; li < r.local_size(); ++li) {
        const ConstArray4 d = divc.fab(li).const_array();
        Array4 rv = r.fab(li).array();
        const int comp = c;
        for_each_cell(r.box(li), [=] ADC_HD(int i, int j) { rv(i, j, comp) = -d(i, j, 0); });
      }
    }
  }

  /// A zero-initialized RHS scratch with the SAME layout (box array / distribution / ghosts) as @p u,
  /// so the subsequent axpy(u, ., r) combines identical layouts.
  MultiFab rhs_scratch_like(const MultiFab& u) const {
    return MultiFab(u.box_array(), u.dmap(), u.ncomp(), u.n_grow());
  }

  /// A zero-initialized scratch STATE with the same layout as @p u: an intermediate stage state of a
  /// multi-stage scheme (SSPRK/RK). Same allocation as rhs_scratch_like; named for the codegen's
  /// intent. Starts at zero, so a stage `sum_i c_i V_i` is built by axpy-ing each term onto it.
  MultiFab scratch_state_like(const MultiFab& u) const { return rhs_scratch_like(u); }

  /// u <- u + a r over the valid cells (linear combine; forwards to adc::saxpy).
  void axpy(MultiFab& u, Real a, const MultiFab& r) const { adc::saxpy(u, a, r); }

  /// z <- a x + b y over the valid cells (assignment, not accumulation; z may alias x or y).
  /// Forwards to adc::lincomb. The codegen uses it for the committed stage: the block state becomes
  /// z = c_base * z + 1 * acc, where acc holds the non-base terms (self-alias z==x is safe).
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    adc::lincomb(z, a, x, b, y);
  }

  /// Register (idempotent) the history @p name with maximum lag @p lag, allocating the ring buffer
  /// WITHOUT reading it. The codegen emits this ONCE at the top of the step body for each declared
  /// history, so the ring depth is locked before the first store (the cold-start fill then broadcasts
  /// the first stored value into every -- already allocated -- slot). Forwards to
  /// System::register_history. A read-only counterpart of @ref history (no fail-loud on uninitialized).
  void register_history(const std::string& name, int lag) const {
    sys_->register_history(name, lag);
  }

  /// The history slot @p lag macro-steps back (the SYSTEM-OWNED ring buffer, ADC-406a): lag 1 = the
  /// previous step's stored value (e.g. R_{n-1} for Adams-Bashforth), lag 0 = the current slot. The
  /// codegen emits ``ctx.history("<name>", <lag>)``; the read registers the ring on first use
  /// (idempotent) and forwards to System::read_history, which throws if the history was never stored
  /// (spec error 17). @p lag defaults to 1 (the common one-step-back read).
  MultiFab& history(const std::string& name, int lag = 1) const {
    sys_->register_history(name, lag);  // idempotent: allocate the ring on first use
    return sys_->read_history(name, lag);
  }

  /// Store @p value into the CURRENT slot of history @p name (ADC-406a). Registers the ring on first
  /// use (at least a current slot; the lag the program reads via @ref history sets the real depth) and
  /// forwards to System::store_history (which fills every slot on the first store -- the cold start).
  /// The codegen emits ``ctx.store_history("<name>", <value>)`` near the end of the step body.
  void store_history(const std::string& name, const MultiFab& value) const {
    sys_->register_history(name, 1);  // idempotent: at least a current slot exists before the store
    sys_->store_history(name, value);
  }

  /// Shift every history ring one macro-step (slot k <- slot k-1). Forwards to
  /// System::rotate_histories. The codegen emits ``ctx.rotate_histories()`` as the LAST statement of
  /// the step body (after the commit), so the next step reads lag k as the value k stores ago.
  void rotate_histories() const { sys_->rotate_histories(); }

  /// @name Reductions (spec op 16)
  /// COLLECTIVE all_reduce over one component of a field (sum / signed max / signed min). The codegen
  /// lowers P.sum / P.sum_component / P.max / P.min DIRECTLY to the adc:: free functions (like norm2 ->
  /// adc::dot), but these wrappers expose them on the context for hand-rolled C++ stages and mirror
  /// norm2 / dot above. MANDATORY UNDER MPI: called on EVERY rank (empty ranks included), like dot.
  /// @{
  Real sum_component(const MultiFab& u, int comp) const { return adc::reduce_sum(u, comp); }
  Real max_component(const MultiFab& u, int comp) const { return adc::reduce_max(u, comp); }
  Real min_component(const MultiFab& u, int comp) const { return adc::reduce_min(u, comp); }
  Real sum(const MultiFab& u) const { return adc::reduce_sum(u, 0); }
  Real max(const MultiFab& u) const { return adc::reduce_max(u, 0); }
  Real min(const MultiFab& u) const { return adc::reduce_min(u, 0); }
  /// @}

  /// Fill the ghost cells (halos) of @p x in place: the transport BC (periodic by default), the SAME
  /// exchange laplacian / gradient / divergence run internally before differencing (spec op 22). The
  /// valid cells are untouched; only the halos change. Forwards to the shared adc::fill_ghosts.
  void fill_boundary(MultiFab& x) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(x, gc.geom.domain, gc.bc);
  }

  /// Apply block @p b's post-step positivity projection to @p u in place: U <- project(U, aux) over the
  /// valid cells, the SAME Zhang-Shu / floor projection the native per-step path runs (ADC-177, spec
  /// op 21). REUSES the block's own projection closure (set at add_block time); a block WITHOUT a
  /// projection is a no-op. Forwards to System::block_project -- it reimplements no positivity.
  void apply_projection(int b, MultiFab& u) const { sys_->block_project(b, u); }

  /// Store a runtime Scalar @p value into the System diagnostics map under @p name (spec op 23),
  /// retrievable after the step via System::program_diagnostic / program_diagnostics (exposed to
  /// Python as sim.program_diagnostic / sim.program_diagnostics). A pure side effect: the scalar is
  /// recorded for inspection / logging, it does not feed the numerics. Forwards to
  /// System::record_program_diagnostic.
  void record_scalar(const std::string& name, Real value) const {
    sys_->record_program_diagnostic(name, value);
  }

  /// The current value of RUNTIME parameter @p name (spec ParamValue param(string), ADC-435), set by the
  /// host via sim.set_param. A compiled time Program kernel reading a dsl.Param(kind='runtime') lowers to
  /// ONE host call of this -- ``const adc::Real <name> = ctx.param("<name>");`` -- BEFORE the per-cell
  /// loop (captured by value), so the value can be changed at runtime WITHOUT recompiling the .so (a
  /// FROZEN const param stays baked as a literal). Forwards to System::param (throws if the host never
  /// seeded @p name -- fail-loud, not silent 0). A pure read: it does not feed the numerics by itself.
  Real param(const std::string& name) const { return sys_->param(name); }

 private:
  /// BC of the coefficient / flux fields (ADC-421): periodic preserved, physical boundary -> zero-
  /// gradient (Foextrap). Identical to GeometricMG::eps_bc and the native Schur coeff_bc -- the face
  /// value at the domain boundary equals the interior value, consistent with the elliptic operator.
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo);
    b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo);
    b.yhi = fo(bc.yhi);
    return b;
  }

  System* sys_;
};

}  // namespace program
}  // namespace runtime
}  // namespace adc
