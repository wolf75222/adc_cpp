#pragma once

#include <adc/core/state/state.hpp>        // kAuxBaseComps (base component of the aux channel)
#include <adc/core/foundation/types.hpp>        // Real
#include <adc/mesh/storage/multifab.hpp>     // MultiFab, Array4, ConstArray4
#include <adc/mesh/index/box2d.hpp>        // Box2D
#include <adc/mesh/execution/for_each.hpp>     // device_fence
#include <adc/mesh/boundary/physical_bc.hpp>  // BCRec, fill_ghosts, fill_boundary
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/polar/polar_poisson_solver.hpp>  // PolarPoissonSolver (direct polar Poisson)
#include <adc/parallel/comm.hpp>                           // n_ranks() (FFT MPI guard)
#include <adc/runtime/builders/block/block_builder_polar.hpp>  // derive_aux_polar (polar aux in local basis)
#include <adc/runtime/context/wall_predicate.hpp>       // detail::wall_predicate

#include <cstdio>   // ADC_TRACE_SOLVE_FIELDS: device diagnostic trace (env-gated, inert by default)
#include <cstdlib>  // getenv
#include <functional>
#include <map>  // named_aux_: NAMED aux fields (comp -> field), re-applied after channel realloc
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/// @file
/// @brief SystemFieldSolver: the ELLIPTIC SOLVE + FIELD DERIVATION responsibility extracted
///        from the god-class System::Impl (audit Lot B, cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2).
///        Extracted VERBATIM from python/system.cpp: no change to the numerics, to the order of
///        operations, to fill_ghosts/fill_boundary, to device_fence or to tolerance. STRICTLY
///        bit-identical -- the code is moved as is, only access to the SHARED members of Impl
///        (aux, sp, cfg, geom, pgeom_, ba, dm, bc_, dom, per_, periodic_, polar_) goes through the
///        back-pointer owner_->.
///
/// CONTRACT / INVARIANTS
/// - OWNS: the elliptic solvers (ell_ cartesian GeometricMG/PoissonFFTSolver; pell_ polar
///   PolarPoissonSolver), the Poisson configuration tokens (p_rhs/p_solver/p_bc/p_wall/
///   p_wall_radius/p_eps_), the coefficient fields and flags (eps(x), eps_x/eps_y, kappa) as well as
///   the aux field APPLICATION buffers (bz_field_ and te_src_) with their methods apply_bz /
///   apply_te.
/// - READS (without owning) the SHARED aux and the block list via owner_->: the aux is populated by
///   solve_fields (phi, grad phi, B_z, T_e) then its halos are filled; the block list provides the
///   right-hand side of the Poisson (sum of the per-block elliptic bricks) and the fluid source of T_e.
/// - DISPATCH cartesian vs polar: solve_fields() routes to solve_fields_polar() when owner_->polar_,
///   otherwise the cartesian path. The two paths are independent (ell_ never touched in polar and
///   vice versa). ensure_elliptic / ensure_elliptic_polar build the solver lazily.
/// - CRITICAL device INVARIANT: the device_fence() between ell_solve() and the derivation of grad phi MUST
///   stay atomic (without it, the GPU V-cycle is not finished when phi is read). Same in polar
///   after pell_->solve(). DO NOT reorder.
/// - MPI INVARIANT: the derivation / population loops (B_z, T_e, eps, kappa) iterate over the LOCAL
///   fabs (local_size()), never fab(0) hardcoded: no-op on a rank without a box, bit-identical to the
///   owner. This guard is PRESERVED by the extraction.
///
/// Since System::Impl stays PRIVATE to python/system.cpp, this helper is a TEMPLATE parametrized on the real
/// Impl type (same technique as native_loader): python/system.cpp instantiates it with System::Impl after
/// defining Impl. owner_ is an Impl* (the lifetime of the helper is subordinate to that of Impl).

namespace adc {
namespace field_solver {

/// True if the DIAGNOSTIC trace of the solve_fields path is active (environment variable
/// ADC_TRACE_SOLVE_FIELDS set). Added for a CUDA device-crash diagnostic: writes to stderr with immediate flush
/// to locate the last marker before a device crash. INERT by default: no effect on the
/// outputs or on the numerics. Diagnostic KEPT (env-gated): useful for a future device crash.
inline bool adc_trace_sf() {
  static const bool on = std::getenv("ADC_TRACE_SOLVE_FIELDS") != nullptr;
  return on;
}
/// Writes the marker @p w to stderr (with flush) ONLY if adc_trace_sf(); no-op otherwise.
inline void adc_sf_mark(const char* w) {
  if (adc_trace_sf()) {
    std::fprintf(stderr, "[sf] %s\n", w);
    std::fflush(stderr);
  }
}

/// SystemFieldSolver<Impl>: see contract above. All methods are MEMBERS (not free
/// functions) because they share the elliptic state owned by this class; accesses to the SHARED
/// state of Impl go through owner_-> verbatim. Templated on Impl to stay free of any dependency on the
/// (private) definition of System::Impl.
template <class Impl>
class SystemFieldSolver {
 public:
  /// @param owner back-pointer to System::Impl (lifetime subordinate to that of Impl).
  explicit SystemFieldSolver(Impl* owner) : owner_(owner) {}

  /// Canonical component of T_e (after phi/grad/B_z); cf. adc::Aux and AUX_CANONICAL on the DSL side.
  static constexpr int kTeComp = kAuxBaseComps + 1;  // = 4

  // --- OWNED state (elliptic solve + coefficient fields + application buffers) --------
  // Poisson configuration (elliptic solver built lazily).
  std::string p_rhs = "charge_density";
  std::string p_solver = "geometric_mg";
  std::string p_bc = "auto";
  std::string p_wall = "none";
  double p_wall_radius = 0.0;
  Real p_eps_ = 1;  // CONSTANT permittivity: div(eps grad phi) = f <=> lap phi = f/eps
  // ABSOLUTE floor of the GeometricMG V-cycle stopping criterion (same units as the residual). Default 0:
  // purely relative criterion (historical bit-identical behavior). Set > 0 (problem scale),
  // it makes the off-step solve_fields exit WITHOUT cycling on an already converged state (diagnostics, oracles,
  // restart). Inert for the FFT solver (direct, without iterative tolerance).
  Real p_abs_tol_ = 0;
  bool has_eps_field_ = false;  // VARIABLE permittivity eps(x) provided (carried by the operator)
  std::vector<double> p_eps_field_;  // field eps(x), n*n row-major (if has_eps_field_)
  bool has_eps_xy_field_ =
      false;  // ANISOTROPIC permittivity eps_x(x), eps_y(x) (operator div(diag(eps_x,eps_y) grad phi))
  std::vector<double>
      p_eps_x_field_;  // field eps_x(x), n*n row-major (faces normal to x; if has_eps_xy_field_)
  std::vector<double>
      p_eps_y_field_;  // field eps_y(x), n*n row-major (faces normal to y; if has_eps_xy_field_)
  bool has_kappa_field_ = false;  // REACTION term kappa(x) provided: div(eps grad phi) - kappa phi
  std::vector<double> p_kappa_field_;  // field kappa(x), n*n row-major (if has_kappa_field_)
  // GAUSS POLICY (restart-free Gauss-evolution option). Default "restart":
  // solve_fields re-solves -Delta phi = f (Gauss) on EVERY call -- historical behavior,
  // BIT-IDENTICAL. "evolve": after the FIRST solve (phi^0), solve_fields NO LONGER re-solves the
  // Poisson; it only DERIVES the aux (phi, grad phi) from the CURRENT phi -- the one that the condensed
  // source stage (Schur) evolves IN-PLACE in ell_phi() (cf. run_source_stage). Thus it gives a
  // restart-free evolution of -Delta phi (the Gauss constraint is only imposed at t=0).
  // INERT without a Schur stage (phi would stay frozen after t=0). The lock gauss_solved_once_ guarantees that
  // the first solve (the init of phi^0) always solves, whatever the policy.
  bool gauss_evolve_ = false;
  bool gauss_solved_once_ = false;
  std::optional<std::variant<GeometricMG, PoissonFFTSolver, RemappedFFTSolver>> ell_;
  // Direct POLAR Poisson solver (FFT-in-theta + tridiag-in-r), built lazily when
  // polar_ (cf. ensure_elliptic_polar). SEPARATE from ell_ (geom() returns a PolarGeometry, not a
  // Geometry): the cartesian path is never touched. INERT (nullopt) in cartesian.
  std::optional<PolarPoissonSolver> pell_;
  // phi buffer of the POLAR condensed SOURCE STAGE (Path A step 2c). The direct PolarPoissonSolver
  // (pell_->phi()) is WITHOUT ghost (valid box = allocation); but the PolarCondensedSchurSourceStepper
  // needs a phi WITH 1 ghost (fill_ghosts + apply_polar_tensor + centered grad + write of
  // phi^{n+1}). So we pass it this dedicated buffer (1 ghost), fed with phi^n (= aux[0] after
  // solve_fields_polar) before the source stage, which carries phi^{n+1} on output (warm start of the next
  // step). In CARTESIAN this buffer is INERT (nullopt): ell_phi() routes to ell_->phi() as
  // before, BIT-IDENTICAL.
  std::optional<MultiFab> phi_src_polar_;
  std::vector<Real> bz_field_;  // field B_z(x) n*n row-major (empty if not provided)
  int te_src_ = -1;             // index of the fluid block source of T_e (-1 = none)
  // NAMED aux fields (ADC-70 phase 1) provided by the user via System::set_aux_field: key =
  // canonical component (>= kAuxNamedBase = 5), value = field n*n (cartesian) / nr*ntheta (polar)
  // row-major. PERSISTENT like bz_field_: solve_fields touches ONLY components 0..2 (phi,
  // grad) and 4 (T_e via apply_te), so components >= 5 survive from one step to the next; but a
  // REALLOCATION of the aux channel (ensure_aux_width) starts again from a zeroed MultiFab -> we re-apply
  // them then (apply_named_aux), exactly like apply_bz / apply_te.
  std::map<int, std::vector<Real>> named_aux_;
  // Per-field aux HALO policy (ADC-369): key = canonical component (>= kAuxNamedBase), value = the
  // uniform boundary policy declared via adc.AuxHalo. Applied by apply_named_aux_bc() AFTER the shared
  // aux ghost fill, overriding only that component's PHYSICAL-face ghosts (periodic faces -- Cartesian
  // periodic, polar theta -- keep their wrap). Empty -> shared aux BC for every field, bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;

  // NAMED multi-elliptic fields (ADC-428): a SECOND elliptic solve (beyond the default Poisson) for a
  // user-named field m.elliptic_field("phi2", rhs=..., aux=[...]). Each named field owns:
  //   - phi_comp / gx_comp / gy_comp: the aux channel components (>= kAuxNamedBase) its solution and
  //     centered gradient are written to (the model declares them as aux_field slots; a source then
  //     reads them like any other named aux). gx_comp / gy_comp < 0 => the field declares fewer than 3
  //     aux slots and the gradient is not derived (only phi is written).
  //   - a DEDICATED elliptic solver instance (cartesian GeometricMG / FFT), built lazily the SAME way
  //     as the default ell_ (ensure_named_elliptic mirrors ensure_elliptic's poisson operator), so the
  //     native solver is REUSED, not reimplemented. The default Poisson path (ell_) is untouched.
  // The RHS = sum over blocks of s.named_poisson_rhs[name] (the per-field brick, ADC-428), assembled
  // exactly like assemble_poisson_rhs but reading the named closures.
  struct NamedField {
    int phi_comp = -1;
    int gx_comp = -1;
    int gy_comp = -1;
    std::optional<std::variant<GeometricMG, PoissonFFTSolver, RemappedFFTSolver>> ell;
  };
  std::map<std::string, NamedField> named_fields_;

  /// Register a named elliptic field (ADC-428): records the aux output components (where the field's
  /// solved phi and centered gradient land). @p gx_comp / @p gy_comp < 0 => only phi is written (the
  /// model declared fewer than 3 aux slots for the field). Idempotent (re-register overwrites the
  /// component map, drops the lazily-built solver so the next solve rebuilds it). The DEDICATED solver
  /// is built on first solve, never here.
  void register_named_field(const std::string& field, int phi_comp, int gx_comp, int gy_comp) {
    NamedField nf;
    nf.phi_comp = phi_comp;
    nf.gx_comp = gx_comp;
    nf.gy_comp = gy_comp;
    named_fields_[field] = std::move(nf);  // solver built lazily by ensure_named_elliptic
  }

  /// Re-applies the per-field aux HALO policies (ADC-369) onto the shared channel, AFTER the shared
  /// fill_ghosts/fill_boundary. For each declared component, overrides ONLY that component's
  /// physical-face ghosts (aux_halo_override keeps periodic faces periodic). No-op when empty.
  void apply_named_aux_bc() {
    if (named_aux_bc_.empty())
      return;  // hot-path fast exit (parity with the AMR counterparts)
    for (const auto& kv : named_aux_bc_) {
      if (kv.first >= owner_->aux_ncomp_)
        continue;
      fill_physical_bc(owner_->aux, owner_->dom, aux_halo_override(owner_->bc_, kv.second),
                       kv.first);
    }
  }

  /// Populates the B_z component (index kAuxBaseComps) of the shared channel from bz_field_, over the
  /// valid cells. No-op if B_z not provided or if no block reads it (base width). The
  /// halos of B_z are filled by solve_fields (like grad); field_postprocess only writes comp 0..2.
  void apply_bz() {
    if (bz_field_.empty() || owner_->aux_ncomp_ <= kAuxBaseComps)
      return;
    // ROW WIDTH (fast axis i) of the row-major array bz_field_: n in cartesian (square n x n,
    // BIT-IDENTICAL), nr in POLAR (ring nr x ntheta, i = r of size nr, cf. set_magnetic_field).
    // The index stays flat[j * row + i]: in cartesian row == n (unchanged); in polar row == nr.
    const int row = owner_->polar_ ? owner_->aux.box(0).nx() : owner_->cfg.n;
    // LOCAL population on the owner rank (cf. solve_fields): iteration over the local fabs of the
    // aux channel instead of fab(0) hardcoded (no-op on a rank without a local box at np>1, bit-identical to the
    // owner).
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          a(i, j, kAuxBaseComps) = bz_field_[static_cast<std::size_t>(j) * row + i];
    }
  }

  /// Populates the T_e component (electron temperature) = p/rho of the fluid block source te_src_.
  /// RECOMPUTED on each solve_fields (T_e varies with the fluid, unlike the static B_z).
  /// No-op if no source or if no block reads T_e (insufficient width). The source block is
  /// compressible (4 var); p = (gamma-1)(E - 0.5 rho|v|^2), T = p/rho.
  void apply_te() {
    if (te_src_ < 0 || owner_->aux_ncomp_ <= kTeComp)
      return;
    const auto& s = owner_->sp[static_cast<std::size_t>(te_src_)];
    const Real gm1 = Real(s.gamma) - Real(1);
    // LOCAL population on the owner rank (cf. solve_fields): we iterate over the local fabs of the
    // aux channel instead of fab(0) hardcoded (no-op on a rank without a local box at np>1, bit-identical to the
    // owner). s.U and aux share the same DistributionMapping -> same local indexing.
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 us = s.U.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const Real rho = us(i, j, 0), mx = us(i, j, 1), my = us(i, j, 2), E = us(i, j, 3);
          const Real p = gm1 * (E - Real(0.5) * (mx * mx + my * my) / rho);
          a(i, j, kTeComp) = p / rho;  // T = p / rho
        }
    }
  }

  /// Populates ONE NAMED aux component (canonical index @p comp >= kAuxNamedBase) of the shared channel
  /// from @p field (row-major), over the valid cells. No-op if the channel is too narrow
  /// (no block reads this component) or if the field is empty. SAME pattern as apply_bz: STATIC field
  /// provided by the user, never rewritten by solve_fields; its halos are filled by
  /// solve_fields (fill_ghosts/fill_boundary over the whole channel). LOCAL population on the rank (iteration
  /// over the local fabs, no-op on a rank without a box at np>1).
  void apply_named_aux_one(int comp, const std::vector<Real>& field) {
    if (field.empty() || owner_->aux_ncomp_ <= comp)
      return;
    // ROW WIDTH (fast axis i): n in cartesian (square n x n), nr in polar (ring nr x
    // ntheta). Index flat[j * row + i], identical to apply_bz / set_density.
    const int row = owner_->polar_ ? owner_->aux.box(0).nx() : owner_->cfg.n;
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          a(i, j, comp) = field[static_cast<std::size_t>(j) * row + i];
    }
  }

  /// Re-applies ALL the stored named aux fields (cf. named_aux_). Called by ensure_aux_width
  /// after a reallocation of the aux channel (which starts again from a zeroed MultiFab), like apply_bz / apply_te.
  void apply_named_aux() {
    for (const auto& kv : named_aux_)
      apply_named_aux_one(kv.first, kv.second);
  }

  // --- elliptic solver (system Poisson) -----------------------------
  /// Resolves the BC mode into a BCRec: "auto" -> dirichlet if wall/non-periodic, otherwise periodic;
  /// "periodic"|"dirichlet"|"neumann" (Foextrap). @throws std::runtime_error on an unknown mode.
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto")
      mode = (p_wall == "circle" || !owner_->cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic")
      return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson: unknown bc '" + mode + "'");
  }
  /// "Conductor interior" predicate from p_wall / p_wall_radius / cfg.L (cf. wall_predicate);
  /// empty if no wall.
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, owner_->cfg.L, "System::set_poisson");
  }
  /// Builds the cartesian elliptic solver (ell_) LAZILY according to p_solver: GeometricMG
  /// (carries eps(x)/aniso/kappa if provided) or PoissonFFTSolver (constant coefficient, single-rank,
  /// no wall; kind 'fft' = discrete stencil, 'fft_spectral' = continuous spectral symbol).
  /// No-op if ell_ already exists. @throws std::runtime_error on unknown rhs/solver or
  /// unsupported combination (fft + MPI/wall/variable eps/kappa; kappa + constant eps != 1).
  void ensure_elliptic() {
    if (ell_)
      return;
    // The system right-hand side is ALWAYS f = Sum_s elliptic_rhs_s(u_s), assembled by
    // solve_fields from the elliptic brick of EACH block (charge q n, background alpha (n-n0),
    // gravity coupling 4piG (rho-rho0)). The token is thus NOT a computation mode but a LABEL
    // of this composite right-hand side. "composite" names this behavior honestly; "charge_density"
    // stays the historical alias (default, bit-identical) since the usual case is a charge block.
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson: unknown rhs '" + p_rhs +
                               "' (charge_density|composite; the right-hand side = sum of the "
                               "per-block elliptic bricks)");
    const BCRec pbc = poisson_bc();
    std::function<bool(Real, Real)> active = wall_active();
    if (p_solver == "fft" || p_solver == "fft_spectral") {
      // The FFT path is CONSTANT-coefficient and PERIODIC only: reject walls, variable / anisotropic
      // permittivity and reaction kappa FIRST (each guard fires identically on all ranks -> no
      // deadlock), so only the pure periodic constant-coefficient case reaches the layout switch below.
      if (active)
        throw std::runtime_error("System: solver '" + p_solver +
                                 "' incompatible with a wall -> 'geometric_mg'");
      if (has_eps_field_)
        throw std::runtime_error("System: solver '" + p_solver +
                                 "' has a CONSTANT coefficient, incompatible with a "
                                 "variable eps(x) field -> use solver='geometric_mg'");
      if (has_eps_xy_field_)
        throw std::runtime_error(
            "System: solver '" + p_solver +
            "' has a CONSTANT coefficient, incompatible with an "
            "ANISOTROPIC permittivity eps_x(x), eps_y(x) -> use solver='geometric_mg'");
      if (has_kappa_field_)
        throw std::runtime_error("System: solver '" + p_solver +
                                 "' (pure Poisson) incompatible with a "
                                 "reaction term kappa(x) -> use solver='geometric_mg'");
      // 'fft_spectral': same plumbing, CONTINUOUS symbol -(kx^2+ky^2) (fidelity to the spectral
      // references, e.g. poisson_fft.m of RIEMOM2D); 'fft' keeps the discrete stencil (bit-identical).
      const bool spectral = (p_solver == "fft_spectral");
      if (n_ranks() > 1) {
        // ADC-287: distributed periodic FFT via a box<->slab remap. System distributes ONE box
        // round-robin (DistributionMapping(1, n_ranks())), so a direct PoissonFFTSolver would
        // dereference a nonexistent fab(0) on a rank without a box (SIGSEGV). RemappedFFTSolver presents
        // the SAME single-box layout outward (rhs()/phi() on owner_->ba/dm, aligned with the aux) and
        // hides a scatter/gather around PoissonFFT inside solve() -> the field-solve path is untouched.
        // COLLECTIVE: every rank constructs the same type; its Ny % n_ranks() guard throws on all ranks
        // (no deadlock). NOTE: pending the ADC-273 design vote (structural change to the ell_ variant).
        ell_.emplace(std::in_place_type<RemappedFFTSolver>, owner_->geom, owner_->ba, pbc, active,
                     spectral);
      } else {
        // Single-rank: the proven direct FFT on the System box. PoissonFFTSolver keeps a hard guard in
        // its constructor (rejects n_ranks()>1 / ba.size()!=1).
        ell_.emplace(std::in_place_type<PoissonFFTSolver>, owner_->geom, owner_->ba, pbc, active,
                     spectral);
      }
    } else if (p_solver == "geometric_mg") {
      ell_.emplace(std::in_place_type<GeometricMG>, owner_->geom, owner_->ba, pbc,
                   std::move(active));
      std::get<GeometricMG>(*ell_).set_abs_tol(
          p_abs_tol_);  // absolute floor of the V-cycle (0 = relative only)
      if (has_eps_field_)
        apply_epsilon_field();  // operator div(eps grad phi) with variable eps(x)
      if (has_eps_xy_field_)
        apply_epsilon_anisotropic_field();  // div(diag(eps_x, eps_y) grad phi)
      if (has_kappa_field_)
        apply_reaction_field();  // term - kappa phi (screened Poisson / Helmholtz)
      // Guard: with kappa and a CONSTANT permittivity eps != 1 (without an eps(x) field), the rhs
      // would be scaled by 1/eps (shortcut lap phi = f/eps) -- inconsistent with the term -kappa phi.
      // We then require eps = 1 or an eps(x) field (carried by the operator, without scaling).
      if (has_kappa_field_ && !has_eps_field_ && !has_eps_xy_field_ && p_eps_ != Real(1))
        throw std::runtime_error(
            "System: reaction term kappa(x) + CONSTANT permittivity eps != 1 "
            "unsupported; use eps = 1 or an eps(x) field (set_epsilon_field)");
    } else {
      throw std::runtime_error("System::set_poisson: unknown solver '" + p_solver +
                               "' (geometric_mg|fft|fft_spectral)");
    }
  }
  /// Installs the eps(x) field (n*n row-major) on the GeometricMG: the operator becomes
  /// div(eps grad phi) = f, eps CARRIED BY THE OPERATOR (harmonic face coefficient, order 2),
  /// without 1/eps scaling of the right-hand side. Only GeometricMG supports this variable coefficient.
  void apply_epsilon_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // Filling of the source field LOCAL to the owner rank (iteration over the local fabs, never
    // fab(0) hardcoded): no-op on a rank without a local box at np>1, identical to before on the
    // owner. mg.set_epsilon is then COLLECTIVE (local copy + MPI-safe restriction).
    for (int li = 0; li < eps_fine.local_size(); ++li) {
      Array4 e = eps_fine.fab(li).array();
      const Box2D v = eps_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          e(i, j, 0) = static_cast<Real>(p_eps_field_[static_cast<std::size_t>(j) * n + i]);
    }
    mg.set_epsilon(
        eps_fine);  // copy on the fine level + restriction (average_down) to the coarse ones
  }
  /// Installs the eps_x(x), eps_y(x) fields (n*n row-major each) on the GeometricMG: the operator
  /// becomes div(diag(eps_x, eps_y) grad phi) = f. The faces normal to x read eps_x, those
  /// normal to y read eps_y (harmonic face coefficients, order 2), CARRIED BY THE OPERATOR
  /// without 1/eps scaling of the right-hand side. GeometricMG only (variable tensor coefficient).
  void apply_epsilon_anisotropic_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_x_fine(owner_->ba, owner_->dm, 1, 0), eps_y_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // LOCAL filling on the owner rank (cf. apply_epsilon_field): iteration over the local fabs
    // (no-op on an empty rank at np>1). eps_x_fine and eps_y_fine share ba/dm -> same
    // local indexing. mg.set_epsilon_anisotropic is then COLLECTIVE (copy + restriction).
    for (int li = 0; li < eps_x_fine.local_size(); ++li) {
      Array4 ex = eps_x_fine.fab(li).array();
      Array4 ey = eps_y_fine.fab(li).array();
      const Box2D v = eps_x_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const std::size_t k = static_cast<std::size_t>(j) * n + i;
          ex(i, j, 0) = static_cast<Real>(p_eps_x_field_[k]);
          ey(i, j, 0) = static_cast<Real>(p_eps_y_field_[k]);
        }
    }
    mg.set_epsilon_anisotropic(eps_x_fine,
                               eps_y_fine);  // faces x <- eps_x, faces y <- eps_y (+ restriction)
  }
  /// Installs the reaction term kappa(x) (n*n row-major) on the GeometricMG: the operator becomes
  /// div(eps grad phi) - kappa phi = f (screened Poisson / Helmholtz; kappa = 1/lambda_D^2 for Debye).
  /// kappa is DIAGONAL (read at cell), restricted by averaging to the coarse levels. GeometricMG only.
  void apply_reaction_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab kappa_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // LOCAL filling on the owner rank (cf. apply_epsilon_field): iteration over the local fabs
    // (no-op on an empty rank at np>1). mg.set_reaction is then COLLECTIVE.
    for (int li = 0; li < kappa_fine.local_size(); ++li) {
      Array4 k = kappa_fine.fab(li).array();
      const Box2D v = kappa_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          k(i, j, 0) = static_cast<Real>(p_kappa_field_[static_cast<std::size_t>(j) * n + i]);
    }
    mg.set_reaction(kappa_fine);
  }
  /// Right-hand side f of the active cartesian elliptic solver (GeometricMG or FFT), via std::visit.
  MultiFab& ell_rhs() {
    return std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *ell_);
  }
  /// Potential phi read (and rewritten) by the condensed source stage. CARTESIAN: the phi of the active
  /// elliptic solver (GeometricMG/FFT, WITH ghosts), BIT-IDENTICAL. POLAR: a dedicated 1-ghost buffer
  /// (phi_src_polar_), fed with phi^n (= aux[0], set by solve_fields_polar) at call time
  /// -- the direct PolarPoissonSolver has no ghosts, so we cannot expose pell_->phi()
  /// directly to a stepper that does fill_ghosts/apply_polar_tensor. The stepper writes phi^{n+1} into it
  /// (warm start of the next step; aux[0] will be rewritten anyway by the next solve_fields).
  MultiFab& ell_phi() {
    if (owner_->polar_) {
      // Allocate lazily (1 ghost) on the System layout, then copy phi^n from aux[0].
      if (!phi_src_polar_)
        phi_src_polar_.emplace(owner_->ba, owner_->dm, 1, 1);
      for (int li = 0; li < phi_src_polar_->local_size(); ++li) {
        const ConstArray4 a = owner_->aux.fab(li).const_array();
        Array4 p = phi_src_polar_->fab(li).array();
        const Box2D v = phi_src_polar_->box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            p(i, j, 0) = a(i, j, 0);  // aux[0] = phi^n
      }
      return *phi_src_polar_;
    }
    return std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *ell_);
  }
  /// Solves the active cartesian Poisson (GeometricMG V-cycle or direct FFT). Sets the trace
  /// markers; the device_fence after ell_solve is carried by the CALLER (solve_fields), not here.
  void ell_solve() {
    adc_sf_mark("ell_solve: before std::visit");
    std::visit(
        [](auto& e) {
          using T = std::decay_t<decltype(e)>;
          if (adc_trace_sf())
            adc_sf_mark(std::is_same_v<T, GeometricMG> ? "ell_solve: GeometricMG::solve() start"
                                                       : "ell_solve: FFT solver::solve() start");
          e.solve();
          adc_sf_mark("ell_solve: solve() return");
        },
        *ell_);
    adc_sf_mark("ell_solve: after std::visit");
  }

  // --- direct POLAR Poisson (PolarPoissonSolver) -------------------------
  /// Builds the direct POLAR Poisson (PolarPoissonSolver, single-rank, single box covering
  /// the ring) LAZILY. The radial BC comes from poisson_bc() (Foextrap -> homogeneous Neumann, wall; the
  /// circular cartesian 'wall' makes no sense on a global ring and is not applied). theta is
  /// PERIODIC (handled by the FFT-in-theta, no azimuthal BC). ADDITIVE: never touches ell_.
  /// @throws std::runtime_error on unknown rhs/solver or variable/aniso/reaction permittivity (unsupported
  /// by the direct polar Poisson).
  void ensure_elliptic_polar() {
    if (pell_)
      return;
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson (polar): unknown rhs '" + p_rhs +
                               "' (charge_density|composite)");
    if (p_solver != "geometric_mg" && p_solver != "polar")
      throw std::runtime_error(
          "System::set_poisson (polar): solver '" + p_solver +
          "' unsupported on a ring; the polar Poisson is direct (FFT-in-theta + tridiag-in-r). "
          "Leave the default ('geometric_mg') or request 'polar'.");
    if (has_eps_field_ || has_eps_xy_field_ || has_kappa_field_)
      throw std::runtime_error(
          "System::set_poisson (polar): variable / anisotropic permittivity / reaction unsupported "
          "by the direct polar Poisson (Phase 2b; operator (1/r) d_r(r d_r) + (1/r^2) d_theta^2)");
    // MULTI-BOX GUARD (ADC-67): the DIRECT polar Poisson (FFT-in-theta + tridiag-in-r) requires
    // complete theta ROWS and radial COLUMNS on ONE box (PolarPoissonSolver already rejects
    // ba.size()!=1). We raise a clear UPSTREAM message HERE -- before the construction of the solver, so from the
    // 1st solve_fields / potential / set_potential of a System with theta_boxes > 1 -- rather than letting
    // the low-level rejection of PolarPoissonSolver bubble up. The theta SPLITTING (theta_boxes > 1) only serves
    // the TRANSPORT; for a multi-box electrostatic field, go through the tensor Schur stage.
    if (owner_->ba.size() != 1)
      throw std::runtime_error(
          "System: DIRECT polar Poisson incompatible with the theta splitting (theta_boxes=" +
          std::to_string(owner_->ba.size()) +
          " boxes); it requires a single-box grid (theta_boxes=1). For a multi-box theta "
          "splitting, "
          "use the polar tensor Schur stage (adc.Split + adc.CondensedSchur), multi-box, or "
          "go back to theta_boxes=1.");
    // Radial BC: Dirichlet/Neumann from poisson_bc() (xlo/xhi). theta always periodic.
    const BCRec pbc = poisson_bc();
    pell_.emplace(owner_->pgeom_, owner_->ba, pbc);
  }

  /// Assembles the system Poisson RIGHT-HAND SIDE into @p rhs: f = Sum_s elliptic_rhs_s(U_s),
  /// the elliptic brick of EACH block, summed in place (rhs is zeroed first). When
  /// @p target_block >= 0 and @p U_stage != nullptr, the target block reads @p U_stage INSTEAD of
  /// its live s.U (the rest of the blocks keep s.U); this is what solve_fields_from_state needs to
  /// re-solve a field-coupled multi-stage scheme from a per-stage state. With the default
  /// (target_block = -1) it is BIT-IDENTICAL to the historical inline loop (every block from s.U).
  /// STRIDE: a held (hold-then-catch-up) block stays FROZEN at its last advance, so its charge enters
  /// the sum with a STALE state until its next catch-up (loose Poisson coupling, assumed).
  void assemble_poisson_rhs(MultiFab& rhs, int target_block = -1,
                            const MultiFab* U_stage = nullptr) {
    rhs.set_val(Real(0));
    for (std::size_t b = 0; b < owner_->sp.size(); ++b) {
      auto& s = owner_->sp[b];
      const bool override_here = (U_stage != nullptr && static_cast<int>(b) == target_block);
      s.add_poisson_rhs(override_here ? *U_stage : s.U, rhs);
    }
  }

  /// POLAR solve_fields: assembles f = Sum_s elliptic_rhs_s(U_s) (host loop per cell), solves the
  /// polar Poisson, then DERIVES the aux in the local basis (e_r, e_theta):
  ///   aux[0] = phi;  aux[1] = grad_r = d phi/dr;  aux[2] = grad_theta = (1/r) d phi/d theta.
  /// This is the layout expected by ExBVelocityPolar (v_r = -grad_theta/B, v_theta = grad_r/B).
  /// @p target_block / @p U_stage: per-stage state override for the target block (default -1: every
  /// block from s.U, bit-identical to the historical path).
  void solve_fields_polar(int target_block = -1, const MultiFab* U_stage = nullptr) {
    ensure_elliptic_polar();
    MultiFab& rhs = pell_->rhs();
    assemble_poisson_rhs(rhs, target_block, U_stage);
    // CONSTANT permittivity eps != 1: lap phi = f/eps (1/eps scaling of the rhs), like the
    // cartesian. (variable/aniso eps(x) is refused by ensure_elliptic_polar.)
    if (p_eps_ != Real(1)) {
      const Real inv = Real(1) / p_eps_;
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const Box2D v = rhs.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            r(i, j, 0) *= inv;
      }
    }
    pell_->solve();
    device_fence();
    // Derivation (phi, grad_r, grad_theta) in the local basis (e_r, e_theta) via the SAME helper as the C++
    // test (derive_aux_polar of block_builder_polar.hpp). phi is WITHOUT ghost (direct single-box solver):
    // the helper thus never reads an out-of-domain index (radial OFFSET at the walls, theta WRAPPED in
    // periodic) -- that was the bug: the centered difference read phi(lo-1)/phi(hi+1)/phi(.,jlo-1) out of
    // allocation -> spurious gradient -> divergent velocity -> nan.
    derive_aux_polar(pell_->phi(), owner_->aux, owner_->pgeom_);
    apply_te();  // inert in polar ExB (no fluid block source of T_e), kept by symmetry
    // Aux ghosts: theta PERIODIC (joint 0/2pi), r PHYSICAL (extrapolation at the boundary). fill_ghosts
    // already routes through bc_ (xlo/xhi Foextrap, ylo/yhi Periodic) -> correct periodic azimuthal halo.
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();  // ADC-369: per-field halo override on the RADIAL faces (theta stays periodic)
  }

  /// Solves the system Poisson then DERIVES the aux = (phi, grad phi[, B_z, T_e]). Routes to
  /// solve_fields_polar() in polar geometry. device INVARIANT: the device_fence() between ell_solve()
  /// and the derivation of grad phi MUST stay atomic (without it the GPU V-cycle is not finished when
  /// phi is read); the derivation / population loops iterate over the LOCAL fabs (MPI-safe).
  ///
  /// @p target_block / @p U_stage: when set (target_block >= 0, U_stage != nullptr), the target
  /// block's Poisson RHS is assembled from @p U_stage INSTEAD of its live s.U -- the seam
  /// solve_fields_from_state uses so a field-coupled multi-stage scheme re-solves phi from each STAGE
  /// state (the compiled Program runs stages sequentially: stage k's solve overwrites the shared aux
  /// before stage k's RHS reads it). The default (-1 / nullptr) keeps every block at s.U: the
  /// historical solve_fields(), BIT-IDENTICAL. Inert under the gauss_evolve_ skip (no RHS assembled).
  void solve_fields(int target_block = -1, const MultiFab* U_stage = nullptr) {
    if (owner_->polar_)
      // ring: polar Poisson + aux in local basis (e_r, e_theta)
      return solve_fields_polar(target_block, U_stage);
    adc_sf_mark("solve_fields: start");
    ensure_elliptic();
    adc_sf_mark("solve_fields: after ensure_elliptic");
    // GAUSS POLICY: "restart" (default, gauss_evolve_==false) re-solves -Delta phi = f on
    // EVERY call (bit-identical to history). "evolve": after the FIRST solve (phi^0), we SKIP
    // the RHS assembly + the elliptic solve -> ell_phi() keeps the current phi (the one that the
    // Schur source stage evolves in-place), giving a restart-free -Delta phi evolution (Gauss imposed only at t=0).
    // The derivation of the aux (phi, grad phi) below ALWAYS runs, on the current phi.
    if (!(gauss_evolve_ && gauss_solved_once_)) {
      MultiFab& rhs = ell_rhs();
      // f = Sum_s elliptic_rhs_s(U_s). By default the CURRENT state of each block; with a
      // target_block / U_stage override the target block reads U_stage (per-stage field solve, ADC-409).
      assemble_poisson_rhs(rhs, target_block, U_stage);
      adc_sf_mark("solve_fields: after add_poisson_rhs");
      // CONSTANT permittivity: div(eps grad phi) = f <=> lap phi = f/eps, so we scale the rhs by
      // 1/eps. With a VARIABLE or ANISOTROPIC eps(x) field we DO NOT do it: the GeometricMG
      // operator carries eps directly (apply_epsilon_field / apply_epsilon_anisotropic_field), the
      // rhs stays f as is.
      if (!has_eps_field_ && !has_eps_xy_field_ && p_eps_ != Real(1)) {
        const Real inv = Real(1) / p_eps_;
        for (int li = 0; li < rhs.local_size(); ++li) {
          Array4 r = rhs.fab(li).array();
          const Box2D v = rhs.box(li);
          for (int j = v.lo[1]; j <= v.hi[1]; ++j)
            for (int i = v.lo[0]; i <= v.hi[0]; ++i)
              r(i, j, 0) *= inv;
        }
      }
      adc_sf_mark("solve_fields: before ell_solve");
      ell_solve();
      gauss_solved_once_ = true;
      adc_sf_mark("solve_fields: after ell_solve, before device_fence");
    }
    device_fence();
    adc_sf_mark("solve_fields: after device_fence (aux derivation)");
    const Real dx = owner_->geom.dx(), dy = owner_->geom.dy();
    // Per-cell derivation (phi, grad phi) -> aux channel: LOCAL to the owner rank. System
    // distributes ONE box (round-robin DistributionMapping(1, n_ranks())), so at np>1 a single rank
    // owns it; the others have local_size()==0 and HAVE NO fab to derive. We iterate over the LOCAL
    // fabs (never fab(0) hardcoded): no-op on an empty rank, identical to before on the owner
    // (loop executed once, bit-identical to np=1). ell_phi() and aux share the same
    // DistributionMapping -> same local indexing.
    MultiFab& phi_mf = ell_phi();
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 p = phi_mf.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          a(i, j, 0) = p(i, j);
          a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
          a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
        }
    }
    adc_sf_mark("solve_fields: after aux derivation (phi, grad phi)");
    apply_te();  // T_e = p/rho of the fluid block source, recomputed on each solve (B_z, comp 3, preserved)
    adc_sf_mark("solve_fields: after apply_te");
    if (owner_->periodic_)
      fill_boundary(owner_->aux, owner_->dom, owner_->per_);
    else
      fill_ghosts(owner_->aux, owner_->dom,
                  owner_->bc_);  // extrapolation at the boundary (wall / free outflow)
    apply_named_aux_bc();  // ADC-369: per-field halo override (after the shared fill; no-op if none)
    adc_sf_mark("solve_fields: end (fill ghosts aux)");
  }

  /// Per-stage field solve (ADC-409): SAME solve + derive-aux as solve_fields(), but the target
  /// block @p block_idx assembles its Poisson RHS from @p U_stage instead of its live s.U (the other
  /// blocks keep s.U). This re-fills the SHARED aux (phi, grad phi) with phi(U_stage); a compiled
  /// Program's stage-k RHS, called right after this, then reads phi solved from stage k's own state --
  /// removing the "solve from current state only" limitation for sequential multi-stage schemes. The
  /// next stage's solve overwrites the aux, so no distinct per-stage buffer is needed. With
  /// block_idx == 0 and U_stage == U^n (the first stage) this is identical to solve_fields().
  /// @throws std::out_of_range if @p block_idx is not a valid block index.
  void solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
    if (block_idx < 0 || block_idx >= static_cast<int>(owner_->sp.size()))
      throw std::out_of_range("solve_fields_from_state: block index " + std::to_string(block_idx) +
                              " out of range (" + std::to_string(owner_->sp.size()) + " blocks)");
    solve_fields(block_idx, &U_stage);
  }

  // --- NAMED multi-elliptic field (ADC-428) ----------------------------------
  /// Builds the DEDICATED cartesian elliptic solver of named field @p nf, the SAME poisson operator as
  /// the default ell_ (ensure_elliptic): GeometricMG (default) or PoissonFFTSolver/RemappedFFTSolver,
  /// with the System's Poisson BC and wall. REUSES the native solver -- no operator is reimplemented.
  /// The variable / anisotropic permittivity and reaction coefficients of the DEFAULT Poisson are NOT
  /// carried onto a named field (a named field is a plain Laplacian; its own coefficients are a future
  /// extension). Built lazily; no-op if already built.
  void ensure_named_elliptic(NamedField& nf) {
    if (nf.ell)
      return;
    const BCRec pbc = poisson_bc();
    std::function<bool(Real, Real)> active = wall_active();
    if (p_solver == "fft" || p_solver == "fft_spectral") {
      if (active)
        throw std::runtime_error("System: named elliptic field solver '" + p_solver +
                                 "' incompatible with a wall -> 'geometric_mg'");
      const bool spectral = (p_solver == "fft_spectral");
      if (n_ranks() > 1)
        nf.ell.emplace(std::in_place_type<RemappedFFTSolver>, owner_->geom, owner_->ba, pbc, active,
                       spectral);
      else
        nf.ell.emplace(std::in_place_type<PoissonFFTSolver>, owner_->geom, owner_->ba, pbc, active,
                       spectral);
    } else if (p_solver == "geometric_mg") {
      nf.ell.emplace(std::in_place_type<GeometricMG>, owner_->geom, owner_->ba, pbc,
                     std::move(active));
      std::get<GeometricMG>(*nf.ell).set_abs_tol(p_abs_tol_);
    } else {
      throw std::runtime_error("System: named elliptic field solver '" + p_solver +
                               "' unsupported (geometric_mg|fft|fft_spectral)");
    }
  }

  /// Assembles the RIGHT-HAND SIDE of named field @p field into @p rhs: f = Sum_s
  /// named_poisson_rhs_s[field](U_s), the per-field elliptic brick of EACH block that declares it
  /// (ADC-428). When @p target_block >= 0 and @p U_stage != nullptr, the target block reads @p U_stage
  /// instead of its live s.U (per-stage field solve, like assemble_poisson_rhs). @throws if no block
  /// declares this field (a named field with no contributing block would solve a zero RHS silently).
  void assemble_named_poisson_rhs(const std::string& field, MultiFab& rhs, int target_block,
                                  const MultiFab* U_stage) {
    rhs.set_val(Real(0));
    bool any = false;
    for (std::size_t b = 0; b < owner_->sp.size(); ++b) {
      auto& s = owner_->sp[b];
      auto it = s.named_poisson_rhs.find(field);
      if (it == s.named_poisson_rhs.end() || !it->second)
        continue;
      const bool override_here = (U_stage != nullptr && static_cast<int>(b) == target_block);
      it->second(override_here ? *U_stage : s.U, rhs);
      any = true;
    }
    if (!any)
      throw std::runtime_error("System: named elliptic field '" + field +
                               "' has no contributing block (declare m.elliptic_field on the block "
                               "model)");
  }

  /// Solves named field @p field's SECOND elliptic problem from block @p block_idx's stage state
  /// @p U_stage and writes phi (+ centered grad) into the field's OWN aux components (ADC-428):
  /// assemble f = Sum_s named_poisson_rhs_s[field] -> ell.solve() (the dedicated native solver) ->
  /// aux[phi_comp] = phi, aux[gx_comp]/aux[gy_comp] = centered grad. The SHARED phi/grad (components
  /// 0..2) and the default Poisson (ell_) are NOT touched. The named aux components are then ghost-
  /// filled (the shared aux fill + the per-field halo override). CARTESIAN only (the polar named path is
  /// a future extension); @throws on the polar geometry or an unknown field.
  void solve_named_field_from_state(const std::string& field, int block_idx,
                                    const MultiFab& U_stage) {
    if (block_idx < 0 || block_idx >= static_cast<int>(owner_->sp.size()))
      throw std::out_of_range("solve_fields_from_state (named): block index " +
                              std::to_string(block_idx) + " out of range (" +
                              std::to_string(owner_->sp.size()) + " blocks)");
    auto it = named_fields_.find(field);
    if (it == named_fields_.end())
      throw std::runtime_error("System: unknown named elliptic field '" + field +
                               "' (register it via m.elliptic_field + the compiled block)");
    if (owner_->polar_)
      throw std::runtime_error("System: named elliptic field '" + field +
                               "' on a polar (ring) grid is not supported yet (cartesian only)");
    NamedField& nf = it->second;
    if (nf.phi_comp < 0 || nf.phi_comp >= owner_->aux_ncomp_)
      throw std::runtime_error(
          "System: named elliptic field '" + field +
          "' aux component out of the channel width (add the block that declares "
          "its aux fields before solving)");
    ensure_named_elliptic(nf);
    MultiFab& rhs = std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *nf.ell);
    assemble_named_poisson_rhs(field, rhs, block_idx, &U_stage);
    // CONSTANT permittivity eps != 1: lap phi = f/eps (1/eps scaling), like the default Poisson. The
    // variable / anisotropic eps(x) and reaction kappa of the DEFAULT Poisson are NOT applied to a
    // named field (plain Laplacian operator; see ensure_named_elliptic).
    if (p_eps_ != Real(1)) {
      const Real inv = Real(1) / p_eps_;
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const Box2D v = rhs.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            r(i, j, 0) *= inv;
      }
    }
    std::visit([](auto& e) { e.solve(); }, *nf.ell);
    device_fence();  // CRITICAL: the V-cycle must finish before phi is read (same invariant as ell_)
    MultiFab& phi_mf = std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *nf.ell);
    const Real dx = owner_->geom.dx(), dy = owner_->geom.dy();
    const int cphi = nf.phi_comp, cgx = nf.gx_comp, cgy = nf.gy_comp;
    const bool grad =
        (cgx >= 0 && cgx < owner_->aux_ncomp_ && cgy >= 0 && cgy < owner_->aux_ncomp_);
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 p = phi_mf.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          a(i, j, cphi) = p(i, j);
          if (grad) {
            a(i, j, cgx) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
            a(i, j, cgy) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
          }
        }
    }
    // Ghost-fill the named field's aux components: the shared aux fill (same routing as solve_fields)
    // then the per-field halo override (ADC-369). This re-fills ALL components -- the shared phi/grad
    // were last written by the default solve_fields, so their valid cells are unchanged and only the
    // halos are recomputed (idempotent for those components).
    if (owner_->periodic_)
      fill_boundary(owner_->aux, owner_->dom, owner_->per_);
    else
      fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();
  }

 private:
  Impl* owner_;
};

}  // namespace field_solver
}  // namespace adc
