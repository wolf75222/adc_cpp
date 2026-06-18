#include <adc/runtime/system.hpp>

#include <adc/core/variables.hpp>  // VariableSet + VariableRole: role descriptor carried by each block
#include <adc/runtime/abi_key.hpp>  // adc::abi_key + detail::abi_key_string (ABI boundary of the native loader)
#include <adc/runtime/block_builder.hpp>  // GridContext + make_block/make_max_speed (compiled closures)
#include <adc/runtime/block_seam.hpp>  // ADC-335: per-transport build seam (build_block_exb/.../polar)
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <adc/coupling/condensed_schur_source_stepper.hpp>  // Schur-condensed source stage (adc.Split / CondensedSchur, #126)
#include <adc/coupling/polar_condensed_schur_source_stepper.hpp>  // POLAR counterpart of the condensed source stage (Path A step 2c, #212)
#include <adc/coupling/coupled_source_program.hpp>  // CoupledSourceKernel: generic coupled source (DSL P5, bytecode)
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/polar_poisson_solver.hpp>  // PolarPoissonSolver (direct polar Poisson, REUSED)
#include <adc/runtime/system_field_solver.hpp>  // SystemFieldSolver: elliptic solve + field derivation (Batch B)
#include <adc/runtime/system_stepper.hpp>  // SystemStepper: time advance (step/advance/step_cfl/step_adaptive) (Batch B)
#include <adc/runtime/system_block_store.hpp>  // SystemBlockStore: block management (BlockState + registry + index/copy/write) (Batch B.3)
#include <adc/runtime/block_builder_polar.hpp>  // POLAR block closures (assemble_rhs_polar, REUSED)
#include <adc/numerics/time/implicit_stepper.hpp>   // backward_euler_source
#include <adc/numerics/time/time_steppers.hpp>      // ForwardEuler, SSPRK2Step (core RK math)
#include <adc/numerics/spatial_operator.hpp>     // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts, fill_boundary
#include <adc/runtime/dynamic_model.hpp>  // IModel: model loaded at runtime (dynamic block)
#include <adc/runtime/native_loader.hpp>  // .so loading (JIT/AOT/native) + ABI guard: VERBATIM, included after the Impl def below (templates instantiated lower down)
#include <adc/runtime/wall_predicate.hpp>  // detail::wall_predicate (wall shared by System/AmrSystem)

#include <algorithm>
#include <cmath>
#include <cstdio>   // ADC_TRACE_SOLVE_FIELDS: device diagnostic trace (env-gated, inert by default)
#include <cstdlib>  // getenv
#include <adc/runtime/dynlib.hpp>  // portable dlopen<->LoadLibraryW layer (ADC-99); <dlfcn.h> on POSIX
#include <functional>
#include <limits>  // std::numeric_limits (per-block CFL: dt = min over blocks)
#include <map>     // std::map (per-block runtime params registry, P7-b)
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace adc {

// The DIAGNOSTIC trace of the solve_fields path (adc_trace_sf / adc_sf_mark, milestone #93) was extracted
// with SystemFieldSolver into include/adc/runtime/system_field_solver.hpp (namespace field_solver);
// it stays env-gated (ADC_TRACE_SOLVE_FIELDS) and inert by default.
// resolve_implicit_components moved to model_factory.hpp (adc::detail) so the per-transport seam TUs
// (python/system_<transport>.cpp, ADC-335) share one definition; it is otherwise unchanged.

// MODULE ABI key (frozen at compile time of this TU). Defined here so the _adc module
// exports it (ADC_EXPORT): add_native_block compares it to the key baked into the loader .so.
ADC_EXPORT std::string abi_key() { return detail::abi_key_string(); }

// Convenience static method (Python binding + add_native_block): delegates to the module's free key.
std::string System::abi_key() { return adc::abi_key(); }

namespace {
// Index of the component carrying @p role in @p vs, or @p fallback if the block does not provide
// this role (dynamic / compiled block: descriptor without roles). Lets couplings target
// a component by its MEANING without hard-coding the index, while staying backward-compatible.
int role_index(const VariableSet& vs, VariableRole role, int fallback) {
  const int c = vs.index_of(role);
  return c >= 0 ? c : fallback;
}
}  // namespace

struct System::Impl {
  // BLOCK MANAGEMENT extracted into SystemBlockStore (Batch B.3, last P0 extraction from the god-class):
  // the block struct (formerly Species, renamed BlockState), the ordered registry (blocks_.blocks), the
  // by-name access (index / find) and the state marshaling (copy_comp0 / copy_state / write_state) now
  // live there. See include/adc/runtime/system_block_store.hpp.
  //
  // COMPATIBILITY ALIASES. The already-extracted header templates (SystemFieldSolver, SystemStepper,
  // native_loader) iterate `owner_->sp` / `P->sp` and name `Impl::Species`; we keep these two
  // access points identical (zero churn outside this file):
  //  - `Species` = the block type carried by the store (init via positional aggregate unchanged);
  //  - `sp` = a REFERENCE to the store registry (same object, same iteration, same indexing).
  using Species = SystemBlockStore::BlockState;

  SystemConfig cfg;
  Geometry geom;
  // POLAR GEOMETRY (diocotron polar-grid project, Phase 2b). polar_ == true when
  // cfg.geometry == "polar": the System then runs on a global ring (r, theta), with the polar
  // transport (assemble_rhs_polar) and the polar Poisson (PolarPoissonSolver) instead of the Cartesian path.
  // pgeom_ is the ring (r_min, r_max, nr, ntheta); INERT (never read) in Cartesian -> bit-identical
  // path. dom/ba/dm always cover the INDEX space (nx() x ny()), common to both
  // geometries: only the indices -> physical space mapping (geom vs pgeom_) changes.
  bool polar_;
  PolarGeometry pgeom_;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // transport BC (periodic or Foextrap per cfg.periodic; polar: physical r, periodic theta)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  int aux_ncomp_ = kAuxBaseComps;     // width of the SHARED aux channel (max over blocks; >= 3)

  // DISC DOMAIN MASK (project T2, contract inert by default). disc_set_ == false: no fixed
  // disc -> the mask is "all active" and the transport path stays BIT-IDENTICAL. When
  // set_disc_domain is called, disc_ carries the descriptor (center + radius, SINGLE SOURCE reusing
  // the conducting-wall level set) and disc_mask_ materializes the 0/1 cell-centered field (1 ghost,
  // so the mask-aware transport reads neighbors). The mask is BUILT but NOT yet wired
  // into step() (scaffolding); it is queryable (disc_mask()) and consumed by assemble_rhs_masked.
  detail::DiscDomain disc_;
  bool disc_set_ = false;
  MultiFab disc_mask_;  // 0/1 cell-centered, same layout as the blocks (ba/dm), 1 ghost; empty as long as !disc_set_
  // At least one block requested wave_speed_cache (ADC-199, opt-in HLL cache). The cache is only wired
  // on the FULL Cartesian advance (advance); the disc advances (advance_masked / advance_eb) do not
  // carry it. This flag locks the switch to a disc transport mode (staircase/cutcell) -> explicit
  // rejection in set_disc_domain / set_geometry_mode rather than a silently ignored cache.
  bool ws_cache_block_ = false;
  // TRANSPORT GEOMETRY MODE (project T5-PR3, wiring the disc into step()). None (default):
  // full Cartesian transport (assemble_rhs) -> BIT-IDENTICAL. Staircase: assemble_rhs_masked (0/1
  // mask). CutCell: assemble_rhs_eb (cut-cell EB). The stepper reads this mode to ROUTE the transport
  // advance of each block (advance vs advance_masked vs advance_eb). Set by set_disc_domain(mode=)
  // / set_geometry_mode; has effect only if a disc is fixed (disc_set_) AND the block carries the
  // matching disc advance. None as long as no disc mode is requested.
  GeometryMode geometry_mode_ = GeometryMode::None;
  // aux APPLICATION fields (bz_field_, te_src_) and apply_bz/apply_te buffers EXTRACTED into
  // fields_ (SystemFieldSolver, Batch B); the SHARED aux and its width stay here (common channel).
  // Block registry OWNED by the store (Batch B.3). `sp` is a REFERENCE to blocks_.blocks: same
  // object (no copy), so owner_->sp / P->sp in the header templates stay bit-identical.
  SystemBlockStore blocks_;
  std::vector<Species>& sp = blocks_.blocks;
  // P7-b: RUNTIME parameter values per AOT block (block name -> vector of current values).
  // The vector is SHARED (shared_ptr) with the compiled block closures: writing into it
  // (set_block_params) changes the block behavior at the next step WITHOUT recompiling. Absent for a
  // block without runtime param or for the other paths (native / dynamic). Set by add_compiled_block.
  std::map<std::string, std::shared_ptr<std::vector<double>>> block_params_;
  // Newton diagnostics (IMEX, OPT-IN): per-block report, owned HERE in shared_ptr (STABLE address
  // even when sp reallocates); the block AdvanceImex* closures write into it via raw pointer.
  // Absent (missing key) for a block without newton_diagnostics -> newton_report raises a clear error.
  std::map<std::string, std::shared_ptr<NewtonReport>> newton_reports_;
  double t = 0;
  int macro_step_ = 0;  // macro-step counter (0-indexed): feeds the per-block stride filter
  std::vector<std::function<void(Real)>> couplings;  // inter-species coupled sources (splitting)
  // GLOBAL time-step bounds (System::add_dt_bound): evaluated ONCE per step (host) by
  // step_cfl / step_adaptive. Hook for non-cell-local constraints (multi-block coupling,
  // Schur/Poisson, scheduler). Empty (default) -> historical step policy, bit-identical.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  // DECLARED frequencies of coupled sources (CoupledSource.frequency, audit wave 3): the
  // couplings apply ONCE per MACRO-step (apply_couplings(dt)), so the bound is on the
  // macro-dt: dt <= cfl / mu, WITHOUT a substeps/stride factor. Empty (default) -> no bound.
  struct CoupledFreq {
    std::string label;
    double mu;
  };
  std::vector<CoupledFreq> coupled_freqs_;
  // PER-CELL frequencies of coupled sources (CoupledSource.frequency with an Expr, refinement
  // of the CONSTANT frequency above): a bytecode program mu(U) evaluated per cell at EVERY
  // step (MAX reduction, global all_reduce_max), bound dt <= cfl / max(mu). The inputs REUSE the
  // resolve() resolution of the input registers (sidx, comp); the constants are the same as the
  // source. Empty (default) -> no per-cell bound (historical path). Stored AFTER full
  // validation (same anti-phantom-bound rule as the scalar).
  struct CoupledFreqExpr {
    std::string label;
    CsProgram prog;
    struct In { int sidx, comp; };
    std::vector<In> ins;  // (species, component) of the inputs (same as the source; resolved once)
    int n_in = 0;
    std::vector<Real> kconsts;  // constants loaded into r[n_in ..] (same as the source)
  };
  std::vector<CoupledFreqExpr> coupled_freq_exprs_;

  // stride_due (hold-then-catch-up cadence filter) EXTRACTED into stepper_ (SystemStepper, Batch B):
  // it serves exclusively the time advance. macro_step_ (above) stays a SHARED member of Impl
  // (read by time() indirectly via t, incremented by stepper_ via owner_->macro_step_).

  // Elliptic solve + field derivation EXTRACTED into fields_ (SystemFieldSolver, Batch B,
  // cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2): the Poisson configuration (p_rhs/p_solver/
  // p_bc/p_wall/p_wall_radius/p_eps_), the coefficient fields (eps(x), eps_x/eps_y, kappa), the
  // solvers (Cartesian ell_, polar pell_) and the aux application buffers (B_z, T_e) live there
  // now. fields_ reads the SHARED aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_ of Impl via its
  // back-pointer. Declared after the shared members it captures (initialized in the constructor).

  // Number of radial / azimuthal cells in POLAR (0 => fall back to cfg.n, cf. SystemConfig).
  static int polar_nr(const SystemConfig& c) { return c.nr > 0 ? c.nr : c.n; }
  static int polar_ntheta(const SystemConfig& c) { return c.ntheta > 0 ? c.ntheta : c.n; }
  // INDEX domain: n x n square in Cartesian; nr x ntheta in polar (i = r, j = theta).
  static Box2D index_domain(const SystemConfig& c) {
    if (c.geometry == "polar") return Box2D::from_extents(polar_nr(c), polar_ntheta(c));
    return Box2D::from_extents(c.n, c.n);
  }
  // Number of cells of a cell-defined field (n*n Cartesian / nr*ntheta polar), for the
  // size check of named aux fields (set_aux_field).
  std::size_t aux_field_cell_count() const;
  // BoxArray of the INDEX domain. Cartesian (and polar mono-box, theta_boxes <= 1): ONE box covering
  // the whole domain -> STRICTLY bit-identical to the historical (ba = {index_domain}). Polar with
  // theta_boxes > 1: split into theta BANDS -- each box covers the whole radius [0, nr-1] and one
  // contiguous azimuthal band (same bounds as theta_split in test_polar_schur_multibox). The bands
  // tile [0, ntheta-1] EXACTLY (base + remainder lengths, but theta_boxes divides ntheta -> equal
  // bands). check_geometry already validates 1 <= theta_boxes <= ntheta and the divisibility.
  static BoxArray index_boxarray(const SystemConfig& c) {
    if (c.geometry != "polar" || c.theta_boxes <= 1)
      return BoxArray(std::vector<Box2D>{index_domain(c)});
    const int nr = polar_nr(c), nth = polar_ntheta(c), nseg = c.theta_boxes;
    std::vector<Box2D> boxes;
    boxes.reserve(static_cast<std::size_t>(nseg));
    int base = nth / nseg, rem = nth % nseg, cur = 0;
    for (int k = 0; k < nseg; ++k) {
      const int len = base + (k < rem ? 1 : 0);
      boxes.push_back(Box2D{{0, cur}, {nr - 1, cur + len - 1}});
      cur += len;
    }
    return BoxArray(std::move(boxes));
  }

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        polar_(c.geometry == "polar"),
        pgeom_{index_domain(c), Real(c.r_min), Real(c.r_max)},
        // ba: ONE box (Cartesian or polar mono-box); theta bands if polar theta_boxes > 1
        // (TRANSPORT split). dm: round-robin (nboxes, nranks) -- 1 box -> dm(1, n_ranks())
        // bit-identical to the historical.
        // SINGLE-BOX INVARIANT (Cartesian): ba.size()==1 with the round-robin dm puts box 0 on rank
        // dm[0] (the owner), so under MPI (n_ranks()>1) every other rank holds an empty fab
        // (local_size()==0). RemappedFFTSolver -- the "fft"/"fft_spectral" Poisson under MPI -- is
        // COUPLED to exactly this layout: it presents rhs()/phi() on this same ba/dm outward and hides
        // the box<->slab scatter/gather inside solve(), so the field-solve path stays layout-agnostic
        // (np==1 uses the single-rank PoissonFFTSolver; a genuinely slab-distributed domain would use
        // DistributedFFTSolver instead). Stated here so the coupling is visible at the layout's source;
        // see include/adc/numerics/elliptic/poisson_fft_solver.hpp (RemappedFFTSolver CONTRACT).
        ba(index_boxarray(c)),
        dm(ba.size(), n_ranks()),
        bc_(make_bc(c)),
        dom(index_domain(c)),
        per_{!polar_ && c.periodic, !polar_ && c.periodic},
        periodic_(!polar_ && c.periodic),
        aux(ba, dm, kAuxBaseComps, 1),
        fields_(this),
        stepper_(this) {}

  // Elliptic solve + field derivation (Batch B). OWNS the solvers (ell_/pell_), the Poisson
  // config, the coefficient fields and the aux application buffers (B_z, T_e). owner_ = this: the
  // helper reads the SHARED aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_/periodic_/polar_ of Impl. None of
  // these accesses dereferences Impl at CONSTRUCTION (pure back-pointer) -> init at end of list without
  // ordering dependency. See include/adc/runtime/system_field_solver.hpp.
  field_solver::SystemFieldSolver<Impl> fields_;

  // Time advance (Batch B). ORCHESTRATES step / advance / step_cfl / step_adaptive, the cadence filter
  // (stride_due), the condensed source stage (run_source_stage) and the couplings (apply_couplings). owner_
  // = this: the stepper reads the SHARED sp / fields_ / aux / couplings / t / macro_step_ / geom / pgeom_ / polar_
  // of Impl via its back-pointer. Pure back-pointer at construction (no dereferencing) ->
  // init at end of list without ordering dependency. See include/adc/runtime/system_stepper.hpp.
  stepper::SystemStepper<Impl> stepper_;

  // Guarantees an aux width >= ncomp (SHARED channel). Reallocating the aux KEEPS its address (member:
  // the block closures capture &aux via grid_ctx) and re-applies B_z. No-op if already wide enough.
  void ensure_aux_width(int ncomp) {
    if (ncomp <= aux_ncomp_) return;
    aux_ncomp_ = ncomp;
    aux = MultiFab(ba, dm, aux_ncomp_, 1);
    fields_.apply_bz();
    fields_.apply_te();
    fields_.apply_named_aux();  // re-applies the NAMED aux fields (ADC-70): the redistributed MultiFab starts at zero
  }

  // apply_bz (population of the B_z component of the aux channel) EXTRACTED into fields_ (SystemFieldSolver).

  // Guarantees that the state U of block @p name carries at least @p ng ghosts (spatial scheme stencil).
  // WENO5 reads 3 ghosts, > the 2 allocated by default in install_block; without this width,
  // fill_ghosts + assemble_rhs would read out of bounds (cf. AmrSystem which allocates with Limiter::n_ghost,
  // PR #22). Reallocates the MultiFab and COPIES the valid cells (set_density may have preceded);
  // no-op if U already has enough ghosts -> allocation and data bit-identical to before for MUSCL.
  void set_block_ghosts(const std::string& name, int ng) {
    Species& s = find(name);
    if (s.U.n_grow() >= ng) return;
    MultiFab nu(s.U.box_array(), s.U.dmap(), s.ncomp, ng);
    nu.set_val(Real(0));
    for (int li = 0; li < s.U.local_size(); ++li) {
      const ConstArray4 old = s.U.fab(li).const_array();
      Array4 dst = nu.fab(li).array();
      const Box2D v = s.U.box(li);  // valid cells (excluding ghost): copied as-is
      for (int c = 0; c < s.ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) dst(i, j, c) = old(i, j, c);
    }
    s.U = std::move(nu);
  }

  // kTeComp (canonical T_e component) and apply_te (population of T_e = p/rho of the source block)
  // EXTRACTED into fields_ (SystemFieldSolver): T_e is part of the aux field application.

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodic by default
    if (c.geometry == "polar") {
      // POLAR: r (dir 0, xlo/xhi) carries a PHYSICAL BC (wall / free outflow, Foextrap); theta
      // (dir 1, ylo/yhi) is PERIODIC (the ring covers [0, 2pi)). This is the convention of
      // test_polar_transport_mms and of assemble_rhs_polar (periodic theta, physical r).
      b.xlo = b.xhi = BCType::Foextrap;
      b.ylo = b.yhi = BCType::Periodic;
      return b;
    }
    if (!c.periodic) b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  // By-name access DELEGATED to the store (Batch B.3): same linear search, same indexing by
  // insertion order, same error message ("System: unknown block '...'").
  Species& find(const std::string& name) { return blocks_.find(name); }
  const Species& find(const std::string& name) const { return blocks_.find(name); }
  int index(const std::string& name) const { return blocks_.index(name); }

  // apply_couplings (inter-species coupling sources by splitting, AFTER transport) and
  // run_source_stage (Schur-condensed source stage, OPT-IN) EXTRACTED into stepper_ (SystemStepper,
  // Batch B): these are time-advance steps, invoked by step / step_cfl / step_adaptive.
  // They read the SHARED state via owner_-> (couplings, fields_.ell_phi(), aux, kAuxBaseComps). The
  // couplings list (above) stays a member of Impl (populated by add_ionization / add_collision / ...).

  // --- elliptic solver (system Poisson) -----------------------------
  // poisson_bc / wall_active / ensure_elliptic / apply_epsilon_field / apply_epsilon_anisotropic_field
  // / apply_reaction_field / ell_rhs / ell_phi / ell_solve / ensure_elliptic_polar / solve_fields_polar
  // / solve_fields EXTRACTED into fields_ (SystemFieldSolver, Batch B). See the header.

  // --- compiled spatial schemes -------------------------------------------
  // Method-of-lines evaluator of a block (L/F/Model frozen): ghosts then R = -div F + S.
  // Construction of the block closures (advance + residual + Poisson) moved to the header
  // (adc/runtime/block_builder.hpp: make_block / make_max_speed / make_poisson_rhs) so that the
  // production template path is instantiable outside this unit (AOT compilation of a
  // generated model). Here we only provide the grid context to pass to them.
  // GridContext: mesh + BC + aux + DISC geometry (project T5-PR3). disc_mask_ / disc_ are
  // STABLE-address MEMBERS -> the block closures (build_block) read them by pointer at each
  // step, so the add_block / set_disc_domain order is irrelevant (the mask is materialized / the
  // radius set before the 1st step; as long as !disc_set_ the stepper does not select the disc advance).
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux, &disc_mask_, &disc_}; }

  // POLAR grid context (ring pgeom_ + r/theta BC + aux) for the polar block closures
  // (block_builder_polar.hpp). Counterpart of grid_ctx(); never called in Cartesian.
  PolarGridContext grid_ctx_polar() { return PolarGridContext{dom, bc_, pgeom_, &aux}; }

  // ensure_elliptic_polar / solve_fields_polar / solve_fields (body) EXTRACTED into fields_
  // (SystemFieldSolver, Batch B). Pure delegation: the Cartesian/polar dispatch, the device_fence and
  // the order of fill_ghosts/fill_boundary now live in the header (bit-identical).
  void solve_fields() { fields_.solve_fields(); }

  // State marshaling DELEGATED to the store (Batch B.3): copy_comp0 / copy_state / write_state carry the
  // device_fence, the layout (component-major) and the size error identically. Kept as
  // helpers of Impl because native_loader and the facade methods call them via P->copy_state /
  // P->write_state / P->copy_comp0 (unchanged access point, zero churn outside this file).
  //
  // MULTI-BOX (theta split of the polar transport, ADC-67). The store marshals via fab(0) -- valid
  // for the unique local box of the Cartesian and of the polar mono-box (local_size() <= 1, including MPI mono-
  // box where a rank without a box returns {}). With theta_boxes > 1, a rank carries SEVERAL local boxes
  // (local_size() > 1): we then rebuild the GLOBAL field (size dom.nx() x dom.ny()) placing
  // each box at its GLOBAL indices, exactly like density_global / state_global (collective gather,
  // all_reduce_sum; mono-rank identity). We DO NOT TOUCH the store (VERBATIM bit-identical extraction):
  // the local_size() <= 1 branch delegates as-is -> Cartesian and polar mono-box UNCHANGED.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    if (mf.local_size() <= 1) return blocks_.copy_comp0(mf);
    device_fence();
    const int gnx = dom.nx(), gny = dom.ny();
    std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
    for (int li = 0; li < mf.local_size(); ++li) {
      const ConstArray4 u = mf.fab(li).const_array();
      const Box2D v = mf.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(u(i, j, 0));
    }
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    if (mf.local_size() <= 1) return blocks_.copy_state(mf, ncomp);
    device_fence();
    const int gnx = dom.nx(), gny = dom.ny();
    std::vector<double> out(static_cast<std::size_t>(ncomp) * gnx * gny, 0.0);
    for (int li = 0; li < mf.local_size(); ++li) {
      const ConstArray4 u = mf.fab(li).const_array();
      const Box2D v = mf.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            out[(static_cast<std::size_t>(c) * gny + j) * gnx + i] = static_cast<double>(u(i, j, c));
    }
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    if (mf.local_size() <= 1) { blocks_.write_state(mf, ncomp, in); return; }
    // Multi-box SCATTER: @p in is the GLOBAL field (component-major (c*gny + j)*gnx + i, same layout
    // as copy_state). Each rank writes ONLY the cells of its local boxes (reading at the
    // global indices) -- no communication. Mono-rank: writes all bands.
    const int gnx = dom.nx(), gny = dom.ny();
    const std::size_t need = static_cast<std::size_t>(ncomp) * gnx * gny;
    if (in.size() != need)
      throw std::runtime_error("System::set_state : size != ncomp*nr*ntheta (multi-box theta)");
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 u = mf.fab(li).array();
      const Box2D v = mf.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            u(i, j, c) = in[(static_cast<std::size_t>(c) * gny + j) * gnx + i];
    }
  }

  // push_dynamic<NV> (DYNAMIC IModel<NV> block loaded from a .so) was EXTRACTED VERBATIM into
  // adc::native_loader::push_dynamic (include/adc/runtime/native_loader.hpp, template over Impl);
  // add_dynamic_block below instantiates it with System::Impl. See SYSTEM_CPP_EXTRACTION_PLAN.md.
};

namespace {
// Geometry guard (polar-grid project). The geometry CHOICE is carried by the config
// (adc.CartesianMesh / adc.PolarMesh). "cartesian": historical path, bit-identical. "polar": global
// ring (r, theta) wired into System.step (Phase 2b): polar transport (assemble_rhs_polar) +
// polar Poisson (PolarPoissonSolver) + aux in local basis (e_r, e_theta). We validate HERE the radial
// bounds of the ring (r_max > r_min >= 0); the Python (PolarMesh) already validates them, but a caller
// that builds the SystemConfig by hand must also be protected. Any other token is an error.
void check_geometry(const SystemConfig& c) {
  if (c.geometry == "cartesian") return;
  if (c.geometry == "polar") {
    if (!(c.r_max > c.r_min && c.r_min >= 0.0))
      throw std::runtime_error(
          "System : geometry='polar' requires a ring r_max > r_min >= 0 (r_min > 0 avoids the "
          "r=0 coordinate singularity) ; cf. adc.PolarMesh");
    // nr >= 3 ENFORCED: the radial derivative of the aux (derive_aux_polar) uses a 2nd-order
    // OFF-CENTERED stencil at both walls (reads phi(i+1),phi(i+2) at r_min and phi(i-1),phi(i-2) at r_max). phi is
    // allocated WITHOUT ghost by the direct solver (its valid box IS its allocation): nr < 3 would read
    // phi out of bounds (UB). We reject it HERE (same fallback computation as Impl::polar_nr: nr or n).
    const int nr = c.nr > 0 ? c.nr : c.n;
    if (nr < 3)
      throw std::runtime_error(
          "System : geometry='polar' requires nr >= 3 (2nd-order off-centered radial stencil at the walls ; "
          "phi without ghost) ; cf. adc.PolarMesh");
    // THETA SPLIT of the transport (theta_boxes, ADC-67). 1 (default) = mono-box, bit-identical. > 1:
    // theta bands -- we require 1 <= theta_boxes <= ntheta (at least one azimuthal cell per band) AND
    // theta_boxes DIVIDES ntheta (EQUAL bands: the per-box split must not depend on the remainder,
    // and the periodic ring stitches back cleanly). PolarMesh already validates on the Python side; a caller that
    // builds the SystemConfig by hand is protected here.
    const int nth = c.ntheta > 0 ? c.ntheta : c.n;
    if (c.theta_boxes < 1)
      throw std::runtime_error("System : geometry='polar' requires theta_boxes >= 1 (cf. adc.PolarMesh)");
    if (c.theta_boxes > nth)
      throw std::runtime_error(
          "System : geometry='polar' requires theta_boxes <= ntheta (at least one azimuthal cell per "
          "band) ; cf. adc.PolarMesh");
    if (nth % c.theta_boxes != 0)
      throw std::runtime_error(
          "System : geometry='polar' requires that theta_boxes DIVIDES ntheta (equal azimuthal bands) ; "
          "cf. adc.PolarMesh");
    return;
  }
  throw std::runtime_error("System : geometry '" + c.geometry +
                           "' unknown (cartesian | polar) ; cf. adc.CartesianMesh / adc.PolarMesh");
}
}  // namespace

System::System(const SystemConfig& c) : p_(std::make_unique<Impl>(c)) { check_geometry(c); }
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

void System::add_block(const std::string& name, const ModelSpec& model,
                       const std::string& limiter, const std::string& riemann,
                       const std::string& recon, const std::string& time, int substeps,
                       bool evolve, int stride, const std::vector<std::string>& implicit_vars,
                       const std::vector<std::string>& implicit_roles,
                       const NewtonOptions& newton, bool newton_diagnostics,
                       double positivity_floor, bool wave_speed_cache) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (stride < 1) throw std::runtime_error("System::add_block : stride >= 1");
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error("System::add_block : positivity_floor >= 0 and finite (0 = inactive)");
  // Validation of the NEWTON OPTIONS (grouped into a POD, ADC-214): same bounds as before, read on the
  // POD fields. fail_policy is already a valid integer (NewtonOptions::kFail* ; the bindings
  // resolve it from the string "none"/"warn"/"throw"). We validate its range to stay defensive.
  if (newton.max_iters < 1)
    throw std::runtime_error("System::add_block : newton_max_iters >= 1");
  if (newton.rel_tol < 0.0 || newton.abs_tol < 0.0 || newton.fd_eps <= 0.0)
    throw std::runtime_error("System::add_block : newton_rel_tol/abs_tol >= 0 and newton_fd_eps > 0");
  if (!(newton.damping > 0.0 && newton.damping <= 1.0))
    throw std::runtime_error("System::add_block : newton_damping in (0, 1]");
  if (newton.fail_policy != NewtonOptions::kFailNone &&
      newton.fail_policy != NewtonOptions::kFailWarn &&
      newton.fail_policy != NewtonOptions::kFailThrow)
    throw std::runtime_error("System::add_block : newton_fail_policy invalid "
                             "(NewtonOptions::kFailNone|kFailWarn|kFailThrow)");
  // @p time carries the TREATMENT and, in explicit, the RK SCHEME: "explicit"/"ssprk2" = SSPRK2
  // (historical default), "ssprk3" = SSPRK3 (order 3), "euler" = ForwardEuler (order 1, fidelity to
  // first-order references -- validation), "imex" = explicit transport + local backward-Euler implicit
  // stiff source (order 1), "imexrk_ars222" = IMEX-RK family scheme ARS(2,2,2)
  // (order 2, distinct PARALLEL advance, Cartesian only). The RK math stays a CORE FUNCTOR
  // (build_block). "imex" and "imexrk_ars222" share the @c imex flag; @c method distinguishes them.
  if (time != "explicit" && time != "ssprk2" && time != "ssprk3" && time != "euler" &&
      time != "imex" && time != "imexrk_ars222")
    throw std::runtime_error(
        "System::add_block : time 'explicit'|'ssprk2'|'ssprk3'|'euler'|'imex'|'imexrk_ars222' (received '" +
        time + "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (received '" +
                             recon + "')");
  const bool imexrk = (time == "imexrk_ars222");
  const bool imex = (time == "imex" || imexrk);  // both go through the implicit source step
  const bool recon_prim = (recon == "primitive");
  // Wave speed cache (opt-in): only engages for the HLL flux and the explicit advance. Requesting it
  // elsewhere would be SILENTLY without effect -> explicit error (no silent ignore). The polar path has
  // its own factory (make_block_polar) without this cache.
  if (wave_speed_cache) {
    if (riemann != "hll")
      throw std::runtime_error("System::add_block : wave_speed_cache requires riemann='hll' (the wave "
                               "speed cache only applies to the HLL flux ; received riemann='" +
                               riemann + "')");
    if (imex)
      throw std::runtime_error("System::add_block : wave_speed_cache not supported with time='" + time +
                               "' (wired on the explicit advance ; use time "
                               "'explicit'/'ssprk2'/'ssprk3'/'euler')");
    if (P->polar_)
      throw std::runtime_error("System::add_block : wave_speed_cache not supported on the polar "
                               "geometry (ring)");
    // DISC transport mode already active: the stepper routes to advance_masked / advance_eb, which do
    // not carry the cache -> requesting it would be WITHOUT EFFECT. Explicit rejection (no silent
    // ignore). The reverse order (set_disc_domain AFTER a cached block) is rejected by set_disc_domain
    // / set_geometry_mode.
    if (P->disc_set_ && P->geometry_mode_ != GeometryMode::None)
      throw std::runtime_error("System::add_block : wave_speed_cache incompatible with an active disc "
                               "transport mode (staircase/cutcell) ; the cache is only wired on the full "
                               "Cartesian advance (remove wave_speed_cache or mode='none')");
    P->ws_cache_block_ = true;  // a block requested the cache -> locks the switch to disc mode
  }
  const std::string method = imexrk ? std::string("imexrk_ars222")
                                     : ((time == "ssprk3") ? std::string("ssprk3")
                                        : (time == "euler") ? std::string("euler")
                                                            : std::string("ssprk2"));
  // The implicit mask (implicit_vars / implicit_roles) applies only to the IMEX source step. Requesting
  // it in explicit is an ERROR (no silent ignore): the explicit has no implicit step.
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error("System::add_block : implicit_vars / implicit_roles require time='imex' "
                             "(the implicit mask applies only to the IMEX source step ; received time='" +
                             time + "')");
  // IMEX-RK ARS(2,2,2): FULLY implicit source (the stage consistency relation assumes a homogeneous
  // solve). A partial mask would be SILENTLY ignored there -> we reject it explicitly. The
  // partial mask stays available on time='imex' (local backward-Euler).
  if (imexrk && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles (partial IMEX mask) unsupported by "
        "time='imexrk_ars222' (its source is FULLY implicit). Use time='imex' for a "
        "partial mask, or remove implicit_vars / implicit_roles.");
  // Same rules for the Newton options/diagnostics: they only drive the IMEX source step.
  // Non-default values in explicit would be SILENTLY ignored -> explicit error.
  const bool newton_non_default = newton.max_iters != 2 || newton.rel_tol != 0.0 ||
                                  newton.abs_tol != 0.0 || newton.fd_eps != 1e-7 ||
                                  newton_diagnostics || newton.damping != 1.0 ||
                                  newton.fail_policy != NewtonOptions::kFailNone;
  if (!imex && newton_non_default)
    throw std::runtime_error("System::add_block : the Newton options (newton_max_iters/rel_tol/"
                             "abs_tol/fd_eps/diagnostics) require time='imex' (received time='" +
                             time + "')");

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  std::function<Real(const MultiFab&)> src_freq, stab_dt;  // optional step bounds (model traits)
  CellConvert prim_to_cons, cons_to_prim;  // pointwise model conversions (set/get_primitive_state)
  VariableSet cons_vs, prim_vs;
  detail::BuiltBlock bb;
  if (P->polar_) {
    // POLAR PATH (ring): closures built by block_builder_polar.hpp (assemble_rhs_polar + scalar polar
    // transport ExBVelocityPolar OR fluid IsothermalFluxPolar + scalar polar Poisson), via the polar
    // seam (python/system_polar.cpp, ADC-335). IMEX is not supported on the ring at this stage: the
    // electrostatic coupling goes through an explicit LOCAL source (non-stiff regime, Path A step 1);
    // we reject it explicitly rather than silently running the transport alone.
    if (imex)
      throw std::runtime_error(
          "System::add_block (polar) : time='" + time + "' (IMEX / IMEX-RK ARS(2,2,2)) unsupported "
          "(ring : coupling by explicit local source, no stiff source to handle implicitly "
          "at this stage). Use 'explicit'/'ssprk2'/'ssprk3'.");
    const PolarGridContext pctx = P->grid_ctx_polar();
    bb = detail::build_block_polar(model, limiter, riemann, pctx, recon_prim, method,
                                   static_cast<Real>(positivity_floor), &P->aux);
  } else {
    const GridContext ctx = P->grid_ctx();
    // Newton options of the IMEX implicit source (defaults = historical constants, bit-identical).
    // The report (OPT-IN diagnostics) lives in Impl::newton_reports_ in a shared_ptr -> STABLE address
    // captured by the closures even when the map reallocates at a later add_block.
    const NewtonOptions& nopts = newton;
    NewtonReport* nreport = nullptr;
    if (newton_diagnostics) {
      auto rep = std::make_shared<NewtonReport>();
      P->newton_reports_[name] = rep;
      nreport = rep.get();
    }
    // Transport-axis seam (ADC-335): each per-transport TU (python/system_<transport>.cpp) runs the
    // SAME source/elliptic dispatch + make_block + makers as before (detail::build_block_for), but
    // instantiates ONLY its own transport's leaves -- so the combinatorial product splits across files
    // for `-j`. This string if/else mirrors detail::dispatch_transport (same unknown-transport message).
    // aux_width is widened host-side AFTER the build (was P->ensure_aux_width inside the visitor;
    // ensure_aux_width keeps the aux ADDRESS captured by the closures, so order vs make_block is
    // immaterial -- byte-identical).
    const detail::BlockBuildArgs args{name, limiter, riemann, ctx, imex, recon_prim, method,
                                      implicit_vars, implicit_roles, nopts, nreport,
                                      static_cast<Real>(positivity_floor), wave_speed_cache};
    if (model.transport == "exb") {
      bb = detail::build_block_exb(model, args);
    } else if (model.transport == "compressible") {
      bb = detail::build_block_compressible(model, args);
    } else if (model.transport == "isothermal") {
      bb = detail::build_block_isothermal(model, args);
    } else {
      throw std::runtime_error("unknown transport '" + model.transport +
                               "' (exb|compressible|isothermal)");
    }
    P->ensure_aux_width(bb.aux_width);
  }
  ncomp = bb.ncomp;
  cons_vs = std::move(bb.cons_vs);
  prim_vs = std::move(bb.prim_vs);
  clo = std::move(bb.clo);
  max_speed = std::move(bb.max_speed);
  add_poisson_rhs = std::move(bb.add_poisson_rhs);
  src_freq = std::move(bb.src_freq);
  stab_dt = std::move(bb.stab_dt);
  prim_to_cons = std::move(bb.prim_to_cons);
  cons_to_prim = std::move(bb.cons_to_prim);
  // Common installation (same path as add_compiled_model for a DSL-generated model):
  // the closures run on the REAL System MultiFabs (MPI halos via fill_boundary, device
  // via Kokkos), without copy.
  install_block(name, ncomp, cons_vs, prim_vs, model.gamma, std::move(clo), std::move(max_speed),
                std::move(add_poisson_rhs), substeps, evolve, stride);
  set_block_conversion(name, std::move(prim_to_cons), std::move(cons_to_prim));
  set_block_dt_bounds(name, std::move(src_freq), std::move(stab_dt));
  // SCHEME GHOSTS: WENO5 reads a 5-point stencil (3 ghosts) > the 2 allocated by default in
  // install_block. We reallocate the block state with block_n_ghost(limiter) if needed (cf. AmrSystem which
  // allocates with Limiter::n_ghost, PR #22) so that fill_ghosts + assemble_rhs do not read out of
  // bounds. minmod/vanleer (2 ghosts): no-op, allocation and result bit-identical to before.
  P->set_block_ghosts(name, block_n_ghost(limiter));
}

// Real grid context (mesh + BC + aux): used by the add_compiled_model template to build
// the closures of an AOT-compiled model on the real System fields (native parity, without marshaling).
ADC_EXPORT GridContext System::grid_context() { return p_->grid_ctx(); }

// Installs a block from already-built closures (by dispatch_model on the add_block side, or by
// block_builder on the add_compiled_model side). Centralizes the creation of the species (U, names, scheme).
ADC_EXPORT void System::install_block(const std::string& name, int ncomp,
                                      const VariableSet& cons_vars,
                                      const VariableSet& prim_vars, double gamma,
                                      BlockClosures closures,
                                      std::function<Real(const MultiFab&)> max_speed,
                                      std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                                      int substeps, bool evolve, int stride) {
  if (stride < 1) throw std::runtime_error("System::install_block : stride >= 1");
  Impl* P = p_.get();
  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps, evolve,
                                stride, gamma, std::move(closures.advance),
                                std::move(closures.rhs_into), std::move(max_speed),
                                std::move(poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
  P->sp.back().cons_vars = cons_vars;
  P->sp.back().prim_vars = prim_vars;
  // DISC transport advances (project T5-PR3): empty unless build_block built them (Cartesian
  // block with disc_mask_/disc_ provided). Empty -> the stepper falls back on advance (bit-identical).
  P->sp.back().advance_masked = std::move(closures.advance_masked);
  P->sp.back().advance_eb = std::move(closures.advance_eb);
  P->sp.back().hotspot = std::move(closures.hotspot);  // dt_hotspot diagnostic (ADC-182)
  // Projection ponctuelle post-pas (ADC-177) : vide sauf si le modele declare le trait
  // HasPointwiseProjection (make_block). Vide -> le stepper ne l'interroge pas (bit-identique).
  P->sp.back().project = std::move(closures.project);
}

// Width-aware reallocation of a block state (delegates to Impl::set_block_ghosts). Exposed
// (ADC_EXPORT) so that the add_compiled_model header template (native path, .so loader) can
// widen the compiled block to block_n_ghost(limiter) -- 3 for weno5 -- as add_block does.
ADC_EXPORT void System::set_block_ghosts(const std::string& name, int n_ghost) {
  p_->set_block_ghosts(name, n_ghost);
}

// OPTIONAL step bounds of a block (model traits): set after install_block, read by
// step_cfl / step_adaptive. Empty functions = the block imposes no bound (historical).
void System::set_block_dt_bounds(const std::string& name,
                                 std::function<Real(const MultiFab&)> source_frequency,
                                 std::function<Real(const MultiFab&)> stability_dt) {
  Impl::Species& s = p_->find(name);  // raises if unknown block
  s.source_frequency = std::move(source_frequency);
  s.stability_dt = std::move(stability_dt);
}

// GLOBAL step bound (host, one evaluation per step): multi-block coupling, Schur/Poisson,
// scheduler, user policy. cf. SystemStepper::step_cfl for the aggregation.
void System::add_dt_bound(const std::string& label, std::function<double()> fn) {
  if (!fn) throw std::runtime_error("System::add_dt_bound : empty bound function");
  p_->dt_bounds_.push_back(Impl::GlobalDtBound{label, std::move(fn)});
}

// ACTIVE bound of the last step_cfl (step-policy diagnostic). "" before the first step.
std::string System::last_dt_bound() const { return p_->stepper_.last_dt_reason(); }

// dt_hotspot diagnostic (ADC-182): the GLOBAL cell (i, j) that dominates the transport CFL
// bound of block @p name, and its speed w = max(wx, wy). ON DEMAND (two reduction
// passes, cf. max_wave_speed_hotspot_mf) -- step/step_cfl do not touch it. Block without
// closure (historical non-rewireable paths, e.g. dynamic) -> EXPLICIT error.
std::array<double, 3> System::dt_hotspot(const std::string& name) {
  Impl::Species& s = p_->find(name);
  if (!s.hotspot)
    throw std::runtime_error("System::dt_hotspot : block '" + name +
                             "' without hotspot diagnostic (non-rewireable add path)");
  Real w = 0;
  int i = -1, j = -1;
  s.hotspot(s.U, w, i, j);
  return {static_cast<double>(w), static_cast<double>(i), static_cast<double>(j)};
}

// Newton report (OPT-IN IMEX diagnostics) of the block: flat copy of the NewtonReport aggregated by the
// LAST advance of the block (reset at the start of the advance by AdvanceImex*). Clear error if the block did
// not enable newton_diagnostics (no silently empty report).
System::SourceNewtonReport System::newton_report(const std::string& name) const {
  p_->index(name);  // raises if unknown block
  const auto it = p_->newton_reports_.find(name);
  if (it == p_->newton_reports_.end())
    throw std::runtime_error(
        "System::newton_report : Newton diagnostics not enabled for block '" + name +
        "' ; add the block with newton_diagnostics=true (adc.IMEX(newton_diagnostics=True) / "
        "adc.SourceImplicit(newton_diagnostics=True))");
  const NewtonReport& r = *it->second;
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp};
}

// Body EXTRACTED VERBATIM into adc::native_loader::add_dynamic_block (native_loader.hpp); instantiated
// here with System::Impl (defined above, private to this TU). Bit-identical: pure delegation.
void System::add_dynamic_block(const std::string& name, const std::string& so_path, int substeps,
                               const std::vector<std::string>& names, const std::string& recon) {
  native_loader::add_dynamic_block(this, p_.get(), name, so_path, substeps, names, recon);
}

// Body EXTRACTED VERBATIM into adc::native_loader::add_compiled_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation.
void System::add_compiled_block(const std::string& name, const std::string& so_path,
                                const std::string& limiter, const std::string& riemann,
                                const std::string& recon, const std::string& time, int substeps,
                                const std::vector<std::string>& names, double positivity_floor) {
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_compiled_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_compiled_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                    substeps, names, positivity_floor);
}

// P7-b: overwrites the SHARED vector of runtime parameter values of block @p name. add_compiled_block
// registered this vector in p_->block_params_ AND captured it in the block closures: writing
// into it suffices to change the behavior at the next step, WITHOUT recompiling the .so. Explicit error if
// the block has no runtime params (vector absent) or if values does not have the right size.
void System::set_block_params(const std::string& name, const std::vector<double>& values) {
  // index() raises "System: unknown block '...'" if the block does not exist (same diagnostic as everywhere).
  (void)p_->blocks_.index(name);
  auto it = p_->block_params_.find(name);
  if (it == p_->block_params_.end())
    throw std::runtime_error(
        "System::set_block_params : block '" + name +
        "' has no runtime parameter (declare dsl.Param(..., kind='runtime') and wire via "
        "backend='aot' / add_compiled_block ; const params are frozen at compile time)");
  std::vector<double>& pv = *it->second;
  if (values.size() != pv.size())
    throw std::runtime_error(
        "System::set_block_params : block '" + name + "' expects " + std::to_string(pv.size()) +
        " runtime parameters, received " + std::to_string(values.size()));
  pv = values;  // the vector is SHARED with the closures (shared_ptr): effect at the next step
}

// Body EXTRACTED VERBATIM into adc::native_loader::add_native_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation (this marshals to the unchanged native loader).
void System::add_native_block(const std::string& name, const std::string& so_path,
                              const std::string& limiter, const std::string& riemann,
                              const std::string& recon, const std::string& time, double gamma,
                              int substeps, bool evolve, int stride, double positivity_floor) {
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_native_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_native_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                  gamma, substeps, evolve, stride, positivity_floor);
}

void System::set_poisson(const std::string& rhs, const std::string& solver,
                         const std::string& bc, const std::string& wall, double wall_radius,
                         double epsilon, double abs_tol) {
  if (epsilon == 0.0) throw std::runtime_error("System::set_poisson : epsilon != 0 required");
  if (abs_tol < 0.0) throw std::runtime_error("System::set_poisson : abs_tol >= 0 required");
  p_->fields_.p_rhs = rhs;
  p_->fields_.p_solver = solver;
  p_->fields_.p_bc = bc;
  p_->fields_.p_wall = wall;
  p_->fields_.p_wall_radius = wall_radius;
  p_->fields_.p_eps_ = static_cast<Real>(epsilon);
  p_->fields_.p_abs_tol_ = static_cast<Real>(abs_tol);  // absolute floor of the V-cycle (0 = relative only)
  p_->fields_.ell_.reset();
}

namespace {
// Translates the Python disc transport mode ("none"|"staircase"|"cutcell") into a GeometryMode. EXPLICIT
// error on an unknown mode (never a silent fallback). Single source of the name table.
GeometryMode parse_geometry_mode(const std::string& mode, const char* err_context) {
  if (mode == "none") return GeometryMode::None;
  if (mode == "staircase") return GeometryMode::Staircase;
  if (mode == "cutcell") return GeometryMode::CutCell;
  throw std::runtime_error(std::string(err_context) + " : unknown geometry mode '" + mode +
                           "' (none|staircase|cutcell)");
}
}  // namespace

void System::set_disc_domain(double cx, double cy, double R, const std::string& mode) {
  Impl* P = p_.get();
  // CARTESIAN only: polar already bounds the ring by its radial walls (r_min / r_max,
  // zero radial flux) -> a Cartesian disc mask makes no sense on the (r, theta) grid.
  if (P->polar_)
    throw std::runtime_error(
        "System::set_disc_domain : polar geometry (the ring is already bounded by its radial "
        "walls r_min/r_max ; the Cartesian disc mask does not apply)");
  if (!(R > 0.0))
    throw std::runtime_error("System::set_disc_domain : radius R > 0 required");
  // Validate the mode BEFORE any mutation (an unknown mode must not leave the disc half-set).
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_disc_domain");
  // wave_speed_cache (ADC-199) is only wired on the full Cartesian advance: a disc mode
  // (staircase/cutcell) borrows advance_masked / advance_eb which ignore the cache -> explicit rejection.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error("System::set_disc_domain : mode '" + mode +
                             "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
                             "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
                             "or use mode='none')");
  P->disc_ = detail::DiscDomain{cx, cy, R};
  P->disc_set_ = true;
  // Materializes the 0/1 cell-centered mask (1 ghost, so the mask-aware transport reads the
  // i-1/i+1/j-1/j+1 neighbors up to the edge). Same layout as the blocks (ba/dm). Cell active when
  // its CENTER is inside the disc (level set < 0, SAME convention as the conducting wall).
  P->disc_mask_ = MultiFab(P->ba, P->dm, 1, 1);
  const detail::DiscDomain disc = P->disc_;
  const Geometry geom = P->geom;
  for (int li = 0; li < P->disc_mask_.local_size(); ++li) {
    Array4 m = P->disc_mask_.fab(li).array();
    // box WITH ghosts: we also classify the ghosts (the mask-aware transport reads the edge neighbors).
    const Box2D g = P->disc_mask_.fab(li).grown_box();
    for_each_cell(g, [=] ADC_HD(int i, int j) {
      m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
    });
  }
  // TRANSPORT ROUTING (project T5-PR3). mode == "none": the mask is materialized (queryable
  // via disc_mask()) but the transport stays FULL Cartesian -> bit-identical. mode != "none": the
  // stepper routes the advance to assemble_rhs_masked (staircase) / assemble_rhs_eb (cutcell).
  P->geometry_mode_ = gmode;
}

void System::set_geometry_mode(const std::string& mode) {
  Impl* P = p_.get();
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_geometry_mode");
  // A disc mode (staircase/cutcell) only makes sense with a fixed disc: otherwise the stepper would fall
  // back on the full transport (the mask / level set does not exist), a silent footgun -> we reject.
  if (gmode != GeometryMode::None && !P->disc_set_)
    throw std::runtime_error(
        "System::set_geometry_mode : mode '" + mode +
        "' requested without a fixed disc ; call set_disc_domain(cx, cy, R) first");
  // wave_speed_cache (ADC-199) is not carried by the disc advances -> explicit rejection (cf.
  // set_disc_domain) rather than a cache silently ignored in staircase/cutcell mode.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error("System::set_geometry_mode : mode '" + mode +
                             "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
                             "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
                             "or use mode='none')");
  P->geometry_mode_ = gmode;
}

std::vector<double> System::disc_mask() const {
  Impl* P = p_.get();
  device_fence();
  const Box2D v = P->dom;
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  if (!P->disc_set_) {
    // CONTRACT: without a fixed disc, the transport subdomain is the whole domain -> all active.
    out.assign(static_cast<std::size_t>(v.nx()) * v.ny(), 1.0);
    return out;
  }
  const ConstArray4 m = P->disc_mask_.fab(0).const_array();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(static_cast<double>(m(i, j, 0)));
  return out;
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : size != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivity eps(x) > 0 required");
  p_->fields_.p_eps_field_ = eps;
  p_->fields_.has_eps_field_ = true;
  p_->fields_.ell_.reset();  // the operator will be rebuilt with the eps field at the next solve_fields
}

void System::set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                           const std::vector<double>& eps_y) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps_x.size()) != n * n || static_cast<int>(eps_y.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_anisotropic_field : size != n*n (eps_x and eps_y)");
  for (double e : eps_x)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_anisotropic_field : permittivity eps_x(x) > 0 required");
  for (double e : eps_y)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_anisotropic_field : permittivity eps_y(x) > 0 required");
  p_->fields_.p_eps_x_field_ = eps_x;
  p_->fields_.p_eps_y_field_ = eps_y;
  p_->fields_.has_eps_xy_field_ = true;
  p_->fields_.ell_.reset();  // operator rebuilt as div(diag(eps_x, eps_y) grad phi) at the next solve_fields
}

void System::set_reaction_field(const std::vector<double>& kappa) {
  const int n = p_->cfg.n;
  if (static_cast<int>(kappa.size()) != n * n)
    throw std::runtime_error("System::set_reaction_field : size != n*n");
  for (double k : kappa)
    if (!(k >= 0.0))
      throw std::runtime_error("System::set_reaction_field : reaction term kappa(x) >= 0 required "
                               "(well-posed elliptic operator and convergent multigrid)");
  p_->fields_.p_kappa_field_ = kappa;
  p_->fields_.has_kappa_field_ = true;
  p_->fields_.ell_.reset();  // operator rebuilt with - kappa phi at the next solve_fields
}

ADC_EXPORT void System::ensure_aux_width(int ncomp) { p_->ensure_aux_width(ncomp); }

void System::set_magnetic_field(const std::vector<double>& bz) {
  // Expected size of the B_z(x) field row-major (slow axis = 2nd box index, fast axis = 1st):
  //   Cartesian = n * n (square, BIT-IDENTICAL); POLAR = nr * ntheta (ring, i = r fast, cf.
  //   apply_bz / polar set_density). The layout is the SAME as set_density (flat[j * nr + i]).
  if (p_->polar_) {
    const int nr = Impl::polar_nr(p_->cfg), nth = Impl::polar_ntheta(p_->cfg);
    if (static_cast<int>(bz.size()) != nr * nth)
      throw std::runtime_error("System::set_magnetic_field : size != nr*ntheta (polar)");
  } else {
    const int n = p_->cfg.n;
    if (static_cast<int>(bz.size()) != n * n)
      throw std::runtime_error("System::set_magnetic_field : size != n*n");
  }
  p_->fields_.bz_field_.assign(bz.begin(), bz.end());
  p_->fields_.apply_bz();  // apply right away if a block already reads B_z; otherwise keep for ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  const int idx = p_->index(name);  // raises if unknown block
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error("System::set_electron_temperature_from : block '" + name +
                             "' must be compressible (4 vars : rho, rho u, rho v, E) for T = p/rho");
  p_->fields_.te_src_ = idx;
  // T_e (canonical comp 4) DERIVED: recomputed at each solve_fields. Inert as long as no block
  // reads T_e (n_aux=5 -> ensure_aux_width(5)), like set_magnetic_field for B_z.
  p_->fields_.apply_te();
}

// Expected size of a cell-defined field (Cartesian n*n / polar nr*ntheta). Member of Impl:
// a free caller could not name the private type System::Impl.
std::size_t System::Impl::aux_field_cell_count() const {
  if (polar_) {
    const int nr = polar_nr(cfg), nth = polar_ntheta(cfg);
    return static_cast<std::size_t>(nr) * nth;
  }
  return static_cast<std::size_t>(cfg.n) * cfg.n;
}

void System::set_aux_field_component(int comp, const std::vector<double>& field) {
  Impl* P = p_.get();
  // RESERVED components (phi/grad/B_z/T_e): a named aux field starts at kAuxNamedBase (= 5).
  // B_z and T_e keep their dedicated paths -> redirecting message (the Python facade already intercepts
  // the canonical names, this guard covers a direct C++ call).
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::set_aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase) + " (B_z -> set_magnetic_field, T_e -> "
        "set_electron_temperature_from)");
  const std::size_t expect = P->aux_field_cell_count();
  if (field.size() != expect)
    throw std::runtime_error("System::set_aux_field : size " + std::to_string(field.size()) +
                             " != " + std::to_string(expect) + " (grid cells)");
  // The aux channel must be wide enough: a block declaring this field (n_aux = kAuxNamedBase + k + 1) has
  // already called ensure_aux_width at its add time. Otherwise the field would be read by no model -> error.
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::set_aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp) +
        " (add the block that reads it before set_aux_field)");
  std::vector<Real> f(field.begin(), field.end());
  p_->fields_.apply_named_aux_one(comp, f);  // populate right away (channel wide enough)
  p_->fields_.named_aux_[comp] = std::move(f);  // keep for a later reallocation of the channel
}

std::vector<double> System::aux_field_component(int comp) const {
  Impl* P = p_.get();
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; read phi via potential(), a named aux field starts "
        "at index " + std::to_string(kAuxNamedBase));
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp));
  device_fence();
  // Rank without a box (MPI mono-box): EMPTY return (cf. potential / copy_comp0). The Python facade is
  // mono-rank; the multi-rank global field would be a dedicated collective accessor (follow-up).
  if (P->aux.local_size() == 0) return {};
  const ConstArray4 a = P->aux.fab(0).const_array();
  const Box2D v = P->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(static_cast<double>(a(i, j, comp)));
  return out;
}

void System::add_ionization(const std::string& electron, const std::string& ion,
                            const std::string& neutral, double rate) {
  Impl* P = p_.get();
  const int ie = P->index(electron), ii = P->index(ion), ig = P->index(neutral);
  const Real k = static_cast<Real>(rate);
  // Density resolved by ROLE (like add_collision / add_thermal_exchange), fallback comp 0 if the
  // block does not provide its roles (dynamic / compiled block). A block storing its density elsewhere
  // than index 0 stays correctly coupled.
  const int de = role_index(P->sp[ie].cons_vars, VariableRole::Density, 0);
  const int di = role_index(P->sp[ii].cons_vars, VariableRole::Density, 0);
  const int dg = role_index(P->sp[ig].cons_vars, VariableRole::Density, 0);
  // Ionization (operator-split, on the density): rate r = k n_e n_g. One neutral disappears, one ion and
  // one electron appear: n_g -= dt r, n_i += dt r, n_e += dt r. Mass is transferred from the
  // neutral to the ion (n_i + n_g conserved). First coupling brick; the momentum
  // / energy transfer (fluid species) is a later refinement.
  P->couplings.push_back([P, ie, ii, ig, k, de, di, dg](Real dt) {
    // MPI / multi-box-safe: iteration over the LOCAL fabs (local_size()==0 on a rank without a box ->
    // no-op), SAME pattern as add_coupled_source -- no more hard-coded fab(0)/box(0), which existed only
    // on the rank owning box 0 and would become wrong if System went multi-box. The blocks
    // share the System DistributionMapping -> same local_size(), co-located fabs.
    MultiFab& Ue = P->sp[ie].U;
    for (int li = 0; li < Ue.local_size(); ++li) {
      Array4 ue = Ue.fab(li).array();
      Array4 ui = P->sp[ii].U.fab(li).array();
      Array4 ug = P->sp[ig].U.fab(li).array();
      for_each_cell(Ue.box(li), [=] ADC_HD(int i, int j) {  // on device (reads n_e, n_g)
        const Real dn = dt * k * ue(i, j, de) * ug(i, j, dg);
        ug(i, j, dg) -= dn;
        ui(i, j, di) += dn;
        ue(i, j, de) += dn;
      });
    }
  });
}

void System::add_collision(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp < 3 || P->sp[ib].ncomp < 3)
    throw std::runtime_error("System::add_collision : both blocks must carry a momentum "
                             "(fluid transport >= 3 variables)");
  const Real k = static_cast<Real>(rate);
  // Components resolved by ROLE (momentum x/y, density) rather than by literal index: a
  // block that stores its variables differently stays correctly coupled. Fallback to the historical
  // indices (1, 2, 0) if the block does not provide its roles (dynamic / compiled block).
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Inter-species friction (operator-split): force F = k (u_a - u_b) on the momentum,
  // opposite on each species (total momentum conserved); the velocities relax
  // toward each other. Frictional heating (energy) is a later refinement
  // (neglected: fits isothermal species, without an energy eq.).
  P->couplings.push_back([P, ia, ib, k, mxa, mya, da, mxb, myb, db](Real dt) {
    // MPI / multi-box-safe: LOCAL fabs, same pattern as add_coupled_source (cf. add_ionization).
    MultiFab& Ua = P->sp[ia].U;
    for (int li = 0; li < Ua.local_size(); ++li) {
      Array4 ua = Ua.fab(li).array();
      Array4 ub = P->sp[ib].U.fab(li).array();
      for_each_cell(Ua.box(li), [=] ADC_HD(int i, int j) {  // on device
        const Real fx = dt * k * (ua(i, j, mxa) / ua(i, j, da) - ub(i, j, mxb) / ub(i, j, db));
        ua(i, j, mxa) -= fx; ub(i, j, mxb) += fx;
        const Real fy = dt * k * (ua(i, j, mya) / ua(i, j, da) - ub(i, j, myb) / ub(i, j, db));
        ua(i, j, mya) -= fy; ub(i, j, myb) += fy;
      });
    }
  });
}

void System::add_thermal_exchange(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp != 4 || P->sp[ib].ncomp != 4)
    throw std::runtime_error("System::add_thermal_exchange : both blocks must carry an "
                             "energy (compressible Euler, 4 variables)");
  const Real k = static_cast<Real>(rate);
  const Real ga = static_cast<Real>(P->sp[ia].gamma), gb = static_cast<Real>(P->sp[ib].gamma);
  // Components resolved by ROLE (energy, momentum x/y, density) rather than by literal index.
  // Fallback to the historical indices (3, 1, 2, 0) if the block does not provide its roles.
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int ea = role_index(va_set, VariableRole::Energy, 3);
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int eb = role_index(vb_set, VariableRole::Energy, 3);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Thermal exchange (operator-split): heat flux q = k (T_a - T_b) on the energy, opposite
  // on each species (total energy conserved); the temperatures relax. T = p/rho (up to a
  // constant), p = (gamma-1)(E - 1/2 rho |u|^2). Transfers the INTERNAL energy (u unchanged).
  P->couplings.push_back([P, ia, ib, k, ga, gb, ea, mxa, mya, da, eb, mxb, myb, db](Real dt) {
    // MPI / multi-box-safe: LOCAL fabs, same pattern as add_coupled_source (cf. add_ionization).
    MultiFab& Ua = P->sp[ia].U;
    for (int li = 0; li < Ua.local_size(); ++li) {
      Array4 ua = Ua.fab(li).array();
      Array4 ub = P->sp[ib].U.fab(li).array();
      for_each_cell(Ua.box(li), [=] ADC_HD(int i, int j) {  // on device
        const Real ra = ua(i, j, da), rb = ub(i, j, db);
        const Real pa = (ga - Real(1)) * (ua(i, j, ea) -
            Real(0.5) * (ua(i, j, mxa) * ua(i, j, mxa) + ua(i, j, mya) * ua(i, j, mya)) / ra);
        const Real pb = (gb - Real(1)) * (ub(i, j, eb) -
            Real(0.5) * (ub(i, j, mxb) * ub(i, j, mxb) + ub(i, j, myb) * ub(i, j, myb)) / rb);
        const Real q = dt * k * (pa / ra - pb / rb);  // k (T_a - T_b), T = p/rho
        ua(i, j, ea) -= q;
        ub(i, j, eb) += q;
      });
    }
  });
}

void System::add_coupled_source(const CoupledSourceProgram& prog_desc, double frequency,
                                const std::string& label) {
  // Bytecode description grouped into a POD (ADC-214): local aliases to keep the body readable (the
  // names and the semantics are strictly those of the old flat parameters).
  const std::vector<std::string>& in_blocks = prog_desc.in_blocks;
  const std::vector<std::string>& in_roles = prog_desc.in_roles;
  const std::vector<double>& consts = prog_desc.consts;
  const std::vector<std::string>& out_blocks = prog_desc.out_blocks;
  const std::vector<std::string>& out_roles = prog_desc.out_roles;
  const std::vector<int>& prog_ops = prog_desc.prog_ops;
  const std::vector<int>& prog_args = prog_desc.prog_args;
  const std::vector<int>& prog_lens = prog_desc.prog_lens;
  const std::vector<int>& freq_prog_ops = prog_desc.freq_prog_ops;
  const std::vector<int>& freq_prog_args = prog_desc.freq_prog_args;
  Impl* P = p_.get();
  const int n_in = static_cast<int>(in_blocks.size());
  const int n_const = static_cast<int>(consts.size());
  const int n_terms = static_cast<int>(out_blocks.size());
  // --- shape validation (before any step, EXPLICIT errors) ------------------------------------
  if (n_terms == 0)
    throw std::runtime_error("System::add_coupled_source : no source term (out_blocks empty)");
  if (static_cast<int>(in_roles.size()) != n_in)
    throw std::runtime_error("System::add_coupled_source : in_blocks / in_roles of different sizes");
  if (static_cast<int>(out_roles.size()) != n_terms || static_cast<int>(prog_lens.size()) != n_terms)
    throw std::runtime_error("System::add_coupled_source : out_blocks / out_roles / prog_lens of different "
                             "sizes");
  if (prog_ops.size() != prog_args.size())
    throw std::runtime_error("System::add_coupled_source : prog_ops / prog_args of different sizes");
  if (n_in + n_const > kCsMaxReg)
    throw std::runtime_error("System::add_coupled_source : too many registers (inputs + constants > " +
                             std::to_string(kCsMaxReg) + ")");
  if (n_terms > kCsMaxTerms)
    throw std::runtime_error("System::add_coupled_source : too many source terms (> " +
                             std::to_string(kCsMaxTerms) + ")");
  // Resolves role -> component via the CONSERVATIVE descriptor of the block (like add_collision); fallback
  // comp 0 if the block does not provide the role. An unknown block raises via P->index().
  auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
    const int sidx = P->index(block);  // raises if unknown block
    const VariableRole r = role_from_name(role);
    if (r == VariableRole::Custom)
      throw std::runtime_error("System::add_coupled_source : role '" + role + "' unknown (block '" +
                               block + "')");
    // STRICT (no silent fallback): a DSL coupled source targets a (block, role) EXPLICITLY
    // requested by the user. If the block does NOT expose this role, it is an error: a fallback on
    // component 0 would apply the source to the wrong field SILENTLY (the false-positive identified at
    // review). We raise. Distinct from the NAMED couplings (add_collision/add_pair) which deliberately assume
    // the canonical layout via role_index(..., fallback) and stay unchanged.
    const int comp = P->sp[static_cast<std::size_t>(sidx)].cons_vars.index_of(r);
    if (comp < 0)
      throw std::runtime_error("System::add_coupled_source : block '" + block +
                               "' does not expose role '" + role +
                               "' (no silent fallback on component 0)");
    return {sidx, comp};
  };
  // Inputs: (species, component) read per cell. Captured by INDEX (the fabs may be
  // reallocated between registration and application: we rebuild the Array4 at EACH step).
  struct InRef { int sidx, comp; };
  std::vector<InRef> ins(static_cast<std::size_t>(n_in));
  for (int c = 0; c < n_in; ++c) {
    auto [s, comp] = resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
    ins[static_cast<std::size_t>(c)] = {s, comp};
  }
  struct OutRef { int sidx, comp; CsProgram prog; };
  std::vector<OutRef> outs(static_cast<std::size_t>(n_terms));
  int off = 0;
  for (int t = 0; t < n_terms; ++t) {
    auto [s, comp] = resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
    const int len = prog_lens[static_cast<std::size_t>(t)];
    if (len < 0 || len > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : program of term " + std::to_string(t) +
                               " too long (> " + std::to_string(kCsMaxProg) + ")");
    if (off + len > static_cast<int>(prog_ops.size()))
      throw std::runtime_error("System::add_coupled_source : prog_lens inconsistent with prog_ops");
    CsProgram pg;
    pg.len = len;
    for (int k = 0; k < len; ++k) {
      const int opc = prog_ops[static_cast<std::size_t>(off + k)];
      const int a = prog_args[static_cast<std::size_t>(off + k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error("System::add_coupled_source : register out of bounds in the program");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    outs[static_cast<std::size_t>(t)] = {s, comp, pg};
    off += len;
  }
  // All touched species (inputs + outputs) share the System DistributionMapping (one box
  // round-robin distributed), so same local_size() and same local indexing -> we would iterate in parallel
  // over the local fabs. Conversion to CAPTURED values (no reference to the C++ lambda's 'this').
  std::vector<Real> kconsts(consts.begin(), consts.end());
  // Optional PER-CELL frequency (CoupledSource.frequency with an Expr, refinement of the
  // CONSTANT frequency): a bytecode program mu(U) on the SAME register table as the terms
  // (inputs then constants). Validates HERE its SHAPE (opcodes / bounded registers) BEFORE any push -- the
  // bound must be registered only after a complete validation (anti-phantom-bound rule). Empty
  // (default) -> no per-cell frequency (historical path).
  const bool has_freq_expr = !freq_prog_ops.empty() || !freq_prog_args.empty();
  CsProgram freq_pg;
  if (has_freq_expr) {
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error("System::add_coupled_source : freq_prog_ops / freq_prog_args of different "
                               "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : frequency program too long (> " +
                               std::to_string(kCsMaxProg) + ")");
    freq_pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < freq_pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error("System::add_coupled_source : register out of bounds in the frequency");
      freq_pg.op[k] = opc;
      freq_pg.arg[k] = a;
    }
  }
  // CONSTANT declared frequency of the coupling (audit wave 3): registered for the step bound of
  // step_cfl / step_adaptive (dt <= cfl/mu on the MACRO-step). <= 0 = no bound (historical). Pushed
  // AFTER all the validation (source AND frequency have raised if invalid): a rejected coupling must
  // leave NO phantom bound -- otherwise a script that try/excepts the failure would keep a throttled step without
  // matching physics.
  if (frequency > 0.0) P->coupled_freqs_.push_back(Impl::CoupledFreq{label, frequency});
  // PER-CELL frequency: same rule (push after complete validation). The inputs REUSE the
  // resolve() resolution (ins); the constants are the same as the source (kconsts). The program
  // mu(U) is reduced (MAX) at each step in step_cfl / step_adaptive.
  if (has_freq_expr) {
    Impl::CoupledFreqExpr ce;
    ce.label = label;
    ce.prog = freq_pg;
    ce.n_in = n_in;
    ce.ins.resize(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c)
      ce.ins[static_cast<std::size_t>(c)] = {ins[static_cast<std::size_t>(c)].sidx,
                                             ins[static_cast<std::size_t>(c)].comp};
    ce.kconsts = kconsts;
    P->coupled_freq_exprs_.push_back(std::move(ce));
  }
  P->couplings.push_back(
      [P, ins, outs, kconsts, n_in, n_const, n_terms](Real dt) {
        // MPI-safe: iteration over the LOCAL fabs of the first input block (or output if no
        // input). local_size()==0 on a rank without a box -> empty loop, no-op (no hard-coded fab(0)).
        const int sref = n_in > 0 ? ins[0].sidx : outs[0].sidx;
        MultiFab& Uref = P->sp[static_cast<std::size_t>(sref)].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = n_in;
          kern.n_const = n_const;
          kern.n_terms = n_terms;
          for (int c = 0; c < n_in; ++c) {
            kern.in[c] = P->sp[static_cast<std::size_t>(ins[static_cast<std::size_t>(c)].sidx)].U.fab(li).array();
            kern.in_comp[c] = ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < n_const; ++c) kern.consts[c] = kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < n_terms; ++t) {
            kern.out[t] = P->sp[static_cast<std::size_t>(outs[static_cast<std::size_t>(t)].sidx)].U.fab(li).array();
            kern.out_comp[t] = outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li), kern);  // NAMED functor (device-clean), additive forward-Euler
        }
      });
}

void System::set_source_stage(const std::string& name, const std::string& kind, double theta,
                              double alpha, const SourceStageOptions& opts) {
  // Settings grouped into a POD (ADC-214): local aliases to keep the body readable (the names and the
  // semantics are strictly those of the old flat parameters).
  const double krylov_tol = opts.krylov_tol;
  const int krylov_max_iters = opts.krylov_max_iters;
  const std::string& density = opts.density;
  const std::string& momentum_x = opts.momentum_x;
  const std::string& momentum_y = opts.momentum_y;
  const std::string& energy = opts.energy;
  const int bz_aux_component = opts.bz_aux_component;
  Impl* P = p_.get();
  Impl::Species& s = P->find(name);  // raises if unknown block
  // ONLY kind wired for now: ElectrostaticLorentzCondensation (cf. CondensedSchurSourceStepper).
  // Other kinds may be added without touching the facade (explicit rejection, no silent ignore).
  if (kind != "electrostatic_lorentz")
    throw std::runtime_error("System::set_source_stage : kind '" + kind +
                             "' unknown (only 'electrostatic_lorentz' is supported)");
  if (!(theta > 0.0 && theta <= 1.0))
    throw std::runtime_error("System::set_source_stage : theta must be in (0, 1] (received " +
                             std::to_string(theta) + ")");
  // Tolerance / budget of the stage Krylov solve (audit 2026-06: the constants 1e-10 / 400 (cart)
  // / 600 (polar) are no longer frozen). krylov_tol <= 0 / krylov_max_iters <= 0 = "historical
  // stepper default" (we do not touch the setting of the constructed stepper).
  if (krylov_tol > 0.0 && !(krylov_tol < 1.0))
    throw std::runtime_error("System::set_source_stage : krylov_tol must be in (0, 1)");
  // GEOMETRY: the condensed source stage is wired in CARTESIAN (CondensedSchurSourceStepper, #126) AND in
  // POLAR (PolarCondensedSchurSourceStepper, #212, Path A step 2c). The dispatch below builds the
  // stepper adapted to the System geometry. Any other geometry is REJECTED explicitly (no
  // silent ignore).
  const bool polar = (P->cfg.geometry == "polar");
  if (P->cfg.geometry != "cartesian" && !polar)
    throw std::runtime_error("System::set_source_stage : condensed source stage supports the "
                             "cartesian and polar geometries (received '" + P->cfg.geometry + "')");
  // The POLAR condensed source stage is now MULTI-RANK MPI (PolarTensorKrylovSolver / polar
  // Schur distributed by AZIMUTHAL split; check_radial_columns layout guard in the
  // solver). On the FACADE side, the System builds for now ONE box covering the ring (P->ba mono-box),
  // so under MPI the box lives on rank 0 and the other ranks have local_size()==0: the solve stays CORRECT
  // (collective dot/project_mean called on all ranks, zero contributions from the empty ranks) and
  // BIT-IDENTICAL to the mono-rank, but without real parallelism at this level. The effective theta split
  // (true multi-rank scaling) takes place at the C++ API level (PolarCondensedSchurSourceStepper with a
  // BoxArray split in theta); the facade-side theta distribution is deferred (Extend). No mono-rank
  // guard here: the PolarTensorKrylovSolver raises a clear error if the layout ever cuts r.
  // ROLE CONTRACT: the block must expose Density / MomentumX / MomentumY (Energy optional). We read the
  // CONSERVATIVE descriptor of the block (populated by add_block / the .so with roles, including the compiled DSL which
  // declares the electrons with roles). A required role absent raises an EXPLICIT error HERE (before the step)
  // -- the stepper constructor would raise it too, but we diagnose on the named-block side.
  const VariableSet& vs = s.cons_vars;
  // DESCRIPTOR RESOLUTION (audit wave 2: roles/transported fields in the ABI). An
  // EMPTY descriptor = canonical role (historical, bit-identical). Otherwise: stable ROLE name
  // first (role_from_name), then block VARIABLE name. Failure = explicit error with remedy.
  auto resolve_field = [&](const std::string& spec, VariableRole canonical,
                           const char* label) -> int {
    if (spec.empty()) {
      const int idx = vs.index_of(canonical);
      if (idx < 0)
        throw std::runtime_error(
            "System::set_source_stage : block '" + name + "' does not expose the role " + label +
            " required by adc.CondensedSchur (the model must declare Density / MomentumX / "
            "MomentumY ; Energy optional), and no explicit descriptor is provided (pass "
            "density=/momentum=... with a role name or a block variable name).");
      return idx;
    }
    const VariableRole r = role_from_name(spec);
    if (r != VariableRole::Custom) {
      const int idx = vs.index_of(r);
      if (idx < 0)
        throw std::runtime_error("System::set_source_stage : block '" + name +
                                 "' does not expose role '" + spec + "' (" + label + ")");
      return idx;
    }
    for (std::size_t i = 0; i < vs.names.size(); ++i)
      if (vs.names[i] == spec) return static_cast<int>(i);
    throw std::runtime_error("System::set_source_stage : '" + spec +
                             "' is neither a stable role nor a variable of block '" + name +
                             "' (" + label + ")");
  };
  const int c_rho = resolve_field(density, VariableRole::Density, "Density");
  const int c_mx = resolve_field(momentum_x, VariableRole::MomentumX, "MomentumX");
  const int c_my = resolve_field(momentum_y, VariableRole::MomentumY, "MomentumY");
  const int c_E = (energy == "none")
                      ? -1
                      : (energy.empty() ? vs.index_of(VariableRole::Energy)
                                        : resolve_field(energy, VariableRole::Energy, "Energy"));
  // B_z MANDATORY: the Lorentz stage reads Omega = B_z. We require set_magnetic_field called
  // (bz_field_ provided) and we widen the aux channel to the B_z channel (kAuxBaseComps) so that apply_bz
  // populates it and solve_fields fills its ghosts. An absent B_z raises an EXPLICIT error.
  if (P->fields_.bz_field_.empty())
    throw std::runtime_error(
        "System::set_source_stage : block '" + name + "' has no B_z field (aux Omega) ; "
        "adc.CondensedSchur requires set_magnetic_field(B_z) (the Lorentz term reads Omega = B_z).");
  // Aux channel of the magnetic field: canonical (kAuxBaseComps) by default, redirectable by
  // bz_aux_component (transported descriptor). NOTE: apply_bz populates the CANONICAL channel; a
  // different component assumes the caller populates it itself (derived/custom aux field).
  const int c_bz = bz_aux_component >= 0 ? bz_aux_component : kAuxBaseComps;
  P->ensure_aux_width(c_bz + 1);  // guarantees the channel in the shared aux + re-applies B_z
  // Builds the condensed source stage on the REAL System layout (ba/dm/geom) with the Poisson BC.
  // The stepper allocates its buffers ONCE; step() reuses them (cf. its lifecycle). alpha =
  // electrostatic coupling constant of the source subsystem.
  if (polar) {
    // POLAR (Path A step 2c): PolarCondensedSchurSourceStepper on the ring pgeom_, SAME Poisson BC
    // (radial Dirichlet/Neumann, theta always periodic on the solver side). RadialLine preconditioner
    // (default). run_source_stage invokes it exactly like the Cartesian (identical step() signature).
    // schur stays nullptr (Cartesian path untouched). EXPLICIT components resolved above
    // (empty descriptors -> canonical roles -> bit-identical): the POLAR stepper accepts the
    // overrides since wave 3 (ctor with explicit components, Cartesian parity).
    s.schur_polar = std::make_shared<PolarCondensedSchurSourceStepper>(
        vs, c_rho, c_mx, c_my, c_E, P->pgeom_, P->ba, P->fields_.poisson_bc(),
        static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur_polar->set_krylov(krylov_tol > 0.0 ? static_cast<Real>(krylov_tol) : Real(1e-10),
                                krylov_max_iters > 0 ? krylov_max_iters : 600);
  } else {
    // CARTESIAN (#126): EXPLICIT components resolved above (empty descriptors -> canonical
    // roles -> same indices as the historical, bit-identical).
    s.schur = std::make_shared<CondensedSchurSourceStepper>(vs, c_rho, c_mx, c_my, c_E, P->geom,
                                                            P->ba, P->fields_.poisson_bc(),
                                                            static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur->set_krylov(krylov_tol > 0.0 ? static_cast<Real>(krylov_tol) : Real(1e-10),
                          krylov_max_iters > 0 ? krylov_max_iters : 400);
  }
  s.schur_bz_comp = c_bz;
  s.schur_theta = theta;
}

void System::set_time_scheme(const std::string& scheme) {
  // Routes the splitting policy of the system stepper (default Lie = bit-identical). The Strang
  // scheme reuses the SAME bricks (s.advance for the transport half-advances, run_source_stage
  // for the full source stage); it RE-SOLVES solve_fields between the stages (cf. SystemStepper::step_strang
  // and docs/HOFFART_STEP_SEQUENCE.md). An unknown scheme raises an EXPLICIT error (no silent ignore).
  if (scheme == "lie") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Lie);
  } else if (scheme == "strang") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Strang);
  } else {
    throw std::runtime_error("System::set_time_scheme : scheme '" + scheme +
                             "' unknown (expected 'lie' or 'strang')");
  }
}

void System::set_gauss_policy(const std::string& policy) {
  // Gauss's law policy (project R0, Hoffart reproduction). "restart" (default): solve_fields
  // re-solves -Delta phi = f at each step (bit-identical to the historical). "evolve": after the first
  // solve (phi^0), solve_fields NO LONGER re-solves the Poisson; it derives the aux from the CURRENT phi that
  // the Schur source stage evolves in-place in ell_phi() -> -Delta phi evolution without restart of the
  // paper (the Gauss constraint is imposed only at t=0). Has effect ONLY with a condensed source stage
  // (without it phi would stay frozen after t=0). The gauss_solved_once_ lock is reset to zero here so
  // that a policy change BEFORE the first solve stays consistent (the 1st solve always solves).
  if (policy == "restart") {
    p_->fields_.gauss_evolve_ = false;
  } else if (policy == "evolve") {
    p_->fields_.gauss_evolve_ = true;
  } else {
    throw std::runtime_error("System::set_gauss_policy : policy '" + policy +
                             "' unknown (expected 'restart' or 'evolve')");
  }
  p_->fields_.gauss_solved_once_ = false;
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const Real gm1 = Real(s.gamma) - Real(1);
  // Local helper: sets density + rest state on ONE cell (same formulas as the historical).
  auto set_cell = [&](Array4& u, int i, int j, Real r) {
    u(i, j, 0) = r;
    if (s.ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }  // momentum at rest
    if (s.ncomp == 4) u(i, j, 3) = r / gm1;                // E = p/(g-1), p = rho
  };
  // MULTI-BOX (theta_boxes > 1, polar): @p rho is the GLOBAL field (nr x ntheta, layout flat[j*gnx+i]
  // identical to the mono-box below). We write each local box at its GLOBAL indices. local_size() <= 1
  // (Cartesian / polar mono-box, including MPI mono-box): historical path UNCHANGED, bit-identical.
  if (s.U.local_size() > 1) {
    const int gnx = p_->dom.nx(), gny = p_->dom.ny();
    if (static_cast<int>(rho.size()) != gnx * gny)
      throw std::runtime_error("System::set_density : size != nr*ntheta (multi-box theta)");
    for (int li = 0; li < s.U.local_size(); ++li) {
      Array4 u = s.U.fab(li).array();
      const Box2D b = s.U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          set_cell(u, i, j, rho[static_cast<std::size_t>(j) * gnx + i]);
    }
    return;
  }
  // Row-major layout of the input array: (ni x nj) = extents of the state box. In Cartesian
  // ni = nj = cfg.n (indexing and size bit-identical to before). In polar ni = nr, nj = ntheta:
  // we index by the real extents of the box (and not n*n), so nr != ntheta is correctly handled.
  const Box2D v = s.U.box(0);
  const int ni = v.nx(), nj = v.ny();
  if (static_cast<int>(rho.size()) != ni * nj)
    throw std::runtime_error("System::set_density : size != nr*ntheta (or n*n in Cartesian)");
  Array4 u = s.U.fab(0).array();
  // LAYOUT CONVENTION (unchanged vs the historical): slow axis = 2nd box index (j), fast axis =
  // 1st (i), i.e. flat[(j-lo) * ni + (i-lo)]. In Cartesian ni = n, lo = 0 -> flat[j*n+i] (bit-identical
  // to before). In polar the array is thus (nr, ntheta) radial-line-by-line: j = theta (slow
  // axis), i = r (fast axis), SAME order as density()/copy_comp0 -> consistent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      set_cell(u, i, j, rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])]);
}

ADC_EXPORT void System::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                             CellConvert cons_to_prim) {
  Impl::Species& s = p_->find(name);
  s.prim_to_cons = std::move(prim_to_cons);
  s.cons_to_prim = std::move(cons_to_prim);
}

void System::set_primitive_state(const std::string& name, const std::vector<double>& prim) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn = static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error("System::set_primitive_state : size != ncomp*nr*ntheta (n*n Cartesian) (block '" + name +
                             "' has " + std::to_string(nc) + " variables)");
  if (!s.prim_to_cons)
    throw std::runtime_error("System::set_primitive_state : the model of block '" + name +
                             "' does not expose a primitive -> conservative conversion (.so generated before "
                             "this project ?) ; use set_state (direct conservative state)");
  // CELL-BY-CELL conversion via the block model: we read the nc primitives component-major
  // (prim[c*nn + k]) into a small contiguous buffer, convert, and write the conservatives at the
  // same place in an output buffer. Then write_state pushes everything to the MultiFab (set_state
  // path, identical marshaling). Reuses therefore the existing marshaling (copy/write_state).
  std::vector<double> cons(prim.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c) cell_in[c] = prim[static_cast<std::size_t>(c) * nn + k];
    s.prim_to_cons(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c) cons[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  p_->write_state(s.U, nc, cons);
}

std::vector<double> System::get_primitive_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn = static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (!s.cons_to_prim)
    throw std::runtime_error("System::get_primitive_state : the model of block '" + name +
                             "' does not expose a conservative -> primitive conversion (.so generated before "
                             "this project ?) ; use get_state (direct conservative state)");
  const std::vector<double> cons = p_->copy_state(s.U, nc);  // get_state path (same marshaling)
  std::vector<double> prim(cons.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c) cell_in[c] = cons[static_cast<std::size_t>(c) * nn + k];
    s.cons_to_prim(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c) prim[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  return prim;
}

void System::solve_fields() { p_->solve_fields(); }

// Time advance EXTRACTED into stepper_ (SystemStepper, Batch B). Pure delegation: the Cartesian/polar
// dispatch of the physical step h, the per-block CFL formula (substeps/stride), the
// hold-then-catch-up semantics of the macro-step counter, the condensed source stage and the couplings live
// now in the header (bit-identical). The public API stays unchanged.
void System::step(double dt) { p_->stepper_.step(dt); }
void System::advance(double dt, int nsteps) { p_->stepper_.advance(dt, nsteps); }
double System::step_cfl(double cfl) { return p_->stepper_.step_cfl(cfl); }
double System::step_adaptive(double cfl) { return p_->stepper_.step_adaptive(cfl); }

// System clock (IO v1, audit wave 2): macro_step is REQUIRED by the restart (the
// hold-then-catch-up stride cadence reads macro_step % stride; t alone is not enough).
int System::macro_step() const { return p_->macro_step_; }

// Potential phi restoration (IO v1, restart): writes the VALID cells of component 0 of the
// solver phi (multigrid warm start; physical state in gauss_policy="evolve"). Mono-box
// (same marshaling convention as potential / set_density).
void System::set_potential(const std::vector<double>& phi) {
  Impl* P = p_.get();
  device_fence();
  if (P->polar_) {
    P->fields_.ensure_elliptic_polar();
    MultiFab& ph = P->fields_.pell_->phi();
    // Rank without a box (MPI mono-box): NO-OP (the owning rank restores phi). Allows restart on
    // all ranks with the GLOBAL field. Mono-rank: local_size()==1, UNCHANGED.
    if (ph.local_size() == 0) return;
    const Box2D v = ph.box(0);
    if (static_cast<int>(phi.size()) != v.nx() * v.ny())
      throw std::runtime_error("System::set_potential : size != nr*ntheta");
    Array4 a = ph.fab(0).array();
    std::size_t k = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) a(i, j, 0) = phi[k++];
    return;
  }
  P->fields_.ensure_elliptic();
  MultiFab& ph = P->fields_.ell_phi();
  if (ph.local_size() == 0) return;  // rank without a box: no-op (cf. polar branch)
  const Box2D v = ph.box(0);
  if (static_cast<int>(phi.size()) != v.nx() * v.ny())
    throw std::runtime_error("System::set_potential : size != n*n");
  Array4 a = ph.fab(0).array();
  std::size_t k = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) a(i, j, 0) = phi[k++];
}
void System::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("System::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
  return p_->copy_state(R, s.ncomp);
}
std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const { return p_->find(name).ncomp; }
std::vector<std::string> System::variable_names(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative") return s.cons_vars.names;
  if (kind == "primitive") return s.prim_vars.names;
  throw std::runtime_error("System::variable_names : kind 'conservative' | 'primitive' (received '" +
                           kind + "')");
}
std::vector<std::string> System::variable_roles(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative") vs = &s.cons_vars;
  else if (kind == "primitive") vs = &s.prim_vars;
  else throw std::runtime_error("System::variable_roles : kind 'conservative' | 'primitive' (received '" +
                                kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i) out.push_back(role_name(vs->at(i).role));  // 'custom' if absent
  return out;
}
double System::block_gamma(const std::string& name) const { return p_->find(name).gamma; }

int System::nx() const { return p_->cfg.n; }
// SLOW axis of the field (rows of the (ny, nx) array). We read it from the INDEX domain (dom = nx() x ny()),
// SINGLE SOURCE of the extents for both geometries: Cartesian dom = n x n -> ny() == nx() == n (square,
// UNCHANGED); polar dom = nr x ntheta -> nx() == nr (fast, i), ny() == ntheta (slow, j). It is this
// dimension that sizes the numpy array on the bindings side: a polar field has nx()*ny() = nr*ntheta
// values, and with nr != ntheta the square reshape (nx, nx) overflows the buffer (teardown bug).
int System::ny() const { return p_->dom.ny(); }
double System::time() const { return p_->t; }
int System::n_species() const { return p_->blocks_.size(); }
std::vector<std::string> System::block_names() const {
  // SINGLE block registry (store), populated by all add paths: a block loaded via
  // add_dynamic_block / add_compiled_block (.so) appears there just like an add_block.
  return p_->blocks_.names();
}
double System::mass(const std::string& name) const {
  const Impl::Species& s = p_->find(name);
  if (!p_->polar_) return sum(s.U, 0);  // Cartesian: bare sum of the cells (bit-identical)
  // POLAR: FV mass = Sum_ij n_ij r_i dr dtheta (annular cell volume r dr dtheta). This is the
  // quantity CONSERVED by assemble_rhs_polar (cf. test_polar_transport_mms). Host loop over the valid
  // cells (mono-rank: a single local fab), reduced over the ranks by symmetry (n_ranks==1).
  device_fence();
  const PolarGeometry& g = p_->pgeom_;
  const Real dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += static_cast<double>(u(i, j, 0)) * static_cast<double>(g.r_cell(i) * dr * dth);
  }
  return all_reduce_sum(m);
}
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  device_fence();
  // POLAR: phi comes from the polar Poisson (pell_), not from the Cartesian solver (ell_). We build it
  // lazily if needed (a call before any step) and we read phi() of PolarPoissonSolver.
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    // Rank without a box (MPI mono-box): EMPTY return (no fab(0)). Cf. copy_comp0; the multi-rank
    // global field goes through System::potential_global.
    if (p_->aux.local_size() == 0) return {};
    const ConstArray4 ph = p_->fields_.pell_->phi().fab(0).const_array();
    const Box2D v = p_->aux.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
    return out;
  }
  p_->fields_.ensure_elliptic();
  if (p_->aux.local_size() == 0) return {};  // rank without a box: empty (cf. potential_global)
  const ConstArray4 ph = p_->fields_.ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

// --- GLOBAL accessors (collective MPI-safe), IO v1 multi-rank --------------------------------
// Common pattern (cf. header system.hpp): GLOBAL buffer of size gny*gnx (or nc*gny*gnx) initialized
// to 0, filled by the LOCAL fabs in GLOBAL INDICES (the box carries its global indices; a rank without a
// box -> local_size()==0 -> no write), then all_reduce_sum_inplace: each cell being
// owned by EXACTLY one rank (disjoint boxes), the sum = the EXACT global field on each
// rank. Mono-rank: the box covers the whole domain and all_reduce = identity -> array bit-identical
// to the non-global accessors (density / get_state / potential). IDENTICAL layout (density: j*gnx + i;
// state: (c*gny + j)*gnx + i, component-major; cf. copy_comp0 / copy_state).
std::vector<double> System::density_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  const int gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(u(i, j, 0));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}
std::vector<double> System::state_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  const int nc = s.ncomp, gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(nc) * gnx * gny, 0.0);
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int c = 0; c < nc; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[(static_cast<std::size_t>(c) * gny + j) * gnx + i] = static_cast<double>(u(i, j, c));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}
std::vector<double> System::potential_global() {
  device_fence();
  const int gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
  // Solves the Poisson (polar or Cartesian) if needed: COLLECTIVE, like potential_global as a whole.
  const MultiFab* phi = nullptr;
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    phi = &p_->fields_.pell_->phi();
  } else {
    p_->fields_.ensure_elliptic();
    phi = &p_->fields_.ell_phi();
  }
  for (int li = 0; li < phi->local_size(); ++li) {
    const ConstArray4 ph = phi->fab(li).const_array();
    const Box2D v = phi->box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(ph(i, j));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// --- LOCAL per-fab accessors (NON collective): parallel HDF5 write by hyperslabs (PR-IO-3) --
// Local counterpart of the _global accessors: they aggregate nothing (no MPI comm), they expose per rank
// the LOCAL boxes (in GLOBAL indices, as carried by the fab box) and the state of each fab.
// The facade sim.write(format='hdf5', parallel=True) creates the global datasets then each rank writes
// ITS boxes in hyperslabs. A rank without a box -> local_size()==0 -> empty list (never a hard-coded fab(0)).
std::vector<std::array<int, 4>> System::local_boxes(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  std::vector<std::array<int, 4>> out;
  out.reserve(s.U.local_size());
  for (int li = 0; li < s.U.local_size(); ++li) {
    const Box2D v = s.U.box(li);
    out.push_back({v.lo[0], v.lo[1], v.hi[0], v.hi[1]});  // (ilo, jlo, ihi, jhi) GLOBAL
  }
  return out;
}
std::vector<double> System::local_state(const std::string& name, int li) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  if (li < 0 || li >= s.U.local_size())
    throw std::out_of_range("System::local_state : local fab index out of bounds (0.." +
                            std::to_string(s.U.local_size() - 1) + ")");
  const int nc = s.ncomp;
  const ConstArray4 u = s.U.fab(li).const_array();
  const Box2D v = s.U.box(li);
  const int bnx = v.nx(), bny = v.ny();  // dimensions of the LOCAL box (valid cells)
  std::vector<double> out(static_cast<std::size_t>(nc) * bnx * bny, 0.0);
  // Layout = state_global mapped to the local box: (c*bny + jl)*bnx + il, component-major, so
  // reshapeable into (nc, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1].
  for (int c = 0; c < nc; ++c)
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[(static_cast<std::size_t>(c) * bny + (j - v.lo[1])) * bnx + (i - v.lo[0])] =
            static_cast<double>(u(i, j, c));
  return out;
}

}  // namespace adc
