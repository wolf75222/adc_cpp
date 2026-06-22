#pragma once

#include <adc/amr/regridding/regrid.hpp>   // tag_cells, grow_tags (per-block tags + phi for the union regrid)
#include <adc/amr/tagging/tag_box.hpp>  // TagBox, tag_union (cell-by-cell OR of the tags of all blocks)
#include <adc/core/state/state.hpp>   // kAuxBaseComps
#include <adc/core/state/variables.hpp>  // VariableSet, VariableRole, role_from_name (role -> component of coupled sources)
#include <adc/coupling/amr/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb (aux injection coarse->fine)
#include <adc/coupling/amr/amr_regrid_coupler.hpp>  // regrid_compute_fine_layout + regrid_field_on_layout (split bricks)
#include <adc/coupling/static_system/amr_system_coupler.hpp>  // detail::same_layout_or_throw (shared-layout guard)
#include <adc/coupling/base/aux_fill.hpp>            // detail::derive_aux_bc (BC of the aux channel)
#include <adc/coupling/source/coupled_source_program.hpp>  // CoupledSourceKernel + CsProgram (flat ABI, P5 bytecode)
#include <adc/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, mf_average_down_mb
#include <adc/numerics/time/implicit_stepper.hpp>  // NewtonReport (OPT-IN IMEX diagnostics, aggregated per block)
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/patch_box.hpp>  // PatchBox: index-space signature of a fine patch (patch_boxes())
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/boundary/fill_boundary.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>

#include <algorithm>  // std::max (substeps/stride-aware CFL step)
#include <cmath>      // std::isfinite (reject a degenerate dt)
#include <cstddef>
#include <functional>
#include <limits>  // std::numeric_limits (initial dt = +inf, min over the blocks)
#include <map>  // named_aux_: model-named aux fields (comp -> coarse field), re-applied each solve
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief AMR multi-block engine at RUNTIME (type-erased registry keyed by name).
///
/// Runtime counterpart of System::Impl (python/system.cpp): where System type-erases the species
/// (struct Species) on a SINGLE-LEVEL grid, AmrRuntime type-erases N blocks on a SHARED AMR
/// hierarchy. It FAITHFULLY reproduces the AmrSystemCoupler::solve_fields / step algorithm
/// (include/adc/coupling/amr_system_coupler.hpp), but over type-erased closures (the runtime facade
/// does not know the blocks' Model/Limiter/Flux types at compile time) rather than over a
/// compile-time CoupledSystem<Blocks...>.
///
/// INVARIANTS (multi-block capstone, docs/AMR_MULTIBLOCK_DESIGN.md):
///  - ONE single shared AMR hierarchy (AmrHierarchyLayout, same_layout_or_throw guard): all
///    blocks live on EXACTLY the same BoxArray + DistributionMapping + dx/dy per level;
///  - ALL blocks live on ALL patches (never a local spatial absence of a block);
///  - SYSTEM Poisson with a SUMMED and CO-LOCATED right-hand side: rhs[coarse] = Sum_b
///    elliptic_rhs_b(U_b) read at the SAME cells of the shared coarse;
///  - aux SHARED per level (phi, grad phi); a single coarse Poisson solve then coarse->fine
///    injection (coupler_inject_aux_mb), exactly like AmrSystemCoupler;
///  - PER-BLOCK conservation (reflux + average_down of the AMR engine, in the advance closure).
///
/// SCOPE (capstone). We carry blocks with potentially DIFFERENT spatial schemes over the FROZEN
/// hierarchy (no regrid: AmrSystemCoupler has none), with per-block MULTIRATE: substeps (explicit
/// substeps) and stride (hold-then-catch-up cadence), honored in step() mirroring
/// AmrSystemCoupler::step (#140). The TEMPORAL TREATMENT is PER BLOCK: explicit (forward-Euler
/// source, carried by the AMR step) OR IMEX (stiff source treated IMPLICITLY by
/// backward_euler_source, transport staying explicit; capstone vii), selected in step().
///
/// IMEX SEMANTICS UNDER substeps (integration decision, follow-up review #184). At substeps=1 AND
/// stride=1 the runtime IMEX branch COINCIDES with the IMEX branch of the compile-time engine
/// AmrSystemCoupler::step (a SOURCE-FREE transport + a backward_euler_source over the effective step).
/// FOR substeps>1 the two paths DIVERGE DELIBERATELY:
///   - the COMPILE-TIME engine IGNORES substeps on the IMEX branch: it does ONE single source-free
///     transport then ONE single implicit_advance over the whole effective step bdt (cf.
///     amr_system_coupler.hpp: the substep loop exists only in the Explicit branch);
///   - the RUNTIME SUB-CYCLES the IMEX splitting: it applies imex_advance K=substeps times, each over
///     bdt/K, i.e. K Lie steps [transport(dt/K); implicit source(dt/K)].
/// This choice is INTENTIONAL and SOUND (it is NOT a bug): (a) the source-free explicit transport
/// becomes SAFER in CFL (each substep carries dt/K, so a wave speed K times larger stays
/// admissible); (b) backward-Euler is UNCONDITIONALLY STABLE whatever the step, so sub-cycling never
/// destabilizes the source; (c) refining the backward-Euler step BRINGS the stiff relaxation CLOSER
/// to its continuous trajectory (splitting error and implicit temporal error both O(dt), both
/// reduced). The runtime thus does NOT mirror the compile-time bit-for-bit once substeps>1; it
/// honors substeps CONSISTENTLY with the explicit branch (same split into K equal substeps), which is
/// the behavior expected by a user setting substeps. Non-regression guard:
/// test_amr_multiblock_imex compares a substeps=4 trajectory to substeps=1 and requires them to
/// DIFFER (the sub-cycling is intentional, not accidental).
///
/// The union-tags regrid and the compiled multi-block production DSL remain LATER PRs. The runtime
/// facade (AmrSystem) explicitly REFUSES multi-block + regrid_every > 0 as long as the union regrid
/// does not exist.

namespace adc {

/// Type-erased closures of ONE AMR block, placed on the shared hierarchy. AMR counterpart of the
/// Species struct of System::Impl: a name + its level stack (on the shared layout) + its closures
/// (advance / elliptic-rhs / max_speed / mass / density). The closures capture the CONCRETE
/// Model/Limiter/Flux of the block (resolved at build): the kernel stays COMPILED, only the block
/// list is type-erased. Produced by detail::build_amr_block (amr_dsl_block.hpp).
struct AmrRuntimeBlock {
  std::string name;
  int ncomp = 1;
  double gamma = 1.4;
  /// EXPLICIT substeps of the block within ITS effective macro-step: the effective step (stride * dt)
  /// is split into substeps equal pieces and each piece is advanced by ONE advance_amr (cf.
  /// AmrRuntime::step). substeps=1 => a single advance_amr over the whole effective step (bit-identical).
  int substeps = 1;
  /// HOLD-THEN-CATCH-UP cadence of the block (multirate). stride=1 (default): the block advances at
  /// EVERY macro-step (bit-identical). stride=M>1: the block is HELD at macro-steps 0..M-2 (not
  /// advanced) then CATCHES UP at macro-step M-1, where (macro_step+1)%M==0, by an effective step
  /// M*dt. Same semantics as block_stride_v / AmrSystemCoupler::step (#140). The INVARIANT of the
  /// end-of-window catch-up: at macro-step k the system time is (k+1)*dt and the block that catches
  /// up has then accumulated (k+1)*dt, so it stays temporally CONSISTENT with the fast blocks (never
  /// "in the future"), which keeps the Poisson coupling (summed RHS) meaningful: a held block
  /// contributes with its FROZEN state (its last advance), not with an anticipated state that would
  /// falsify q_b n_b in the sum.
  int stride = 1;
  /// Width of the aux channel READ by the block model (aux_comps<Model>(); >= kAuxBaseComps). The aux
  /// channel SHARED per level is sized to the MAX of this width over all blocks, so that a block
  /// reading an extra field (B_z, T_e; n_aux > 3) never reads out of bounds.
  int aux_ncomp = kAuxBaseComps;

  /// Descriptor of the model CONSERVATIVE variables (names + physical ROLES, Model::conservative_vars()).
  /// Single source of truth to resolve a role (Density, MomentumX, ...) -> component index in
  /// add_coupled_source, like System::add_coupled_source reads Species::cons_vars. The resolution is
  /// STRICT (#181): if the block does NOT expose the requested canonical role (index_of < 0),
  /// add_coupled_source THROWS instead of falling back to component 0 (a silent fallback would apply
  /// the source to the wrong field).
  VariableSet cons_vars;

  /// Level stack of the block (level 0 = coarse, > 0 = fine patches), ON the shared layout. The aux
  /// pointer of each AmrLevelMP is (re)wired by AmrRuntime to the SHARED aux of the level. shared_ptr:
  /// AmrRuntimeBlock stays MOVABLE (a std::vector<AmrLevelMP> is heavy to move into a std::function,
  /// and the engine ctor needs a stable address for the closures).
  std::shared_ptr<std::vector<AmrLevelMP>> levels;

  /// Advances the block by ONE substep of size dt: AMR transport (Berger-Oliger + conservative reflux
  /// + average_down) over the block level stack, with ITS spatial scheme (Limiter, Flux). Captures
  /// advance_amr<Limiter, Flux> on the concrete Model. The substep loop and the stride cadence are
  /// carried by AmrRuntime::step (runtime counterpart of AmrSystemCoupler::step): the closure does ONE
  /// advance_amr, the engine calls it substeps times (dt = effective step/substeps). The signature
  /// passes the base domain + periodicity + coarse ownership policy, rewired by the engine.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> advance;

  /// TEMPORAL TREATMENT of the block: false (default) = EXPLICIT (forward-Euler source, in advance);
  /// true = IMEX (stiff source treated IMPLICITLY by backward_euler_source). The facade (AmrSystem)
  /// freezes it from time="imex". Selected EXPLICITLY in AmrRuntime::step (runtime counterpart of the
  /// constexpr block_time_treatment_v dispatch of AmrSystemCoupler::step): an explicit block goes
  /// through advance, an IMEX block through imex_advance. false everywhere -> bit-identical trajectory
  /// to the historical one.
  bool imex = false;

  /// IMEX advance of the block by ONE substep of size dt: (1) EXPLICIT TRANSPORT on the SOURCE-FREE
  /// model (-div F only, SourceFreeModel<Model>) by the AMR engine (Berger-Oliger + conservative
  /// reflux + average_down), then (2) IMPLICIT STIFF SOURCE backward_euler_source AT EACH LEVEL (local
  /// Newton, finite-difference jacobian; implicit mask CARRIED BY THE BLOCK for partial IMEX),
  /// followed by a fine -> coarse cascade (mf_average_down_mb). ONE call = ONE Lie step [transport;
  /// implicit source] over dt. The SEMANTICS of this splitting (source-free transport then
  /// backward-Euler) mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel +
  /// AmrImplicitSourceStepper); at substeps=1 it is IDENTICAL to it. But step() calls THIS closure
  /// substeps times (over dt = effective step / substeps), so for substeps>1 the runtime SUB-CYCLES the
  /// IMEX splitting where the compile-time applies it once over the whole effective step: DIVERGENCE
  /// INTENTIONAL (cf. IMEX SEMANTICS UNDER substeps, file header). Captures the CONCRETE
  /// Model/Limiter/Flux + the mask (build_amr_block); the kernel stays COMPILED, only the block
  /// registry is type-erased. CONSERVATION INVARIANT (LOCAL source): the source is cell-local (outside
  /// face fluxes), so OUTSIDE the reflux registers -> conservation at coarse-fine interfaces stays
  /// intact; a COVERED coarse cell becomes again the 2x2 average of its children through the final
  /// cascade (otherwise the mass diagnostic, sum of the coarse only, would count a phantom source).
  /// Empty for an explicit block (imex == false): step() never calls it.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> imex_advance;

  /// POINTWISE PROJECTION post-pas (ADC-177) : U <- project(U, aux) appliquee PAR NIVEAU a la FIN
  /// de l'avance complete du bloc (substeps + reflux/cascade faits). Vide -> aucune projection
  /// (modele sans HasPointwiseProjection : trajectoire bit-identique). Locale par niveau (aucun
  /// collectif MPI). Cf. detail::apply_pointwise_project_amr, cable par build_amr_block.
  std::function<void(std::vector<AmrLevelMP>&)> project_per_level;

  /// NEWTON DIAGNOSTICS (OPT-IN, wave 3: AMR counterpart of System::newton_report). false (default) ->
  /// imex_advance passes report=nullptr to backward_euler_source: FAST bit-identical path, no extra
  /// allocation or reduction. true -> imex_advance passes @c newton_report.get() (STABLE address since
  /// shared_ptr) to the backward_euler_source of EACH level; the report is AGGREGATED (max residual,
  /// max iterations, sum of failed cells, MPI all_reduce) over all levels AND all substeps of a
  /// macro-step. AmrRuntime::step RESETS the report at the head of the block advance (parity with
  /// System::AdvanceImex which resets at the head of operator()). MULTI-BLOCK native only (the
  /// single-block coupler and the .so loaders reject it at build / at the facade). STABLE address
  /// (shared_ptr): captured by the imex_advance closure AND read by AmrRuntime::newton_report.
  bool newton_diagnostics = false;
  std::shared_ptr<NewtonReport> newton_report;

  /// Contribution of the block to the Poisson right-hand side: rhs += elliptic_rhs_b(U_b) on the
  /// coarse. CO-LOCATED: the loop reads U_b and writes rhs AT THE SAME cells (same shared coarse
  /// BoxArray). The SUM of the contributions of all blocks forms the system Poisson RHS.
  std::function<void(const MultiFab&, MultiFab&)> add_elliptic_rhs;

  /// Speed driving the block CFL on the coarse. By default max_wave_speed (historical); when the
  /// model declares the HasStabilitySpeed trait, it is lambda* (stability_speed) that the closure
  /// reduces -- SAME policy as System (make_max_speed), cf. build_amr_block.
  std::function<Real(const MultiFab&, const MultiFab&)> max_speed;

  /// OPTIONAL STEP BOUNDS of the block (AMR StabilityPolicy, audit 2026-06): evaluated on the COARSE
  /// (level 0, where the AMR CFL lives -- cf. step_cfl: h = dx_coarse). EMPTY (default) -> step_cfl
  /// keeps the transport bound only, bit-identical. Filled by build_amr_block / build_amr_compiled when
  /// the model declares HasSourceFrequency / HasStabilityDt (same semantics as System: mu in 1/s ->
  /// dt <= cfl*substeps/(stride*mu), without h; direct admissible step -> dt <=
  /// dt_adm*substeps/stride, without cfl).
  std::function<Real(const MultiFab&, const MultiFab&)> source_frequency;
  std::function<Real(const MultiFab&, const MultiFab&)> stability_dt;

  /// Mass of component 0 of the block coarse (sum u*dV; cross-rank reduced if distributed).
  std::function<Real()> mass;

  /// Coarse density (component 0) of the block as a global n*n row-major field (diagnostic).
  std::function<std::vector<double>()> density;

  /// Coarse potential read from the shared aux (component 0) as an n*n row-major field (diagnostic).
  /// Identical for all blocks (shared aux); carried per block for API symmetry.
  std::function<std::vector<double>(const MultiFab&)> potential;
};

/// AMR multi-block engine at runtime. Owns the SHARED aux per level, the coarse Poisson
/// (GeometricMG), the geometry + BC, and the type-erased block REGISTRY. Reproduces the
/// AmrSystemCoupler algorithm (solve_fields + step) over closures rather than a CoupledSystem.
class AmrRuntime {
 public:
  /// @param geom        geometry of the coarse level (domain + physical extents).
  /// @param ba_coarse   BoxArray of the coarse (the coarse Poisson lives on it).
  /// @param bcPhi       BC of the coarse Poisson.
  /// @param blocks      block registry (>= 1), all on the SAME layout (guarded at the ctor).
  /// @param base_per    periodicity of the base domain (transport).
  /// @param replicated_coarse  ownership of level 0 (replicated single-box, or distributed multi-box).
  /// @param active      conductive-wall predicate (passed to MG; empty = none).
  AmrRuntime(const Geometry& geom, const BoxArray& ba_coarse, const BCRec& bcPhi,
             std::vector<AmrRuntimeBlock> blocks, Periodicity base_per = Periodicity{true, true},
             bool replicated_coarse = true, std::function<bool(Real, Real)> active = {})
      : geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        mg_(geom, ba_coarse, bcPhi, std::move(active), replicated_coarse),
        blocks_(std::move(blocks)) {
    if (blocks_.empty())
      throw std::runtime_error("AmrRuntime : at least one block required");
    for (const auto& b : blocks_)
      if (!b.levels || b.levels->empty())
        throw std::runtime_error(
            "AmrRuntime : each block must carry at least one level "
            "(coarse) on the shared layout");
    nlev_ = static_cast<int>(blocks_[0].levels->size());

    // EXACT layout consistency between blocks (the aux is shared per level): same number of levels,
    // and per level same BoxArray (boxes AND order), same DistributionMapping, same dx/dy. SAME guard
    // as AmrSystemCoupler (detail::same_layout_or_throw): all blocks live on ALL patches of the
    // UNIQUE shared hierarchy. A single block matches itself trivially (the loop over the other blocks
    // is empty).
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_)
        ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // Width of the SHARED aux channel: max of the blocks' aux_comps (>= kAuxBaseComps). Counterpart of
    // AmrSystemCoupler::system_aux_comps: a block reading an extra field (B_z, T_e) has the room at
    // each level, a base block ignores the extra components. PR1 does not POPULATE multi-block B_z (no
    // bz_ here), but we size the channel to the widest anyway so that load_aux<aux_comps<Model>> never
    // reads out of bounds. Without an extra-field block -> kAuxBaseComps (3) -> allocation strictly
    // identical to the base case.
    aux_ncomp_ = kAuxBaseComps;
    for (const auto& b : blocks_)
      if (b.aux_ncomp > aux_ncomp_)
        aux_ncomp_ = b.aux_ncomp;

    // SHARED aux: one MultiFab (phi, grad phi) per level, on the common grid. Sized once -> stable
    // addresses for the blocks' aux pointers. The shared layout is that of block 0
    // (same_layout_or_throw guard: identical for all).
    aux_.resize(nlev_);
    const auto& L0 = *blocks_[0].levels;
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(L0[k].U.box_array(), L0[k].U.dmap(), aux_ncomp_, 1);
    for (auto& b : blocks_)
      for (int k = 0; k < nlev_; ++k)
        (*b.levels)[k].aux = &aux_[k];

    // Tag predicates of the union regrid: one empty slot per block (set_block_tag_predicate fills
    // them). Empty by default -> no tag -> frozen hierarchy (regrid is not called anyway as long as
    // set_regrid has not activated regrid_every_ > 0).
    block_tag_.resize(blocks_.size());
  }

  int nlev() const { return nlev_; }
  std::size_t n_blocks() const { return blocks_.size(); }
  /// Conservative VariableSet (names + physical roles, Model::conservative_vars()) of block @p b. The
  /// SAME cons_vars that add_coupled_source resolves (block, role) against; exposed read-only so the
  /// facade can resolve a name/role-selected regrid variable into a component per block (ADC-296).
  /// @throws if @p b is out of bounds.
  const VariableSet& block_cons_vars(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_cons_vars : block index out of bounds");
    return blocks_[b].cons_vars;
  }
  std::size_t n_coupled_sources() const { return coupled_sources_.size(); }
  MultiFab& phi() { return mg_.phi(); }
  // System Poisson right-hand side after the last solve_fields: f = Sum_b elliptic_rhs_b(U_b) on the
  // shared coarse. Exposed to check the CO-LOCATED SUM (PR1 test); same grid as the coarse (the
  // blocks' contributions are accumulated there at the same cells).
  MultiFab& poisson_rhs() { return mg_.rhs(); }
  const MultiFab& aux(int k) const { return aux_[k]; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return *blocks_[b].levels; }
  Real mass(std::size_t b) const { return blocks_[b].mass(); }
  std::vector<double> density(std::size_t b) const { return blocks_[b].density(); }
  int solve_count() const { return solve_count_; }
  int regrid_count() const { return regrid_count_; }

  /// Tag predicate of the union regrid: (ConstArray4 of the read field, i, j) -> should we refine ?
  /// HOST type (evaluated in the host loop of tag_cells, never on device): a std::function capturing a
  /// concrete functor is licit (nvcc-safe -- the predicate does not enter a kernel). We use it for the
  /// PER-BLOCK criterion (read on the block density/U, component 0) and for the phi criterion (read on
  /// the shared aux). docs/AMR_REGRID_UNION_TAGS_DESIGN.md (D1, D4).
  using TagPredicate = std::function<bool(const ConstArray4&, int, int)>;

  /// Activates the UNION-TAGS REGRID at the cadence @p every (in macro-steps): every @p every
  /// macro-steps, BEFORE the macro-step's step(dt) (D2, consistent with the single-block
  /// amr_dsl_block.hpp:104), the shared hierarchy is re-gridded from the UNION of the tags of all
  /// blocks + phi. @p every == 0 (DEFAULT) -> FROZEN hierarchy, regrid never called -> BIT-IDENTICAL
  /// trajectory to the historical one (the feature is opt-in). @p grow: tag dilation (nesting +
  /// anticipation); @p margin: nesting (clamp the patches to the boundaries). Must be called BEFORE
  /// the first step.
  void set_regrid(int every, int grow = 2, int margin = 2) {
    if (every < 0)
      throw std::runtime_error("AmrRuntime::set_regrid : regrid_every >= 0");
    regrid_every_ = every;
    regrid_grow_ = grow;
    regrid_margin_ = margin;
  }

  /// Registers the TAG PREDICATE of block @p b (D1: PER-BLOCK union criterion). The predicate is
  /// evaluated on the block U (component 0 = density, or a discrete gradient at the caller's charge) at
  /// the PARENT level during the regrid; the UNION (OR) of the predicates of all blocks + the phi
  /// criterion drives the clustering. A block WITHOUT a registered predicate tags nothing on ITS side
  /// (it stays re-gridded as background, present everywhere, by the union of the other criteria).
  /// @throws if @p b is out of bounds.
  void set_block_tag_predicate(std::size_t b, TagPredicate crit) {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::set_block_tag_predicate : block index out of bounds");
    block_tag_[b] = std::move(crit);
  }

  /// Registers the PHI TAG PREDICATE (D4: SEPARATE phi criterion, on |grad phi|). The predicate is
  /// evaluated on the shared aux of the parent level (components 1,2 = grad phi in x,y) during the
  /// regrid; it adds to the union of the blocks' tags. Not registered -> phi does not contribute to
  /// the union.
  void set_phi_tag_predicate(TagPredicate crit) { phi_tag_ = std::move(crit); }

  /// Registers a model-NAMED aux field (ADC-291) at shared-channel component @p comp (= kAuxNamedBase
  /// + k for the k-th named field of a block), as a coarse base-level field @p field (n*n row-major,
  /// global cell index j*nx+i). The field is STATIC (external to the elliptic): solve_fields re-applies
  /// it onto the coarse aux every macro-step AFTER field_postprocess (which only writes phi/grad,
  /// comps 0..2) and BEFORE the coarse->fine injection, so it reaches every level and SURVIVES a regrid
  /// (regrid re-solves). AMR counterpart of System::set_aux_field_component. No-op default: without a
  /// named field the map is empty and the path is bit-identical. @p comp must be >= kAuxNamedBase and
  /// within the channel (the facade validates and resolves the name).
  void set_named_aux(int comp, std::vector<Real> field) {
    named_aux_[comp] = std::move(field);
    if (!aux_.empty())
      apply_named_aux();  // reflect immediately if the hierarchy already exists
  }

  /// Registers a per-field aux HALO policy (ADC-369) for the named component @p comp: solve_fields
  /// applies it onto the COARSE aux AFTER the shared fill_ghosts, overriding only that component's
  /// physical-face ghosts (periodic faces stay periodic). Coarse-level scope (fine patches touching the
  /// domain boundary inherit the shared BC). No-op default. AMR counterpart of
  /// System::set_aux_field_halo_component.
  void set_named_aux_bc(int comp, AuxHaloPolicy policy) { named_aux_bc_[comp] = policy; }

  /// Registers an inter-species COUPLED SOURCE (DSL CoupledSource, P5 bytecode) on the runtime facade,
  /// counterpart of System::add_coupled_source. The ABI is FLAT (postfix bytecode): we resolve each
  /// (block, role) into (block index, component) then store a closure that, at each macro-step AFTER
  /// the transport, applies the source by additive forward-Euler splitting via coupled_source_step. The
  /// coupling is ENTIRELY baked into a stack machine (device-clean functor CoupledSourceKernel): NO
  /// per-cell Python callback in the hot path.
  ///
  /// CONSERVATION (conservative exchange): with an add_pair construction (one +expr term on one block,
  /// -expr exactly on the other, SAME cell), the two per-cell contributions are opposite up to sign, so
  /// n_a + n_b is conserved PER CELL (and globally) to machine precision, independent of dt and of the
  /// state. The engine does not enforce it (an ionization creating a pair is licit): conservation is a
  /// property of the constructed coupling, checked test-side.
  ///
  /// @param in_blocks/in_roles  READ fields (one register per (block, role)), in register order.
  /// @param consts              constants (parameters), loaded into the registers after the inputs.
  /// @param out_blocks/out_roles target (block, role) of each source term.
  /// @param prog_ops/prog_args  CONCATENATED postfix bytecode of all the terms (split by prog_lens).
  /// @param prog_lens           program length of each term (size == out_blocks).
  /// @throws std::runtime_error on an inconsistent form, an unknown role, an unknown block, an opcode
  ///         or register out of bounds, or a program too long (same guards as System).
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens) {
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    const int n_terms = static_cast<int>(out_blocks.size());
    // --- form validation (before any step, EXPLICIT errors); mirror of System::add_coupled_source.
    if (n_terms == 0)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : no source term (out_blocks empty)");
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : in_blocks / in_roles of different sizes");
    if (static_cast<int>(out_roles.size()) != n_terms ||
        static_cast<int>(prog_lens.size()) != n_terms)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : out_blocks / out_roles / prog_lens of "
          "different sizes");
    if (prog_ops.size() != prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : prog_ops / prog_args of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (n_terms > kCsMaxTerms)
      throw std::runtime_error("AmrRuntime::add_coupled_source : too many source terms (> " +
                               std::to_string(kCsMaxTerms) + ")");
    // Resolves (block, role) -> (block index, component) by the block CONSERVATIVE descriptor, like
    // System (#181). An unknown block throws immediately; an unknown (non-canonical) role too.
    auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_source : no block named '" + block + "'");
      // STRICT (no silent fallback; mirror of System::add_coupled_source #181): a DSL coupled source
      // targets a (block, role) EXPLICITLY requested by the user. The role is addressed BY NAME: a
      // canonical role name OR a user-defined role label (index_of(string), ADC-292). If the block does
      // NOT expose this role, a fallback to component 0 would apply the source to the wrong field
      // SILENTLY (the false-positive identified at the Lot E review). We throw, listing what the block
      // exposes.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : block '" + block + "' does not expose role '" + role +
            "' (roles: " + (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
            ", no silent fallback to component 0)");
      return {b, comp};
    };
    // Inputs: (block, component) read per cell. Captured by INDEX -> we rebuild the Array4 at EACH
    // application (the fabs live in the level stack, repointed per level in the splitting).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      auto [b, comp] =
          resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    std::vector<CsRef> outs(static_cast<std::size_t>(n_terms));
    int off = 0;
    for (int t = 0; t < n_terms; ++t) {
      auto [b, comp] =
          resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
      const int len = prog_lens[static_cast<std::size_t>(t)];
      if (len < 0 || len > kCsMaxProg)
        throw std::runtime_error("AmrRuntime::add_coupled_source : program of term " +
                                 std::to_string(t) + " too long (> " + std::to_string(kCsMaxProg) +
                                 ")");
      if (off + len > static_cast<int>(prog_ops.size()))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : prog_lens inconsistent with prog_ops");
      CsProgram pg;
      pg.len = len;
      for (int k = 0; k < len; ++k) {
        const int opc = prog_ops[static_cast<std::size_t>(off + k)];
        const int a = prog_args[static_cast<std::size_t>(off + k)];
        if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
          throw std::runtime_error("AmrRuntime::add_coupled_source : invalid opcode");
        if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
          throw std::runtime_error(
              "AmrRuntime::add_coupled_source : register out of bounds in the program");
        pg.op[k] = opc;
        pg.arg[k] = a;
      }
      outs[static_cast<std::size_t>(t)] = {b, comp, pg};
      off += len;
    }
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_sources_.push_back(CoupledSourceSpec{std::move(ins), std::move(outs),
                                                 std::move(kconsts), n_in, n_const, n_terms});
  }

  /// Applies ALL the registered coupled sources of a step dt, by forward-Euler splitting. Runtime
  /// counterpart of AmrSystemCoupler::coupled_source_step: we refresh the fields (aux per level) then,
  /// source by source, we apply the bytecode INDEPENDENTLY AT EACH LEVEL of the shared hierarchy (the
  /// blocks live on ALL levels), followed by a fine -> coarse cascade.
  ///
  /// COVERAGE INVARIANT (#169): the source was applied independently on EACH level, so a coarse cell
  /// COVERED by a fine patch would otherwise carry its own coarse source, unrelated to the source seen
  /// by its fine children. A covered coarse cell MUST be the 2x2 average of its children (it does not
  /// represent matter on its own). We restore this consistency by the SAME fine -> coarse cascade
  /// (mf_average_down_mb) as solve_fields and the compile-time engine: without it, the mass diagnostic
  /// (sum of the coarse only) would count a phantom coarse source under the patch. Single-level
  /// hierarchy: no covered cell, the cascade loops do not run -> bit-identical to the no-patch case.
  ///
  /// PER-CELL CONSERVATION: at a given level, each term writes out(i,j,comp) += dt * S(reg(i,j)) on
  /// the SAME cell (i,j) read by the inputs; an add_pair exchange lays +S on one block and -S on the
  /// other AT THE SAME (i,j), so the sum of the two blocks is unchanged cell by cell. Without a
  /// registered source (coupled_sources_ empty): total no-op -> bit-identical trajectory to the
  /// historical one.
  void coupled_source_step(Real dt) {
    if (coupled_sources_.empty())
      return;        // opt-in: no source -> bit-identical path
    solve_fields();  // aux per level up to date (a term may read phi/grad via a future input)
    for (const auto& cs : coupled_sources_) {
      // PER-LEVEL application: at each level k, the blocks share EXACTLY the same layout
      // (same_layout_or_throw guard), so same local_size() and same local indexing -> we iterate in
      // parallel over the local fabs. local_size()==0 on a rank without a box -> empty loop (MPI-safe).
      for (int k = 0; k < nlev_; ++k) {
        const int sref = cs.n_in > 0 ? cs.ins[0].block : cs.outs[0].block;
        MultiFab& Uref = (*blocks_[static_cast<std::size_t>(sref)].levels)[k].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = cs.n_in;
          kern.n_const = cs.n_const;
          kern.n_terms = cs.n_terms;
          for (int c = 0; c < cs.n_in; ++c) {
            kern.in[c] =
                (*blocks_[static_cast<std::size_t>(cs.ins[static_cast<std::size_t>(c)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.in_comp[c] = cs.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < cs.n_const; ++c)
            kern.consts[c] = cs.kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < cs.n_terms; ++t) {
            kern.out[t] =
                (*blocks_[static_cast<std::size_t>(cs.outs[static_cast<std::size_t>(t)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.out_comp[t] = cs.outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = cs.outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li),
                        kern);  // NAMED functor (device-clean), additive forward-Euler
        }
      }
      // Restore the consistency of the covered coarse cells (cf. COVERAGE INVARIANT above).
      for (auto& b : blocks_)
        for (int k = nlev_ - 1; k >= 1; --k)
          mf_average_down_mb((*b.levels)[k].U, (*b.levels)[k - 1].U);
    }
  }

  /// sync_down (per block) + system coarse Poisson (CO-LOCATED SUMMED RHS) + coarse aux + fine
  /// injection. Reproduces AmrSystemCoupler::solve_fields identically, but the system RHS is assembled
  /// by the blocks' add_elliptic_rhs closures (Sum_b elliptic_rhs_b(U_b)) instead of a compile-time
  /// RhsAssembler.
  void solve_fields() {
    ++solve_count_;
    // 1. average_down per block (fine -> coarse) over the whole hierarchy.
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(L[k].U, L[k - 1].U);
    }

    // 2. SUMMED and CO-LOCATED system RHS: f = Sum_b elliptic_rhs_b(U_b) on the coarse. We reset to
    // zero then each block ACCUMULATES (+=) its contribution on the SAME cells of the shared coarse
    // (mg_.rhs() shares the coarse layout).
    mg_.rhs().set_val(Real(0));
    for (auto& b : blocks_)
      b.add_elliptic_rhs((*b.levels)[0].U, mg_.rhs());
    mg_.solve();

    // 3. coarse aux = (phi, grad phi) via the SAME clean path as AmrSystemCoupler: fill the ghosts of
    // phi according to bcPhi_, field_postprocess (phi + grad), fill the ghosts of aux according to
    // aux_bc_ (derived from bcPhi_). Handles the non-periodic case (Foextrap).
    fill_ghosts(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    // 3b. model-NAMED aux (ADC-291): re-apply the static named fields onto the coarse valid cells
    // BEFORE fill_ghosts (so their ghosts are filled) and the injection (so they reach every level).
    // No-op when no named field was set; field_postprocess wrote only comps 0..2, so this never clobbers
    // phi/grad. This is what makes named aux survive a regrid (regrid re-solves -> re-applies).
    apply_named_aux();
    fill_ghosts(aux_[0], dom_, aux_bc_);
    apply_named_aux_bc();  // ADC-369: per-field halo override on the coarse physical ghosts (after the
                           // shared fill, before injection); no-op when no policy declared.
    // 4. coarse->fine injection of the aux (parent replicated only at level 1 if coarse replicated).
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  /// UNION-TAGS REGRID (capstone Phase 2, C.6; docs/AMR_REGRID_UNION_TAGS_DESIGN.md, steps R0-R8).
  /// Re-grids the SHARED hierarchy from the UNION (cell-by-cell OR) of the tags of ALL blocks (per-block
  /// predicate, D1) + the phi tags (on |grad phi|, D4), followed by ONE SINGLE Berger-Rigoutsos
  /// clustering -> ONE SINGLE new fine layout applied to ALL blocks (including those held by their
  /// stride, D3) AND to the shared aux. Maintains the shared-layout PRECONDITION (same_layout_or_throw)
  /// after the regrid. v1 with 2 LEVELS (coarse + 1 fine, D5): no-op if nlev < 2. No-op (grid
  /// unchanged) if the union of the tags is empty (nothing to refine).
  void regrid() {
    if (nlev_ < 2)
      return;  // 2 levels required (D5): nothing to re-grid in single-level
    const int fk = nlev_ - 1, pk = fk - 1;  // fine + its parent (pk == 0 in v1 with 2 levels)

    // (R0) PRECONDITION: fields up to date (aux per level, for the |grad phi| criterion). The per-block
    // mass snapshot is NOT needed by the engine (conservation is checked test-side V1).
    solve_fields();

    // (R1)+(R2) PER-BLOCK TAGS (on the block U at the parent level) + PHI TAGS (on the shared aux).
    const int PNX = dom_.nx() << pk, PNY = dom_.ny() << pk;
    const Box2D pdom = Box2D::from_extents(PNX, PNY);
    std::vector<TagBox> parts;
    parts.reserve(blocks_.size() + 1);
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      const TagPredicate& crit = block_tag_[b];
      if (!crit)
        continue;  // block without a criterion: tags nothing on its side (re-gridded as background)
      parts.push_back(tag_cells((*blocks_[b].levels)[pk].U, pdom, crit));
    }
    if (phi_tag_)
      parts.push_back(tag_cells(aux_[pk], pdom, phi_tag_));
    if (parts.empty())
      return;  // no active criterion -> no tagged cell -> grid unchanged

    // (R3) UNION (OR) of the tags + dilation (nesting + anticipation of the structures moving).
    TagBox grown = grow_tags(tag_union(parts), regrid_grow_, pdom);

    // (R4)+(R5) cross-rank collective reduction (if coarse distributed) + UNIQUE clustering -> SHARED
    // fine layout. all_reduce_or_inplace is called INSIDE regrid_compute_fine_layout for distributed
    // pk==0: all ranks start from the SAME tag grid -> IDENTICAL fb/dmap per rank (otherwise MPI
    // desync).
    auto [fb, dmap] =
        regrid_compute_fine_layout(std::move(grown), pdom, pk, regrid_margin_, replicated_coarse_);
    if (fb.size() == 0)
      return;  // nothing to refine: we keep the current grid (no-op)

    // (R6) COHERENT PROLONG / RESTRICT of ALL blocks on the SAME fb/dmap (including the blocks held by
    // their stride: their frozen state is present everywhere and contributes to the Poisson, D3). The
    // ghost width is INHERITED per block (a MUSCL order-2 block carries 2 ghosts; a Minmod block and a
    // VanLeer one may differ), so the scheme does not read out of bounds at the next step (V2 / risk
    // X4).
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      const int ngf = L[fk].U.n_grow();
      L[fk].U = regrid_field_on_layout(fb, dmap, L[pk].U, L[fk].U, pk, ngf, replicated_coarse_);
    }

    // (R7) REBUILD OF THE SHARED AUX (one only, width aux_ncomp_) on the new layout + RE-WIRING of the
    // aux pointer of EACH block. The address &aux_[fk] stays stable (in-place reallocation of the
    // MultiFab in the existing std::vector) -> the pointers of the other levels do not move.
    aux_[fk] = MultiFab(fb, dmap, aux_ncomp_, 1);
    for (auto& b : blocks_)
      (*b.levels)[fk].aux = &aux_[fk];

    // (V3) SHARED-LAYOUT INVARIANT: all blocks MUST live on EXACTLY the same fb/dmap (boxes, order,
    // rank per box) after the regrid. Collective guard (cross-block); catches any inconsistent
    // reconstruction before it corrupts the shared aux / the summed Poisson.
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_)
        ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // (R8) RESTORATION OF THE COVERAGE INVARIANT: re-solve so that phi / grad phi are consistent with
    // the new grid AND to trigger the fine -> coarse cascade (mf_average_down_mb, in solve_fields) that
    // restores the covered coarse cells (otherwise a mass diagnostic, sum of the coarse only, would
    // count a phantom coarse value under the new patch, X5).
    solve_fields();
    ++regrid_count_;
  }

  /// Advances the system by one macro-step dt. We first solve the fields (co-located summed Poisson,
  /// ONCE per macro-step: OncePerStep cadence), then each block advances over ITS level stack with ITS
  /// scheme, honoring its stride cadence and its substeps, and ITS temporal treatment. Runtime
  /// counterpart of AmrSystemCoupler::step (OncePerStep): the compile-time version carries
  /// substeps/stride in block_substeps_v / block_stride_v and chooses the treatment by the constexpr
  /// block_time_treatment_v; here the engine carries the substep loop, the stride filter AND the
  /// IMEX-vs-explicit selection.
  ///
  /// TREATMENT SELECTION (capstone vii):
  ///  - EXPLICIT block (b.imex == false): the advance closure does ONE advance_amr (transport +
  ///    forward-Euler source), called substeps times;
  ///  - IMEX block (b.imex == true): the imex_advance closure does ONE SOURCE-FREE advance_amr then the
  ///    IMPLICIT stiff source backward_euler_source per level + cascade (cf.
  ///    AmrRuntimeBlock::imex_advance), called substeps times. Unconditionally stable on a stiff
  ///    relaxation (where the explicit, of factor |1 - dt/eps|, DIVERGES as soon as dt > 2 eps).
  /// The substep loop is COMMON to both treatments (substeps applications of h = bdt/substeps), so the
  /// runtime also SUB-CYCLES the IMEX splitting. At substeps=1 this sub-cycling is a no-op and the IMEX
  /// path coincides with the IMEX branch of the compile-time engine AmrSystemCoupler::step; for
  /// substeps>1 it DIVERGES deliberately from that engine (which itself ignores substeps on its IMEX
  /// branch): see IMEX SEMANTICS UNDER substeps in the header (CFL-safe on the transport,
  /// backward-Euler stable at any step, stiff relaxation more accurate). imex == false everywhere ->
  /// advance path only -> bit-identical trajectory to the historical one (the IMEX is opt-in).
  void step(Real dt) {
    solve_count_ = 0;
    // UNION-TAGS REGRID (capstone Phase 2, C.6; D2: BEFORE the macro-step's step, consistent with the
    // single-block amr_dsl_block.hpp:108). regrid_every_ cadence in MACRO-STEPS, OUTSIDE the substep
    // loops and the stride windows (macro-step granularity ONLY, D3). regrid_every_ == 0 -> FROZEN
    // hierarchy, regrid never called -> BIT-IDENTICAL trajectory to the historical one. The guard
    // macro_step_ > 0 (like the single-block) avoids a regrid at the very first step (the initial grid
    // is already the build one). The regrid sits BEFORE solve_fields below: it does its own
    // solve_fields (R0/R8), then the step's solve_fields recomputes phi on the re-gridded grid.
    if (regrid_every_ > 0 && macro_step_ > 0 && macro_step_ % regrid_every_ == 0)
      regrid();
    // System Poisson solved ONCE on the current state (OncePerStep cadence). A HELD block (stride > 1,
    // outside end-of-window) contributed with its FROZEN state since its last advance: loose coupling
    // assumed by the multirate, exactly like System::step / AmrSystemCoupler in OncePerStep. phi stays
    // frozen during the blocks' advance (no per-substep re-solve here). When reached from step_cfl this
    // re-solves an unchanged state (a second solve), kept on purpose; see the ADC-318 note in step_cfl.
    solve_fields();
    for (auto& b : blocks_) {
      // HOLD-THEN-CATCH-UP cadence (cf. AmrRuntimeBlock::stride, #140): the block is HELD as long as
      // (macro_step_+1) % stride != 0, then CATCHES UP at end-of-window by an effective step stride*dt.
      // The end-of-window catch-up keeps the block temporally consistent with the fast ones at the
      // coupling point (never in the future). stride=1: always true -> every step, bit-identical.
      if ((macro_step_ + 1) % b.stride != 0)
        continue;
      // NEWTON DIAGNOSTICS (OPT-IN): RESET of the report at the HEAD of the block advance (parity with
      // System::AdvanceImex::operator() which resets nreport before its substep loop). The report then
      // AGGREGATES over all the levels AND substeps of THIS advance (imex_advance accumulates per level
      // via backward_euler_source; step() calls imex_advance substeps times without re-resetting).
      // Placed AFTER the stride skip: a HELD block keeps the report of its LAST advance ("last advance"
      // semantics of System). No-op for a block without diagnostics (newton_report null).
      if (b.newton_diagnostics && b.newton_report)
        b.newton_report->reset();
      const Real bdt = dt * static_cast<Real>(b.stride);  // catch-up: effective step stride*dt
      // substeps equal substeps of bdt/substeps. The chosen closure does ONE advance per call;
      // substeps=1 -> a single advance of bdt (bit-identical to the single-substep case). Per-block
      // treatment SELECTION: IMEX (source-free transport + implicit stiff source, mirrors the IMEX
      // branch of AmrSystemCoupler::step) if b.imex, otherwise EXPLICIT (transport + forward-Euler
      // source). The test is PER BLOCK and stable: a single IMEX block changes nothing for the
      // neighboring explicit blocks.
      // NOTE substeps>1: the loop below calls step_block substeps times for BOTH treatments, so the
      // IMEX splitting is SUB-CYCLED (K Lie steps over bdt/K). The compile-time, for its part, applies
      // its IMEX only once over bdt (it ignores substeps on its IMEX branch): divergence INTENTIONAL
      // and sound for substeps>1 (cf. IMEX SEMANTICS UNDER substeps in the file header).
      const Real h = bdt / static_cast<Real>(b.substeps);
      auto& step_block = b.imex ? b.imex_advance : b.advance;
      for (int s = 0; s < b.substeps; ++s)
        step_block(*b.levels, dom_, h, base_per_, replicated_coarse_);
      // PROJECTION PONCTUELLE post-pas (ADC-177) : par niveau, APRES substeps + reflux/cascade.
      // Cell-local + idempotente -> conservation preservee (flux-registres deja regles). No-op si vide.
      if (b.project_per_level)
        b.project_per_level(*b.levels);
    }
    // Inter-species coupled sources AFTER the transport (same order as AmrSystemCoupler: transport then
    // coupled_source_step), by forward-Euler splitting. No-op if no source registered -> bit-identical
    // trajectory to the historical one (the feature is opt-in).
    coupled_source_step(dt);
    ++macro_step_;
  }

  /// substeps/stride-aware CFL step (runtime counterpart of System::step_cfl, EXACT mirror of its
  /// formula). A block of stride cadence advances by an effective step stride*dt in substeps substeps,
  /// so each substep is worth stride*dt/substeps; the per-substep stability condition
  /// stride*dt/substeps <= cfl*h/w_b gives dt <= cfl*h*substeps_b/(stride_b*w_b). The GLOBAL dt is the
  /// min over the blocks (the most constraining). We first solve the fields (per-block max_speed
  /// requires the aux up to date), compute dt, then advance by one step(dt). @p h = coarse mesh spacing
  /// (dx_coarse). Returns the dt used. Single-block (a single block, stride=1): if w_b is the only
  /// constraining one, dt = cfl*h*substeps/w (identical to System::step_cfl single-block).
  Real step_cfl(Real cfl, Real h) {
    // NOTE (ADC-318): this pre-solve plus step(dt)'s own head solve below is a DOUBLE Poisson solve on
    // the SAME unchanged state (regrid_every=0 freezes the grid in between). It looks redundant but is
    // NOT, and is INTENTIONALLY kept. GeometricMG::solve() is warm-started and iterates to a RELATIVE
    // tolerance (rel_tol 1e-8; abs_tol 0 by default, so its off-step early-exit never fires here), so the
    // second solve does not recompute identical phi: starting from the first solve's iterate it
    // over-converges it by ~rel_tol. Skipping the second solve would therefore NOT be bit-identical; it
    // drifts the trajectory by ~3e-10 over 20 steps (below the solver tolerance and far below the O(dt^2)
    // scheme error, but nonzero). The de-dup was declined to preserve the exact historical bit-stream
    // (SystemStepper::step_cfl avoids the double solve by INLINING its advance, not by skipping a solve).
    solve_fields();  // aux up to date: each block's max_speed reads it on the current coarse
    Real dt = std::numeric_limits<Real>::infinity();
    last_dt_reason_ = "degenerate";
    for (auto& b : blocks_) {
      const Real w = std::max(b.max_speed((*b.levels)[0].U, aux_[0]), kCflSpeedFloor);
      Real dt_b = cfl * h * static_cast<Real>(b.substeps) / (static_cast<Real>(b.stride) * w);
      const char* why = "transport";
      // OPTIONAL block BOUNDS (AMR StabilityPolicy, audit 2026-06): same substeps/stride formulas as
      // SystemStepper::step_cfl, evaluated on the COARSE. Empty closures (model without the trait) ->
      // not queried, transport bound only (bit-identical).
      if (b.source_frequency) {
        const Real mu = b.source_frequency((*b.levels)[0].U, aux_[0]);
        if (mu > Real(0)) {
          const Real dt_src =
              cfl * static_cast<Real>(b.substeps) / (static_cast<Real>(b.stride) * mu);
          if (dt_src < dt_b) {
            dt_b = dt_src;
            why = "source_frequency";
          }
        }
      }
      if (b.stability_dt) {
        const Real db = b.stability_dt((*b.levels)[0].U, aux_[0]);
        if (db > Real(0)) {
          const Real dt_adm = db * static_cast<Real>(b.substeps) / static_cast<Real>(b.stride);
          if (dt_adm < dt_b) {
            dt_b = dt_adm;
            why = "stability_dt";
          }
        }
      }
      if (dt_b < dt) {
        dt = dt_b;
        last_dt_reason_ = std::string(why) + ":" + b.name;
      }
    }
    // Declared frequencies of the coupled sources (CoupledSource.frequency): bound on the MACRO-step
    // (the couplings apply once per macro-step), dt <= cfl / mu, without substeps/stride.
    for (const auto& cs : coupled_freqs_) {
      const Real dt_cs = cfl / cs.mu;
      if (dt_cs < dt) {
        dt = dt_cs;
        last_dt_reason_ = "coupled_source:" + cs.label;
      }
    }
    // PER-CELL frequencies (CoupledSource.frequency with an Expr): mu(U) reduced (MAX) on the COARSE
    // level of the input blocks (where the AMR CFL lives), GLOBAL all_reduce_max (ALL ranks, neutral
    // without a local box), bound dt <= cfl / max(mu). Same reason "coupled_source:<label>" as the
    // constant. No per-cell source -> empty loop (bit-identical). The Array4 are rebuilt at EACH step
    // (the hierarchy fabs are repointed by the regrid), like coupled_source_step.
    for (const auto& ce : coupled_freq_exprs_) {
      Real m = 0;
      if (ce.n_in > 0) {
        auto& Uref =
            (*blocks_[static_cast<std::size_t>(ce.ins[0].block)].levels)[0].U;  // coarse (lev 0)
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledFreqKernel kern;
          kern.n_in = ce.n_in;
          kern.n_const = ce.n_const;
          for (int c = 0; c < ce.n_in; ++c) {
            kern.in[c] =
                (*blocks_[static_cast<std::size_t>(ce.ins[static_cast<std::size_t>(c)].block)]
                      .levels)[0]
                    .U.fab(li)
                    .array();
            kern.in_comp[c] = ce.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < ce.n_const; ++c)
            kern.consts[c] = ce.kconsts[static_cast<std::size_t>(c)];
          kern.prog = ce.prog;
          m = std::max(m, reduce_max_cell(Uref.box(li), kern));
        }
      } else {
        // Program WITHOUT an input field (constant in bytecode): evaluated once on the constants.
        Real reg[kCsMaxReg];
        for (int c = 0; c < ce.n_const; ++c)
          reg[c] = ce.kconsts[static_cast<std::size_t>(c)];
        const Real mu0 = ce.prog.eval(reg);
        if (mu0 > Real(0))
          m = mu0;
      }
      const double mu = all_reduce_max(static_cast<double>(m));  // ALL ranks (collective symmetry)
      if (mu > 0.0) {
        const Real dt_cs = cfl / static_cast<Real>(mu);
        if (dt_cs < dt) {
          dt = dt_cs;
          last_dt_reason_ = "coupled_source:" + ce.label;
        }
      }
    }
    // GLOBAL bounds (AmrRuntime::add_dt_bound, parity with System::add_dt_bound): evaluated PER RANK
    // then reduced all_reduce_min (dt identical on all ranks; <= 0/non-finite = inert).
    for (const auto& g : dt_bounds_) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (static_cast<Real>(v) < dt) {
        dt = static_cast<Real>(v);
        last_dt_reason_ = "global:" + g.label;
      }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * h / kCflSpeedFloor;  // guard (no block: impossible here)
      last_dt_reason_ = "degenerate";
    }
    step(dt);
    return dt;
  }

  /// MACRO-STEP counter of the engine (regrid + hold-then-catch-up stride cadence: regrid when
  /// macro_step_ % regrid_every == 0, stride catch-up when (macro_step_+1) % stride == 0).
  int macro_step() const { return macro_step_; }
  /// RESTORES the macro-step counter (IO v1, reserved for restart via AmrSystem::set_clock): without
  /// it the regrid/stride cadence would restart from phase 0 after a resume. No effect on the level
  /// state; only sets the cadence phase.
  void set_macro_step(int s) { macro_step_ = s; }

  /// GLOBAL step bound (AMR counterpart of System::add_dt_bound): fn() evaluated once per step_cfl,
  /// all_reduce_min, <= 0/non-finite = inert. For user coupling/scheduler/policies.
  void add_dt_bound(const std::string& label, std::function<double()> fn) {
    dt_bounds_.push_back(GlobalDtBound{label, std::move(fn)});
  }

  /// DECLARED frequency of a coupled source (CoupledSource.frequency, wave-3 audit): step bound
  /// dt <= cfl / mu on the MACRO-step (the couplings apply once per macro-step). mu <= 0 = inert (no
  /// bound).
  void add_coupled_frequency(const std::string& label, Real mu) {
    if (mu > Real(0))
      coupled_freqs_.push_back(CoupledFreqDecl{label, mu});
  }

  /// PER-CELL COUPLED frequency (CoupledSource.frequency with an Expr, refinement of the CONSTANT
  /// frequency above): a bytecode program mu(U) on the SAME register table as the source (inputs
  /// in_blocks/in_roles then constants consts). Evaluated at each step_cfl on the COARSE level of the
  /// input blocks (where the AMR CFL lives: h = dx_coarse), MAX reduction + global all_reduce_max,
  /// bound dt <= cfl / max(mu) on the macro-step. The bound is thus evaluated on the COARSE (not on the
  /// fine patches): consistent with the AMR transport CFL, but a local under-estimate of mu under a
  /// fine patch is not seen (assumed choice, documented). Empty program -> ignored (no bound). Form
  /// validation (opcodes / register bounds) and STRICT role resolution, like add_coupled_source.
  void add_coupled_frequency_expr(const std::string& label,
                                  const std::vector<std::string>& in_blocks,
                                  const std::vector<std::string>& in_roles,
                                  const std::vector<double>& consts,
                                  const std::vector<int>& freq_prog_ops,
                                  const std::vector<int>& freq_prog_args) {
    if (freq_prog_ops.empty() && freq_prog_args.empty())
      return;  // no per-cell frequency
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : in_blocks / in_roles of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : freq_prog_ops / freq_prog_args of different "
          "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : frequency program too long (> " +
          std::to_string(kCsMaxProg) + ")");
    // Resolves (block, role) -> (block index, component), STRICT (mirror of add_coupled_source).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      const std::string& block = in_blocks[static_cast<std::size_t>(c)];
      const std::string& role = in_roles[static_cast<std::size_t>(c)];
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : no block named '" +
                                 block + "'");
      // Role addressed BY NAME: a canonical role name OR a user-defined role label (ADC-292), STRICT.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : block '" + block +
                                 "' does not expose role '" + role + "' (roles: " +
                                 (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
                                 ", no silent fallback to component 0)");
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    CsProgram pg;
    pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : register out of bounds in the frequency");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_freq_exprs_.push_back(
        CoupledFreqExprDecl{label, std::move(ins), pg, n_in, n_const, std::move(kconsts)});
  }

  /// ACTIVE bound of the last step_cfl ("transport:<block>" / "source_frequency:<block>" /
  /// "stability_dt:<block>" / "global:<label>" / "degenerate" / "" before the first step).
  const std::string& last_dt_bound() const { return last_dt_reason_; }

  /// NEWTON REPORT (OPT-IN IMEX diagnostics) of block @p name, AGGREGATED over the levels and substeps
  /// of its LAST advance (cf. AmrRuntimeBlock::newton_report). AMR counterpart of System::newton_report.
  /// @throws std::runtime_error if the block is unknown, or if it was not added with
  ///         newton_diagnostics=true (no silently empty report).
  const NewtonReport& newton_report(const std::string& name) const {
    const int b = block_index(name);
    if (b < 0)
      throw std::runtime_error("AmrRuntime::newton_report : no block named '" + name + "'");
    const AmrRuntimeBlock& blk = blocks_[static_cast<std::size_t>(b)];
    if (!blk.newton_diagnostics || !blk.newton_report)
      throw std::runtime_error(
          "AmrRuntime::newton_report : Newton diagnostics not enabled for block '" + name +
          "' ; add the block with newton_diagnostics=True (adc.IMEX(newton_diagnostics=True))");
    return *blk.newton_report;
  }

  /// Coarse potential (component 0 of the shared aux) as an n*n row-major field. Solves the fields if
  /// needed (counterpart of AmrSystem::potential), then reads aux(0). Identical for all blocks.
  std::vector<double> potential() {
    solve_fields();
    return blocks_[0].potential(aux_[0]);
  }

  /// Max SYSTEM wave speed (max over the blocks) on the current coarse. Requires the aux up to date.
  Real max_speed() {
    solve_fields();
    Real w = Real(1e-12);
    for (auto& b : blocks_) {
      const Real wb = b.max_speed((*b.levels)[0].U, aux_[0]);
      if (wb > w)
        w = wb;
    }
    return w;
  }

  int n_patches() const {
    const auto& L = *blocks_[0].levels;
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  }

  // Index-space signatures of the fine patches (level + inclusive lo/hi corners), for ALL fine levels.
  // Read-only of the GLOBAL BoxArray (all boxes/all ranks) already stored -> rank-independent, zero
  // communication, NO hot-path cost (query between steps). Mirror of n_patches(): the same box_array()
  // that gives the COUNT gives the BOXES. Block 0 representative (SHARED layout, same_layout_or_throw
  // guard). Loop k = 1..nlev-1: a single fine level today (ratio 2), correct if a future adds levels
  // (the level field disambiguates the spacing dx = L / (n << level) Python-side).
  std::vector<PatchBox> patch_boxes() const {
    const auto& L = *blocks_[0].levels;
    std::vector<PatchBox> out;
    for (int k = 1; k < static_cast<int>(L.size()); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const Box2D& b : bxs)
        out.push_back(PatchBox{k, b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  }

  // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic). Block 0 is the SHARED layout
  // (same_layout_or_throw), so its level-0 MultiFab carries the base BoxArray + DistributionMapping
  // common to all blocks. local_size() = base boxes OWNED by this rank; box_array().size() = total base
  // boxes (all ranks). Mirror of n_patches(): a query between steps, no communication, no hot-path cost.
  int coarse_local_boxes() const { return (*blocks_[0].levels)[0].U.local_size(); }
  int coarse_total_boxes() const { return (*blocks_[0].levels)[0].U.box_array().size(); }

 private:
  // Re-applies the model-NAMED aux fields (ADC-291) onto the COARSE shared aux valid cells. Mirror of
  // SystemFieldSolver::apply_named_aux_one (cartesian System): per LOCAL fab (MPI-safe), valid cells
  // only, global flat index j*nx+i. The coarse layout is frozen across regrid (only fine levels are
  // rebuilt), so the stored coarse field stays valid; solve_fields runs the coarse->fine injection
  // right after, carrying the named comps to every level. No-op without a named field.
  void apply_named_aux() {
    if (named_aux_.empty() || aux_.empty())
      return;
    const int row = dom_.nx();
    for (const auto& [comp, field] : named_aux_) {
      if (field.empty() || comp >= aux_ncomp_)
        continue;
      for (int li = 0; li < aux_[0].local_size(); ++li) {
        Array4 a = aux_[0].fab(li).array();
        const Box2D v = aux_[0].box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            a(i, j, comp) = field[static_cast<std::size_t>(j) * row + i];
      }
    }
  }

  // Per-field aux HALO override (ADC-369) on the COARSE aux, AFTER the shared fill_ghosts. Overrides
  // only each declared component's physical-face ghosts (aux_halo_override keeps periodic faces
  // periodic). No-op without a policy. Mirror of SystemFieldSolver::apply_named_aux_bc.
  void apply_named_aux_bc() {
    if (named_aux_bc_.empty() || aux_.empty())
      return;
    for (const auto& [comp, policy] : named_aux_bc_) {
      if (comp >= aux_ncomp_)
        continue;
      fill_physical_bc(aux_[0], dom_, aux_halo_override(aux_bc_, policy), comp);
    }
  }

  // Index of the block named @p name in the registry (-1 if absent). Counterpart of
  // AmrSystem::Impl::block_index (the facade names the blocks; the coupled sources target them by name,
  // resolved once at registration).
  int block_index(const std::string& name) const {
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (blocks_[i].name == name)
        return static_cast<int>(i);
    return -1;
  }

  // Resolved reference of a coupled-source field: (block index, component) + the term bytecode program
  // (empty for an input). Inputs carry only block/comp; outputs carry in addition the postfix program
  // evaluated per cell. We capture the block INDEX (not a fab pointer): the Array4 are rebuilt at each
  // application, per level.
  struct CsRef {
    int block;
    int comp;
    CsProgram prog;  // outputs: term program; inputs: unused (CsProgram{})
  };
  // A registered coupled source: its inputs, its output terms and its constants, ready to be marshaled
  // into a CoupledSourceKernel per level / per fab at application.
  struct CoupledSourceSpec {
    std::vector<CsRef> ins;
    std::vector<CsRef> outs;
    std::vector<Real> kconsts;
    int n_in = 0;
    int n_const = 0;
    int n_terms = 0;
  };

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  GeometricMG mg_;
  std::vector<AmrRuntimeBlock> blocks_;
  // GLOBAL step bounds (add_dt_bound, parity with System) + ACTIVE bound of the last step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  // Declared frequencies of the coupled sources (bound dt <= cfl/mu on the macro-step, wave 3).
  struct CoupledFreqDecl {
    std::string label;
    Real mu;
  };
  std::vector<CoupledFreqDecl> coupled_freqs_;
  // PER-CELL frequencies of the coupled sources (CoupledSource.frequency with an Expr): bytecode
  // program mu(U) evaluated on the coarse at each step_cfl (MAX + all_reduce_max -> dt <= cfl/max(mu)).
  // ins = (block, comp) of the inputs (prog unused); kconsts = constants (same as the source).
  struct CoupledFreqExprDecl {
    std::string label;
    std::vector<CsRef> ins;
    CsProgram prog;
    int n_in = 0;
    int n_const = 0;
    std::vector<Real> kconsts;
  };
  std::vector<CoupledFreqExprDecl> coupled_freq_exprs_;
  std::string last_dt_reason_;
  std::vector<MultiFab> aux_;  // [level], shared by all blocks
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse base-level field
  // (n*n row-major). STATIC user fields re-applied by solve_fields each macro-step (so they persist
  // across regrid). Empty by default -> bit-identical. cf. set_named_aux / apply_named_aux.
  std::map<int, std::vector<Real>> named_aux_;
  // Per-field aux HALO policy (ADC-369): component -> uniform boundary policy, applied to the coarse aux
  // after the shared fill (apply_named_aux_bc). Empty by default -> bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  std::vector<CoupledSourceSpec>
      coupled_sources_;  // registered coupled sources (applied after transport)
  // UNION-TAGS REGRID (capstone Phase 2, C.6). regrid_every_ == 0 -> FROZEN hierarchy (default,
  // bit-identical). block_tag_: PER-BLOCK tag predicate (D1; same size as blocks_, empty = this block
  // tags nothing on its side). phi_tag_: phi tag predicate on |grad phi| (D4; empty = phi does not
  // contribute to the union).
  std::vector<TagPredicate> block_tag_;
  TagPredicate phi_tag_;
  int regrid_every_ = 0;
  int regrid_grow_ = 2;
  int regrid_margin_ = 2;
  int aux_ncomp_ = kAuxBaseComps;
  int nlev_ = 0;
  int macro_step_ = 0;
  mutable int solve_count_ = 0;
  int regrid_count_ = 0;
};

}  // namespace adc
