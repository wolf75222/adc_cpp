#pragma once

#include <pops/coupling/schur/amr/amr_condensed_schur_source_stepper.hpp>  // GLOBAL condensed source stage (amr-schur)
#include <pops/coupling/amr/amr_coupler_mp.hpp>                      // AmrCouplerMP, AmrLevelMP
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/layout/refinement.hpp>  // coarsen_index
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>  // SourceFreeModel (explicit IMEX half-step, transport only)
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // backward_euler_source + ImplicitMask (stiff IMEX source)
#include <pops/parallel/comm.hpp>                   // n_ranks
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntimeBlock (type-erased multi-block registry)
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>  // detail::make_poisson_rhs (rhs += elliptic_rhs(U))
#include <pops/runtime/config/dispatch_tags.hpp>  // UNIQUE tag registry (validate_limiter/riemann)

#include <algorithm>  // std::find, std::sort (resolving the partial IMEX mask of a compiled block)
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model on the AmrSystem side: wires a COMPILED model (a CompositeModel, generated
///        by the DSL or hand-written, known at COMPILE time) as a block of an AMR hierarchy,
///        EXACTLY the production path of AmrSystem::add_block but WITHOUT going through the ModelSpec
///        dispatch (the model is already a concrete type). A SINGLE compiled block -> historical
///        mono-block AmrCouplerMP path (bit-identical); SEVERAL compiled blocks or a MIX of compiled +
///        native (capstone v, multi-block production DSL) -> AmrRuntime runtime engine on the shared
///        hierarchy, the compiled block being materialized there as a type-erased AmrRuntimeBlock.
///
/// Refined counterpart of add_compiled_model(System&, ...) (dsl_block.hpp). The AMR coupler build
/// machinery (AmrCouplerMP<Model> + conservative reflux + regrid) is instantiated HERE, from the CALLING
/// translation unit, on the concrete Model type -- like block_builder.hpp for the flat System.
/// The type-erased closures enter AmrSystem through AmrSystem::set_compiled_block (a non-template method)
/// which freezes TWO builders: the mono-block one (detail::build_amr_compiled / dispatch_amr_compiled,
/// SHARED with the native ModelSpec path of add_block once the type is resolved by detail::dispatch_model)
/// AND the multi-block one (detail::dispatch_amr_block, also SHARED with add_block in native multi-block).

namespace pops {

/// Bundle (limiter, Riemann flux) expected by AmrCouplerMP::step<Disc>. Unique definition: the
/// native path of amr_system.cpp goes through this same header (no more DiscLF duplicated on the .cpp side).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

// Projection ponctuelle post-pas appliquee PAR NIVEAU (ADC-177) : miroir de PointwiseProject
// (block_builder.hpp) mais sur la pile de niveaux AMR ; aux = lev.aux (cable par AmrRuntime).
// Defini en tete du namespace : utilise par build_amr_compiled (mono-bloc) ET build_amr_block
// (multi-bloc natif), tous deux situes plus bas (la recherche qualifiee detail:: exige la
// declaration AVANT le point d'usage). No-op (else) si le modele ne declare pas m.project.
template <class Model>
void apply_pointwise_project_amr(const Model& m, std::vector<AmrLevelMP>& levels) {
  if constexpr (HasPointwiseProjection<Model>) {
    for (auto& lev : levels) {
      MultiFab& U = lev.U;
      const MultiFab& a = *lev.aux;
      for (int li = 0; li < U.local_size(); ++li)
        for_each_cell(U.box(li),
                      ProjectCellKernel<Model>{m, U.fab(li).array(), U.fab(li).const_array(),
                                               a.fab(li).const_array()});
    }
  } else {
    (void)m;
    (void)levels;
  }
}

/// Fills the COARSE B_z field (component 0, n*n row-major in GLOBAL indices) from @p field.
/// Scalar counterpart of coupler_write_coarse (identical box traversal, replicated mono-box AND
/// distributed multi-box): B_z is required by the Schur-condensed source stage (Lorentz term).
inline void amr_write_coarse_bz(MultiFab& bz, const std::vector<double>& field, int n) {
  if (static_cast<int>(field.size()) != n * n)
    throw std::runtime_error(
        "AMR amr-schur: B_z field of size != n*n (call set_magnetic_field before the first step)");
  device_fence();
  for (int li = 0; li < bz.local_size(); ++li) {
    Array4 b = bz.fab(li).array();
    const Box2D v = bz.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        b(i, j, 0) = field[static_cast<std::size_t>(j) * n + i];
  }
}

/// A GLOBAL condensed source STAGE on the mono-block coupler hierarchy. Seeds the warm-start phi^n
/// (= aux0 component 0, i.e. the coarse Poisson solve of the last update()), then runs the
/// condensed stage (AmrCondensedSchurSourceStepper) which assembles/solves its OWN condensed operator on
/// the coarse grid and reconstructs the velocity (rho frozen, mom/E updated). In mono-level (no fine patch)
/// this is bit-for-bit the uniform stage #126.
template <class Coupler>
void amr_schur_source(Coupler& cpl, AmrCondensedSchurSourceStepper& schur, MultiFab& bz_coarse,
                      MultiFab& phi_coarse, double theta, double dt) {
  device_fence();
  for (int li = 0; li < phi_coarse.local_size(); ++li)
    for_each_cell(phi_coarse.box(li),
                  CopyComp0Kernel{phi_coarse.fab(li).array(), cpl.aux0().fab(li).const_array()});
  schur.step(cpl.levels(), phi_coarse, bz_coarse, /*c_bz=*/0, static_cast<Real>(theta),
             static_cast<Real>(dt));
}

/// Builds the AMR coupler for a composite Model + concrete (Limiter, Flux) and fills the type-erased
/// hooks. Two levels: coarse + one central seed fine patch, reshaped by the regrid. This is the header
/// counterpart of AmrSystem::Impl::build, instantiated from the calling TU on the Model type. The
/// coarse helpers (layout, write/read/inject) are SHARED with the native path via
/// amr_coupler_mp.hpp (detail::coupler_*), so replicated and distributed follow exactly the same logic.
template <class Model, class Limiter, class Flux>
AmrCompiledHooks build_amr_compiled(const Model& model, const AmrBuildParams& bp) {
  using Coupler = AmrCouplerMP<Model>;
  const int nc = Model::n_vars;
  const Geometry g{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  // Level 0 (coarse): layout decided by the ownership policy (replicated mono-box by default,
  // distributed multi-box if bp.distribute_coarse). When replicated, dmap = my_rank() everywhere (the box
  // lives on each rank; a round-robin would place it on rank 0 only -> out-of-bounds fab elsewhere,
  // segfault under np>1). The fine seed (allocated below ONLY when refinement is configured) starts on the
  // SAME dmap as the coarse; the initial regrid REBUILDS it then REDISTRIBUTES round-robin
  // (DistributionMapping(nfine, n_ranks())) -> multi-GPU distribution of the fine patches. When distributed,
  // the coarse is distributed TOO (AMR strong-scaling).
  const auto [bac, dm] = coupler_make_coarse_layout(bp.n, bp.distribute_coarse, bp.coarse_max_grid);
  const int ng = Limiter::n_ghost;  // limiter stencil (1 NoSlope, 2 MUSCL): scheme parity
  MultiFab Uc(bac, dm, nc, ng);
  Uc.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  // Level 1 (central seed fine patch, reshaped by the regrid): allocated ONLY on the explicit/imex path
  // AND when refinement is actually configured (set_refinement was called -> refine_threshold below its
  // 1e30 "no refinement" sentinel). Two reasons to gate it:
  //   - amr-schur (Step 2/3) runs MONO-LEVEL: the condensed source stage does not yet carry the multi-level
  //     case (cf. AmrCondensedSchurSourceStepper, guard on the number of fine patches), so a seed would trip
  //     that guard at the first step. Multi-level amr-schur (fine reconstruction + cascade + composite
  //     Schur/Poisson) is Step 4.
  //   - NO refinement (ADC-324): with the 1e30 sentinel the build-time regrid below (cpl->regrid(crit)) tags
  //     nothing and amr_regrid_finest is a deliberate no-op on zero tags, so the seed would NEVER be reshaped
  //     or removed -- it would persist as a SINGLE un-chopped fine box on the coarse dmap (box 0 -> rank 0),
  //     dead weight that starves MPI strong-scaling (rank 0 carries its coarse boxes PLUS the whole fine
  //     patch). Gating on refine_threshold keeps the no-refinement hierarchy MONO-LEVEL (n_patches()==0, like
  //     the amr-schur path), so the coarse distributes cleanly. When refinement IS configured the seed is
  //     allocated and the first build regrid chops + distributes it round-robin exactly as before (UNCHANGED).
  if (!bp.schur && bp.refine_threshold < 1e30) {
    const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
    Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    BoxArray baf(std::vector<Box2D>{fb});
    MultiFab Uf(baf, dm, nc, ng);
    Uf.set_val(Real(0));
    levels.push_back({std::move(Uf), nullptr, dxf, dxf});
  }

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson_bc, std::move(levels), bp.wall,
                                       !bp.distribute_coarse);
  // Coarse seed: COMPLETE conservative state (preferred, set_conservative_state) otherwise density
  // only (historical). coupler_inject_coarse_to_fine_mb prolongs ALL components (loop k<nc), so the
  // momentum of the seed propagates freely to the fine levels -- no change of prolongation.
  // has_state==false -> bit-identical density path (NO-DEFAULT-CHANGE).
  if (bp.has_state)
    coupler_write_coarse_state(cpl->coarse(), bp.state, bp.n, nc);
  else if (bp.has_density)
    coupler_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k)
    coupler_inject_coarse_to_fine_mb(cpl->coarse(), Lv[k].U, !bp.distribute_coarse);

  const double thr = bp.refine_threshold;
  auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
  if (cpl->levels().size() > 1)
    cpl->regrid(crit);  // no regrid on a mono-level hierarchy (amr-schur)
  // model-NAMED aux (ADC-291): seed the static named fields onto the coupler's shared aux BEFORE the
  // first update/step (like density/B_z seeding). The coupler re-applies them in compute_aux each
  // update, so they persist across regrid and reach every level via the aux injection. Empty -> no-op.
  for (const auto& kv : bp.named_aux)
    cpl->set_named_aux(kv.first, std::vector<Real>(kv.second.begin(), kv.second.end()));
  // ADC-369: per-field aux halo policies (compute_aux applies them after the shared fill).
  for (const auto& kv : bp.named_aux_bc)
    cpl->set_named_aux_bc(kv.first, kv.second);
  cpl->update();

  AmrCompiledHooks h;
  h.coupler_holder = cpl;  // lifetime: the closures capture cpl (shared_ptr)
  const int sub = bp.substeps;
  const bool rprim = bp.recon_prim;
  const bool imex = bp.imex;  // implicit stiff source (backward_euler) rather than forward Euler
  const int regrid_every = bp.regrid_every;
  // NEWTON OPTIONS of the mono-block IMEX source (wave 3): threaded to cpl->step -> advance_amr ->
  // backward_euler_source. DEFAULT {} (newton_options not set) = historical constants (2 iters) ->
  // bit-identical path (2a). Captured BY VALUE (POD) in the h.step closure.
  const NewtonOptions nopts = bp.newton_options;
  // TIME METHOD mono-block: integer of the flat ABI (bp.time_method) -> AmrTimeMethod, threaded to
  // cpl->step -> advance_amr. 0 (default / older .so loader) = historical kEuler, bit-identical.
  const AmrTimeMethod tmethod =
      bp.time_method == 1 ? AmrTimeMethod::kSsprk3 : AmrTimeMethod::kEuler;
  // Zhang-Shu positivity floor (ADC-259): threaded to cpl->step / advance_transport -> advance_amr ->
  // compute_face_fluxes + C/F ghost clamp. bp.pos_floor == 0 (default / older flat-ABI .so loader that
  // never sets this append-only field) -> inactive, bit-identical historical path.
  const Real pf = static_cast<Real>(bp.pos_floor);
  auto step_state = std::make_shared<int>(0);  // step counter shared by the closure
  if (bp.schur) {
    // amr-schur PATH: GLOBAL condensed source stage (electrostatic/Lorentz) instead of the LOCAL
    // explicit/imex source. The stage is built on the COARSE grid by COMPOSING the uniform stage #126
    // (Density/MomentumX/MomentumY roles of the Model -> clear error HERE if missing). Coarse B_z required
    // (set_magnetic_field). The model must be SOURCE-FREE (NoSource source brick): advance_transport
    // then runs NO source (the source is the condensed stage alone), mirror of the uniform path where the
    // block is added with its transport only (time.hyperbolic) + the separate condensed source stage.
    //
    // WARNING: OPTION A = INTERMEDIATE. The condensed stage solves the elliptic on the COARSE grid (like the
    // AMR Poisson compute_aux/solve_fields), then grad phi is injected (piecewise constant) to the fines: the
    // fine patches refine the TRANSPORT but NOT the elliptic coupling. For a FAITHFUL paper/AMR reproduction
    // a multi-level COMPOSITE Schur/Poisson will be needed (condensed elliptic solved at the patch
    // resolution, composite MG crossing the levels) -- infrastructure absent today (GeometricMG
    // coarsens ONE grid, != AMR hierarchy). This is the fidelity lock, to do AFTER the mono-level parity.
    // RESOLUTION of the field descriptors (wave 3 audit, parity with System::set_source_stage):
    // "" = canonical role (historical, bit-identical); otherwise stable ROLE name then block VARIABLE name.
    // Failure = explicit error at build (never a silent ignore).
    const VariableSet schur_vs = Model::conservative_vars();
    auto resolve_schur = [&schur_vs](const std::string& spec, VariableRole canonical,
                                     const char* label) -> int {
      if (spec.empty()) {
        const int idx = schur_vs.index_of(canonical);
        if (idx < 0)
          throw std::runtime_error(
              std::string("AmrSystem::set_source_stage: the block does not expose "
                          "the role ") +
              label + " (declare the roles, or pass an explicit descriptor)");
        return idx;
      }
      const VariableRole r = role_from_name(spec);
      if (r != VariableRole::Custom) {
        const int idx = schur_vs.index_of(r);
        if (idx < 0)
          throw std::runtime_error("AmrSystem::set_source_stage: role '" + spec + "' missing (" +
                                   label + ")");
        return idx;
      }
      for (std::size_t i = 0; i < schur_vs.names.size(); ++i)
        if (schur_vs.names[i] == spec)
          return static_cast<int>(i);
      throw std::runtime_error("AmrSystem::set_source_stage: '" + spec +
                               "' is neither a stable role nor a block variable (" + label + ")");
    };
    const int sc_rho = resolve_schur(bp.schur_density, VariableRole::Density, "Density");
    const int sc_mx = resolve_schur(bp.schur_momentum_x, VariableRole::MomentumX, "MomentumX");
    const int sc_my = resolve_schur(bp.schur_momentum_y, VariableRole::MomentumY, "MomentumY");
    const int sc_E = (bp.schur_energy == "none")
                         ? -1
                         : (bp.schur_energy.empty()
                                ? schur_vs.index_of(VariableRole::Energy)
                                : resolve_schur(bp.schur_energy, VariableRole::Energy, "Energy"));
    auto schur = std::make_shared<AmrCondensedSchurSourceStepper>(
        schur_vs, sc_rho, sc_mx, sc_my, sc_E, g, bac, bp.poisson_bc,
        static_cast<Real>(bp.schur_alpha));
    if (bp.schur_krylov_tol > 0.0 || bp.schur_krylov_max_iters > 0)
      schur->set_krylov(
          bp.schur_krylov_tol > 0.0 ? static_cast<Real>(bp.schur_krylov_tol) : Real(1e-10),
          bp.schur_krylov_max_iters > 0 ? bp.schur_krylov_max_iters : 400);
    auto bz_coarse = std::make_shared<MultiFab>(bac, dm, 1, 1);
    amr_write_coarse_bz(*bz_coarse, bp.bz_field, bp.n);
    auto phi_coarse = std::make_shared<MultiFab>(bac, dm, 1, 1);
    phi_coarse->set_val(Real(0));
    const double theta = bp.schur_theta;
    const bool strang = bp.schur_strang;
    h.step = [cpl, crit, sub, rprim, regrid_every, step_state, schur, bz_coarse, phi_coarse, theta,
              strang, model, pf](double dt) {
      // amr-schur Step 2/3: MONO-LEVEL hierarchy (the condensed stage does not carry the multi-level case).
      // So we do NOT regrid (a regrid would create a fine patch -> multi-level guard of the stage). The
      // amr-schur regrid will come with the composite Schur/Poisson (Step 4). cf. levels().size() > 1.
      if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0 &&
          cpl->levels().size() > 1)
        cpl->regrid(crit);
      const double h2 = dt / sub;
      for (int s = 0; s < sub; ++s) {
        if (strang) {
          // STRANG (2nd order): H(dt/2); S(dt); H(dt/2), with update() (= sync_down + coarse Poisson
          // + grad inject, the AMR counterpart of solve_fields) RE-SOLVED BEFORE each stage that
          // consumes phi -- exactly SystemStepper::step_strang (3 solves: head, pre-source, post-source).
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(Real(0.5) * h2, rprim, pf);
          cpl->update();
          amr_schur_source(*cpl, *schur, *bz_coarse, *phi_coarse, theta, h2);
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(Real(0.5) * h2, rprim, pf);
        } else {
          // LIE (Godunov, 1st order): H(dt); S(dt). A single update() at the head (the source stage reads
          // the head phi), mirror of SystemStepper::step Lie (a single solve_fields, transport, source).
          cpl->update();
          cpl->template advance_transport<AmrDiscLF<Limiter, Flux>>(h2, rprim, pf);
          amr_schur_source(*cpl, *schur, *bz_coarse, *phi_coarse, theta, h2);
        }
      }
      // PROJECTION PONCTUELLE post-pas (ADC-177) PAR NIVEAU, APRES transport + source condensee de
      // tous les substeps. No-op si le modele ne declare pas m.project (HasPointwiseProjection false).
      detail::apply_pointwise_project_amr(model, cpl->levels());
      ++*step_state;
    };
  } else {
    h.step = [cpl, crit, sub, rprim, imex, regrid_every, step_state, nopts, tmethod, model,
              pf](double dt) {
      if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0)
        cpl->regrid(crit);
      const double h2 = dt / sub;
      // NEWTON OPTIONS threaded to the coupler (mono-block): nopts={} by default => iters=2 historical,
      // bit-identical; non-default nopts (set_density + pops.IMEX(newton_*)) drives the local Newton.
      // tmethod (kEuler default) selects SSPRK3 if requested (time='ssprk3'); kEuler bit-identical.
      for (int s = 0; s < sub; ++s)
        cpl->template step<AmrDiscLF<Limiter, Flux>>(h2, rprim, imex, nopts, tmethod, pf);
      // PROJECTION PONCTUELLE post-pas (ADC-177) PAR NIVEAU, APRES transport + source de tous les
      // substeps. No-op si le modele ne declare pas m.project (HasPointwiseProjection false).
      detail::apply_pointwise_project_amr(model, cpl->levels());
      ++*step_state;
    };
  }
  // RESTORATION of the CADENCE PHASE (IO v1, parity with System::set_clock): AmrSystem::set_clock sets
  // the macro-step counter of the mono-block (the regrid cadence reads *step_state) on restart. Shares the
  // SAME step_state as the step closure above -> the regrid phase resumes exactly. Without the call,
  // *step_state stays at 0 (default, bit-identical).
  h.set_macro_step = [step_state](int s) { *step_state = s; };
  // CFL SPEED: lambda* (HasStabilitySpeed trait) if declared, otherwise max_wave_speed of the coupler
  // (historical fallback, bit-identical) -- SAME policy as System/make_max_speed, evaluated on the
  // COARSE grid (the AMR mono-block CFL lives at the coarse step).
  if constexpr (HasStabilitySpeed<Model>) {
    h.max_speed = [cpl, model] {
      return static_cast<double>(max_stability_speed_mf(model, cpl->coarse(), cpl->aux0()));
    };
  } else {
    h.max_speed = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
  }
  // OPTIONAL STEP BOUNDS (AMR mono-block StabilityPolicy): same reductions as System,
  // hooks left EMPTY without the trait (AmrSystem::step_cfl then keeps the historical formula).
  if constexpr (HasSourceFrequency<Model>) {
    h.source_frequency = [cpl, model] {
      return static_cast<double>(max_source_frequency_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    h.stability_dt = [cpl, model] {
      return static_cast<double>(min_stability_dt_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  h.mass = [cpl] { return static_cast<double>(cpl->mass()); };
  h.n_patches = [cpl] {
    auto& L = cpl->levels();
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  };
  // Index-space signatures of the fine patches (mono-block counterpart of AmrRuntime::patch_boxes).
  // Captures the SAME cpl as the other hooks (no new lifetime concern), reads the already materialized
  // BoxArray -> query between steps, zero cost on the hot path (h.step untouched).
  h.patch_boxes = [cpl] {
    auto& L = cpl->levels();
    std::vector<pops::PatchBox> out;
    for (std::size_t k = 1; k < L.size(); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const pops::Box2D& b : bxs)
        out.push_back(pops::PatchBox{static_cast<int>(k), b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  };
  // Coarse-level (base) box counts (ADC-319, MPI ownership diagnostic): per-rank OWNED fabs of level 0
  // (local_size()) and the GLOBAL base box count (box_array().size()). Same cpl capture as the other
  // hooks (no new lifetime concern); a query between steps, zero cost on the hot path. distribute_coarse
  // -> local < total per rank (distributed coarse transport); replicated/single-box -> local == total.
  h.coarse_local_boxes = [cpl] { return cpl->coarse().local_size(); };
  h.coarse_total_boxes = [cpl] { return cpl->coarse().box_array().size(); };
  // AMR CHECKPOINT / RESTART single-rank (ADC-65): COMPLETE conservative state per level + phi
  // (warm-start) + imposing the saved fine hierarchy. Capture the SAME cpl (shared_ptr) as the
  // other hooks (no new lifetime concern). Single-rank: the coupler accessors loop over local_size()
  // (no gather) -- the facade rejects np>1 / multi-block upstream. These hooks are QUERIES/SETTERS
  // between steps: zero cost on the hot path (h.step untouched).
  h.n_levels = [cpl] { return cpl->nlev(); };
  h.n_vars = [] { return Model::n_vars; };
  h.level_state = [cpl](int k) { return cpl->level_state(k); };
  h.set_level_state = [cpl](int k, const std::vector<double>& s) { cpl->set_level_state(k, s); };
  h.level_potential = [cpl](int k) { return cpl->level_potential(k); };
  h.set_level_potential = [cpl](int k, const std::vector<double>& p) {
    cpl->set_level_potential(k, p);
  };
  h.set_hierarchy = [cpl](const std::vector<pops::PatchBox>& boxes) {
    // Mono-block: all patches live at level 1 -> we filter level == 1 and convert to Box2D
    // (INCLUSIVE corners, fine-level index space), then impose this BoxArray on the coupler.
    std::vector<pops::Box2D> fb;
    for (const pops::PatchBox& b : boxes)
      if (b.level == 1)
        fb.push_back(pops::Box2D{{b.ilo, b.jlo}, {b.ihi, b.jhi}});
    cpl->set_hierarchy(fb);
  };
  const int nn = bp.n;
  const bool repl = !bp.distribute_coarse;
  h.density = [cpl, nn, repl] { return coupler_read_coarse(cpl->coarse(), nn, repl); };
  // Coarse phi: we refresh (update() = sync_down + compute_aux, hence coarse Poisson solve)
  // then read aux0 component 0. Counterpart of System::potential() which calls ensure_elliptic: the
  // value is current even if no step has run yet. update() is already called at each step,
  // so the overhead exists only on a call outside the loop (diagnostic).
  h.potential = [cpl, nn, repl] {
    cpl->update();
    return coupler_read_coarse_phi(cpl->aux0(), nn, repl);
  };
  return h;
}

/// SHARED layout of a multi-block AMR hierarchy (PR1 capstone), frozen at construction. All
/// blocks allocate their levels on EXACTLY this layout (same BoxArray + DistributionMapping +
/// dx/dy per level) -> same_layout_or_throw passes by construction. PR1: coarse + ONE central FIXED
/// fine patch (placement by union of tags is a later PR). We expose the BoxArrays /
/// dmaps / dx/dy per level, the coarse grid (Geometry + ba) for the Poisson, and the ownership
/// policy. build_amr_block allocates the block on top of it.
struct SharedAmrLayout {
  Geometry geom;                         // geometry of the coarse level (Poisson)
  BoxArray ba_coarse;                    // BoxArray of the coarse grid
  DistributionMapping dm_coarse;         // DistributionMapping of the coarse grid
  std::vector<BoxArray> ba;              // [level] shared BoxArray (coarse + fines)
  std::vector<DistributionMapping> dm;   // [level] shared DistributionMapping
  std::vector<Real> dx, dy;              // [level] mesh spacing
  bool replicated_coarse = true;         // ownership of level 0
  BCRec poisson_bc;                      // BC of the coarse Poisson
  std::function<bool(Real, Real)> wall;  // conducting-wall predicate (empty = none)
  int n = 128;                           // coarse cells per direction
  Periodicity base_per{true, true};      // periodicity of the base domain

  int nlev() const { return static_cast<int>(ba.size()); }
};

/// Builds the SHARED layout (PR1): coarse (per the ownership policy) + ONE central FIXED
/// fine patch (the seed of build_amr_compiled, BEFORE its regrid). Identical to the geometry of the
/// mono-block path, but WITHOUT the initial regrid (multi-block PR1 = frozen hierarchy). All blocks
/// then settle onto it via build_amr_block.
inline SharedAmrLayout make_shared_amr_layout(const AmrBuildParams& bp) {
  SharedAmrLayout S;
  S.geom = Geometry{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  S.n = bp.n;
  S.replicated_coarse = !bp.distribute_coarse;
  S.poisson_bc = bp.poisson_bc;
  S.wall = bp.wall;
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  const auto [bac, dmc] =
      detail::coupler_make_coarse_layout(bp.n, bp.distribute_coarse, bp.coarse_max_grid);
  S.ba_coarse = bac;
  S.dm_coarse = dmc;
  // Central FIXED fine patch: same signatures as build_amr_compiled (coarse cells
  // [n/4 .. 3n/4-1]^2, refined x2). DISTRIBUTED round-robin DistributionMapping(nfine, n_ranks()),
  // EXACTLY like the regrid of the mono-block path (amr_regrid_finest): the fine patches are
  // distributed over the ranks (one per GPU). This is ESSENTIAL under MPI: on the REPLICATED coarse,
  // if the fine were placed on the same replicated dmap ({my_rank()}), EACH rank would hold a copy
  // of the fine box and the reflux (all_reduce_sum_inplace of the flux registers) would sum the SAME
  // contribution n_ranks() times -> mass over-counted (grows with np). In serial (np=1) the round-robin
  // dmap places the box on rank 0, identical to {my_rank()}: bit-identical.
  const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
  const Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  BoxArray baf(std::vector<Box2D>{fb});
  DistributionMapping dmf(baf.size(),
                          n_ranks());  // fine distributed round-robin (one patch per rank)
  S.ba = {bac, baf};
  S.dm = {dmc, dmf};
  S.dx = {dxc, dxf};
  S.dy = {dxc, dxf};
  return S;
}

/// Builds ONE type-erased AMR block (AmrRuntimeBlock) on the SHARED layout @p S, for a composite
/// Model + concrete (Limiter, Flux). Multi-block counterpart of build_amr_compiled: allocates the level
/// stack of the block on the SAME BoxArray/dmap as all the others (guarantees same_layout_or_throw),
/// sets the initial density (component 0) + coarse->fine injection, and CAPTURES the concrete scheme
/// in the closures (advance via advance_amr<Limiter, Flux>, add_elliptic_rhs via PoissonRhs).
/// The kernel stays COMPILED; only the block list is type-erased (AMR analog of make_block /
/// PoissonRhs on the flat System side). @p density (empty = coarse at zero), @p substeps sub-steps of the
/// block, @p stride hold-then-catch-up cadence of the block (1 = each macro-step). substeps and stride are
/// carried by AmrRuntime::step (the advance closure does just ONE advance_amr): they thus do NOT touch
/// the scheme capture, only the substeps/stride fields of the AmrRuntimeBlock.
///
/// TIME TREATMENT (capstone vii): @p imex selects the SOURCE treatment. We populate
/// TWO distinct closures set on the AmrRuntimeBlock and AmrRuntime::step chooses (b.imex):
///   - advance: AMR transport + EXPLICIT source (forward Euler) -- historical path unchanged;
///   - imex_advance: SOURCE-FREE AMR transport + stiff IMPLICIT source backward_euler_source per
///     level (mask @p implicit_components for partial IMEX) + cascade. The SEMANTICS of the splitting
///     mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper), and
///     AT substeps=1 is IDENTICAL to it. This closure does ONE Lie step; AmrRuntime::step calls it
///     substeps times (on the effective step / substeps), so for substeps>1 the runtime SUB-CYCLES the
///     IMEX splitting where compile-time applies it once on the effective step. ASSUMED divergence
///     and sound (cf. IMEX SEMANTICS UNDER substeps in amr_runtime.hpp).
/// @p implicit_components: indices of the components treated IMPLICITLY (partial IMEX, carried by the
/// BLOCK, takes priority over the model default); EMPTY (default) -> inactive mask -> full backward-Euler
/// (all components implicit), bit-identical behavior to IMEX without a mask. Ignored if imex==false.
template <class Model, class Limiter, class Flux>
AmrRuntimeBlock build_amr_block(
    const Model& model, const SharedAmrLayout& S, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps,
    bool recon_prim, bool imex, int stride = 1, const std::vector<int>& implicit_components = {},
    const NewtonOptions& nopts = {}, const std::vector<double>* state = nullptr,
    bool newton_diagnostics = false, AmrTimeMethod time_method = AmrTimeMethod::kEuler,
    double pos_floor = 0.0) {
  const int nc = Model::n_vars;
  const int ng = Limiter::n_ghost;  // limiter stencil (scheme parity, like build_amr_compiled)
  const int nlev = S.nlev();
  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->reserve(nlev);
  for (int k = 0; k < nlev; ++k) {
    MultiFab U(S.ba[k], S.dm[k], nc, ng);
    U.set_val(Real(0));
    levels->push_back(AmrLevelMP{std::move(U), nullptr, S.dx[k], S.dy[k]});
  }
  // Coarse seed + piecewise-constant injection to the fines, exactly like
  // build_amr_compiled: COMPLETE CONSERVATIVE STATE (set_conservative_state, wave 3: now
  // wired in multi-block, preferred) otherwise density (component 0, rest at rest) otherwise zero.
  if (state && !state->empty())
    detail::coupler_write_coarse_state((*levels)[0].U, *state, S.n, nc);
  else if (has_density)
    detail::coupler_write_coarse((*levels)[0].U, density, S.n, nc, gamma);
  for (int k = 1; k < nlev; ++k)
    detail::coupler_inject_coarse_to_fine_mb((*levels)[0].U, (*levels)[k].U, S.replicated_coarse);

  AmrRuntimeBlock b;
  b.name = name;
  b.ncomp = nc;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.imex = imex;  // time treatment of the block: selects advance vs imex_advance in step()
  b.aux_ncomp = aux_comps<Model>();  // aux width READ by the model (B_z/T_e -> > kAuxBaseComps)
  b.cons_vars =
      Model::conservative_vars();  // names + ROLES: role resolution -> component of coupled sources
  b.levels = levels;

  const bool rprim = recon_prim;
  // advance: ONE AMR transport sub-step of the block (conservative Berger-Oliger + reflux + average_down)
  // of size dt, with ITS scheme (Limiter, Flux) on ITS level stack, source in
  // FORWARD EULER (imex=false always here: the IMEX path lives in imex_advance, selected by
  // step()). The sub-step loop (substeps) and the stride cadence are CARRIED by AmrRuntime::step,
  // not by this closure: thus the multirate semantics are in ONE place in the engine (mirror
  // of AmrSystemCoupler::step) and stay disableable / testable there. Implicit FUNCTOR:
  // advance_amr<Limiter, Flux> is a named template function (no cross-TU extended lambda);
  // we capture it in a std::function from THIS TU (device-clean recipe #64/#97).
  // tmethod (kEuler default) selects SSPRK3 (time='ssprk3') for the explicit transport of the block;
  // kEuler -> historical forward Euler, bit-identical. The explicit source stays carried by advance_amr.
  b.advance = [model, rprim, time_method, pos_floor](std::vector<AmrLevelMP>& L, const Box2D& dom,
                                                     Real dt, Periodicity per, bool repl) {
    advance_amr<Limiter, Flux>(model, L, dom, dt, per, repl, rprim, /*imex=*/false, NewtonOptions{},
                               time_method, static_cast<Real>(pos_floor));
  };
  // imex_advance (capstone vii): ONE Lie step [source-free transport; implicit source] whose
  // SEMANTICS mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper),
  // populated ONLY if imex. (1) EXPLICIT transport on the SOURCE-FREE model (SourceFreeModel<Model>:
  // flux/CFL of the model, null source) by the SAME AMR engine (conservative reflux); (2) stiff source
  // IMPLICIT backward_euler_source AT EACH LEVEL (local Newton), with the mask @p implicit_components
  // carried by the BLOCK (partial IMEX); (3) cascade fine -> coarse (mf_average_down_mb) for the coherence
  // of the covered coarse cells. AmrRuntime::step calls this closure substeps times: at
  // substeps=1 this is exactly the compile-time IMEX branch, for substeps>1 the runtime SUB-CYCLES the
  // splitting (assumed decision, cf. IMEX SEMANTICS UNDER substeps in amr_runtime.hpp).
  // We CAPTURE the mask in an ImplicitMask<Model::n_vars> (device-clean POD) once here (the
  // width n_vars is known only at build, the mask is inactive if implicit_components is empty ->
  // full backward-Euler, bit-identical to IMEX without a mask). SourceFreeModel<Model> is a concrete
  // type instantiated IN this TU: its advance_amr<Limiter, Flux> stays compiled (no cross-TU extended
  // lambda), captured in the std::function of identical signature to advance. The reconstruction
  // of the source-free half-step stays CONSERVATIVE (recon_prim=false): SAME choice as AmrSystemCoupler::step
  // (which calls advance_amr on SourceFreeModel with the default), and SourceFreeModel does not expose
  // the primitive variables anyway (cf. its header). The EXPLICIT block, for its part, keeps recon_prim=rprim.
  if (imex) {
    ImplicitMask<Model::n_vars> mask;
    for (int c : implicit_components)
      if (c >= 0 && c < Model::n_vars) {
        mask.active = true;
        mask.flag[c] = true;
      }
    // NEWTON DIAGNOSTICS (OPT-IN, wave 3): we allocate the AGGREGATE report of the block in a shared_ptr
    // (STABLE address even after moving the AmrRuntimeBlock into the engine registry) and we
    // capture its raw pointer in the imex_advance closure. newton_diagnostics==false (default) ->
    // nreport=nullptr -> backward_euler_source FAST path, bit-identical. The RESET of the report is the
    // responsibility of AmrRuntime::step (head of the block advance), like System::AdvanceImex.
    std::shared_ptr<NewtonReport> nrep;
    if (newton_diagnostics) {
      nrep = std::make_shared<NewtonReport>();
      b.newton_diagnostics = true;
      b.newton_report = nrep;
    }
    NewtonReport* nreport = nrep.get();  // null without diagnostics; stable address otherwise
    b.imex_advance = [model, mask, nopts, nreport, pos_floor](std::vector<AmrLevelMP>& L,
                                                              const Box2D& dom, Real dt,
                                                              Periodicity per, bool repl) {
      // (1) explicit source-free transport (-div F only), reflux carries the hyperbolic conservation.
      // The Zhang-Shu floor (ADC-259) applies to the source-free TRANSPORT (the half-step that
      // reconstructs faces); the stiff implicit source backward_euler_source below stays unfloored
      // (cell-local, parity with the uniform System IMEX). SourceFreeModel<Model> forwards
      // conservative_vars(), so positivity_comp resolves the SAME Density-role component.
      advance_amr<Limiter, Flux>(SourceFreeModel<Model>{model}, L, dom, dt, per, repl,
                                 /*recon_prim=*/false, /*imex=*/false, NewtonOptions{},
                                 AmrTimeMethod::kEuler, static_cast<Real>(pos_floor));
      // (2) stiff implicit source backward-Euler PER LEVEL (local Newton, block mask). The report
      // nreport (null without diagnostics) AGGREGATES over the levels: backward_euler_source does its own
      // max/sum + MPI all_reduce into *nreport (no reset here -> it also accumulates over the sub-steps,
      // step() having reset at the head of the advance). nreport==nullptr -> fast bit-identical path.
      const int nlev_l = static_cast<int>(L.size());
      for (int k = 0; k < nlev_l; ++k)
        backward_euler_source<Model>(model, *L[k].aux, L[k].U, dt, nopts, mask, nreport);
      // (3) COVERAGE INVARIANT (cf. AmrImplicitSourceStepper): the implicit source was solved
      // level by level, so a COVERED coarse cell would carry a phantom coarse source
      // instead of the 2x2 average of its children. Cascade fine -> coarse for the coherence (the mass,
      // sum of the coarse grid alone, then does not count the patch source twice). Mono-level: empty loop
      // -> bit-identical. The source remaining CELL-LOCAL (not a face flux), it does NOT enter
      // the reflux registers: conservation at the coarse-fine interfaces stays intact.
      for (int k = nlev_l - 1; k >= 1; --k)
        mf_average_down_mb(L[k].U, L[k - 1].U);
    };
  }
  // PROJECTION PONCTUELLE post-pas (ADC-177) : cablee SEULEMENT si le modele declare m.project
  // (HasPointwiseProjection). AmrRuntime::step l'applique PAR NIVEAU a la FIN de l'avance du bloc
  // (substeps + reflux/cascade faits). Vide sinon -> trajectoire bit-identique. Capture le `model`
  // concret comme advance / imex_advance (foncteur device-clean, pas de lambda etendue cross-TU).
  if constexpr (HasPointwiseProjection<Model>)
    b.project_per_level = [model](std::vector<AmrLevelMP>& L) {
      detail::apply_pointwise_project_amr(model, L);
    };
  // Contribution of the block to the SUMMED Poisson RHS: rhs += elliptic_rhs(U) on the coarse grid (pure
  // host loop). SAME functor as the flat System (make_poisson_rhs -> detail::PoissonRhs) -> each
  // block accumulates (+=) into the SAME cells of the shared coarse grid (per-cell co-location).
  b.add_elliptic_rhs = make_poisson_rhs(model);
  // CFL SPEED of the block: SAME policy as System (make_max_speed) -- stability lambda*
  // (HasStabilitySpeed trait) if the model declares it, otherwise max_wave_speed (historical fallback,
  // bit-identical). The Riemann solvers always read max_wave_speed.
  if constexpr (HasStabilitySpeed<Model>) {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_stability_speed_mf(model, U, aux);
    };
  } else {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_wave_speed_mf(model, U, aux);
    };
  }
  // OPTIONAL STEP BOUNDS (AMR StabilityPolicy): same reductions as System
  // (max_source_frequency_mf / min_stability_dt_mf), evaluated by AmrRuntime::step_cfl on the
  // COARSE grid. Closures left EMPTY when the model does not declare the trait (bit-identical).
  if constexpr (HasSourceFrequency<Model>) {
    b.source_frequency = [model](const MultiFab& U, const MultiFab& aux) {
      return max_source_frequency_mf(model, U, aux);
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    b.stability_dt = [model](const MultiFab& U, const MultiFab& aux) {
      return min_stability_dt_mf(model, U, aux);
    };
  }
  const Geometry g = S.geom;
  const bool repl = S.replicated_coarse;
  b.mass = [levels, g, repl] {
    const MultiFab& U = (*levels)[0].U;
    const Real dV = g.dx() * g.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(U.box(li),
                                    [u, dV] POPS_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return repl ? M : all_reduce_sum(M);
  };
  const int nn = S.n;
  b.density = [levels, nn, repl] { return detail::coupler_read_coarse((*levels)[0].U, nn, repl); };
  b.potential = [nn, repl](const MultiFab& aux0) {
    return detail::coupler_read_coarse_phi(aux0, nn, repl);
  };
  return b;
}

// ADC-359 per-flux branches of dispatch_amr_block, factored so the compressible AMR seam compiles ONE
// flux per TU (build_amr_block_for_flux -> these). Each body is the corresponding `if (riem == "<flux>")`
// branch of dispatch_amr_block VERBATIM (same leaves, same hllc/roe `if constexpr` capability guards, same
// messages); validate_riemann/limiter run in the caller (dispatch_amr_block, or the compressible thin
// dispatcher python/amr_block_compressible.cpp). dispatch_amr_block (below, unchanged) still serves the
// exb/isothermal seam, where the if constexpr guards prune hllc/roe.
template <class Model>
AmrRuntimeBlock dispatch_amr_block_rusanov(
    const Model& m, const std::string& lim, const SharedAmrLayout& S, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps,
    bool recon_prim, bool imex, int stride, const std::vector<int>& implicit_components,
    const NewtonOptions& nopts, const std::vector<double>* state, bool newton_diagnostics,
    AmrTimeMethod time_method, double pos_floor) {
  if (lim == "none")
    return build_amr_block<Model, NoSlope, RusanovFlux>(
        m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
        implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
  if (lim == "minmod")
    return build_amr_block<Model, Minmod, RusanovFlux>(
        m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
        implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
  if (lim == "vanleer")
    return build_amr_block<Model, VanLeer, RusanovFlux>(
        m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
        implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
  if (lim == "weno5")
    return build_amr_block<Model, Weno5, RusanovFlux>(
        m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
        implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "limiteur", lim);
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_hll(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, bool imex,
                                       int stride, const std::vector<int>& implicit_components,
                                       const NewtonOptions& nopts, const std::vector<double>* state,
                                       bool newton_diagnostics, AmrTimeMethod time_method,
                                       double pos_floor) {
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    if (lim == "none")
      return build_amr_block<Model, NoSlope, HLLFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "minmod")
      return build_amr_block<Model, Minmod, HLLFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "vanleer")
      return build_amr_block<Model, VanLeer, HLLFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "weno5")
      return build_amr_block<Model, Weno5, HLLFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'hll' requires signed wave "
        "speeds (model.wave_speeds); this transport -> 'rusanov'");
  }
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_hllc(const Model& m, const std::string& lim,
                                        const SharedAmrLayout& S, const std::string& name,
                                        const std::vector<double>& density, bool has_density,
                                        double gamma, int substeps, bool recon_prim, bool imex,
                                        int stride, const std::vector<int>& implicit_components,
                                        const NewtonOptions& nopts,
                                        const std::vector<double>* state, bool newton_diagnostics,
                                        AmrTimeMethod time_method, double pos_floor) {
  if constexpr (HasHLLCStructure<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_amr_block<Model, NoSlope, HLLCFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "minmod")
      return build_amr_block<Model, Minmod, HLLCFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "vanleer")
      return build_amr_block<Model, VanLeer, HLLCFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "weno5")
      return build_amr_block<Model, Weno5, HLLCFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'hllc' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's HLLC capability (pressure + wave_speeds + contact_speed + "
        "hllc_star_state, cf. HasHLLCStructure); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_roe(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, bool imex,
                                       int stride, const std::vector<int>& implicit_components,
                                       const NewtonOptions& nopts, const std::vector<double>* state,
                                       bool newton_diagnostics, AmrTimeMethod time_method,
                                       double pos_floor) {
  if constexpr (HasRoeDissipation<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_amr_block<Model, NoSlope, RoeFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "minmod")
      return build_amr_block<Model, Minmod, RoeFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "vanleer")
      return build_amr_block<Model, VanLeer, RoeFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    if (lim == "weno5")
      return build_amr_block<Model, Weno5, RoeFlux>(
          m, S, name, density, has_density, gamma, substeps, recon_prim, imex, stride,
          implicit_components, nopts, state, newton_diagnostics, time_method, pos_floor);
    throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'roe' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's Roe capability (roe_dissipation, cf. HasRoeDissipation); "
        "this transport -> 'hll'/'rusanov'");
  }
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> build_amr_block. SAME guards as
/// dispatch_amr_compiled (hllc/roe require the model's Riemann capability HasHLLCStructure /
/// HasRoeDissipation, OR the canonical Euler 2D layout: 4 variables + pressure).
/// Multi-block counterpart of dispatch_amr_compiled. @p implicit_components: partial IMEX mask carried
/// by the block (indices of the implicit components; empty = full backward-Euler), threaded to build_amr_block.
template <class Model>
AmrRuntimeBlock dispatch_amr_block(
    const Model& m, const std::string& lim, const std::string& riem, const SharedAmrLayout& S,
    const std::string& name, const std::vector<double>& density, bool has_density, double gamma,
    int substeps, bool recon_prim, bool imex, int stride = 1,
    const std::vector<int>& implicit_components = {}, const NewtonOptions& nopts = {},
    const std::vector<double>* state = nullptr, bool newton_diagnostics = false,
    AmrTimeMethod time_method = AmrTimeMethod::kEuler, double pos_floor = 0.0) {
  // CENTRALIZED VALIDATION (dispatch_tags.hpp registry) BEFORE the dispatch: same tags accepted /
  // rejected as before, identical messages. The template if/else dispatch that follows is UNCHANGED; the
  // capability guards (hllc/roe: 2D Euler or capability) stay `if constexpr` PER MODEL.
  validate_riemann(riem, /*polar=*/false, "add_block(AmrSystem, multi-block)");
  validate_limiter(lim, "add_block(AmrSystem, multi-block)");
  // ADC-359: delegate to the flux-pinned dispatch_amr_block_<flux> helpers above (factored so the
  // compressible seam compiles one flux per TU). Behavior is unchanged: same leaves, same hllc/roe
  // capability guards, same throws. exb/isothermal route here as before (their guards prune hllc/roe).
  if (riem == "rusanov")
    return dispatch_amr_block_rusanov(m, lim, S, name, density, has_density, gamma, substeps,
                                      recon_prim, imex, stride, implicit_components, nopts, state,
                                      newton_diagnostics, time_method, pos_floor);
  if (riem == "hll")
    return dispatch_amr_block_hll(m, lim, S, name, density, has_density, gamma, substeps,
                                  recon_prim, imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
  if (riem == "hllc")
    return dispatch_amr_block_hllc(m, lim, S, name, density, has_density, gamma, substeps,
                                   recon_prim, imex, stride, implicit_components, nopts, state,
                                   newton_diagnostics, time_method, pos_floor);
  if (riem == "roe")
    return dispatch_amr_block_roe(m, lim, S, name, density, has_density, gamma, substeps,
                                  recon_prim, imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "flux", riem);
}

// ADC-359 per-flux branches of dispatch_amr_compiled, factored so the compressible compiled AMR seam
// compiles ONE flux per TU (build_amr_compiled_for_flux -> these). Each body is the corresponding
// `if (riem == "<flux>")` branch of dispatch_amr_compiled VERBATIM (same leaves, guards, messages);
// validate_* run in the caller. dispatch_amr_compiled (below, unchanged) still serves exb/isothermal.
template <class Model>
AmrCompiledHooks dispatch_amr_compiled_rusanov(const Model& m, const std::string& lim,
                                               const AmrBuildParams& bp) {
  if (lim == "none")
    return build_amr_compiled<Model, NoSlope, RusanovFlux>(m, bp);
  if (lim == "minmod")
    return build_amr_compiled<Model, Minmod, RusanovFlux>(m, bp);
  if (lim == "vanleer")
    return build_amr_compiled<Model, VanLeer, RusanovFlux>(m, bp);
  if (lim == "weno5")
    return build_amr_compiled<Model, Weno5, RusanovFlux>(m, bp);
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_hll(const Model& m, const std::string& lim,
                                           const AmrBuildParams& bp) {
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    if (lim == "none")
      return build_amr_compiled<Model, NoSlope, HLLFlux>(m, bp);
    if (lim == "minmod")
      return build_amr_compiled<Model, Minmod, HLLFlux>(m, bp);
    if (lim == "vanleer")
      return build_amr_compiled<Model, VanLeer, HLLFlux>(m, bp);
    if (lim == "weno5")
      return build_amr_compiled<Model, Weno5, HLLFlux>(m, bp);
    throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'hll' requires signed wave "
        "speeds (model.wave_speeds: declare a primitive 'p'); "
        "this transport -> 'rusanov'");
  }
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_hllc(const Model& m, const std::string& lim,
                                            const AmrBuildParams& bp) {
  if constexpr (HasHLLCStructure<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_amr_compiled<Model, NoSlope, HLLCFlux>(m, bp);
    if (lim == "minmod")
      return build_amr_compiled<Model, Minmod, HLLCFlux>(m, bp);
    if (lim == "vanleer")
      return build_amr_compiled<Model, VanLeer, HLLCFlux>(m, bp);
    if (lim == "weno5")
      return build_amr_compiled<Model, Weno5, HLLCFlux>(m, bp);
    throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'hllc' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's HLLC capability (pressure + wave_speeds + contact_speed + "
        "hllc_star_state, cf. HasHLLCStructure); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_roe(const Model& m, const std::string& lim,
                                           const AmrBuildParams& bp) {
  if constexpr (HasRoeDissipation<Model> ||
                (Model::n_vars == 4 &&
                 requires(const Model mm, typename Model::State s) { mm.pressure(s); })) {
    if (lim == "none")
      return build_amr_compiled<Model, NoSlope, RoeFlux>(m, bp);
    if (lim == "minmod")
      return build_amr_compiled<Model, Minmod, RoeFlux>(m, bp);
    if (lim == "vanleer")
      return build_amr_compiled<Model, VanLeer, RoeFlux>(m, bp);
    if (lim == "weno5")
      return build_amr_compiled<Model, Weno5, RoeFlux>(m, bp);
    throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "limiteur", lim);
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'roe' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's Roe capability (roe_dissipation, cf. HasRoeDissipation); "
        "this transport -> 'hll'/'rusanov'");
  }
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> build_amr_compiled. Same guards as
/// AmrSystem::add_block (hllc/roe require the model's Riemann capability HasHLLCStructure /
/// HasRoeDissipation, OR the canonical Euler 2D layout: 4 variables + pressure).
template <class Model>
AmrCompiledHooks dispatch_amr_compiled(const Model& m, const std::string& lim,
                                       const std::string& riem, const AmrBuildParams& bp) {
  // CENTRALIZED VALIDATION (dispatch_tags.hpp registry) BEFORE the dispatch: same tags accepted /
  // rejected as before. Template if/else dispatch UNCHANGED; per-model hllc/roe capability guards.
  validate_riemann(riem, /*polar=*/false, "add_compiled_model(AmrSystem)");
  validate_limiter(lim, "add_compiled_model(AmrSystem)");
  // ADC-359: delegate to the flux-pinned dispatch_amr_compiled_<flux> helpers above. Behavior unchanged
  // (same leaves, guards, throws); exb/isothermal route here as before (their guards prune hllc/roe).
  if (riem == "rusanov")
    return dispatch_amr_compiled_rusanov(m, lim, bp);
  if (riem == "hll")
    return dispatch_amr_compiled_hll(m, lim, bp);
  if (riem == "hllc")
    return dispatch_amr_compiled_hllc(m, lim, bp);
  if (riem == "roe")
    return dispatch_amr_compiled_roe(m, lim, bp);
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "flux", riem);
}

}  // namespace detail

/// Resolves the partial IMEX MASK (implicit_vars / implicit_roles) of a COMPILED block into indices of
/// conserved components, against the conservative descriptor @p cons of the CONCRETE Model (known here).
/// SAME strict logic as resolve_implicit_components of amr_system.cpp (missing name/role -> error;
/// unique sorted indices) -- replicated here because this header does not depend on the facade .cpp. EMPTY
/// input -> empty -> inactive mask (full backward-Euler). Used by the multi-block runtime builder.
inline std::vector<int> resolve_implicit_components_compiled(
    const std::string& block, const VariableSet& cons, const std::vector<std::string>& names,
    const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end())
      out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) {
        idx = i;
        break;
      }
    if (idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem): implicit_vars: variable '" + nm +
                               "' missing from block '" + block + "'");
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem): implicit_roles: role '" + rn +
                               "' missing from block '" + block + "'");
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Wires @p model (concrete CompositeModel) as an AMR block of @p sys, with the requested scheme. The
/// build is DEFERRED (like add_block): the captured closures are invoked at the first
/// step/mass/density via ensure_built(), after set_refinement / set_poisson / set_density.
///
/// MONO-BLOCK (a single add_compiled_model): historical AmrCouplerMP<Model> path (mono_builder),
/// bit-identical. MULTI-BLOCK (>= 2 blocks, compiled and/or native mixed; capstone v): the block is
/// materialized as a type-erased AmrRuntimeBlock on the layout SHARED by the multi_builder, exactly
/// like native add_block. We freeze BOTH builders here (the facade chooses the routing at ensure_built).
/// @p time: "explicit" (forward Euler source) or "imex" (stiff implicit source via
/// backward_euler_source, explicit transport carried by the reflux). Any other treatment is refused.
/// @p stride: HOLD-THEN-CATCH-UP cadence of the block in multi-block (1 = each macro-step).
/// @p implicit_vars / @p implicit_roles: partial IMEX mask of the block (multi-block; requires time=imex).
/// @p pos_floor: Zhang-Shu positivity floor (ADC-322; 0 = inactive, bit-identical). Stored on the block
///   (mono path reads AmrBuildParams::pos_floor) AND forwarded to the multi-block builder, so the .so
///   floors the Density-role face states like a native add_block.
/// @throws std::runtime_error if the system is already built or if time/recon are out of domain.
template <class Model>
void add_compiled_model(AmrSystem& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4, int substeps = 1,
                        int stride = 1, const std::vector<std::string>& implicit_vars = {},
                        const std::vector<std::string>& implicit_roles = {},
                        double pos_floor = 0.0) {
  if (substeps < 1)
    throw std::runtime_error("add_compiled_model(AmrSystem): substeps >= 1");
  // PROJECTION PONCTUELLE post-pas (ADC-177) : DESORMAIS CABLEE sur AmrSystem. Appliquee PAR NIVEAU
  // a la fin de l'avance du pas (apres le reflux), aussi bien sur le coupleur mono-bloc
  // (build_amr_compiled -> cpl->levels()) que sur le multi-bloc natif (build_amr_block ->
  // AmrRuntime::step -> project_per_level). Cell-local + idempotente : conservation preservee (les
  // flux-registres sont deja regles). No-op si le modele ne declare pas m.project.
  // SSPRK3 IS NOT carried by the COMPILED path: neither the mono_builder nor the multi_builder
  // freezes AmrBuildParams::time_method / passes AmrTimeMethod to dispatch_amr_block (the flat ABI of the
  // .so loader does not marshal the method). EXPLICIT rejection rather than a silent kEuler fallback; an
  // SSPRK3 block must be NATIVE (AmrSystem::add_block / dispatch_amr_block, which threads it).
  if (time == "ssprk3")
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): time='ssprk3' not carried by the "
        "compiled path (.so); use a native block pops.Model(...).");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("add_compiled_model(AmrSystem): time '" + time +
                             "' unknown (explicit|imex)");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem): recon unknown '" + recon +
                             "' (conservative|primitive)");
  const bool recon_prim = (recon == "primitive");
  const bool imex = (time == "imex");
  // (1) MONO-BLOCK builder: captures the concrete Model + the scheme, materializes the AmrCouplerMP at the
  // lazy build (refine/poisson/density parameters frozen at that point). Historical path, untouched.
  auto mono_builder = [model, limiter, riemann, recon_prim, imex](const AmrBuildParams& bp) {
    AmrBuildParams p = bp;
    p.recon_prim = recon_prim;
    p.imex = imex;
    return detail::dispatch_amr_compiled(model, limiter, riemann, p);
  };
  // (2) MULTI-BLOCK builder: captures the SAME concrete Model/scheme, materializes the AmrRuntimeBlock of the
  // block on the SHARED layout (common to all blocks, created once at ensure_built). Resolves ITSELF
  // the partial IMEX mask against cons_vars of the concrete Model (known here), then calls dispatch_amr_block
  // -- EXACTLY the native path of add_block, only the point of type resolution differs (here at
  // the add, there from a ModelSpec at build). FUNCTOR without a cross-TU extended lambda in the kernel:
  // dispatch_amr_block captures advance_amr<Limiter, Flux> (named template function), device-clean
  // recipe #64/#97; the outer lambda only orchestrates (no device kernel in its body).
  auto multi_builder = [model, limiter, riemann](
                           const detail::SharedAmrLayout& S, const std::string& bname,
                           const std::vector<double>& density, bool has_density, double bgamma,
                           int bsub, bool brecon_prim, bool bimex, int bstride,
                           const std::vector<std::string>& ivars,
                           const std::vector<std::string>& iroles, double bpos_floor) {
    const std::vector<int> impl_components =
        bimex
            ? resolve_implicit_components_compiled(bname, Model::conservative_vars(), ivars, iroles)
            : std::vector<int>{};
    // pos_floor (ADC-322): the .so flat ABI now carries the Zhang-Shu floor; forward it to the SAME
    // dispatch_amr_block -> build_amr_block leaf as a native multi-block. The compiled path transports
    // NEITHER Newton options/state/diagnostics NOR SSPRK3 (rejected at the facade / add_compiled_model),
    // so those intermediate arguments stay at their historical defaults (kEuler, no Newton, no state).
    return detail::dispatch_amr_block(
        model, limiter, riemann, S, bname, density, has_density, bgamma, bsub, brecon_prim, bimex,
        bstride, impl_components, NewtonOptions{},
        /*state=*/nullptr, /*newton_diagnostics=*/false, AmrTimeMethod::kEuler, bpos_floor);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(mono_builder),
                         std::move(multi_builder), name, recon_prim, imex, stride, implicit_vars,
                         implicit_roles, pos_floor);
}

}  // namespace pops
