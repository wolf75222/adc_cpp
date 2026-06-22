#include <adc/runtime/amr_system.hpp>

#include <adc/runtime/detail/abi_key.hpp>  // detail::abi_key_string: ABI key (header-only), compared to the loader's
#include <adc/runtime/builders/amr_dsl_block.hpp>  // detail::dispatch_amr_compiled + build_amr_compiled (shared path)
#include <adc/runtime/amr/amr_runtime.hpp>  // AmrRuntime + AmrRuntimeBlock (multi-block runtime engine)
#include <adc/runtime/builders/amr_block_seam.hpp>  // ADC-335: per-transport AMR build seam (build_amr_block/_compiled_<transport>)
#include <adc/runtime/builders/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <adc/runtime/detail/model_registry.hpp>  // unknown_transport_msg: single-source transport rejection (ADC-331)
#include <adc/runtime/detail/wall_predicate.hpp>  // detail::wall_predicate (wall shared System/AmrSystem)
#include <adc/numerics/time/integrators/implicit_stepper.hpp>  // NewtonOptions + validate_newton_options (shared range check)

#include <algorithm>  // std::find, std::sort (partial IMEX mask resolution: sorted unique indices)
#include <cmath>
#include <cstddef>
#include <limits>  // std::numeric_limits (global step bounds: neutralization to +inf before the min)
#include <adc/runtime/detail/dynlib.hpp>  // portable dlopen<->LoadLibraryW layer (ADC-99); <dlfcn.h> on POSIX
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace adc {

// resolve_implicit_components (AMR) moved to amr_block_seam.hpp (adc::detail::
// resolve_implicit_components_amr) so the per-transport seam TUs share one definition; otherwise
// unchanged (AmrSystem-specific error wording preserved verbatim).

struct AmrSystem::Impl {
  AmrSystemConfig cfg;

  // Specification of ONE block (frozen at add_block, materialized at lazy build). The facade holds
  // a REGISTRY BY NAME of blocks (cf. design AMR_MULTIBLOCK_DESIGN.md section 0/3): the single-block
  // path goes through AmrCouplerMP (untouched, bit-identical); two or more blocks go through the
  // multi-block runtime engine AmrRuntime (shared hierarchy, co-located summed Poisson).
  struct BlockSpec {
    std::string name;
    // Native ModelSpec path (composed bricks) OR compiled path (.so / add_compiled_model).
    bool is_compiled = false;
    ModelSpec spec;  // ModelSpec path (is_compiled == false)
    std::string limiter = "minmod", riemann = "rusanov";
    bool recon_prim = false;  // recon == "primitive"
    bool imex = false;        // time == "imex": implicit stiff source
    // Partial IMEX mask CARRIED BY THE BLOCK (cf. System::add_block): conserved components handled
    // implicitly, by NAME (implicit_vars) or by physical ROLE (implicit_roles). We STORE the raw
    // strings here (the concrete Model type -- thus cons_vars -- is only resolved at lazy build, in
    // build_multi via dispatch_model); the names/roles -> indices resolution happens there, against the
    // block's conservative descriptor. Empty (default) -> full backward-Euler (all implicit).
    std::vector<std::string> implicit_vars, implicit_roles;
    int substeps = 1;
    int stride = 1;  // hold-then-catch-up cadence (multi-block; cf. AmrRuntimeBlock)
    double gamma = 1.4;
    // Compiled SINGLE-BLOCK path: type-erasing builder (AmrCompiledHooks of a concrete AmrCouplerMP),
    // invoked at lazy build when the compiled block is ALONE (AmrCouplerMP path, bit-identical).
    std::function<AmrCompiledHooks(const AmrBuildParams&)> compiled_hooks_builder;
    // Compiled MULTI-BLOCK path (capstone v, multi-block production DSL): type-erasing builder that, on
    // the SHARED layout materialized at lazy build (build_multi), produces the AmrRuntimeBlock of the
    // compiled block -- exactly like dispatch_amr_block for a native block, but with the Model/Limiter/
    // Flux CONCRETE types already captured at add (add_compiled_model) instead of a ModelSpec dispatch.
    // The partial IMEX mask (implicit_vars/roles above) is resolved into indices IN this builder (the
    // concrete Model type -- thus cons_vars -- is known there), just as the native path resolves it in
    // build_multi. Empty for a native block (is_compiled == false).
    AmrCompiledBlockBuilder compiled_block_builder;
    // Initial density of the block (component 0), n*n row-major; targeted by set_density(name, rho).
    bool has_density = false;
    std::vector<double> density;
    // FULL initial conservative state (all components), ncomp*n*n component-major; set by
    // set_conservative_state(name, U). Takes priority over density at seed (cf. make_build_params /
    // build_amr_compiled). SINGLE-BLOCK only (build_multi throws if has_state).
    bool has_state = false;
    std::vector<double> state;
    // SOURCE STAGE condensed by Schur (amr-schur path, set by set_source_stage). schur==false ->
    // explicit/imex path unchanged. theta/alpha frozen at the call. The splitting (lie/strang) and the
    // B_z field are CARRIED BY THE SYSTEM (Impl), not the block: a single condensed stage in single-block.
    bool schur = false;
    double schur_theta = 0.5, schur_alpha = 1.0;
    double schur_krylov_tol = 0.0;   // <= 0 = historical default (1e-10)
    int schur_krylov_max_iters = 0;  // <= 0 = historical default (400)
    std::string schur_density, schur_momentum_x, schur_momentum_y, schur_energy;  // "" = canonical
    NewtonOptions newton{};  // IMEX source Newton options (wave 3; single-block AND multi-block)
    bool newton_non_default = false;  // true -> non-default options (.so loader REJECTED: flat ABI)
    bool newton_diagnostics = false;  // newton_report: native MULTI-BLOCK (single/.so REJECTED)
    // TEMPORAL METHOD of the block (time == "ssprk3" -> kSsprk3). 0 == historical forward Euler (default),
    // 1 == kSsprk3 (order 3 + per-stage reflux). Materialized to AmrTimeMethod at build (single-block via
    // make_build_params -> bp.time_method; multi-block via dispatch_amr_block). Mutually exclusive with imex.
    int time_method = 0;
    // Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors the Density-role face
    // states + C/F fine ghost means to >= pos_floor. 0 (default) = inactive, bit-identical. Threaded
    // to dispatch_amr_block (multi-block) and to AmrBuildParams::pos_floor (single-block, build_amr_compiled).
    // COMPILED blocks carry it too (ADC-322): set_compiled_block stores it here from the regenerated
    // .so loader (adc_install_native_amr -> add_compiled_model), so both routings floor like a native block.
    double pos_floor = 0.0;
  };

  std::vector<BlockSpec> blocks;
  double gamma = 1.4;  // gamma of the FIRST block (compat: read by the single-block path)

  // Coupled inter-species sources (compiled adc.dsl.CoupledSource, flat P5 bytecode ABI) FROZEN at
  // add_coupled_source and injected into the AmrRuntime runtime engine at lazy build (build_multi).
  // The runtime does not yet exist at registration (built at ensure_built): so we store the flat
  // spec here, then replay it on the runtime right after its construction (multi-block only).
  struct CoupledSourceSpec {
    std::vector<std::string> in_blocks, in_roles;
    std::vector<double> consts;
    std::vector<std::string> out_blocks, out_roles;
    std::vector<int> prog_ops, prog_args, prog_lens;
    double frequency = 0.0;  // CONSTANT declared mu (bound dt <= cfl/mu; 0 = no bound)
    std::string label = "coupled_source";
    // Optional PER-CELL frequency mu(U): bytecode program (same inputs/constants/register table
    // as the source). EMPTY = constant frequency only. Replayed on the runtime at build.
    std::vector<int> freq_prog_ops, freq_prog_args;
  };
  std::vector<CoupledSourceSpec> coupled_sources;

  double refine_threshold = 1e30;  // 1e30 => no refinement by default
  // ADC-296: refinement variable selected by NAME (refine_var_name) XOR by physical ROLE
  // (refine_var_role). BOTH empty (default) => component 0 (historical density criterion, bit-identical).
  // Resolved PER BLOCK at build_multi against the block's cons_vars (STRICT, no silent comp-0 fallback).
  std::string refine_var_name;
  std::string refine_var_role;
  // PHI tag threshold on |grad phi| (D4): <= 0 => phi does NOT contribute to the tag union (default,
  // bit-identical). > 0 => in multi-block + regrid_every > 0, build_multi sets the engine's phi predicate
  // (set_phi_tag_predicate): refines where |grad phi| (components 1,2 of the shared aux) exceeds this threshold.
  double phi_grad_threshold = 0.0;

  // SOURCE STAGE condensed by Schur (amr-schur path): splitting policy (Lie/Strang) + B_z field,
  // CARRIED BY THE SYSTEM (single-block: a single condensed stage, shared B_z). bz_field is required
  // as soon as a block carries schur==true (checked at build: amr_write_coarse_bz throws if size != n*n).
  bool schur_strang = false;     // false: Lie (H(dt) S(dt)); true: Strang (H(dt/2) S(dt) H(dt/2))
  std::vector<double> bz_field;  // coarse B_z(x,y), n*n row-major (set_magnetic_field)
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse field (n*n row-major).
  // Pending until build: seeded into the single-block coupler (make_build_params -> bp.named_aux) AND
  // pushed to the multi-block runtime (build_multi). Empty -> bit-identical. cf. set_aux_field_component.
  std::map<int, std::vector<double>> named_aux_;
  // Per-field aux HALO policies (ADC-369): component -> uniform policy. Pending until build, then seeded
  // into the engine (bp.named_aux_bc for the coupler; runtime->set_named_aux_bc for the runtime).
  std::map<int, AuxHaloPolicy> named_aux_bc_;

  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto", p_wall = "none";
  double p_wall_radius = 0.0;

  bool built = false;
  // --- single-block path (AmrCouplerMP, untouched: bit-identical to history) ---
  std::shared_ptr<void> coupler_holder;  // keeps the hooks' AmrCouplerMP<Model> alive
  std::function<void(double)> step_fn;
  std::function<double()> max_speed_fn;
  std::function<double()> mass_fn;
  std::function<int()> n_patches_fn;
  std::function<std::vector<PatchBox>()> patch_boxes_fn;
  std::function<int()>
      coarse_local_boxes_fn;  ///< per-rank owned coarse fab count (ADC-319 diagnostic)
  std::function<int()> coarse_total_boxes_fn;  ///< global coarse box count (ADC-319 diagnostic)
  std::function<std::vector<double>()> density_fn;
  std::function<std::vector<double>()> potential_fn;
  // OPTIONAL step bounds of the single block (AMR StabilityPolicy, audit 2026-06): EMPTY hooks if
  // the model does not declare the traits -> single-block step_cfl keeps the historical formula.
  std::function<double()> source_frequency_fn;
  std::function<double()> stability_dt_fn;
  // GLOBAL bounds (AmrSystem::add_dt_bound): registered BEFORE the lazy build, passed to the
  // multi-block engine at its construction; read directly by the single-block step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds;
  std::string
      last_dt_reason;  // ACTIVE bound of the last single-block step_cfl (multi: via runtime)
  // Restoration of the single-block regrid cadence phase (IO v1, parity System::set_clock): the
  // builder populates this hook (writes the coupler's step_state). EMPTY until the block is installed.
  std::function<void(int)> set_macro_step_fn;
  // AMR single-rank CHECKPOINT / RESTART (ADC-65): per-level state accessors + hierarchy
  // imposition, populated by build_amr_compiled (single-block, AmrCouplerMP coupler). The multi-block
  // (runtime engine) does NOT populate them -> the facade methods reject runtime != nullptr.
  std::function<int()> n_levels_fn;
  std::function<int()> n_vars_fn;
  std::function<std::vector<double>(int)> level_state_fn;
  std::function<void(int, const std::vector<double>&)> set_level_state_fn;
  std::function<std::vector<double>(int)> level_potential_fn;
  std::function<void(int, const std::vector<double>&)> set_level_potential_fn;
  std::function<void(const std::vector<PatchBox>&)> set_hierarchy_fn;
  // --- multi-block path (AmrRuntime, shared hierarchy + summed Poisson) ---
  std::shared_ptr<adc::AmrRuntime> runtime;
  double t = 0;
  // AUTHORITATIVE MACRO-STEP counter (parity System::Impl::macro_step_): incremented by
  // AmrSystem::step / step_cfl, read by macro_step(). The engines (AmrRuntime; single-block step_state)
  // hold their OWN cadence counter, synchronized from this one at build and at set_clock.
  int macro_step_ = 0;
  bool clock_restore_pending_ =
      false;  // a set_clock is waiting to be pushed to the engine (at the next step)

  explicit Impl(const AmrSystemConfig& c) : cfg(c) {}

  // Pushes macro_step_ to the engine's cadence counter (regrid/stride): multi-block runtime OR
  // single-block coupler step_state. Called at the 1st step after a set_clock (clock_restore_pending_).
  // Without restoration the cadence starts from 0 (default, bit-identical).
  void push_macro_step_to_engine() {
    if (runtime)
      runtime->set_macro_step(macro_step_);
    else if (set_macro_step_fn)
      set_macro_step_fn(macro_step_);
  }

  // Index of the block named @p name in the registry (-1 if absent). Empty name -> first block (compat
  // single-block: the name was cosmetic). Multiple unnamed blocks is not ambiguous at add (an empty
  // name always targets block 0), but set_density(name) resolves it precisely.
  int block_index(const std::string& name) const {
    if (name.empty())
      return blocks.empty() ? -1 : 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
      if (blocks[i].name == name)
        return static_cast<int>(i);
    return -1;
  }

  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto")
      mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
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
    throw std::runtime_error("AmrSystem::set_poisson : unknown bc '" + mode + "'");
  }
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, cfg.L, "AmrSystem::set_poisson");
  }

  // Materializes the frozen parameters of the SINGLE-BLOCK path lazy build (AmrCouplerMP) from
  // the config + the refine/poisson/density choices + the SINGLE block (blocks[0]). Common to both
  // single-block paths (native ModelSpec and add_compiled_model): both instantiate the SAME shared
  // builder (detail::build_amr_compiled). UNTOUCHED by multi-block -> single-block bit-identical.
  AmrBuildParams make_build_params() {
    const BlockSpec& b = blocks[0];
    AmrBuildParams bp;
    bp.n = cfg.n;
    bp.L = cfg.L;
    bp.regrid_every = cfg.regrid_every;
    bp.gamma = b.gamma;
    bp.substeps = b.substeps;
    bp.recon_prim = b.recon_prim;
    bp.imex = b.imex;
    bp.time_method =
        b.time_method;  // SSPRK3 (1) / forward Euler (0) -> build_amr_compiled -> cpl->step
    bp.refine_threshold = refine_threshold;
    bp.poisson_bc = poisson_bc();
    bp.wall = wall_active();
    bp.has_density = b.has_density;
    bp.density = b.density;
    bp.has_state =
        b.has_state;  // full conservative state (set_conservative_state), takes priority at seed
    bp.state = b.state;
    bp.distribute_coarse = cfg.distribute_coarse;
    bp.coarse_max_grid = cfg.coarse_max_grid;
    // SOURCE STAGE condensed by Schur (amr-schur): block theta/alpha + system splitting/B_z.
    bp.schur = b.schur;
    bp.schur_theta = b.schur_theta;
    bp.schur_alpha = b.schur_alpha;
    bp.schur_krylov_tol = b.schur_krylov_tol;
    bp.schur_krylov_max_iters = b.schur_krylov_max_iters;
    bp.schur_density = b.schur_density;
    bp.schur_momentum_x = b.schur_momentum_x;
    bp.schur_momentum_y = b.schur_momentum_y;
    bp.schur_energy = b.schur_energy;
    bp.schur_strang = schur_strang;
    bp.bz_field = bz_field;
    bp.named_aux =
        named_aux_;  // ADC-291: model-named aux fields seeded onto the single-block coupler
    bp.named_aux_bc = named_aux_bc_;  // ADC-369: per-field aux halo policies
    // NEWTON OPTIONS of the single-block IMEX source (wave 3: now WIRED on the AmrCouplerMP
    // coupler). build_amr_compiled captures them from bp and passes them to cpl->step. Default
    // (add_block without option) = historical NewtonOptions{} -> path (2a) bit-identical.
    bp.newton_options = b.newton;
    // Zhang-Shu positivity floor (ADC-259): consumed by build_amr_compiled (mono-block ->
    // cpl->step / advance_transport). 0 (default) -> inactive, bit-identical historical path.
    bp.pos_floor = b.pos_floor;
    return bp;
  }

  // Installs the type-erased closures of the SINGLE-BLOCK path (AmrCouplerMP).
  void install(AmrCompiledHooks&& h) {
    coupler_holder = std::move(h.coupler_holder);
    step_fn = std::move(h.step);
    max_speed_fn = std::move(h.max_speed);
    mass_fn = std::move(h.mass);
    n_patches_fn = std::move(h.n_patches);
    patch_boxes_fn = std::move(h.patch_boxes);
    coarse_local_boxes_fn = std::move(h.coarse_local_boxes);  // ADC-319 MPI ownership diagnostic
    coarse_total_boxes_fn = std::move(h.coarse_total_boxes);
    density_fn = std::move(h.density);
    potential_fn = std::move(h.potential);
    source_frequency_fn = std::move(h.source_frequency);  // empty without trait (bit-identical)
    stability_dt_fn = std::move(h.stability_dt);
    set_macro_step_fn = std::move(h.set_macro_step);  // cadence phase restoration (IO v1)
    n_levels_fn = std::move(h.n_levels);              // AMR single-rank checkpoint/restart (ADC-65)
    n_vars_fn = std::move(h.n_vars);
    level_state_fn = std::move(h.level_state);
    set_level_state_fn = std::move(h.set_level_state);
    level_potential_fn = std::move(h.level_potential);
    set_level_potential_fn = std::move(h.set_level_potential);
    set_hierarchy_fn = std::move(h.set_hierarchy);
    built = true;
  }

  bool multi_block() const { return blocks.size() >= 2; }

  // Builds the MULTI-BLOCK runtime engine (AmrRuntime): one common SharedAmrLayout (shared
  // hierarchy, frozen), then EACH block materializes its type-erased AmrRuntimeBlock on it (via its
  // block_builder, which captures the concrete Model/Limiter/Flux). The coarse Poisson is SUMMED and
  // CO-LOCATED (Sum_b elliptic_rhs_b(U_b) read at the same cells of the shared coarse grid).
  void build_multi() {
    // MULTI-BLOCK set_conservative_state (wave 3 audit): the full state is now THREADED to the
    // NATIVE builder (dispatch_amr_block -> build_amr_block, seed coupler_write_coarse_state +
    // injection to the fine levels, takes priority over density). The COMPILED (.so) path does not
    // transport it (frozen loader ABI) -> explicit rejection, never a silent density fallback.
    for (const auto& b : blocks)
      if (b.has_state && b.is_compiled)
        throw std::runtime_error(
            "AmrSystem::set_conservative_state : not transported by the compiled .so loader (block "
            "'" +
            b.name + "') in multi-block ; use a native block adc.Model(...), or set_density.");
    AmrBuildParams bp = make_build_params();  // geometry + poisson_bc + wall + common ownership
    const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
    std::vector<adc::AmrRuntimeBlock> rblocks;
    rblocks.reserve(blocks.size());
    for (auto& b : blocks) {
      if (b.is_compiled) {
        // A compiled block without a runtime builder (empty multi_builder) CANNOT go multi-block:
        // this is the case of an OLD .so loader (generated/compiled against a header earlier than this
        // capstone, which only called set_compiled_block with the mono_builder). We raise a CLEAR
        // error rather than calling an empty std::function (opaque std::bad_function_call); the
        // remedy is to regenerate the loader (dsl.compile_native(target='amr_system')).
        if (!b.compiled_block_builder)
          throw std::runtime_error(
              "AmrSystem : compiled block '" + b.name +
              "' without multi-block builder (.so loader predating multi-block production DSL "
              "support). "
              "Regenerate the loader via dsl.compile_native(target='amr_system') / "
              "compile(backend='production', target='amr_system').");
        // Compiled MULTI-BLOCK path (capstone v): the CONCRETE Model/Limiter/Flux are already captured
        // in the builder (add_compiled_model), we invoke it on the SHARED layout. It allocates the
        // level stack of the block on S and captures the scheme, EXACTLY like dispatch_amr_block for a
        // native block (the builder CALLS it internally). It resolves the partial IMEX mask ITSELF into
        // component indices against cons_vars of the concrete Model (the raw implicit_vars/roles are
        // passed to it). No throw: the 2nd compiled block (or a mix of compiled + native) is wired.
        // Newton options NOT transported by the .so loader builder (ABI frozen at generation):
        // explicit rejection rather than a silent iters=2 (regenerate the loader = dedicated follow-up).
        if (b.newton_non_default)
          throw std::runtime_error(
              "AmrSystem : Newton options are not transported by the compiled .so loader "
              "(block '" +
              b.name + "') ; use a native block adc.Model(...) in multi-block.");
        // newton_diagnostics report likewise: the .so loader builder allocates no NewtonReport nor
        // threads it (flat ABI). Explicit rejection (defense in depth; the Python facade already
        // filters it upstream) rather than a silently empty report.
        if (b.newton_diagnostics)
          throw std::runtime_error(
              "AmrSystem : newton_diagnostics (newton_report) is not transported by the "
              "compiled .so loader (block '" +
              b.name + "') ; use a native block adc.Model(...).");
        // Zhang-Shu positivity floor (ADC-322): the AmrCompiledBlockBuilder now carries a floor slot,
        // so a loader regenerated against this header floors the Density-role face states like a native
        // block (forwarded to dispatch_amr_block -> build_amr_block). b.pos_floor == 0 for an OLDER .so
        // (it never marshals the field) -> inactive, bit-identical. No reject.
        rblocks.push_back(b.compiled_block_builder(S, b.name, b.density, b.has_density, b.gamma,
                                                   b.substeps, b.recon_prim, b.imex, b.stride,
                                                   b.implicit_vars, b.implicit_roles, b.pos_floor));
        continue;
      }
      // Native ModelSpec path: model dispatch -> concrete type, then spatial scheme dispatch
      // -> build_amr_block (allocates the block's level stack on the SHARED layout + closures).
      // The block density is carried by the BlockSpec (set_density(name) targets it). The partial IMEX
      // mask (implicit_vars / implicit_roles) is resolved HERE into component indices, against the
      // conservative descriptor of the concrete Model type (cons_vars), then threaded to build_amr_block.
      // Transport-axis seam (ADC-335): each per-transport TU (python/amr_block_<transport>.cpp) runs the
      // SAME dispatch_amr_block as before (build_amr_block_for), but instantiates only its transport's
      // leaves. The impl-mask resolution + temporal-method mapping move into the seam. The string if/else
      // mirrors detail::dispatch_transport (same unknown-transport message).
      const detail::AmrBlockBuildArgs ba{b.spec,
                                         b.name,
                                         b.limiter,
                                         b.riemann,
                                         b.density,
                                         b.has_density,
                                         b.gamma,
                                         b.substeps,
                                         b.recon_prim,
                                         b.imex,
                                         b.stride,
                                         b.implicit_vars,
                                         b.implicit_roles,
                                         b.newton,
                                         b.has_state ? &b.state : nullptr,
                                         b.newton_diagnostics,
                                         b.time_method,
                                         b.pos_floor};
      if (b.spec.transport == "exb") {
        rblocks.push_back(detail::build_amr_block_exb(ba, S));
      } else if (b.spec.transport == "compressible") {
        rblocks.push_back(detail::build_amr_block_compressible(ba, S));
      } else if (b.spec.transport == "isothermal") {
        rblocks.push_back(detail::build_amr_block_isothermal(ba, S));
      } else {
        throw std::runtime_error(unknown_transport_msg(b.spec.transport));
      }
    }
    runtime =
        std::make_shared<adc::AmrRuntime>(S.geom, S.ba_coarse, S.poisson_bc, std::move(rblocks),
                                          S.base_per, S.replicated_coarse, S.wall);
    // Model-NAMED aux fields (ADC-291): push the pending coarse fields into the runtime engine, which
    // re-applies them onto the shared aux each solve_fields (so they persist across the union regrid)
    // and injects them to the fine levels. Empty -> no-op (bit-identical).
    for (const auto& kv : named_aux_)
      runtime->set_named_aux(kv.first, std::vector<Real>(kv.second.begin(), kv.second.end()));
    for (const auto& kv : named_aux_bc_)
      runtime->set_named_aux_bc(kv.first, kv.second);  // ADC-369
    // GLOBAL bounds registered BEFORE the lazy build (add_dt_bound): passed to the engine
    // (which aggregates them in its step_cfl, all_reduce_min). Those added AFTER go in directly.
    for (const auto& g : dt_bounds)
      runtime->add_dt_bound(g.label, g.fn);
    // Declared frequencies of the coupled sources (CoupledSource.frequency, wave 3): step bound
    // dt <= cfl/mu on the runtime engine macro-step. CONSTANT frequency then PER-CELL frequency
    // (Expr): the second is evaluated on the coarse grid at each step_cfl (add_coupled_frequency_expr
    // resolves the inputs / validates the bytecode; empty program -> ignored).
    for (const auto& cs : coupled_sources) {
      if (cs.frequency > 0.0)
        runtime->add_coupled_frequency(cs.label, static_cast<Real>(cs.frequency));
      runtime->add_coupled_frequency_expr(cs.label, cs.in_blocks, cs.in_roles, cs.consts,
                                          cs.freq_prog_ops, cs.freq_prog_args);
    }
    // TAG-UNION REGRID (capstone Phase 2, C.6): if regrid_every > 0, we ACTIVATE the engine's
    // cadence and set the PER-BLOCK tag predicate (D1). The criterion tags where the SELECTED variable
    // of the block exceeds refine_threshold -> the UNION of the block tags refines where ANY block
    // exceeds it. By DEFAULT the variable is component 0 (historical density criterion, like the
    // single-block path AmrCouplerMP which tags a(i,j,0) > threshold); ADC-296 lets set_refinement pick
    // it PER BLOCK by name/role, resolved against the block's cons_vars (STRICT: absent -> explicit
    // error, never a silent comp-0 fallback). refine_threshold == 1e30 (default, no refinement) -> no
    // tag -> grid unchanged even if regrid_every > 0 (consistent no-op). regrid_every == 0 ->
    // set_regrid(0) -> FROZEN hierarchy, bit-identical to before this PR.
    const Real thr = static_cast<Real>(refine_threshold);
    runtime->set_regrid(cfg.regrid_every);
    if (cfg.regrid_every > 0) {
      const bool selected = !refine_var_name.empty() || !refine_var_role.empty();
      for (std::size_t b = 0; b < blocks.size(); ++b) {
        int comp = 0;  // default: component 0 (bit-identical density criterion)
        if (selected) {
          // The compiled .so flat-ABI block carries no role table on its runtime side: a non-default
          // selector there is REFUSED (comp-0 only), not silently ignored (mirror of the other .so rejects).
          if (blocks[b].is_compiled)
            throw std::runtime_error(
                "AmrSystem::set_refinement : variable/role selector not supported on the compiled "
                ".so "
                "block '" +
                blocks[b].name + "' (component 0 only) ; use a native block adc.Model(...)");
          comp = detail::resolve_selected_component("AmrSystem::set_refinement", blocks[b].name,
                                                    runtime->block_cons_vars(b), refine_var_name,
                                                    refine_var_role);
        }
        runtime->set_block_tag_predicate(
            b, [thr, comp](const ConstArray4& a, int i, int j) { return a(i, j, comp) > thr; });
      }
      // PHI PREDICATE (D4): if the user set a |grad phi| threshold (set_phi_refinement > 0),
      // we wire the engine's phi predicate (read on the shared aux, components 1,2 = grad phi in x,y).
      // It is ADDED to the union of the per-block density predicates: the grid refines where any
      // block exceeds refine_threshold OR |grad phi| exceeds gthr. Physical diocotron criterion (ring
      // edge = potential gradient). <= 0 (default) -> not wired -> phi does not contribute (bit-identical).
      if (phi_grad_threshold > 0.0) {
        const Real gthr = static_cast<Real>(phi_grad_threshold);
        runtime->set_phi_tag_predicate([gthr](const ConstArray4& a, int i, int j) {
          const Real gx = a(i, j, 1), gy = a(i, j, 2);
          return std::sqrt(gx * gx + gy * gy) > gthr;
        });
      }
    }
    // Replays the coupled sources frozen at add_coupled_source on the just-built runtime engine:
    // each resolves (block, role) -> (index, component) against the blocks' cons_vars and stores its
    // closure (applied after transport at each macro-step). No source -> no-op (the loop is empty),
    // so multi-block without coupling stays bit-identical to before.
    for (const auto& cs : coupled_sources)
      runtime->add_coupled_source(cs.in_blocks, cs.in_roles, cs.consts, cs.out_blocks, cs.out_roles,
                                  cs.prog_ops, cs.prog_args, cs.prog_lens);
    built = true;
  }

  void ensure_built() {
    if (built)
      return;
    if (blocks.empty())
      throw std::runtime_error("AmrSystem : call add_block first");
    // UNLOCK (capstone Phase 2, C.6): multi-block + regrid_every > 0 IS NOW SUPPORTED
    // (the AmrRuntime engine carries the tag-union regrid, cf. build_multi -> set_regrid +
    // set_block_tag_predicate). The old REFUSAL (the multi-block hierarchy was FROZEN) is lifted: the
    // grid actually re-grids from the union of the tags of all blocks. regrid_every == 0
    // stays a frozen hierarchy (regrid never called), bit-identical to Phase 1.
    if (multi_block()) {
      build_multi();
      return;
    }

    // --- SINGLE-BLOCK path (AmrCouplerMP, untouched: bit-identical to history) ---
    const BlockSpec& b = blocks[0];
    // ADC-296: the name/role refinement selector is resolved per block by the MULTI-BLOCK runtime engine
    // (build_multi -> resolve_selected_component). The single-block AmrCouplerMP path tags on component 0
    // only; a selector requested but staying single-block would be SILENTLY ignored (the comp-0 fallback
    // this milestone forbids), so we REFUSE it explicitly -- same pattern as the implicit-mask reject
    // below. The default (empty selector) stays component 0, bit-identical.
    if (!refine_var_name.empty() || !refine_var_role.empty())
      throw std::runtime_error(
          "AmrSystem::set_refinement : the variable/role selector is only wired in MULTI-BLOCK "
          "(>= 2 add_block, union-of-tags runtime engine). In single-block the AmrCouplerMP path "
          "refines on component 0 only : drop variable=/role= or add a 2nd block.");
    // The single-block IMEX goes through AmrCouplerMP (advance_amr imex flag), which carries NO partial
    // IMEX mask (FULL backward-Euler). A mask requested but staying SINGLE-BLOCK would thus be SILENTLY
    // ignored -> we REFUSE it explicitly (the partial mask requires the multi-block runtime engine).
    if (!b.implicit_vars.empty() || !b.implicit_roles.empty())
      throw std::runtime_error(
          "AmrSystem : implicit_vars / implicit_roles (partial IMEX mask) are only wired in "
          "MULTI-BLOCK (>= 2 add_block, runtime engine). In single-block the IMEX handles ALL "
          "components implicitly (full backward-Euler) : remove the mask or add a 2nd block.");
    // SINGLE-BLOCK NEWTON OPTIONS (wave 3, settled): NOW WIRED on the AmrCouplerMP coupler
    // (make_build_params -> bp.newton_options -> build_amr_compiled -> cpl->step -> advance_amr ->
    // backward_euler_source). No more rejection of b.newton_non_default here: a single IMEX block with
    // newton_max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy runs correctly. Default = historical
    // iters=2 (bit-identical). Still NOT wired in single-block: the newton_diagnostics REPORT
    // (aggregated newton_report = multi-block engine only; threading it through the coupler subcycling
    // would be invasive) -> EXPLICIT rejection rather than a silently empty report.
    if (b.newton_diagnostics)
      throw std::runtime_error(
          "AmrSystem : newton_diagnostics (newton_report) is only wired in MULTI-BLOCK "
          "(AmrRuntime runtime engine). In single-block the Newton OPTIONS "
          "(newton_max_iters/rel_tol/"
          "abs_tol/fd_eps/damping/fail_policy) are wired, but not the aggregated report : add a "
          "2nd block for newton_report, use newton_fail_policy='warn'/'throw', or a single-level "
          "System for the full report.");
    const AmrBuildParams bp = make_build_params();
    if (b.is_compiled) {  // compiled path: the builder freezes the types (Model, Limiter, Flux)
      // Zhang-Shu positivity floor (ADC-322): bp.pos_floor (= b.pos_floor, set by set_compiled_block
      // from the regenerated loader) flows into the SAME build_amr_compiled leaf as a native block ->
      // cpl->step / advance_transport floor the Density-role face states. An OLDER .so never marshals
      // the field, so b.pos_floor stays 0 -> inactive, bit-identical. No reject.
      install(b.compiled_hooks_builder(bp));
      return;
    }
    // Native ModelSpec path: the model dispatch resolves the concrete type, then the SAME
    // spatial scheme dispatch + coupler build as add_compiled_model (detail, shared).
    // Transport-axis seam (ADC-335): the single-block (AmrCouplerMP) build, one per-transport TU
    // (python/amr_compiled_<transport>.cpp). bp already bundles every single-block parameter. Same
    // unknown-transport message as detail::dispatch_transport.
    if (b.spec.transport == "exb") {
      install(detail::build_amr_compiled_exb(b.spec, b.limiter, b.riemann, bp));
    } else if (b.spec.transport == "compressible") {
      install(detail::build_amr_compiled_compressible(b.spec, b.limiter, b.riemann, bp));
    } else if (b.spec.transport == "isothermal") {
      install(detail::build_amr_compiled_isothermal(b.spec, b.limiter, b.riemann, bp));
    } else {
      throw std::runtime_error(unknown_transport_msg(b.spec.transport));
    }
  }
};

namespace {
// UPSTREAM configuration guard (ADC-299): validate the AmrSystemConfig invariants BEFORE constructing
// Impl. The AMR Impl ctor is trivial (it only stores cfg, allocating nothing from n), so unlike System
// nothing is built before the check; we still validate ahead of Impl for parity with System and to keep
// every config rejection at a single upstream point. n was already guarded (n == 0 -> nn = n*n = 0 -> a
// division by zero in set_conservative_state, U.size() % nn, and an empty coarse grid downstream); L,
// regrid_every and coarse_max_grid were unchecked and reach the lazy build (dx, regrid cadence, coarse
// tiling) as is.
void validate_amr_system_config(const AmrSystemConfig& c) {
  if (c.n < 1)
    throw std::runtime_error("AmrSystem : n >= 1 required (coarse cells per direction) ; got n = " +
                             std::to_string(c.n));
  if (!(c.L > 0.0))
    throw std::runtime_error("AmrSystem : L > 0 required (square domain [0,L]^2) ; got L = " +
                             std::to_string(c.L));
  if (c.regrid_every < 0)
    throw std::runtime_error(
        "AmrSystem : regrid_every >= 0 required (0 = never regrid after init) ; "
        "got regrid_every = " +
        std::to_string(c.regrid_every));
  if (c.coarse_max_grid < 0)
    throw std::runtime_error(
        "AmrSystem : coarse_max_grid >= 0 required (0 = default n/2 tile, "
        "distribute_coarse only) ; got coarse_max_grid = " +
        std::to_string(c.coarse_max_grid));
}
}  // namespace

AmrSystem::AmrSystem(const AmrSystemConfig& c) {
  validate_amr_system_config(c);  // BEFORE Impl (parity with System; single upstream config guard)
  p_ = std::make_unique<Impl>(c);
}
AmrSystem::~AmrSystem() = default;
AmrSystem::AmrSystem(AmrSystem&&) noexcept = default;
AmrSystem& AmrSystem::operator=(AmrSystem&&) noexcept = default;

void AmrSystem::add_block(const std::string& name, const ModelSpec& model,
                          const std::string& limiter, const std::string& riemann,
                          const std::string& recon, const std::string& time, int substeps,
                          int stride, const std::vector<std::string>& implicit_vars,
                          const std::vector<std::string>& implicit_roles,
                          const NewtonOptions& newton, bool newton_diagnostics,
                          double positivity_floor) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::add_block : the system is already built (call "
        "add_block before any step/mass/density)");
  // Completeness contract of the model (ADC-290, parity with System::add_block): transport / elliptic
  // must be chosen explicitly. Validated before the transport string routing (build_multi /
  // build_amr_compiled), so a default-constructed ModelSpec fails clearly instead of silently
  // selecting Euler + Poisson-charge.
  detail::validate_model_spec(model);
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::add_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("AmrSystem::add_block : stride >= 1");
  // Zhang-Shu positivity floor (ADC-259): eager validation (parity with System::add_block). 0 =
  // inactive (bit-identical). The Density-role probe + the compiled-.so rejection happen at lazy build.
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "AmrSystem::add_block : positivity_floor >= 0 and finite (0 = inactive)");
  // IMEX source Newton options grouped into a POD (ADC-214; wave 3 audit, parity
  // System::add_block). Defaults {} = historical constants (2 / 0 / 0 / 1e-7 / 1.0 / none),
  // bit-identical. SUPPORT: MULTI-BLOCK engine (AmrRuntime) only -- the single-block (AmrCouplerMP)
  // keeps iters=2 frozen, non-default options are REJECTED there at build (ensure_built), never ignored.
  // Range check shared with System::add_block (validate_newton_options, in implicit_stepper.hpp).
  validate_newton_options(newton, "AmrSystem::add_block");
  const bool newton_non_default = newton.max_iters != 2 || newton.rel_tol != 0.0 ||
                                  newton.abs_tol != 0.0 || newton.fd_eps != 1e-7 ||
                                  newton.damping != 1.0 ||
                                  newton.fail_policy != NewtonOptions::kFailNone;
  if (time != "imex" && newton_non_default)
    throw std::runtime_error("AmrSystem::add_block : Newton options require time='imex'");
  // newton_diagnostics (newton_report) requires time='imex' (the report comes from the IMEX source
  // Newton), parity with System::add_block. SUPPORT: native MULTI-BLOCK only -- the single-block
  // (coupler) and the .so loaders reject it (at build / at the facade), never an empty report.
  if (time != "imex" && newton_diagnostics)
    throw std::runtime_error("AmrSystem::add_block : newton_diagnostics requires time='imex'");
  // time == "ssprk3": SSPRK3 (order 3 + per-stage reflux), explicit transport -> MUTUALLY EXCLUSIVE
  // with imex (single time.kind selector, parity with System). The implicit stiff source (imex) does
  // NOT combine with SSPRK3 (unvalidated combination): the engine also rejects it as defense in depth.
  if (time != "explicit" && time != "imex" && time != "ssprk3") {
    if (time == "imexrk_ars222")
      throw std::runtime_error(
          "AmrSystem : time 'imexrk_ars222' (IMEX-RK family, ARS(2,2,2) scheme) not wired on AMR "
          "(scope = Cartesian System). Use 'explicit'|'ssprk3'|'imex' on AMR, or a "
          "Cartesian System for IMEX-RK.");
    throw std::runtime_error("AmrSystem : time '" + time +
                             "' unknown on AMR (explicit|ssprk3|imex)");
  }
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem : unknown recon '" + recon + "' (conservative|primitive)");
  const bool imex = (time == "imex");
  const int time_method = (time == "ssprk3") ? 1 : 0;  // adc::AmrTimeMethod (0 kEuler, 1 kSsprk3)
  // The partial IMEX mask (implicit_vars / implicit_roles) only applies to the IMEX source step:
  // requesting it in explicit is an ERROR (no silent ignore; same guard as System::add_block).
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::add_block : implicit_vars / implicit_roles require time='imex' "
        "(the implicit mask only applies to the IMEX source step ; got time='" +
        time + "')");
  // MULTI-BLOCK (capstone v): a 2nd block (or more) switches to the AmrRuntime runtime engine
  // (shared hierarchy, co-located summed Poisson). The single block stays on AmrCouplerMP
  // (bit-identical). An already COMPILED block (set_compiled_block / add_compiled_model) CAN now mix
  // with a native block: its runtime builder (compiled_block_builder) materializes an
  // AmrRuntimeBlock on the SAME shared layout, exactly like a native block (cf. build_multi). A
  // single hard guard: the compiled block must have been registered WITH a runtime builder (an .so
  // loader recompiled against this header provides it; build_multi throws clearly otherwise).
  // MULTI-BLOCK: the name INDEXES the block (set_density/mass/density) -> it must be UNIQUE and non
  // empty as soon as there is (or we create) more than one block. Single-block: the name stays cosmetic.
  if (!p_->blocks.empty()) {
    if (name.empty())
      throw std::runtime_error(
          "AmrSystem::add_block : in multi-block each block must have a NON "
          "empty name (the name indexes the block : set_density/mass/density)");
    if (p_->block_index(name) >= 0)
      throw std::runtime_error("AmrSystem::add_block : block name already used '" + name +
                               "' (block names must be unique in multi-block)");
  }
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = false;
  b.spec = model;
  b.limiter = limiter;
  b.riemann = riemann;
  b.recon_prim = (recon == "primitive");
  b.imex = imex;
  b.time_method = time_method;  // SSPRK3 (1) or forward Euler (0); threaded at build (single/multi)
  b.implicit_vars =
      implicit_vars;  // partial IMEX mask (resolved into indices at build, build_multi)
  b.implicit_roles = implicit_roles;
  b.newton =
      newton;  // Newton options grouped into a POD (ADC-214; wave 3, single-block AND multi-block)
  b.newton_non_default = newton_non_default;
  b.newton_diagnostics =
      newton_diagnostics;          // newton_report (native multi-block; single/.so rejected)
  b.pos_floor = positivity_floor;  // Zhang-Shu floor (ADC-259); threaded at build (single/multi)
  b.substeps = substeps;
  b.stride = stride;
  b.gamma = model.gamma;  // adiabatic index of the block (Euler), read by coupler_write_coarse
  if (p_->blocks.empty())
    p_->gamma = model.gamma;  // compat: gamma of the 1st block (single-block path)
  p_->blocks.push_back(std::move(b));
}

ADC_EXPORT void AmrSystem::set_compiled_block(
    int ncomp, double gamma, int substeps,
    std::function<AmrCompiledHooks(const AmrBuildParams&)> mono_builder,
    AmrCompiledBlockBuilder multi_builder, const std::string& name, bool recon_prim, bool imex,
    int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles, double pos_floor) {
  (void)ncomp;  // the number of variables is carried by the concrete Model (Model::n_vars) in the
                // type-erasing builders; the parameter stays for API symmetry with System.
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_compiled_block : the system is already built");
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::set_compiled_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("AmrSystem::set_compiled_block : stride >= 1");
  // The partial IMEX mask only applies to the IMEX source step (same guard as add_block):
  // requesting it in explicit is an ERROR (no silent ignore).
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : implicit_vars / implicit_roles require "
        "time='imex' (the implicit mask only applies to the IMEX source step)");
  // MULTI-BLOCK (capstone v): a 2nd block (or more) switches to AmrRuntime; the compiled block is
  // materialized there by multi_builder (its AmrRuntimeBlock on the shared layout). So we STACK the
  // block instead of throwing. Like add_block: as soon as there is (or we create) more than one block,
  // the name INDEXES the block -> it must be UNIQUE and non empty. In single-block the name stays cosmetic.
  if (!p_->blocks.empty()) {
    if (name.empty())
      throw std::runtime_error(
          "AmrSystem::set_compiled_block : in multi-block each block must have "
          "a NON empty name (the name indexes the block : set_density/mass/density)");
    if (p_->block_index(name) >= 0)
      throw std::runtime_error("AmrSystem::set_compiled_block : block name already used '" + name +
                               "' (block names must be unique in multi-block)");
  }
  if (p_->blocks.empty())
    p_->gamma = gamma;  // compat: gamma of the 1st block (single-block path)
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = true;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.recon_prim = recon_prim;
  b.imex = imex;
  b.implicit_vars =
      implicit_vars;  // partial IMEX mask (resolved into indices by multi_builder at build)
  b.implicit_roles = implicit_roles;
  // Zhang-Shu positivity floor (ADC-322): carried by the regenerated .so loader (adc_install_native_amr
  // -> add_compiled_model). Stored on the block so the MONO path reads it via make_build_params ->
  // AmrBuildParams::pos_floor, and the MULTI path forwards it through the AmrCompiledBlockBuilder.
  b.pos_floor = pos_floor;
  b.compiled_hooks_builder = std::move(mono_builder);   // single-block path (AmrCouplerMP)
  b.compiled_block_builder = std::move(multi_builder);  // multi-block path (AmrRuntime)
  p_->blocks.push_back(std::move(b));
}

namespace {
// Module anchor for dladdr: its ADDRESS lives in the image that contains amr_system.cpp (the _adc
// module, or the test binary). add_native_block uses it to locate the module and promote it to
// global scope (RTLD_NOLOAD). A TU-local function suffices (no need to export) and avoids
// depending on a symbol defined elsewhere (adc::abi_key, system.cpp).
void amr_native_anchor() {}
}  // namespace

void AmrSystem::add_native_block(const std::string& name, const std::string& so_path,
                                 const std::string& limiter, const std::string& riemann,
                                 const std::string& recon, const std::string& time, double gamma,
                                 int substeps, double positivity_floor) {
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::add_native_block : substeps >= 1");
  // Zhang-Shu positivity floor (ADC-322): eager validation (parity with add_block). 0 = inactive,
  // bit-identical. Marshaled down to the loader (adc_install_native_amr) -> add_compiled_model; an
  // older .so (no floor slot) ignores it, so a non-zero floor on such a loader is a silent no-op.
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "AmrSystem::add_native_block : positivity_floor >= 0 and finite (0 = inactive)");
  // UPSTREAM scheme validation (like add_block): add_compiled_model(AmrSystem&) already rejects
  // time outside {explicit, imex} and recon outside {conservative, primitive}, but we diagnose HERE a
  // typo before the C++ boundary. time == "imex" => stiff source handled IMPLICITLY
  // (backward_euler_source), explicit transport carried by the reflux. limiter (including weno5, wired #105)
  // and riemann (including hllc/roe, wired at parity #113) are validated by dispatch_amr_compiled in the
  // loader (clear exception).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error(
        "AmrSystem::add_native_block : recon 'conservative' | 'primitive' "
        "(got '" +
        recon + "')");
  // time == "ssprk3": SSPRK3 IS NOT transported by the AMR .so loader flat ABI -- the extern "C"
  // installer (adc_install_native_amr) only marshals the time STRING, and the add_compiled_model
  // template it inlines only maps {explicit, imex} (it does NOT freeze AmrBuildParams::time_method). Rather
  // than SILENTLY FALLING BACK to kEuler (option ignored), we REJECT explicitly here: an SSPRK3 block
  // must go through a NATIVE block adc.Model(...) (AmrSystem.add_block), not a compiled .so loader.
  if (time == "ssprk3")
    throw std::runtime_error(
        "AmrSystem::add_native_block : time='ssprk3' not transported by the compiled .so loader "
        "(flat "
        "ABI : only {explicit, imex} is marshaled) ; use a native block adc.Model(...) with "
        "adc.Explicit(ssprk3=True) via AmrSystem.add_block.");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error(
        "AmrSystem::add_native_block : time 'explicit' | 'imex' on AMR (got '" + time + "')");
  // DSL "production" path on the AMR side: the generated .so loader (emit_cpp_native_loader with
  // target="amr_system") inlines the header template add_compiled_model(AmrSystem&, ...), which
  // materializes a concrete AmrCouplerMP<Model> at lazy build and installs its hooks via
  // set_compiled_block -- NATIVE path, SAME AMR hierarchy as add_block (reflux, regrid). The loader
  // thus calls set_compiled_block (out-of-line method of adc::AmrSystem) DEFINED in THIS module;
  // it must be resolved through the dlopen against the already-loaded _adc module.
  // ELF PORTABILITY (Linux): CPython loads _adc with RTLD_LOCAL, so its symbols are NOT in
  // the global scope. We PROMOTE the current module to global scope (RTLD_NOLOAD = without
  // reloading it; RTLD_GLOBAL OR'd into the flags of the already-loaded object), located by dladdr on
  // an ADDRESS of THIS module: amr_native_anchor (TU-local function). We thus avoid depending
  // on adc::abi_key (defined in system.cpp) -- which would link-couple any test compiling
  // amr_system.cpp alone. On macOS, harmless (the loader resolves via dynamic_lookup).
#if defined(_WIN32)
  // Windows (ADC-100): no RTLD_GLOBAL. The generated AMR .dll is linked against _adc.lib (symbol
  // AmrSystem::set_compiled_block ADC_EXPORT) + kokkoscore.lib (shared Kokkos). Undefined symbols resolved
  // by the OS loader against the already-loaded _adc.pyd + kokkos*.dll. We load + resolve adc_install_native_amr.
  adc::dynlib::handle h = adc::dynlib::open(so_path);
  if (!h)
    throw std::runtime_error("AmrSystem::add_native_block : LoadLibrary('" + so_path +
                             "') : " + adc::dynlib::last_error() +
                             " (.dll linked against _adc.lib + kokkoscore.lib ; cf. ADC-100)");
  {
    auto key_fn = reinterpret_cast<const char* (*)()>(adc::dynlib::sym(h, "adc_native_abi_key"));
    if (!key_fn) {
      adc::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::add_native_block : adc_native_abi_key missing from the .dll");
    }
    const std::string loader_key = key_fn();
    const std::string module_key = detail::abi_key_string();
    if (loader_key != module_key) {
      adc::dynlib::close(h);
      throw std::runtime_error("AmrSystem::add_native_block : incompatible ABI -- loader '" +
                               loader_key + "' != module '" + module_key + "'");
    }
    using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                  const char*, double, int, double);
    auto install = reinterpret_cast<install_fn_t>(adc::dynlib::sym(h, "adc_install_native_amr"));
    if (!install) {
      adc::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::add_native_block : adc_install_native_amr missing from the .dll");
    }
    install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
            time.c_str(), gamma, substeps, positivity_floor);
  }
#else
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&amr_native_anchor), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // RTLD_GLOBAL: places the loader's symbols in the global scope AND lets the loader resolve
  // its undefined symbols (set_compiled_block exported ADC_EXPORT) against the already-loaded images. RTLD_NOW:
  // immediate resolution -> a missing AmrSystem symbol fails HERE, not in flight.
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error(
        "AmrSystem::add_native_block : dlopen('" + so_path + "') : " + std::string(e ? e : "?") +
        " (the symbol adc::AmrSystem::set_compiled_block must be exported AND the "
        "_adc module loaded globally ; cf. ADC_EXPORT)");
  }
  // EXPLICIT ABI GUARD: the key baked into the loader (at ITS compilation) must equal the module's
  // key. A mismatch = divergent headers / compiler / standard -> potentially different memory layout
  // of AmrSystem/AmrBuildParams/AmrCompiledHooks at the boundary -> UB. We raise
  // a CLEAR error rather than let an incompatible loader through. SAME key symbol as the
  // System path (adc_native_abi_key): only the installer (adc_install_native_amr) differs.
  auto key_fn = reinterpret_cast<const char* (*)()>(dlsym(h, "adc_native_abi_key"));
  if (!key_fn) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : adc_native_abi_key missing from the .so "
        "(regenerate via dsl.compile_native(target='amr_system') / "
        "compile(backend='production', target='amr_system'))");
  }
  const std::string loader_key = key_fn();
  // Module key = SAME computation as adc::abi_key() (header-only detail::abi_key_string()): avoids the
  // dependency on the out-of-line symbol adc::abi_key (system.cpp). The loader bakes its own at ITS compile.
  const std::string module_key = detail::abi_key_string();
  if (loader_key != module_key) {
    dlclose(h);
    throw std::runtime_error("AmrSystem::add_native_block : incompatible ABI -- loader key '" +
                             loader_key + "' != module key '" + module_key +
                             "'. Recompile the loader with the SAME compiler, C++ standard and "
                             "adc headers as the _adc module.");
  }
  // AMR native installer of the loader: reinterpret_cast<AmrSystem*>(this) then
  // add_compiled_model<ProdModel>(*amrsys, ...). Scheme marshaled as flat extern "C" arguments. No
  // evolve parameter (single-block AMR, no frozen background block like System). DISTINCT SYMBOL
  // (adc_install_native_amr, vs adc_install_native on the System side): a loader generated for System
  // does NOT export it, so wiring it here fails clearly instead of an inconsistent cast. The trailing
  // double is the Zhang-Shu positivity floor (ADC-322): old 8-argument loaders carry an ABI key from
  // the pre-floor headers and are REJECTED above, so the 9-argument call never reaches a stale .so.
  using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, double);
  auto install = reinterpret_cast<install_fn_t>(dlsym(h, "adc_install_native_amr"));
  if (!install) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : adc_install_native_amr missing from the .so "
        "(loader generated for System, or regenerate via "
        "dsl.compile_native(target='amr_system'))");
  }
  install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps, positivity_floor);
  // The .so stays loaded (RTLD_GLOBAL) for the duration of the process: the type-erasing builder
  // installed by set_compiled_block captures code (header template) that lives there. We do NOT close it.
#endif  // _WIN32 (production AMR POSIX-only; Windows = throw, ADC-100)
}

void AmrSystem::set_refinement(double threshold, const std::string& variable,
                               const std::string& role) {
  // Reject the ambiguous double selector immediately (fast feedback); cons_vars is only known at the
  // lazy build, so an absent name/role is caught there (build_multi -> resolve_selected_component).
  if (!variable.empty() && !role.empty())
    throw std::runtime_error(
        "AmrSystem::set_refinement : select the refinement variable by NAME (variable=) or by ROLE "
        "(role=), not both");
  p_->refine_threshold = threshold;
  p_->refine_var_name = variable;
  p_->refine_var_role = role;
}

void AmrSystem::set_phi_refinement(double grad_threshold) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_phi_refinement : the system is already built (set the "
        "refinement criterion before any step/mass/density)");
  // <= 0 (default) -> phi DISABLED (build_multi does not set the phi predicate); bit-identical. > 0 ->
  // phi tag on |grad phi| added to the tag union (D4), set by build_multi. We DO NOT impose a
  // guard on the number of blocks here: the single/multi routing is only decided at ensure_built (>= 2
  // add_block), and the order of configuration calls is free (set_phi_refinement may precede the
  // 2nd add_block). In SINGLE-BLOCK the threshold stays without effect: the AmrCouplerMP path regrids on
  // density alone (no separate phi predicate) and does not call build_multi -> phi_grad_threshold is ignored,
  // without illusion (the phi predicate only makes sense on the multi-block runtime engine).
  p_->phi_grad_threshold = grad_threshold;
}

void AmrSystem::set_poisson(const std::string& rhs, const std::string& solver,
                            const std::string& bc, const std::string& wall, double wall_radius) {
  // single-block/explicit CONTRACT (cf. set_compiled_block): AMR wires a SINGLE elliptic
  // solver (GeometricMG, the AmrCouplerMP template default) and a SINGLE right-hand side
  // (f = model.elliptic_rhs(U), assembled by coupler_eval_rhs). We thus explicitly REFUSE
  // any rhs/solver value outside the actually wired domain, instead of storing it
  // silently (the historical no-op suggested solver='fft' worked on the hierarchy).
  // bc/wall are actually consumed by poisson_bc()/wall_active(): validated there.
  if (rhs != "charge_density" && rhs != "composite")
    throw std::runtime_error("AmrSystem::set_poisson : unknown rhs '" + rhs +
                             "' (charge_density|composite ; the right-hand side = sum of the "
                             "block's elliptic bricks)");
  if (solver != "geometric_mg")
    throw std::runtime_error("AmrSystem::set_poisson : solver '" + solver +
                             "' unsupported on AMR (only 'geometric_mg' is wired on the "
                             "hierarchy ; 'fft' only exists on a single-level grid, cf. System)");
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
}

void AmrSystem::set_density(const std::string& name, const std::vector<double>& rho) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_density : the system is already built (set the "
        "density before any step/mass/density)");
  if (p_->blocks.empty())
    throw std::runtime_error("AmrSystem::set_density : call add_block first");
  // SINGLE-BLOCK: the name is COSMETIC (historical compat, add_compiled_model /
  // add_native_block paths that do not name the BlockSpec) -> the density targets the single block,
  // whatever name is passed. MULTI-BLOCK: the name INDEXES the block (each block has ITS initial
  // density, for the summed Poisson RHS Sum_b q_b n_b); an unknown name is an explicit error.
  std::size_t idx = 0;
  if (p_->blocks.size() >= 2) {
    const int i = p_->block_index(name);
    if (i < 0)
      throw std::runtime_error("AmrSystem::set_density : no block named '" + name +
                               "' (multi-block : the name indexes the block)");
    idx = static_cast<std::size_t>(i);
  }
  p_->blocks[idx].density = rho;
  p_->blocks[idx].has_density = true;
}

void AmrSystem::set_conservative_state(const std::string& name, const std::vector<double>& U) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_conservative_state : the system is already built "
        "(set the state before any step/mass/density)");
  if (p_->blocks.empty())
    throw std::runtime_error("AmrSystem::set_conservative_state : call add_block first");
  // UPSTREAM size guard: NON empty state and multiple of n*n. The exact size ncomp*n*n is checked
  // at build (coupler_write_coarse_state), the only place where ncomp == Model::n_vars is known -- same
  // deferral as the n*n guard of set_density. We explicitly reject an EMPTY state (0 % nn == 0 would
  // otherwise set has_state=true with an empty state, which would only throw deep in the 1st step).
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (U.empty())
    throw std::runtime_error(
        "AmrSystem::set_conservative_state : empty state (expected ncomp*n*n)");
  if (U.size() % nn != 0)
    throw std::runtime_error("AmrSystem::set_conservative_state : state size (" +
                             std::to_string(U.size()) + ") not a multiple of n*n (" +
                             std::to_string(nn) + ") ; expected ncomp*n*n component-major");
  // SINGLE-BLOCK: cosmetic name. MULTI-BLOCK: the name indexes the block (but build_multi will then throw,
  // the full state only being wired on the single-block path).
  std::size_t idx = 0;
  if (p_->blocks.size() >= 2) {
    const int i = p_->block_index(name);
    if (i < 0)
      throw std::runtime_error("AmrSystem::set_conservative_state : no block named '" + name +
                               "' (multi-block : the name indexes the block)");
    idx = static_cast<std::size_t>(i);
  }
  p_->blocks[idx].state = U;
  p_->blocks[idx].has_state = true;
}

void AmrSystem::set_magnetic_field(const std::vector<double>& bz) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_magnetic_field : the system is already built "
        "(set B_z before any step)");
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (bz.size() != nn)
    throw std::runtime_error("AmrSystem::set_magnetic_field : B_z of size " +
                             std::to_string(bz.size()) + " (expected n*n = " + std::to_string(nn) +
                             ", coarse row-major)");
  p_->bz_field = bz;
}

void AmrSystem::set_aux_field_component(int comp, const std::vector<double>& field) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_aux_field : the system is already built (set named aux "
        "fields before any step)");
  // RESERVED components (phi/grad/B_z/T_e): a model-named aux field starts at kAuxNamedBase. B_z keeps
  // its dedicated path (the Python facade intercepts canonical names; this guard covers a direct call).
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "AmrSystem::set_aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase) + " (B_z -> set_magnetic_field)");
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (field.size() != nn)
    throw std::runtime_error("AmrSystem::set_aux_field : field of size " +
                             std::to_string(field.size()) +
                             " (expected n*n = " + std::to_string(nn) + ", coarse row-major)");
  p_->named_aux_[comp] = field;  // pending: seeded into the engine at build (single + multi block)
}

void AmrSystem::set_aux_field_halo_component(int comp, int bc_type, double value) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_aux_field (halo) : the system is already built (set named "
        "aux halos before any step)");
  if (comp < kAuxNamedBase)
    throw std::runtime_error("AmrSystem::set_aux_field (halo) : component " + std::to_string(comp) +
                             " reserved ; a named aux field starts at index " +
                             std::to_string(kAuxNamedBase));
  if (bc_type != static_cast<int>(BCType::Foextrap) &&
      bc_type != static_cast<int>(BCType::Dirichlet))
    throw std::runtime_error("AmrSystem::set_aux_field (halo) : unsupported halo type " +
                             std::to_string(bc_type) + " ; use foextrap or dirichlet");
  p_->named_aux_bc_[comp] = AuxHaloPolicy{static_cast<BCType>(bc_type), static_cast<Real>(value)};
}

void AmrSystem::set_source_stage(const std::string& name, const std::string& kind, double theta,
                                 double alpha, const SourceStageOptions& opts) {
  // Settings grouped into a POD (ADC-214): local aliases to keep the body readable (same names /
  // semantics as the old flat parameters). bz_aux_component of the POD is ignored here (the single-block
  // AMR stage reads the canonical B_z channel, cf. set_source_stage in amr_system.hpp).
  const double krylov_tol = opts.krylov_tol;
  const int krylov_max_iters = opts.krylov_max_iters;
  const std::string& density = opts.density;
  const std::string& momentum_x = opts.momentum_x;
  const std::string& momentum_y = opts.momentum_y;
  const std::string& energy = opts.energy;
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_source_stage : the system is already built "
        "(configure the source stage before any step)");
  // ONLY wired kind: ElectrostaticLorentzCondensation (cf. CondensedSchurSourceStepper), like System.
  if (kind != "electrostatic_lorentz")
    throw std::runtime_error("AmrSystem::set_source_stage : unknown kind '" + kind +
                             "' (only 'electrostatic_lorentz' is wired)");
  if (!(theta > 0.0 && theta <= 1.0))
    throw std::runtime_error("AmrSystem::set_source_stage : theta must be in (0, 1] (got " +
                             std::to_string(theta) + ")");
  // SINGLE-BLOCK only: the AMR condensed stage is wired on the single-block coupler (AmrCouplerMP). The
  // multi-block path (AmrRuntime) does not carry it yet -> clear error rather than a silent no-op.
  if (p_->multi_block())
    throw std::runtime_error(
        "AmrSystem::set_source_stage : the Schur-condensed source stage is "
        "only wired in SINGLE-BLOCK (1 add_block) ; this system has >= 2 blocks.");
  const int idx = p_->block_index(name);
  if (idx < 0)
    throw std::runtime_error("AmrSystem::set_source_stage : no block named '" + name + "'");
  Impl::BlockSpec& b = p_->blocks[static_cast<std::size_t>(idx)];
  // The condensed stage REPLACES the source: it requires SOURCE-FREE transport (explicit, NoSource model),
  // not the local IMEX. Requesting it on an imex block is an ERROR (both sources would stack).
  if (b.imex)
    throw std::runtime_error(
        "AmrSystem::set_source_stage : the condensed source stage requires "
        "EXPLICIT source-free transport (the block '" +
        name +
        "' is time='imex' ; the "
        "local IMEX source and the condensed stage would stack). Add the block in "
        "time='explicit' (model with NoSource source brick).");
  // The Density/MomentumX/MomentumY roles and the presence of B_z are validated AT BUILD (where the
  // concrete Model type -- thus cons_vars -- is known): the ctor of AmrCondensedSchurSourceStepper throws if a
  // role is missing, amr_write_coarse_bz throws if B_z is absent. Clear errors, like System.
  b.schur = true;
  b.schur_theta = theta;
  b.schur_alpha = alpha;
  // Transported settings (wave 3): Krylov tolerances + field descriptors ("" = canonical).
  // Resolution/validation of the descriptors AT BUILD against Model::conservative_vars() (where the
  // concrete type is known), like the canonical roles. krylov_tol validated here (form).
  if (krylov_tol > 0.0 && !(krylov_tol < 1.0))
    throw std::runtime_error("AmrSystem::set_source_stage : krylov_tol must be in (0, 1)");
  b.schur_krylov_tol = krylov_tol;
  b.schur_krylov_max_iters = krylov_max_iters;
  b.schur_density = density;
  b.schur_momentum_x = momentum_x;
  b.schur_momentum_y = momentum_y;
  b.schur_energy = energy;
}

void AmrSystem::set_time_scheme(const std::string& scheme) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_time_scheme : the system is already built "
        "(choose the splitting before any step)");
  if (scheme == "lie")
    p_->schur_strang = false;
  else if (scheme == "strang")
    p_->schur_strang = true;
  else
    throw std::runtime_error("AmrSystem::set_time_scheme : unknown scheme '" + scheme +
                             "' (expected 'lie' or 'strang')");
}

void AmrSystem::add_coupled_source(const CoupledSourceProgram& prog, double frequency,
                                   const std::string& label) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::add_coupled_source : the system is already built "
        "(register the source before any step/mass/density)");
  // MULTI-BLOCK only: a COUPLED source reads/writes SEVERAL named blocks; the single-block path
  // (AmrCouplerMP) has no block registry and carries its source via the model. We thus refuse a
  // coupled source as long as there are fewer than two blocks (EXPLICIT error rather than a silent no-op).
  if (p_->blocks.size() < 2)
    throw std::runtime_error(
        "AmrSystem::add_coupled_source : inter-species coupled source supported "
        "only in MULTI-BLOCK (>= 2 add_block) ; the single-block carries its source "
        "via the block model");
  // Bytecode description grouped into a POD (ADC-214). MINIMAL form validation here (list size);
  // the FINE validation (roles, blocks, opcodes, registers) is done by
  // AmrRuntime::add_coupled_source at injection (lazy build), exactly as System delegates to
  // CoupledSourceKernel. We store the flat spec as-is (POD fields copied one by one).
  if (prog.out_blocks.empty())
    throw std::runtime_error("AmrSystem::add_coupled_source : no source term (out_blocks empty)");
  p_->coupled_sources.push_back(Impl::CoupledSourceSpec{
      prog.in_blocks, prog.in_roles, prog.consts, prog.out_blocks, prog.out_roles, prog.prog_ops,
      prog.prog_args, prog.prog_lens, frequency, label, prog.freq_prog_ops, prog.freq_prog_args});
}

void AmrSystem::step(double dt) {
  p_->ensure_built();
  // PENDING cadence phase restoration (set_clock before the 1st step): now that the
  // engine exists (ensure_built), we push macro_step_ to its regrid/stride counter.
  if (p_->clock_restore_pending_) {
    p_->push_macro_step_to_engine();
    p_->clock_restore_pending_ = false;
  }
  if (p_->runtime)
    p_->runtime->step(static_cast<Real>(dt));
  else
    p_->step_fn(dt);
  p_->t += dt;
  ++p_->macro_step_;  // authoritative counter (parity System: one macro-step = one increment)
}
void AmrSystem::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s)
    step(dt);
}
double AmrSystem::step_cfl(double cfl) {
  p_->ensure_built();
  if (p_->clock_restore_pending_) {  // pending phase restoration (cf. step)
    p_->push_macro_step_to_engine();
    p_->clock_restore_pending_ = false;
  }
  const double hx = p_->cfg.L / p_->cfg.n;  // coarse grid spacing (dx_coarse)
  if (p_->runtime) {
    // MULTI-BLOCK: SUBSTEPS/STRIDE-AWARE CFL step, EXACT mirror of System::step_cfl. A block of
    // stride cadence advances an effective step stride*dt in substeps sub-steps, so each sub-step
    // is worth stride*dt/substeps; the per-sub-step stability condition gives the per-block dt
    // dt_b = cfl*h*substeps_b/(stride_b*w_b), and the GLOBAL dt is the min over the blocks (the most
    // constraining). AmrRuntime::step_cfl carries the formula (the w_b/substeps/stride live there), then
    // advances by a step(dt) which re-applies stride and substeps. BACKWARD-COMPAT: with substeps=1 and
    // stride=1 everywhere, dt = cfl*h*min_b(1/w_b) = cfl*h/w_max, identical to the old formula
    // (max_speed = max_b w_b) -> multi-block facade substeps=1/stride=1 bit-identical.
    const double dt =
        static_cast<double>(p_->runtime->step_cfl(static_cast<Real>(cfl), static_cast<Real>(hx)));
    p_->t += dt;
    ++p_->macro_step_;  // authoritative counter (parity System: one macro-step = one increment)
    return dt;
  }
  // SINGLE-BLOCK (AmrCouplerMP): historical TRANSPORT bound dt = cfl*h/w_max (UNCHANGED formula,
  // bit-identical without optional bounds), then StabilityPolicy aggregation (audit 2026-06):
  //  - BLOCK bounds (source_frequency / stability_dt hooks, EMPTY without model trait) applied
  //    to the effective SUB-STEP dt/substeps (step_fn splits dt into substeps pieces; no stride
  //    in single-block): dt <= cfl*substeps/mu and dt <= dt_adm*substeps;
  //  - GLOBAL bounds (add_dt_bound), all_reduce_min like System (identical dt on all ranks).
  double h = cfl * hx / p_->max_speed_fn();
  p_->last_dt_reason = "transport:" + p_->blocks[0].name;
  const double sub = static_cast<double>(p_->blocks[0].substeps);
  if (p_->source_frequency_fn) {
    const double mu = p_->source_frequency_fn();
    if (mu > 0.0) {
      const double dt_src = cfl * sub / mu;
      if (dt_src < h) {
        h = dt_src;
        p_->last_dt_reason = "source_frequency:" + p_->blocks[0].name;
      }
    }
  }
  if (p_->stability_dt_fn) {
    const double db = p_->stability_dt_fn();
    if (db > 0.0) {
      const double dt_adm = db * sub;
      if (dt_adm < h) {
        h = dt_adm;
        p_->last_dt_reason = "stability_dt:" + p_->blocks[0].name;
      }
    }
  }
  for (const auto& g : p_->dt_bounds) {
    if (!g.fn)
      continue;
    double v = g.fn();
    if (!(v > 0.0) || !std::isfinite(v))
      v = std::numeric_limits<double>::infinity();
    v = all_reduce_min(v);
    if (v < h) {
      h = v;
      p_->last_dt_reason = "global:" + g.label;
    }
  }
  p_->step_fn(h);
  p_->t += h;
  ++p_->macro_step_;  // authoritative counter (parity System: one macro-step = one increment)
  return h;
}

// GLOBAL step bound (AMR counterpart of System::add_dt_bound): registered BEFORE or AFTER the build
// (single-block: read by step_cfl at each step; multi-block: passed to the engine, or added hot
// if it already exists). fn() is evaluated PER RANK then reduced all_reduce_min on the consumer side.
void AmrSystem::add_dt_bound(const std::string& label, std::function<double()> fn) {
  if (!fn)
    throw std::runtime_error("AmrSystem::add_dt_bound : empty bound function");
  p_->dt_bounds.push_back(Impl::GlobalDtBound{label, fn});
  if (p_->runtime)
    p_->runtime->add_dt_bound(label, std::move(fn));
}

// ACTIVE bound of the last step_cfl ("" before the first CFL step).
std::string AmrSystem::last_dt_bound() const {
  if (p_->runtime)
    return p_->runtime->last_dt_bound();
  return p_->last_dt_reason;
}

// Newton report (OPT-IN IMEX diagnostics) of the block, AGGREGATED over the levels/sub-steps of its
// LAST advance (reset at the head of advance by AmrRuntime::step). Native MULTI-BLOCK only: the
// single-block (AmrCouplerMP coupler) rejects it at build (ensure_built); a call HERE without a runtime
// engine (thus single-block built) raises a clear error (parity System::newton_report: never an empty report).
AmrSystem::SourceNewtonReport AmrSystem::newton_report(const std::string& name) {
  p_->ensure_built();  // materializes the engine (multi-block) or the coupler (single-block)
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::newton_report : Newton diagnostics not available in single-block (the "
        "AmrCouplerMP "
        "coupler rejects newton_diagnostics at build) ; add a 2nd block (multi-block engine), "
        "or use a single-level System for the full report.");
  const NewtonReport& r =
      p_->runtime->newton_report(name);  // throws if unknown block / diagnostics off
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp};
}

int AmrSystem::nx() const {
  return p_->cfg.n;
}
double AmrSystem::time() const {
  return p_->t;
}
int AmrSystem::macro_step() const {
  return p_->macro_step_;
}
void AmrSystem::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("AmrSystem::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
  // Pushes the cadence phase (regrid/stride) to the engine: right away if it is already built, otherwise at
  // the 1st step (clock_restore_pending_). set_clock is typically called BEFORE the 1st step (restart of a
  // replayed composition, lazy build), hence the flag.
  if (p_->built)
    p_->push_macro_step_to_engine();
  else
    p_->clock_restore_pending_ = true;
}
int AmrSystem::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}
std::vector<std::string> AmrSystem::block_names() const {
  std::vector<std::string> out;
  out.reserve(p_->blocks.size());
  for (const auto& b : p_->blocks)
    out.push_back(b.name);
  return out;
}
int AmrSystem::n_patches() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->n_patches();
  return p_->n_patches_fn();
}
std::vector<PatchBox> AmrSystem::patch_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->patch_boxes();  // MULTI-BLOCK: AmrRuntime engine
  return p_->patch_boxes_fn();          // SINGLE-BLOCK: AmrCouplerMP hook
}
int AmrSystem::coarse_local_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->coarse_local_boxes();  // MULTI-BLOCK: shared layout, block 0
  return p_->coarse_local_boxes_fn();          // SINGLE-BLOCK: AmrCouplerMP hook
}
int AmrSystem::coarse_total_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->coarse_total_boxes();
  return p_->coarse_total_boxes_fn();
}
double AmrSystem::mass() {
  return mass(std::string());
}
double AmrSystem::mass(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime)
    return p_->mass_fn();                 // SINGLE-BLOCK: cosmetic name -> single block
  const int idx = p_->block_index(name);  // MULTI-BLOCK: the name indexes the block
  if (idx < 0)
    throw std::runtime_error("AmrSystem::mass : no block named '" + name + "'");
  return static_cast<double>(p_->runtime->mass(static_cast<std::size_t>(idx)));
}
std::vector<double> AmrSystem::density() {
  return density(std::string());
}
std::vector<double> AmrSystem::density(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime)
    return p_->density_fn();              // SINGLE-BLOCK: cosmetic name -> single block
  const int idx = p_->block_index(name);  // MULTI-BLOCK: the name indexes the block
  if (idx < 0)
    throw std::runtime_error("AmrSystem::density : no block named '" + name + "'");
  return p_->runtime->density(static_cast<std::size_t>(idx));
}
std::vector<double> AmrSystem::potential() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->potential();  // shared aux (common to all blocks)
  return p_->potential_fn();
}

namespace {
// Common message of the AMR checkpoint accessors (ADC-65): BIT-IDENTICAL restart is only wired
// in SINGLE-BLOCK (AmrCouplerMP coupler). The multi-block (AmrRuntime engine) shares the layout AND
// the aux between blocks and does not yet expose the per-level/per-block state nor hierarchy imposition on
// the shared grid -> EXPLICIT rejection rather than a silent partial/false state (documented follow-up).
const char* const kAmrCkptMonoOnly =
    "AMR checkpoint/restart (per-level state, hierarchy imposition) wired in SINGLE-BLOCK only "
    "(AmrCouplerMP coupler) ; this system is multi-block (AmrRuntime engine : shared layout + aux "
    "= follow-up). Use a single add_block, or a single-level System (bit-identical "
    "checkpoint/restart).";
}  // namespace

int AmrSystem::n_levels() {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->n_levels_fn();
}
int AmrSystem::n_vars() {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->n_vars_fn();
}
std::vector<double> AmrSystem::level_state(int k) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->level_state_fn(k);
}
void AmrSystem::set_level_state(int k, const std::vector<double>& s) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->set_level_state_fn(k, s);
}
std::vector<double> AmrSystem::level_potential(int k) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->level_potential_fn(k);
}
void AmrSystem::set_level_potential(int k, const std::vector<double>& p) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->set_level_potential_fn(k, p);
}
void AmrSystem::set_hierarchy(const std::vector<PatchBox>& boxes) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->set_hierarchy_fn(boxes);
}

}  // namespace adc
