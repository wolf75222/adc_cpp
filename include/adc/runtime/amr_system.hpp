#pragma once

#include <adc/mesh/patch_box.hpp>    // PatchBox: index-space signature of a fine patch (patch_boxes())
#include <adc/mesh/physical_bc.hpp>  // BCRec
#include <adc/numerics/time/implicit_stepper.hpp>  // NewtonOptions (Newton options of the IMEX source)
#include <adc/runtime/export.hpp>    // ADC_EXPORT: set_compiled_block resolved by the native AMR loader
#include <adc/runtime/facade_options.hpp>  // SourceStageOptions / CoupledSourceProgram (facade PODs, ADC-214)
#include <adc/runtime/model_spec.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Multi-species composition on AMR at runtime: the refined counterpart of System.
///
/// One or SEVERAL blocks (species, described by ModelSpec of generic bricks) carried on an
/// AMR hierarchy. Like System but on an adaptive mesh.
///
/// MONO-BLOCK (1 add_block): a single-model AmrCouplerMP<Model> (coarse + one fine level tracked by
/// regrid, conservative reflux). Historical path, UNTOUCHED -> bit-identical.
///
/// MULTI-BLOCK (>= 2 add_block, capstone, docs/AMR_MULTIBLOCK_DESIGN.md): N blocks co-located on
/// ONE SHARED AMR hierarchy (same BoxArray + DistributionMapping + dx/dy per level, guarded by
/// same_layout_or_throw). All blocks live on ALL patches. A single aux per level (phi,
/// grad phi) and a single coarse Poisson whose right-hand side is the CO-LOCATED SUM of the blocks'
/// elliptic bricks (f = Sum_b q_b n_b read at the same cells). Conservation PER BLOCK (reflux +
/// average_down). AmrRuntime runtime engine (type-erased registry by name). Blocks with potentially
/// DIFFERENT spatial schemes and with PER-BLOCK TEMPORAL TREATMENT (explicit or IMEX, local stiff
/// implicit source; capstone vii) with union-of-tags regrid (multi-block + regrid_every > 0 is NOW
/// SUPPORTED: the mesh re-grids from the union of the tags; regrid_every == 0 = frozen hierarchy). Multirate (substeps/stride), inter-species
/// coupled sources: already wired. Multiple COMPILED blocks (add_compiled_model) and a MIX of
/// compiled + native: wired (capstone v, multi-block production DSL). Union regrid: later PR.
///
/// @note Two levels (ratio 2); explicit OR imex temporal treatment (per block).

namespace adc {

// Forward declarations of the runtime multi-block engine (definitions in amr_runtime.hpp /
// amr_dsl_block.hpp). set_compiled_block stores a DEFERRED runtime-block BUILDER which, given the
// SHARED layout materialized at lazy build, returns the type-erased AmrRuntimeBlock of the compiled
// block: this is what lets SEVERAL compiled blocks (multi-block production DSL) co-exist on the
// SAME AMR hierarchy, exactly like add_block in native multi-block. We forward-declare so as NOT
// to weigh down this public header (read by bindings.cpp and the loaders) with amr_runtime.hpp: only
// the TUs that BUILD/CALL the builder (amr_dsl_block.hpp and python/amr_system.cpp) include the
// complete definitions; a std::function with incomplete-type signature is legal as long as it is
// not instantiated with a concrete callable outside those TUs (PIMPL std::function recipe).
struct AmrRuntimeBlock;
namespace detail {
struct SharedAmrLayout;
}

/// AMR mesh and cadence (per-block physical parameters live in the ModelSpec).
struct AmrSystemConfig {
  int n = 128;            ///< coarse-level cells per direction
  double L = 1.0;         ///< size of the square domain [0,L]^2
  int regrid_every = 20;  ///< re-refinement every N steps (0 = never after init)
  bool periodic = true;   ///< periodic domain
  /// OWNERSHIP POLICY of the coarse level (cf. AmrCouplerMP::replicated_coarse).
  /// false (DEFAULT, historical): coarse mono-box REPLICATED on all ranks. The coarse Poisson
  ///   and the coarse transport are REDUNDANT on each GPU (zero communication,
  ///   better geometric MG) but DO NOT SCALE: only the fine patches are distributed.
  /// true (strong-scaling mode): coarse MULTI-BOX (BoxArray::from_domain, tile size
  ///   coarse_max_grid) distributed round-robin across the ranks. The coarse Poisson and the coarse
  ///   transport distribute (each rank carries only its tiles), which removes the redundancy
  ///   and enables AMR strong-scaling. The geometric MG then operates on a multi-box coarse
  ///   (cf. geometric_mg.hpp): convergence to be measured (may require more cycles).
  bool distribute_coarse = false;
  /// Coarse tile size when distribute_coarse=true (BoxArray::from_domain). 0 => n/2
  /// (minimal 2x2 split, least aggressive for the MG). Ignored if distribute_coarse=false.
  int coarse_max_grid = 0;
};

/// Frozen parameters passed to the deferred build of the compiled path (add_compiled_model). Materialized
/// by AmrSystem at ensure_built time: the geometry + the refine/poisson/density choices known
/// at that moment. The amr_dsl_block header consumes them to instantiate AmrCouplerMP<Model>.
struct AmrBuildParams {
  int n = 128;
  double L = 1.0;
  int regrid_every = 20;
  double gamma = 1.4;
  int substeps = 1;
  bool recon_prim = false;            ///< recon == "primitive" (frozen by add_compiled_model)
  bool imex = false;                  ///< time == "imex": stiff implicit source (backward_euler)
  double refine_threshold = 1e30;     ///< 1e30 => no refinement
  BCRec poisson_bc;                   ///< coarse Poisson BC (resolved by set_poisson)
  std::function<bool(Real, Real)> wall;  ///< conductive wall predicate (empty = none)
  bool has_density = false;
  std::vector<double> density;        ///< initial coarse density (component 0), n*n
  bool distribute_coarse = false;     ///< distributed multi-box coarse (AMR strong-scaling)
  int coarse_max_grid = 0;            ///< tile size of the distributed coarse (0 => n/2)
  // FULL initial conservative state (all components), takes priority over `density` when
  // has_state. ADDED AT THE END OF THE STRUCT: the offsets of the preceding fields are unchanged, so an
  // older mono-block .so loader (which COPIES bp into its own layout then does not read these fields)
  // falls back SILENTLY to the historical density path -- no corruption (append-only).
  bool has_state = false;
  std::vector<double> state;          ///< ncomp*n*n, component-major c*n*n + j*n + i; ncomp == Model::n_vars
  // Schur-CONDENSED SOURCE STAGE (amr-schur path, counterpart of System::set_source_stage). ADDED AT THE
  // END OF THE STRUCT (append-only, same reason as has_state: an older .so loader does not read these
  // fields and falls back to the historical explicit/imex path). schur==false -> path unchanged.
  bool schur = false;                 ///< true: GLOBAL condensed source stage (instead of local explicit/imex)
  double schur_theta = 0.5;           ///< theta-scheme of the condensed stage (0.5 = Crank-Nicolson)
  double schur_alpha = 1.0;           ///< electrostatic coupling constant of the condensed stage
  bool schur_strang = false;          ///< true: Strang splitting H(dt/2) S(dt) H(dt/2); false: Lie H(dt) S(dt)
  std::vector<double> bz_field;       ///< coarse B_z(x,y) field, n*n row-major (required by the condensed stage)
  // Settings of the condensed stage TRANSPORTED by the ABI (audit wave 3, append-only like has_state).
  double schur_krylov_tol = 0.0;      ///< tolerance of the coarse Krylov solve (<= 0 = default 1e-10)
  int schur_krylov_max_iters = 0;     ///< iteration budget (<= 0 = default 400)
  // Field descriptors of the stage ("" = canonical role, bit-identical; otherwise stable role
  // name OR block variable name, resolved at build against Model::conservative_vars()).
  std::string schur_density, schur_momentum_x, schur_momentum_y, schur_energy;
  // NEWTON OPTIONS of the IMEX source on the MONO-BLOCK path (wave 3: mono-block AMR options wired).
  // ADDED AT THE END OF THE STRUCT (append-only, same reason as has_state / schur_*: an older .so loader
  // does not read this field and falls back to the historical Newton with 2 frozen iters). DEFAULT {} =
  // historical constants (2 / 0 / 0 / 1e-7 / 1.0 / none) -> backward_euler_source path (2a)
  // bit-identical. Consumed by build_amr_compiled (the mono-block closure passes it to cpl->step).
  NewtonOptions newton_options{};
  // TEMPORAL METHOD of the block, transported as an INTEGER by the flat ABI (0 == kEuler, historical
  // before, 1 == kSsprk3). ADDED AT THE END OF THE STRUCT (append-only, same reason as has_state /
  // schur_* / newton_options: an older .so loader does not read this field and falls back SILENTLY
  // to 0 == kEuler -- no corruption). Consumed by build_amr_compiled (mono-block -> cpl->step).
  int time_method = 0;  // adc::AmrTimeMethod: 0 kEuler (default), 1 kSsprk3
  // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on the AMR
  // transport. ADDED AT THE END OF THE STRUCT (append-only, same reason as has_state / schur_* /
  // newton_options / time_method: an older flat-ABI .so loader does not read this field and falls
  // back SILENTLY to 0 -- inactive, bit-identical). Consumed by build_amr_compiled (mono-block ->
  // cpl->step / advance_transport). The COMPILED multi-block path rejects pos_floor > 0 at the facade
  // (the AmrCompiledBlockBuilder ABI carries no floor slot); native blocks thread it through.
  double pos_floor = 0.0;
};

/// Type-erased closures of a compiled AMR block, produced by amr_dsl_block::build_amr_compiled and
/// installed via AmrSystem::set_compiled_block. Symmetric with the std::function hooks of AmrSystem::Impl.
struct AmrCompiledHooks {
  std::shared_ptr<void> coupler_holder;   ///< keeps the AmrCouplerMP<Model> alive
  std::function<void(double)> step;       ///< one macro-step (periodic regrid included)
  std::function<double()> max_speed;      ///< max wave speed (CFL step)
  std::function<double()> mass;           ///< coarse mass
  std::function<int()> n_patches;         ///< number of fine patches
  std::function<std::vector<double>()> density;  ///< coarse density, n*n row-major
  std::function<std::vector<double>()> potential;  ///< coarse-level phi, n*n row-major
  // ADDED AT THE TAIL (additive, moves no existing field): index-space signatures of the fine
  // patches. Mirror of n_patches (same box_array(), the COUNT becomes the BOXES). The .so loader that
  // builds this struct is guarded by adc_native_abi_key: a .so generated BEFORE this addition must be
  // recompiled (the guard already diagnoses it clearly); the tail addition makes it purely additive.
  std::function<std::vector<PatchBox>()> patch_boxes;  ///< index-space signatures of the fine patches
  // ADDED AT THE TAIL (AMR StabilityPolicy, audit 2026-06, additive like patch_boxes): OPTIONAL step
  // bounds of the block, evaluated on the COARSE level by AmrSystem::step_cfl mono-block. EMPTY if
  // the model does not declare the HasSourceFrequency / HasStabilityDt traits (bit-identical). The
  // adc_native_abi_key guard forces regeneration of older .so files (purely additive addition).
  std::function<double()> source_frequency;  ///< coarse max of mu [1/s] (0 = does not constrain)
  std::function<double()> stability_dt;      ///< coarse min of the admissible step (0 = does not constrain)
  // ADDED AT THE TAIL (IO v1, parity with System::set_clock): restoration of the macro-step counter of
  // the MONO-BLOCK engine (the AmrCouplerMP coupler carries the regrid-cadence phase in a step_state;
  // this hook writes it at restart). EMPTY is never the case (the builder always populates it); the
  // adc_native_abi_key guard forces regeneration of older .so files (purely additive tail addition).
  std::function<void(int)> set_macro_step;   ///< restores the cadence (regrid) phase of the mono-block
  // ADDED AT THE TAIL (mono-rank AMR checkpoint/restart, ADC-65; additive like set_macro_step): FULL
  // CONSERVATIVE state per level (all components) + phi (warm-start) + imposition of a SAVED fine
  // hierarchy. The mono-block coupler (AmrCouplerMP) carries them; the builder always populates them
  // (never empty). The adc_native_abi_key guard forces regeneration of older .so files.
  std::function<int()> n_levels;                                ///< number of levels (>= 1)
  std::function<int()> n_vars;                                  ///< conserved components of the block
  std::function<std::vector<double>(int)> level_state;          ///< full state of level k (c*nf*nf+j*nf+i)
  std::function<void(int, const std::vector<double>&)> set_level_state;       ///< restores the state of level k
  std::function<std::vector<double>(int)> level_potential;      ///< phi of level k (nf*nf row-major)
  std::function<void(int, const std::vector<double>&)> set_level_potential;   ///< restores phi of level k
  std::function<void(const std::vector<PatchBox>&)> set_hierarchy;  ///< imposes the saved fine patches
  // ADDED AT THE TAIL (ADC-319, MPI ownership diagnostic; additive like the fields above): COARSE-level
  // (base) box counts. coarse_local_boxes = cpl->coarse().local_size() (level-0 fabs OWNED by this rank);
  // coarse_total_boxes = cpl->coarse().box_array().size() (total base boxes, all ranks). They reveal
  // whether distribute_coarse actually distributes the base across ranks (local < total) or replicates
  // it (local == total on every rank). The adc_native_abi_key guard forces regeneration of older .so
  // files (purely additive tail addition); the builder always populates them (never empty).
  std::function<int()> coarse_local_boxes;  ///< per-rank owned coarse (level-0) fab count
  std::function<int()> coarse_total_boxes;  ///< global coarse box count (identical on all ranks)
};

/// DEFERRED builder of a COMPILED block on the multi-block hierarchy: receives the SHARED layout (created
/// ONCE at lazy build, common to all blocks) plus the block parameters frozen at
/// add time (name, initial density, gamma, substeps/stride, recon/imex, partial IMEX mask resolved into
/// component indices), and returns the type-erased AmrRuntimeBlock of the block (captures the CONCRETE
/// Model/Limiter/Flux via detail::dispatch_amr_block, the kernel stays COMPILED). Symmetric with the
/// native add_block path: the (sole) difference is only that the types are known at add time (compiled
/// model) rather than resolved from a ModelSpec at build. The SIGNATURE mentions FORWARD-DECLARED types:
/// it is instantiated with a concrete callable only in add_compiled_model(AmrSystem&) (header
/// amr_dsl_block.hpp) where those types are complete, and invoked only in python/amr_system.cpp.
using AmrCompiledBlockBuilder = std::function<AmrRuntimeBlock(
    const detail::SharedAmrLayout& layout, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps, bool recon_prim,
    bool imex, int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles)>;

/// Single block carried on an AMR hierarchy, composed at runtime.
///
/// @code{.cpp}
/// adc::AmrSystemConfig cfg;                // base level: n x n on [0, L]^2
/// cfg.n = 64;
/// adc::AmrSystem amr(cfg);
///
/// adc::ModelSpec ne;
/// ne.transport = "exb"; ne.source = "none"; ne.elliptic = "charge";
/// amr.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// amr.set_poisson("charge_density", "geometric_mg");
/// amr.set_refinement(0.1);                 // refine where any block's field exceeds the threshold
///
/// amr.set_density("ne", rho0);             // rho0: initial density on the base level
/// amr.step_cfl(0.4);                       // conservative refluxed step + composite FAC Poisson
/// @endcode
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  // RULE OF FIVE (C.21): move-only (PIMPL unique_ptr). The copy was already IMPLICITLY deleted
  // (move ctor declared); we make it EXPLICIT for intent. No API change (the copy was
  // already unusable).
  AmrSystem(const AmrSystem&) = delete;
  AmrSystem& operator=(const AmrSystem&) = delete;
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// GLOBAL time-step bound (AMR counterpart of System::add_dt_bound): fn() evaluated ONCE
  /// per step_cfl (host), all_reduce_min (identical dt on all ranks), <= 0 / non-finite =
  /// inert this step. Hook for non-local constraints (coupling, scheduler, user ramp).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
  /// "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
  std::string last_dt_bound() const;

  /// Adds a block carried on the AMR. Same spatial-scheme parameters as System
  /// (limiter x riemann x recon), applied to each level/patch of the hierarchy. The FIRST
  /// add_block defines the block; a 2nd (or more) switches to the multi-block engine (shared
  /// hierarchy, co-located sum Poisson). Blocks can have DIFFERENT SPATIAL SCHEMES.
  /// @param name    block name: INDEXES the block (set_density(name), mass(name), density(name)). In
  ///                multi-block the name must be unique; mono-block an empty name targets the single block.
  /// @param model   composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" (weno5 = WENO5-Z, 3 ghosts; rusanov)
  /// @param riemann "rusanov" | "hll" (generic signed-wave, requires model.wave_speeds) | "hllc"
  ///                | "roe" (hllc/roe require a compressible transport)
  /// @param time    "explicit" (SSPRK2, forward-Euler source carried by the AMR step) | "ssprk3"
  ///                (SSPRK3, order 3, reflux per stage; explicit transport, EXCLUSIVE of imex) |
  ///                "imex" (stiff source handled IMPLICITLY by backward_euler_source; the transport
  ///                stays explicit, carried by the conservative reflux; cf. capstone vii). Any other
  ///                treatment is refused.
  /// @param substeps explicit substeps of the block (>= 1): the effective step is split into substeps
  ///                equal pieces (MULTI-BLOCK only; in mono-block, carried by AmrCouplerMP).
  /// @param stride  HOLD-THEN-CATCH-UP cadence of the block (>= 1; default 1 = each macro-step). stride=M
  ///                holds the block M-1 macro-steps then catches it up by an effective step M*dt (multirate).
  ///                MULTI-BLOCK only (a single block always advances every step). step_cfl honors
  ///                the cadence: dt = cfl*h*min_b(substeps_b/(stride_b*w_b)), mirror of System::step_cfl.
  /// @param implicit_vars / implicit_roles  partial IMEX mask CARRIED BY THE BLOCK (cf. System::add_block):
  ///                conserved components handled IMPLICITLY, by NAME (implicit_vars) or by physical
  ///                ROLE (implicit_roles). EMPTY (default) -> full backward-Euler (all
  ///                components implicit). Only meaningful with time="imex": requesting them in explicit
  ///                is an ERROR (no silent ignore). MULTI-BLOCK only (the mono-block
  ///                AmrCouplerMP carries its IMEX without a mask; a mask there is therefore refused).
  /// @throws std::runtime_error if a block is already defined, if substeps < 1, if stride < 1, if time
  ///         is not in {explicit, ssprk3, imex}, if recon is not in {conservative,
  ///         primitive}, or if an implicit mask is requested outside IMEX / with a name-role absent from the block.
  /// @param newton  options of the IMEX source Newton grouped in a POD (ADC-214; cf.
  ///                 NewtonOptions; parity with System::add_block): max_iters / rel_tol / abs_tol /
  ///                 fd_eps / damping / fail_policy. Default {} = historical constants, bit-identical.
  ///                 SUPPORT (wave 3, settled): these OPTIONS are wired in MONO-BLOCK (coupler
  ///                 AmrCouplerMP) AND in MULTI-BLOCK (AmrRuntime engine); the .so loaders
  ///                 reject them (flat ABI). fail_policy warn/throw works everywhere.
  /// @param newton_diagnostics  aggregated Newton report (newton_report): wired in NATIVE MULTI-BLOCK
  ///                 only (the mono-block rejects it at build, the .so loaders at the facade). Stays
  ///                 flat (a separate bool, outside the homogeneous family of convergence options).
  /// @param positivity_floor  Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors
  ///                 the Density-role face states (reconstruct_pp / zhang_shu_scale) AND the C/F fine
  ///                 ghost means to >= floor. Default 0 = inactive, bit-identical. Guarantee = face /
  ///                 ghost-state Density positivity only (order-1 fallback), NOT updated-mean nor
  ///                 pressure positivity (parity with System::add_block). A model without a Density
  ///                 role rejects floor > 0; the COMPILED .so multi-block path rejects it at the facade.
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0);

  /// Report of the implicit (IMEX) source Newton of a block, AGGREGATED over the levels and substeps of
  /// the block's LAST advance. Exists only if the block was added with newton_diagnostics=true IN
  /// NATIVE MULTI-BLOCK (explicit error otherwise: mono-block, .so loader, or block without diagnostics).
  /// Flat copy (no dependence on the numerics header on the caller side), parity with System::SourceNewtonReport.
  struct SourceNewtonReport {
    bool enabled;          ///< a report was computed (at least one IMEX advance played)
    bool converged;        ///< no cell failed on the last advance
    double max_residual;   ///< max over cells/levels/substeps of ||F||_inf at the Newton exit
    double max_iters_used; ///< max over cells/levels/substeps of the iterations consumed
    double n_failed;       ///< count (cells x levels x substeps) failed (non-finite / pivot / non-convergence)
    double failed_i;       ///< i of ONE faulty cell (-1 if none; max index encoded)
    double failed_j;       ///< j of the same cell (-1 if none)
    double failed_comp;    ///< conserved component of the worst residual of that cell (-1 unknown)
  };
  /// @throws std::runtime_error if the block is unknown, in mono-block, on a .so loader, or if the block
  ///         did not enable newton_diagnostics. Forces the lazy build (ensure_built).
  SourceNewtonReport newton_report(const std::string& name);

  /// Registers a COMPILED block (add_compiled_model path, header amr_dsl_block.hpp). TWO type-erased
  /// builders are frozen here, for the TWO routings of the facade:
  ///  - @p mono_builder: given the AmrBuildParams frozen at lazy build, returns the
  ///    AmrCompiledHooks of a concrete AmrCouplerMP<Model>. Used IN MONO-BLOCK (1 single compiled block)
  ///    -> historical AmrCouplerMP path, UNTOUCHED, bit-identical.
  ///  - @p multi_builder: given the SHARED layout materialized at lazy build (common to all
  ///    blocks), returns the type-erased AmrRuntimeBlock of the block. Used IN MULTI-BLOCK (>= 2 blocks,
  ///    compiled and/or native mixed) -> AmrRuntime runtime engine, exactly like add_block.
  /// @p recon_prim / @p imex / @p stride / @p implicit_vars / @p implicit_roles: metadata of the block
  /// (temporal scheme, multirate, partial IMEX mask) frozen at add time, consumed by the
  /// multi-block routing (the mono-block already carries them in the AmrBuildParams via mono_builder).
  /// DO NOT call directly: go through the free function add_compiled_model(AmrSystem&, ...).
  /// @throws std::runtime_error if the system is already built.
  /// DEFAULT VISIBILITY (ADC_EXPORT): the ONLY method called by the header template
  /// add_compiled_model(AmrSystem&) (cf. amr_dsl_block.hpp). A generated .so loader (DSL
  /// "production" path on the AMR side, emit_cpp_native_loader(target="amr_system") / add_native_block) inlines this
  /// template and must resolve this symbol from the already-loaded _adc module; compiled with
  /// -fvisibility=hidden (pybind11), the module would not export it without this annotation and the dlopen
  /// of the loader would fail. Symmetric with the ADC_EXPORT methods of System (grid_context/install_block).
  ADC_EXPORT void set_compiled_block(
      int ncomp, double gamma, int substeps,
      std::function<AmrCompiledHooks(const AmrBuildParams&)> mono_builder,
      AmrCompiledBlockBuilder multi_builder = {}, const std::string& name = std::string(),
      bool recon_prim = false, bool imex = false, int stride = 1,
      const std::vector<std::string>& implicit_vars = {},
      const std::vector<std::string>& implicit_roles = {});

  /// Wires a NATIVE AMR block from a .so loader generated by the DSL (backend "production", target
  /// "amr_system": dsl.compile_native(target="amr_system") / compile(backend="production",
  /// target="amr_system")). AMR counterpart of System::add_native_block: the .so inlines the header template
  /// add_compiled_model(AmrSystem&, ...), which materializes a concrete AmrCouplerMP<Model> at lazy
  /// build and installs its hooks via set_compiled_block -- NATIVE path, SAME AMR hierarchy as
  /// add_block (conservative reflux, regrid), no flat-array marshaling.
  ///
  /// The _adc module is PROMOTED to global scope (RTLD_NOLOAD) then the loader is dlopen-ed in
  /// RTLD_GLOBAL to resolve set_compiled_block; the ABI key baked in the loader
  /// (adc_native_abi_key) is compared to the module's (abi_key()) -- mismatch => clear error (no
  /// silent UB at the C++ boundary). Same scheme guard-rails as System (upstream validation).
  ///
  /// MULTI-BLOCK (capstone v): add_native_block CAN now be called several times (or mixed
  /// with native add_block) -> the compiled blocks co-exist on the shared hierarchy via AmrRuntime
  /// (the loader recompiled against this header provides the runtime builder; cf. set_compiled_block). The
  /// name then INDEXES the block (set_density/mass/density), like add_block.
  /// time is wired to {explicit, imex} (imex = stiff implicit source via backward_euler_source; any
  /// other treatment is rejected by add_compiled_model). The multirate (stride) and the partial IMEX
  /// mask do NOT transit through the flat ABI of the loader (ABI unchanged): this .so path now REJECTS
  /// them at the Python facade level (AmrSystem.add_equation raises ValueError on stride>1 or a
  /// non-empty IMEX mask, rather than ignoring them silently). For these parameters, use
  /// native add_block (ModelSpec) or add_compiled_model(AmrSystem&) DIRECTLY (header), which expose
  /// stride and the mask. recon "primitive" and flux "roe"/"hllc" are WIRED at parity (#113:
  /// dispatch_amr_compiled accepts them; the Python facade applies a pressure guard for hllc/roe).
  /// limiter "weno5" (WENO5-Z, 3 ghosts) is WIRED on rusanov (#105: the coupler levels are
  /// allocated to Limiter::n_ghost and the regrid inherits n_grow(): no out-of-bounds read).
  /// @throws std::runtime_error if the ABI diverges, if a symbol is missing, or substeps < 1.
  /// @param name block name: cosmetic in mono-block, INDEXES the block in multi-block (set_density/
  ///             mass/density; must be unique and non-empty from the 2nd block on, like add_block).
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1);

  /// Refines the cells where the density (component 0) exceeds @p threshold.
  void set_refinement(double threshold);

  /// Adds to the regrid criterion the PHI tag on |grad phi| (D4 of the design
  /// docs/AMR_REGRID_UNION_TAGS_DESIGN.md): also refines the cells where the norm of the gradient of the
  /// electrostatic potential |grad phi| (components 1,2 of the shared aux) exceeds @p grad_threshold.
  /// MULTI-BLOCK only (the AmrRuntime runtime engine carries the union-of-tags regrid; the mono-block
  /// path AmrCouplerMP has no separate phi predicate). The phi tag is ADDED to the union of the density
  /// tags per block (set_refinement): the mesh refines where ANY block exceeds its
  /// density threshold OR |grad phi| exceeds @p grad_threshold. PHYSICAL criterion of the diocotron: the ring
  /// edge follows the gradient of the potential, not the density alone.
  /// @param grad_threshold threshold of |grad phi|. <= 0 (DEFAULT) -> the phi tag is DISABLED (phi does not
  ///        contribute to the union; bit-identical to before this call). Without regrid_every > 0, no
  ///        effect (the regrid is never called). To be called BEFORE the first step.
  void set_phi_refinement(double grad_threshold);

  /// Configures the coarse Poisson (cf. System::set_poisson). On AMR the elliptic solver is
  /// ALWAYS GeometricMG and the right-hand side ALWAYS f = sum of the block's elliptic bricks.
  /// @param rhs    "charge_density" | "composite" (same composed right-hand side as System)
  /// @param solver "geometric_mg" only (the only one wired on the hierarchy; no FFT)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" (circular conductive wall, requires wall_radius > 0)
  /// @throws std::runtime_error if rhs, solver, bc or wall is outside the supported domain.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Sets the initial density on the coarse level (component 0), n*n row-major.
  /// @param name cosmetic label (mono-block AMR: the density targets the single block).
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Sets the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level, then
  /// prolongs it to the fine levels at build (constant injection, like the density). @p U is flat
  /// component-major (c*n*n + j*n + i) of size ncomp*n*n; ncomp == n_vars of the model (checked at
  /// build, where only Model::n_vars is known). Takes priority over set_density: allows starting the AMR
  /// from the paper's drift state (rho, rho*u, rho*v) instead of m=0 (Problem 2). The conversion
  /// primitive -> conservative (rho_u = rho*u) is done on the Python side (the caller already supplies the
  /// conservative). Wired on the NATIVE blocks (mono-block as well as multi-block: threaded to the native builder,
  /// seed the coarse then inject to the fine); in multi-block @p name indexes the target block. A
  /// COMPILED (.so) block carrying a state raises at build in multi-block (the .so loader does not transport
  /// the state): use a native block adc.Model(...) or set_density.
  /// @throws std::runtime_error if the system is already built, if U is empty, or if its size
  ///         is not a multiple of n*n.
  void set_conservative_state(const std::string& name, const std::vector<double>& U);

  /// Sets the magnetic field B_z(x, y) of the coarse level (n*n row-major), required by the Schur-condensed
  /// source stage (Lorentz term Omega = B_z). AMR counterpart of System::set_magnetic_field.
  /// MONO-BLOCK only (the condensed AMR stage is wired on the mono-block coupler AmrCouplerMP).
  /// @throws std::runtime_error if the system is already built or if bz is not of size n*n.
  void set_magnetic_field(const std::vector<double>& bz);

  /// Enables the Schur-CONDENSED SOURCE STAGE (amr-schur path) on block @p name. AMR counterpart of
  /// System::set_source_stage: assembles and solves the GLOBAL electrostatic/Lorentz condensed operator
  /// (on the coarse, by composing the uniform stage #126), instead of the LOCAL cell-by-cell IMEX
  /// source. This is the OPT-IN of the amr-schur path, the refined equivalent of the
  /// uniform time=Strang(Explicit(ssprk3), CondensedSchur(theta, alpha)). The block transport must
  /// be SOURCE-FREE (model with NoSource source brick): the source is played separately by the stage.
  /// @param kind  only "electrostatic_lorentz" wired (cf. CondensedSchurSourceStepper).
  /// @param theta theta-scheme in (0, 1] (0.5 = Crank-Nicolson).
  /// @param alpha electrostatic coupling constant.
  /// @throws std::runtime_error if the system is already built, in MULTI-BLOCK, if kind/theta are
  ///         out of domain, or (at build) if the block does not expose Density/MomentumX/MomentumY or if
  ///         set_magnetic_field was not called.
  /// @param opts  settings grouped in a POD (ADC-214; cf. SourceStageOptions; parity with
  ///        System::set_source_stage): krylov_tol / krylov_max_iters (COARSE Krylov solve; <= 0 =
  ///        historical defaults 1e-10 / 400; the FAC composite solve keeps its own tolerances,
  ///        Phase 4) and descriptors density / momentum_x / momentum_y / energy ("" = canonical role,
  ///        bit-identical; otherwise stable role name or block variable name, resolved at build). The
  ///        POD's bz_aux_component field is IGNORED here (the mono-block AMR stage reads the canonical
  ///        B_z channel). Default {} = historical behavior.
  void set_source_stage(const std::string& name, const std::string& kind, double theta,
                        double alpha, const SourceStageOptions& opts = {});

  /// Chooses the time-splitting policy of the condensed source stage: "lie" (default, H(dt) S(dt))
  /// or "strang" (H(dt/2) S(dt) H(dt/2), 2nd order). AMR counterpart of System::set_time_scheme. Without a condensed
  /// source stage (set_source_stage not called), no effect. @throws if scheme unknown / already built.
  void set_time_scheme(const std::string& scheme);

  /// Registers an inter-species COUPLED SOURCE (compiled adc.dsl.CoupledSource, flat bytecode ABI
  /// P5), refined counterpart of System::add_coupled_source but on the SHARED AMR hierarchy. The source
  /// is applied at EACH macro-step AFTER the transport, by forward-Euler splitting, level by
  /// level, followed by a fine -> coarse cascade (consistency of the covered coarse cells, #169).
  /// The coupling is baked into a device-clean stack machine (CoupledSourceKernel): NO per-cell Python
  /// callback in the hot path. MULTI-BLOCK only (>= 2 add_block: the coupling reads/writes
  /// SEVERAL named blocks). Must be called BEFORE the first step (the runtime engine is built
  /// at lazy build; the source is injected into it).
  ///
  /// CONSERVATION: an add_pair construction (a term +expr on a block, -expr exactly on the other,
  /// SAME cell) makes the sum of the two blocks conserved PER CELL (and globally) to machine
  /// precision. The engine does NOT IMPOSE it (an ionization creating an e/i pair is legal): it is a
  /// property of the constructed coupling (verify_conservation on the DSL side checks it symbolically).
  ///
  /// @throws std::runtime_error if called in mono-block, if the system is already built, or if the
  ///         shape of the bytecode / a role / a block is invalid (same guards as System).
  /// @param prog      bytecode description of the coupling grouped in a POD (ADC-214; cf.
  ///                  CoupledSourceProgram; parity with System::add_coupled_source): in_blocks /
  ///                  in_roles / consts / out_blocks / out_roles + prog_ops / prog_args / prog_lens
  ///                  (stack machine) + freq_prog_ops / freq_prog_args (PER-CELL frequency mu(U)
  ///                  optional; EMPTY = constant frequency only, bit-identical; non-empty:
  ///                  evaluated on the COARSE LEVEL of the input blocks at each step_cfl, MAX +
  ///                  all_reduce_max, bound dt <= cfl / max(mu) on the coarse, not the patches).
  /// @param frequency CONSTANT declared frequency mu [1/s] of the coupling (wave 3): bound
  ///                  dt <= cfl/mu on the macro-step of step_cfl; <= 0 (default) = no bound.
  /// @param label     name of the coupling (reason "coupled_source:<label>" of last_dt_bound).
  void add_coupled_source(const CoupledSourceProgram& prog, double frequency = 0.0,
                          const std::string& label = "coupled_source");

  void step(double dt);  ///< one AMR macro-step (periodic regrid included)
  void advance(double dt, int nsteps);
  /// Advances at dt = cfl * coarse_dx / max wave speed. @return the dt used.
  double step_cfl(double cfl);

  int nx() const;
  double time() const;
  /// MACRO-STEP counter (0-indexed; incremented by step / advance / step_cfl), parity with
  /// System::macro_step. Required for checkpoint/restart (the stride / regrid cadence depends on
  /// macro_step % stride|regrid_every, not only on t). Prerequisite IO PR-IO-3 (audit 2026-06).
  int macro_step() const;
  /// RESTORES the AMR clock (t, macro_step) -- parity with System::set_clock. Sets the time AND the
  /// macro-step counter (propagated to the regrid/stride cadence of the engine, mono-block as well as multi-block). Useful
  /// alone (stride cadence + clock resumption). @throws if macro_step < 0.
  void set_clock(double t, int macro_step);
  int n_blocks() const;           ///< number of blocks (1 = mono-block AmrCouplerMP; >= 2 = AmrRuntime)
  /// Names of the blocks in add order (parity with System::block_names): the IO facade iterates over them
  /// to write EACH block by its name (an empty name -> block 0, historical mono-block compat).
  std::vector<std::string> block_names() const;
  int n_patches();                ///< number of current fine patches (of the shared hierarchy)
  /// Index-space signatures of the current fine patches: one PatchBox (level, ilo, jlo, ihi, jhi) per
  /// fine box, for ALL fine levels (level >= 1). INCLUSIVE corners in the index space of the
  /// level (n << level cells/direction, ratio 2). SAME source as n_patches() (the GLOBAL fine
  /// BoxArray, all boxes/all ranks -> rank-independent, MPI-safe, zero communication). It is a
  /// QUERY (between steps): read-only of the already-stored boxes, NO hot-path cost. The
  /// conversion to [0, L]^2 is done on the Python side (which knows n via nx() and L). Forces the lazy
  /// build (ensure_built) like n_patches()/mass()/density().
  std::vector<PatchBox> patch_boxes();
  /// COARSE-level (base) box counts, MPI ownership diagnostic (ADC-319). coarse_local_boxes() = number
  /// of base boxes OWNED by this rank (level-0 MultiFab local_size()); coarse_total_boxes() = total base
  /// boxes across all ranks (BoxArray size, identical on every rank). With distribute_coarse=true the
  /// base is split into several boxes spread round-robin, so local < total per rank and the coarse
  /// transport distributes (MPI strong-scaling); a single-box or replicated base gives local == total on
  /// every rank. coarse_local_boxes() is rank-dependent, coarse_total_boxes() is rank-independent.
  /// Forces the lazy build (ensure_built) like n_patches()/mass()/density().
  int coarse_local_boxes();
  int coarse_total_boxes();

  /// MONO-RANK AMR CHECKPOINT / RESTART (ADC-65), MONO-BLOCK: per-level STATE accessors +
  /// hierarchy imposition for a BIT-IDENTICAL resumption (cf. AmrSystem.checkpoint/restart on the
  /// Python side). All REJECT multi-block (AmrRuntime engine: SHARED layout + aux, documented
  /// follow-up); the facade additionally rejects MPI np>1 (per-level gather: follow-up). Force the lazy build
  /// (ensure_built) like patch_boxes()/mass(). @p k: level (0 = coarse, >= 1 = fine).
  int n_levels();                 ///< number of levels of the hierarchy (>= 1)
  int n_vars();                   ///< number of conserved components (mono-block)
  /// FULL conservative state of level @p k, flat component-major c*nf*nf + j*nf + i (nf = n << k;
  /// zeros outside the patches at the fine level -- only the patch interior is defined).
  std::vector<double> level_state(int k);
  void set_level_state(int k, const std::vector<double>& s);  ///< restores the state of level @p k (as is)
  /// Potential phi of level @p k, flat nf*nf row-major. Level 0 = warm-start of the multigrid
  /// (bit-identical resumption); level >= 1 = aux comp 0 (recomputed at update).
  std::vector<double> level_potential(int k);
  void set_level_potential(int k, const std::vector<double>& p);  ///< restores phi of level @p k
  /// Imposes the SAVED fine hierarchy (at restart) instead of Berger-Rigoutsos clustering: @p boxes
  /// are the patch_boxes() signatures of the checkpoint (filtered to level 1 in mono-block).
  void set_hierarchy(const std::vector<PatchBox>& boxes);

  double mass();                  ///< mass of the 1st block on the coarse (conserved at reflux)
  double mass(const std::string& name);     ///< mass of the named block on the coarse (conserved PER BLOCK)
  std::vector<double> density();  ///< coarse density of the 1st block (component 0), n*n row-major
  std::vector<double> density(const std::string& name);  ///< coarse density of the named block, n*n
  /// Electrostatic potential phi of the COARSE LEVEL (base), n*n row-major. Level 0 covers
  /// the whole domain: enough to sample a median circle (azimuthal FFT), SAME
  /// observable as System::potential() on a single-level mesh. Solves the coarse Poisson if
  /// needed (cf. System::potential / ensure_elliptic), so current value even before any step.
  /// MULTI-BLOCK: phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located); shared by all
  /// the blocks (single aux). The block name therefore does not intervene.
  std::vector<double> potential();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
