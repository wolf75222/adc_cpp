#pragma once

#include <pops/core/state/variables.hpp>  // VariableSet (role-bearing descriptor carried by each block)
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonOptions (options of the IMEX source Newton)
#include <pops/runtime/export.hpp>  // POPS_EXPORT (methods resolved by the native loader through dlopen)
#include <pops/runtime/facade_options.hpp>  // SourceStageOptions / CoupledSourceProgram (facade PODs, ADC-214)
#include <pops/runtime/context/grid_context.hpp>  // GridContext + BlockClosures (AOT-compiled block seam)
#include <pops/runtime/config/model_spec.hpp>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Runtime multi-species composition: one coupled system, block by block.
///
/// Each block is a species (one state U) described by a ModelSpec (composition of generic
/// bricks: transport + source + elliptic right-hand side), with its spatial scheme
/// (limiter + Riemann flux), its time treatment and its substeps. All blocks share a
/// Poisson whose right-hand side is the sum of the per-block elliptic_rhs; the source S
/// acts per block. The core names no scenario; scenarios are compositions defined on the
/// application side (adc_cases).
///
/// Python composes (brick objects); the per-cell computation (assemble_rhs<L,F>, Newton of the
/// implicit source, multigrid/FFT) stays C++-compiled and is frozen when the block is added. No
/// Python callback in the hot path, except a time integrator written in Python via
/// eval_rhs / get_state / set_state.

namespace pops {

namespace runtime::program {
class Profiler;       // per-node wall-clock profiler (ADC-459); full type in program/profiler.hpp
class CacheManager;   // scheduler value cache (ADC-458); full type in program/cache_manager.hpp
}  // namespace runtime::program

/// Mesh and domain shared by all blocks (physical parameters are per block, in the ModelSpec).
///
/// GEOMETRY ("polar grid" work, Phase 1): the CHOICE of geometry lives HERE, in the mesh config,
/// NOT in the scheme (FiniteVolume stays reconstruction + Riemann + variables). Default
/// "cartesian": square domain [0,L]^2, behavior and numerics STRICTLY UNCHANGED (bit-identical).
/// "polar" describes a global ring r in [r_min, r_max] x theta in [0, 2pi) (cf. PolarGeometry); it is
/// carried by pops.PolarMesh on the Python side and is NOT yet wired into System::step (Phase 1 ships the
/// geometry + the polar transport operator + its MMS validation; polar transport through
/// System, which would also require the polar Poisson, is a later phase). Polar fields are
/// ignored while geometry == "cartesian".
struct SystemConfig {
  int n = 64;            ///< cells per direction (n x n domain) -- for polar: n_r = n_theta = n
  double L = 1.0;        ///< size of the square domain [0,L]^2 (cartesian)
  bool periodic = true;  ///< periodic domain, otherwise free outflow in transport (cartesian)
  // --- opt-in geometry (Phase 1): "cartesian" (default, bit-identical) | "polar" (global ring) ---
  std::string geometry =
      "cartesian";     ///< geometry choice (carried by pops.CartesianMesh / pops.PolarMesh)
  int nr = 0;          ///< radial cells (polar; 0 => takes n)
  int ntheta = 0;      ///< azimuthal cells (polar; 0 => takes n)
  double r_min = 0.0;  ///< inner radius of the ring (polar)
  double r_max = 1.0;  ///< outer radius of the ring (polar)
  // --- multi-box split of the polar TRANSPORT (split into theta BANDS, ADC-67) -----------------
  // Number of boxes of the ring, split in theta (each box covers the whole radius [0, nr-1] and one
  // azimuthal band). 1 (default) = mono-box STRICTLY bit-identical to history. theta_boxes > 1:
  // polar transport (assemble_rhs_polar + collective fill_ghosts) runs multi-box. CONSTRAINTS (cf.
  // PolarMesh / check_geometry): 1 <= theta_boxes <= ntheta AND theta_boxes divides ntheta. INERT in
  // cartesian (the cartesian split goes through AmrSystem / the historical mono-box MPI multi-box).
  // SCOPE: multi-box transport OK; DIRECT polar Poisson mono-box only (clear UPSTREAM rejection if
  // theta_boxes > 1, cf. ensure_elliptic_polar); polar tensor Schur stage multi-box.
  int theta_boxes = 1;  ///< boxes of the theta split of polar transport (1 = mono-box)
};

/// Coupled multi-species system, composed at runtime from generic bricks.
///
/// @code{.cpp}
/// pops::SystemConfig cfg;                  // n x n cells on [0, L]^2, periodic
/// cfg.n = 96;
/// pops::System sys(cfg);
///
/// pops::ModelSpec ne;                       // scalar density advected by E x B
/// ne.transport = "exb";
/// ne.source = "none";
/// ne.elliptic = "charge";
/// sys.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// sys.set_poisson("charge_density", "geometric_mg");
///
/// sys.set_density("ne", rho0);             // rho0: initial density, flattened row-major (n*n)
/// const double dt = sys.step_cfl(0.4);     // one CFL-limited step of the coupled system
/// @endcode
class System {
 public:
  explicit System(const SystemConfig& cfg);
  ~System();
  System(System&&) noexcept;
  System& operator=(System&&) noexcept;

  /// Adds an equation block (one species).
  /// @param model    composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter  reconstruction: "none" | "minmod" | "vanleer" | "weno5"
  /// @param riemann  numerical flux: "rusanov" (minimal generic) | "hll" (generic, requires
  ///                 model.wave_speeds) | "hllc" | "roe" (generic when the model supplies the
  ///                 HasHLLCStructure / HasRoeDissipation hooks; canonical Euler 2D fallback
  ///                 otherwise: 4 variables + ideal-gas pressure)
  /// @param recon    reconstructed variables: "conservative" | "primitive" (Euler: primitive
  ///                 more robust, positivity of rho and p)
  /// @param time     "explicit" (SSPRK2) | "ssprk3" | "imex" (explicit transport, local implicit
  ///                 backward-Euler source, order 1) | "imexrk_ars222" (IMEX-RK family, ARS(2,2,2)
  ///                 scheme, order 2, cartesian only; source FULLY implicit -> incompatible
  ///                 with implicit_vars/implicit_roles)
  /// @param substeps substeps per macro-step: the block advances N times per macro-step, each
  ///                 substep of length dt/N (fast electrons: substeps=10, step dt/10).
  /// @param stride   block cadence, HOLD-THEN-CATCH-UP semantics: 1 = every macro-step (default,
  ///                 bit-identical); M > 1 = block HELD (not advanced) while (macro_step + 1) % M != 0,
  ///                 then advanced by one effective step M*dt at the macro-step where (macro_step + 1) % M == 0
  ///                 (end of an M-step window), thus temporally consistent with the fast blocks (slow block,
  ///                 e.g. neutrals on stride=20). substeps and stride are ORTHOGONAL: stride=M,
  ///                 substeps=N -> N substeps of M*dt/N, once at the end of the window. COUPLING: between two
  ///                 catch-ups, the held block enters the Poisson sum with its STALE state (last frozen
  ///                 advance). step_cfl honors the cadence (dt <= cfl*h*substeps / (stride*w)).
  /// @param evolve   false = FROZEN species (fixed background): not advanced in time, but seen by the
  ///                 system Poisson (and, in the future, by coupled sources)
  /// @param implicit_vars  IMEX only: names of the conservative variables to treat IMPLICITLY in
  ///                 the source step (backward-Euler); the others stay explicit (forward Euler). The
  ///                 mask is CARRIED BY THE BLOCK / time policy (and NOT by the model): the
  ///                 SAME model can thus be reused with different implicit treatments. EMPTY
  ///                 (default) + EMPTY implicit_roles -> model default (Model::is_implicit, or all
  ///                 implicit absent a trait) -> bit-identical. Resolved against the conservative names
  ///                 of the block; an absent name raises an EXPLICIT error.
  /// @param implicit_roles IMEX only: same implicit mask but by physical ROLE ("density",
  ///                 "momentum_x", "energy", ...) instead of the name (cf. variable_roles). Union with
  ///                 implicit_vars. A role absent from the block raises an EXPLICIT error.
  /// @param newton IMEX only: options of the local Newton of the implicit source (backward-Euler),
  ///                 grouped in a POD (ADC-214; cf. NewtonOptions). max_iters (default 2 = historical
  ///                 constant), rel_tol / abs_tol (PER-CELL stopping criterion
  ///                 ||F||_inf <= abs_tol + rel_tol*||F0||_inf, 0/0 = disabled -> historical fixed
  ///                 iterations), fd_eps (step of the finite-difference Jacobian, default 1e-7),
  ///                 damping (damping W -= damping*delta in (0, 1], 1 = full Newton),
  ///                 fail_policy (kFailNone / kFailWarn / kFailThrow: reaction to failed cells).
  ///                 Default {} = historical constants, bit-identical.
  /// @param newton_diagnostics IMEX only: enables the block's Newton report (max residual,
  ///                 max iterations, failed cells -- non-finite / degenerate pivot / non-convergence),
  ///                 aggregated over the substeps of each advance and available via newton_report(name).
  ///                 OPT-IN: false (default) = historical path with no extra cost. Stays
  ///                 flat (a separate bool, outside the homogeneous family of convergence options).
  /// @param wave_speed_cache riemann='hll' + explicit ONLY: pre-computes model.wave_speeds ONCE per cell
  ///                 and direction in a scratch, then bounds each face by min/max of the two neighbor
  ///                 cells, instead of recalling wave_speeds per face. Net gain when wave_speeds is
  ///                 expensive (moment hierarchy). With recon='conservative' + limiter 'none' it is
  ///                 BIT-IDENTICAL to the per-face path; with a 2nd-order+ limiter it is a Davis bound on
  ///                 the cell values (different result). false (default) = per-face path unchanged. Wired
  ///                 on the FULL cartesian advance only: refused if riemann != 'hll', time IMEX, polar
  ///                 geometry, or a staircase/cutcell disc transport mode is active (explicit error,
  ///                 never a silent ignore).
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative", const std::string& time = "explicit",
                 int substeps = 1, bool evolve = true, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0, bool wave_speed_cache = false);

  /// Report of the implicit source Newton (IMEX) of a block, AGGREGATED over the substeps of the
  /// LAST advance of the block. Only exists if the block was added with newton_diagnostics=true
  /// (explicit error otherwise). Flat copy (no dependency on the numerics header).
  struct SourceNewtonReport {
    bool enabled;           ///< a report was computed (at least one IMEX advance played)
    bool converged;         ///< no failed cell on the last advance
    double max_residual;    ///< max over cells/substeps of ||F||_inf at the Newton exit
    double max_iters_used;  ///< max over cells/substeps of the iterations consumed
    double
        n_failed;  ///< number of (cells x substeps) failed (non-finite / pivot / non-convergence)
    double failed_i;     ///< i of ONE faulty cell (-1 if none; max index encoded)
    double failed_j;     ///< j of the same cell (-1 if none)
    double failed_comp;  ///< conservative component of the worst residual of that cell (-1 unknown)
  };
  SourceNewtonReport newton_report(const std::string& name) const;

  /// Adds a block whose model is LOADED AT RUNTIME from a shared library (.so)
  /// generated by the DSL (emit_cpp_brick -> ModelAdapter -> extern "C" factory). The .so must expose
  /// pops_model_nvars(), pops_make_model() (returns an IModel<NV>*) and pops_destroy_model(void*).
  /// HOST PATH (virtual dispatch, global periodic Rusanov a_max, explicit Euler): to
  /// prototype a brand-new model, written as formulas on the Python side, without recompiling the core. cf.
  /// dynamic_model.hpp.
  /// @param names variable names (introspection); default u0..u{NV-1}.
  /// @param recon MUSCL reconstruction of the face states (conservative): "none" (order 1) | "minmod"
  ///              | "vanleer" (order 2, TVD). The FLUX choice (HLLC/Roe) stays on the compiled path.
  void add_dynamic_block(const std::string& name, const std::string& so_path, int substeps = 1,
                         const std::vector<std::string>& names = {},
                         const std::string& recon = "none");

  /// Adds a block whose model is COMPILED AOT from a .so generated by the DSL
  /// (dsl.compile_aot / compile_or_jit(mode="compile")). Unlike the dynamic block (host
  /// path, IModel virtual dispatch, order-1 Rusanov), this block runs the PRODUCTION path: the .so
  /// runs assemble_rhs<Limiter, Flux> (HLLC/Roe by choice, order 2) and the core's SSPRK2/IMEX on the
  /// generated model; only flat arrays cross (extern "C" ABI, cf. compiled_block_abi.hpp).
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" (weno5: the .so allocates block_n_ghost = 3
  ///                ghosts, cf. compiled_block_abi.hpp)
  /// @param riemann "rusanov" | "hll" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"
  /// @param time    "explicit" | "imex" ONLY: the .so's extern "C" ABI only wires SSPRK2 in
  ///                explicit -- "ssprk3" / "euler" are rejected (cf. add_native_block, which carries them)
  void add_compiled_block(const std::string& name, const std::string& so_path,
                          const std::string& limiter = "minmod",
                          const std::string& riemann = "rusanov",
                          const std::string& recon = "conservative",
                          const std::string& time = "explicit", int substeps = 1,
                          const std::vector<std::string>& names = {},
                          double positivity_floor = 0.0);

  /// Changes the values of the RUNTIME parameters of an AOT block (add_compiled_block) WITHOUT recompiling the
  /// .so (P7-b). @p values is the COMPLETE block of values (order = sorted order of the names on the DSL side, cf.
  /// CompiledModel.runtime_param_names). The block must have been added via add_compiled_block AND
  /// declare at least one runtime parameter (dsl.Param(..., kind="runtime")); otherwise explicit error
  /// (a silent set_param on a block without a runtime param would mask a bug). Effect on the next step
  /// (the block closures read the SHARED block of values). @throws std::runtime_error if the block
  /// is unknown, has no runtime params, or if @p values does not have the right length.
  void set_block_params(const std::string& name, const std::vector<double>& values);

  /// Adds a block whose model is compiled into a NATIVE LOADER .so generated by the DSL
  /// (dsl.compile_native / compile(backend="production")). This is the PRODUCTION path: the loader
  /// inlines the header template pops::add_compiled_model<ProdModel>, which builds the closures on the
  /// REAL CONTEXT of the System (grid_context) and installs the block via install_block. The block then
  /// runs EXACTLY the native path of add_block (fill_boundary = MPI halos, assemble_rhs device),
  /// ZERO-COPY -- unlike add_compiled_block (.so + marshaling of flat arrays, extern "C" host
  /// ABI). Since the inline loader calls out-of-line methods of this module (install_block
  /// /grid_context/ensure_aux_width, exported POPS_EXPORT), the boundary is NOT a flat ABI:
  /// loader and module MUST share the same C++ ABI. add_native_block reads the loader's ABI key
  /// (pops_native_abi_key) and COMPARES it to abi_key(); a mismatch raises an EXPLICIT error (no UB).
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" (weno5: add_compiled_model reallocates
  ///                the block state to block_n_ghost = 3 ghosts after install_block, like add_block)
  /// @param riemann "rusanov" | "hll" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"
  /// @param time    "explicit" (SSPRK2) | "ssprk3" | "euler" | "imex" (the template marshals the explicit
  ///                RK scheme down to the loader's make_block, parity with add_block)
  /// @param gamma   adiabatic index of the block (set_density / inter-species couplings)
  /// @param stride  block cadence (1 = every step, default; cf. add_block)
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4, int substeps = 1,
                        bool evolve = true, int stride = 1, double positivity_floor = 0.0);

  /// ABI key of the module (compiler + C++ standard + signature of the pops headers, frozen at
  /// compilation). Compared to the key baked into a native loader .so by add_native_block; also exposed
  /// on the Python side so that emit_cpp_native_loader (or a diagnostic) can consult it.
  static std::string abi_key();

  /// @name AOT-COMPILED block seam (native parity, no marshaling)
  /// To wire a DSL-generated model by COMPOSING at COMPILATION time (production Kokkos + MPI + AMR
  /// binary), via the free template pops::add_compiled_model<Model> of
  /// pops/runtime/dsl_block.hpp: it builds the closures with block_builder.hpp on the REAL
  /// CONTEXT of the System (grid_context) -- so the block runs the same path as add_block (fill_boundary
  /// = MPI halos, assemble_rhs device), without copying the arrays. That is the difference with
  /// add_compiled_block (.so + host marshaling, runtime CPU prototyping).
  /// @{
  /// DEFAULT VISIBILITY (POPS_EXPORT): grid_context / install_block / ensure_aux_width are the
  /// only methods called by the header template add_compiled_model. A generated loader .so (DSL
  /// "production" path, cf. emit_cpp_native_loader / add_native_block) inlines this template and must
  /// resolve these symbols from the already loaded _pops module. Compiled with -fvisibility=hidden (pybind11),
  /// the module would not export them without this annotation and the loader's dlopen would fail.
  POPS_EXPORT GridContext grid_context();  ///< REAL mesh + BC + aux of the System (aux not owned)
  /// Installs a block from already-built closures (cf. add_compiled_model). The
  /// cons/prim descriptors carry the names AND the roles (M::conservative_vars()), used
  /// by inter-species couplings.
  POPS_EXPORT void install_block(const std::string& name, int ncomp, const VariableSet& cons_vars,
                                const VariableSet& prim_vars, double gamma, BlockClosures closures,
                                std::function<Real(const MultiFab&)> max_speed,
                                std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                                int substeps, bool evolve, int stride = 1);
  /// Guarantees that the state U of block @p name carries at least @p n_ghost ghosts (width of the
  /// spatial stencil). WENO5 reads 3 ghosts, > the 2 allocated by install_block; called by add_compiled_model
  /// (header) with block_n_ghost(limiter) AFTER install_block, so the native compiled path
  /// (loader .so) accepts weno5 -- SAME mechanism as add_block. No-op if U already has enough ghosts
  /// (none/minmod/vanleer, <= 2): allocation and data bit-identical to history. POPS_EXPORT:
  /// called by the header template add_compiled_model -> must be exported for the loader .so.
  POPS_EXPORT void set_block_ghosts(const std::string& name, int n_ghost);
  /// @}

  /// Configures the shared Poisson.
  /// @param rhs    only mode: "charge_density", f = sum_s elliptic_rhs_s(u_s)
  /// @param solver "geometric_mg" (any case, wall included) | "fft" (periodic, n = 2^k)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle": conducting wall at (L/2, L/2), radius wall_radius
  /// @param epsilon CONSTANT permittivity of the operator div(eps grad phi) = f. eps != 1 solves
  ///                eps lap phi = f (i.e. lap phi = f/eps). For a VARIABLE permittivity eps(x),
  ///                cf. set_epsilon_field (variable-coefficient operator, GeometricMG).
  /// @param abs_tol ABSOLUTE floor of the stopping criterion of the GeometricMG V-cycle (same units as the
  ///                residual). Default 0: purely relative criterion, historical behavior unchanged.
  ///                Set > 0 (problem scale), it makes solve_fields exit without cycling OUT OF
  ///                STEP on an already-converged state. No effect on the FFT solver (direct).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg", const std::string& bc = "auto",
                   const std::string& wall = "none", double wall_radius = 0.0, double epsilon = 1.0,
                   double abs_tol = 0.0);

  /// Configured field (Poisson) solver token, e.g. "geometric_mg" | "fft" | "fft_spectral"
  /// (the @p solver of the last set_poisson; default "geometric_mg"). Read by install_program for the
  /// Spec criterion-24 solver requirement check (a field operator that requires a named solver is
  /// rejected at install when the configured solver does not match) and exposed for introspection.
  std::string poisson_solver() const;

  /// Sets the TRANSPORT DOMAIN as a DISC centered at (@p cx, @p cy) with radius @p R
  /// (T2 work, CONTRACT inert by default). Materializes a 0/1 cell-centered mask (cell
  /// active when its center is inside the disc, level set hypot(x-cx, y-cy) - R < 0, SAME convention
  /// as the conducting wall of the Poisson). It is the FV counterpart of the elliptic wall: it lets the
  /// FV transport act on the true disc instead of the full cartesian square (otherwise the circle lives
  /// only in the Poisson wall -- the "cartesian ring edges" lock, cf. docs/HOFFART_FIDELITY.md). The
  /// mask makes possible a CONSERVATIVE mask-aware transport (zero normal flux at active/inactive faces).
  ///
  /// DISC TRANSPORT MODE (T5-PR3 work, @p mode): dispatches the transport advance of step() to
  /// the corresponding disc operator. Default "none" -> full cartesian path (assemble_rhs), BIT-
  /// IDENTICAL to history even after set_disc_domain (the mask is materialized but transport
  /// ignores it while the mode is "none"). "staircase" -> conservative masked transport (assemble_rhs_
  /// masked, 0/1 face gate, jagged boundary). "cutcell" -> cut-cell / embedded-boundary transport
  /// (assemble_rhs_eb, alpha_f apertures + kappa volume fraction, smooth boundary, order 2 interior).
  /// The mode is honored under Lie AND Strang (cf. set_time_scheme). A mode != "none" without a transportable
  /// cartesian block raises an EXPLICIT error at the step (never a silent full transport). Unknown mode
  /// -> error. R > 0 required; cartesian only (polar already bounds the ring by its radial
  /// walls -> explicit error).
  void set_disc_domain(double cx, double cy, double R, const std::string& mode = "none");

  /// Sets ONLY the disc transport mode (without (re)defining the disc): "none" | "staircase" |
  /// "cutcell". Useful to toggle the mode after set_disc_domain, or to reset it to "none"
  /// (back to the full cartesian path, bit-identical). Requesting a mode != "none" without a defined disc
  /// (set_disc_domain) raises an EXPLICIT error (the mode alone has no geometry to apply).
  void set_geometry_mode(const std::string& mode);

  /// @return the 0/1 cell-centered domain mask, ny*nx row-major (j slow, i fast). Without
  /// set_disc_domain, returns an ALL-ACTIVE mask (only 1.0): the transport sub-domain is
  /// the entire domain (default path). Diagnostic / contract verification.
  std::vector<double> disc_mask() const;

  /// Sets a VARIABLE permittivity eps(x), n*n row-major field (> 0), at the cell CENTER.
  /// The system Poisson operator becomes div(eps grad phi) = f, eps CARRIED BY THE OPERATOR
  /// (harmonic face coefficient, order 2) without 1/eps scaling of the right-hand side. Only
  /// the 'geometric_mg' solver supports it; requesting it with 'fft' (constant coefficient) raises an
  /// error. Takes precedence over the constant permittivity of set_poisson. Call before solve_fields.
  void set_epsilon_field(const std::vector<double>& eps);

  /// Sets an ANISOTROPIC permittivity eps_x(x), eps_y(x), two n*n row-major fields (> 0), at the CENTER
  /// of the cells. The system Poisson operator becomes div(diag(eps_x, eps_y) grad phi) = f:
  /// faces normal to x carry eps_x, those normal to y carry eps_y (harmonic face coefficients,
  /// order 2), CARRIED BY THE OPERATOR without 1/eps scaling of the right-hand side.
  /// eps_x == eps_y gives back the isotropic operator div(eps grad phi). Only 'geometric_mg' supports it;
  /// requesting it with 'fft' (constant coefficient) raises an error. Call before solve_fields.
  void set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                     const std::vector<double>& eps_y);

  /// Enables a REACTION term kappa(x) >= 0: the system Poisson operator goes from
  /// div(eps grad phi) = f to div(eps grad phi) - kappa phi = f (SCREENED Poisson / Helmholtz;
  /// kappa = 1/lambda_D^2 for Debye screening). n*n row-major field, carried by the operator
  /// GeometricMG (diagonal kappa, restricted to coarse levels). Only 'geometric_mg' supports it
  /// (error with 'fft'). Composable with set_epsilon_field. kappa = 0 everywhere => Poisson unchanged.
  void set_reaction_field(const std::vector<double>& kappa);

  /// Sets an out-of-plane magnetic field B_z(x, y) SHARED by the blocks, n*n row-major. Populates the
  /// extra aux component (B_z channel) read by the models that declare it (n_aux > 3);
  /// inert if no block reads B_z (aux channel stays at base width). B_z is static
  /// (external to the elliptic): derive_aux does not touch it. Call after having added the block
  /// (or before: the value is kept and applied when the aux channel widens).
  void set_magnetic_field(const std::vector<double>& bz);

  /// Designates a COMPRESSIBLE fluid block (4 var) as the source of the electron temperature T_e:
  /// the T_e aux channel (next canonical component) is filled with T = p/rho of this block, RECOMPUTED
  /// at each solve_fields. Has effect only if a block declares it reads T_e (n_aux > 4); otherwise stored
  /// and inert. It is the second EXTRA aux field (after B_z), populated by DERIVATION from a
  /// block (and not supplied by the user as B_z is).
  void set_electron_temperature_from(const std::string& name);

  /// Guarantees that the SHARED aux channel has at least @p ncomp components. Called by
  /// add_compiled_model (cf. dsl_block.hpp) with aux_comps<Model> when adding a block that reads extra
  /// auxiliary fields. Reallocating preserves the ADDRESS of the System's aux (the already-installed
  /// block closures point to &aux), and re-applies B_z if it was supplied.
  /// POPS_EXPORT: called by add_compiled_model (header) -> must be exported for the loader .so.
  POPS_EXPORT void ensure_aux_width(int ncomp);

  /// Sets a NAMED aux field (ADC-70 phase 1) on the canonical component @p comp (>= kAuxNamedBase
  /// = 5), row-major n*n (cartesian) / nr*ntheta (polar) array. The System does NOT know the
  /// names: the FACADE (pops.System.set_aux_field) resolves name -> comp via the block's table (from
  /// CompiledModel.aux_extra_names) and calls this. PERSISTENT STATIC field: stored (re-applied
  /// after a channel reallocation) and populated right away if the channel is wide enough. @throws if
  /// comp < kAuxNamedBase (components reserved for phi/grad/B_z/T_e: dedicated paths), if the size does not
  /// match the grid, or if no block declares a field at this index (channel too narrow).
  void set_aux_field_component(int comp, const std::vector<double>& field);

  /// Declares a per-field aux HALO policy (ADC-369) for the NAMED component @p comp (>= kAuxNamedBase):
  /// @p bc_type is pops::BCType (Foextrap=1 / Dirichlet=2), @p value the Dirichlet boundary value
  /// (ignored for Foextrap). Applied by solve_fields AFTER the shared aux ghost fill, overriding only
  /// this component's PHYSICAL-face ghosts (periodic faces -- periodic domain, polar theta -- keep their
  /// wrap). The FACADE (pops.System.set_aux_field(..., halo=pops.AuxHalo(...))) resolves name -> comp and
  /// calls this. No policy declared -> the shared aux BC, bit-identical. @throws on a reserved/too-narrow
  /// component or an unsupported type.
  void set_aux_field_halo_component(int comp, int bc_type, double value);

  /// Reads a NAMED aux field (component @p comp >= kAuxNamedBase): valid cells of the aux channel,
  /// row-major n*n (cartesian) / nr*ntheta (polar). Counterpart of potential() for a named
  /// component. Is 0 everywhere as long as no set_aux_field_component has written this component (the aux
  /// channel is initialized to zero and solve_fields never touches components >= kAuxNamedBase).
  std::vector<double> aux_field_component(int comp) const;

  /// Sets the density of a species (component 0), n*n row-major array. The other
  /// components (momentum, energy) are set to the at-rest equilibrium.
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Initializes the state of a block from its PRIMITIVE variables (rho, u, v, p ...): @p prim is
  /// a flat ncomp*n*n component-major array in the order of primitive_vars(name). Each cell
  /// is converted to CONSERVATIVE variables by the block's MODEL conversion (M.to_conservative),
  /// then written into the state. Ergonomic counterpart of set_density for a model with several primitives
  /// (compressible 4 var: p; isothermal 3 var; scalar 1 var: identity). cf. get_primitive_state.
  void set_primitive_state(const std::string& name, const std::vector<double>& prim);

  /// Reads the CONSERVATIVE state of the block and converts it to PRIMITIVE variables via the model
  /// conversion (M.to_primitive). @return a flat ncomp*n*n component-major array in the order of
  /// primitive_vars(name) (diagnostics: velocities, pressure). Exact round-trip with set_primitive_state.
  std::vector<double> get_primitive_state(const std::string& name);

  /// Type-erasure of the POINTWISE (one cell) cons <-> prim conversion of a block: in/out are
  /// arrays of ncomp doubles. Installed by install_block / add_compiled_model / push_dynamic from
  /// the block's model, consumed by set_primitive_state / get_primitive_state.
  using CellConvert = std::function<void(const double* in, double* out)>;
  /// Installs the pointwise cons <-> prim conversions of a block (after install_block). Called by
  /// the header template add_compiled_model (compiled model); the native path add_block and the dynamic
  /// .so path set them directly. POPS_EXPORT: resolved by the native loader through dlopen.
  POPS_EXPORT void set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                       CellConvert cons_to_prim);

  /// Installs the optional STEP BOUNDS of a block (after install_block): reduction of the
  /// max source frequency (HasSourceFrequency trait, bound dt <= cfl*substeps/(stride*mu)) and of the
  /// min admissible step (HasStabilityDt trait, bound dt <= dt_adm*substeps/stride, without cfl).
  /// EMPTY functions = the block imposes no bound (historical step policy, bit-identical).
  /// Called by add_block and by the template add_compiled_model (cf. dsl_block.hpp) with the
  /// compiled closures of block_builder (make_source_frequency / make_stability_dt).
  /// POPS_EXPORT: resolved by the native loader through dlopen.
  POPS_EXPORT void set_block_dt_bounds(const std::string& name,
                                      std::function<Real(const MultiFab&)> source_frequency,
                                      std::function<Real(const MultiFab&)> stability_dt);

  /// Adds a GLOBAL time-step bound, evaluated ONCE per step (host) by step_cfl /
  /// step_adaptive: dt <= fn() when fn() > 0 and finite (otherwise the bound does not constrain this step).
  /// It is the hook for NON cell-local constraints: multi-block coupling, Schur/Poisson
  /// stage, AMR/scheduler, or a user policy (startup ramp...). @p label
  /// names the bound in last_dt_bound() ("global:<label>"). A Python callback is acceptable HERE
  /// (one evaluation per step, never per cell).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// Name of the ACTIVE bound (the one that set dt) of the last step_cfl: "transport:<block>",
  /// "source_frequency:<block>", "stability_dt:<block>", "global:<label>", "degenerate" (no evolving
  /// block), or "" if no step_cfl has run. Diagnostic of the step policy.
  std::string last_dt_bound() const;

  /// Adds an IONIZATION coupling (operator-split): rate k n_e n_g; a neutral becomes an ion
  /// and an electron. Mass transferred from the neutral to the ion (n_i + n_g conserved). The three blocks
  /// must exist. First inter-species source brick (on the density, comp 0).
  void add_ionization(const std::string& electron, const std::string& ion,
                      const std::string& neutral, double rate);

  /// Adds an inter-species COLLISION / friction (operator-split): force k (u_a - u_b) on the
  /// momentum, opposite on each species (total momentum conserved). The two
  /// blocks must be fluids (>= 3 variables). Frictional heating neglected (refinement).
  void add_collision(const std::string& a, const std::string& b, double rate);

  /// Adds an inter-species THERMAL EXCHANGE (operator-split): heat flux k (T_a - T_b)
  /// on the energy, opposite on each species (total energy conserved); T = p/rho. The two
  /// blocks must be compressible Euler (4 variables, energy equation).
  void add_thermal_exchange(const std::string& a, const std::string& b, double rate);

  /// Enables a Schur-condensed SOURCE STAGE on block @p name (EXPLICIT / IMPLICIT splitting,
  /// cf. docs/SCHUR_CONDENSATION_DESIGN.md sections 5-6). It is the OPT-IN of the pops.Split(
  /// hyperbolic=Explicit, source=CondensedSchur) policy: at each step, the block does its EXPLICIT
  /// hyperbolic transport as today, THEN this source stage replaces the explicit /
  /// IMEX source by the condensed stage (CondensedSchurSourceStepper, #126), AUTONOMOUS and solved in C++ (no
  /// per-cell Python callback). The default path (without this call) stays BIT-IDENTICAL.
  ///
  /// CONTRACT (validated HERE, before any step): the block MUST expose the Density / MomentumX /
  /// MomentumY roles (Energy optional) in its conservative descriptor; a missing role raises an
  /// EXPLICIT error. The B_z field must be available (set_magnetic_field called): the aux is widened to
  /// the B_z channel and an absent B_z raises an error. The block reuses the phi potential of the system
  /// Poisson as a warm start (solve_fields runs at the head of step), but the source stage solves its
  /// OWN condensed elliptic operator (it does not duplicate solve_fields).
  /// @param kind  only "electrostatic_lorentz" for now (ElectrostaticLorentzCondensation).
  /// @param theta theta-scheme in (0,1] (0.5 = Crank-Nicolson, 1 = backward Euler).
  /// @param alpha electrostatic coupling constant of the source sub-system (d_t(-Lap phi) =
  ///              -alpha div(rho v)).
  /// @param opts  settings grouped in a POD (ADC-214; cf. SourceStageOptions): tolerance / budget of the
  ///              Krylov solve (krylov_tol / krylov_max_iters, <= 0 = historical defaults 1e-10;
  ///              400 cartesian, 600 polar), field DESCRIPTORS (density / momentum_x /
  ///              momentum_y / energy; "" = canonical role, otherwise stable role name or variable
  ///              name of the block; energy == "none" disables the energy; CARTESIAN only for an
  ///              override, the polar stepper rejects it), and bz_aux_component (< 0 = canonical
  ///              B_z channel). Default {} = historical bit-identical behavior. These fields were a
  ///              long list of four adjacent `std::string` interchangeable at the call site.
  void set_source_stage(const std::string& name, const std::string& kind, double theta,
                        double alpha, const SourceStageOptions& opts = {});

  /// Time SPLITTING POLICY of the macro-step (hyperbolic transport H + source stage S):
  ///  - "lie"    (default): H(dt); S(dt) once (Godunov, 1st order). BIT-IDENTICAL to history.
  ///  - "strang": H(dt/2); S(dt); H(dt/2) (symmetric, 2nd order as soon as H and S are).
  /// The Strang scheme RE-SOLVES solve_fields between the stages so that each half-advance reads a phi
  /// consistent with the current density (the SINGLE head solve_fields, sufficient for Lie, does not suffice
  /// for the 2nd Strang half-advance); cf. docs/HOFFART_STEP_SEQUENCE.md and SystemStepper::step_strang.
  /// Reuses the SAME bricks (s.advance, source stage): no new stepper. An unknown scheme
  /// raises an EXPLICIT error. Without a call, the path stays BIT-IDENTICAL (Lie).
  void set_time_scheme(const std::string& scheme);

  /// Gauss-law policy. "restart" (default): solve_fields
  /// re-solves -Delta phi = f at each step (BIT-IDENTICAL to history). "evolve": after the first
  /// solve (phi^0), solve_fields no longer re-solves the Poisson and derives the aux from the CURRENT phi
  /// that the Schur source stage evolves in-place -> -Delta phi evolution without restart (Gauss
  /// imposed only at t=0). Has effect only with a condensed source stage. Unknown scheme -> explicit error.
  void set_gauss_policy(const std::string& policy);

  /// Adds a GENERIC inter-species COUPLED SOURCE described by a BYTECODE (pops.dsl.CoupledSource,
  /// P5 phase 1, EXPLICIT forward-Euler splitting after transport). Unlike the named couplings
  /// (add_ionization / add_collision / add_thermal_exchange) which freeze a formula, this one reads
  /// (block, role) fields as INPUT and writes source terms (block, role) computed by symbolic
  /// EXPRESSIONS compiled to postfix bytecode (stack machine, evaluated in the same
  /// for_each_cell device; no per-cell Python callback). Reuses EXACTLY the coupling
  /// application seam (P->couplings); MPI-safe (iteration over the local fabs,
  /// local_size()==0 -> no-op).
  ///
  /// FLAT ABI (no C++ object crosses the boundary):
  /// @param prog      bytecode description of the coupling grouped in a POD (ADC-214; cf.
  ///                  CoupledSourceProgram): in_blocks / in_roles (inputs read and their roles),
  ///                  consts (.param() parameters, loaded after the inputs), out_blocks / out_roles
  ///                  (targets of each term), prog_ops / prog_args / prog_lens (concatenated opcodes
  ///                  of ALL terms, stack machine cf. CsOp, parallel arguments, and length
  ///                  per term), and freq_prog_ops / freq_prog_args (OPTIONAL program of a
  ///                  PER-CELL frequency mu(U), same stack machine / register table; EMPTY =
  ///                  constant frequency only, bit-identical). These arrays were a long list
  ///                  of `std::vector` of the same type, interchangeable at the call site.
  /// @param frequency  declared CONSTANT frequency mu [1/s] of the coupling (audit wave 3,
  ///                   CoupledSource.frequency): step bound dt <= cfl / mu aggregated by step_cfl /
  ///                   step_adaptive (the couplings apply ONCE per macro-step, the bound
  ///                   is on the macro-dt, without a substeps/stride factor). <= 0 (default) = no
  ///                   bound, bit-identical. Stays flat (a double, outside the homogeneous family).
  /// @param label      name of the coupling (reason "coupled_source:<label>" of last_dt_bound). Stays
  ///                   flat (a string, outside the homogeneous family). When prog.freq_prog_ops/_args are
  ///                   non-empty, step_cfl / step_adaptive reduces the MAX of mu over the cells
  ///                   (global all_reduce_max) and bounds dt <= cfl / max(mu) (reason
  ///                   "coupled_source:<label>"). max(mu) <= 0 = no bound this step.
  /// Unknown blocks / roles, an exceeded capacity or a malformed program raise an EXPLICIT
  /// error (before any step). Without a call, the default path stays BIT-IDENTICAL.
  void add_coupled_source(const CoupledSourceProgram& prog, double frequency = 0.0,
                          const std::string& label = "coupled_source");

  POPS_EXPORT void solve_fields();  ///< solves Poisson then derives aux = (phi, grad phi); exported
                                   ///< so a compiled program .so resolves it via ProgramContext
                                   ///< (the other seam accessors below are likewise POPS_EXPORT)
  /// Per-stage field solve (ADC-409): SAME elliptic solve + aux derivation as solve_fields(), but
  /// block @p block_idx assembles its Poisson RHS from @p U_stage instead of its live state (the
  /// other blocks keep theirs). This re-fills the SHARED aux with phi(U_stage) so a field-coupled
  /// multi-stage compiled Program can re-solve the fields from each STAGE state -- the stages run
  /// sequentially, so stage k's RHS (called right after this) reads phi from stage k's own state
  /// before the next stage overwrites the aux. With block_idx 0 and U_stage = U^n (the first stage)
  /// it is identical to solve_fields(). POPS_EXPORT: resolved by a compiled program .so (ProgramContext)
  /// across the dlopen boundary. @throws std::out_of_range if @p block_idx is not a valid block.
  POPS_EXPORT void solve_fields_from_state(int block_idx, const MultiFab& U_stage);
  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): SAME elliptic solve + aux
  /// derivation as solve_fields(), but the system Poisson RHS is assembled from the SIMULTANEOUS stage
  /// states of MULTIPLE blocks at once -- every coupled block reads its OWN stage state, not a single-
  /// target override. @p U_stages is indexed BY BLOCK INDEX (its size must equal n_blocks()); entry b
  /// != nullptr -> block b contributes its stage state, entry b == nullptr -> block b contributes its
  /// live state. With every entry pointing at the corresponding live state it is bit-identical to
  /// solve_fields(). The codegen lowers P.solve_fields_from_blocks([...]) to this -- the seam a multi-
  /// species field-coupled step uses (the IR commit_many guarantee: no operator observes a partially
  /// committed group). POPS_EXPORT: resolved by a compiled program .so (ProgramContext) across the
  /// dlopen boundary. @throws std::invalid_argument if @p U_stages is not sized to n_blocks().
  POPS_EXPORT void solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages);
  /// @name Named multi-elliptic fields (ADC-428)
  /// A SECOND elliptic solve (beyond the default Poisson) for a user-named field
  /// (m.elliptic_field("phi2", rhs=..., aux=[...])). The named field owns its RHS (a per-block brick,
  /// distinct from the default elliptic coupling), a DEDICATED native elliptic solver instance, and its
  /// OWN aux output components (the model's named aux slots). The default Poisson path
  /// (solve_fields / solve_fields_from_state) is untouched / bit-identical. POPS_EXPORT: resolved by the
  /// generated problem.so / native loader across the dlopen boundary.
  /// @{
  /// Solve named @p field's elliptic problem from block @p block_idx's stage state @p U_stage and write
  /// its solved phi (+ centered gradient) into the field's own aux components. The codegen lowers
  /// P.solve_fields(field=name, state=U) to this. @throws if @p field is unregistered, the block index
  /// is invalid, or the geometry is polar (cartesian only for now).
  POPS_EXPORT void solve_fields_from_state(const std::string& field, int block_idx,
                                          const MultiFab& U_stage);
  /// Register named @p field's aux output components (where its solved phi / centered grad land). Called
  /// by the native loader for each m.elliptic_field once the block is installed. @p gx_comp / @p gy_comp
  /// < 0 => only phi is written (the field declared fewer than 3 aux slots).
  POPS_EXPORT void register_elliptic_field(const std::string& field, int phi_comp, int gx_comp,
                                          int gy_comp);
  /// Attach named @p field's RHS closure (+= elliptic_field_rhs(U)) to block @p block_name. Called by
  /// the native loader (make_poisson_rhs of the per-field brick). @throws if the block is unknown.
  POPS_EXPORT void set_block_elliptic_field(const std::string& block_name, const std::string& field,
                                           std::function<void(const MultiFab&, MultiFab&)> rhs);
  /// @}
  void step(double dt);  ///< solve_fields, then advances each block according to its scheme
  void advance(double dt, int nsteps);

  /// Advances one step at dt = cfl * h / max wave speed of the system. @return the dt used.
  double step_cfl(double cfl);
  /// Diagnostic (ADC-182): {w, i, j} of the GLOBAL cell that dominates the transport
  /// CFL bound of the block -- to locate a realizability erosion / a collapsing dt.
  /// On demand, off the hot path (step/step_cfl unchanged).
  std::array<double, 3> dt_hotspot(const std::string& name);

  /// Advances one MULTIRATE macro-step: the slowest block sets the macro-step, each block
  /// that is faster is sub-cycled n = ceil(w_block / w_min) times. @return the macro-step.
  double step_adaptive(double cfl);

  /// @name Profiling (Spec 3 section 29-30, ADC-459)
  /// Per-phase / per-brick wall-clock timing of the step. Disabled by default (no hot-path cost
  /// when off). enable_profiling() then step()/step_cfl() then profile_report() returns the table;
  /// reset_profiling() clears it. Per-rank (no MPI reduction); the per-Program-node granularity is
  /// wired through the compiled-program path as a follow-up.
  /// @{
  void enable_profiling();
  void disable_profiling();
  bool is_profiling() const;
  void reset_profiling();
  std::string profile_report() const;
  /// The System-owned Profiler (a non-owning reference; lives as long as the System). A compiled time
  /// Program reaches it through ProgramContext::profile_node to time each Program node into the SAME
  /// table sim.profile_report() renders -- so per-node scopes ("node:rhs2", ...) accumulate alongside
  /// the coarse "step" / "field_solve" phases. POPS_EXPORT: a generated problem.so resolves it across
  /// the dlopen boundary like the other ProgramContext seam accessors (block_state, grid_context).
  POPS_EXPORT runtime::program::Profiler& profiler();
  /// @}

  /// @name Primitives for a time integrator written in Python
  /// solve_fields(); R = eval_rhs(name); U = get_state(name); ...; set_state(name, U).
  /// @{
  std::vector<double> eval_rhs(const std::string& name);   ///< -div F + S, size ncomp*n*n
  std::vector<double> get_state(const std::string& name);  ///< U, ncomp*n*n (component-major)
  void set_state(const std::string& name, const std::vector<double>& u);
  int n_vars(const std::string& name) const;
  /// Variable names of a block (introspection): kind = "conservative" | "primitive".
  std::vector<std::string> variable_names(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// PHYSICAL roles of the variables of a block (parallel to variable_names): "density",
  /// "momentum_x", "energy", ... or "custom" if the block does not provide its roles. This is what
  /// the inter-species couplings resolve (index_of(role)) instead of a literal index.
  std::vector<std::string> variable_roles(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// Adiabatic index (gamma) of the block, read by the inter-species couplings (collision, thermal
  /// exchange, T_e). Equals the historical default 1.4 unless the block declares it (add_block: ModelSpec
  /// gamma; compiled / dynamic block: optional symbol pops_compiled_gamma of the .so ABI).
  double block_gamma(const std::string& name) const;
  /// @}

  /// @name Compiled time-program seam (epic ADC-399 / ADC-401)
  /// Lets a generated problem.so (via pops::runtime::program::ProgramContext) run a time Program during
  /// sim.step(dt): install a macro-step body and reach per-block storage. The .so reimplements nothing
  /// -- it composes these primitives (solve_fields(); block_rhs_into(b, U, R); saxpy(U, dt, R)).
  /// @{
  /// Install the macro-step body. When set, SystemStepper::step calls it instead of the historical
  /// path (and keeps t / macro_step coherent). Pass an empty std::function to clear it.
  /// POPS_EXPORT: a generated problem.so resolves these across the dlopen boundary (RTLD_GLOBAL), like
  /// install_block / grid_context; without default visibility the .so could not find them (_pops is
  /// built with hidden visibility).
  POPS_EXPORT void install_program_step(std::function<void(double)> step);
  /// Set the compiled-Program macro-step cadence (ADC-411): SYSTEM-level @p substeps and @p stride
  /// around the installed program closure (cf. SystemStepper::step). @p substeps subdivides each
  /// effective step into @p substeps calls program_step_(eff_dt/substeps); @p stride runs the whole
  /// program once per @p stride macro-steps with eff_dt = stride*dt (GLOBAL hold-then-catch-up, the
  /// clock still ticks every macro-step). Both must be >= 1 (throws std::invalid_argument otherwise).
  /// Default 1/1 -> byte-identical to a single program_step_(dt) call. Kept SEPARATE from
  /// install_program so the generated .so ABI is untouched (the cadence is runtime metadata).
  /// NOTE: substeps > 1 is bit-exact vs native substeps ONLY for an UNCOUPLED / transport-only program
  /// (program_step_ re-runs the whole program, solve_fields included); stride is GLOBAL (whole-system),
  /// equal to native per-block stride only for a single-block system. See SystemStepper::step.
  POPS_EXPORT void set_program_cadence(int substeps, int stride);
  /// Number of blocks (species) installed.
  POPS_EXPORT int n_blocks() const;
  /// The conservative state MultiFab of block @p b (zero-copy, non-owning reference).
  POPS_EXPORT MultiFab& block_state(int b);
  /// @name Compiled-Program NAME-based block binding (Spec 3 criterion 23, ADC-457)
  /// A compiled Program numbers its blocks in P.state declaration order (the .so's
  /// pops_program_block_name table); the System numbers its blocks in add_block / add_equation order
  /// (block_names). They need NOT agree. install_program reads the .so's block names, matches each to
  /// the System block of that name, and stores the resulting program-index -> system-index map here so
  /// ProgramContext::state / rhs_into / commit resolve a Program block index to the name-matched System
  /// block -- NOT the positional index. An EMPTY map is the identity (a single-block or order-matching
  /// Program lowers byte-identically; ProgramContext built directly, e.g. in a C++ test, also sees
  /// identity). Lives in Impl (private to the _pops TU) so it survives the dlopen boundary; the seam is
  /// POPS_EXPORT so the generated .so and ProgramContext resolve it via RTLD_GLOBAL.
  /// @{
  /// Install the program-index -> system-index map (entry p = the System block index of Program block
  /// p). Empty clears it (identity). Set by install_program after matching the .so's block names.
  POPS_EXPORT void set_program_block_map(const std::vector<int>& prog_to_sys);
  /// The installed program-index -> system-index map (empty = identity). Read by ProgramContext.
  POPS_EXPORT const std::vector<int>& program_block_map() const;
  /// @}
  /// R <- -div F(U) + S(U, aux) for block @p b (the block's frozen-Poisson residual closure).
  POPS_EXPORT void block_rhs_into(int b, MultiFab& U, MultiFab& R);
  /// R <- -div F(U) for block @p b -- the SAME flux divergence as block_rhs_into but WITHOUT the
  /// model's default/composite source (Poisson frozen, ghosts filled identically). The block's
  /// flux-only closure is the rhs_into path on SourceFreeModel<Model> (the zero-source adapter the
  /// IMEX explicit half-step already uses), so the flux / ghost / geometry handling is bit-identical
  /// -- only the source is dropped (with limiter='none'; the HLL wave-speed cache -- rejected on the
  /// aot/production backends compiled Programs use -- is the only path where cached cell-center speeds
  /// differ from the per-face reconstruction). A compiled time Program's hyperbolic stage
  /// (ProgramContext::neg_div_flux_default_into) reads it so a Lie/Strang split assembles "flux but no
  /// source" without the default source leaking in (epic ADC-399 / ADC-425, spec criterion 17). FAILS
  /// LOUD (std::runtime_error) on a block whose path did not build the closure (the host .so prototype
  /// loader) -- never a silent source leak. POPS_EXPORT: resolved by the generated problem.so across the
  /// dlopen boundary, like block_rhs_into.
  POPS_EXPORT void block_neg_div_flux_into(int b, MultiFab& U, MultiFab& R);
  /// R <- S(U, aux) for block @p b -- the model's default/composite SOURCE only, WITHOUT the flux
  /// divergence (the exact MIRROR of block_neg_div_flux_into, which is flux without source). Together
  /// they split block_rhs_into = -div F + S into its two halves (ADC-430, sibling of ADC-425). The
  /// block's source-only closure evaluates m.source per cell into R (the SAME source term assemble_rhs
  /// adds), with NO numerical-flux dispatch -- so it is flux-template agnostic (unlike a zero-flux model
  /// adapter, which HLL/Roe would not zero) and bit-identical to the source term of rhs_into. A compiled
  /// time Program's source stage (ProgramContext::source_default_into) reads it so a Lie/Strang split
  /// assembles "the default source but no flux" -- P.rhs(flux=False, sources with "default") -- without
  /// the -div F base leaking in (epic ADC-399 / ADC-430, spec: rhs flux=False is source-only). FAILS
  /// LOUD (std::runtime_error) on a block whose path did not build the closure (the host .so prototype
  /// loader) -- never a silent flux leak. POPS_EXPORT: resolved by the generated problem.so across the
  /// dlopen boundary, like block_neg_div_flux_into.
  POPS_EXPORT void block_source_into(int b, MultiFab& U, MultiFab& R);
  /// The maximum |wave speed| of block @p b evaluated on @p U -- the SAME per-block reduction
  /// step_cfl reads (BlockState::max_speed, the HasStabilitySpeed / max_wave_speed closure set at
  /// add_block time): a collective reduction over the block's cells. A compiled time Program reads it
  /// (ProgramContext::max_wave_speed) to express its own dt bound (epic ADC-399 / ADC-417, spec s18).
  /// REUSES the block's wave-speed closure -- it does not recompute the speed. POPS_EXPORT: resolved by
  /// the generated problem.so across the dlopen boundary, like the other seam accessors.
  POPS_EXPORT Real block_max_speed(int b, const MultiFab& U) const;
  /// The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)) --
  /// the SAME hmin the native CFL uses (SystemStepper::cfl_grid_h). A compiled time Program reads it
  /// (ProgramContext::hmin) to express its own dt bound (epic ADC-399 / ADC-417, spec s18). POPS_EXPORT:
  /// resolved by the generated problem.so across the dlopen boundary.
  POPS_EXPORT Real cfl_min_dx() const;
  /// A fresh scalar field co-distributed with the System mesh: block 0's BoxArray and
  /// DistributionMapping, @p n_comp components, @p n_ghost ghost layers, zero-initialized. Scratch a
  /// compiled time Program allocates for a matrix-free Krylov solve (the residual / search-direction
  /// fields that feed cg_solve / bicgstab_solve via ProgramContext::laplacian); shares the block
  /// (ba, dm) so a per-cell kernel pairs it with the state and aux by local fab index.
  POPS_EXPORT MultiFab alloc_scalar_field(int n_comp, int n_ghost);
  /// @name Multistep history (epic ADC-399 / ADC-406a)
  /// SYSTEM-OWNED history ring buffers for multistep schemes (Adams-Bashforth and friends): a named
  /// field carried ACROSS macro-steps (e.g. the previous RHS R_{n-1}). The history lives in the System
  /// (a HistoryManager in Impl), NOT in the .so closure, so a later checkpoint slice (ADC-406b) can
  /// serialize it. A generated problem.so reaches it through ProgramContext::history / store_history /
  /// rotate_histories; these are POPS_EXPORT so the .so resolves them across the dlopen boundary.
  /// @{
  /// Register (idempotent) a history named @p name with maximum lag @p lag (>= 1): a ring buffer of
  /// depth @p lag + 1 (slot 0 = the CURRENT value, slot k = the value k macro-steps back after the
  /// rotates), each slot a MultiFab co-distributed with block 0 (the block state's ncomp, so it can
  /// hold a full RHS / state) and zero-initialized. Re-registering with the SAME lag returns the
  /// existing current slot; a DIFFERENT lag throws (a name binds one depth). Returns the current slot
  /// [0] -- the read target for lag = 1 after one rotate. @throws if @p lag < 1 or no block exists yet.
  POPS_EXPORT MultiFab& register_history(const std::string& name, int lag);
  /// The history slot @p lag macro-steps back (lag 0 = the current slot, lag 1 = the previous step's
  /// stored value, ...). @throws if @p name is unknown, @p lag exceeds the registered depth, or the
  /// history has not been stored yet ("history '<name>' with lag=<lag> was requested but not
  /// initialized") -- a read before the first store is a fail-loud configuration error (spec error 17).
  POPS_EXPORT MultiFab& read_history(const std::string& name, int lag);
  /// Copy @p value (valid cells) into the CURRENT slot [0] of history @p name and mark it initialized.
  /// On the FIRST store the value is also broadcast into EVERY deeper slot (the cold-start fill: a
  /// multistep scheme's step 0 then reads the same value at every lag, degenerating to a one-step
  /// method -- deterministic and machine-precision reproducible). @throws if @p name is unknown. The
  /// caller is responsible for layout compatibility: the ring slots share the block's (ba, dm, ncomp),
  /// so a value built from the same block matches (lincomb is a valid-cell copy, no layout check).
  POPS_EXPORT void store_history(const std::string& name, const MultiFab& value);
  /// Shift every history ring buffer one step (slot k <- slot k-1, for k = depth-1 .. 1), called ONCE
  /// at the end of each macro-step (the generated step body emits ctx.rotate_histories() last). The
  /// current slot [0] is recycled (it gets the oldest buffer; the next store overwrites it before any
  /// read). O(1) handle swaps -- no deep copy. No-op when no history exists.
  POPS_EXPORT void rotate_histories();
  /// @name Multistep history checkpoint/restart (epic ADC-399 / ADC-406b)
  /// SERIALIZE / RESTORE the System-owned history rings across a checkpoint. The history lives in the
  /// System (HistoryManager in Impl), so the checkpoint facade (sim.checkpoint / sim.restart) gathers
  /// and restores it DIRECTLY -- no .so checkpoint_extra ABI is needed for the buffers (only the program
  /// hash, installed_program_hash() below, is recorded to reject a restart against a different Program).
  /// @{
  /// Names of every registered history (the keys of the HistoryManager), so the facade can iterate the
  /// rings to serialize. Empty when no history exists (the single-step paths). Order is the map order.
  POPS_EXPORT std::vector<std::string> history_names() const;
  /// Ring depth (max lag + 1) of history @p name. @throws if @p name is unknown.
  POPS_EXPORT int history_depth(const std::string& name) const;
  /// Component count of the slots of history @p name (the block's ncomp). @throws if unknown.
  POPS_EXPORT int history_ncomp(const std::string& name) const;
  /// GLOBAL (collective, MPI-safe) gather of slot @p slot (0 = current, k = k macro-steps back) of
  /// history @p name into a component-major buffer of size ncomp*ny*nx, EXACTLY like state_global
  /// (every rank fills its local boxes then all_reduce_sum). All ranks MUST call it. @throws if @p name
  /// is unknown or @p slot is out of range. Reads the slot even before the first store (the checkpoint
  /// of a never-stored ring is its zero fill); the initialized flag is serialized separately.
  POPS_EXPORT std::vector<double> history_global(const std::string& name, int slot) const;
  /// Whether history @p name has been stored at least once (the cold-start fill already happened). The
  /// facade records it so a restart restores the initialized state without a phantom re-fill. @throws
  /// if @p name is unknown.
  POPS_EXPORT bool history_initialized(const std::string& name) const;
  /// RESTORE (restart) slot @p slot of history @p name from a GLOBAL component-major buffer (same layout
  /// as history_global / set_state): the owner rank writes its box, the others are no-ops (MPI-safe, all
  /// ranks call it). Registers the ring (depth = max(slot)+1) if @p name is unknown yet, so the restart
  /// rebuilds the rings the program will re-register on its first step. @throws on a size mismatch.
  POPS_EXPORT void restore_history(const std::string& name, int slot,
                                  const std::vector<double>& values);
  /// Mark history @p name initialized (or not) after a restart: a restored, already-stored ring must
  /// read at lag without a phantom cold-start re-fill on its first post-restart store. @throws if
  /// @p name is unknown (restore its slots first).
  POPS_EXPORT void set_history_initialized(const std::string& name, bool initialized);
  /// @}
  /// Load a generated problem.so and install its compiled time Program. dlopens @p so_path, checks
  /// its ABI key against this module (fail-loud on mismatch), and calls its pops_install_program(this),
  /// which wraps the System in a ProgramContext and installs the macro-step closure. The .so resolves
  /// the seam accessors above via the global scope (same self-promotion as the native loader). Mirrors
  /// add_native_block; the .so stays loaded for the process lifetime.
  POPS_EXPORT void install_program(const std::string& so_path);
  /// IR hash of the installed compiled Program (the string returned by the .so's pops_program_hash),
  /// or "" if no program is installed. Recorded in the checkpoint (sim.checkpoint) so a restart against
  /// a DIFFERENT compiled Program is rejected fail-loud (the buffers / cadence would be meaningless).
  POPS_EXPORT std::string installed_program_hash() const;
  /// @name Scheduler value cache (epic ADC-399 / ADC-458, Spec 3 section 17-18 + 30)
  /// The held-node value cache (every(N).hold / accumulate_dt) lives in the SYSTEM (one CacheManager
  /// per installed Program), NOT in the .so step closure -- so the checkpoint can reach it, exactly the
  /// way the history rings do. Every ProgramContext (the step closure's copy and any fresh one) forwards
  /// its cache_* seam ops to this single manager. POPS_EXPORT so the generated problem.so resolves it
  /// across the dlopen boundary like the other ProgramContext seam accessors.
  /// @{
  /// The System-owned scheduler cache (a non-owning reference; lives as long as the System). A compiled
  /// Program's cache_store_aux / cache_restore_aux / cache_should_update reach it through ProgramContext.
  POPS_EXPORT runtime::program::CacheManager& program_cache();
  /// @name Scheduler-cache checkpoint/restart (Spec 3 section 30, ADC-458)
  /// SERIALIZE / RESTORE the System-owned cache across a checkpoint, mirroring the history seam: the
  /// facade (sim.checkpoint / sim.restart) gathers each VALID slot (gather_global, MPI-safe) and scatters
  /// it back (write_state) alongside the block state and histories. The program-hash guard
  /// (installed_program_hash) rejects a restart against a different compiled Program; a held scheduled
  /// node the checkpoint never recorded fails loud at restart (the facade compares the restored ids).
  /// @{
  /// Node ids of every VALID cached slot (ascending). Empty when no schedule cached a value.
  POPS_EXPORT std::vector<int> program_cache_nodes() const;
  /// The scheduled node name of slot @p node_id ("fields_from_state"), or "node_<id>" if it was stored
  /// without one (the current nameless codegen). Names a missing node verbatim at restart.
  POPS_EXPORT std::string program_cache_name(int node_id) const;
  /// The macro step at slot @p node_id's last recompute. @throws if absent.
  POPS_EXPORT int program_cache_last_update_step(int node_id) const;
  /// The accumulated skipped dt of slot @p node_id (accumulate_dt policy). 0 if none.
  POPS_EXPORT double program_cache_accumulated_dt(int node_id) const;
  /// The component count of slot @p node_id's cached value. @throws if absent.
  POPS_EXPORT int program_cache_ncomp(int node_id) const;
  /// The ghost-cell width of slot @p node_id's cached value (1 for the aux, the block-state width for a
  /// held scratch) -- serialized so restore rebuilds with the same ngrow. @throws if absent.
  POPS_EXPORT int program_cache_ngrow(int node_id) const;
  /// GLOBAL (collective, MPI-safe) gather of slot @p node_id's cached MultiFab into a component-major
  /// buffer of size ncomp*ny*nx, EXACTLY like state_global / history_global. All ranks MUST call it.
  /// @throws if @p node_id is absent.
  POPS_EXPORT std::vector<double> program_cache_global(int node_id) const;
  /// RESTORE (restart) slot @p node_id from a GLOBAL component-major buffer (same layout as
  /// program_cache_global / set_state): allocate a value MultiFab co-distributed with block 0 (@p ncomp
  /// components), scatter the buffer into it (owner rank writes, others no-op -- MPI-safe, all ranks
  /// call it), and re-key the slot with its bookkeeping (@p name may be empty). @throws if no block
  /// exists yet (the cache value is co-distributed with block 0's storage).
  POPS_EXPORT void restore_program_cache(int node_id, int ncomp, int ngrow, int last_update_step,
                                        double accumulated_dt, const std::string& name,
                                        const std::vector<double>& values);
  /// @}
  /// @}
  /// Apply block @p b's post-step positivity projection to @p u in place (ADC-177): U <- project(U,
  /// aux) over the valid cells, the SAME closure the native per-step path runs (s.project). A compiled
  /// time Program reaches it through ProgramContext::apply_projection (spec op 21). REUSES the block's
  /// own projection (set at add_block time); a block without a projection is a NO-OP (cost-free).
  /// POPS_EXPORT so a generated problem.so resolves it across the dlopen boundary.
  POPS_EXPORT void block_project(int b, MultiFab& u);
  /// @name Compiled-Program scalar diagnostics (epic ADC-399 / ADC-414, spec op 23)
  /// A name -> Real map a compiled Program writes via P.record_scalar (ProgramContext::record_scalar),
  /// retrievable AFTER sim.step for inspection / logging. Lives in Impl (private to the _pops TU) so it
  /// survives across the dlopen boundary; the .so writes it through the POPS_EXPORT setter below.
  /// @{
  /// Store @p value under @p name (overwrites a prior value of the same name). Called by the installed
  /// program closure each step. POPS_EXPORT: the generated problem.so resolves it via RTLD_GLOBAL.
  POPS_EXPORT void record_program_diagnostic(const std::string& name, Real value);
  /// The recorded value of diagnostic @p name. @throws std::out_of_range if @p name was never
  /// recorded (a typo / a diagnostic the installed program does not write fails loud, not 0).
  POPS_EXPORT Real program_diagnostic(const std::string& name) const;
  /// All recorded diagnostics (name -> last recorded value). Empty when the program records none.
  /// Exposed to Python as sim.program_diagnostics() (a dict); program_diagnostic(name) reads one.
  POPS_EXPORT std::map<std::string, Real> program_diagnostics() const;
  /// @}
  /// @}

  /// @name Diagnostics
  /// @{
  int nx() const;
  /// MACRO-STEP counter (0-indexed; incremented by step / step_cfl / step_adaptive). Necessary
  /// for checkpoint/restart: the stride cadence (hold-then-catch-up) depends on macro_step % stride,
  /// not only on the time t (audit 2026-06, IO v1). POPS_EXPORT: a scheduled (every(N)/hold) program
  /// `.so` calls it for the cadence decision, so it must be in the loader's flat ABI like the other
  /// seam accessors (grid_context / solve_fields_from_state); without it the held-schedule `.so`
  /// fails to dlopen ("symbol not found in flat namespace"), caught by the Spec 3 runtime e2e test.
  POPS_EXPORT int macro_step() const;
  /// RESTORES the clock (t, macro_step) -- reserved for the RESTART (sim.restart). Restoring macro_step
  /// is MANDATORY to resume the stride cadence exactly; a restart that would only restore
  /// t would desynchronize the blocks at stride > 1. @throws if macro_step < 0.
  void set_clock(double t, int macro_step);
  /// Extent of the SLOW axis of the field (rows of the (ny, nx) row-major array returned by density / potential
  /// / get_state). Cartesian: ny() == nx() == n (square, UNCHANGED). Polar (ring): ny() == ntheta
  /// (slow azimuthal axis) while nx() == nr (fast radial axis) -- with nr != ntheta the field has
  /// nr*ntheta values, NOT nx()^2: it is this dimension that correctly sizes the numpy
  /// array on the bindings side (without it, a (nx, nx) reshape overflows the buffer when nr != ntheta).
  int ny() const;
  double time() const;
  int n_species() const;
  /// Block names, in the order of addition. SINGLE SOURCE: the internal block registry, populated by
  /// ALL the addition paths (add_block / add_dynamic_block / add_compiled_block / install_block).
  /// An integrator written in Python iterates over it, so it must also see the blocks loaded from
  /// a .so (JIT / AOT); counting them by n_species() alone would skip them.
  std::vector<std::string> block_names() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  ///< ny*nx row-major (j slow, i fast)
  std::vector<double> potential();  ///< phi, ny*nx row-major (j slow, i fast)
  /// RESTORES the potential phi (IO v1, reserved for restart): without it the multigrid would restart
  /// from a blank phi and the resume would not be bit-identical (warm start lost); in
  /// gauss_policy="evolve", phi IS the physical state and its restoration is indispensable. Field
  /// ny*nx row-major (same layout as potential()).
  void set_potential(const std::vector<double>& phi);

  /// @name GLOBAL accessors (MPI-safe collectives) -- outputs / multi-rank checkpoint (IO v1)
  /// The System builds ONE box covering the whole domain (cf. ctor: mono-box ba, round-robin dm ->
  /// box 0 on rank 0). The accessors above (density / get_state / potential) read fab(0):
  /// VALID on the owner rank (mono-rank OR rank 0 under MPI), but fab(0) is OUT OF BOUNDS on
  /// a rank without a box (local_size()==0). The _global variants fill a GLOBAL buffer from the
  /// LOCAL fabs (in GLOBAL indices; nothing on an empty rank) then all_reduce_sum_inplace -> EACH
  /// rank holds the complete field (AMR reflux pattern, comm.hpp). They are COLLECTIVE: all the
  /// ranks MUST call them. On mono-rank they return EXACTLY the same array as the non-global
  /// accessors (all_reduce = identity, box = complete domain) -> bit-identical output.
  /// The IO facade (sim.write / sim.checkpoint) uses them then writes the file only on rank 0.
  /// @{
  std::vector<double> density_global(const std::string& name) const;  ///< comp0, ny*nx global
  std::vector<double> state_global(const std::string& name) const;    ///< U, ncomp*ny*nx global
  std::vector<double> potential_global();                             ///< phi, ny*nx global
  /// @}

  /// @name LOCAL per-fab accessors -- PARALLEL HDF5 write by hyperslabs (IO PR-IO-3, opt-in)
  /// Local counterpart of the _global accessors: instead of gathering the whole field by all_reduce_sum,
  /// they expose per rank the list of LOCAL boxes and the state of EACH fab. The parallel HDF5
  /// facade (sim.write(format='hdf5', parallel=True)) creates the GLOBAL datasets then each rank
  /// writes ITS boxes as hyperslabs -- no global gather. They are NON COLLECTIVE (purely
  /// local: no MPI comm; a rank without a box returns an empty list). The cartesian System is
  /// MONO-BOX (one box covering the domain, on rank 0): local_boxes thus returns ONE box on
  /// rank 0 and nothing elsewhere -- true hyperslab parallelism appears only on a MULTI-BOX
  /// geometry (cf. AMR). The API stays correct in the general case (iteration over all the local fabs,
  /// GLOBAL indices in the box). Layout of local_state IDENTICAL to state_global but
  /// relative to the local box: (c*bny + (j - jlo))*bnx + (i - ilo), component-major.
  /// @{
  std::vector<std::array<int, 4>> local_boxes(
      const std::string& name) const;  ///< (ilo,jlo,ihi,jhi) per local fab
  std::vector<double> local_state(const std::string& name,
                                  int li) const;  ///< U of fab li, flat (ncomp*bny*bnx)
                                                  /// @}
                                                  /// @}

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
