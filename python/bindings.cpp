// pybind11 bindings of the adc_cpp LIB: compiles the `_adc` module. Exposes the
// runtime composition facade `System` (the tutor's "coupler / system") + its
// config. Python composes WHAT to assemble (model + spatial scheme + temporal
// treatment + per-block substeps, system Poisson); all the cell-by-cell compute
// stays in the compiled lib. The readable sugar (Spatial, Explicit, IMEX,
// System) is added by the Python package adc/__init__.py.
// Built only with -DADC_BUILD_PYTHON=ON.

#include <pybind11/functional.h>  // std::function<double()> <- Python callable (add_dt_bound)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/core/kokkos_env.hpp>  // Kokkos_Core under ADC_HAS_KOKKOS (kokkos_is_initialized)
#include <adc/parallel/comm.hpp>    // adc::my_rank / n_ranks: rank-0 guard of the multi-rank IO facade
#include <adc/runtime/abi_key.hpp>  // adc::abi_key: ABI key exposed to the DSL ("production" path)
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/system.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>  // std::tuple: argument of AmrSystem.set_hierarchy (patch_boxes boxes) (ADC-65)
#include <vector>

namespace py = pybind11;
using namespace adc;

// field (ny*nx row-major, j slow / i fast) -> numpy array (ny, nx) (copy). We size the buffer
// with BOTH real extents of the index domain (rows = ny, cols = nx): square n x n in Cartesian
// (UNCHANGED), but nr x ntheta in polar where nr != ntheta. A square reshape (n, n) would allocate nx^2
// slots for ny*nx values -> memcpy overflows the numpy buffer (heap overflow, crash at teardown). We
// CHECK buffer size == source size before the memcpy (explicit guard).
static py::array_t<double> to_2d(const std::vector<double>& v, int rows, int cols) {
  py::array_t<double> a({rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings): field size (" + std::to_string(v.size()) +
                             ") != rows*cols (" + std::to_string(rows) + "*" + std::to_string(cols) +
                             "); inconsistent 2D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
// state (ncomp*ny*nx, component-major order, j slow / i fast) -> numpy array (ncomp, ny, nx).
// Same guard as to_2d: rows = ny, cols = nx (square in Cartesian, nr x ntheta in polar).
static py::array_t<double> to_3d(const std::vector<double>& v, int ncomp, int rows, int cols) {
  py::array_t<double> a({ncomp, rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings): state size (" + std::to_string(v.size()) +
                             ") != ncomp*rows*cols (" + std::to_string(ncomp) + "*" +
                             std::to_string(rows) + "*" + std::to_string(cols) +
                             "); inconsistent 3D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
static std::vector<double> flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

// ADC-214: the Python SURFACE keeps the newton_fail_policy kwarg as a STRING ("none"/"warn"/"throw");
// the NewtonOptions POD carries an integer (NewtonOptions::kFail*). This conversion table therefore
// lives in the bindings (where the flat kwargs are assembled into a POD), with the SAME explicit error
// message as before this work. @p where names the calling method in the message.
static int newton_fail_policy_from_string(const std::string& policy, const char* where) {
  if (policy == "none") return NewtonOptions::kFailNone;
  if (policy == "warn") return NewtonOptions::kFailWarn;
  if (policy == "throw") return NewtonOptions::kFailThrow;
  throw std::runtime_error(std::string(where) + ": newton_fail_policy 'none'|'warn'|'throw' (got '" +
                           policy + "')");
}

PYBIND11_MODULE(_adc, m) {
  m.doc() =
      "adc_cpp (lib): runtime multi-species composition. System composes a "
      "system block by block; the compute stays compiled C++.";

  // Module ABI key (compiler + C++ standard + signature of the adc headers). The DSL reads it
  // (diagnostic); add_native_block compares it to the key baked into a native loader.
  m.def("abi_key", &adc::abi_key,
        "Module ABI key (compiler, C++ standard, signature of the adc headers).");

  // MPI rank / rank count of the communicator (0 / 1 in serial or when MPI is not initialized, cf.
  // adc/parallel/comm.hpp). Exposed so the IO facade (sim.write / sim.checkpoint) writes the file
  // only on rank 0 after a collective gather (state_global / potential_global).
  m.def("my_rank", &adc::my_rank, "MPI rank of the process (0 in serial).");
  m.def("n_ranks", &adc::n_ranks, "Number of MPI ranks (1 in serial).");

  // C++ standard of the LOADER (ADC_CXX_STD injected by the build: 20 under Kokkos, 23 otherwise). The
  // DSL backend="production" MUST compile the native model with this SAME standard, otherwise __cplusplus
  // diverges -> different ABI key -> add_native_block raises "incompatible ABI". We expose it as an
  // integer (20/23); dsl.compile derives the -std=c++NN flag from it instead of hardcoding c++23.
#ifdef ADC_CXX_STD
  m.attr("__cxx_std__") = static_cast<int>(ADC_CXX_STD);
#else
  // Manual build without -DADC_CXX_STD: we fall back on __cplusplus to stay consistent with the ABI
  // key (which itself always encodes __cplusplus). 202002L -> 20, beyond -> 23.
  m.attr("__cxx_std__") = static_cast<int>(__cplusplus > 202002L ? 23 : 20);
#endif

  // Compute backend COMPILED into the module: True if _adc was built with Kokkos
  // (-DADC_USE_KOKKOS=ON -> ADC_HAS_KOKKOS), hence capable of multi-thread (OpenMP device) / GPU.
  // adc.set_threads / adc.parallel_info use it to warn that a SERIAL module ignores the thread
  // setting. A serial build exposes False; no false negative.
#ifdef ADC_HAS_KOKKOS
  m.attr("__has_kokkos__") = true;
#else
  m.attr("__has_kokkos__") = false;
#endif

  // MPI seam COMPILED into the module (ADC_HAS_MPI via the adc INTERFACE under -DADC_USE_MPI=ON) plus
  // the MPI include dir(s) used by the build (ADC_MPI_INCLUDE, baked by CMake; '|'-joined). The DSL
  // "production"/"aot" loaders are compiled OUTSIDE CMake and inherit none of this: dsl.py reads these
  // attributes (_native_mpi_flags) to re-bake -DADC_HAS_MPI + -I<inc> so the loader uses comm.hpp's
  // REAL MPI rather than its serial stubs (n_ranks()=1). Without it a distributed layout built inside
  // the loader replicates on every rank (ADC-319). A serial module exposes False / empty.
#if defined(ADC_HAS_MPI)
  m.attr("__has_mpi__") = true;
#if defined(ADC_MPI_INCLUDE)
  m.attr("__mpi_include__") = ADC_MPI_INCLUDE;
#else
  m.attr("__mpi_include__") = "";
#endif
#else
  m.attr("__has_mpi__") = false;
  m.attr("__mpi_include__") = "";
#endif

  // Path of the COMPILER that built this module (ADC_CXX_COMPILER, injected by CMake). Since the ABI
  // key encodes __VERSION__, the "production" DSL MUST recompile its loaders with THIS compiler:
  // dsl.py prefers it to the PATH's `which c++` (which, in a conda env, often designates another
  // compiler -> "-std=c++23 invalid" or ABI rejection). Manual build without -D: empty string, dsl.py
  // then falls back on its historical detection.
#ifdef ADC_CXX_COMPILER
  m.attr("__cxx_compiler__") = ADC_CXX_COMPILER;
#else
  m.attr("__cxx_compiler__") = "";
#endif

  // Project version (ADC_VERSION = CMake PROJECT_VERSION, single source). Re-exposed as
  // adc.__version__ by the package; "unknown" on a manual build without -D.
#ifdef ADC_VERSION
  m.attr("__version__") = ADC_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

  // REAL state of the Kokkos init (lazy: first Fab allocation, through ANY path --
  // System, AmrSystem, DSL .so...). adc.set_threads relies on this rather than on a Python
  // flag that only saw System/AmrSystem: the "too late" warning becomes reliable.
  // Serial build: always False (nothing to initialize, the thread setting is moot).
  m.def("kokkos_is_initialized", []() {
#ifdef ADC_HAS_KOKKOS
    return Kokkos::is_initialized();
#else
    return false;
#endif
  }, "True if the module's Kokkos runtime is already initialized (set_threads then arrives too late).");

  py::class_<SystemConfig>(m, "SystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &SystemConfig::n)
      .def_readwrite("L", &SystemConfig::L)
      .def_readwrite("periodic", &SystemConfig::periodic)
      // Opt-in geometry ("polar grid" work, Phase 1). "cartesian" (default) = bit-identical;
      // "polar" = global ring carried by adc.PolarMesh. Polar fields ignored if geometry=="cartesian".
      .def_readwrite("geometry", &SystemConfig::geometry)
      .def_readwrite("nr", &SystemConfig::nr)
      .def_readwrite("ntheta", &SystemConfig::ntheta)
      .def_readwrite("r_min", &SystemConfig::r_min)
      .def_readwrite("r_max", &SystemConfig::r_max)
      .def_readwrite("theta_boxes", &SystemConfig::theta_boxes);

  // ModelSpec: composition of generic bricks (transport/source/elliptic + parameters).
  // No named scenario; the adc.Model(...) sugar on the Python side fills these fields.
  py::class_<ModelSpec>(m, "ModelSpec")
      .def(py::init<>())
      .def_readwrite("transport", &ModelSpec::transport)
      .def_readwrite("source", &ModelSpec::source)
      .def_readwrite("elliptic", &ModelSpec::elliptic)
      .def_readwrite("B0", &ModelSpec::B0)
      .def_readwrite("gamma", &ModelSpec::gamma)
      .def_readwrite("cs2", &ModelSpec::cs2)
      .def_readwrite("qom", &ModelSpec::qom)
      .def_readwrite("q", &ModelSpec::q)
      .def_readwrite("alpha", &ModelSpec::alpha)
      .def_readwrite("n0", &ModelSpec::n0)
      .def_readwrite("sign", &ModelSpec::sign)
      .def_readwrite("four_pi_G", &ModelSpec::four_pi_G)
      .def_readwrite("rho0", &ModelSpec::rho0);

  py::class_<System>(m, "System")
      .def(py::init<const SystemConfig&>())
      // Per-block composition: model (bricks) + spatial scheme (limiter/riemann) + time
      // (explicit/imex) + substeps. Python says WHAT, the compiled C++ does the compute.
      // ADC-214: the Python SURFACE is UNCHANGED (same flat newton_* kwargs, same defaults). The
      // lambda receives them flat and BUILDS the NewtonOptions POD internally before calling the
      // new C++ method (which groups these homogeneous parameters). adc_cases sees no change.
      .def("add_block",
           [](System& s, const std::string& name, const ModelSpec& model,
              const std::string& limiter, const std::string& riemann, const std::string& recon,
              const std::string& time, int substeps, bool evolve, int stride,
              const std::vector<std::string>& implicit_vars,
              const std::vector<std::string>& implicit_roles, int newton_max_iters,
              double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
              bool newton_diagnostics, double newton_damping, const std::string& newton_fail_policy,
              double positivity_floor, bool wave_speed_cache) {
             NewtonOptions newton;
             newton.max_iters = newton_max_iters;
             newton.rel_tol = static_cast<Real>(newton_rel_tol);
             newton.abs_tol = static_cast<Real>(newton_abs_tol);
             newton.fd_eps = static_cast<Real>(newton_fd_eps);
             newton.damping = static_cast<Real>(newton_damping);
             newton.fail_policy = newton_fail_policy_from_string(newton_fail_policy, "System::add_block");
             s.add_block(name, model, limiter, riemann, recon, time, substeps, evolve, stride,
                         implicit_vars, implicit_roles, newton, newton_diagnostics, positivity_floor,
                         wave_speed_cache);
           },
           py::arg("name"), py::arg("model"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative",
           py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("evolve") = true, py::arg("stride") = 1,
           // Implicit mask CARRIED BY THE BLOCK (IMEX): conserved variables treated implicitly by
           // NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default) -> model default,
           // bit-identical. Resolved on the C++ side against the block's names/roles (error on a missing name/role).
           py::arg("implicit_vars") = std::vector<std::string>{},
           py::arg("implicit_roles") = std::vector<std::string>{},
           // Options of the implicit IMEX source Newton (defaults = historical constants 2 / 1e-7,
           // bit-identical). newton_diagnostics=True enables the report (newton_report(name)).
           py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
           py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
           py::arg("newton_diagnostics") = false, py::arg("newton_damping") = 1.0,
           py::arg("newton_fail_policy") = "none",
           // Zhang-Shu POSITIVITY limiter (ADC-76): density floor of the reconstructed face states
           // (conservative scaling toward the cell mean). 0 (default) = inactive,
           // bit-identical path. Requires a model exposing the Density role.
           py::arg("positivity_floor") = 0.0,
           // HLL wave speed cache (opt-in): evaluates model.wave_speeds once per cell instead of per
           // face. riemann='hll' + explicit only (explicit error otherwise). NoSlope + conservative
           // recon -> bit-identical to the per-face path. False (default) = path unchanged.
           py::arg("wave_speed_cache") = false)
      // Newton report (IMEX diagnostics OPT-IN): dict {enabled, converged, max_residual,
      // max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the substeps of the
      // LAST advance of the block. failed_cell = (i, j) of ONE faulty cell or None.
      .def("newton_report",
           [](const System& s, const std::string& name) {
             const System::SourceNewtonReport r = s.newton_report(name);
             py::dict d;
             d["enabled"] = r.enabled;
             d["converged"] = r.converged;
             d["max_residual"] = r.max_residual;
             d["max_iters_used"] = r.max_iters_used;
             d["n_failed"] = r.n_failed;
             if (r.failed_i >= 0)
               d["failed_cell"] = py::make_tuple(static_cast<int>(r.failed_i),
                                                 static_cast<int>(r.failed_j));
             else
               d["failed_cell"] = py::none();
             d["failed_component"] = static_cast<int>(r.failed_comp);
             return d;
           },
           py::arg("name"))
      // Block whose model is loaded at runtime from a .so generated by the DSL (host path).
      .def("add_dynamic_block", &System::add_dynamic_block, py::arg("name"), py::arg("so_path"),
           py::arg("substeps") = 1, py::arg("names") = std::vector<std::string>{},
           py::arg("recon") = "none")
      .def("add_compiled_block", &System::add_compiled_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("names") = std::vector<std::string>{}, py::arg("positivity_floor") = 0.0)
      // P7-b: changes the RUNTIME parameters of an AOT block WITHOUT recompiling the .so. values =
      // whole block (sorted name order on the DSL side). cf. System::set_block_params.
      .def("set_block_params", &System::set_block_params, py::arg("name"), py::arg("values"))
      // NATIVE block loaded from a .so loader generated by the DSL (backend "production",
      // dsl.compile_native): the .so inlines add_compiled_model<ProdModel> -> zero-copy block on the
      // real System context, ABI key verified. cf. System::add_native_block.
      .def("add_native_block", &System::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1, py::arg("evolve") = true, py::arg("stride") = 1,
           py::arg("positivity_floor") = 0.0)
      .def("add_ionization", &System::add_ionization, py::arg("electron"), py::arg("ion"),
           py::arg("neutral"), py::arg("rate"))
      .def("add_collision", &System::add_collision, py::arg("a"), py::arg("b"), py::arg("rate"))
      .def("add_thermal_exchange", &System::add_thermal_exchange, py::arg("a"), py::arg("b"),
           py::arg("rate"))
      // Schur-condensed source stage (OPT-IN, adc.Split(source=adc.CondensedSchur(...))): replaces
      // the block's explicit / IMEX source with the C++ condensed stage (CondensedSchurSourceStepper, #126)
      // after the hyperbolic transport. kind='electrostatic_lorentz'. Default (without the call) unchanged.
      // ADC-214: Python surface UNCHANGED (same flat krylov_* kwargs / descriptors, same
      // defaults). The lambda receives them flat and BUILDS the SourceStageOptions POD before the C++ call.
      .def("set_source_stage",
           [](System& s, const std::string& name, const std::string& kind, double theta, double alpha,
              double krylov_tol, int krylov_max_iters, const std::string& density,
              const std::string& momentum_x, const std::string& momentum_y, const std::string& energy,
              int bz_aux_component) {
             SourceStageOptions opts;
             opts.krylov_tol = krylov_tol;
             opts.krylov_max_iters = krylov_max_iters;
             opts.density = density;
             opts.momentum_x = momentum_x;
             opts.momentum_y = momentum_y;
             opts.energy = energy;
             opts.bz_aux_component = bz_aux_component;
             s.set_source_stage(name, kind, theta, alpha, opts);
           },
           py::arg("name"), py::arg("kind"), py::arg("theta"), py::arg("alpha"),
           // Tolerance / budget of the stage's Krylov solve (audit 2026-06): <= 0 = historical
           // stepper defaults (1e-10; 400 Cartesian / 600 polar).
           py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0,
           // Field descriptors (wave 2 audit: roles carried in the ABI): "" = canonical
           // role (bit-identical); otherwise a stable role name or a block variable name.
           // bz_aux_component < 0 = canonical B_z channel. Honored in Cartesian as in polar.
           py::arg("density") = "", py::arg("momentum_x") = "", py::arg("momentum_y") = "",
           py::arg("energy") = "", py::arg("bz_aux_component") = -1)
      // Time splitting policy: "lie" (default, bit-identical) or "strang" (H(dt/2) S(dt)
      // H(dt/2), 2nd order). Cf. System::set_time_scheme / SystemStepper::step_strang.
      .def("set_time_scheme", &System::set_time_scheme, py::arg("scheme"))
      // (System) -- see also AmrSystem.add_coupled_source below for the AMR counterpart.
      // GLOBAL time-step bound (step_cfl audit): fn() evaluated ONCE per step (host) by
      // step_cfl / step_adaptive; dt <= fn() when fn() > 0 and finite. Hook for non
      // cell-local constraints (coupling, Schur/Poisson, scheduler, user ramp). A Python
      // callback is acceptable here (never per cell).
      .def("add_dt_bound", &System::add_dt_bound, py::arg("label"), py::arg("fn"))
      // ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
      // "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
      .def("last_dt_bound", &System::last_dt_bound)
      // Clock (IO v1): macro_step exposed + restoration (t, macro_step) for the restart -- the
      // stride cadence depends on macro_step % stride, not only on t.
      .def("macro_step", &System::macro_step)
      .def("set_clock", &System::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("set_potential", &System::set_potential, py::arg("phi"))
      // Gauss law policy (R0, Hoffart repro): "restart" (default, re-solves Poisson every
      // step, bit-identical) or "evolve" (after phi^0, no more re-solve; the Schur stage evolves phi
      // without restart, like the paper). Cf. System::set_gauss_policy.
      .def("set_gauss_policy", &System::set_gauss_policy, py::arg("policy"))
      // Generic COUPLED source (adc.dsl.CoupledSource, P5): flat ABI (postfix bytecode). Reads
      // fields (block, role) and writes source terms compiled into a stack machine, applied by
      // explicit splitting after the transport (same seam as add_ionization). Without the call, unchanged.
      // ADC-214: Python surface UNCHANGED (same flat kwargs in_blocks/.../freq_prog_args, same
      // defaults). The lambda assembles the CoupledSourceProgram POD before the C++ call (frequency / label
      // stay flat, distinct types outside the homogeneous family).
      .def("add_coupled_source",
           [](System& s, const std::vector<std::string>& in_blocks,
              const std::vector<std::string>& in_roles, const std::vector<double>& consts,
              const std::vector<std::string>& out_blocks, const std::vector<std::string>& out_roles,
              const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
              const std::vector<int>& prog_lens, double frequency, const std::string& label,
              const std::vector<int>& freq_prog_ops, const std::vector<int>& freq_prog_args) {
             CoupledSourceProgram prog{in_blocks,  in_roles,  consts,        out_blocks,
                                       out_roles,  prog_ops,  prog_args,     prog_lens,
                                       freq_prog_ops, freq_prog_args};
             s.add_coupled_source(prog, frequency, label);
           },
           py::arg("in_blocks"), py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"),
           py::arg("out_roles"), py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
           // CONSTANT declared frequency mu of the coupling (CoupledSource.frequency, wave 3): step
           // bound dt <= cfl/mu on the macro-step; <= 0 = no bound (historical).
           py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
           // Optional PER-CELL frequency mu(U): bytecode program (same stack machine / register
           // table as the terms). EMPTY (default) = constant frequency only, bit-identical.
           py::arg("freq_prog_ops") = std::vector<int>{},
           py::arg("freq_prog_args") = std::vector<int>{})
      .def("variable_names", &System::variable_names,
           "Variable names of a block (introspection). kind = 'conservative' | 'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("variable_roles", &System::variable_roles,
           "PHYSICAL roles of a block's variables, parallel to variable_names: 'density', "
           "'momentum_x', 'energy', ... or 'custom' if the block does not declare its roles. This is what "
           "the inter-species couplings resolve (index_of(role)). kind = 'conservative' | "
           "'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("block_gamma", &System::block_gamma, py::arg("name"))
      .def("set_poisson", &System::set_poisson,
           "Configures the shared system Poisson. rhs: 'charge_density' | 'composite' (labels "
           "of the SAME right-hand side f = sum of the elliptic bricks per block; charge_density = historical "
           "alias). solver: 'geometric_mg' (any case, wall included) | 'fft' (periodic, "
           "discrete stencil; n = 2^k for the fast FFT, otherwise direct DFT O(n^2)) | 'fft_spectral' "
           "(periodic, continuous symbol -(kx^2+ky^2)). bc: 'auto' | 'periodic' | 'dirichlet' | "
           "'neumann'. wall: 'none' | "
           "'circle' (conducting wall centered at (L/2, L/2), radius wall_radius). epsilon: "
           "CONSTANT permittivity of div(eps grad phi) = f (for variable eps(x): set_epsilon_field). "
           "abs_tol: absolute floor of the GeometricMG V-cycle stopping criterion (0 = relative criterion, "
           "historical; no effect on FFT).",
           py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0, py::arg("epsilon") = 1.0,
           py::arg("abs_tol") = 0.0)
      // DISC transport domain (T2 / T5-PR3 work): materializes a cell-centered 0/1 mask (cell
      // active if its center is in hypot(x-cx, y-cy) - R < 0) and WIRES the transport according to
      // mode=: 'none' (default, full Cartesian transport, bit-identical even with the disc set),
      // 'staircase' (assemble_rhs_masked, 0/1 face gate), 'cutcell' (assemble_rhs_eb, cut-cell EB,
      // apertures + kappa). Honored under Lie AND Strang (set_time_scheme). cf. System::set_disc_domain.
      .def("set_disc_domain", &System::set_disc_domain, py::arg("cx"), py::arg("cy"), py::arg("R"),
           py::arg("mode") = "none")
      // Toggles ONLY the disc transport mode ('none'|'staircase'|'cutcell') without (re)defining the
      // disc. A mode != 'none' requires a disc already set (set_disc_domain) -> error otherwise.
      .def("set_geometry_mode", &System::set_geometry_mode, py::arg("mode"))
      // Domain 0/1 mask (ny, nx) row-major (diagnostic / contract verification). All 1.0 without
      // set_disc_domain.
      .def("disc_mask", [](const System& s) { return to_2d(s.disc_mask(), s.ny(), s.nx()); })
      .def("set_epsilon_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_epsilon_field(flat(arr));
           },
           py::arg("eps"))
      .def("set_epsilon_anisotropic_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> eps_x,
              py::array_t<double, py::array::c_style | py::array::forcecast> eps_y) {
             s.set_epsilon_anisotropic_field(flat(eps_x), flat(eps_y));
           },
           py::arg("eps_x"), py::arg("eps_y"))
      .def("set_reaction_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_reaction_field(flat(arr));
           },
           py::arg("kappa"))
      .def("set_magnetic_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_magnetic_field(flat(arr));
           },
           py::arg("bz"))
      // NAMED aux fields (ADC-70 phase 1): by canonical COMPONENT (>= 5). The name -> comp
      // resolution lives in the Python facade (adc.System.set_aux_field), which calls these two methods.
      .def("set_aux_field_component",
           [](System& s, int comp,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_aux_field_component(comp, flat(arr));
           },
           py::arg("comp"), py::arg("field"))
      .def("aux_field_component",
           [](const System& s, int comp) {
             return to_2d(s.aux_field_component(comp), s.ny(), s.nx());
           },
           py::arg("comp"))
      .def("set_electron_temperature_from", &System::set_electron_temperature_from,
           py::arg("name"))
      .def("set_density",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      // Init from the PRIMITIVES: prim = array (ncomp, n, n) component-major in the order of
      // primitive_vars(name); converted to conservative by the block's model. The Python facade
      // (adc.System.set_primitive_state(**prims)) assembles this array from the named kwargs.
      .def("set_primitive_state",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_primitive_state(name, flat(arr));
           },
           py::arg("name"), py::arg("prim"))
      // Diagnostic: conservative state -> primitive (ncomp, n, n), order of primitive_vars(name).
      .def("get_primitive_state",
           [](System& s, const std::string& name) {
             return to_3d(s.get_primitive_state(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("solve_fields", &System::solve_fields)
      .def("step", &System::step, py::arg("dt"))
      .def("advance", &System::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &System::step_cfl,
           "Advances by ONE step at dt = cfl * h / max wave speed of the system (also honors the "
           "optional bounds: substeps, stride, source_frequency, couplings, add_dt_bound). Returns the dt "
           "used.",
           py::arg("cfl"))
      .def("dt_hotspot", &System::dt_hotspot,
           "Diagnostic (ADC-182): (w, i, j) of the GLOBAL cell that dominates the transport CFL bound "
           "of block 'name' -- to locate a collapsing dt. On demand, off the hot path.",
           py::arg("name"))
      .def("step_adaptive", &System::step_adaptive,
           "Advances by ONE MULTIRATE macro-step: the slowest block sets the macro-step, each faster "
           "block is sub-cycled n = ceil(w_block / w_min) times. Returns the macro-step.",
           py::arg("cfl"))
      // Primitives for a CUSTOM time integrator in Python (take_step):
      .def("eval_rhs",
           [](System& s, const std::string& name) {
             return to_3d(s.eval_rhs(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("get_state",
           [](System& s, const std::string& name) {
             return to_3d(s.get_state(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("set_state",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_state(name, flat(arr));
           },
           py::arg("name"), py::arg("u"))
      .def("n_vars", &System::n_vars, py::arg("name"))
      .def("nx", &System::nx)
      .def("ny", &System::ny)
      .def("time", &System::time)
      .def("n_species", &System::n_species)
      .def("block_names", &System::block_names)
      .def("mass", &System::mass, py::arg("name"))
      .def("density",
           [](const System& s, const std::string& name) {
             return to_2d(s.density(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("potential", [](System& s) { return to_2d(s.potential(), s.ny(), s.nx()); })
      // GLOBAL accessors (MPI-safe collectives): multi-rank outputs / checkpoint (IO v1). Each
      // rank MUST call them (internal all_reduce); they return the COMPLETE field (rank-0 gather
      // implicit via all_reduce_sum) -- single-rank: bit-identical to density / get_state / potential.
      // The sim.write / sim.checkpoint facade uses them then writes the file only on rank 0.
      .def("density_global",
           [](const System& s, const std::string& name) {
             return to_2d(s.density_global(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("state_global",
           [](const System& s, const std::string& name) {
             return to_3d(s.state_global(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("potential_global", [](System& s) { return to_2d(s.potential_global(), s.ny(), s.nx()); })
      // LOCAL per-fab accessors (NOT collective): PARALLEL HDF5 writing by hyperslabs (PR-IO-3,
      // sim.write(format='hdf5', parallel=True)). local_boxes returns the list of local boxes
      // (ilo, jlo, ihi, jhi) in GLOBAL indices; local_state returns the state of fab li reshaped
      // (n_vars, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1]. A rank without a box returns an
      // empty list. Since the System is single-box, real parallelism only appears on a multi-box
      // geometry (cf. AMR); the API stays correct in the general case.
      .def("local_boxes", &System::local_boxes, py::arg("name"))
      .def("local_state",
           [](const System& s, const std::string& name, int li) {
             const auto boxes = s.local_boxes(name);
             if (li < 0 || li >= static_cast<int>(boxes.size()))
               throw std::out_of_range("System.local_state: local fab index out of bounds");
             const int bnx = boxes[li][2] - boxes[li][0] + 1;  // ihi - ilo + 1
             const int bny = boxes[li][3] - boxes[li][1] + 1;  // jhi - jlo + 1
             return to_3d(s.local_state(name, li), s.n_vars(name), bny, bnx);
           },
           py::arg("name"), py::arg("li"))
      .def_static("abi_key", &System::abi_key,
                  "Module ABI key (cf. adc.abi_key); compared to that of a native loader.");

  // --- AMR: single-species composition on multi-patch AMR (generic composable brick) ---
  // adc_cases DRIVES it from Python (no C++ on the cases side) just like System.
  //
  // NB: the two-fluid AP integrator (BESPOKE asymptotic-preserving scheme, not composable
  // block by block) has left the core: it is not a generic brick but a SCENARIO. It now lives
  // in adc_cases (cf. adc_cases/two_fluid_ap/), compiled on the fly against the generic
  // headers of adc_cpp; it is no longer exposed by the _adc module.

  // AmrSystem: generic single-species composition on AMR.
  py::class_<AmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &AmrSystemConfig::n)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("periodic", &AmrSystemConfig::periodic)
      .def_readwrite("distribute_coarse", &AmrSystemConfig::distribute_coarse)
      .def_readwrite("coarse_max_grid", &AmrSystemConfig::coarse_max_grid);

  py::class_<AmrSystem>(m, "AmrSystem")
      .def(py::init<const AmrSystemConfig&>())
      // ADC-214: Python surface UNCHANGED (same flat newton_* kwargs, same defaults). The lambda
      // assembles the NewtonOptions POD before the C++ call (parity with System.add_block).
      .def("add_block",
           [](AmrSystem& s, const std::string& name, const ModelSpec& model,
              const std::string& limiter, const std::string& riemann, const std::string& recon,
              const std::string& time, int substeps, int stride,
              const std::vector<std::string>& implicit_vars,
              const std::vector<std::string>& implicit_roles, int newton_max_iters,
              double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
              double newton_damping, const std::string& newton_fail_policy, bool newton_diagnostics,
              double positivity_floor) {
             NewtonOptions newton;
             newton.max_iters = newton_max_iters;
             newton.rel_tol = static_cast<Real>(newton_rel_tol);
             newton.abs_tol = static_cast<Real>(newton_abs_tol);
             newton.fd_eps = static_cast<Real>(newton_fd_eps);
             newton.damping = static_cast<Real>(newton_damping);
             newton.fail_policy =
                 newton_fail_policy_from_string(newton_fail_policy, "AmrSystem::add_block");
             s.add_block(name, model, limiter, riemann, recon, time, substeps, stride, implicit_vars,
                         implicit_roles, newton, newton_diagnostics, positivity_floor);
           },
           py::arg("name"), py::arg("model"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit",
           py::arg("substeps") = 1, py::arg("stride") = 1,
           // Partial IMEX mask CARRIED BY THE BLOCK (capstone vii): conserved variables treated
           // implicitly by NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default)
           // -> full backward-Euler. Only meaningful with time="imex" and MULTI-BLOCK (cf. add_block).
           py::arg("implicit_vars") = std::vector<std::string>{},
           py::arg("implicit_roles") = std::vector<std::string>{},
           // IMEX Newton options (wave 3, System parity): OPTIONS wired in MONO-BLOCK (coupler)
           // AND MULTI-BLOCK (engine). newton_diagnostics (newton_report report): native MULTI-BLOCK
           // only (mono-block rejected at build; .so loaders rejected at the Python facade).
           py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
           py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
           py::arg("newton_damping") = 1.0, py::arg("newton_fail_policy") = "none",
           py::arg("newton_diagnostics") = false,
           // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on
           // the AMR transport. 0 (default) = inactive, bit-identical. Marshaled from spatial.positivity_floor
           // by the AmrSystem.add_block / add_equation Python facade.
           py::arg("positivity_floor") = 0.0)
      // Newton report (IMEX diagnostics OPT-IN, native MULTI-BLOCK): dict {enabled, converged,
      // max_residual, max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the
      // levels/substeps of the LAST advance of the block. failed_cell = (i, j) or None. EXACT shape of
      // the System.newton_report binding (parity, including failed_cell tuple/None).
      .def("newton_report",
           [](AmrSystem& s, const std::string& name) {
             const AmrSystem::SourceNewtonReport r = s.newton_report(name);
             py::dict d;
             d["enabled"] = r.enabled;
             d["converged"] = r.converged;
             d["max_residual"] = r.max_residual;
             d["max_iters_used"] = r.max_iters_used;
             d["n_failed"] = r.n_failed;
             if (r.failed_i >= 0)
               d["failed_cell"] = py::make_tuple(static_cast<int>(r.failed_i),
                                                 static_cast<int>(r.failed_j));
             else
               d["failed_cell"] = py::none();
             d["failed_component"] = static_cast<int>(r.failed_comp);
             return d;
           },
           py::arg("name"))
      // NATIVE AMR block loaded from a .so loader generated by the DSL (backend "production",
      // target="amr_system"): the .so inlines add_compiled_model(AmrSystem&) -> native block on the
      // AMR hierarchy (reflux, regrid), ABI key verified. cf. AmrSystem::add_native_block. NO
      // evolve (mono-block AMR). The AMR LIMITS (primitive/roe/hllc/weno5) are guarded on the Python
      // facade side (AmrSystem.add_equation) before this binding.
      .def("add_native_block", &AmrSystem::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1)
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"))
      // PHI tag on |grad phi| (D4) added to the union of regrid tags: also refines where the
      // norm of the potential gradient exceeds grad_threshold (diocotron ring edge). MULTI-BLOCK
      // + regrid_every > 0. <= 0 (default) -> phi DISABLED (bit-identical). cf. AmrSystem::set_phi_refinement.
      .def("set_phi_refinement", &AmrSystem::set_phi_refinement, py::arg("grad_threshold"))
      .def("set_poisson", &AmrSystem::set_poisson,
           "Configures the coarse Poisson of the AMR hierarchy (cf. System.set_poisson). On AMR the "
           "solver is ALWAYS GeometricMG and the right-hand side ALWAYS the sum of the elliptic "
           "bricks. rhs: 'charge_density' | 'composite'. solver: 'geometric_mg' only (no "
           "FFT on the hierarchy). bc: 'auto' | 'periodic' | 'dirichlet' | 'neumann'. wall: "
           "'none' | 'circle' (circular conducting wall, requires wall_radius > 0).",
           py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      // GLOBAL step bound + ACTIVE bound (AMR StabilityPolicy, System.add_dt_bound parity).
      .def("add_dt_bound", &AmrSystem::add_dt_bound, py::arg("label"), py::arg("fn"))
      .def("last_dt_bound", &AmrSystem::last_dt_bound)
      // amr-schur PATH (AMR counterpart of System.set_magnetic_field / set_source_stage / set_time_scheme).
      // GLOBAL Schur-condensed source stage (electrostatic/Lorentz) on the mono-block hierarchy,
      // instead of the local IMEX source. B_z (Lorentz term) accepts a flattened numpy (n, n).
      .def("set_magnetic_field",
           [](AmrSystem& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_magnetic_field(flat(arr));
           },
           py::arg("bz"))
      // ADC-214: Python surface UNCHANGED (same flat krylov_* kwargs / descriptors, same
      // defaults; no bz_aux_component on the AMR side). The lambda assembles the SourceStageOptions POD.
      .def("set_source_stage",
           [](AmrSystem& s, const std::string& name, const std::string& kind, double theta,
              double alpha, double krylov_tol, int krylov_max_iters, const std::string& density,
              const std::string& momentum_x, const std::string& momentum_y,
              const std::string& energy) {
             SourceStageOptions opts;
             opts.krylov_tol = krylov_tol;
             opts.krylov_max_iters = krylov_max_iters;
             opts.density = density;
             opts.momentum_x = momentum_x;
             opts.momentum_y = momentum_y;
             opts.energy = energy;
             s.set_source_stage(name, kind, theta, alpha, opts);
           },
           py::arg("name"), py::arg("kind"), py::arg("theta"), py::arg("alpha"),
           // Carried settings (wave 3, System parity): Krylov tolerances of the coarse solve
           // (<= 0 = defaults 1e-10/400) + field descriptors ("" = canonical role).
           py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0,
           py::arg("density") = "", py::arg("momentum_x") = "", py::arg("momentum_y") = "",
           py::arg("energy") = "")
      .def("set_time_scheme", &AmrSystem::set_time_scheme, py::arg("scheme"))
      .def("set_density",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      // Full initial conservative state (ncomp, n, n) -> starts the AMR from the paper's drift
      // state (rho, rho*u, rho*v) instead of m=0. Keeps ndim==3 EXPLICIT: flat() flattens
      // any C-contiguous array, so a 2D density (n, n) passed by mistake would become a
      // 1-component state (comp 0 = density, momentum left at 0) -- a silent density masquerade
      // with the wrong physics. We require (ncomp, n, n). flat() then flattens in
      // component-major c*n*n + j*n + i (same convention as to_3d / set_state).
      .def("set_conservative_state",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             if (arr.ndim() != 3)
               throw std::runtime_error(
                   "AmrSystem.set_conservative_state: state expected of shape (ncomp, n, n); got "
                   "a " + std::to_string(arr.ndim()) + "D array (a 2D density? use "
                   "set_density)");
             s.set_conservative_state(name, flat(arr));
           },
           py::arg("name"), py::arg("U"))
      // Inter-species COUPLED source (compiled adc.dsl.CoupledSource, P5 bytecode), MULTI-BLOCK on the
      // SHARED AMR hierarchy: applied after the transport at each macro-step, by explicit
      // splitting, level by level + fine -> coarse cascade (consistent covered cells). SAME
      // flat ABI as System.add_coupled_source. Without the call, unchanged. cf. AmrSystem::add_coupled_source.
      // ADC-214: Python surface UNCHANGED (same flat kwargs, same defaults). The lambda assembles the
      // CoupledSourceProgram POD before the C++ call (parity with System.add_coupled_source).
      .def("add_coupled_source",
           [](AmrSystem& s, const std::vector<std::string>& in_blocks,
              const std::vector<std::string>& in_roles, const std::vector<double>& consts,
              const std::vector<std::string>& out_blocks, const std::vector<std::string>& out_roles,
              const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
              const std::vector<int>& prog_lens, double frequency, const std::string& label,
              const std::vector<int>& freq_prog_ops, const std::vector<int>& freq_prog_args) {
             CoupledSourceProgram prog{in_blocks,  in_roles,  consts,        out_blocks,
                                       out_roles,  prog_ops,  prog_args,     prog_lens,
                                       freq_prog_ops, freq_prog_args};
             s.add_coupled_source(prog, frequency, label);
           },
           py::arg("in_blocks"), py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"),
           py::arg("out_roles"), py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
           py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
           // Optional PER-CELL frequency mu(U): evaluated on the coarse level (cf. System).
           py::arg("freq_prog_ops") = std::vector<int>{},
           py::arg("freq_prog_args") = std::vector<int>{})
      .def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl,
           "Advances by one AMR macro-step at dt = cfl * dx_coarse / max wave speed (also honors the "
           "substeps/stride cadence in multi-block and the optional bounds). Returns the dt used.",
           py::arg("cfl"))
      .def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      // AMR clock (IO v1, System parity): macro-step counter + restoration (t, macro_step) ->
      // the regrid/stride cadence resumes exactly after a set_clock. Prerequisite PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("n_blocks", &AmrSystem::n_blocks)
      .def("block_names", &AmrSystem::block_names)
      .def("n_patches", &AmrSystem::n_patches)
      // Index-space footprints of the fine patches: list of tuples (level, ilo, jlo, ihi, jhi), INCLUSIVE
      // corners, in the index space of the level (n << level cells/direction, ratio 2). SAME
      // source as n_patches() (the GLOBAL fine BoxArray) -> rank-independent, MPI-safe. Query between
      // steps, zero cost on the hot path. The Python wrapper converts to [0, L]^2 (it knows n via nx() and
      // L); cf. AmrSystem.patch_rectangles() on the facade side.
      .def("patch_boxes",
           [](AmrSystem& s) {
             py::list out;
             for (const adc::PatchBox& b : s.patch_boxes())
               out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
             return out;
           })
      // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic): coarse_local_boxes() = base
      // boxes OWNED by this rank (level-0 local_size()); coarse_total_boxes() = total base boxes (BoxArray
      // size, identical on all ranks). distribute_coarse=True -> local < total per rank (distributed
      // coarse transport); replicated / single-box -> local == total. Query between steps, no hot cost.
      .def("coarse_local_boxes", &AmrSystem::coarse_local_boxes)
      .def("coarse_total_boxes", &AmrSystem::coarse_total_boxes)
      // mass / density: overload by BLOCK NAME (multi-block; empty name -> 1st block, mono-block
      // compat or cosmetic name). The name INDEXES the block in multi-block (each block has its mass /
      // density, conserved PER BLOCK at reflux). Without argument -> 1st block (mono-block back-compat).
      .def("mass", [](AmrSystem& s) { return s.mass(); })
      .def("mass", [](AmrSystem& s, const std::string& name) { return s.mass(name); },
           py::arg("name"))
      // AMR: SQUARE domain (n x n), no polar geometry -> rows == cols == nx() (unchanged).
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.nx(), s.nx()); })
      .def("density",
           [](AmrSystem& s, const std::string& name) {
             return to_2d(s.density(name), s.nx(), s.nx());
           },
           py::arg("name"))
      // phi of the coarse (base) level, (n, n). SAME observable as System.potential(): level 0
      // covers the whole domain -> enough to sample a median circle (azimuthal FFT). In
      // multi-block, phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located), shared by all.
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.nx(), s.nx()); })
      // AMR CHECKPOINT / RESTART single-rank (ADC-65): full conservative state per level + phi
      // (warm-start) + imposition of the saved fine hierarchy. SERIAL MONO-BLOCK (multi-block: C++
      // rejection; np>1: facade rejection -- per-level gather = future). level_state / level_potential return
      // FLAT fields (c*nf*nf + j*nf + i / nf*nf, nf = nx << k); the facade reshapes. set_*
      // flatten any C-contiguous array (flat). set_hierarchy: list of tuples
      // (level, ilo, jlo, ihi, jhi) like patch_boxes() (the coupler filters level 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("n_vars", [](AmrSystem& s) { return s.n_vars(); })
      .def("level_state", [](AmrSystem& s, int k) { return s.level_state(k); }, py::arg("k"))
      .def("set_level_state",
           [](AmrSystem& s, int k,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_level_state(k, flat(arr));
           },
           py::arg("k"), py::arg("state"))
      .def("level_potential", [](AmrSystem& s, int k) { return s.level_potential(k); }, py::arg("k"))
      .def("set_level_potential",
           [](AmrSystem& s, int k,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_level_potential(k, flat(arr));
           },
           py::arg("k"), py::arg("phi"))
      .def("set_hierarchy",
           [](AmrSystem& s, const std::vector<std::tuple<int, int, int, int, int>>& boxes) {
             std::vector<adc::PatchBox> bx;
             bx.reserve(boxes.size());
             for (const auto& b : boxes)
               bx.push_back(adc::PatchBox{std::get<0>(b), std::get<1>(b), std::get<2>(b),
                                          std::get<3>(b), std::get<4>(b)});
             s.set_hierarchy(bx);
           },
           py::arg("boxes"));
}
