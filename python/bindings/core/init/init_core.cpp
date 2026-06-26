#include "../bindings_detail.hpp"

#include <pops/core/state/aux_names.hpp>  // ADC-291: canonical aux name<->component table + bounds

// ADC-365: module attributes/globals + SystemConfig + ModelSpec (registered first so System/
// AmrSystem signatures resolve them).
void init_core(py::module_& m) {
  m.doc() =
      "adc_cpp (lib): runtime multi-species composition. System composes a "
      "system block by block; the compute stays compiled C++.";

  // Module ABI key (compiler + C++ standard + signature of the adc headers). The DSL reads it
  // (diagnostic); add_native_block compares it to the key baked into a native loader.
  m.def("abi_key", &pops::abi_key,
        "Module ABI key (compiler, C++ standard, signature of the adc headers).");

  // MPI rank / rank count of the communicator (0 / 1 in serial or when MPI is not initialized, cf.
  // adc/parallel/comm.hpp). Exposed so the IO facade (sim.write / sim.checkpoint) writes the file
  // only on rank 0 after a collective gather (state_global / potential_global).
  m.def("my_rank", &pops::my_rank, "MPI rank of the process (0 in serial).");
  m.def("n_ranks", &pops::n_ranks, "Number of MPI ranks (1 in serial).");

  // C++ standard of the LOADER (POPS_CXX_STD injected by the build: 20 under Kokkos, 23 otherwise). The
  // DSL backend="production" MUST compile the native model with this SAME standard, otherwise __cplusplus
  // diverges -> different ABI key -> add_native_block raises "incompatible ABI". We expose it as an
  // integer (20/23); dsl.compile derives the -std=c++NN flag from it instead of hardcoding c++23.
#ifdef POPS_CXX_STD
  m.attr("__cxx_std__") = static_cast<int>(POPS_CXX_STD);
#else
  // Manual build without -DADC_CXX_STD: we fall back on __cplusplus to stay consistent with the ABI
  // key (which itself always encodes __cplusplus). 202002L -> 20, beyond -> 23.
  m.attr("__cxx_std__") = static_cast<int>(__cplusplus > 202002L ? 23 : 20);
#endif

  // Compute backend COMPILED into the module: True if _pops was built with Kokkos
  // (-DADC_USE_KOKKOS=ON -> POPS_HAS_KOKKOS), hence capable of multi-thread (OpenMP device) / GPU.
  // pops.set_threads / pops.parallel_info use it to warn that a SERIAL module ignores the thread
  // setting. A serial build exposes False; no false negative.
#ifdef POPS_HAS_KOKKOS
  m.attr("__has_kokkos__") = true;
#else
  m.attr("__has_kokkos__") = false;
#endif

  // MPI seam COMPILED into the module (POPS_HAS_MPI via the adc INTERFACE under -DADC_USE_MPI=ON) plus
  // the MPI include dir(s) used by the build (POPS_MPI_INCLUDE, baked by CMake; '|'-joined). The DSL
  // "production"/"aot" loaders are compiled OUTSIDE CMake and inherit none of this: dsl.py reads these
  // attributes (_native_mpi_flags) to re-bake -DADC_HAS_MPI + -I<inc> so the loader uses comm.hpp's
  // REAL MPI rather than its serial stubs (n_ranks()=1). Without it a distributed layout built inside
  // the loader replicates on every rank (ADC-319). A serial module exposes False / empty.
#if defined(POPS_HAS_MPI)
  m.attr("__has_mpi__") = true;
#if defined(POPS_MPI_INCLUDE)
  m.attr("__mpi_include__") = POPS_MPI_INCLUDE;
#else
  m.attr("__mpi_include__") = "";
#endif
#else
  m.attr("__has_mpi__") = false;
  m.attr("__mpi_include__") = "";
#endif

  // Path of the COMPILER that built this module (POPS_CXX_COMPILER, injected by CMake). Since the ABI
  // key encodes __VERSION__, the "production" DSL MUST recompile its loaders with THIS compiler:
  // dsl.py prefers it to the PATH's `which c++` (which, in a conda env, often designates another
  // compiler -> "-std=c++23 invalid" or ABI rejection). Manual build without -D: empty string, dsl.py
  // then falls back on its historical detection.
#ifdef POPS_CXX_COMPILER
  m.attr("__cxx_compiler__") = POPS_CXX_COMPILER;
#else
  m.attr("__cxx_compiler__") = "";
#endif

  // Project version (POPS_VERSION = CMake PROJECT_VERSION, single source). Re-exposed as
  // pops.__version__ by the package; "unknown" on a manual build without -D.
#ifdef POPS_VERSION
  m.attr("__version__") = POPS_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

  // AUX channel limits + canonical name table (ADC-291), exposed from the SINGLE C++ source
  // (adc/core/state.hpp + aux_names.hpp). The DSL/capabilities() read these so the Python mirrors
  // (AUX_NAMED_MAX / AUX_NAMED_BASE / AUX_CANONICAL in dsl.py) cannot SILENTLY drift from C++:
  // test_capabilities.py asserts they match. kAuxMaxExtra is the only remaining compile-time aux
  // limit and is now declarative + introspectable here.
  m.attr("__aux_base_comps__") = static_cast<int>(pops::kAuxBaseComps);
  m.attr("__aux_named_base__") = static_cast<int>(pops::kAuxNamedBase);
  m.attr("__aux_max_extra__") = static_cast<int>(pops::kAuxMaxExtra);
  m.attr("__aux_max_comps__") = static_cast<int>(pops::kAuxMaxComps);
  {
    py::dict canon;
    for (const auto& [name, comp] : pops::kAuxCanonicalNames)
      canon[py::str(std::string(name))] = static_cast<int>(comp);
    m.attr("__aux_canonical__") = canon;
  }

  // REAL state of the Kokkos init (lazy: first Fab allocation, through ANY path --
  // System, AmrSystem, DSL .so...). pops.set_threads relies on this rather than on a Python
  // flag that only saw System/AmrSystem: the "too late" warning becomes reliable.
  // Serial build: always False (nothing to initialize, the thread setting is moot).
  m.def(
      "kokkos_is_initialized",
      []() {
#ifdef POPS_HAS_KOKKOS
        return Kokkos::is_initialized();
#else
        return false;
#endif
      },
      "True if the module's Kokkos runtime is already initialized (set_threads then arrives too "
      "late).");

  py::class_<SystemConfig>(m, "SystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &SystemConfig::n)
      .def_readwrite("L", &SystemConfig::L)
      .def_readwrite("periodic", &SystemConfig::periodic)
      // Opt-in geometry ("polar grid" work, Phase 1). "cartesian" (default) = bit-identical;
      // "polar" = global ring carried by pops.PolarMesh. Polar fields ignored if geometry=="cartesian".
      .def_readwrite("geometry", &SystemConfig::geometry)
      .def_readwrite("nr", &SystemConfig::nr)
      .def_readwrite("ntheta", &SystemConfig::ntheta)
      .def_readwrite("r_min", &SystemConfig::r_min)
      .def_readwrite("r_max", &SystemConfig::r_max)
      .def_readwrite("theta_boxes", &SystemConfig::theta_boxes);

  // ModelSpec: composition of generic bricks (transport/source/elliptic + parameters).
  // No named scenario; the pops.Model(...) sugar on the Python side fills these fields.
  py::class_<ModelSpec>(m, "ModelSpec")
      .def(py::init<>())
      .def_readwrite("transport", &ModelSpec::transport)
      .def_readwrite("source", &ModelSpec::source)
      .def_readwrite("elliptic", &ModelSpec::elliptic)
      .def_readwrite("B0", &ModelSpec::B0)
      .def_readwrite("gamma", &ModelSpec::gamma)
      .def_readwrite("cs2", &ModelSpec::cs2)
      .def_readwrite("vacuum_floor", &ModelSpec::vacuum_floor)
      .def_readwrite("qom", &ModelSpec::qom)
      .def_readwrite("q", &ModelSpec::q)
      .def_readwrite("alpha", &ModelSpec::alpha)
      .def_readwrite("n0", &ModelSpec::n0)
      .def_readwrite("sign", &ModelSpec::sign)
      .def_readwrite("four_pi_G", &ModelSpec::four_pi_G)
      .def_readwrite("rho0", &ModelSpec::rho0);
}
