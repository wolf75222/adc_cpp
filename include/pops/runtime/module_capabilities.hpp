#pragma once

/// @file
/// @brief Authoritative STATIC capability facts of the built _pops module (Spec 5 sec.13.12 /
///        sec.13.12.1, criteria #36/#37).
///
/// MOTIVATION. ``pops._capabilities.inspect_capabilities`` walks the inert Python descriptor catalog;
/// that walk is "Python-derived, not authoritative" (Spec 5 sec.13.12). The transport capabilities a
/// module actually provides -- which backend it was compiled with, whether MPI / GPU is real, whether
/// the route carries a stride, named aux fields, a partial IMEX mask -- are decided by the C++ build,
/// not by Python. This header sources those facts from the SAME compile-time tokens the module attrs
/// already expose (``POPS_HAS_KOKKOS`` / ``POPS_HAS_MPI`` in init_core.cpp) so the Python read side can
/// cross-check its descriptor walk against the C++ truth and FAIL LOUD on a disagreement.
///
/// HONESTY (non-negotiable, Spec 5 sec.13.12). A capability is reported TRUE only when a C++ path backs
/// it. ``supports_partial_imex_mask`` is FALSE: a tree-wide grep finds NO partial-IMEX-mask code path,
/// so claiming it would be a lie. ``supports_gpu`` is TRUE only under Kokkos AND a real device backend
/// token (CUDA/HIP); a Kokkos-Serial / OpenMP CPU build reports FALSE. ``supports_stride`` is
/// route-dependent (the production / native route carries a stride; the AOT / prototype route hardcodes
/// stride=1), so the facts are queried per @p target.
///
/// This is a pure free-function / POD header: no System state, no out-of-line definition (kept out of
/// System::Impl on purpose, so the C++ MockImpl in tests/test_strang_splitting.cpp is untouched).

namespace pops {

/// Discrete, monotonic ABI revision of the module capability contract. Bump when the SHAPE of
/// ModuleCapabilities (its fields / their meaning) changes, so a per-artifact manifest baked into an
/// older .so (pops_compiled_manifest) can be told apart from a newer module at load time. Distinct from
/// the textual pops::abi_key() (compiler / std / header signature): that detects a toolchain ABI break,
/// this versions the capability *vocabulary*.
inline constexpr int kAbiVersion = 1;

/// The lowering route whose static capabilities are queried. The two generated backends differ in one
/// honest way: the production / native loader carries a real cell stride, the AOT / prototype flat-array
/// path hardcodes stride=1 (cf. compiled_block_abi.hpp make_grid). ``kModule`` reports the route-agnostic
/// facts (the AND-conservative stride, i.e. false).
enum class CapabilityTarget { kModule, kProduction, kAot };

/// The STATIC transport capabilities the built _pops module provides (Spec 5 sec.13.12). A small POD of
/// booleans + the ABI version, sourced from compile-time tokens only -- it allocates nothing, touches no
/// System, runs no kernel.
struct ModuleCapabilities {
  int abi_version;                  ///< pops::kAbiVersion (this build's capability-contract revision).
  bool supports_uniform;            ///< single-level uniform grid (always available).
  bool supports_amr;                ///< adaptive mesh refinement runtime (AmrSystem; always built in).
  bool supports_mpi;                ///< real MPI transport (POPS_HAS_MPI); false on a serial module.
  bool supports_gpu;               ///< real GPU device backend (Kokkos AND a CUDA/HIP token).
  bool supports_stride;             ///< the route carries a cell stride (production: yes; aot: no).
  bool supports_named_fields;       ///< named aux-field transport (named_aux, aux_field; always built).
  bool supports_partial_imex_mask;  ///< partial IMEX mask -- FALSE: no C++ path backs it (do not lie).
};

namespace detail {

/// True iff this translation unit is compiled for a real GPU device backend. Conservative and honest:
/// a Kokkos build is necessary but NOT sufficient (Kokkos-Serial / OpenMP is a CPU build). __CUDACC__ /
/// __HIPCC__ are the device-compiler tokens (cf. core/foundation/types.hpp); absent them we report
/// false rather than fabricate GPU support from the mere presence of Kokkos.
inline constexpr bool kHasGpuBackend =
#if defined(POPS_HAS_KOKKOS) && (defined(__CUDACC__) || defined(__HIPCC__))
    true;
#else
    false;
#endif

inline constexpr bool kHasMpi =
#if defined(POPS_HAS_MPI)
    true;
#else
    false;
#endif

}  // namespace detail

/// The module's STATIC capability facts for a given lowering route @p target (Spec 5 sec.13.12 / #36).
///
/// All values come from compile-time tokens, never a Python computation:
///   - ``abi_version`` = pops::kAbiVersion;
///   - ``supports_uniform`` / ``supports_amr`` = true (both runtimes are built into _pops);
///   - ``supports_mpi`` = POPS_HAS_MPI;
///   - ``supports_gpu`` = POPS_HAS_KOKKOS AND a device token (else false, conservatively honest);
///   - ``supports_stride`` = true for the production / native route, false for the AOT / prototype route
///     (which hardcodes stride=1) and for the route-agnostic ``kModule`` query;
///   - ``supports_named_fields`` = true (the named-aux transport exists, kAuxNamedBase / aux_field);
///   - ``supports_partial_imex_mask`` = false (NO C++ path backs it -- reporting true would be a lie).
inline ModuleCapabilities module_capabilities(CapabilityTarget target = CapabilityTarget::kModule) {
  ModuleCapabilities caps{};
  caps.abi_version = kAbiVersion;
  caps.supports_uniform = true;
  caps.supports_amr = true;
  caps.supports_mpi = detail::kHasMpi;
  caps.supports_gpu = detail::kHasGpuBackend;
  caps.supports_stride = (target == CapabilityTarget::kProduction);
  caps.supports_named_fields = true;
  caps.supports_partial_imex_mask = false;
  return caps;
}

}  // namespace pops
