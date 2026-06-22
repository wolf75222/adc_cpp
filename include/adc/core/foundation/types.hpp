#pragma once

/// @file
/// @brief Base scalar types and the ADC_HD macro (host+device portability). Minimal foundation
///        with no external dependency; the switch to pde_core::Real waits for the distributed mesh.
///
/// `Real`: centralized double alias. All numerical computation uses it; do not write `double`
/// directly in the physics layer or the kernels.
///
/// `ADC_HD`: annotation for functions called inside Kokkos kernels on host AND device.
/// - Kokkos: KOKKOS_FUNCTION (portable Cuda/HIP/SYCL/CPU, without manual CUDA syntax).
///   KOKKOS_FUNCTION is preferred over KOKKOS_INLINE_FUNCTION so as not to add an implicit `inline`
///   on sites already marked `ADC_HD inline ...`.
/// - Direct CUDA/HIP (without Kokkos): __host__ __device__.
/// - Pure CPU: empty expansion.
/// INVARIANT: ADC_HD can only wrap device-clean code (no host object,
/// no std::vector, no vtable).

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Macros.hpp>
#define ADC_HD KOKKOS_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define ADC_HD __host__ __device__
#else
#define ADC_HD
#endif

namespace adc {

using Real = double;

/// Speed FLOOR for the CFL step policies (audit 2026-06, explicit constant instead of the
/// scattered literal 1e-30): w = max(reduced_speed, kCflSpeedFloor) avoids the division by zero
/// when a block has no wave (frozen transport / null field). WARNING: a system in which ALL
/// the speeds are null then receives a step ~cfl*h/1e-30, enormous -- that is the historical
/// behavior assumed (such a step transports nothing); diagnose it via last_dt_bound() ==
/// "degenerate" on the System side. Shared by System::step_cfl/step_adaptive and AmrRuntime::step_cfl.
inline constexpr Real kCflSpeedFloor = Real(1e-30);

}  // namespace adc
