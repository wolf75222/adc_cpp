#pragma once

/// @file
/// @brief Scheme tags (SSPRK2, SSPRK3, UserTimeIntegrator), TimeTreatment enum and per-block
///        time policies: template TimePolicy<Method, Treatment, Substeps, Stride>, its traits
///        (TimePolicyTraits) and the aliases ExplicitTime / ImplicitTime / IMEXTime /
///        PrescribedTime.
///
/// Layer: `include/pops/numerics/time`.
/// Role: separate TWO levels -- the mathematical scheme (SSPRK, IMEX, user implicit) and the
///        usage policy in a coupled system (explicit/implicit, substeps, cadence, or prescribed
///        field). The core keeps the generic schemes and the scheduler; cases choose a policy
///        per block without changing the local PhysicalModel.
///
/// Invariants:
/// - SubstepsT >= 1 and StrideT >= 1 (static_assert);
/// - SubstepsT = MORE FREQUENT substeps (n steps of dt/n); StrideT = SLOWER cadence (advances 1
///   macro-step every StrideT, thus by a step of StrideT*dt). Both are ORTHOGONAL;
/// - StrideT=1 (default) = historical behavior; the default TimePolicyTraits treats any type as
///   Explicit, substeps=1, stride=1.

namespace pops {

struct SSPRK2 {};  // Shu-Osher SSP-RK2 (2 stages, order 2)
struct SSPRK3 {};  // Shu-Osher SSP-RK3 (3 stages, order 3)

struct UserTimeIntegrator {};  // extension point: take_step provided by the case

enum class TimeTreatment { Explicit, Implicit, IMEX, Prescribed };

// SubstepsT: MORE FREQUENT substeps (n steps of dt/n per macro-step, fast electrons).
// StrideT: SLOWER cadence (the block advances only 1 macro-step every StrideT, thus by a
//   step of StrideT*dt, a slow "gas" we do not solve at every step, guardian return).
//   Both are orthogonal; StrideT=1 = historical behavior.
template <class MethodT, TimeTreatment TreatmentT, int SubstepsT = 1, int StrideT = 1>
struct TimePolicy {
  static_assert(SubstepsT >= 1, "a TimePolicy must have at least one substep");
  static_assert(StrideT >= 1, "a TimePolicy must have a cadence (stride) >= 1");
  using Method = MethodT;
  static constexpr TimeTreatment treatment = TreatmentT;
  static constexpr int substeps = SubstepsT;
  static constexpr int stride = StrideT;
};

template <class T>
struct TimePolicyTraits {
  using Method = T;
  static constexpr TimeTreatment treatment = TimeTreatment::Explicit;
  static constexpr int substeps = 1;
  static constexpr int stride = 1;
};

template <class MethodT, TimeTreatment TreatmentT, int SubstepsT, int StrideT>
struct TimePolicyTraits<TimePolicy<MethodT, TreatmentT, SubstepsT, StrideT>> {
  using Method = MethodT;
  static constexpr TimeTreatment treatment = TreatmentT;
  static constexpr int substeps = SubstepsT;
  static constexpr int stride = StrideT;
};

template <class MethodT = SSPRK2, int SubstepsT = 1, int StrideT = 1>
using ExplicitTime = TimePolicy<MethodT, TimeTreatment::Explicit, SubstepsT, StrideT>;

template <class MethodT = UserTimeIntegrator, int SubstepsT = 1, int StrideT = 1>
using ImplicitTime = TimePolicy<MethodT, TimeTreatment::Implicit, SubstepsT, StrideT>;

template <class MethodT = UserTimeIntegrator, int SubstepsT = 1, int StrideT = 1>
using IMEXTime = TimePolicy<MethodT, TimeTreatment::IMEX, SubstepsT, StrideT>;

using PrescribedTime = TimePolicy<UserTimeIntegrator, TimeTreatment::Prescribed, 1, 1>;

}  // namespace pops
