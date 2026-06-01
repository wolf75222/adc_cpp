#pragma once

// Tags et politiques d'integration en temps.
//
// Deux niveaux sont separes :
//   - le schema mathematique (SSPRK2, SSPRK3, IMEX, implicite utilisateur...) ;
//   - la politique d'emploi dans un systeme couple : explicite / implicite,
//     nombre de sous-pas, ou champ prescrit.
//
// Le coeur garde les schemas generiques et le scheduler ; les cas choisissent une
// politique par bloc d'equation. Exemple attendu pour un plasma multi-especes :
//   electrons : ImplicitTime<UserImplicit, 10>
//   ions      : ExplicitTime<SSPRK2, 1>
// sans changer le PhysicalModel local.

namespace adc {

struct SSPRK2 {};  // Shu-Osher SSP-RK2 (2 etages, ordre 2)
struct SSPRK3 {};  // Shu-Osher SSP-RK3 (3 etages, ordre 3)

struct UserTimeIntegrator {};  // point d'extension : take_step fourni par le cas

enum class TimeTreatment {
  Explicit,
  Implicit,
  IMEX,
  Prescribed
};

template <class MethodT, TimeTreatment TreatmentT, int SubstepsT = 1>
struct TimePolicy {
  static_assert(SubstepsT >= 1, "un TimePolicy doit avoir au moins un sous-pas");
  using Method = MethodT;
  static constexpr TimeTreatment treatment = TreatmentT;
  static constexpr int substeps = SubstepsT;
};

template <class T>
struct TimePolicyTraits {
  using Method = T;
  static constexpr TimeTreatment treatment = TimeTreatment::Explicit;
  static constexpr int substeps = 1;
};

template <class MethodT, TimeTreatment TreatmentT, int SubstepsT>
struct TimePolicyTraits<TimePolicy<MethodT, TreatmentT, SubstepsT>> {
  using Method = MethodT;
  static constexpr TimeTreatment treatment = TreatmentT;
  static constexpr int substeps = SubstepsT;
};

template <class MethodT = SSPRK2, int SubstepsT = 1>
using ExplicitTime = TimePolicy<MethodT, TimeTreatment::Explicit, SubstepsT>;

template <class MethodT = UserTimeIntegrator, int SubstepsT = 1>
using ImplicitTime = TimePolicy<MethodT, TimeTreatment::Implicit, SubstepsT>;

template <class MethodT = UserTimeIntegrator, int SubstepsT = 1>
using IMEXTime = TimePolicy<MethodT, TimeTreatment::IMEX, SubstepsT>;

using PrescribedTime = TimePolicy<UserTimeIntegrator, TimeTreatment::Prescribed, 1>;

}  // namespace adc
