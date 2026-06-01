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

// SubstepsT : sous-pas PLUS FREQUENTS (n pas de dt/n par macro-pas — electrons rapides).
// StrideT   : cadence PLUS LENTE (le bloc n'avance qu'1 macro-pas sur StrideT, alors d'un
//   pas de StrideT*dt — un « gaz » lent qu'on ne resout pas a chaque pas, retour tuteur).
//   Les deux sont orthogonaux ; StrideT=1 = comportement historique.
template <class MethodT, TimeTreatment TreatmentT, int SubstepsT = 1, int StrideT = 1>
struct TimePolicy {
  static_assert(SubstepsT >= 1, "un TimePolicy doit avoir au moins un sous-pas");
  static_assert(StrideT >= 1, "un TimePolicy doit avoir une cadence (stride) >= 1");
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

}  // namespace adc
