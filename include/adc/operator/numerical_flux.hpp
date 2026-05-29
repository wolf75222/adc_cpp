#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

// Flux numerique a une interface, exprime en POLITIQUE (template), au meme titre
// que le limiteur de reconstruction. assemble_rhs<Limiter, NumericalFlux> choisit
// les deux independamment, au lieu d'appeler en dur rusanov_flux.
//
// Contrat d'une politique de flux : un foncteur device-callable (ADC_HD)
//   operator()(model, UL, AL, UR, AR, dir) -> Model::State
// qui rend le flux numerique a l'interface entre l'etat gauche (UL, aux AL) et
// droit (UR, aux AR) dans la direction dir. Etats par valeur, aucun virtuel :
// utilisable tel quel dans un kernel.
//
//   RusanovFlux : Lax-Friedrichs local, alpha = max des |vitesses d'onde| des deux
//                 etats. Robuste, diffusif. Seul flux ne demandant que
//                 max_wave_speed (donc compatible avec le concept PhysicalModel
//                 actuel sans extension).
//
// HLL / HLLC arriveront avec model/euler.hpp : ils exigent les vitesses d'onde
// SIGNEES s_L, s_R (et l'onde de contact s_* pour HLLC), donc une extension du
// concept (p.ex. wave_speeds(U, aux, dir) -> {sL, sR}). Rusanov n'en a pas besoin.

namespace adc {

struct RusanovFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    const Real sL = m.max_wave_speed(UL, AL, dir);
    const Real sR = m.max_wave_speed(UR, AR, dir);
    const Real alpha = sL > sR ? sL : sR;  // max device-safe (pas de std::max)
    typename Model::State F;
    for (int c = 0; c < Model::n_vars; ++c)
      F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * alpha * (UR[c] - UL[c]);
    return F;
  }
};

// Estimation des vitesses de signal (Davis) : sL = min, sR = max des vitesses
// d'onde signees des deux etats. Requiert model.wave_speeds (cf. Euler).
template <class Model>
ADC_HD inline void hll_speeds(const Model& m, const typename Model::State& UL,
                              const Aux& AL, const typename Model::State& UR,
                              const Aux& AR, int dir, Real& sL, Real& sR) {
  Real lL, hL, lR, hR;
  m.wave_speeds(UL, AL, dir, lL, hL);
  m.wave_speeds(UR, AR, dir, lR, hR);
  sL = lL < lR ? lL : lR;
  sR = hL > hR ? hL : hR;
}

// HLL (Harten-Lax-van Leer) : une seule onde intermediaire (pas d'onde de
// contact). Moins diffusif que Rusanov sur chocs et detentes ; lisse encore les
// discontinuites de contact.
struct HLLFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    Real sL, sR;
    hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    if (sL >= 0) return FL;
    if (sR <= 0) return FR;
    typename Model::State F;
    const Real inv = Real(1) / (sR - sL);
    for (int c = 0; c < Model::n_vars; ++c)
      F[c] = (sR * FL[c] - sL * FR[c] + sL * sR * (UR[c] - UL[c])) * inv;
    return F;
  }
};

// HLLC (HLL + onde de Contact) : restitue l'onde de contact (Toro), donc capture
// nettement la discontinuite de densite. Requiert model.pressure et wave_speeds.
// n_vars == 4 attendu (Euler 2D) : indices normal/tangentiel selon dir.
struct HLLCFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // composante de qte de mvt normale
    const int it = (dir == 0) ? 2 : 1;  // tangentielle
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real pL = m.pressure(UL), pR = m.pressure(UR);
    Real sL, sR;
    hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    if (sL >= 0) return FL;
    if (sR <= 0) return FR;

    // vitesse de l'onde de contact (Toro 10.37)
    const Real sStar = (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
                       (rL * (sL - unL) - rR * (sR - unR));
    typename Model::State F;
    if (sStar >= 0) {  // etat star gauche
      const Real fac = rL * (sL - unL) / (sL - sStar);
      typename Model::State Us;
      Us[0] = fac;
      Us[in] = fac * sStar;
      Us[it] = fac * (UL[it] / rL);
      Us[3] = fac * (UL[3] / rL + (sStar - unL) * (sStar + pL / (rL * (sL - unL))));
      for (int c = 0; c < 4; ++c) F[c] = FL[c] + sL * (Us[c] - UL[c]);
    } else {  // etat star droit
      const Real fac = rR * (sR - unR) / (sR - sStar);
      typename Model::State Us;
      Us[0] = fac;
      Us[in] = fac * sStar;
      Us[it] = fac * (UR[it] / rR);
      Us[3] = fac * (UR[3] / rR + (sStar - unR) * (sStar + pR / (rR * (sR - unR))));
      for (int c = 0; c < 4; ++c) F[c] = FR[c] + sR * (Us[c] - UR[c]);
    }
    return F;
  }
};

}  // namespace adc
