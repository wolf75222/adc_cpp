/// @file
/// @brief Politiques de flux numerique a une interface : Rusanov, HLL, HLLC, Roe.
///
/// Chaque politique est un foncteur ADC_HD sans etat satisfaisant le contrat :
///   operator()(model, UL, AL, UR, AR, dir) -> Model::State
/// qui rend le flux numerique a l'interface entre l'etat gauche (UL, aux AL) et droit (UR, AR)
/// dans la direction dir (0 = x, 1 = y). Etats et auxiliaires passes par valeur ; aucun virtuel.
///
/// Hierarchie de precision (coupe des ondes intermediaires) ET de generalite :
///   RusanovFlux  : GENERIQUE minimal ; ne demande que max_wave_speed (tout PhysicalModel).
///   HLLFlux      : GENERIQUE avec ondes signees ; requiert model.wave_speeds (sL, sR).
///   HLLCFlux     : EULER 2D SEULEMENT (n_vars == 4, layout rho/m/E, pression) -- alias
///                  EulerHLLCFlux2D. Comportement indefini sur tout autre modele.
///   RoeFlux      : EULER 2D GAZ PARFAIT SEULEMENT (eigenstructure Euler codee en dur, gamma-1
///                  deduit de l'EOS gaz parfait, entropy fix de Harten eps = 0.1*c -- une politique
///                  d'entropie SPECIFIQUE Euler/Roe) -- alias EulerRoeFlux2D.
///
/// Pour un modele NON Euler (systeme de moments, isotherme, scalaire...), le chemin generique est
/// RusanovFlux, ou HLLFlux des que le modele expose wave_speeds.
///
/// CAPABILITIES RIEMANN (audit 2026-06, generalisation long terme) : HLLC et Roe ne sont plus des
/// algorithmes Euler-only DEGUISES en generiques -- ce sont des ALGORITHMES generiques dont la
/// STRUCTURE PHYSIQUE est fournie par le modele via des traits optionnels :
///   - HasHLLCStructure : le modele fournit contact_speed (vitesse de l'onde de contact) et
///     hllc_star_state (son etat etoile). HLLCFlux devient alors un solveur contact-resolving
///     GENERIQUE (l'algorithme F* = F_k + s_k (U*_k - U_k) ne connait plus le layout).
///   - HasRoeDissipation : le modele fournit sa dissipation de Roe complete
///     d = |A_roe| (U_R - U_L) (linearisation, eigenstructure et entropy fix INCLUS, qui sont des
///     proprietes du MODELE, pas du coeur). RoeFlux devient F = 1/2 (F_L + F_R) - 1/2 d.
/// Le chemin Euler 2D canonique (n_vars == 4 + pressure, sans hooks) reste l'implementation
/// historique BIT-IDENTIQUE : les capabilities ne font qu'OUVRIR ces solveurs aux autres systemes.
///
/// INVARIANT device : pas de vtable, pas de std:: dans les chemins critiques (std::sqrt
/// est authorise dans RoeFlux pour la moyenne de Roe, device-clean sous Kokkos/nvcc).

#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>     // std::sqrt (RoeFlux : moyenne de Roe) ; libstdc++ ne le tire pas transitivement
#include <concepts>  // capabilities Riemann (HasHLLCStructure / HasRoeDissipation)

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

/// RusanovFlux (Lax-Friedrichs local) : flux robuste, compatible avec tout PhysicalModel minimal.
///
/// Fhat = 1/2 (F(UL) + F(UR)) - 1/2 alpha (UR - UL), alpha = max(sL, sR).
/// Ne requiert que model.max_wave_speed -- compatible TOUT PhysicalModel du concept de base
/// (pas de wave_speeds, pas de pressure). Diffusif (alpha borne haute) mais universel.
/// ADC_HD. INVARIANT : traitement composante par composante (upwind scalaire), sans couplage.
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
/// hll_speeds : estimees de Davis pour les vitesses de signal des solveurs HLL/HLLC.
///
/// sL = min(sL_gauche, sL_droit), sR = max(sR_gauche, sR_droit).
/// Requiert model.wave_speeds(U, aux, dir, lo, hi) -> vitesses signees (cf. Euler).
/// Partagee par HLLFlux et HLLCFlux. ADC_HD.
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
/// HLLFlux (Harten-Lax-van Leer) : 2 vitesses de signal, moins diffusif que Rusanov.
///
/// Requiert model.wave_speeds (vitesses signees sL, sR). Moins diffusif que Rusanov sur
/// chocs et detentes ; capte encore imparfaitement les discontinuites de contact (une seule
/// region etoile). Retourne FL si sL >= 0, FR si sR <= 0, flux HLL sinon. ADC_HD.
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

// ---------------------------------------------------------------------------------------------
// CAPABILITIES RIEMANN : traits OPTIONNELS par lesquels un modele NON Euler fournit la structure
// physique exigee par un solveur contact-resolving (HLLC) ou Roe-like. cf. l'en-tete du fichier.
// ---------------------------------------------------------------------------------------------

/// Capability HLLC : le modele fournit la vitesse de l'onde de CONTACT et l'ETAT ETOILE du cote k.
///  - contact_speed(UL, UR, pL, pR, sL, sR, dir) -> s* (vitesse de l'onde intermediaire) ;
///  - hllc_star_state(U, p, s, s_star, dir) -> U*_k (etat etoile du cote k, s = sL ou sR).
/// Avec pressure + wave_speeds, HLLCFlux applique alors l'algorithme GENERIQUE
/// F* = F_k + s_k (U*_k - U_k) sans aucune hypothese de layout. Les deux methodes doivent etre
/// ADC_HD (appelees dans les kernels).
template <class M>
concept HasHLLCStructure =
    requires(const M m, const typename M::State u, const typename M::State v, const Aux a, Real p,
             Real q, Real sl, Real sr, Real ss, int dir) {
      { m.pressure(u) } -> std::convertible_to<Real>;
      m.wave_speeds(u, a, dir, sl, sr);  // vitesses signees (hll_speeds, bornes des ondes externes)
      { m.contact_speed(u, v, p, q, sl, sr, dir) } -> std::convertible_to<Real>;
      { m.hllc_star_state(u, p, sl, ss, dir) } -> std::same_as<typename M::State>;
    };

/// Capability Roe : le modele fournit sa dissipation de Roe COMPLETE
/// d = |A_roe(UL, UR)| (UR - UL) -- moyenne de Roe, decomposition en ondes, entropy fix inclus
/// (ce sont des proprietes du systeme physique, pas du coeur). RoeFlux devient alors
/// F = 1/2 (F_L + F_R) - 1/2 d, sans aucune hypothese Euler. ADC_HD requis.
template <class M>
concept HasRoeDissipation =
    requires(const M m, const typename M::State ul, const Aux al, const typename M::State ur,
             const Aux ar, int dir) {
      { m.roe_dissipation(ul, al, ur, ar, dir) } -> std::same_as<typename M::State>;
    };

// HLLC (HLL + onde de Contact) : restitue l'onde de contact (Toro), donc capture
// nettement la discontinuite de densite. Requiert model.pressure et wave_speeds.
// CHEMIN CANONIQUE : n_vars == 4 (Euler 2D), indices normal/tangentiel selon dir.
// CHEMIN CAPABILITY (HasHLLCStructure) : algorithme generique, structure fournie par le modele.
/// HLLCFlux (HLL + onde de Contact, Toro) : 3 ondes, capture la discontinuite de densite.
///
/// Requiert model.pressure et model.wave_speeds. Cible Euler 2D (n_vars == 4) ;
/// indices de quantite de mouvement normale/tangentielle selon dir. Vitesse etoile sStar
/// calculee par la formule de Toro eq. 10.37. Retombe sur FL / FR en zone supersonique. ADC_HD.
/// INVARIANT : n_vars == 4 suppose ; comportement indefini sur d'autres modeles.
struct HLLCFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    // CHEMIN CAPABILITY (HasHLLCStructure) : algorithme HLLC GENERIQUE -- la vitesse de contact et
    // les etats etoiles viennent du MODELE, le coeur n'assume ni layout ni EOS. Un modele Euler
    // canonique SANS hooks prend la branche historique ci-dessous, bit-identique.
    if constexpr (HasHLLCStructure<Model>) {
      Real sL, sR;
      hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
      const auto FL = m.flux(UL, AL, dir);
      const auto FR = m.flux(UR, AR, dir);
      if (sL >= 0) return FL;
      if (sR <= 0) return FR;
      const Real pL = m.pressure(UL), pR = m.pressure(UR);
      const Real sStar = m.contact_speed(UL, UR, pL, pR, sL, sR, dir);
      typename Model::State F;
      if (sStar >= 0) {
        const typename Model::State Us = m.hllc_star_state(UL, pL, sL, sStar, dir);
        for (int c = 0; c < Model::n_vars; ++c) F[c] = FL[c] + sL * (Us[c] - UL[c]);
      } else {
        const typename Model::State Us = m.hllc_star_state(UR, pR, sR, sStar, dir);
        for (int c = 0; c < Model::n_vars; ++c) F[c] = FR[c] + sR * (Us[c] - UR[c]);
      }
      return F;
    } else {
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
    }  // fin du chemin canonique Euler 2D (else du if constexpr HasHLLCStructure)
  }
};

/// Largeur du lissage de l'entropy fix de Harten du RoeFlux, en fraction de la vitesse du son de
/// Roe (eps = kRoeEntropyFixFraction * c). Constante DOCUMENTEE plutot que cachee dans le noyau ;
/// SPECIFIQUE Euler/Roe (cf. le commentaire dans RoeFlux::operator()).
inline constexpr Real kRoeEntropyFixFraction = Real(0.1);

// Roe (linearisation de Roe + correction d'entropie de Harten sur les ondes acoustiques). Capture
// nettement contacts et chocs (comme HLLC), mais via la decomposition complete en ondes : pour un
// etat supersonique, F* = flux amont EXACT (propriete de Roe : F_R - F_L = A_roe (U_R - U_L)).
// Euler 2D (n_vars == 4) : requiert m.pressure. gamma-1 est deduit de l'etat (gaz parfait), donc
// aucun membre gamma n'est requis du modele. Indices normal/tangentiel selon dir.
/// RoeFlux : linearisation de Roe + correction d'entropie de Harten (ondes acoustiques).
///
/// Decomposition COMPLETE en ondes propres : F_R - F_L = A_roe (U_R - U_L) exactement.
/// Pour un etat supersonique, F* = flux amont exact. Correction d'entropie de Harten
/// (eps = 0.1*c) sur ondes acoustiques : evite les chocs non-entropiques (sonic glitch).
/// Requiert model.pressure ; gamma-1 deduit de l'etat courant (hypothese gaz parfait).
/// std::sqrt utilise pour la moyenne de Roe (device-clean sous Kokkos/nvcc). ADC_HD.
/// INVARIANT : n_vars == 4 suppose (Euler 2D) ; comportement indefini sur autres modeles.
struct RoeFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                          const Aux& AL, const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    // CHEMIN CAPABILITY (HasRoeDissipation) : solveur Roe-like GENERIQUE -- la dissipation
    // d = |A_roe| (UR - UL) (linearisation + eigenstructure + entropy fix) vient du MODELE.
    // Un modele Euler canonique SANS hook prend la branche historique ci-dessous, bit-identique.
    if constexpr (HasRoeDissipation<Model>) {
      const auto FL = m.flux(UL, AL, dir);
      const auto FR = m.flux(UR, AR, dir);
      const typename Model::State d = m.roe_dissipation(UL, AL, UR, AR, dir);
      typename Model::State F;
      for (int c = 0; c < Model::n_vars; ++c)
        F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * d[c];
      return F;
    } else {
    const int in = (dir == 0) ? 1 : 2;  // qte de mvt normale
    const int it = (dir == 0) ? 2 : 1;  // tangentielle
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real utL = UL[it] / rL, utR = UR[it] / rR;
    const Real pL = m.pressure(UL), pR = m.pressure(UR);
    const Real HL = (UL[3] + pL) / rL, HR = (UR[3] + pR) / rR;

    // moyenne de Roe (ponderee par sqrt(rho))
    const Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), den = sqL + sqR;
    const Real un = (sqL * unL + sqR * unR) / den;
    const Real ut = (sqL * utL + sqR * utR) / den;
    const Real H = (sqL * HL + sqR * HR) / den;
    const Real rho = sqL * sqR;
    const Real q2 = un * un + ut * ut;
    // gamma-1 deduit du gaz parfait : p = (gamma-1) (E - 1/2 rho |v|^2)
    const Real gm1 = pL / (UL[3] - Real(0.5) * rL * (unL * unL + utL * utL));
    const Real c2 = gm1 * (H - Real(0.5) * q2);
    const Real c = std::sqrt(c2);

    // sauts et amplitudes d'onde
    const Real dr = rR - rL, dp = pR - pL, dun = unR - unL, dut = utR - utL;
    const Real a1 = (dp - rho * c * dun) / (Real(2) * c2);  // onde un - c
    const Real a2 = dr - dp / c2;                           // entropie, un
    const Real a3 = rho * dut;                              // cisaillement, un
    const Real a5 = (dp + rho * c * dun) / (Real(2) * c2);  // onde un + c

    // |valeur propre| avec correction d'entropie de Harten sur les ondes acoustiques (1, 5).
    // kRoeEntropyFixFraction = 0.1 est une POLITIQUE D'ENTROPIE EULER/ROE-SPECIFIQUE (largeur du
    // lissage parabolique en fraction de la vitesse du son de Roe, valeur usuelle de la litterature) :
    // elle n'a pas de sens pour un autre systeme hyperbolique et ne doit pas etre presentee comme
    // un parametre generique du coeur.
    const Real eps = kRoeEntropyFixFraction * c;
    auto absfix = [eps](Real l) {
      const Real al = l < 0 ? -l : l;
      return al < eps ? Real(0.5) * (l * l / eps + eps) : al;
    };
    const Real al1 = absfix(un - c), al2 = (un < 0 ? -un : un), al5 = absfix(un + c);

    // dissipation Sum |lambda_k| a_k r_k, base (rho, mom_n, mom_t, E)
    const Real d_rho = al1 * a1 + al2 * a2 + al5 * a5;
    const Real d_mn = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);
    const Real d_mt = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;
    const Real d_E = al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + a3 * ut) +
                     al5 * a5 * (H + un * c);

    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    typename Model::State F;
    F[0] = Real(0.5) * (FL[0] + FR[0]) - Real(0.5) * d_rho;
    F[in] = Real(0.5) * (FL[in] + FR[in]) - Real(0.5) * d_mn;
    F[it] = Real(0.5) * (FL[it] + FR[it]) - Real(0.5) * d_mt;
    F[3] = Real(0.5) * (FL[3] + FR[3]) - Real(0.5) * d_E;
    return F;
    }  // fin du chemin canonique Euler 2D gaz parfait (else du if constexpr HasRoeDissipation)
  }
};

/// Alias EXPLICITES du domaine de validite : HLLCFlux et RoeFlux sont des solveurs EULER 2D
/// (n_vars == 4, layout rho/m_x/m_y/E, pression gaz parfait), PAS des solveurs generiques.
/// Les noms courts restent pour compatibilite (make_block et les .so generes les referencent) ;
/// le code neuf qui veut nommer l'hypothese peut employer ces alias.
using EulerHLLCFlux2D = HLLCFlux;
using EulerRoeFlux2D = RoeFlux;

}  // namespace adc
