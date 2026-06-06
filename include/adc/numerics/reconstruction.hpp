/// @file
/// @brief Politiques de reconstruction aux interfaces : limiteurs MUSCL et WENO5-Z.
///
/// Chaque politique expose :
///   - `n_ghost` : rayon de stencil requis (1 = premier ordre, 2 = MUSCL lineaire, 3 = WENO5).
///   - `operator()(am, ap)` : pente limitee depuis les differences arriere (am) et avant (ap).
///
/// Toutes les politiques sont ADC_HD (pas de std::, pas de branchement a UB). Le limiteur est
/// un parametre de template dans assemble_rhs / reconstruct (polymorphisme statique, inline sur
/// device). INVARIANT : une politique de reconstruction est PONCTUELLE -- elle ne boucle pas sur
/// la grille, n'accede a aucun tableau global. L'acces au stencil maillage est dans reconstruct
/// (spatial_operator.hpp).

#pragma once

#include <adc/core/types.hpp>

#include <cmath>

// Limiteurs de pente pour la reconstruction MUSCL. Chaque limiteur expose son
// nombre de ghosts requis (1 = premier ordre sans pente, 2 = reconstruction
// lineaire limitee) et une fonction (diff arriere, diff avant) -> pente limitee.
//
// Choix de conception : le limiteur est un parametre de template (polymorphisme
// statique, kernel inlinable, device-safe). NoSlope court-circuite tout calcul
// de pente, donc le chemin premier ordre ne lit qu'un ghost.

namespace adc {

/// Reconstruction premier ordre (constante par morceaux) : pente nulle, 1 ghost.
///
/// Politique minimale : aucun calcul de pente, aucune lecture de voisin a distance >= 2.
/// Le chemin n_ghost == 1 dans reconstruct ne touche pas les cellules a +/-2. ADC_HD.
/// INVARIANT : renvoie toujours Real(0) -- l'etat de cellule n'est pas modifie.
struct NoSlope {
  static constexpr int n_ghost = 1;
  ADC_HD Real operator()(Real, Real) const { return Real(0); }
};

// minmod : TVD, robuste, mais ecrete les extrema lisses (ordre 1 local aux pics).
/// Limiteur minmod : TVD (Total Variation Diminishing), 2 ghosts, ordre 2 en zone lisse.
///
/// Renvoie min(|a|,|b|)*sgn(a) si a et b sont de meme signe, 0 sinon. Implemente sans
/// std::min / std::abs pour rester device-safe (pas de <cmath> requis). Ordre 1 local
/// aux extrema (efface les pics lisses) : privilegier VanLeer pour les modes de Diocotron.
struct Minmod {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    if (a * b <= Real(0)) return Real(0);
    const Real fa = a < 0 ? -a : a, fb = b < 0 ? -b : b;  // |.| device-safe
    return (fa < fb) ? a : b;
  }
};

// van Leer : limiteur lisse, meilleur ordre aux extrema que minmod.
/// Limiteur van Leer : lisse, 2 ghosts, meilleur ordre aux extrema que Minmod.
///
/// Moyenne harmonique des differences : 2ab/(a+b) si meme signe, 0 sinon. Pas de
/// branchement de signe (pas de std::abs). Prefere a Minmod pour la preservation des
/// modes de croissance Diocotron (moins dissipatif aux extrema du profil de densite).
struct VanLeer {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    const Real ab = a * b;
    if (ab <= Real(0)) return Real(0);
    return Real(2) * ab / (a + b);
  }
};

// WENO5 (Weighted Essentially Non-Oscillatory, ordre 5 en zones lisses) : la valeur
// reconstruite a l'interface +x (resp. +y) de la cellule centrale d'un stencil 5 points
// est une combinaison NON LINEAIRE de trois reconstructions d'ordre 3, ponderee par des
// indicateurs de regularite (beta). En zone lisse les poids non lineaires tendent vers les
// poids lineaires optimaux (1/10, 6/10, 3/10) -> ordre 5 ; pres d'un front raide (le bord
// d'anneau) les poids ecartent le stencil qui traverse le saut -> capture sans oscillation.
// Variante WENO-Z (Borges 2008) : moins dissipative que Jiang-Shu en zone lisse (poids via
// tau5 = |beta0 - beta2|), donc meilleure preservation du taux de croissance d'un mode lisse.
//
// weno5z(vm2, vm1, v0, vp1, vp2) rend la valeur a l'interface ENTRE v0 et vp1 (face +x de la
// cellule v0). Pour la face -x, passer le stencil RENVERSE (vp2, vp1, v0, vm1, vm2). Stencil
// 5 points -> 3 ghosts. Lisse (pas de branchements de signe) donc device-callable.

/// weno5z : reconstruction WENO5-Z (Borges 2008) a une interface, sur stencil 5 points.
///
/// Rend la valeur reconstruite a la face ENTRE v0 et vp1 (face +dir de la cellule v0).
/// Pour la face -dir, appeler weno5z(vp2, vp1, v0, vm1, vm2) (stencil renverse). ADC_HD.
/// INVARIANT : calcul purement combinatoire, sans branchement sur les signes -- les
/// indicateurs beta et tau5 sont des carres, toujours >= 0 ; seule la valeur absolue de
/// (b0-b2) est prise via un ternaire (device-safe, evite std::abs).
/// Ne doit PAS etre appele directement par un utilisateur du maillage : passer par la
/// politique Weno5 et la fonction reconstruct de spatial_operator.hpp.
ADC_HD inline Real weno5z(Real vm2, Real vm1, Real v0, Real vp1, Real vp2) {
  const Real eps = Real(1e-40);
  // trois reconstructions d'ordre 3 de la face +x de v0
  const Real q0 = (Real(2) * vm2 - Real(7) * vm1 + Real(11) * v0) / Real(6);
  const Real q1 = (-vm1 + Real(5) * v0 + Real(2) * vp1) / Real(6);
  const Real q2 = (Real(2) * v0 + Real(5) * vp1 - vp2) / Real(6);
  // indicateurs de regularite (Jiang-Shu)
  const Real b0 = Real(13) / Real(12) * (vm2 - 2 * vm1 + v0) * (vm2 - 2 * vm1 + v0) +
                  Real(0.25) * (vm2 - 4 * vm1 + 3 * v0) * (vm2 - 4 * vm1 + 3 * v0);
  const Real b1 = Real(13) / Real(12) * (vm1 - 2 * v0 + vp1) * (vm1 - 2 * v0 + vp1) +
                  Real(0.25) * (vm1 - vp1) * (vm1 - vp1);
  const Real b2 = Real(13) / Real(12) * (v0 - 2 * vp1 + vp2) * (v0 - 2 * vp1 + vp2) +
                  Real(0.25) * (3 * v0 - 4 * vp1 + vp2) * (3 * v0 - 4 * vp1 + vp2);
  // poids WENO-Z : alpha_k = d_k (1 + (tau5/(eps+beta_k))^2), tau5 = |beta0 - beta2|
  const Real tau5 = (b0 - b2 < 0 ? b2 - b0 : b0 - b2);
  const Real a0 = (Real(1) / Real(10)) * (Real(1) + (tau5 / (eps + b0)) * (tau5 / (eps + b0)));
  const Real a1 = (Real(6) / Real(10)) * (Real(1) + (tau5 / (eps + b1)) * (tau5 / (eps + b1)));
  const Real a2 = (Real(3) / Real(10)) * (Real(1) + (tau5 / (eps + b2)) * (tau5 / (eps + b2)));
  const Real inv = Real(1) / (a0 + a1 + a2);
  return (a0 * q0 + a1 * q1 + a2 * q2) * inv;
}

// Politique de reconstruction WENO5 (ordre 5, 3 ghosts). N'utilise PAS l'interface lim(am, ap)
// des limiteurs de pente MUSCL (la reconstruction lit tout le stencil) : operator() est un
// no-op present pour rester un Limiter valide. spatial_operator::reconstruct route sur weno5z
// quand n_ghost >= 3.
/// Politique de tag WENO5 : marque le stencil a 3 ghosts, delegue a weno5z.
///
/// N'implemente pas lim(am, ap) au sens MUSCL (operator() est un no-op) : la reconstruction
/// WENO5 lit directement le stencil 5 points depuis reconstruct (chemin n_ghost >= 3).
/// L'operator() factice est present pour satisfaire le concept Limiter (compatible avec
/// toutes les fonctions template qui attendent un limiteur).
struct Weno5 {
  static constexpr int n_ghost = 3;
  ADC_HD Real operator()(Real, Real) const { return Real(0); }
};

}  // namespace adc
