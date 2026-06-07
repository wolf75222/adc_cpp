#pragma once

#include <adc/core/types.hpp>  // Real

#include <cmath>       // std::hypot
#include <functional>  // std::function
#include <stdexcept>   // std::runtime_error
#include <string>

/// @file
/// @brief Predicat de paroi (conducteur embedded) partage par les runtimes System et AmrSystem.
///        Les deux derivaient le meme predicat depuis les memes parametres (wall, rayon, L) ;
///        seul le prefixe du message d'erreur differait. Centralise ici par extraction PURE :
///        le corps (cercle centre, comparaison std::hypot < R) est repris a l'identique.

namespace adc {
namespace detail {

/// Construit le predicat "interieur du conducteur" (paroi embedded pour le solveur de Poisson)
/// depuis le mode de paroi @p wall, le rayon @p wall_radius et la taille de domaine @p L.
///   - "none"   : pas de paroi -> predicat vide.
///   - "circle" : disque centre en (L/2, L/2) de rayon @p wall_radius.
///   - autre    : erreur, prefixee par @p err_context (p.ex. "System::set_poisson").
/// Corps repris a l'identique des runtimes System / AmrSystem (bit-identique).
inline std::function<bool(Real, Real)> wall_predicate(const std::string& wall,
                                                      double wall_radius, double L,
                                                      const std::string& err_context) {
  if (wall == "none") return {};
  if (wall == "circle") {
    const double cx = 0.5 * L, cy = 0.5 * L, R = wall_radius;
    return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
  }
  throw std::runtime_error(err_context + " : wall inconnu '" + wall + "'");
}

/// Descripteur de geometrie DISQUE : SOURCE UNIQUE de verite du domaine physique reel du papier
/// (Hoffart et al., arXiv:2510.11808, Sec 5.3 : disque D de rayon R). C'est le pendant "transport" du
/// mur de Poisson : le mur n'agit que sur l'elliptique (cf. wall_predicate / geometric_mg cut_cell),
/// alors que ce descripteur sert a construire un MASQUE DE DOMAINE cellule-centre pour rendre le
/// chemin de transport FV conscient du disque (verrou "bords d'anneau cartesiens", cf.
/// docs/HOFFART_FIDELITY.md ligne 39).
///
/// REUTILISE EXACTEMENT le level set du mur conducteur (geometric_mg.hpp ligne 71) :
///   ls(x, y) = hypot(x - cx, y - cy) - R, < 0 a l'INTERIEUR.
/// Une cellule est ACTIVE quand son CENTRE est dans le disque (ls < 0), exactement comme le predicat
/// d'interieur de GeometricMG (active = ls < 0). Le centre par defaut (cx, cy) = (L/2, L/2) coincide
/// avec celui de wall_predicate("circle", ...).
///
/// CONTRAT (chantier T2, inerte par defaut) : ce descripteur ne change RIEN au comportement par
/// defaut. Il n'est materialise (mask MultiFab, transport mask-aware) que sur opt-in explicite
/// (System::set_disc_domain). Tant qu'aucun disque n'est fixe, le masque est "tout actif" et le
/// chemin FV/AMR/MPI reste BIT-IDENTIQUE.
struct DiscDomain {
  double cx = 0.0;  ///< centre x (defaut L/2 quand construit depuis L)
  double cy = 0.0;  ///< centre y (defaut L/2 quand construit depuis L)
  double R = 0.0;   ///< rayon du disque

  /// Disque centre dans une boite carree [0, L]^2 de rayon @p radius (meme centre que
  /// wall_predicate("circle", radius, L) : (L/2, L/2)).
  static DiscDomain centered_in_box(double L, double radius) {
    return DiscDomain{0.5 * L, 0.5 * L, radius};
  }

  /// Level set ls(x, y) = hypot(x - cx, y - cy) - R : < 0 a l'interieur, 0 au bord, > 0 dehors.
  /// Identique a la convention du mur conducteur (geometric_mg cut_cell).
  ADC_HD Real level_set(Real x, Real y) const {
    return static_cast<Real>(std::hypot(static_cast<double>(x) - cx, static_cast<double>(y) - cy) - R);
  }

  /// Cellule ACTIVE : son centre (x, y) est dans le disque (ls < 0). Meme test d'interieur que
  /// l'active de GeometricMG (ls < 0).
  ADC_HD bool cell_active(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

}  // namespace detail
}  // namespace adc
