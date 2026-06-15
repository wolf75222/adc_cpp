#pragma once

#include <adc/core/types.hpp>  // Real, ADC_HD

/// @file
/// @brief Primitive PARTAGEE de fraction de coupe (cut-cell / embedded boundary).
///
/// Une SEULE computation de croisement de face (face crossing) entre :
///   - le solveur elliptique (geometric_mg.hpp : poids de Shortley-Weller du mur de Poisson),
///   - le futur transport EB (aperture FV des faces du disque).
/// Les deux DOIVENT lire la MEME geometrie d'ouverture pour que l'aperture FV soit bit-coherente
/// avec le mur elliptique (verrou "bords d'anneau cartesiens" ; cf. docs/HOFFART_FIDELITY.md,
/// ligne "Domain (disc of radius R)" du tableau de fidelite, le "Cartesian-ring-edge lock").
///
/// Le level-set canonique est detail::DiscDomain::level_set (wall_predicate.hpp, chantier T2) :
///   ls(x, y) = hypot(x - cx, y - cy) - R, < 0 a l'INTERIEUR, signe du bord.
/// Cette primitive est header-only, ADC_HD (device-safe) et SANS etat : elle prend un callback de
/// level-set par valeur et la cellule, et rend les distances/aperture purement geometriques.

namespace adc {
namespace detail {

/// Distance de coupe d'UNE face le long d'une direction, en partant du centre actif (ls < 0).
///
/// Convention HISTORIQUE de GeometricMG (geometric_mg.hpp, lambda 'cut') reprise A L'IDENTIQUE pour
/// garantir des poids Shortley-Weller bit-identiques :
///   - voisin INTERIEUR (ln < 0) : pas de coupe, la face est pleine -> distance = h ;
///   - le level-set change de signe (ln >= 0) : fraction lineaire theta = lc / (lc - ln) (crossing
///     lineaire entre le centre lc < 0 et le voisin ln >= 0), distance = theta * h ;
///   - garde-fou anti division par 0 : theta plante a [1e-3, 1] (theta -> 0 ferait diverger le poids).
/// lc est suppose < 0 (cellule active) ; les bornes du clamp sont celles d'origine.
ADC_HD inline Real cut_distance(Real lc, Real ln, Real h) {
  if (ln < Real(0)) return h;             // voisin interieur : pas de coupe (face pleine)
  Real th = lc / (lc - ln);               // ls change de signe : fraction de coupe lineaire
  if (th < Real(1e-3)) th = Real(1e-3);   // garde-fou anti division par 0 (theta -> 0)
  if (th > Real(1)) th = Real(1);
  return th * h;
}

/// Resultat geometrique du croisement d'une cellule coupee : 4 distances de coupe par face, les 4
/// apertures alpha_f normalisees dans [0, 1] (alpha_f = distance_face / h), et la fraction de volume
/// kappa de la cellule (part de la cellule dans le domaine actif). axm/axp portent la direction x
/// (voisins i-1 / i+1), aym/ayp la direction y (voisins j-1 / j+1).
///
/// alpha_f et kappa sont les quantites que le TRANSPORT EB consommera (flux de face attenue par
/// alpha_f, volume de cellule kappa) ; axm/axp/aym/ayp sont ce que l'ELLIPTIQUE consomme deja pour
/// les poids de Shortley-Weller. Tout est derive de la MEME cut_distance -> coherence bit-a-bit.
struct CutFraction {
  Real axm;    ///< distance de coupe face x- (vers i-1), dans [1e-3*dx, dx]
  Real axp;    ///< distance de coupe face x+ (vers i+1)
  Real aym;    ///< distance de coupe face y- (vers j-1)
  Real ayp;    ///< distance de coupe face y+ (vers j+1)
  Real alpha_xm;  ///< aperture face x- = axm / dx, dans [1e-3, 1]
  Real alpha_xp;  ///< aperture face x+ = axp / dx
  Real alpha_ym;  ///< aperture face y- = aym / dy
  Real alpha_yp;  ///< aperture face y+ = ayp / dy
  Real kappa;     ///< fraction de volume de la cellule (part dans le domaine actif), dans (0, 1]
};

/// Calcule la geometrie de coupe d'une cellule ACTIVE (centre (xc, yc) avec ls < 0) a partir d'un
/// level-set @p ls evalue au centre et aux 4 voisins cardinaux distants de @p dx / @p dy.
///
/// @tparam LevelSet callable Real(Real, Real) device-safe (p.ex. DiscDomain::level_set capture).
///
/// La cellule est supposee ACTIVE (l'appelant a deja teste ls(xc, yc) < 0, comme GeometricMG saute
/// les cellules conductrices). Les 4 distances de face reutilisent cut_distance (donc strictement la
/// logique d'origine). Les apertures normalisent par le pas. kappa est une fraction de volume DERIVEE
/// des memes apertures (moyenne des deux demi-faces par direction, produit des deux directions) :
/// loin du bord (toutes apertures = 1) kappa = 1 ; pres du bord kappa < 1. kappa n'altere PAS
/// l'elliptique (qui n'utilise que axm/axp/aym/ayp) ; il est fourni pour le transport EB a venir.
template <class LevelSet>
ADC_HD inline CutFraction cut_fraction(const LevelSet& ls, Real xc, Real yc, Real dx, Real dy) {
  const Real lc = ls(xc, yc);
  const Real axm = cut_distance(lc, ls(xc - dx, yc), dx);
  const Real axp = cut_distance(lc, ls(xc + dx, yc), dx);
  const Real aym = cut_distance(lc, ls(xc, yc - dy), dy);
  const Real ayp = cut_distance(lc, ls(xc, yc + dy), dy);
  const Real alpha_xm = axm / dx;
  const Real alpha_xp = axp / dx;
  const Real alpha_ym = aym / dy;
  const Real alpha_yp = ayp / dy;
  // Fraction de volume : moyenne des demi-faces par axe (etendue moyenne de la cellule selon chaque
  // direction, normalisee), produit des deux axes. Loin du bord -> 1 ; cellule coupee -> < 1.
  const Real kappa = Real(0.5) * (alpha_xm + alpha_xp) * Real(0.5) * (alpha_ym + alpha_yp);
  return CutFraction{axm, axp, aym, ayp, alpha_xm, alpha_xp, alpha_ym, alpha_yp, kappa};
}

/// Poids de Shortley-Weller (stencil 5 points cut-cell) a partir des 4 distances de coupe. Renvoie
/// EXACTEMENT les 5 coefficients que GeometricMG ecrit dans son champ coef (composantes 0..4) :
///   w_xm = 2 / (axm (axm + axp)),  w_xp = 2 / (axp (axm + axp)),
///   w_ym = 2 / (aym (aym + ayp)),  w_yp = 2 / (ayp (aym + ayp)),
///   w_diag = 2 / (axm axp) + 2 / (aym ayp).
/// Centralise la formule pour que l'assemblage elliptique reste l'unique source de verite cut-cell.
struct ShortleyWellerWeights {
  Real w_xm, w_xp, w_ym, w_yp, w_diag;
};

ADC_HD inline ShortleyWellerWeights shortley_weller(const CutFraction& cf) {
  const Real sx = cf.axm + cf.axp;
  const Real sy = cf.aym + cf.ayp;
  return ShortleyWellerWeights{
      Real(2) / (cf.axm * sx),                          // w_xm sur p(i-1)
      Real(2) / (cf.axp * sx),                          // w_xp sur p(i+1)
      Real(2) / (cf.aym * sy),                          // w_ym sur p(i,j-1)
      Real(2) / (cf.ayp * sy),                          // w_yp sur p(i,j+1)
      Real(2) / (cf.axm * cf.axp) + Real(2) / (cf.aym * cf.ayp)};  // w_diag
}

}  // namespace detail
}  // namespace adc
