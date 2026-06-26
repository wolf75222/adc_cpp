// Primitive PARTAGEE de fraction de coupe (cut_fraction.hpp) : deux verifications.
//
// (1) UNITE ANALYTIQUE : sur un cercle, le croisement de face AXIAL est exactement lineaire (le
//     level-set ls(x, y) = hypot(x - cx, y - cy) - R restreint a l'axe vaut |x - cx| - R, lineaire),
//     donc cut_fraction doit rendre la distance EXACTE au cercle. On verifie l'aperture axiale, la
//     face interieure (pleine, distance = h), et kappa = 1 loin du bord.
//
// (2) BIT-IDENTITE : detail::shortley_weller(detail::cut_fraction(...)) reproduit EXACTEMENT (diff
//     0.0, pas de tolerance) les 5 poids Shortley-Weller assembles par la formule INLINE historique
//     de GeometricMG (l'ancienne lambda 'cut' + 2/(axm*(axm+axp)) ...). C'est la garantie "no behavior
//     change" du refactor : l'elliptique lit des coef byte-identiques avant/apres.

#include <pops/numerics/elliptic/eb/cut_fraction.hpp>

#include <cmath>
#include <cstdio>
#include <functional>

using namespace pops;

// level-set canonique du disque (meme convention que DiscDomain::level_set et le mur de Poisson).
struct DiscLS {
  double cx, cy, R;
  Real operator()(Real x, Real y) const {
    return static_cast<Real>(std::hypot(static_cast<double>(x) - cx, static_cast<double>(y) - cy) -
                             R);
  }
};

// Reference INLINE : copie EXACTE de l'ancien corps de GeometricMG (lambda 'cut' + formule des poids)
// AVANT le refactor. Sert d'oracle bit-identite : si la primitive devie d'un seul ULP, le test casse.
struct RefWeights {
  Real w_xm, w_xp, w_ym, w_yp, w_diag;
};
template <class LS>
static RefWeights ref_inline(const LS& ls, Real xc, Real yc, Real dx, Real dy) {
  auto cut = [](Real lc, Real ln, Real h) -> Real {
    if (ln < Real(0))
      return h;
    Real th = lc / (lc - ln);
    if (th < Real(1e-3))
      th = Real(1e-3);
    if (th > Real(1))
      th = Real(1);
    return th * h;
  };
  const Real lc = ls(xc, yc);
  const Real axm = cut(lc, ls(xc - dx, yc), dx);
  const Real axp = cut(lc, ls(xc + dx, yc), dx);
  const Real aym = cut(lc, ls(xc, yc - dy), dy);
  const Real ayp = cut(lc, ls(xc, yc + dy), dy);
  RefWeights w;
  w.w_xm = Real(2) / (axm * (axm + axp));
  w.w_xp = Real(2) / (axp * (axm + axp));
  w.w_ym = Real(2) / (aym * (aym + ayp));
  w.w_yp = Real(2) / (ayp * (aym + ayp));
  w.w_diag = Real(2) / (axm * axp) + Real(2) / (aym * ayp);
  return w;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // ---------------------------------------------------------------------------
  // (1) UNITE ANALYTIQUE : cercle unite centre a l'origine, R = 1.
  // ---------------------------------------------------------------------------
  const DiscLS ls{0.0, 0.0, 1.0};
  const Real dx = Real(0.2), dy = Real(0.2);

  // Cellule sur l'axe x, centre (0.9, 0) : ls = -0.1 (interieur). Voisin +x (1.1, 0) : ls = +0.1
  // (dehors). Croisement axial exact en x = R = 1.0 -> distance = 0.1, aperture = 0.5.
  {
    const detail::CutFraction cf = detail::cut_fraction(ls, Real(0.9), Real(0.0), dx, dy);
    // face +x coupee : distance analytique = R - 0.9 = 0.1, aperture = 0.5.
    chk(std::fabs(cf.axp - Real(0.1)) < Real(1e-12), "axp_distance_analytique");
    chk(std::fabs(cf.alpha_xp - Real(0.5)) < Real(1e-12), "alpha_xp_analytique");
    // face -x : voisin (0.7, 0) ls = -0.3 < 0 -> interieur, face PLEINE = dx, aperture = 1.
    chk(std::fabs(cf.axm - dx) < Real(1e-12), "axm_face_pleine");
    chk(std::fabs(cf.alpha_xm - Real(1.0)) < Real(1e-12), "alpha_xm_pleine");
    // faces y : voisins (0.9, +-0.2) a hypot 0.9219 < 1 -> interieurs, faces pleines.
    chk(std::fabs(cf.aym - dy) < Real(1e-12), "aym_face_pleine");
    chk(std::fabs(cf.ayp - dy) < Real(1e-12), "ayp_face_pleine");
    // kappa = 0.5(alpha_xm + alpha_xp) * 0.5(alpha_ym + alpha_yp) = 0.5(1+0.5)*0.5(1+1) = 0.75.
    chk(std::fabs(cf.kappa - Real(0.75)) < Real(1e-12), "kappa_cellule_coupee");
    // poids de Shortley-Weller coherents avec ces distances (axp = 0.1, axm = 0.2).
    const detail::ShortleyWellerWeights w = detail::shortley_weller(cf);
    const Real sx = cf.axm + cf.axp;  // 0.3
    chk(std::fabs(w.w_xp - Real(2) / (cf.axp * sx)) < Real(1e-12), "w_xp_formule");
    chk(w.w_xp > w.w_xm, "petite_face_poids_plus_grand");  // axp < axm => w_xp > w_xm
  }

  // Cellule PROFONDE (0, 0) : les 4 voisins (+-0.2, 0)/(0, +-0.2) sont a distance 0.2 < 1 -> tous
  // interieurs. Toutes les faces pleines, apertures = 1, kappa = 1 (cellule non coupee).
  {
    const detail::CutFraction cf = detail::cut_fraction(ls, Real(0.0), Real(0.0), dx, dy);
    chk(std::fabs(cf.alpha_xm - Real(1.0)) < Real(1e-12) &&
            std::fabs(cf.alpha_xp - Real(1.0)) < Real(1e-12) &&
            std::fabs(cf.alpha_ym - Real(1.0)) < Real(1e-12) &&
            std::fabs(cf.alpha_yp - Real(1.0)) < Real(1e-12),
        "interieur_apertures_unite");
    chk(std::fabs(cf.kappa - Real(1.0)) < Real(1e-12), "interieur_kappa_unite");
    // loin du bord les poids degenerent au stencil uniforme : 1/dx^2 = 25, diag = 2/dx^2 + 2/dy^2 = 100.
    const detail::ShortleyWellerWeights w = detail::shortley_weller(cf);
    chk(std::fabs(w.w_xm - Real(25.0)) < Real(1e-9) &&
            std::fabs(w.w_diag - Real(100.0)) < Real(1e-9),
        "interieur_stencil_uniforme");
  }

  // garde-fou anti division par 0 : theta plante a 1e-3 quand le centre est TRES pres du bord.
  {
    // centre a 1e-6 a l'interieur, voisin loin dehors -> theta brut ~ 5e-6 < 1e-3 -> clampe a 1e-3.
    const detail::CutFraction cf = detail::cut_fraction(ls, Real(1.0 - 1e-6), Real(0.0), dx, dy);
    chk(std::fabs(cf.alpha_xp - Real(1e-3)) < Real(1e-9), "clamp_theta_min");
  }

  // ---------------------------------------------------------------------------
  // (2) BIT-IDENTITE primitive vs formule inline historique, sur une grille de centres balayant
  //     le bord du disque (cx, cy) = (0.5, 0.5), R = 0.4, dx = dy = 1/64 : on traverse beaucoup de
  //     cellules coupees (toutes les configurations de croisement de face). Diff EXACTE 0.0.
  // ---------------------------------------------------------------------------
  {
    const DiscLS d{0.5, 0.5, 0.4};
    const int nc = 64;
    const Real h = Real(1.0) / nc;
    long active = 0, cut_cells = 0;
    Real max_diff = Real(0);
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        const Real xc = (i + Real(0.5)) * h, yc = (j + Real(0.5)) * h;
        if (d(xc, yc) >= Real(0))
          continue;  // conducteur : GeometricMG saute (coef = 0)
        ++active;
        const detail::CutFraction cf = detail::cut_fraction(d, xc, yc, h, h);
        const detail::ShortleyWellerWeights w = detail::shortley_weller(cf);
        const RefWeights r = ref_inline(d, xc, yc, h, h);
        // comparaison EXACTE (operator!=), pas de tolerance : doit etre byte-identique.
        if (w.w_xm != r.w_xm || w.w_xp != r.w_xp || w.w_ym != r.w_ym || w.w_yp != r.w_yp ||
            w.w_diag != r.w_diag) {
          max_diff = std::max(max_diff, std::fabs(w.w_xm - r.w_xm));
          max_diff = std::max(max_diff, std::fabs(w.w_xp - r.w_xp));
          max_diff = std::max(max_diff, std::fabs(w.w_diag - r.w_diag));
        }
        if (cf.alpha_xm < Real(1) || cf.alpha_xp < Real(1) || cf.alpha_ym < Real(1) ||
            cf.alpha_yp < Real(1))
          ++cut_cells;
      }
    std::printf("bit-identite : %ld cellules actives, %ld coupees, max_diff=%.3e\n", active,
                cut_cells, static_cast<double>(max_diff));
    chk(max_diff == Real(0), "bit_identite_poids_diff_exacte_0");
    chk(active > 1800, "balayage_couvre_le_disque");  // pi*R^2/h^2 = pi*0.4^2*64^2 ~ 2058 cellules
    chk(cut_cells > 100, "beaucoup_de_cellules_coupees");  // le bord du disque genere des coupures
  }

  if (fails == 0)
    std::printf("OK test_cut_fraction\n");
  return fails == 0 ? 0 : 1;
}
