// Chantier T5-PR2 : TRANSPORT FV EMBEDDED-BOUNDARY (EB) conservatif sur disque (assemble_rhs_eb).
//
// L'operateur assemble_rhs_eb generalise le chemin masque T2 (assemble_rhs_masked, portes de face 0/1)
// a un schema CUT-CELL : ouvertures de face alpha_f in [0, 1] et fraction de volume kappa derivees de
// detail::cut_fraction (T5-PR1), flux de paroi immergee no-penetration, residu divise par kappa (clampe).
//
// On valide TROIS proprietes du contrat (vraies assertions, pas de no-op) :
//
//   (a) MMS / SOLUTION MANUFACTUREE sur le disque : pour un transport LISSE a vitesse constante a
//       l'INTERIEUR du disque (loin du bord, ou alpha_f = 1 et kappa = 1), le residu discret -R converge
//       vers la divergence ANALYTIQUE -div F a l'ORDRE 2 (rapport d'erreur ~4 sous raffinement x2). On
//       mesure l'erreur sur les cellules PROFONDES (a >= 2 cellules du bord) pour isoler l'ordre du
//       schema EB volumique de l'erreur O(1) attendue sur la couche cut-cell elle-meme (le clamp et la
//       reconstruction non centree au bord plafonnent l'ordre la-bas ; PR2 cible le transport interieur).
//
//   (b) CONSERVATION DE MASSE a la PRECISION MACHINE sur le disque : aucun flux ne franchit la frontiere
//       immergee (faces vers l'inactif fermees + paroi no-penetration). La masse discrete coherente avec
//       le schema, Sum_active n_ij kappa_eff_ij dx dy, est conservee A LA MACHINE sur K pas Euler avant,
//       pour une advection a vitesse constante (div v = 0). C'est l'invariant EXACT du schema EB (le
//       kappa_eff au denominateur du residu se simplifie avec le kappa_eff de la masse -> telescopage des
//       flux de face, bords fermes).
//
//   (c) BIT-IDENTITE du chemin PAR DEFAUT : assemble_rhs_eb avec un level set qui rend TOUTE la grille
//       active et SANS aucune cellule coupee (disque englobant -> alpha_f = 1, kappa = 1 partout) est
//       STRICTEMENT egal (diff bit a bit = 0) a assemble_rhs (chemin cartesien historique). C'est
//       l'invariant "EB additif et inerte sans coupe" : pas de disque coupant => operateur cartesien pur.
//
// Modele jouet INLINE : advection scalaire a vitesse (vx, vy) constante. Aucune source, aucun elliptique.

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/spatial_operator_eb.hpp>
#include <adc/runtime/detail/wall_predicate.hpp>  // detail::DiscDomain (level set source-unique)

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kL = 1.0;       // boite carree [0, L]^2
static constexpr double kRdisc = 0.35;  // rayon du disque EB (cellules actives ET inactives)
static constexpr double kVx = 0.7;      // vitesse de transport (div v = 0)
static constexpr double kVy = -0.4;

// Advection scalaire a vitesse constante (vx, vy). Flux F = (vx u, vy u).
struct Advect {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real vx = 0.0, vy = 0.0;
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? vx : vy) * u[0]};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    return std::fabs(dir == 0 ? vx : vy);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<Advect>, "Advect est un PhysicalModel");

// Level set "dalle mince" |y - yc| - h : interieur ssi |y - yc| < h. Centre dans la dalle (ls < 0),
// les DEUX voisins en y dehors -> les deux demi-faces y sont coupees pres du centre, donc kappa =
// (1/2)(alpha_xm+alpha_xp)(1/2)(alpha_ym+alpha_yp) tres petit. Sert au test (d) du CLAMP small-cell :
// une dalle d'epaisseur h << dy force kappa < kappa_min, ce qu'un disque lisse n'atteint pas (chaque
// cellule active du disque garde au moins une demi-face pleine par axe). FONCTEUR NOMME, device-safe.
struct SlabLevelSet {
  Real yc, h;
  ADC_HD Real operator()(Real, Real y) const {
    const Real ay = (y - yc) < Real(0) ? -(y - yc) : (y - yc);
    return ay - h;
  }
};

// --- (a) Champs de la solution manufacturee : densite lisse u(x, y), divergence analytique de F = v u.
// u(x, y) = 1 + 0.5 sin(2pi x) cos(2pi y) (lisse, periodique sur la boite). div F = v . grad u car v
// constant : d_x(vx u) + d_y(vy u) = vx d_x u + vy d_y u.
static double mms_u(double x, double y) {
  return 1.0 + 0.5 * std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y);
}
static double mms_div_ref(double x, double y) {
  const double dux = 0.5 * 2.0 * M_PI * std::cos(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y);
  const double duy = -0.5 * 2.0 * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
  return kVx * dux + kVy * duy;
}

// Erreur L1 (ponderee aire) du residu EB vs divergence analytique, mesuree sur les cellules PROFONDES
// (centre a distance >= margin du bord du disque) : isole l'ordre du schema VOLUMIQUE interieur de la
// couche cut-cell (ordre reduit au bord par le clamp + reconstruction non centree). Renvoie l'erreur L1.
template <class Limiter>
static double mms_error(int n, double margin) {
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, kL, 0.0, kL};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(kL, kRdisc);

  const int ng = Limiter::n_ghost;
  MultiFab U(ba, dm, 1, ng), aux(ba, dm, kAuxBaseComps, ng), R(ba, dm, 1, 0);
  aux.set_val(0.0);
  U.set_val(0.0);
  // u EXACT partout (valides + ghosts) : le stencil voit des donnees exactes -> erreur = troncature.
  {
    Array4 a = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        a(i, j, 0) = mms_u(geom.x_cell(i), geom.y_cell(j));
  }

  const Advect model{kVx, kVy};
  assemble_rhs_eb<Limiter, RusanovFlux>(model, U, aux, detail::disc_level_set(disc), geom, R);

  sync_host();
  const ConstArray4 r = R.fab(0).const_array();
  const double dxa = geom.dx() * geom.dy();
  double l1 = 0.0, area = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      const double ls = double(disc.level_set(Real(x), Real(y)));
      if (ls > -margin)
        continue;  // cellule au bord / dehors : exclue de la mesure d'ordre interieur
      const double div_discrete = -double(r(i, j, 0));  // R = -div F (source nulle) -> div = -R
      const double e = std::fabs(div_discrete - mms_div_ref(x, y));
      l1 += e * dxa;
      area += dxa;
    }
  return area > 0.0 ? l1 / area : 0.0;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  std::printf("=== T5-PR2 : transport FV embedded-boundary conservatif sur disque ===\n");
  std::printf("Boite [0, %.1f]^2, disque centre rayon %.2f, v=(%.2f, %.2f)\n", kL, kRdisc, kVx,
              kVy);

  // ----------------------------------------------------------------------
  // (a) MMS sur le disque : ordre 2 interieur (rapport d'erreur ~4 sous raffinement x2).
  // ----------------------------------------------------------------------
  std::printf("\n--- (a) MMS divergence EB (ordre 2 a l'interieur du disque) ---\n");
  {
    // margin = ~2 cellules de la resolution la plus grossiere, en coordonnees physiques, pour exclure la
    // couche cut-cell des DEUX grilles a la meme distance physique (ordre mesure sur le coeur lisse).
    const double margin = 3.0 * (kL / 64.0);
    const double e0 = mms_error<Minmod>(64, margin);
    const double e1 = mms_error<Minmod>(128, margin);
    const double e2 = mms_error<Minmod>(256, margin);
    const double p1 = std::log2(e0 / e1);
    const double p2 = std::log2(e1 / e2);
    std::printf("  minmod L1 : n=64 %.4e  n=128 %.4e  n=256 %.4e\n", e0, e1, e2);
    std::printf(
        "  ordre observe : %.2f (64->128), %.2f (128->256)  (rapport e0/e1=%.2f, e1/e2=%.2f)\n", p1,
        p2, e0 / e1, e1 / e2);
    chk(e0 > 0.0 && e1 > 0.0 && e2 > 0.0,
        "(a) erreurs MMS strictement positives (mesure non vide)");
    // Ordre 2 : rapport d'erreur ~4 (2^2). On exige p in [1.7, 2.3] sur les DEUX raffinements (la borne
    // haute exclut un ordre artificiellement gonfle ; la basse, un schema sous-convergent / une metrique
    // EB erronee qui donnerait un ordre 0 ou 1).
    auto order2_ok = [](double p) { return p >= 1.7 && p <= 2.3; };
    chk(order2_ok(p1), "(a) ordre ~2 (64->128) : transport EB du 2e ordre a l'interieur du disque");
    chk(order2_ok(p2), "(a) ordre ~2 (128->256) : convergence soutenue (rapport d'erreur ~4)");
  }

  // ----------------------------------------------------------------------
  // (b) CONSERVATION de masse a la machine sur le disque (aucun flux ne franchit le mur immerge).
  // ----------------------------------------------------------------------
  std::printf(
      "\n--- (b) Conservation de masse EB (Sum n kappa_eff dx dy, mur no-penetration) ---\n");
  {
    const int n = 96;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(kL, kRdisc);
    BCRec bc;  // periodique : le disque EB ferme la frontiere, la BC physique de boite n'influe pas

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    // Bosse lisse couvrant le disque. Aux nul (flux ignore aux).
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          a(i, j, 0) =
              1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
        }
    }

    const Advect model{kVx, kVy};

    // Masse discrete COHERENTE avec le schema : Sum_active n_ij kappa_eff_ij dx dy. kappa_eff au
    // denominateur du residu se simplifie avec celui de la masse -> telescopage exact des flux de face
    // (bords fermes) -> invariant EXACT a la machine. C'est la masse que l'operateur EB conserve.
    auto eb_mass = [&](const MultiFab& F) {
      device_fence();
      const ConstArray4 f = F.fab(0).const_array();
      const double dx = geom.dx(), dy = geom.dy();
      double s = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          if (double(disc.level_set(Real(x), Real(y))) >= 0.0)
            continue;  // inactive : hors masse
          detail::CutFraction cf = detail::cut_fraction(detail::disc_level_set(disc), Real(x),
                                                        Real(y), Real(dx), Real(dy));
          const double kappa_eff = std::max(double(cf.kappa), double(detail::kEbKappaMin));
          s += double(f(i, j, 0)) * kappa_eff * dx * dy;
        }
      return s;
    };

    const double m0 = eb_mass(U);
    chk(m0 > 0.0, "(b) masse EB initiale strictement positive (bosse couvre le disque)");

    // Compte des cellules COUPEES (kappa < 1) : le test n'a de sens EB que s'il y a de vraies coupes.
    int n_cut = 0, n_active = 0;
    {
      const double dx = geom.dx(), dy = geom.dy();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          if (double(disc.level_set(Real(x), Real(y))) >= 0.0)
            continue;
          ++n_active;
          detail::CutFraction cf = detail::cut_fraction(detail::disc_level_set(disc), Real(x),
                                                        Real(y), Real(dx), Real(dy));
          if (double(cf.kappa) < 1.0 - 1e-9)
            ++n_cut;
        }
    }
    std::printf("  cellules actives=%d dont coupees (kappa<1)=%d\n", n_active, n_cut);
    chk(n_active > 0 && n_cut > 0,
        "(b) le disque produit de vraies cellules coupees (test EB non vide)");

    // Avance EXPLICITE Euler avant sur le residu EB : U^{n+1} = U^n + dt R_eb(U^n).
    const double v = std::hypot(kVx, kVy);
    const double dt =
        0.2 * geom.dx() / v;  // CFL transport (pas FIXE : le clamp garantit la stabilite)
    bool any_nan = false;
    for (int s = 0; s < 60; ++s) {
      fill_ghosts(U, geom.domain, bc);
      MultiFab R(ba, dm, 1, 0);
      assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, detail::disc_level_set(disc), geom, R);
      {  // garde-fou small-cell : le residu reste FINI (le clamp 1/kappa_min borne l'amplification)
        device_fence();
        const ConstArray4 rr = R.fab(0).const_array();
        for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
          for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
            if (!std::isfinite(double(rr(i, j, 0))))
              any_nan = true;
      }
      saxpy(U, Real(dt), R);
    }
    chk(!any_nan,
        "(b) residu FINI sur toutes les cellules (clamp small-cell : pas de NaN sur la couche "
        "r0/r1)");

    const double m1 = eb_mass(U);
    const double rel_drift = std::fabs(m1 - m0) / std::fabs(m0);
    std::printf("  masse EB : m0=%.15e  m1=%.15e  drift relatif=%.3e\n", m0, m1, rel_drift);
    // Conservation a la machine : la masse EB coherente avec le schema est conservee (telescopage exact
    // des flux internes, bords fermes). Borne juste au-dessus du bruit machine (accumulation 60 pas).
    chk(rel_drift < 1e-12,
        "(b) masse EB conservee a la machine (aucun flux ne franchit le mur immerge ; drift < "
        "1e-12)");
    // Temoin que la dynamique a TOURNE (conservation non triviale) : l'etat a bouge.
    {
      device_fence();
      const ConstArray4 u = U.fab(0).const_array();
      double max_dev = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          if (double(disc.level_set(Real(x), Real(y))) >= 0.0)
            continue;
          max_dev = std::max(max_dev, std::fabs(double(u(i, j, 0)) - 1.0));
        }
      chk(max_dev > 1e-3,
          "(b) le transport EB a effectivement avance l'etat (conservation non triviale)");
    }
  }

  // ----------------------------------------------------------------------
  // (c) BIT-IDENTITE : un disque ENGLOBANT (aucune coupe : alpha_f = 1, kappa = 1 partout) rend
  //     assemble_rhs_eb STRICTEMENT egal a assemble_rhs (diff bit a bit = 0).
  // ----------------------------------------------------------------------
  std::printf("\n--- (c) Bit-identite : sans coupe, EB == operateur cartesien historique ---\n");
  {
    const int n = 48;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;
    // Disque ENGLOBANT : rayon largement > diagonale de la boite -> level set < 0 sur TOUTE la grille (y
    // compris les ghosts), et tous les voisins actifs a distance >> R du bord -> cut_distance rend h
    // (alpha_f = 1) et kappa = 1 partout. L'operateur EB doit alors reproduire le cartesien BIT a BIT.
    const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(kL, 1000.0);

    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = geom.x_cell(i), y = geom.y_cell(j);
          a(i, j, 0) =
              1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
        }
    }

    const Advect model{kVx, kVy};
    fill_ghosts(U, geom.domain, bc);  // memes ghosts pour les deux chemins
    MultiFab R_ref(ba, dm, 1, 0), R_eb(ba, dm, 1, 0);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R_ref);
    assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, detail::disc_level_set(disc), geom, R_eb);

    sync_host();
    double max_abs_diff = 0.0;
    const ConstArray4 rr = R_ref.fab(0).const_array();
    const ConstArray4 re = R_eb.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        max_abs_diff = std::max(max_abs_diff, std::fabs(double(rr(i, j, 0)) - double(re(i, j, 0))));
    std::printf("  max|R_eb - R_cartesien| = %.3e (attendu 0)\n", max_abs_diff);
    // Egalite BIT A BIT : sans coupe (alpha = 1, kappa = 1), l'EB emprunte le MEME flux/reconstruction,
    // la division par kappa = 1 et la ponderation par alpha = 1 sont l'identite -> diff EXACTEMENT 0.
    chk(max_abs_diff == 0.0,
        "(c) sans coupe : residu EB BIT-IDENTIQUE au residu cartesien (alpha=1, kappa=1 -> "
        "identite)");
  }

  // ----------------------------------------------------------------------
  // (d) CLAMP SMALL-CELL load-bearing : sur une cellule a kappa << kappa_min (dalle mince), le residu EB
  //     est BORNE par le clamp (amplification 1/kappa_eff = 1/kappa_min), PAS par le 1/kappa brut qui
  //     deborderait le pas fixe. On compare le residu CLAMPE (kappa_min = kEbKappaMin) au residu NON
  //     clampe (kappa_min = kappa_brut, donc clamp inactif) sur la MEME cellule : leur rapport doit etre
  //     EXACTEMENT kappa_brut / kappa_min (le clamp n'agit que sur le denominateur). C'est la preuve que
  //     le clamp est ACTIF et borne reellement l'amplification (assertion non vide : sans clamp le
  //     residu serait ~ (kappa_min/kappa_brut) fois plus grand).
  // ----------------------------------------------------------------------
  std::printf(
      "\n--- (d) Clamp small-cell : residu borne sur une cellule a kappa << kappa_min ---\n");
  {
    const int n = 16;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, kL, 0.0, kL};  // dx = dy = 1/16
    const BoxArray ba(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    BCRec bc;

    // Dalle mince centree en yc = milieu d'une rangee de cellules : la rangee centrale est active (centre
    // dans la dalle), ses deux voisins y dehors -> les deux demi-faces y sont coupees pres du centre, donc
    // kappa = (1)(h/dy) est minuscule. On EXIGE kappa_brut < kappa_min ci-dessous, sinon le test (clamp
    // actif) ne prouverait rien.
    const double dy = geom.dy();
    const int jc = n / 2;
    const double yc = geom.y_cell(jc);
    // h = 0.008 dy : la coupe traverse les faces y a alpha_y = h/dy = 0.008 -> kappa = (1)(0.008) = 8e-3,
    // sous kappa_min = 1e-2 (le clamp DOIT agir). Au-dessus du plancher cut_distance (theta >= 1e-3).
    const double h = 0.008 * dy;
    const SlabLevelSet ls{Real(yc), Real(h)};

    // kappa brut de la cellule centrale (le clamp NE doit PAS l'avoir vu).
    detail::CutFraction cf =
        detail::cut_fraction(ls, Real(geom.x_cell(n / 2)), Real(yc), Real(geom.dx()), Real(dy));
    const double kappa_raw = double(cf.kappa);
    std::printf("  cellule centrale : kappa_brut = %.6e (kappa_min = %.1e)\n", kappa_raw,
                double(detail::kEbKappaMin));
    chk(kappa_raw < double(detail::kEbKappaMin),
        "(d) la dalle mince produit bien kappa < kappa_min (le clamp DOIT agir : test non vide)");

    // Etat variant en X (les faces y de la rangee coupee sont FERMEES : seul le flux x compte ; il faut
    // donc une variation en x pour que div_x != 0 -> residu non nul a amplifier par 1/kappa).
    MultiFab U(ba, dm, 1, 2), aux(ba, dm, kAuxBaseComps, 2);
    aux.set_val(0.0);
    {
      Array4 a = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          a(i, j, 0) = 1.0 + 0.3 * std::sin(2.0 * M_PI * geom.x_cell(i));
    }
    const Advect model{kVx, kVy};
    fill_ghosts(U, geom.domain, bc);

    // Residu CLAMPE (defaut) vs NON clampe (kappa_min = kappa_raw -> clamp inactif sur cette cellule).
    MultiFab R_clamp(ba, dm, 1, 0), R_raw(ba, dm, 1, 0);
    assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, ls, geom, R_clamp, false,
                                         detail::kEbKappaMin);
    assemble_rhs_eb<Minmod, RusanovFlux>(model, U, aux, ls, geom, R_raw, false, Real(kappa_raw));
    sync_host();

    const ConstArray4 rc = R_clamp.fab(0).const_array();
    const ConstArray4 rw = R_raw.fab(0).const_array();
    // Sur la rangee centrale coupee, le residu clampe doit etre FINI et, terme a terme, valoir
    // (kappa_raw / kappa_min) fois le residu non clampe (meme flux au numerateur, denominateur clampe).
    const double ratio_expected =
        kappa_raw / double(detail::kEbKappaMin);  // < 1 : le clamp ATTENUE
    double max_rel_err = 0.0, max_abs_clamp = 0.0, max_abs_raw = 0.0;
    bool clamp_finite = true;
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double vc = double(rc(i, jc, 0)), vw = double(rw(i, jc, 0));
      if (!std::isfinite(vc))
        clamp_finite = false;
      max_abs_clamp = std::max(max_abs_clamp, std::fabs(vc));
      max_abs_raw = std::max(max_abs_raw, std::fabs(vw));
      if (std::fabs(vw) >
          1e-300) {  // la, R = -(1/kappa) div ; le ratio des residus = ratio des 1/kappa
        const double r = vc / vw;
        max_rel_err = std::max(max_rel_err, std::fabs(r - ratio_expected) / ratio_expected);
      }
    }
    std::printf(
        "  rangee coupee : max|R_clamp|=%.4e  max|R_raw|=%.4e  ratio attendu=%.4e  err rel=%.3e\n",
        max_abs_clamp, max_abs_raw, ratio_expected, max_rel_err);
    chk(clamp_finite,
        "(d) residu clampe FINI sur la rangee a kappa minuscule (pas de debordement)");
    chk(max_abs_raw > 0.0,
        "(d) residu non trivial sur la rangee coupee (flux y non nul : test reel)");
    // Le clamp ATTENUE le residu d'un facteur kappa_raw/kappa_min < 1 : R_clamp = ratio * R_raw a la
    // tolerance arithmetique. Prouve que le clamp borne l'amplification 1/kappa a 1/kappa_min (sans lui
    // le residu serait 1/ratio ~ %.0f fois plus grand -> instabilite du pas fixe).
    chk(max_rel_err < 1e-12,
        "(d) R_clamp == (kappa_raw/kappa_min) * R_raw : le clamp borne 1/kappa a 1/kappa_min "
        "(exact)");
    chk(ratio_expected < 1.0,
        "(d) le clamp ATTENUE bien (kappa_raw < kappa_min : amplification reduite, stabilite "
        "assuree)");
  }

  std::printf("\n=== VERDICT : %s ===\n", fails == 0 ? "SUCCESS" : "ECHEC");
  if (fails == 0)
    std::printf("OK test_eb_transport\n");
  return fails == 0 ? 0 : 1;
}
