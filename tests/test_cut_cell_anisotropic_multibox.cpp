// Bord embedded cut-cell (Shortley-Weller) + operateur elliptique ANISOTROPE
// div(diag(eps_x, eps_y) grad phi) = f sur un domaine DECOUPE EN PLUSIEURS BOITES.
//
// Le cut-cell mono-box est couvert par test_cut_cell ; le cut-cell anisotrope mono-box par
// test_cut_cell_anisotropic ; le Poisson multi-box reparti (sans cut-cell) par test_mpi_mbox_parity.
// Il manquait la combinaison cut-cell + anisotrope sur un domaine MULTI-BOX : ce test la valide.
// Les champs cut-cell (masque, coefficients Shortley-Weller) et anisotropes (eps_x, eps_y) sont des
// MultiFab construits PAR BOITE depuis les fonctions analytiques ; le lisseur et le residu echangent
// les halos de phi entre boites via fill_ghosts. On verifie qu'aucune hypothese mono-box ne se cache
// dans l'operateur : le decoupage en boites ne change ni l'ordre ni la solution.
//
// MMS : phi = R^2 - r^2 sur le disque r < R, phi = 0 sur le cercle r = R. Pour eps_x, eps_y
// CONSTANTS, div(diag(eps_x, eps_y) grad phi) = eps_x phi_xx + eps_y phi_yy = -2 eps_x - 2 eps_y,
// constant ; phi = R^2 - r^2 reste la solution EXACTE avec f = -2(eps_x + eps_y). On verifie
// (eps_x = 1.5, eps_y = 0.7) :
//   (A) convergence cut-cell ordre ~2 en L2 sur grille MULTI-BOX (raffinement 128/256/512) ;
//   (B) INVARIANCE AU DECOUPAGE : a n fixe, l'erreur L2 sur le disque est BIT-IDENTIQUE entre une
//       grille MONO-box et une grille MULTI-box (le solveur ne depend pas du pavage en boites).
//
// La norme L2 (et non L_inf) porte l'ordre 2 propre de Shortley-Weller : la convergence
// cell-centree est polluee par la pire cellule coupee (small-cell), ce qui rend L_inf erratique ;
// la supraconvergence est en L2 (cf. test_cut_cell).
//
// PORTEE : MONO-RANG. La REPARTITION MPI du cut-cell multi-box n'est PAS couverte : sous le build
// MPI, l'enchainement de plusieurs solves cut-cell multi-box diverge de facon INTERMITTENTE (erreur
// ponctuelle ~1e-5, voire blow-up), y compris a np=1 (donc sans communication inter-rang). Le build
// NON-MPI, lui, est strictement deterministe (cf. plus bas). Le cut-cell reparti MPI reste donc un
// chemin a durcir avant d'etre teste ; ce test verrouille le sous-ensemble SOLIDE (multi-box mono-rang).

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace pops;
static constexpr double kCx = 0.5, kCy = 0.5, kR = 0.4;
static constexpr double kEx = 1.5, kEy = 0.7;  // permittivite anisotrope constante

// Construit le solveur cut-cell + anisotrope sur une BoxArray donnee (mono- ou multi-box).
// Le constructeur evalue masque / coefficients Shortley-Weller / eps_x / eps_y PAR BOITE depuis
// les fonctions analytiques : il n'existe pas de chemin mono-box particulier dans l'operateur.
static GeometricMG make_mg(const Geometry& geom, const BoxArray& ba) {
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  std::function<Real(Real, Real)> ls = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) - kR;
  };
  std::function<bool(Real, Real)> active = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) < kR;
  };
  // (geom, ba, bc, active, replicated, min_coarse, nu1, nu2, nbottom, cut_cell, levelset)
  GeometricMG mg(geom, ba, bc, active, false, 2, 2, 2, 50, /*cut_cell=*/true, ls);
  mg.set_epsilon_anisotropic([](Real, Real) { return Real(kEx); },
                             [](Real, Real) { return Real(kEy); });
  return mg;
}

// Resout le MMS sur n x n decoupe en boites <= max_grid_size, renvoie l'erreur L2 de phi vs
// R^2 - r^2 sur les cellules INTERIEURES au disque (somme sur toutes les boites du niveau fin).
static double solve_l2(int nc, int max_grid_size) {
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  // max_grid_size >= nc -> une seule boite ; < nc -> plusieurs boites pavant le domaine.
  BoxArray ba = BoxArray::from_domain(dom, max_grid_size);

  GeometricMG mg = make_mg(geom, ba);
  mg.rhs().set_val(Real(-2.0 * (kEx + kEy)));  // f = div(diag(ex,ey) grad(R^2-r^2)) = -2(ex+ey)
  mg.phi().set_val(0.0);
  mg.solve_robust(1e-10, 300);

  const double dx = 1.0 / nc;
  double s = 0;
  long cnt = 0;
  for (int li = 0; li < mg.phi().local_size(); ++li) {
    const ConstArray4 p = mg.phi().fab(li).const_array();
    const Box2D b = mg.phi().box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
        const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
        if (r2 < kR * kR) {  // interieur du disque
          const double e = std::fabs(p(i, j) - (kR * kR - r2));
          s += e * e;
          ++cnt;
        }
      }
  }
  return std::sqrt(s / cnt);
}

static double order(double e1, double e2, int n1, int n2) {
  return std::log(e1 / e2) / std::log(double(n2) / n1);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // max_grid_size = 64 sur n = 128/256/512 -> 2x2, 4x4, 8x8 boites (4, 16, 64 boites) pavant le
  // domaine. Chaque boite 64x64 se coarsen proprement dans la hierarchie MG (puissances de 2).
  const int mgs = 64;

  // (A) convergence cut-cell + anisotrope ordre ~2 en L2, sur grille MULTI-BOX.
  const double m128 = solve_l2(128, mgs);
  const double m256 = solve_l2(256, mgs);
  const double m512 = solve_l2(512, mgs);
  const double o = order(m128, m512, 128, 512);
  std::printf("cut-cell+aniso (ex=%.1f ey=%.1f) MULTI-BOX L2 : %.3e %.3e %.3e  ordre=%.2f\n", kEx,
              kEy, m128, m256, m512, o);
  chk(o > 1.7, "cutcell_aniso_multibox_ordre2_L2");  // Shortley-Weller : ~2 en L2
  chk(std::isfinite(m512) && m512 > 0, "cutcell_aniso_multibox_fini");

  // (B) invariance au decoupage : MONO-box (une seule boite couvrant le domaine) vs MULTI-box, a
  // n = 256 fixe. Les deux passent par la meme machinerie ; seul le pavage change. L'erreur L2
  // sur le disque doit etre BIT-IDENTIQUE (le solveur ne depend pas du decoupage en boites).
  const int nc = 256;
  const double l2_mono = solve_l2(nc, nc);    // max_grid_size = nc -> 1 boite
  const double l2_multi = solve_l2(nc, mgs);  // plusieurs boites
  const double gap = std::fabs(l2_mono - l2_multi);
  std::printf("invariance decoupage n=%d : mono=%.12e multi=%.12e  ecart=%.3e\n", nc, l2_mono,
              l2_multi, gap);
  chk(gap <= 1e-12 * (l2_mono + 1.0), "cutcell_aniso_invariance_decoupage");  // bit-identique

  if (fails == 0)
    std::printf("OK test_cut_cell_anisotropic_multibox\n");
  return fails == 0 ? 0 : 1;
}
