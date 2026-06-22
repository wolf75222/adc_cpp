// Cut-cell (Shortley-Weller) sur domaine MULTI-BOX, ENCHAINEMENT de solves, sous MPI (np=1/2/4).
//
// Verrou de non-regression du bug "cut-cell multi-box MPI" : une hierarchie multigrille construite
// sur un domaine multi-box (BoxArray::from_domain, max_grid_size < n) coarsenait les boites JUSQU'A
// 1 cellule, puis UNE FOIS DE TROP -> BoxArray grossier degenere (plusieurs boites en doublon sur la
// meme cellule). average_down lisait alors un bloc 2x2 par cellule grossiere HORS des bornes du fab
// fin de 1 cellule (lecture memoire non initialisee). En serie le tas est stable (bug masque), mais
// sous le build MPI le tas est remue : l'enchainement de solves cut-cell multi-box DIVERGEAIT de
// facon intermittente (ecart ponctuel ~1e-5 jusqu'au blow-up inf/1e47), MEME a np=1. Cf. le fix dans
// geometric_mg.hpp (stop du coarsening quand une boite ne se coarsen plus proprement).
//
// Le test choisit n=128, max_grid_size=32 -> 16 boites qui, sans le fix, degenerent au niveau le plus
// grossier (domaine 2x2). On enchaine K solves successifs (le tas se salit d'un solve a l'autre) et on
// exige : (1) chaque L2 FINI et proche de l'analytique (un solve diverge donnerait >> 1e-3) ; (2) tous
// les K solves donnent un L2 STRICTEMENT identique (determinisme intra-rang : tout flake intermittent
// le casse) ; (3) INVARIANCE au decoupage : L2 multi-box == L2 mono-box (le mono-box ne degenere jamais
// -> reference correcte). Les echecs sont votes par all_reduce sur tous les rangs. Rejoue np=1/2/4 :
// np=1 couvre le chemin qui flakait deja, np=2/4 le multi-box reparti (halos cut-cell cross-rang).

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace adc;
static constexpr double kCx = 0.5, kCy = 0.5, kR = 0.4;

// Erreur L2 (GLOBALE, reduite sur les rangs) de phi vs R^2 - r^2 sur les cellules du disque, pour un
// solve cut-cell sur ba. GeometricMG repartit lui-meme les boites (round-robin sur n_ranks()). Meme
// MMS que test_cut_cell : lap(phi) = -4, phi = 0 sur le cercle.
static double solve_l2(const Geometry& geom, const BoxArray& ba, int nc) {
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  std::function<Real(Real, Real)> ls = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) - kR;
  };
  std::function<bool(Real, Real)> active = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) < kR;
  };
  // (geom, ba, bc, active, replicated, min_coarse, nu1, nu2, nbottom, cut_cell, levelset)
  GeometricMG mg(geom, ba, bc, active, false, 2, 2, 2, 50, true, ls);
  mg.rhs().set_val(-4.0);
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
        if (r2 < kR * kR) {
          const double e = std::fabs(p(i, j) - (kR * kR - r2));
          s += e * e;
          ++cnt;
        }
      }
  }
  const double gs = all_reduce_sum(s);
  const long gc = static_cast<long>(all_reduce_sum(static_cast<long>(cnt)));
  return std::sqrt(gs / static_cast<double>(gc));
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  const int nc = 128, mgs = 32;  // 16 boites : degeneraient au plus grossier sans le fix
  const Box2D dom = Box2D::from_extents(nc, nc);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};

  // reference MONO-BOX (1 boite couvrant le domaine) : ne degenere jamais (1 boite pave toujours).
  BoxArray ba1(std::vector<Box2D>{dom});
  const double ref_mono = solve_l2(geom, ba1, nc);

  // MULTI-BOX : 16 boites pavant le domaine, reparties par GeometricMG sur np rangs.
  BoxArray baK = BoxArray::from_domain(dom, mgs);

  const int K = 50;
  double l2_first = -1;
  int local_fail = 0;
  for (int it = 0; it < K; ++it) {
    const double l2 = solve_l2(geom, baK, nc);
    if (it == 0)
      l2_first = l2;
    const bool finite = std::isfinite(l2);
    const bool small = finite && l2 < 1e-3;        // un solve diverge donne >> 1e-3
    const bool stable = finite && l2 == l2_first;  // determinisme intra-rang (bit-exact)
    const bool invariant =
        finite &&
        std::fabs(l2 - ref_mono) <= 1e-6 * (ref_mono + 1.0);  // == mono-box (decoupage invariant)
    if (!(small && stable && invariant)) {
      ++local_fail;
      if (local_fail <= 3)
        std::printf(
            "[rang %d/%d] it=%d L2=%.10e (ref_mono=%.10e) fini=%d petit=%d stable=%d "
            "invariant=%d\n",
            me, np, it, l2, ref_mono, finite, small, stable, invariant);
    }
  }

  const long total_fail = static_cast<long>(all_reduce_sum(static_cast<long>(local_fail)));
  if (me == 0) {
    std::printf("np=%d boites=%d K=%d ref_mono=%.10e l2_multi=%.10e : %ld echecs\n", np, baK.size(),
                K, ref_mono, l2_first, total_fail);
    if (total_fail == 0)
      std::printf(
          "OK test_mpi_cutcell_multibox np=%d (cut-cell multi-box deterministe, == mono-box)\n",
          np);
  }
  comm_finalize();
  return total_fail ? 1 : 0;
}
