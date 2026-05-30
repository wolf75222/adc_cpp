// Reflux multi-patch 3-NIVEAUX DISTRIBUE via mpirun -np N (chemin recursif
// subcycle_level_mp / amr_step_multilevel_multipatch).
//
// Le niveau 0 (grossier mono-box) est REPLIQUE : chaque rang en detient une copie
// (DistributionMapping par-rang) initialisee de facon deterministe, donc identique. Les
// niveaux 1 (MULTI-BOX, 8 patchs) et 2 (1 patch fin) sont, eux, repartis. Le niveau 1
// joue simultanement le role d'enfant (vis-a-vis du niveau 0) ET de parent (vis-a-vis du
// niveau 2) : c'est la capacite reellement nouvelle par rapport au 2-niveaux. On decoupe
// le niveau 1 en 8 patchs round-robin pour que le PARENT du patch fin tombe sur un rang
// DIFFERENT du patch fin (sinon le chemin cross-rank reste masque).
//
// On execute le MEME pas dans trois repartitions des niveaux 1/2, toutes par tous les
// rangs (collectives appariees) :
//   DIST : patchs round-robin sur les rangs.
//   REF  : patchs tous sur le rang 0 (equivalent np=1 ; les autres rangs a vide).
//   SFC  : patchs repartis par courbe de Morton (make_sfc_distribution, load_balance reel).
// Le niveau 0 etant replique, chaque rang detient les resultats grossiers : comparaison bit
// a bit. DIST == SFC == REF == np=1 prouve l'invariance a la distribution, y compris sous un
// equilibrage de charge par localite (le maillon "load_balance SFC" de la cible distribuee,
// branche sur l'AMR distribue et non plus seulement teste comme algorithme en serie).

#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>  // make_sfc_distribution, load_imbalance

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  const double dxc = 1.0 / nc, dyc = 1.0 / nc;
  BoxArray bac(std::vector<Box2D>{dom});
  DistributionMapping dmc(std::vector<int>(1, me));  // grossier REPLIQUE : box 0 sur chaque rang

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  auto gxv = [&](double dx, int i) { return 0.2 * std::sin(2 * kPi * (i + 0.5) * dx); };
  auto blob = [](double x, double y) {
    return 1.0 + 0.5 * std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02);
  };

  // --- geometrie 3 niveaux ---
  // niveau 1 : region grossiere [8,23]^2 raffinee -> coords niveau 1 [16,47]^2, decoupee
  // en 8 patchs (4 colonnes de largeur 8 x 2 rangees de hauteur 16).
  std::vector<Box2D> l1;
  for (int cy = 0; cy < 2; ++cy)
    for (int cx = 0; cx < 4; ++cx) {
      const int x0 = 16 + 8 * cx, y0 = 16 + 16 * cy;
      l1.push_back(Box2D{{x0, y0}, {x0 + 7, y0 + 15}});
    }
  // niveau 2 : 1 patch fin interieur au patch niveau 1 de colonne [24,31] rangee [16,31].
  // coords niveau 1 [26,29]x[20,27] (strictement interieur) -> coords niveau 2 [52,59]x[40,55].
  std::vector<Box2D> l2 = {Box2D{{52, 40}, {59, 55}}};

  auto fill_aux = [&](MultiFab& a, double dx) {
    for (int li = 0; li < a.local_size(); ++li) {
      Array4 ar = a.fab(li).array();
      const Box2D g = a.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          ar(i, j, 0) = 0; ar(i, j, 1) = gxv(dx, i); ar(i, j, 2) = -1.0;
        }
    }
  };
  auto init = [&](MultiFab& U, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          u(i, j, 0) = blob((i + 0.5) * dx, (j + 0.5) * dx);
    }
  };
  // masse grossiere locale : le grossier est replique (identique sur chaque rang), donc une
  // somme LOCALE donne la vraie masse, sans all_reduce (qui multiplierait par np).
  auto local_mass = [&](const MultiFab& U) {
    const ConstArray4 c = U.fab(0).const_array();
    double M = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += c(i, j, 0) * dxc * dyc;
    return M;
  };

  // aux partage entre les deux runs (memes box_array, memes dmap fournis par run).
  auto run = [&](const DistributionMapping& dm1, const DistributionMapping& dm2) {
    BoxArray ba1(l1), ba2(l2);
    MultiFab U0(bac, dmc, 1, 1), U1(ba1, dm1, 1, 1), U2(ba2, dm2, 1, 1);
    MultiFab a0(bac, dmc, 3, 1), a1(ba1, dm1, 3, 1), a2(ba2, dm2, 3, 1);
    init(U0, dxc); init(U1, dxc / 2); init(U2, dxc / 4);
    fill_aux(a0, dxc); fill_aux(a1, dxc / 2); fill_aux(a2, dxc / 4);
    mf_average_down_mb(U2, U1);  // sync init niveau 1 <- niveau 2
    mf_average_down_mb(U1, U0);  // niveau 0 <- niveau 1 (avant la masse)
    std::vector<AmrLevelMP> LP(3);
    LP[0] = {std::move(U0), &a0, dxc, dyc};
    LP[1] = {std::move(U1), &a1, dxc / 2, dyc / 2};
    LP[2] = {std::move(U2), &a2, dxc / 4, dyc / 4};
    const double M0 = local_mass(LP[0].U);
    const double dt = 0.4 * dxc;
    for (int s = 0; s < 30; ++s)
      amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(model, LP, dom, dt);
    return std::make_pair(std::move(LP[0].U), std::fabs(local_mass(LP[0].U) - M0));
  };

  // DIST : niveaux 1/2 round-robin. REF : tous sur rang 0.
  auto [UcDist, drift] =
      run(DistributionMapping((int)l1.size(), n_ranks()),
          DistributionMapping((int)l2.size(), n_ranks()));
  auto [UcRef, driftRef] =
      run(DistributionMapping(std::vector<int>(l1.size(), 0)),
          DistributionMapping(std::vector<int>(l2.size(), 0)));

  double maxdiff = 0;
  const ConstArray4 ud = UcDist.fab(0).const_array(), ur = UcRef.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(ud(i, j, 0) - ur(i, j, 0)));
  maxdiff = all_reduce_max(maxdiff);
  drift = all_reduce_max(drift);

  if (me == 0)
    std::printf("reflux multipatch 3-niveaux distribue (np=%d) : max|Uc_dist - Uc_ref| = "
                "%.3e, derive masse = %.3e\n", n_ranks(), maxdiff, drift);

  if (maxdiff > 1e-12) { if (me == 0) std::printf("FAIL distribution_invariante\n"); ++fails; }
  if (drift > 1e-10) { if (me == 0) std::printf("FAIL masse_conservee\n"); ++fails; }

  // SFC : niveaux 1/2 repartis par courbe de Morton (load_balance reel, pas round-robin).
  // Le resultat grossier doit rester bit-identique a REF : un equilibrage par localite ne
  // change pas la reponse, seulement quel rang detient quel patch. C'est le maillon
  // "load_balance SFC" de la cible distribuee, branche sur l'AMR distribue (plus seulement
  // teste en serie comme algorithme).
  auto [UcSfc, driftSfc] = run(make_sfc_distribution(BoxArray(l1), n_ranks()),
                               make_sfc_distribution(BoxArray(l2), n_ranks()));
  (void)driftSfc;
  double maxdiffSfc = 0;
  {
    const ConstArray4 us = UcSfc.fab(0).const_array(), ur2 = UcRef.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiffSfc = std::fmax(maxdiffSfc, std::fabs(us(i, j, 0) - ur2(i, j, 0)));
  }
  maxdiffSfc = all_reduce_max(maxdiffSfc);
  const double imb = load_imbalance(BoxArray(l1), make_sfc_distribution(BoxArray(l1), n_ranks()),
                                    n_ranks());
  if (me == 0)
    std::printf("  SFC (Morton) niveau 1/2 : max|Uc_sfc - Uc_ref| = %.3e, desequilibre = %.3f\n",
                maxdiffSfc, imb);
  if (maxdiffSfc > 1e-12) { if (me == 0) std::printf("FAIL sfc_invariante\n"); ++fails; }
  if (imb > 1.6) { if (me == 0) std::printf("FAIL sfc_desequilibre\n"); ++fails; }

  fails = all_reduce_sum(fails);
  if (fails == 0 && me == 0) std::printf("OK test_mpi_amr_multipatch3\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
