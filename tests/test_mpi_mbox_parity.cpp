// Parite MULTI-BOX du chemin compile add_compiled_model (foncteurs nommes). Le residu R = rhs_into(U)
// calcule par les FERMETURES de make_block (la machinerie exacte d'add_compiled_model, instanciee ici
// depuis une UNITE DE TRADUCTION EXTERNE = le chemin device-clean) doit etre INDEPENDANT du decoupage
// du domaine en boites (halos remplis par fill_ghosts). On compare une decomposition 16-boites
// distribuee sur np rangs via SFC a une reference MONO-boite : meme champ initial periodique, memes
// normes GLOBALES du residu. Invariant au nombre de boites ET de rangs : valide les halos (intra-rang
// multi-box ET cross-rang MPI) du chemin make_block / add_compiled_model.
//
// Cette propriete de decoupage est INDEPENDANTE du backend : verte sous Kokkos Serial (CI) sur CPU,
// elle l'est aussi sous Kokkos Cuda (GPU, valide sur ROMEO GH200, cf docs/GPU_RUNTIME_PORT.md). Sous
// Cuda, for_each_cell ne fence pas (async) : on insere une barriere Kokkos::fence() avant la lecture
// HOTE du residu. Lance via ctest a np=1/2/4 ; resultat bit-identique (dmax == 0) a chaque rang count.
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>
#include <adc/physics/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/euler.hpp>   // Euler (transport compressible)
#include <adc/runtime/block_builder.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;
static const double kPi = 3.14159265358979323846;

// Etat conservatif initial : champ primitif periodique (rho>0, p>0), vitesse non nulle -> -div F != 0.
static Euler::State init_state(const Euler& eul, int i, int j, int n) {
  const double X = (i + 0.5) / n, Y = (j + 0.5) / n;
  Euler::Prim P;
  P[0] = 1.0 + 0.2 * std::sin(2 * kPi * X) * std::cos(2 * kPi * Y);
  P[1] = 0.3 * std::sin(2 * kPi * X);
  P[2] = 0.3 * std::cos(2 * kPi * Y);
  P[3] = 1.0;
  return eul.to_conservative(P);
}

// Normes GLOBALES du residu d'un decoupage donne (ba, dm) : meme champ initial, meme schema, meme
// modele. Le residu est calcule par les fermetures de make_block (chemin add_compiled_model).
static void residual_norms(const BoxArray& ba, const DistributionMapping& dm, const Box2D& dom,
                           const Geometry& geom, const Model& model, int n, double& gsum,
                           double& gsumsq, double& gmax) {
  const Euler eul{1.4};
  MultiFab U(ba, dm, Model::n_vars, 2);  // minmod -> 2 fantomes
  MultiFab aux(ba, dm, 3, 1);
  MultiFab R(ba, dm, Model::n_vars, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    Fab2D& Fu = U.fab(li);
    const Box2D b = Fu.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Euler::State s = init_state(eul, i, j, n);
        for (int c = 0; c < Model::n_vars; ++c)
          Fu(i, j, c) = s[c];
      }
    Fab2D& Fa = aux.fab(li);
    const Box2D ag = Fa.box().grow(1);  // aux = 0 partout (valide + fantomes) : source nulle
    for (int c = 0; c < 3; ++c)
      for (int j = ag.lo[1]; j <= ag.hi[1]; ++j)
        for (int i = ag.lo[0]; i <= ag.hi[0]; ++i)
          Fa(i, j, c) = 0.0;
  }
  GridContext ctx{dom, BCRec{}, geom, &aux};  // BCRec{} = tout periodique
  BlockClosures clo =
      make_block(model, "minmod", "rusanov", ctx, /*imex=*/false, /*recon_prim=*/false);
  clo.rhs_into(U, R);  // fill_ghosts(U) [halos multi-box / MPI] + assemble_rhs (foncteurs nommes)
#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();  // barriere avant lecture HOTE du residu device (no-op sous Serial)
#endif
  double s = 0, ss = 0, mx = 0;
  for (int li = 0; li < R.local_size(); ++li) {
    const Fab2D& Fr = R.fab(li);
    const Box2D b = Fr.box();
    for (int c = 0; c < Model::n_vars; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          const double v = Fr(i, j, c);
          s += v;
          ss += v * v;
          const double a = std::fabs(v);
          if (a > mx)
            mx = a;
        }
  }
  gsum = all_reduce_sum(s);
  gsumsq = all_reduce_sum(ss);
  gmax = all_reduce_max(mx);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 64;
  const double L = 1.0;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, L, 0.0, L};
  const Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};

  // (A) decomposition MULTI-BOX distribuee (SFC sur np rangs) : boites <= 16 -> 4x4 = 16 boites.
  BoxArray baK = BoxArray::from_domain(dom, 16);
  DistributionMapping dmK = make_sfc_distribution(baK, np);
  double aSum, aSumsq, aMax;
  residual_norms(baK, dmK, dom, geom, model, n, aSum, aSumsq, aMax);

  // (B) reference MONO-BOX (boite unique sur le rang 0 ; tous les rangs entrent les collectives).
  BoxArray ba1(std::vector<Box2D>{dom});
  DistributionMapping dm1(1, np);
  double bSum, bSumsq, bMax;
  residual_norms(ba1, dm1, dom, geom, model, n, bSum, bSumsq, bMax);

  int fails = 0;
  if (me == 0) {
    const double l2b = std::sqrt(bSumsq);        // norme L2 du residu mono-box
    const double dmax = std::fabs(aMax - bMax);  // max|R| : invariant EXACT
    const double dssq = std::fabs(aSumsq - bSumsq) / (bSumsq + 1e-300);
    const double dsum = std::fabs(aSum - bSum);  // somme ~0 (div periodique) -> absolu
    std::printf(
        "np=%d boxesK=%d | maxK=%.6e max1=%.6e dmax=%.2e | L2=%.6e dssqrel=%.2e | dsum=%.2e\n", np,
        baK.size(), aMax, bMax, dmax, l2b, dssq, dsum);
    if (!(bMax > 1e-6)) {
      std::printf("FAIL residu mono-box trivial\n");
      ++fails;
    }
    if (!(dmax <= 1e-12 * (bMax + 1.0))) {
      std::printf("FAIL max|R| multi != mono\n");
      ++fails;
    }
    if (!(dssq < 1e-10)) {
      std::printf("FAIL L2(R) multi != mono\n");
      ++fails;
    }
    if (!(dsum < 1e-9 * (l2b + 1.0))) {
      std::printf("FAIL sum(R) multi != mono (echelle L2)\n");
      ++fails;
    }
    if (fails == 0)
      std::printf("OK test_mpi_mbox_parity np=%d (multi-box %s == mono-box)\n", np,
                  np > 1 ? "MPI" : "mono-rang");
  }
  comm_finalize();
  return fails ? 1 : 0;
}
