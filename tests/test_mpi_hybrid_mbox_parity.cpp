// Parite MULTI-BOX / MPI du chemin compile add_compiled_model sur un composite HYBRIDE : un
// CompositeModel melant une brique de SOURCE style DSL (apply non-template, lit a.grad_x/grad_y, comme
// l'emet dsl.SourceBrick) avec un transport NATIF (Euler) et une elliptique NATIVE (GravityCoupling).
// C'est le pendant C++/MPI de python/tests/test_dsl_hybrid_amr/_coupling : le composite hybride passe
// par le MEME make_block (foncteurs nommes) qu'add_compiled_model. Le residu R = -div F + S(U, aux)
// doit etre INDEPENDANT du decoupage du domaine en boites ET du nombre de rangs (halos fill_ghosts
// intra-rang multi-box + cross-rang MPI). aux est un champ analytique PERIODIQUE (fonction pure de la
// position -> invariant au decoupage), donc la source contribue ET reste decomposition-invariante.
//
// Pendant strict de test_mpi_mbox_parity (composite 100% natif) : seul le TYPE de brique change. La
// propriete est independante du backend (verte Kokkos Serial CPU, et Cuda GPU). Lance via ctest a
// np=1/2/4 ; resultat bit-identique (dmax == 0) a chaque rang count.
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>
#include <adc/physics/bricks.hpp>  // CompositeModel, GravityCoupling
#include <adc/physics/euler.hpp>   // Euler (transport compressible natif)

#include <adc/runtime/builders/block_builder.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
static const double kPi = 3.14159265358979323846;

// Brique de SOURCE style DSL (telle que l'emet dsl.SourceBrick : apply NON template, 4 variables, lit
// a.grad_x / a.grad_y). Replique la gravite rho g, g = -grad phi. Composee ici avec un transport et une
// elliptique NATIFS -> un CompositeModel HYBRIDE, comme le genere adc.CompositeModel cote Python.
namespace adc_generated {
struct GenGravitySrc {
  ADC_HD adc::StateVec<4> apply(const adc::StateVec<4>& U, const adc::Aux& a) const {
    const adc::Real rho = U[0], rhou = U[1], rhov = U[2];
    const adc::Real gx = a.grad_x, gy = a.grad_y;
    adc::StateVec<4> S{};
    S[0] = adc::Real(0);
    S[1] = -rho * gx;
    S[2] = -rho * gy;
    S[3] = -(rhou * gx + rhov * gy);
    return S;
  }
};
}  // namespace adc_generated

using Model = CompositeModel<Euler, adc_generated::GenGravitySrc, GravityCoupling>;

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

// Normes GLOBALES du residu d'un decoupage donne (ba, dm). aux = grad analytique periodique (fonction
// pure de la position), rempli valid+fantomes -> la SOURCE hybride contribue et reste invariante au
// decoupage. Residu calcule par les fermetures de make_block (chemin add_compiled_model).
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
    const Box2D ag = Fa.box().grow(1);  // grad analytique periodique sur valid + fantomes
    for (int j = ag.lo[1]; j <= ag.hi[1]; ++j)
      for (int i = ag.lo[0]; i <= ag.hi[0]; ++i) {
        const double X = (i + 0.5) / n, Y = (j + 0.5) / n;
        Fa(i, j, 0) = 0.0;                          // phi (inutilise par la source)
        Fa(i, j, 1) = 0.4 * std::sin(2 * kPi * X);  // grad_x
        Fa(i, j, 2) = 0.4 * std::cos(2 * kPi * Y);  // grad_y
      }
  }
  GridContext ctx{dom, BCRec{}, geom, &aux};  // BCRec{} = tout periodique
  BlockClosures clo =
      make_block(model, "minmod", "rusanov", ctx, /*imex=*/false, /*recon_prim=*/false);
  clo.rhs_into(U, R);  // fill_ghosts(U) [halos multi-box / MPI] + assemble_rhs (-div F + source)
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
  const Model model{Euler{1.4}, adc_generated::GenGravitySrc{}, GravityCoupling{-1.0, 1.0, 1.0}};

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
    const double l2b = std::sqrt(bSumsq);
    const double dmax = std::fabs(aMax - bMax);
    const double dssq = std::fabs(aSumsq - bSumsq) / (bSumsq + 1e-300);
    const double dsum = std::fabs(aSum - bSum);
    std::printf(
        "np=%d boxesK=%d | maxK=%.6e max1=%.6e dmax=%.2e | L2=%.6e dssqrel=%.2e | dsum=%.2e\n", np,
        baK.size(), aMax, bMax, dmax, l2b, dssq, dsum);
    if (!(bMax > 1e-6)) {
      std::printf("FAIL residu mono-box trivial (la source hybride doit contribuer)\n");
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
      std::printf(
          "OK test_mpi_hybrid_mbox_parity np=%d (composite hybride multi-box %s == mono-box)\n", np,
          np > 1 ? "MPI" : "mono-rang");
  }
  comm_finalize();
  return fails ? 1 : 0;
}
