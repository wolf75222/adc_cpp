// Etape D (capstone) : un PAS COUPLE Euler-Poisson entier sur GPU. Coupler<Diocotron>
// ::advance enchaine, par etage SSPRK2 : f = elliptic_rhs(U) -> multigrille
// lap(phi)=f -> aux=(phi, grad phi) -> assemble_rhs MUSCL -> mise a jour saxpy/lincomb.
// Toutes ces briques passent par for_each_cell (backend Kokkos -> Cuda) et l'arith
// MultiFab device ; il a suffi d'annoter ADC_HD les deux lambdas du coupleur
// (elliptic_rhs et la derivee centree de phi).
//
// Validation : le MEME source se compile en CPU (sans ADC_HAS_KOKKOS) et en GPU
// (avec). On avance un diocotron periodique lisse de K pas et on imprime des
// sommes de controle (masse, somme des carres, max). La masse est conservee
// (transport conservatif periodique, source nulle) ; surtout, les checksums GPU
// doivent coincider avec ceux du CPU, prouvant que tout le pas couple donne le
// MEME resultat sur les deux backends.

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
  const char* exec = Kokkos::DefaultExecutionSpace::name();
#else
  (void)argc;
  (void)argv;
  const char* exec = "serial-cpu";
#endif
  int rc = 0;
  {
    const int N = 128, ng = 2;  // MUSCL Minmod : 2 ghosts
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);

    Diocotron model;  // B0 = n_i0 = alpha = 1
    BCRec bcU, bcPhi;  // periodique partout (defaut)

    Coupler<Diocotron> cpl(model, geom, ba, bcU, bcPhi);

    // densite initiale lisse, moyenne = n_i0 = 1 (RHS de Poisson a moyenne nulle)
    MultiFab U(ba, dm, 1, ng);
    {
      Array4 u = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      auto wrap = [&](int x) { return (x % N + N) % N; };
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ii = wrap(i), jj = wrap(j);
          const double x = (ii + 0.5) / N, y = (jj + 0.5) / N;
          u(i, j, 0) = 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
        }
    }

    const double m0 = sum(U);
    const double dt = 0.02;
    const int nsteps = 20;
    for (int s = 0; s < nsteps; ++s) cpl.advance<Minmod>(U, dt);

    // sommes de controle (lecture hote -> barriere)
#ifdef ADC_HAS_KOKKOS
    Kokkos::fence();
#endif
    const double mass = sum(U);
    double sumsq = 0, maxabs = 0;
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double v = u(i, j, 0);
        sumsq += v * v;
        maxabs = std::fmax(maxabs, std::fabs(v));
      }

    std::printf(
        "exec=%s  N=%d  pas couples=%d\n"
        "  masse0=%.10e  masse=%.10e  dmasse=%.3e\n"
        "  sum(U^2)=%.10e  max|U|=%.10e\n",
        exec, N, nsteps, m0, mass, mass - m0, sumsq, maxabs);

    const bool finite = std::isfinite(mass) && std::isfinite(sumsq);
    const bool conserv = std::fabs(mass - m0) < 1e-9;
    if (finite && conserv)
      std::printf("OK coupled_kokkos (checksums ci-dessus a comparer CPU vs GPU)\n");
    else {
      std::printf("FAIL coupled_kokkos (finite=%d conserv=%d)\n", finite, conserv);
      rc = 1;
    }
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  return rc;
}
