// Pas deux-fluides isotherme 2D ASYMPTOTIC-PRESERVING entier sur GPU. Le meme
// TwoFluidAP2D<GeometricMG> (integrator/two_fluid_ap.hpp) : transport Rusanov des
// deux especes, continuite + Lorentz implicites, Poisson reformule
// lap(phi) = (ne* - ni*)/(1+beta0) resolu par multigrille. Tout passe par
// for_each_cell (backend Kokkos -> Cuda), fill_boundary et l'arith MultiFab device ;
// l'elliptique GeometricMG est entierement on-device (lisseur GS rb + V-cycle).
//
// Validation : le MEME source se compile en CPU (sans ADC_HAS_KOKKOS) et en GPU
// (avec). Regime RAIDE omega_pe=1e3, dt*omega_pe=5 : le schema AP doit rester borne
// et quasi-neutre (la ou l'explicite exploserait). On imprime des checksums (masse
// par espece, max|dne|, max|charge|, sum(ne^2)) : les valeurs GPU doivent coincider
// avec celles du CPU, prouvant que tout le pas AP donne le MEME resultat sur les deux
// backends.

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/two_fluid_ap.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <cmath>
#include <cstdio>

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
    TwoFluidAP2D<GeometricMG> d(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);  // raide
    const double dt = 5.0 / 1e3;  // dt*omega_pe = 5 : explicite instable
    d.init(1e-3);
    const double m0e = sum(d.e, 0), m0i = sum(d.ion, 0);
    const int nsteps = 300;
    for (int s = 0; s < nsteps; ++s) d.step(dt, true);

#ifdef ADC_HAS_KOKKOS
    Kokkos::fence();  // barriere avant lecture hote (memoire unifiee)
#endif
    const double me = sum(d.e, 0), mi = sum(d.ion, 0);
    double dev = 0, chg = 0, sumsq = 0;
    const ConstArray4 fe = d.e.fab(0).const_array();
    const ConstArray4 fi = d.ion.fab(0).const_array();
    for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
      for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i) {
        const double ne = fe(i, j, 0);
        dev = std::fmax(dev, std::fabs(ne - 1.0));
        chg = std::fmax(chg, std::fabs(fi(i, j, 0) - ne));
        sumsq += ne * ne;
      }

    std::printf(
        "exec=%s  N=64  pas AP=%d  (dt*omega_pe=5, raide)\n"
        "  masse_e=%.10e (d=%.3e)  masse_i=%.10e (d=%.3e)\n"
        "  max|dne|=%.10e  max|charge|=%.10e  sum(ne^2)=%.10e\n",
        exec, nsteps, me, me - m0e, mi, mi - m0i, dev, chg, sumsq);

    const bool finite = std::isfinite(me) && std::isfinite(sumsq);
    const bool bounded = dev < 0.1;             // AP : reste O(eps), pas d'explosion
    const bool neutral = chg < 0.1;             // quasi-neutralite capturee
    const bool conserv = std::fabs(me - m0e) < 1e-7 && std::fabs(mi - m0i) < 1e-7;
    if (finite && bounded && neutral && conserv)
      std::printf("OK two_fluid_ap_kokkos (checksums ci-dessus a comparer CPU vs GPU)\n");
    else {
      std::printf("FAIL two_fluid_ap_kokkos (finite=%d bounded=%d neutral=%d conserv=%d)\n",
                  finite, bounded, neutral, conserv);
      rc = 1;
    }
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  return rc;
}
