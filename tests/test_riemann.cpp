// Tube a choc de Sod : valide le flux numerique en POLITIQUE (Rusanov / HLL / HLLC)
// contre la solution de Riemann EXACTE (echantillonnage de Toro). On verifie :
//   - chaque schema converge vers l'exact (erreur L1 bornee), positivite preservee ;
//   - hierarchie de diffusion : HLLC <= HLL <= Rusanov (HLLC capture le contact).
//
// Probleme uniforme en y : 1D selon x, CL transmissives (Foextrap) en x.

#include <adc/integrator/ssprk.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler.hpp>
#include <adc/operator/numerical_flux.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// etats de Sod
static constexpr double rL = 1.0, uLL = 0.0, pL = 1.0;
static constexpr double rR = 0.125, uRR = 0.0, pR = 0.1;
static constexpr double GAM = 1.4, X0 = 0.5;

// densite exacte de Riemann au point x, temps t (echantillonnage de Toro).
static double sod_exact_rho(double x, double t) {
  const double cL = std::sqrt(GAM * pL / rL), cR = std::sqrt(GAM * pR / rR);
  auto fK = [&](double p, double rK, double pK, double cK) {
    if (p > pK) {
      const double A = 2.0 / ((GAM + 1) * rK), B = (GAM - 1) / (GAM + 1) * pK;
      return (p - pK) * std::sqrt(A / (p + B));
    }
    return 2 * cK / (GAM - 1) * (std::pow(p / pK, (GAM - 1) / (2 * GAM)) - 1);
  };
  auto dfK = [&](double p, double rK, double pK, double cK) {
    if (p > pK) {
      const double A = 2.0 / ((GAM + 1) * rK), B = (GAM - 1) / (GAM + 1) * pK;
      return std::sqrt(A / (B + p)) * (1 - (p - pK) / (2 * (B + p)));
    }
    return 1.0 / (rK * cK) * std::pow(p / pK, -(GAM + 1) / (2 * GAM));
  };
  // Newton pour p* (etoile)
  double p = 0.5 * (pL + pR);
  for (int it = 0; it < 100; ++it) {
    const double f = fK(p, rL, pL, cL) + fK(p, rR, pR, cR) + (uRR - uLL);
    const double df = dfK(p, rL, pL, cL) + dfK(p, rR, pR, cR);
    const double pn = p - f / df;
    if (std::fabs(pn - p) < 1e-12 * (p + pn)) { p = pn; break; }
    p = pn > 0 ? pn : 0.5 * p;
  }
  const double pst = p;
  const double ust = 0.5 * (uLL + uRR) + 0.5 * (fK(pst, rR, pR, cR) - fK(pst, rL, pL, cL));
  const double S = (x - X0) / t;

  if (S < ust) {  // gauche du contact
    if (pst > pL) {  // choc gauche
      const double SL = uLL - cL * std::sqrt((GAM + 1) / (2 * GAM) * pst / pL +
                                             (GAM - 1) / (2 * GAM));
      if (S < SL) return rL;
      return rL * (pst / pL + (GAM - 1) / (GAM + 1)) /
             ((GAM - 1) / (GAM + 1) * pst / pL + 1);
    }
    const double cmL = cL * std::pow(pst / pL, (GAM - 1) / (2 * GAM));
    const double SHL = uLL - cL, STL = ust - cmL;
    if (S < SHL) return rL;
    if (S > STL) return rL * std::pow(pst / pL, 1.0 / GAM);
    const double cf = 2 / (GAM + 1) * (cL + (GAM - 1) / 2 * (uLL - S));
    return rL * std::pow(cf / cL, 2.0 / (GAM - 1));
  } else {  // droite du contact
    if (pst > pR) {  // choc droit
      const double SR = uRR + cR * std::sqrt((GAM + 1) / (2 * GAM) * pst / pR +
                                             (GAM - 1) / (2 * GAM));
      if (S > SR) return rR;
      return rR * (pst / pR + (GAM - 1) / (GAM + 1)) /
             ((GAM - 1) / (GAM + 1) * pst / pR + 1);
    }
    const double cmR = cR * std::pow(pst / pR, (GAM - 1) / (2 * GAM));
    const double SHR = uRR + cR, STR = ust + cmR;
    if (S > SHR) return rR;
    if (S < STR) return rR * std::pow(pst / pR, 1.0 / GAM);
    const double cf = 2 / (GAM + 1) * (cR - (GAM - 1) / 2 * (uRR - S));
    return rR * std::pow(cf / cR, 2.0 / (GAM - 1));
  }
}

// erreur L1 sur rho a t=0.2 pour un flux donne ; renvoie aussi min(rho), min(p).
template <class Flux>
static double run_sod(int Nx, int Ny, double& minrho, double& minp) {
  const double Tend = 0.2, cfl = 0.4, dx = 1.0 / Nx, Ly = Ny * dx;
  Box2D dom = Box2D::from_extents(Nx, Ny);
  Geometry geom{dom, 0.0, 1.0, 0.0, Ly};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  Euler model;

  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 2);
  aux.set_val(0.0);
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const bool left = geom.x_cell(i) < X0;
        const double r = left ? rL : rR, p = left ? pL : pR;
        f(i, j, 0) = r;
        f(i, j, 1) = 0.0;
        f(i, j, 2) = 0.0;
        f(i, j, 3) = p / (GAM - 1);  // u = 0
      }
  }
  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;  // transmissif en x ; periodique en y

  double t = 0;
  while (t < Tend - 1e-12) {
    double vmax = 0;
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        Euler::State s;
        for (int c = 0; c < 4; ++c) s[c] = u(i, j, c);
        vmax = std::fmax(vmax, model.max_wave_speed(s, Aux{}, 0));
      }
    double dt = cfl * dx / vmax;
    if (t + dt > Tend) dt = Tend - t;
    advance_ssprk2<Minmod, Flux>(model, U, aux, geom, bc, dt);
    t += dt;
  }

  double err = 0;
  minrho = 1e30;
  minp = 1e30;
  const ConstArray4 u = U.fab(0).const_array();
  const Box2D v = U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      Euler::State s;
      for (int c = 0; c < 4; ++c) s[c] = u(i, j, c);
      err += std::fabs(s[0] - sod_exact_rho(geom.x_cell(i), Tend));
      minrho = std::fmin(minrho, s[0]);
      minp = std::fmin(minp, model.pressure(s));
    }
  return err / (static_cast<double>(Nx) * Ny);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int Nx = 200, Ny = 4;
  double mr, mp;
  const double eRus = run_sod<RusanovFlux>(Nx, Ny, mr, mp);
  chk(mr > 0 && mp > 0, "rusanov_positif");
  const double eHLL = run_sod<HLLFlux>(Nx, Ny, mr, mp);
  chk(mr > 0 && mp > 0, "hll_positif");
  const double eHLLC = run_sod<HLLCFlux>(Nx, Ny, mr, mp);
  chk(mr > 0 && mp > 0, "hllc_positif");

  std::printf("Sod L1(rho) Nx=%d : Rusanov %.4e | HLL %.4e | HLLC %.4e\n", Nx,
              eRus, eHLL, eHLLC);

  chk(std::isfinite(eHLLC) && eHLLC < 0.02, "hllc_precis");
  chk(eHLL <= eRus * 1.02, "hll_pas_pire_que_rusanov");
  chk(eHLLC <= eHLL * 1.02, "hllc_le_plus_net");

  if (fails == 0) std::printf("OK test_riemann\n");
  return fails == 0 ? 0 : 1;
}
