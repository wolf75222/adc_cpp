// Diocotron UNIFORME haute precision : reconstruction WENO5 (ou MUSCL) + SSPRK3, Poisson resolu
// a CHAQUE etage Runge-Kutta (couplage stade par stade), paroi conductrice circulaire embedded
// (GeometricMG::solve_robust). But : mesurer le taux de croissance gamma du mode l et son ERREUR
// RELATIVE vs l'analytique (g3=0.772, g4=0.911, g5=0.683), a la maniere de Hoffart arXiv:2510.11808
// (qui atteint 0.1 a 0.4 %). M1 a montre que le plafond venait de la DIFFUSION du schema (ordre 1
// en espace ET en temps) ; on monte les deux ordres pour viser < 1 % (idealement < 0.5 %).
//
// Run : ./diocotron_highorder <out> [nc] [nsteps] [l] [recon] [cfl] [delta]
//   recon : 0 NoSlope, 1 VanLeer, 2 Minmod, 3 WENO5 (defaut 3). delta : amplitude de
//   perturbation du mode (defaut 0.1, comme le papier ; plus petit = phase lineaire plus longue).
// Sortie : <out>/ring_amp.csv (t, amplitude du mode l de phi sur le cercle r0). Extraire gamma :
//   python3 scripts/validate_diocotron_growth.py <out>/ring_amp.csv --rhobar 0.9 --target <g_l> --window t0,t1

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

template <class Limiter>
static void run(const std::string& out, int nc, int nsteps, int l, double cfl, double delta) {
  const double L = 1.0, cx = 0.5, cy = 0.5;
  const double r0 = 0.15, r1 = 0.20, Rwall = 0.40, floor = 1e-3;
  const int ng = Limiter::n_ghost;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, L, 0.0, L};
  const double dxc = geom.dx(), dyc = geom.dy();
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  std::filesystem::create_directories(out);

  Diocotron model; model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 0.0;
  auto ne0 = [&](double x, double y) {
    const double r = std::hypot(x - cx, y - cy), th = std::atan2(y - cy, x - cx);
    return (r > r0 && r < r1) ? (1.0 - delta + delta * std::sin(l * th)) : floor;
  };
  std::function<bool(Real, Real)> active = [=](Real x, Real y) { return std::hypot(x - cx, y - cy) < Rwall; };
  BCRec bcPhi; bcPhi.xlo = bcPhi.xhi = bcPhi.ylo = bcPhi.yhi = BCType::Dirichlet;  // paroi phi=0
  BCRec bcU;   bcU.xlo = bcU.xhi = bcU.ylo = bcU.yhi = BCType::Foextrap;            // densite outflow
  BCRec bcAux; bcAux.xlo = bcAux.xhi = bcAux.ylo = bcAux.yhi = BCType::Foextrap;

  MultiFab U(ba, dm, 1, ng), aux(ba, dm, 3, ng);
  {
    Array4 u = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
  }
  GeometricMG mg(geom, ba, bcPhi, active);

  // Poisson(state) -> phi -> aux = (phi, grad phi) ; ghosts remplis. Resolu a chaque etage RK.
  auto solve_aux = [&](const MultiFab& state) {
    Array4 f = mg.rhs().fab(0).array();
    const ConstArray4 u0 = state.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) f(i, j) = model.alpha * (u0(i, j, 0) - model.n_i0);
    mg.solve_robust(1e-8, 50);
    fill_ghosts(mg.phi(), dom, bcPhi);
    const ConstArray4 p = mg.phi().fab(0).const_array();
    Array4 a = aux.fab(0).array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dxc);
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dyc);
      }
    fill_ghosts(aux, dom, bcAux);
  };
  // R = -div F(state, aux) : aux deja a jour pour `state` (solve_aux), ghosts densite remplis.
  auto eval_L = [&](MultiFab& state, MultiFab& R) {
    fill_ghosts(state, dom, bcU);
    assemble_rhs<Limiter, RusanovFlux>(model, state, aux, geom, R);
  };
  auto mode_amplitude = [&]() {
    const ConstArray4 p = mg.phi().fab(0).const_array();
    const int K = 512; double sr = 0, si = 0;
    for (int k = 0; k < K; ++k) {
      const double th = 2 * kPi * k / K;
      const double x = cx + r0 * std::cos(th), y = cy + r0 * std::sin(th);
      const double fx = x / dxc - 0.5, fy = y / dxc - 0.5;
      const int i0 = (int)std::floor(fx), j0 = (int)std::floor(fy);
      const double tx = fx - i0, ty = fy - j0;
      auto P = [&](int ii, int jj) { ii = std::clamp(ii, 0, nc - 1); jj = std::clamp(jj, 0, nc - 1); return p(ii, jj); };
      const double v = (1 - tx) * (1 - ty) * P(i0, j0) + tx * (1 - ty) * P(i0 + 1, j0) +
                       (1 - tx) * ty * P(i0, j0 + 1) + tx * ty * P(i0 + 1, j0 + 1);
      sr += v * std::cos(l * th); si += v * std::sin(l * th);
    }
    return 2.0 * std::hypot(sr, si) / K;
  };
  auto vmax = [&]() {
    const ConstArray4 a = aux.fab(0).const_array();
    double v = 0;
    for (int j = 0; j < nc; ++j) for (int i = 0; i < nc; ++i) v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model.B0);
    return std::max(v, 1e-12);
  };

  MultiFab R(ba, dm, 1, 0), U1(ba, dm, 1, ng), U2(ba, dm, 1, ng);
  solve_aux(U);
  const double dt0 = cfl * dxc / vmax();
  double dt = dt0, t = 0;
  std::ofstream amp(out + "/ring_amp.csv");
  amp << "# diocotron highorder nc=" << nc << " l=" << l << " ng=" << ng << " cfl=" << cfl << "\n";
  amp << "t,amplitude\n";
  const int snap = std::max(1, nsteps / 30);
  std::printf("highorder nc=%d l=%d ng=%d dt0=%.3e\n", nc, l, ng, dt0);
  for (int s = 0; s <= nsteps; ++s) {
    solve_aux(U);  // phi pour l'etat courant : diagnostic ET aux de l'etage 1
    amp << t << ',' << mode_amplitude() << '\n';
    if (s % snap == 0) std::printf("  s=%5d t=%7.3f a=%.4e\n", s, t, mode_amplitude());
    if (s == nsteps) break;
    // SSPRK3 (Shu-Osher), Poisson RE-RESOLU a chaque etage (couplage stade par stade).
    eval_L(U, R);                          // R = L(U)   (aux de solve_aux(U))
    lincomb(U1, Real(1), U, dt, R);        // U1 = U + dt L(U)
    solve_aux(U1); eval_L(U1, R);          // R = L(U1)
    lincomb(U2, Real(1), U1, dt, R);       // U2 = U1 + dt L(U1)
    lincomb(U2, Real(0.75), U, Real(0.25), U2);  // U2 = 3/4 U + 1/4 (U1 + dt L(U1))
    solve_aux(U2); eval_L(U2, R);          // R = L(U2)
    lincomb(U1, Real(1), U2, dt, R);       // U1 = U2 + dt L(U2)
    lincomb(U, Real(1) / 3, U, Real(2) / 3, U1);  // U = 1/3 U + 2/3 (U2 + dt L(U2))
    t += dt;
    if (s % 20 == 0) dt = std::min(dt0, cfl * dxc / vmax());
  }
  amp.close();
  std::printf("ecrit %s/ring_amp.csv\n", out.c_str());
}

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_ho";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 256;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 1500;
  const int l = (argc > 4) ? std::atoi(argv[4]) : 4;
  const int recon = (argc > 5) ? std::atoi(argv[5]) : 3;
  const double cfl = (argc > 6) ? std::atof(argv[6]) : 0.4;
  const double delta = (argc > 7) ? std::atof(argv[7]) : 0.1;  // amplitude de perturbation
  if (recon == 0) run<NoSlope>(out, nc, nsteps, l, cfl, delta);
  else if (recon == 1) run<VanLeer>(out, nc, nsteps, l, cfl, delta);
  else if (recon == 2) run<Minmod>(out, nc, nsteps, l, cfl, delta);
  else run<Weno5>(out, nc, nsteps, l, cfl, delta);
  std::printf("%s nc=%d l=%d recon=%d done\n", out.c_str(), nc, l, recon);
  return 0;
}
