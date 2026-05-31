// Valide AmrCouplerMP (coupleur AMR E x B multi-patch + regrid Berger-Rigoutsos) :
//   1. EQUIVALENCE : sur une hierarchie 2 niveaux MONO-BOX, sans regrid, il doit donner
//      le MEME niveau grossier qu'AmrCoupler (le coupleur mono-box deja valide), a
//      l'arrondi. Prouve que le couplage (sync_down / Poisson / inject / pas AMR) se
//      reduit exactement au cas mono-box.
//   2. CONSERVATION SOUS REGRID : sur une bosse advectee, regrid periodique par BR
//      (le niveau fin change de patchs en cours de route) -> masse grossiere conservee.

#include <adc/coupling/amr_coupler.hpp>     // AmrCoupler (reference)
#include <adc/coupling/amr_coupler_mp.hpp>  // AmrCouplerMP (candidat)
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  BCRec bc;

  Diocotron model;
  model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 1.0;

  // ===== Garde 1 : equivalence mono-box vs AmrCoupler =====
  {
    auto ne0 = [&](double x, double y) {
      return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
    };
    const int CI0 = nc / 4, CI1 = 3 * nc / 4 - 1, CJ0 = nc / 4, CJ1 = 3 * nc / 4 - 1;
    Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};
    auto initc = [&](MultiFab& U) {
      Array4 u = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    };
    auto initf = [&](MultiFab& U) {
      Array4 u = U.fab(0).array();
      const Box2D b = U.box(0);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
    };

    // reference AmrCoupler
    MultiFab Rc(ba, dm, 1, 1), Rf(BoxArray(std::vector<Box2D>{fbox}), dm, 1, 1);
    initc(Rc); initf(Rf); mf_average_down_mb(Rf, Rc);
    std::vector<AmrLevelMP> LR;
    LR.push_back({std::move(Rc), nullptr, dxc, dyc});
    LR.push_back({std::move(Rf), nullptr, dxf, dyf});
    AmrCoupler<Diocotron> ref(model, geom, ba, bc, std::move(LR));

    // candidat AmrCouplerMP (memes donnees)
    MultiFab Pc(ba, dm, 1, 1), Pf(BoxArray(std::vector<Box2D>{fbox}), dm, 1, 1);
    initc(Pc); initf(Pf); mf_average_down_mb(Pf, Pc);
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Pc), nullptr, dxc, dyc});
    LP.push_back({std::move(Pf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> cand(model, geom, ba, bc, std::move(LP));

    ref.update();
    cand.update();  // memes V-cycles : on warm-start les deux multigrilles a l'identique
    const double dt = 0.4 * dxc / ref.max_drift_speed();
    for (int s = 0; s < 20; ++s) { ref.step(dt); cand.step(dt); }
    double maxdiff = 0;
    const ConstArray4 ur = ref.coarse().fab(0).const_array();
    const ConstArray4 up = cand.coarse().fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(ur(i, j, 0) - up(i, j, 0)));
    std::printf("garde 1 (mono-box) : max|dUc| vs AmrCoupler = %.3e\n", maxdiff);
    chk(maxdiff < 1e-12, "couplermp_equiv_amrcoupler");
  }

  // ===== Garde 2 : conservation sous regrid Berger-Rigoutsos dynamique =====
  {
    auto blob = [&](double x, double y) {  // deux bosses -> BR produit plusieurs patchs
      return 1.0 + 0.6 * std::exp(-((x - 0.32) * (x - 0.32) + (y - 0.5) * (y - 0.5)) / 0.004) +
             0.6 * std::exp(-((x - 0.68) * (x - 0.68) + (y - 0.5) * (y - 0.5)) / 0.004);
    };
    auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > model.n_i0 + 0.05; };
    Box2D seed{{2 * (nc / 4), 2 * (nc / 4)}, {2 * (3 * nc / 4) - 1, 2 * (3 * nc / 4) - 1}};
    MultiFab Uc(ba, dm, 1, 1), Uf(BoxArray(std::vector<Box2D>{seed}), dm, 1, 1);
    {
      Array4 u = Uc.fab(0).array();
      const Box2D g = Uc.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dxc, (j + 0.5) * dyc);
      Array4 uf = Uf.fab(0).array();
      const Box2D b = Uf.box(0);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) uf(i, j, 0) = blob((i + 0.5) * dxf, (j + 0.5) * dyf);
    }
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> sim(model, geom, ba, bc, std::move(LP));

    sim.regrid(crit);  // patchs initiaux par BR
    sim.update();
    const double m0 = sim.mass();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    bool finite = true;
    int npatch = 0;
    for (int s = 0; s < 60; ++s) {
      if (s % 10 == 0) sim.regrid(crit);
      sim.step(dt);
      if (!std::isfinite(sim.mass())) finite = false;
      npatch = sim.levels()[1].U.local_size();
    }
    const double drift = std::fabs(sim.mass() - m0);
    std::printf("garde 2 (regrid BR dynamique) : npatch_final=%d drift_masse=%.3e %s\n",
                npatch, drift, finite ? "fini" : "NON-FINI");
    chk(finite, "couplermp_regrid_fini");
    chk(npatch >= 1, "couplermp_regrid_patchs");
    chk(drift < 1e-9, "couplermp_regrid_conservation");
  }

  if (fails == 0) std::printf("OK test_amr_coupler_mp\n");
  return fails == 0 ? 0 : 1;
}
