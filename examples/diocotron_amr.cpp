// Diocotron sur AMR a 2 niveaux (couplage decouple). Le pas couple (sync + Poisson
// grossier + aux = grad phi injecte + sous-cyclage/reflux) est porte par le composant
// AmrCoupler (coupling/amr_coupler.hpp), desormais sur la pile MultiFab + le seam. Cet
// exemple ne garde que ce qui lui est propre : le regrid dynamique (bounding box des
// cellules au-dessus du fond) et l'I/O.
//
// Run : ./build/bin/diocotron_amr /tmp/dio_amr [nc] [nsteps]

#include <adc/coupling/amr_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;
static int crsn(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_amr";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 500;
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  const double A = 1.0, w = 0.05, eta = 0.02;
  const int m = 2;
  auto ne0 = [&](double x, double y) {
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };

  int CI0 = nc / 8, CI1 = 7 * nc / 8 - 1, CJ0 = 7 * nc / 16, CJ1 = 9 * nc / 16 - 1;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  MultiFab Uc(ba, dm, 1, 1), Uf(BoxArray(std::vector<Box2D>{fbox}), dm, 1, 1);
  {
    Array4 uc = Uc.fab(0).array();
    const Box2D g = Uc.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        uc(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    Array4 uf = Uf.fab(0).array();
    for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
      for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
        uf(i, j, 0) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
  }
  mf_average_down_mb(Uf, Uc);
  {
    const ConstArray4 uc = Uc.fab(0).const_array();
    double mean = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) mean += uc(i, j, 0);
    model.n_i0 = mean / (double(nc) * nc);
  }

  std::vector<AmrLevelMP> L0;
  L0.push_back({std::move(Uc), nullptr, dxc, dyc});
  L0.push_back({std::move(Uf), nullptr, dxf, dyf});

  BCRec bc;  // periodique
  AmrCoupler<Diocotron> sim(model, geom, ba, bc, std::move(L0));
  std::vector<AmrLevelMP>& L = sim.levels();

  // --- PROPRE A CET EXEMPLE : regrid dynamique (densite au-dessus du fond) ---
  auto regrid = [&]() {
    const ConstArray4 c = L[0].U.fab(0).const_array();
    int i0 = nc, i1 = -1, j0 = nc, j1 = -1;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        if (c(i, j, 0) > model.n_i0 + 0.12) {
          i0 = std::min(i0, i); i1 = std::max(i1, i);
          j0 = std::min(j0, j); j1 = std::max(j1, j);
        }
    if (i1 < i0) return;
    const int buf = 4;
    const int nCI0 = std::max(2, i0 - buf), nCI1 = std::min(nc - 3, i1 + buf);
    const int nCJ0 = std::max(2, j0 - buf), nCJ1 = std::min(nc - 3, j1 + buf);
    const Box2D oldfb = L[1].U.box(0);  // box fine courante ; sa coarsen = region (niveau 0)
    if (nCI0 == oldfb.lo[0] / 2 && nCI1 == (oldfb.hi[0] - 1) / 2 &&
        nCJ0 == oldfb.lo[1] / 2 && nCJ1 == (oldfb.hi[1] - 1) / 2)
      return;
    Box2D nf{{2 * nCI0, 2 * nCJ0}, {2 * nCI1 + 1, 2 * nCJ1 + 1}};
    MultiFab Ufn(BoxArray(std::vector<Box2D>{nf}), dm, 1, 1);
    const ConstArray4 ofo = L[1].U.fab(0).const_array();
    Array4 a = Ufn.fab(0).array();
    for (int j = nf.lo[1]; j <= nf.hi[1]; ++j)
      for (int i = nf.lo[0]; i <= nf.hi[0]; ++i)
        a(i, j, 0) = oldfb.contains(i, j) ? ofo(i, j, 0) : c(crsn(i), crsn(j), 0);
    L[1].U = std::move(Ufn);  // region portee par le box_array ; aux resync par le stepper
  };

  // --- I/O ---
  std::ofstream boxes(out + "/boxes.csv");
  boxes << "frame,CI0,CI1,CJ0,CJ1,nc\n";
  auto dump = [&](int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    const ConstArray4 u = L[0].U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) f << u(i, j, 0) << (i + 1 < nc ? ' ' : '\n');
    const Box2D fb = L[1].U.box(0);
    boxes << frame << ',' << fb.lo[0] / 2 << ',' << (fb.hi[0] - 1) / 2 << ','
          << fb.lo[1] / 2 << ',' << (fb.hi[1] - 1) / 2 << ',' << nc << '\n';
  };

  sim.update();
  const double M0 = sim.mass();
  double dt = 0.4 * dxc / sim.max_drift_speed();
  const int snap = std::max(1, nsteps / 30);
  std::printf("diocotron AMR 2 niveaux (AmrCoupler MultiFab) nc=%d dt=%.2e\n", nc, dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      const Box2D fb = L[1].U.box(0);
      std::printf("  s=%4d  fine=[%d..%d]x[%d..%d]  drift=%.2e\n", s, fb.lo[0] / 2,
                  (fb.hi[0] - 1) / 2, fb.lo[1] / 2, (fb.hi[1] - 1) / 2,
                  std::fabs(sim.mass() - M0));
    }
    if (s == nsteps) break;
    if (s > 0 && s % 20 == 0) regrid();
    sim.step(dt);
    if (s % 20 == 0) dt = 0.4 * dxc / sim.max_drift_speed();
  }
  std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(), frame,
              std::fabs(sim.mass() - M0));
  return 0;
}
