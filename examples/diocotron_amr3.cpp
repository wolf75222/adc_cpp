// Diocotron sur AMR a 3 niveaux emboites. Le pas couple (sync + Poisson grossier +
// aux = grad phi injecte + sous-cyclage/reflux multi-niveaux) est porte par AmrCoupler
// (coupling/amr_coupler.hpp), desormais sur la pile MultiFab + le seam. Cet exemple ne
// garde que ce qui lui est PROPRE : le critere de raffinement (regrid imbrique par tag
// gradient) et l'I/O.
//
// Run : ./build/bin/diocotron_amr3 /tmp/dio3 [nc] [nsteps]

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

// magnitude du gradient non-divise (proxy d'erreur de troncature pour le tag).
static double gradmag(const ConstArray4& u, int i, int j) {
  const double gx = u(i + 1, j) - u(i - 1, j);
  const double gy = u(i, j + 1) - u(i, j - 1);
  return 0.5 * std::sqrt(gx * gx + gy * gy);
}

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio3";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 500;
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  const double dxf1 = dxc / 2, dyf1 = dyc / 2, dxf2 = dxc / 4, dyf2 = dyc / 4;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  auto MF = [&](const Box2D& b) { return MultiFab(BoxArray(std::vector<Box2D>{b}), dm, 1, 1); };

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  const double A = 1.0, w = 0.05, eta = 0.02;
  const int m = 2;
  auto ne0 = [&](double x, double y) {
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };

  int L1CI0 = nc / 8, L1CI1 = 7 * nc / 8 - 1;
  int L1CJ0 = 7 * nc / 16, L1CJ1 = 9 * nc / 16 - 1;
  Box2D fbox1{{2 * L1CI0, 2 * L1CJ0}, {2 * L1CI1 + 1, 2 * L1CJ1 + 1}};
  int L2CI0 = fbox1.lo[0] + nc / 6, L2CI1 = fbox1.hi[0] - nc / 6;
  int L2CJ0 = fbox1.lo[1] + 2, L2CJ1 = fbox1.hi[1] - 2;
  Box2D fbox2{{2 * L2CI0, 2 * L2CJ0}, {2 * L2CI1 + 1, 2 * L2CJ1 + 1}};

  MultiFab U0 = MF(dom), U1 = MF(fbox1), U2 = MF(fbox2);
  auto init = [&](MultiFab& U, double dx, double dy) {
    Array4 u = U.fab(0).array();
    const Box2D b = U.fab(0).grown_box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        u(i, j, 0) = ne0((i + 0.5) * dx, (j + 0.5) * dy);
  };
  init(U0, dxc, dyc);
  init(U1, dxf1, dyf1);
  init(U2, dxf2, dyf2);
  mf_average_down(U2, U1, L2CI0, L2CI1, L2CJ0, L2CJ1);
  mf_average_down(U1, U0, L1CI0, L1CI1, L1CJ0, L1CJ1);
  {
    const ConstArray4 u0 = U0.fab(0).const_array();
    double mean = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) mean += u0(i, j, 0);
    model.n_i0 = mean / (double(nc) * nc);
  }

  std::vector<AmrLevelMF> L0;
  L0.push_back({std::move(U0), nullptr, dxc, dyc, L1CI0, L1CI1, L1CJ0, L1CJ1, true});
  L0.push_back({std::move(U1), nullptr, dxf1, dyf1, L2CI0, L2CI1, L2CJ0, L2CJ1, true});
  L0.push_back({std::move(U2), nullptr, dxf2, dyf2, 0, 0, 0, 0, false});

  BCRec bc;  // periodique
  AmrCoupler<Diocotron> sim(model, geom, ba, bc, std::move(L0));
  std::vector<AmrLevelMF>& L = sim.levels();

  // --- PROPRE A CET EXEMPLE : regrid imbrique par tag gradient ---
  auto regrid = [&]() {
    fill_boundary(L[0].U, dom, Periodicity{true, true});  // ghosts pour le grad au bord
    const ConstArray4 c0t = L[0].U.fab(0).const_array();
    double gmax1 = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) gmax1 = std::max(gmax1, gradmag(c0t, i, j));
    const double thr1 = 0.20 * gmax1;
    int i0 = nc, i1 = -1, j0 = nc, j1 = -1;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        if (gradmag(c0t, i, j) > thr1) {
          i0 = std::min(i0, i); i1 = std::max(i1, i);
          j0 = std::min(j0, j); j1 = std::max(j1, j);
        }
    if (i1 < i0) return;
    const int buf = 4;
    int nL1CI0 = std::max(2, i0 - buf), nL1CI1 = std::min(nc - 3, i1 + buf);
    int nL1CJ0 = std::max(2, j0 - buf), nL1CJ1 = std::min(nc - 3, j1 + buf);
    Box2D nf1{{2 * nL1CI0, 2 * nL1CJ0}, {2 * nL1CI1 + 1, 2 * nL1CJ1 + 1}};

    MultiFab nU1 = MF(nf1);
    {
      const ConstArray4 c0 = L[0].U.fab(0).const_array();
      const ConstArray4 o1 = L[1].U.fab(0).const_array();
      const Box2D old1 = L[1].U.box(0);
      Array4 a = nU1.fab(0).array();
      for (int j = nf1.lo[1]; j <= nf1.hi[1]; ++j)
        for (int i = nf1.lo[0]; i <= nf1.hi[0]; ++i)
          a(i, j, 0) = old1.contains(i, j) ? o1(i, j, 0) : c0(crsn(i), crsn(j), 0);
    }

    int k0 = nf1.hi[0], k1 = nf1.lo[0], l0 = nf1.hi[1], l1 = nf1.lo[1];
    {
      const ConstArray4 a = nU1.fab(0).const_array();
      double gmax2 = 0;
      for (int j = nf1.lo[1] + 1; j <= nf1.hi[1] - 1; ++j)
        for (int i = nf1.lo[0] + 1; i <= nf1.hi[0] - 1; ++i)
          gmax2 = std::max(gmax2, gradmag(a, i, j));
      const double thr2 = 0.40 * gmax2;
      for (int j = nf1.lo[1] + 1; j <= nf1.hi[1] - 1; ++j)
        for (int i = nf1.lo[0] + 1; i <= nf1.hi[0] - 1; ++i)
          if (gradmag(a, i, j) > thr2) {
            k0 = std::min(k0, i); k1 = std::max(k1, i);
            l0 = std::min(l0, j); l1 = std::max(l1, j);
          }
    }
    const int buf2 = 4;
    int nL2CI0, nL2CI1, nL2CJ0, nL2CJ1;
    if (k1 < k0) {
      nL2CI0 = nf1.lo[0] + nf1.nx() / 3; nL2CI1 = nf1.hi[0] - nf1.nx() / 3;
      nL2CJ0 = nf1.lo[1] + nf1.ny() / 3; nL2CJ1 = nf1.hi[1] - nf1.ny() / 3;
    } else {
      nL2CI0 = std::max(nf1.lo[0] + 1, k0 - buf2);
      nL2CI1 = std::min(nf1.hi[0] - 1, k1 + buf2);
      nL2CJ0 = std::max(nf1.lo[1] + 1, l0 - buf2);
      nL2CJ1 = std::min(nf1.hi[1] - 1, l1 + buf2);
    }
    Box2D nf2{{2 * nL2CI0, 2 * nL2CJ0}, {2 * nL2CI1 + 1, 2 * nL2CJ1 + 1}};

    MultiFab nU2 = MF(nf2);
    {
      const ConstArray4 c1 = nU1.fab(0).const_array();
      const ConstArray4 o2 = L[2].U.fab(0).const_array();
      const Box2D old2 = L[2].U.box(0);
      Array4 a = nU2.fab(0).array();
      for (int j = nf2.lo[1]; j <= nf2.hi[1]; ++j)
        for (int i = nf2.lo[0]; i <= nf2.hi[0]; ++i)
          a(i, j, 0) = old2.contains(i, j) ? o2(i, j, 0) : c1(crsn(i), crsn(j), 0);
    }

    L[1].U = std::move(nU1);  // aux resynchronise par le stepper
    L[2].U = std::move(nU2);
    L[0].rCI0 = nL1CI0; L[0].rCI1 = nL1CI1; L[0].rCJ0 = nL1CJ0; L[0].rCJ1 = nL1CJ1;
    L[1].rCI0 = nL2CI0; L[1].rCI1 = nL2CI1; L[1].rCJ0 = nL2CJ0; L[1].rCJ1 = nL2CJ1;
  };

  // --- I/O : densite par niveau + extents ---
  std::ofstream boxes(out + "/boxes.csv");
  boxes << "frame,x1lo,x1hi,y1lo,y1hi,x2lo,x2hi,y2lo,y2hi\n";
  auto dump = [&](int frame) {
    char nm[64];
    const ConstArray4 c0 = L[0].U.fab(0).const_array();
    std::snprintf(nm, sizeof(nm), "/dens_c_%04d.txt", frame);
    std::ofstream fc(out + nm);
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) fc << c0(i, j, 0) << (i + 1 < nc ? ' ' : '\n');
    const Box2D b1 = L[1].U.box(0), b2 = L[2].U.box(0);
    const ConstArray4 a1 = L[1].U.fab(0).const_array(), a2 = L[2].U.fab(0).const_array();
    std::snprintf(nm, sizeof(nm), "/dens_1_%04d.txt", frame);
    std::ofstream f1(out + nm);
    for (int j = b1.lo[1]; j <= b1.hi[1]; ++j)
      for (int i = b1.lo[0]; i <= b1.hi[0]; ++i)
        f1 << a1(i, j, 0) << (i < b1.hi[0] ? ' ' : '\n');
    std::snprintf(nm, sizeof(nm), "/dens_2_%04d.txt", frame);
    std::ofstream f2(out + nm);
    for (int j = b2.lo[1]; j <= b2.hi[1]; ++j)
      for (int i = b2.lo[0]; i <= b2.hi[0]; ++i)
        f2 << a2(i, j, 0) << (i < b2.hi[0] ? ' ' : '\n');
    boxes << frame << ',' << b1.lo[0] * dxf1 << ',' << (b1.hi[0] + 1) * dxf1 << ','
          << b1.lo[1] * dyf1 << ',' << (b1.hi[1] + 1) * dyf1 << ','
          << b2.lo[0] * dxf2 << ',' << (b2.hi[0] + 1) * dxf2 << ','
          << b2.lo[1] * dyf2 << ',' << (b2.hi[1] + 1) * dyf2 << '\n';
  };

  sim.update();
  const double M0 = sim.mass();
  double dt = 0.4 * dxc / sim.max_drift_speed();
  const int snap = std::max(1, nsteps / 30);
  std::printf("diocotron AMR 3 niveaux (AmrCoupler MultiFab) nc=%d dt=%.2e\n", nc, dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      const Box2D b1 = L[1].U.box(0), b2 = L[2].U.box(0);
      std::printf("  s=%4d  L1=[%d..%d]x[%d..%d] L2=[%d..%d]x[%d..%d] drift=%.2e\n", s,
                  b1.lo[0], b1.hi[0], b1.lo[1], b1.hi[1], b2.lo[0], b2.hi[0], b2.lo[1],
                  b2.hi[1], std::fabs(sim.mass() - M0));
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
