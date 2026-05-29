// Diocotron couple sur AMR 2 niveaux MULTI-PATCH avec regrid Berger-Rigoutsos
// dynamique. C'est le payoff de la pile multi-patch : le niveau fin est un ENSEMBLE
// de boxes (pas une box unique), re-cluster a la volee par BR autour des zones de
// fort gradient. Chaque pas : Poisson grossier (multigrille) -> aux = grad phi
// injecte aux patchs -> amr_step_2level_multipatch (reflux coverage-aware, conservatif).
//
// Run : ./build/bin/diocotron_multipatch /tmp/diomp [nc] [nsteps]
// Sortie : <out>/dens_XXXX.txt (grossier) + <out>/boxes_XXXX.txt (patchs fins).

#include <adc/amr/cluster.hpp>
#include <adc/amr/regrid.hpp>
#include <adc/amr/tag_box.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>
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

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "diomp";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 96;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 400;
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  // deux bandes de charge -> deux structures distinctes -> BR produit plusieurs patchs
  auto ne0 = [&](double x, double y) {
    const double a = std::exp(-((y - 0.35) * (y - 0.35)) / 0.0025) *
                     (1 + 0.02 * std::cos(2 * kPi * 3 * x));
    const double b = std::exp(-((y - 0.65) * (y - 0.65)) / 0.0025) *
                     (1 + 0.02 * std::cos(2 * kPi * 3 * x));
    return 1.0 + 0.8 * (a + b);
  };

  MultiFab Uc(ba, dm, 1, 1), auxc(ba, dm, 3, 1);
  {
    Array4 u = Uc.fab(0).array();
    const Box2D g = Uc.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    const ConstArray4 uc = Uc.fab(0).const_array();
    double mean = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) mean += uc(i, j, 0);
    model.n_i0 = mean / (double(nc) * nc);
  }

  BCRec bc;  // periodique
  GeometricMG mg(geom, ba, bc);
  MultiFab Uf, auxf;  // niveau fin multi-box (rempli par le regrid)

  // Poisson grossier -> aux = (phi, grad phi) grossier.
  auto compute_coarse_aux = [&]() {
    fill_boundary(Uc, dom, Periodicity{true, true});
    Array4 f = mg.rhs().fab(0).array();
    const ConstArray4 u0 = Uc.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) f(i, j) = model.alpha * (u0(i, j, 0) - model.n_i0);
    mg.solve();
    const ConstArray4 p = mg.phi().fab(0).const_array();
    Array4 a = auxc.fab(0).array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dxc);
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dyc);
      }
    fill_boundary(auxc, dom, Periodicity{true, true});
  };

  // injection piecewise-constante de aux grossier -> patchs fins (valides + ghosts).
  auto inject_aux = [&]() {
    const ConstArray4 ac = auxc.fab(0).const_array();
    for (int li = 0; li < auxf.local_size(); ++li) {
      Array4 af = auxf.fab(li).array();
      const Box2D g = auxf.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          for (int k = 0; k < 3; ++k) af(i, j, k) = ac(coarsen_index(i, 2), coarsen_index(j, 2), k);
    }
  };

  // regrid : tag (densite au-dessus du fond) -> BR -> patchs multi-box, en reportant
  // les donnees fines existantes la ou les box se recouvrent, sinon interp grossier.
  auto regrid = [&]() {
    auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > model.n_i0 + 0.10; };
    TagBox tags = tag_cells(Uc, dom, crit);
    TagBox grown = grow_tags(tags, 2, dom);
    std::vector<Box2D> patches = berger_rigoutsos(grown, ClusterParams{});
    std::vector<Box2D> fboxes;
    for (Box2D b : patches) {
      b.lo[0] = std::max(b.lo[0], 2); b.lo[1] = std::max(b.lo[1], 2);
      b.hi[0] = std::min(b.hi[0], nc - 3); b.hi[1] = std::min(b.hi[1], nc - 3);
      if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
      fboxes.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
    }
    if (fboxes.empty()) return;
    MultiFab nUf(BoxArray(fboxes), DistributionMapping((int)fboxes.size(), n_ranks()), 1, 1);
    const ConstArray4 uc = Uc.fab(0).const_array();
    for (int li = 0; li < nUf.local_size(); ++li) {
      Array4 a = nUf.fab(li).array();
      const Box2D nb = nUf.box(li);
      // 1) interp grossier partout, 2) report des anciennes donnees fines la ou possible
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i) a(i, j, 0) = uc(coarsen_index(i, 2), coarsen_index(j, 2), 0);
      for (int ol = 0; ol < Uf.local_size(); ++ol) {
        const ConstArray4 o = Uf.fab(ol).const_array();
        const Box2D ob = Uf.box(ol), inter = nb.intersect(ob);
        if (inter.empty()) continue;
        for (int j = inter.lo[1]; j <= inter.hi[1]; ++j)
          for (int i = inter.lo[0]; i <= inter.hi[0]; ++i) a(i, j, 0) = o(i, j, 0);
      }
    }
    Uf = std::move(nUf);
    auxf = MultiFab(Uf.box_array(), Uf.dmap(), 3, 1);
  };

  compute_coarse_aux();
  regrid();  // patchs initiaux

  auto vmax = [&]() {
    const ConstArray4 a = auxc.fab(0).const_array();
    double v = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model.B0);
    return std::max(v, 1e-12);
  };
  auto mass = [&]() {
    const ConstArray4 u = Uc.fab(0).const_array();
    double M = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += u(i, j, 0) * dxc * dyc;
    return M;
  };
  // resync init (cellules couvertes = moyenne fine) avant de mesurer la masse
  mf_average_down_multi(Uf, Uc);
  const double M0 = mass();
  double dt = 0.4 * dxc / vmax();

  std::ofstream meta(out + "/run.csv");
  meta << "frame,npatch,drift\n";
  auto dump = [&](int frame) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "/dens_%04d.txt", frame);
    std::ofstream fd(out + nm);
    const ConstArray4 u = Uc.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) fd << u(i, j, 0) << (i + 1 < nc ? ' ' : '\n');
    std::snprintf(nm, sizeof(nm), "/boxes_%04d.txt", frame);
    std::ofstream fb(out + nm);
    for (int li = 0; li < Uf.local_size(); ++li) {
      const Box2D b = Uf.box(li);
      fb << b.lo[0] * dxc / 2 << ' ' << b.lo[1] * dyc / 2 << ' ' << (b.hi[0] + 1) * dxc / 2
         << ' ' << (b.hi[1] + 1) * dyc / 2 << '\n';
    }
    meta << frame << ',' << Uf.local_size() << ',' << std::fabs(mass() - M0) << '\n';
  };

  const int snap = std::max(1, nsteps / 30);
  std::printf("diocotron multipatch nc=%d : %d patchs initiaux, dt=%.2e\n", nc,
              Uf.local_size(), dt);
  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      std::printf("  s=%4d  npatch=%d  drift=%.2e\n", s, Uf.local_size(),
                  std::fabs(mass() - M0));
    }
    if (s == nsteps) break;
    if (s > 0 && s % 20 == 0) { compute_coarse_aux(); regrid(); }
    compute_coarse_aux();
    inject_aux();
    amr_step_2level_multipatch<NoSlope, RusanovFlux>(model, Uc, dom, dxc, dyc, Uf, auxc, auxf, dt);
    if (s % 20 == 0) dt = 0.4 * dxc / vmax();
  }
  std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(), frame,
              std::fabs(mass() - M0));
  return 0;
}
