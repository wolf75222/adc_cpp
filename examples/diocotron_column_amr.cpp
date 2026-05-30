// Instabilite diocotron COLONNE CREUSE sur AMR 2 niveaux multi-patch (objectif Hoffart M2).
// Anneau de charge r in [0.15,0.20] dans une cavite conductrice circulaire r=0.40 (paroi
// embedded portee par le multigrille), perturbe au mode azimutal l. Le niveau fin raffine le
// BORD D'ANNEAU (la ou l'instabilite croit), re-cluster par Berger-Rigoutsos. But : recuperer
// le taux de croissance analytique (g_4 = 0.911 pour 0.15:0.20:0.40 = 6:8:16) avec BIEN MOINS
// de cellules que la grille uniforme, qui plafonne a ~60% par diffusion du bord (cf. M1).
//
// Meme binaire pour les deux branches de la comparaison (memes numeriques) :
//   refine=1 : AMR (grossier nc + patchs fins 2x autour de l'anneau).
//   refine=0 : UNIFORME (grossier nc seul, pas de niveau fin) -> reference.
//
// Run : ./build/bin/diocotron_column_amr <out> [nc] [nsteps] [refine] [l]
// Sortie : <out>/ring_amp.csv (t, amplitude du mode l de phi a r=r0) + <out>/cells.txt
//          (cellules grossier + somme patchs ; pour la comparaison de cout).

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
#include <functional>
#include <string>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_col_amr";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 96;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 1200;
  const bool refine = (argc > 4) ? (std::atoi(argv[4]) != 0) : true;
  const int l = (argc > 5) ? std::atoi(argv[5]) : 4;
  std::filesystem::create_directories(out);

  const double L = 1.0, cx = 0.5 * L, cy = 0.5 * L;
  const double r0 = 0.15, r1 = 0.20, Rwall = 0.40, delta = 0.1, floor = 1e-3;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, L, 0.0, L};
  const double dxc = geom.dx(), dyc = geom.dy();
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 0.0;  // colonne d'electrons pure : RHS = alpha*ne (Dirichlet ferme le probleme)

  // CI anneau : ne = 1 - delta + delta*sin(l*theta) dans l'anneau, plancher ailleurs.
  auto ne0 = [&](double x, double y) {
    const double r = std::hypot(x - cx, y - cy), th = std::atan2(y - cy, x - cx);
    return (r > r0 && r < r1) ? (1.0 - delta + delta * std::sin(l * th)) : floor;
  };
  // paroi conductrice : cellule active = interieur du cercle Rwall (passe au MG).
  std::function<bool(Real, Real)> active = [cx, cy, Rwall](Real x, Real y) {
    return std::hypot(x - cx, y - cy) < Rwall;
  };
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // paroi conductrice phi=0

  MultiFab Uc(ba, dm, 1, 1), auxc(ba, dm, 3, 1);
  {
    Array4 u = Uc.fab(0).array();
    const Box2D g = Uc.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
  }

  GeometricMG mg(geom, ba, bc, active);  // multigrille avec paroi embedded
  MultiFab Uf, auxf;                     // niveau fin multi-box (vide si refine=0)

  auto compute_coarse_aux = [&]() {
    fill_boundary(Uc, dom, Periodicity{false, false});
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
    fill_boundary(auxc, dom, Periodicity{false, false});
  };

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

  // tag GEOMETRIQUE : couronne [0.13,0.22] englobant l'anneau -> raffine le bord des t=0.
  auto regrid = [&]() {
    auto crit = [&](const ConstArray4&, int i, int j) {
      const double x = (i + 0.5) * dxc, y = (j + 0.5) * dyc, r = std::hypot(x - cx, y - cy);
      return r > 0.13 && r < 0.22;
    };
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

  // amplitude du mode l de phi echantillonne sur le cercle r=r0 (acces direct mg.phi()).
  auto mode_amplitude = [&]() {
    const ConstArray4 p = mg.phi().fab(0).const_array();
    const int K = 256;
    double sr = 0, si = 0;
    for (int k = 0; k < K; ++k) {
      const double th = 2 * kPi * k / K;
      const double x = cx + r0 * std::cos(th), y = cy + r0 * std::sin(th);
      const double fx = x / dxc - 0.5, fy = y / dyc - 0.5;
      const int i0 = (int)std::floor(fx), j0 = (int)std::floor(fy);
      const double tx = fx - i0, ty = fy - j0;
      auto P = [&](int ii, int jj) {
        ii = std::clamp(ii, 0, nc - 1); jj = std::clamp(jj, 0, nc - 1); return p(ii, jj);
      };
      const double v = (1 - tx) * (1 - ty) * P(i0, j0) + tx * (1 - ty) * P(i0 + 1, j0) +
                       (1 - tx) * ty * P(i0, j0 + 1) + tx * ty * P(i0 + 1, j0 + 1);
      sr += v * std::cos(l * th); si += v * std::sin(l * th);
    }
    return 2.0 * std::hypot(sr, si) / K;
  };

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
  auto fine_cells = [&]() {
    long c = 0;
    for (int li = 0; li < Uf.local_size(); ++li) c += Uf.box(li).num_cells();
    return c;
  };

  compute_coarse_aux();
  if (refine) regrid();
  if (refine) mf_average_down_multi(Uf, Uc);
  const double M0 = mass();
  double dt = 0.4 * dxc / vmax(), t = 0;

  std::ofstream amp(out + "/ring_amp.csv");
  amp << "# diocotron colonne AMR nc=" << nc << " refine=" << refine << " l=" << l
      << " (uniforme effectif " << (refine ? 2 * nc : nc) << ")\n";
  amp << "t,amplitude\n";
  long maxcells = 0;

  const int snap = std::max(1, nsteps / 40);
  std::printf("colonne-AMR nc=%d refine=%d : %d patchs, dt=%.2e\n", nc, refine,
              Uf.local_size(), dt);
  for (int s = 0; s <= nsteps; ++s) {
    const long cells = (long)nc * nc + fine_cells();
    maxcells = std::max(maxcells, cells);
    amp << t << ',' << mode_amplitude() << '\n';
    if (s % snap == 0)
      std::printf("  s=%5d t=%7.3f a=%.4e npatch=%d cells=%ld drift=%.2e\n", s, t,
                  mode_amplitude(), Uf.local_size(), cells, std::fabs(mass() - M0));
    if (s == nsteps) break;
    if (refine && s > 0 && s % 30 == 0) { compute_coarse_aux(); regrid(); }
    compute_coarse_aux();
    if (refine) inject_aux();
    amr_step_2level_multipatch<NoSlope, RusanovFlux>(model, Uc, dom, dxc, dyc, Uf, auxc, auxf, dt);
    t += dt;
    if (s % 20 == 0) dt = 0.4 * dxc / vmax();
  }
  amp.close();
  std::ofstream(out + "/cells.txt") << "max_cells " << maxcells << "\n"
                                    << "uniform_equiv " << (long)(refine ? 2 * nc : nc) *
                                                               (refine ? 2 * nc : nc)
                                    << "\n";
  std::printf("ecrit %s/ring_amp.csv ; max cellules=%ld ; derive masse=%.2e\n", out.c_str(),
              maxcells, std::fabs(mass() - M0));
  return 0;
}
