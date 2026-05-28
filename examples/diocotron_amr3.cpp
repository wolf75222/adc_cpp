// Diocotron sur AMR a 3 niveaux emboites (couplage decouple) : transport
// sous-cycle recursif (Berger-Oliger) avec reflux a chaque interface, regrid
// dynamique imbrique. Poisson resolu sur la grille grossiere uniforme (phi
// lisse), aux = grad phi injecte vers les deux niveaux fins.
//
// Niveau 0 : grille grossiere nc x nc (Poisson + transport).
// Niveau 1 : raffine ratio 2 la bande de charge (suit l'enroulement).
// Niveau 2 : raffine ratio 2 les coeurs denses des tourbillons (le plus fin
//            fait r*r = 4 sous-pas par pas grossier).
//
// A chaque pas : average_down (sync ascendant), regrid imbrique periodique,
// Poisson grossier (multigrille), aux = grad phi (grossier + injection vers les
// fins), amr_step_multilevel sur la pile a 3 niveaux.
//
// Run : ./build/bin/diocotron_amr3 /tmp/dio3 [nc] [nsteps]

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_multilevel.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
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

// magnitude du gradient non-divise (difference centrale) : proxy d'erreur de
// troncature pour le tagging (critere gradient facon Berger-Colella, sect. 4.1).
// Independant de dx -> seuil exprime en unites de densite par cellule.
static double gradmag(const ConstArray4& u, int i, int j) {
  const double gx = u(i + 1, j) - u(i - 1, j);
  const double gy = u(i, j + 1) - u(i, j - 1);
  return 0.5 * std::sqrt(gx * gx + gy * gy);
}

// remplissage periodique multi-composantes d'un Fab2D
static void fill_periodic_mc(Fab2D& F, const Box2D& dom) {
  const int ng = F.n_ghost(), nc = F.ncomp();
  for (int c = 0; c < nc; ++c) {
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int g = 1; g <= ng; ++g) {
        F(dom.lo[0] - g, j, c) = F(dom.hi[0] - g + 1, j, c);
        F(dom.hi[0] + g, j, c) = F(dom.lo[0] + g - 1, j, c);
      }
    for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
      for (int g = 1; g <= ng; ++g) {
        F(i, dom.lo[1] - g, c) = F(i, dom.hi[1] - g + 1, c);
        F(i, dom.hi[1] + g, c) = F(i, dom.lo[1] + g - 1, c);
      }
  }
}

// injection piecewise-constante de aux (3 comp) parent -> enfant (valides +
// ghosts), ratio 2 : aux est un champ grossier (grad phi connu au niveau 0).
static void inject_aux(const Fab2D& parent, Fab2D& child) {
  const ConstArray4 p = parent.const_array();
  Array4 c = child.array();
  const Box2D g = child.grown_box();
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      for (int k = 0; k < 3; ++k) c(i, j, k) = p(crsn(i), crsn(j), k);
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

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  const double A = 1.0, w = 0.05;
  const int m = 2;
  const double eta = 0.02;
  auto ne0 = [&](double x, double y) {
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };

  // --- regions initiales : niveau 1 (coords niveau 0), niveau 2 (coords niveau
  // 1, strictement interieur a la box du niveau 1). Le regrid les recalcule. ---
  int L1CI0 = nc / 8, L1CI1 = 7 * nc / 8 - 1;
  int L1CJ0 = 7 * nc / 16, L1CJ1 = 9 * nc / 16 - 1;
  Box2D fbox1{{2 * L1CI0, 2 * L1CJ0}, {2 * L1CI1 + 1, 2 * L1CJ1 + 1}};
  int L2CI0 = fbox1.lo[0] + nc / 6, L2CI1 = fbox1.hi[0] - nc / 6;
  int L2CJ0 = fbox1.lo[1] + 2, L2CJ1 = fbox1.hi[1] - 2;
  Box2D fbox2{{2 * L2CI0, 2 * L2CJ0}, {2 * L2CI1 + 1, 2 * L2CJ1 + 1}};

  Fab2D U0(dom, 1, 1), U1(fbox1, 1, 1), U2(fbox2, 1, 1);
  Fab2D a0(dom, 3, 1), a1(fbox1, 3, 1), a2(fbox2, 3, 1);
  auto init = [&](Fab2D& U, double dx, double dy) {
    const Box2D b = U.grown_box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        U(i, j) = ne0((i + 0.5) * dx, (j + 0.5) * dy);
  };
  init(U0, dxc, dyc);
  init(U1, dxf1, dyf1);
  init(U2, dxf2, dyf2);
  average_down_fab(U2, U1, L2CI0, L2CI1, L2CJ0, L2CJ1);  // 2 -> 1
  average_down_fab(U1, U0, L1CI0, L1CI1, L1CJ0, L1CJ1);  // 1 -> 0

  double mean = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) mean += U0(i, j);
  mean /= double(nc) * nc;
  model.n_i0 = mean;

  std::vector<AmrLevel> L(3);
  L[0] = {std::move(U0), &a0, dxc,  dyc,  L1CI0, L1CI1, L1CJ0, L1CJ1, true};
  L[1] = {std::move(U1), &a1, dxf1, dyf1, L2CI0, L2CI1, L2CJ0, L2CJ1, true};
  L[2] = {std::move(U2), &a2, dxf2, dyf2, 0,     0,     0,     0,     false};

  BCRec bc;
  GeometricMG mg(geom, ba, bc);

  // Poisson grossier + aux = grad phi, injecte vers les niveaux fins.
  auto compute_aux = [&]() {
    Array4 f = mg.rhs().fab(0).array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        f(i, j) = model.alpha * (L[0].U(i, j) - model.n_i0);
    mg.solve(1e-8, 30);
    const ConstArray4 p = mg.phi().fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        a0(i, j, 0) = p(i, j);
        a0(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dxc);
        a0(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dyc);
      }
    fill_periodic_mc(a0, dom);
    inject_aux(a0, a1);  // 0 -> 1 (valides + ghosts)
    inject_aux(a1, a2);  // 1 -> 2
  };

  auto vmax = [&]() {
    double v = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        v = std::max(v, std::hypot(a0(i, j, 1), a0(i, j, 2)) / model.B0);
    return std::max(v, 1e-12);
  };
  auto mass = [&]() {
    double M = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += L[0].U(i, j) * dxc * dyc;
    return M;
  };
  auto sync_down = [&]() {
    average_down_fab(L[2].U, L[1].U, L[1].rCI0, L[1].rCI1, L[1].rCJ0, L[1].rCJ1);
    average_down_fab(L[1].U, L[0].U, L[0].rCI0, L[0].rCI1, L[0].rCJ0, L[0].rCJ1);
  };

  // --- regrid imbrique : niveau 1 = bbox des cellules grossieres taguees ;
  // niveau 2 = bbox des cellules fines taguees (seuil plus haut, coeurs denses),
  // clippe strictement a l'interieur du niveau 1. ---
  auto regrid = [&]() {
    // niveau 1 : tag gradient (bords des structures) sur le grossier. Seuil
    // relatif au max -> s'adapte a la decroissance des gradients (diffusion).
    fill_periodic_fab(L[0].U, dom);  // ghosts periodiques pour le grad au bord
    const ConstArray4 c0t = L[0].U.const_array();
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

    // transfert niveau 1 (ancien fin la ou il existe, sinon injection niveau 0)
    Fab2D nU1(nf1, 1, 1), na1(nf1, 3, 1);
    {
      const ConstArray4 c0 = L[0].U.const_array();
      const ConstArray4 o1 = L[1].U.const_array();
      const Box2D old1 = L[1].U.box();
      Array4 a = nU1.array();
      for (int j = nf1.lo[1]; j <= nf1.hi[1]; ++j)
        for (int i = nf1.lo[0]; i <= nf1.hi[0]; ++i)
          a(i, j) = old1.contains(i, j) ? o1(i, j) : c0(crsn(i), crsn(j));
    }

    // niveau 2 : tag gradient (bords/filaments les plus raides) sur le NOUVEAU
    // niveau 1, seuil relatif plus haut. Boucle a l'interieur strict de nf1
    // (pas de ghost) -> les cellules taguees sont deja nesting-compatibles.
    int k0 = nf1.hi[0], k1 = nf1.lo[0], l0 = nf1.hi[1], l1 = nf1.lo[1];
    {
      const ConstArray4 a = nU1.const_array();
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
    if (k1 < k0) {  // pas de coeur tague : box centrale de secours
      nL2CI0 = nf1.lo[0] + nf1.nx() / 3; nL2CI1 = nf1.hi[0] - nf1.nx() / 3;
      nL2CJ0 = nf1.lo[1] + nf1.ny() / 3; nL2CJ1 = nf1.hi[1] - nf1.ny() / 3;
    } else {
      nL2CI0 = std::max(nf1.lo[0] + 1, k0 - buf2);
      nL2CI1 = std::min(nf1.hi[0] - 1, k1 + buf2);
      nL2CJ0 = std::max(nf1.lo[1] + 1, l0 - buf2);
      nL2CJ1 = std::min(nf1.hi[1] - 1, l1 + buf2);
    }
    Box2D nf2{{2 * nL2CI0, 2 * nL2CJ0}, {2 * nL2CI1 + 1, 2 * nL2CJ1 + 1}};

    // transfert niveau 2 (ancien fin la ou il existe, sinon injection niveau 1)
    Fab2D nU2(nf2, 1, 1), na2(nf2, 3, 1);
    {
      const ConstArray4 c1 = nU1.const_array();
      const ConstArray4 o2 = L[2].U.const_array();
      const Box2D old2 = L[2].U.box();
      Array4 a = nU2.array();
      for (int j = nf2.lo[1]; j <= nf2.hi[1]; ++j)
        for (int i = nf2.lo[0]; i <= nf2.hi[0]; ++i)
          a(i, j) = old2.contains(i, j) ? o2(i, j) : c1(crsn(i), crsn(j));
    }

    L[1].U = std::move(nU1); a1 = std::move(na1);
    L[2].U = std::move(nU2); a2 = std::move(na2);
    L[0].rCI0 = nL1CI0; L[0].rCI1 = nL1CI1; L[0].rCJ0 = nL1CJ0; L[0].rCJ1 = nL1CJ1;
    L[1].rCI0 = nL2CI0; L[1].rCI1 = nL2CI1; L[1].rCJ0 = nL2CJ0; L[1].rCJ1 = nL2CJ1;
  };

  std::ofstream boxes(out + "/boxes.csv");
  boxes << "frame,x1lo,x1hi,y1lo,y1hi,x2lo,x2hi,y2lo,y2hi\n";
  auto dump = [&](int frame) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "/dens_c_%04d.txt", frame);
    std::ofstream fc(out + nm);
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) fc << L[0].U(i, j) << (i + 1 < nc ? ' ' : '\n');
    const Box2D b1 = L[1].U.box(), b2 = L[2].U.box();
    std::snprintf(nm, sizeof(nm), "/dens_1_%04d.txt", frame);
    std::ofstream f1(out + nm);
    for (int j = b1.lo[1]; j <= b1.hi[1]; ++j)
      for (int i = b1.lo[0]; i <= b1.hi[0]; ++i)
        f1 << L[1].U(i, j) << (i < b1.hi[0] ? ' ' : '\n');
    std::snprintf(nm, sizeof(nm), "/dens_2_%04d.txt", frame);
    std::ofstream f2(out + nm);
    for (int j = b2.lo[1]; j <= b2.hi[1]; ++j)
      for (int i = b2.lo[0]; i <= b2.hi[0]; ++i)
        f2 << L[2].U(i, j) << (i < b2.hi[0] ? ' ' : '\n');
    boxes << frame << ',' << b1.lo[0] * dxf1 << ',' << (b1.hi[0] + 1) * dxf1 << ','
          << b1.lo[1] * dyf1 << ',' << (b1.hi[1] + 1) * dyf1 << ','
          << b2.lo[0] * dxf2 << ',' << (b2.hi[0] + 1) * dxf2 << ','
          << b2.lo[1] * dyf2 << ',' << (b2.hi[1] + 1) * dyf2 << '\n';
  };

  compute_aux();
  const double M0 = mass();
  double dt = 0.4 * dxc / vmax();
  const int snap = std::max(1, nsteps / 30);
  std::printf("diocotron AMR 3 niveaux (regrid imbrique) nc=%d dt=%.2e\n", nc, dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      const Box2D b1 = L[1].U.box(), b2 = L[2].U.box();
      std::printf("  s=%4d  L1=[%d..%d]x[%d..%d] L2=[%d..%d]x[%d..%d] drift=%.2e\n",
                  s, b1.lo[0], b1.hi[0], b1.lo[1], b1.hi[1], b2.lo[0], b2.hi[0],
                  b2.lo[1], b2.hi[1], std::fabs(mass() - M0));
    }
    if (s == nsteps) break;
    if (s > 0 && s % 20 == 0) regrid();  // remaillage imbrique dynamique
    sync_down();                          // sync ascendant pour Poisson
    compute_aux();
    amr_step_multilevel(model, L, dom, dt);
    if (s % 20 == 0) dt = 0.4 * dxc / vmax();
  }
  std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(),
              frame, std::fabs(mass() - M0));
  return 0;
}
