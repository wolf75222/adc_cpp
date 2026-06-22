// Le constructeur de fermetures de bloc (adc/runtime/block_builder.hpp) est extrait de System::Impl
// pour etre instanciable HORS de l'unite system.cpp : c'est la fondation du backend AOT (un modele
// genere par le DSL, compile ahead-of-time, entre dans le System par le chemin de PRODUCTION
// template -- HLLC, ordre 2 -- et non plus par le seul chemin hote virtuel du bloc dynamique).
//
// Ce test exerce le chemin EXTERNE : on assemble un CompositeModel arbitraire et un GridContext a la
// main (sans System), puis on verifie que make_block / make_max_speed / make_poisson_rhs produisent
// exactement le residu / la vitesse d'onde / le second membre de Poisson du chemin direct, et que
// l'avance SSPRK2 conserve la masse. Si ca compile et passe, un .so genere peut faire de meme.
#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, NoSource, GravityForce, GravityCoupling
#include <adc/physics/fluids/euler.hpp>   // Euler (brique hyperbolique compressible)
#include <adc/runtime/builders/block/block_builder.hpp>

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique

  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 1);
  aux.set_val(0.0);  // grad phi nul ici (on teste le cablage, pas la physique du couplage)

  // euler_poisson compile : Euler + force de gravite + couplage GravityCoupling. Modele arbitraire
  // assemble a la main, comme le ferait une unite generee AOT.
  using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;
  Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};

  auto init = [&]() {  // bulle de densite + vitesse douce
    Array4 a = U.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i) - 0.5, y = geom.y_cell(j) - 0.5;
      const double rho = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
      a(i, j, 0) = rho;
      a(i, j, 1) = 0.1 * rho * std::sin(2 * kPi * geom.x_cell(i));
      a(i, j, 2) = 0.0;
      a(i, j, 3) = 1.0 / (1.4 - 1.0) + 0.5 * a(i, j, 1) * a(i, j, 1) / rho;
    });
  };
  init();

  const GridContext ctx{dom, bc, geom, &aux};

  // (1) make_block (minmod + HLLC, primitif) : son residu == assemble_rhs direct du meme schema.
  BlockClosures clo = make_block(model, "minmod", "hllc", ctx, /*imex=*/false, /*recon_prim=*/true);
  MultiFab R1(ba, dm, 4, 0), R2(ba, dm, 4, 0);
  clo.rhs_into(U, R1);
  fill_ghosts(U, dom, bc);
  assemble_rhs<Minmod, HLLCFlux>(model, U, aux, geom, R2, /*recon_prim=*/true);
  double dres = 0, nrm = 0;
  for (int c = 0; c < 4; ++c) {
    const ConstArray4 r1 = R1.fab(0).const_array(), r2 = R2.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        dres = std::fmax(dres, std::fabs(r1(i, j, c) - r2(i, j, c)));
        nrm = std::fmax(nrm, std::fabs(r2(i, j, c)));
      }
  }
  chk(dres < 1e-14, "rhs_into == assemble_rhs direct (HLLC ordre 2)");
  chk(nrm > 1e-6, "residu non trivial");

  // (2) make_max_speed == max_wave_speed_mf direct.
  auto max_speed = make_max_speed(model, ctx);
  chk(std::fabs(max_speed(U) - max_wave_speed_mf(model, U, aux)) < 1e-14, "make_max_speed");

  // (3) make_poisson_rhs : rhs += elliptic_rhs(U) cellule par cellule.
  auto poisson_rhs = make_poisson_rhs(model);
  MultiFab rhs(ba, dm, 1, 0);
  rhs.set_val(0.0);
  poisson_rhs(U, rhs);
  double dpo = 0;
  {
    const ConstArray4 rr = rhs.fab(0).const_array(), u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        dpo =
            std::fmax(dpo, std::fabs(rr(i, j, 0) - model.elliptic_rhs(load_state<Model>(u, i, j))));
  }
  chk(dpo < 1e-14, "make_poisson_rhs == elliptic_rhs");

  // (4) l'avance SSPRK2 tourne (chemin de production) et conserve la masse sur un etat lisse.
  const double mass0 = sum(U);
  for (int s = 0; s < 10; ++s)
    clo.advance(U, 2e-3, 1);
  double mn = 1e300;
  {
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        mn = std::fmin(mn, u(i, j, 0));
  }
  chk(std::fabs(sum(U) - mass0) < 1e-9, "avance conserve la masse");
  chk(mn > 0.0 && std::isfinite(mn), "etat physique apres avance");

  if (fails == 0)
    std::printf("OK test_block_builder (seam AOT instanciable hors System)\n");
  return fails ? 1 : 0;
}
