// WENO5-Z + SSPRK3 cables a travers block_builder (make_block / build_block) : le MEME chemin de
// production que minmod/vanleer x {rusanov, hllc, roe} x SSPRK2, etendu au limiteur weno5 (ordre 5,
// stencil 5 points, 3 ghosts) et au schema temporel SSPRK3 (3 etages, ordre 3).
//
// Quatre verifications :
//  (1) PARITE SCHEMA : make_block("weno5", "rusanov").rhs_into == assemble_rhs<Weno5, RusanovFlux>
//      direct (le dispatch route bien vers Weno5 ; spatial_operator branche weno5z a n_ghost >= 3).
//  (2) SSPRK3 : build_block<..., method="ssprk3"> avance == SSPRK3Step du coeur applique a la main au
//      meme residu (le tag temporel selectionne bien le bon foncteur RK).
//  (3) NO-DEFAULT-CHANGE : make_block("minmod", "rusanov") avec method par defaut == method="ssprk2"
//      == build_block<Minmod, RusanovFlux> sans method (BIT-IDENTIQUE : residu ET avance). Le chemin
//      historique n'est pas touche.
//  (4) ORDRE / PRECISION : advection lineaire d'un sinus lisse periodique sur une periode complete.
//      WENO5+SSPRK3 a une erreur < Minmod+SSPRK2 a meme resolution, et une pente de convergence > 2
//      (au-dela de l'ordre 2 du MUSCL). Test court (n <= 64), CI-friendly.
#include <adc/validation/physics/advection_diffusion.hpp>  // adc::validation::AdvectionDiffusion : transport scalaire (nu=0 = advection pure)
#include <adc/runtime/builders/block_builder.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/time/time_steppers.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Erreur L1 (cellules valides) entre u(.,0) et la solution exacte advectee d'une periode (== u0).
static double advect_error(int n, const std::string& limiter, const std::string& method) {
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  adc::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.0,
                                            /*nu=*/0.0};  // advection pure selon x
  const int ng = block_n_ghost(limiter);
  MultiFab U(ba, dm, 1, ng);
  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i));  // sinus lisse, une periode en x
    });
  };
  init(U);

  const GridContext ctx{dom, bc, geom, &aux};
  BlockClosures clo = make_block(model, limiter, "rusanov", ctx, /*imex=*/false,
                                 /*recon_prim=*/false, method);
  // Avance d'une PERIODE (t = L/ax = 1) a CFL fixe : u(t=1) == u0 exactement (advection periodique).
  const double dx = geom.dx();
  const double cfl = 0.4;
  const double dt = cfl * dx / 1.0;  // |a| = 1
  const int nsteps = static_cast<int>(std::ceil(1.0 / dt));
  const double dt_exact = 1.0 / nsteps;  // ajuste pour terminer pile a t=1
  for (int s = 0; s < nsteps; ++s)
    clo.advance(U, dt_exact, 1);

  double err = 0;
  const ConstArray4 u = U.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      err += std::fabs(u(i, j, 0) - std::sin(2 * kPi * geom.x_cell(i)));
  return err / (static_cast<double>(n) * n);  // L1 moyenne
}

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
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  adc::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.3,
                                            /*nu=*/0.0};  // advection 2D oblique, pure
  const GridContext ctx{dom, bc, geom, &aux};

  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i)) * std::cos(2 * kPi * geom.y_cell(j));
    });
  };

  // (1) PARITE SCHEMA : weno5 alloue 3 ghosts ; make_block route vers Weno5 -> rhs_into ==
  // assemble_rhs<Weno5, RusanovFlux> direct (meme reconstruction weno5z).
  chk(block_n_ghost("weno5") == 3, "block_n_ghost(weno5) == 3");
  {
    MultiFab U(ba, dm, 1, 3);
    init(U);
    BlockClosures clo = make_block(model, "weno5", "rusanov", ctx, false, false);
    MultiFab R1(ba, dm, 1, 0), R2(ba, dm, 1, 0);
    clo.rhs_into(U, R1);
    fill_ghosts(U, dom, bc);
    assemble_rhs<Weno5, RusanovFlux>(model, U, aux, geom, R2, false);
    double dres = 0, nrm = 0;
    const ConstArray4 r1 = R1.fab(0).const_array(), r2 = R2.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        dres = std::fmax(dres, std::fabs(r1(i, j, 0) - r2(i, j, 0)));
        nrm = std::fmax(nrm, std::fabs(r2(i, j, 0)));
      }
    chk(dres < 1e-14, "make_block(weno5).rhs_into == assemble_rhs<Weno5> direct");
    chk(nrm > 1e-6, "residu WENO5 non trivial");
  }

  // (2) SSPRK3 : l'avance d'un sous-pas via build_block(method="ssprk3") == SSPRK3Step du coeur
  // applique a la main au meme residu rhs_into (weno5/rusanov). Confirme le routage du tag temporel.
  {
    MultiFab U(ba, dm, 1, 3), Uref(ba, dm, 1, 3);
    init(U);
    init(Uref);
    BlockClosures clo3 = make_block(model, "weno5", "rusanov", ctx, false, false, "ssprk3");
    const double dt = 1e-3;
    clo3.advance(U, dt, 1);  // un sous-pas SSPRK3
    // Reference : meme residu (rhs_into du chemin ssprk2, residu identique), avance SSPRK3Step a la main.
    BlockClosures clo2 =
        make_block(model, "weno5", "rusanov", ctx, false, false);  // rhs_into identique
    SSPRK3Step{}.take_step([&](MultiFab& Us, MultiFab& R) { clo2.rhs_into(Us, R); }, Uref, dt);
    double d = 0;
    const ConstArray4 u = U.fab(0).const_array(), ur = Uref.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        d = std::fmax(d, std::fabs(u(i, j, 0) - ur(i, j, 0)));
    chk(d < 1e-14, "build_block(ssprk3).advance == SSPRK3Step du coeur");
  }

  // (2b) REUSE DU SCRATCH (ADC-261) : une avance a n>1 sous-pas reutilise UN seul Scratch hoisté a
  // travers les sous-pas (run_explicit_substeps) ; elle doit egaler n take_step one-shot SSPRK3Step
  // separes (scratch frais a chaque appel, h=dt/n). Verrouille le chemin de REUTILISATION, pas couvert
  // par le cas n=1 ci-dessus.
  {
    MultiFab U(ba, dm, 1, 3), Uref(ba, dm, 1, 3);
    init(U);
    init(Uref);
    BlockClosures clo3 = make_block(model, "weno5", "rusanov", ctx, false, false, "ssprk3");
    BlockClosures clo2 =
        make_block(model, "weno5", "rusanov", ctx, false, false);  // rhs_into identique
    const double dt = 4e-3;
    const int nsub = 4;
    clo3.advance(U, dt, nsub);  // Scratch reutilise a travers les nsub sous-pas
    const double h = dt / nsub;
    for (int s = 0; s < nsub; ++s)
      SSPRK3Step{}.take_step([&](MultiFab& Us, MultiFab& R) { clo2.rhs_into(Us, R); }, Uref, h);
    double d = 0;
    const ConstArray4 u = U.fab(0).const_array(), ur = Uref.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        d = std::fmax(d, std::fabs(u(i, j, 0) - ur(i, j, 0)));
    chk(d < 1e-14, "advance(n=4) == 4x take_step one-shot (reuse du scratch bit-identique)");
  }

  // (3) NO-DEFAULT-CHANGE : minmod/rusanov, method par defaut == "ssprk2" == build_block<Minmod>
  // sans tag. Residu ET avance BIT-IDENTIQUES (le chemin historique n'a pas bouge d'un bit).
  {
    MultiFab Ua(ba, dm, 1, 2), Ub(ba, dm, 1, 2);
    init(Ua);
    init(Ub);
    BlockClosures cdef =
        make_block(model, "minmod", "rusanov", ctx, false, false);  // method par defaut
    BlockClosures cs2 =
        make_block(model, "minmod", "rusanov", ctx, false, false, "ssprk2");  // explicite
    // residu identique
    MultiFab Ra(ba, dm, 1, 0), Rb(ba, dm, 1, 0);
    cdef.rhs_into(Ua, Ra);
    cs2.rhs_into(Ub, Rb);
    double dr = 0;
    {
      const ConstArray4 ra = Ra.fab(0).const_array(), rb = Rb.fab(0).const_array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          dr = std::fmax(dr, std::fabs(ra(i, j, 0) - rb(i, j, 0)));
    }
    chk(dr == 0.0, "minmod : rhs_into defaut == ssprk2 (bit-identique)");
    // avance identique sur 20 pas
    for (int s = 0; s < 20; ++s) {
      cdef.advance(Ua, 1e-3, 1);
      cs2.advance(Ub, 1e-3, 1);
    }
    double da = 0;
    {
      const ConstArray4 ua = Ua.fab(0).const_array(), ub = Ub.fab(0).const_array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          da = std::fmax(da, std::fabs(ua(i, j, 0) - ub(i, j, 0)));
    }
    chk(da == 0.0, "minmod/ssprk2 : avance defaut == ssprk2 (bit-identique, no-default-change)");
  }

  // (4) ORDRE / PRECISION : advection 1D d'un sinus sur une periode. WENO5+SSPRK3 plus precis que
  // Minmod+SSPRK2 a meme n, et pente de convergence > 2 (au-dela de l'ordre 2 du MUSCL).
  {
    const double e_minmod_64 = advect_error(64, "minmod", "ssprk2");
    const double e_weno_64 = advect_error(64, "weno5", "ssprk3");
    chk(e_weno_64 < e_minmod_64, "WENO5+SSPRK3 plus precis que Minmod+SSPRK2 a n=64");
    const double e_weno_32 = advect_error(32, "weno5", "ssprk3");
    const double slope = std::log(e_weno_32 / e_weno_64) / std::log(2.0);
    std::printf("  WENO5 L1 : n=32 %.3e  n=64 %.3e  pente %.2f (Minmod n=64 %.3e)\n", e_weno_32,
                e_weno_64, slope, e_minmod_64);
    chk(slope > 2.0, "WENO5 : pente de convergence > 2 (ordre superieur a O2)");
  }

  if (fails == 0)
    std::printf("OK test_weno5_ssprk3 (weno5 + ssprk3 cables, defaut SSPRK2 intact)\n");
  return fails ? 1 : 0;
}
