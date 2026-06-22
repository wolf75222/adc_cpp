// Parite spatiale AmrSystem <-> System : le coeur spatial du chemin AMR
// (compute_face_fluxes<Limiter, Flux> + recon_prim, consomme par le reflux de
// advance_amr) reproduit EXACTEMENT le residu d'assemble_rhs<Limiter, Flux> du
// chemin System, sous reconstruction PRIMITIVE et flux HLLC puis Roe. C'est la
// preuve que la facade raffinee accepte les memes parametres de schema que System
// et les applique a chaque niveau/patch.
//
// Deux parties :
//   A. Coeur spatial (bit-identique) : div(compute_face_fluxes) == assemble_rhs,
//      pour HLLC et Roe en reconstruction primitive (minmod, ordre 2). Et le
//      primitif DIFFERE du conservatif (la reconstruction joue bien).
//   B. Plomberie moteur : advance_amr (un niveau) transmet recon_prim jusqu'a
//      compute_face_fluxes (primitif != conservatif apres un pas), reste fini et
//      conserve la masse sur un etat lisse periodique.

#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, CompressibleFlux, NoSource, ChargeDensity
#include <adc/numerics/numerical_flux.hpp>      // HLLCFlux, RoeFlux
#include <adc/numerics/reconstruction.hpp>      // Minmod
#include <adc/numerics/spatial_operator.hpp>    // assemble_rhs, compute_face_fluxes, load_state
#include <adc/numerics/time/amr_reflux_mf.hpp>  // advance_amr, AmrLevelMP

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/boundary/fill_boundary.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>  // norm_inf
#include <adc/mesh/storage/multifab.hpp>  // sum
#include <adc/mesh/boundary/physical_bc.hpp>

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

  const int n = 32;
  const double L = 1.0;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, L, 0.0, L};
  const double dx = geom.dx(), dy = geom.dy();
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const BCRec bc;  // periodique

  // Euler compressible pur (sans source) : le residu vient entierement du flux
  // hyperbolique reconstruit -> isole la parite reconstruction x flux Riemann.
  using Model = CompositeModel<CompressibleFlux, NoSource, ChargeDensity>;
  const Model model{CompressibleFlux{1.4}, NoSource{}, ChargeDensity{1.0}};

  MultiFab U0(ba, dm, 4, 2), aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  {  // bulle de densite + champ de vitesse doux (etat lisse, positif)
    Array4 a = U0.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i) - 0.5, y = geom.y_cell(j) - 0.5;
      const double rho = 1.0 + 0.4 * std::exp(-(x * x + y * y) / 0.02);
      a(i, j, 0) = rho;
      a(i, j, 1) = 0.2 * rho * std::sin(2 * kPi * geom.x_cell(i));
      a(i, j, 2) = 0.1 * rho * std::cos(2 * kPi * geom.y_cell(j));
      const double ke = 0.5 * (a(i, j, 1) * a(i, j, 1) + a(i, j, 2) * a(i, j, 2)) / rho;
      a(i, j, 3) = 1.0 / (1.4 - 1.0) + ke;
    });
  }

  // residu d'assemble_rhs (chemin System) sur les cellules valides.
  auto rhs_system = [&](auto flux_tag, bool prim, MultiFab& R) {
    using Flux = decltype(flux_tag);
    MultiFab U = U0;
    fill_ghosts(U, dom, bc);
    assemble_rhs<Minmod, Flux>(model, U, aux, geom, R, prim);
  };
  // residu reconstitue du chemin AMR : -div des flux de face de compute_face_fluxes
  // (NoSource -> pas de terme S). Memes (Limiter, Flux, recon_prim) que System.
  auto rhs_amr = [&](auto flux_tag, bool prim, MultiFab& R) {
    using Flux = decltype(flux_tag);
    MultiFab U = U0;
    fill_ghosts(U, dom, bc);
    MultiFab Fx(BoxArray(std::vector<Box2D>{xface_box(dom)}), dm, 4, 0);
    MultiFab Fy(BoxArray(std::vector<Box2D>{yface_box(dom)}), dm, 4, 0);
    compute_face_fluxes<Minmod, Flux>(model, U, aux, Fx, Fy, dx, dy, prim);
    Array4 r = R.fab(0).array();
    const ConstArray4 fx = Fx.fab(0).const_array(), fy = Fy.fab(0).const_array();
    for_each_cell(dom, [=] ADC_HD(int i, int j) {
      for (int c = 0; c < 4; ++c)
        r(i, j, c) = -((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy);
    });
  };

  auto maxdiff = [&](const MultiFab& A, const MultiFab& B) {
    double d = 0;
    const ConstArray4 a = A.fab(0).const_array(), b = B.fab(0).const_array();
    for (int c = 0; c < 4; ++c)
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          d = std::fmax(d, std::fabs(a(i, j, c) - b(i, j, c)));
    return d;
  };

  // --- Partie A : coeur spatial AMR == coeur spatial System (bit-identique) ---
  {
    MultiFab Rs(ba, dm, 4, 0), Ra(ba, dm, 4, 0), Rc(ba, dm, 4, 0);

    // HLLC + primitif : div(face fluxes) == assemble_rhs, exactement.
    rhs_system(HLLCFlux{}, true, Rs);
    rhs_amr(HLLCFlux{}, true, Ra);
    chk(maxdiff(Rs, Ra) < 1e-13, "HLLC+primitif : div(compute_face_fluxes) == assemble_rhs");
    chk(norm_inf(Rs) > 1e-6, "HLLC+primitif : residu non trivial");

    // le primitif change le residu vs le conservatif (la reconstruction joue).
    rhs_system(HLLCFlux{}, false, Rc);
    chk(maxdiff(Rs, Rc) > 1e-9, "HLLC : recon primitif != conservatif");

    // Roe + primitif : meme parite bit-identique.
    rhs_system(RoeFlux{}, true, Rs);
    rhs_amr(RoeFlux{}, true, Ra);
    chk(maxdiff(Rs, Ra) < 1e-13, "Roe+primitif : div(compute_face_fluxes) == assemble_rhs");
    chk(norm_inf(Rs) > 1e-6, "Roe+primitif : residu non trivial");
  }

  // --- Partie B : recon_prim transmis a travers advance_amr (moteur AMR) ---
  {
    const double dt = 1e-3;
    const double sum0 = sum(U0);
    auto one_step = [&](bool prim) {
      std::vector<AmrLevelMP> levels;
      levels.push_back(AmrLevelMP{U0, &aux, dx, dy});  // un seul niveau (grossier)
      advance_amr<Minmod, HLLCFlux>(model, levels, dom, dt, Periodicity{true, true},
                                    /*coarse_replicated=*/true, /*recon_prim=*/prim);
      return std::move(levels[0].U);
    };
    MultiFab Up = one_step(true);
    MultiFab Uc = one_step(false);

    chk(std::isfinite(norm_inf(Up)) && norm_inf(Up) < 1e6, "advance_amr primitif : etat fini");
    chk(std::fabs(sum(Up) - sum0) < 1e-9, "advance_amr primitif : masse conservee (periodique)");
    chk(std::fabs(sum(Uc) - sum0) < 1e-9, "advance_amr conservatif : masse conservee");
    // recon_prim a bien traverse advance_amr -> compute_face_fluxes : les deux pas different.
    chk(maxdiff(Up, Uc) > 1e-9, "advance_amr : recon_prim plumbing (primitif != conservatif)");
  }

  if (fails == 0)
    std::printf("OK test_amr_spatial_parity\n");
  return fails == 0 ? 0 : 1;
}
