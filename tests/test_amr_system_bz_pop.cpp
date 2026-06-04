// Peuplement de B_z PAR NIVEAU dans le coupleur AMR de systeme (AmrSystemCoupler).
//
// PR #37 a rendu le canal aux extensible width-aware sur les chemins AMR : un bloc declarant
// n_aux=4 dispose de la PLACE pour B_z a chaque niveau, mais B_z restait a 0 (sauf propagation
// coarse->fine par coupler_inject_aux_mb). Ce test verifie le nouveau mecanisme : un B_z fourni
// par l'utilisateur (std::function<Real(Real,Real)>, comme le bz_ du SystemAssembler mono-niveau)
// est POSE sur la composante B_z (indice kAuxBaseComps) du canal aux partage de CHAQUE niveau,
// echantillonne aux centres de cellule DU NIVEAU (chaque niveau a sa geometrie / dx).
//
//   (A) PEUPLEMENT PAR NIVEAU, ECHANTILLONNE A LA RESOLUTION DU NIVEAU : avec un B_z spatialement
//       variable bz(x,y)=1+x, on verifie que sim.aux(0) (grossier) et sim.aux(1) (fin) portent
//       chacun B_z = bz(x_cell_DU_NIVEAU). Sur le fin, deux cellules contenues dans une meme
//       cellule grossiere ont des B_z DISTINCTS (echantillonnage fin), ce qu'une simple injection
//       coarse->fine (constante par cellule grossiere) ne produirait pas.
//   (B) PRESERVATION par solve_fields : phi resolu (ici 0), field_postprocess n'ecrit que comp
//       0..2 ; B_z (comp 3) inchange a tous les niveaux apres solve_fields.
//   (C) LA SOURCE LIT B_z SUR LE GROSSIER ET LE FIN : source S = B_z*u, flux nul. Apres un pas,
//       une cellule grossiere NON couverte croit de u0*(1+dt*B_z_grossier), une cellule fine de
//       u0*(1+(dt/2)*B_z_fin)^2 (r=2 sous-pas), chacune avec le B_z DE SON NIVEAU. C'est la
//       verification fonctionnelle demandee : le residu/source = B_z*u correct par niveau.
//   (D) SETTER set_bz : poser B_z apres construction donne le meme resultat que par le ctor.
//   (E) GARDE / BIT-IDENTITE : sans bz fourni, la composante B_z reste 0 (comportement historique).

#include <adc/core/coupled_system.hpp>
#include <adc/core/physical_model.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/amr_system_coupler.hpp>
#include <adc/coupling/elliptic_rhs.hpp>  // ChargeDensityRhs
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

using namespace adc;

// Croissance pilotee par B_z : flux nul, elliptique nul (phi=0), source S = B_z*u. Lit a.B_z
// -> declare n_aux=4. du/dt = B_z u, Euler avant par sous-pas.
struct BzGrowPop {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const { return State{a.B_z * u[0]}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Bloc de base (n_aux defaut = 3) : advection en x, ne lit pas l'aux, source nulle.
struct AdvectXPop {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real v = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? v * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return std::fabs(v); }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrowPop> && PhysicalModel<AdvectXPop>);
static_assert(aux_comps<BzGrowPop>() == 4, "BzGrowPop declare n_aux = 4");
static_assert(aux_comps<AdvectXPop>() == kAuxBaseComps, "AdvectXPop reste au contrat de base");

namespace {

// Geometrie commune des tests : domaine 16x16 sur [0,1]^2, patch fin {{8,8},{23,23}}
// (couvre les cellules grossieres [4..11]^2). Le coin (0,0) grossier est NON couvert.
constexpr int NC = 16;

// B_z spatialement variable : echantillonner a la resolution du niveau change la valeur.
Real bz_field(Real x, Real /*y*/) { return Real(1) + x; }

// Lit la composante B_z (comp kAuxBaseComps) d'un MultiFab a la cellule (i,j) du fab 0.
Real read_bz(const MultiFab& A, int i, int j) {
  return A.fab(0).const_array()(i, j, kAuxBaseComps);
}

}  // namespace

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba_coarse(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dyc = geom.dy();
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_fine(std::vector<Box2D>{fbox});

  using BzBlk = EquationBlock<BzGrowPop, FirstOrder, ExplicitTime<SSPRK2, 1>>;
  using BaseBlk = EquationBlock<AdvectXPop, FirstOrder, ExplicitTime<SSPRK2, 1>>;

  // Fabrique : construit un coupleur frais (etats remis). bz_user vide => garde (E).
  auto build = [&](std::function<Real(Real, Real)> bz_user, bool use_setter, Real u0g) {
    MultiFab UgC(ba_coarse, dm, 1, 2), UgF(ba_fine, dm, 1, 2);
    MultiFab UbC(ba_coarse, dm, 1, 2), UbF(ba_fine, dm, 1, 2);
    UgC.set_val(u0g);     UgF.set_val(u0g);
    UbC.set_val(Real(1)); UbF.set_val(Real(1));

    BzBlk g{"grow", BzGrowPop{}, UgC, BCRec{}};
    BaseBlk b{"base", AdvectXPop{Real(0)}, UbC, BCRec{}};  // v=0 : inerte
    CoupledSystem system{g, b};

    auto make = [&](MultiFab&& U, Real dx, Real dy) {
      return AmrLevelMP{std::move(U), nullptr, dx, dy};
    };
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make(std::move(UgC), dxc, dyc));
    bl.back().push_back(make(std::move(UgF), dxc / 2, dyc / 2));
    bl.emplace_back();
    bl.back().push_back(make(std::move(UbC), dxc, dyc));
    bl.back().push_back(make(std::move(UbF), dxc / 2, dyc / 2));

    ChargeDensityRhs charge{{{Real(0), 0}, {Real(0), 0}}};  // charges nulles -> phi = 0
    using Sim = AmrSystemCoupler<decltype(system), ChargeDensityRhs>;
    auto sim = std::make_unique<Sim>(
        system, geom, ba_coarse, BCRec{}, charge, std::move(bl),
        Periodicity{true, true}, /*replicated_coarse=*/true,
        PoissonCadence::OncePerStep, std::function<bool(Real, Real)>{},
        use_setter ? std::function<Real(Real, Real)>{} : bz_user);
    if (use_setter) sim->set_bz(bz_user);
    return sim;
  };

  // --- (A) peuplement par niveau, echantillonne a la resolution du niveau ------------------
  {
    auto sim = build(bz_field, /*use_setter=*/false, /*u0g=*/Real(2));
    chk(sim->aux_ncomp() == 4, "shared_aux_width_max_4");

    // grossier : B_z(i) = bz(x_cell_grossier(i)). Geometrie grossiere = geom.
    bool coarse_ok = true;
    for (int i = 0; i < NC; ++i)
      if (std::fabs(read_bz(sim->aux(0), i, 4) - bz_field(geom.x_cell(i), 0)) > Real(1e-12))
        coarse_ok = false;
    chk(coarse_ok, "coarse_Bz_sampled_at_coarse_centers");

    // fin : B_z(I) = bz(x_cell_fin(I)) avec la geometrie raffinee (dx/2). Deux cellules fines
    // 2*ci et 2*ci+1 (dans la meme cellule grossiere ci) ont des B_z DISTINCTS -> echantillonnage
    // a la resolution fine, pas une injection constante par cellule grossiere.
    const Geometry gf = geom.refine(2);
    bool fine_ok = true;
    for (int I = fbox.lo[0]; I <= fbox.hi[0]; ++I) {
      const Real got = read_bz(sim->aux(1), I, fbox.lo[1]);
      if (std::fabs(got - bz_field(gf.x_cell(I), 0)) > Real(1e-12)) fine_ok = false;
    }
    chk(fine_ok, "fine_Bz_sampled_at_fine_centers");

    // les deux cellules fines d'une meme cellule grossiere different (B_z variable, fin-echantillonne)
    const Real bz_lo = read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]);
    const Real bz_hi = read_bz(sim->aux(1), fbox.lo[0] + 1, fbox.lo[1]);
    chk(std::fabs(bz_lo - bz_hi) > Real(1e-6), "fine_Bz_resolves_subcoarse_variation");

    // --- (B) preservation par solve_fields (field_postprocess n'ecrit que comp 0..2) --------
    sim->solve_fields();
    chk(std::fabs(read_bz(sim->aux(0), 7, 7) - bz_field(geom.x_cell(7), 0)) < Real(1e-12),
        "coarse_Bz_preserved_after_solve_fields");
    chk(std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]) - bz_lo) < Real(1e-12),
        "fine_Bz_preserved_after_solve_fields");

    // --- (C) la source lit B_z sur le GROSSIER et le FIN ------------------------------------
    const Real u0 = Real(2), dt = Real(0.05);
    sim->step(dt);  // tout explicite, phi gele (OncePerStep)
    device_fence();

    // cellule grossiere NON couverte (0,0) : u <- u0*(1 + dt*B_z(0,0)).
    const Real bzc = bz_field(geom.x_cell(0), 0);
    const Real expect_c = u0 * (Real(1) + dt * bzc);
    const Real uc = sim->levels(0)[0].U.fab(0).const_array()(0, 0, 0);
    chk(std::fabs(uc - expect_c) < Real(1e-12), "coarse_source_reads_levelwise_Bz");
    chk(std::fabs(uc - u0) > Real(1e-3), "coarse_Bz_actually_read");

    // cellule fine (fbox.lo) : r=2 sous-pas Euler avant de dt/2 -> u0*(1+(dt/2)*B_z_fin)^2.
    const Real bzf = bz_field(gf.x_cell(fbox.lo[0]), 0);
    const Real half = dt / 2;
    const Real expect_f = u0 * (Real(1) + half * bzf) * (Real(1) + half * bzf);
    const Real uf = sim->levels(0)[1].U.fab(0).const_array()(fbox.lo[0], fbox.lo[1], 0);
    chk(std::fabs(uf - expect_f) < Real(1e-12), "fine_source_reads_levelwise_Bz");
    // grossier et fin utilisent des B_z DIFFERENTS (bzc != bzf) : verifie qu'on ne lit pas le
    // meme B_z partout (sinon le peuplement par niveau ne serait pas effectif).
    chk(std::fabs(bzc - bzf) > Real(1e-6), "coarse_and_fine_Bz_differ");
  }

  // --- (D) setter set_bz : meme resultat que par le ctor -----------------------------------
  {
    auto sim = build(bz_field, /*use_setter=*/true, /*u0g=*/Real(2));
    const Geometry gf = geom.refine(2);
    chk(std::fabs(read_bz(sim->aux(0), 3, 3) - bz_field(geom.x_cell(3), 0)) < Real(1e-12),
        "set_bz_populates_coarse");
    chk(std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]) -
                  bz_field(gf.x_cell(fbox.lo[0]), 0)) < Real(1e-12),
        "set_bz_populates_fine");
  }

  // --- (E) garde : sans bz fourni, la composante B_z reste 0 (bit-identite historique) ------
  {
    auto sim = build({}, /*use_setter=*/false, /*u0g=*/Real(2));
    // le canal reste alloue a 4 (un bloc declare n_aux=4) mais comp 3 = 0 (MultiFab non
    // initialise par fill_bz -> valeur par defaut 0 de l'allocation).
    sim->solve_fields();
    chk(std::fabs(read_bz(sim->aux(0), 5, 5)) < Real(1e-30), "no_bz_coarse_stays_zero");
    chk(std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1])) < Real(1e-30),
        "no_bz_fine_stays_zero");
  }

  if (fails == 0) std::printf("test_amr_system_bz_pop: OK\n");
  return fails == 0 ? 0 : 1;
}
