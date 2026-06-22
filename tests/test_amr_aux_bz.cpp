// Chantier "Aux extensible", increment AMR : LECTURE et ALLOCATION width-aware du canal aux
// dans les chemins AMR. Avant ce pas, l'AMR allouait l'aux a 3 comp et lisait load_aux(...) a
// 3 comp EN DUR : un modele declarant n_aux=4 (lisant a.B_z dans sa source) ne voyait jamais
// B_z en AMR. On verifie ici, sur un modele jouet BzGrow (n_aux=4, flux nul, elliptic nul,
// source S = B_z * u) :
//   (A) COMPILE-CHECK : aux_comps<BzGrow>() == 4 -> les chemins AMR instancient bien
//       load_aux<aux_comps<Model>()> (la largeur du modele, pas 3 en dur).
//   (B) CHEMIN AMR DIRECT (advance_amr, 2 niveaux) sur un aux ALLOUE A 4 comp, B_z = c pose en
//       constante : la source AMR (mf_apply_source -> load_aux<4>) lit B_z. Avec flux nul,
//       u(t+dt) = u + dt*B_z*u sur grossier ET fin -> pas d'acces hors borne, B_z bien lu.
//   (C) BIT-IDENTITE : un modele de BASE (n_aux=3, advection) sur un aux 3 comp conserve la
//       masse grossiere sous un pas AMR (reflux + average_down), inchange par ce travail.
//   (D) END-TO-END AmrSystemCoupler : un systeme {BzGrow (n_aux=4), base (n_aux=3)} alloue le
//       canal aux PARTAGE a la largeur MAX (4) a chaque niveau ; B_z peuple a c ; le bloc qui
//       lit B_z croit (source = B_z u), le bloc de base sur le meme aux elargi reste inerte.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/amr_system_coupler.hpp>
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/mesh/layout/refinement.hpp>              // coarsen_index
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// Modele jouet : croissance pilotee par B_z. flux nul, pas de couplage elliptique (phi = 0).
// source S = B_z * u -> du/dt = B_z u. Declare n_aux=4 : LIT a.B_z.
struct BzGrow {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Modele de BASE (n_aux defaut = 3) : advection en x a vitesse v (E x B-like, mais vitesse
// constante portee par le flux, aux non lue). Source nulle.
struct AdvectX {
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

static_assert(PhysicalModel<BzGrow> && PhysicalModel<AdvectX>);
// (A) COMPILE-CHECK : la largeur aux lue par les chemins AMR pour BzGrow est 4 (n_aux),
// pour AdvectX 3 (defaut). C'est exactement le NComp passe a load_aux<aux_comps<Model>()>
// dans mf_apply_source / compute_face_fluxes / max_wave_speed AMR.
static_assert(aux_comps<BzGrow>() == 4, "BzGrow declare n_aux = 4");
static_assert(aux_comps<AdvectX>() == kAuxBaseComps, "AdvectX reste au contrat de base (3)");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int NC = 16;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba_coarse(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dyc = geom.dy();
  const Real c = Real(0.7);  // B_z constant

  // patch fin sur les cellules grossieres [4..11]^2 -> box fine {{8,8},{23,23}}.
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_fine(std::vector<Box2D>{fbox});

  // --- (B) chemin AMR direct (advance_amr) sur un aux ALLOUE A 4 comp, B_z = c -----------
  {
    const Real u0 = Real(2);
    MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm, 1, 2);
    Uc.set_val(u0);
    Uf.set_val(u0);

    // aux ALLOUE A 4 COMP (aux_comps<BzGrow>()) : phi/grad = 0, B_z = c (comp 3). Sans cette
    // largeur, load_aux<4> dans la source AMR lirait hors borne ; ici la place existe.
    MultiFab auxc(ba_coarse, dm, aux_comps<BzGrow>(), 1);
    MultiFab auxf(ba_fine, dm, aux_comps<BzGrow>(), 1);
    chk(auxc.ncomp() == 4 && auxf.ncomp() == 4, "amr_aux_allocated_width_4");
    auxc.set_val(Real(0));
    auxf.set_val(Real(0));
    auto set_bz = [&](MultiFab& A) {
      for (int li = 0; li < A.local_size(); ++li) {
        Fab2D& f = A.fab(li);
        const Box2D g = f.grown_box();
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            f(i, j, 3) = c;  // B_z = comp 3
      }
    };
    set_bz(auxc);
    set_bz(auxf);

    std::vector<AmrLevelMP> L;
    L.push_back(AmrLevelMP{std::move(Uc), &auxc, dxc, dyc});
    L.push_back(AmrLevelMP{std::move(Uf), &auxf, dxc / 2, dyc / 2});

    const Real dt = Real(0.05);
    // flux nul -> mf_advance_faces = no-op ; mf_apply_source ajoute dt*B_z*u (Euler avant
    // par sous-pas : le fin fait r=2 sous-pas de dt/2). Grossier : 1 pas dt. Fin : 2 pas dt/2.
    advance_amr(BzGrow{}, L, dom, dt);

    // grossier : u <- u + dt*c*u = u0*(1 + dt*c). Verifie sur une cellule NON couverte
    // par le fin (coin (0,0)), pour eviter l'ecrasement average_down par le fin.
    const Real expect_c = u0 * (Real(1) + dt * c);
    // fin : deux sous-pas Euler avant de dt/2 -> u0*(1 + (dt/2)*c)^2.
    const Real half = dt / 2;
    const Real expect_f = u0 * (Real(1) + half * c) * (Real(1) + half * c);
    {
      device_fence();
      const ConstArray4 uc = L[0].U.fab(0).const_array();
      const ConstArray4 uf = L[1].U.fab(0).const_array();
      chk(std::fabs(uc(0, 0, 0) - expect_c) < Real(1e-12), "amr_coarse_source_reads_Bz");
      chk(std::fabs(uf(fbox.lo[0], fbox.lo[1], 0) - expect_f) < Real(1e-12),
          "amr_fine_source_reads_Bz");
      // sanity : sans lecture de B_z (B_z ignore), u serait reste a u0 -> on rejette ce cas.
      chk(std::fabs(uc(0, 0, 0) - u0) > Real(1e-3), "amr_Bz_actually_read");
    }
  }

  // --- (C) bit-identite : modele de BASE (3 comp) sous un pas AMR -> masse conservee --------
  {
    MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm, 1, 2);
    // motif non trivial mais coherent grossier/fin : u = 1 + 0.25*(i_coarse pair?1:-1).
    auto fillp = [](MultiFab& U, int ratio) {
      for (int li = 0; li < U.local_size(); ++li) {
        Array4 a = U.fab(li).array();
        const Box2D g = U.fab(li).grown_box();
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
            const int ci = (ratio == 1) ? i : coarsen_index(i, ratio);
            a(i, j, 0) = Real(1) + Real(0.25) * ((ci & 1) ? Real(1) : Real(-1));
          }
      }
    };
    fillp(Uc, 1);
    fillp(Uf, 2);
    MultiFab auxc(ba_coarse, dm, 3, 1), auxf(ba_fine, dm, 3, 1);  // largeur 3 (base)
    auxc.set_val(Real(0));
    auxf.set_val(Real(0));
    chk(auxc.ncomp() == 3, "base_aux_width_3");

    std::vector<AmrLevelMP> L;
    L.push_back(AmrLevelMP{std::move(Uc), &auxc, dxc, dyc});
    L.push_back(AmrLevelMP{std::move(Uf), &auxf, dxc / 2, dyc / 2});

    const Real m0 = sum(L[0].U) * dxc * dyc;
    advance_amr(AdvectX{Real(1)}, L, dom, Real(0.01));
    const Real m1 = sum(L[0].U) * dxc * dyc;
    chk(std::fabs(m1 - m0) < Real(1e-12), "base_model_mass_conserved_amr");
  }

  // --- (D) end-to-end AmrSystemCoupler : canal aux partage a la largeur MAX (4) -------------
  {
    using BzBlk = EquationBlock<BzGrow, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    using BaseBlk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

    const Real u0 = Real(2);
    MultiFab UgC(ba_coarse, dm, 1, 2), UgF(ba_fine, dm, 1, 2);  // bloc BzGrow
    MultiFab UbC(ba_coarse, dm, 1, 2), UbF(ba_fine, dm, 1, 2);  // bloc base
    UgC.set_val(u0);
    UgF.set_val(u0);
    UbC.set_val(Real(1));
    UbF.set_val(Real(1));

    BzBlk g{"grow", BzGrow{}, UgC, BCRec{}};
    BaseBlk b{"base", AdvectX{Real(0)}, UbC, BCRec{}};  // v=0 : pas de transport, source nulle
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

    // charges nulles -> f = 0 -> phi = 0 (B_z independant de l'elliptique).
    ChargeDensityRhs charge{{{Real(0), 0}, {Real(0), 0}}};
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, charge, std::move(bl));

    // canal aux PARTAGE : largeur = max(aux_comps) = max(4, 3) = 4, a CHAQUE niveau.
    chk(sim.aux_ncomp() == 4, "shared_aux_width_is_max_amr");
    chk(sim.aux(0).ncomp() == 4 && sim.aux(1).ncomp() == 4, "shared_aux_4_at_each_level");

    sim.solve_fields();  // phi = 0 ; field_postprocess n'ecrit que comp 0..2

    // peuple B_z = c sur le canal partage (comp 3) a CHAQUE niveau (solve_fields preserve B_z).
    for (int k = 0; k < sim.nlev(); ++k) {
      MultiFab& A = sim.aux(k);
      for (int li = 0; li < A.local_size(); ++li) {
        Fab2D& f = A.fab(li);
        const Box2D gb = f.grown_box();
        for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
          for (int i = gb.lo[0]; i <= gb.hi[0]; ++i)
            f(i, j, 3) = c;
      }
    }

    const Real mg0 = sim.mass(0), mb0 = sim.mass(1);
    sim.step(Real(0.05));  // tout explicite ; phi gele (OncePerStep)

    const Real mg1 = sim.mass(0), mb1 = sim.mass(1);
    // bloc BzGrow : source = B_z u > 0 -> masse grossiere CROIT (B_z bien lu en AMR).
    chk(mg1 > mg0 + Real(1e-6), "amr_system_BzGrow_block_grew");
    // bloc base sur le MEME aux elargi : v=0 (pas de flux), source nulle -> masse inchangee
    // (la comp 3 du canal partage est ignoree par AdvectX, n_aux=3).
    chk(std::fabs(mb1 - mb0) < Real(1e-12), "amr_system_base_block_inert");
  }

  if (fails == 0)
    std::printf("test_amr_aux_bz: OK\n");
  return fails == 0 ? 0 : 1;
}
