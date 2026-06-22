// Garde-fou de coherence de LAYOUT d'AmrSystemCoupler (premier pas MINIMAL du capstone AMR
// multi-blocs : type AmrHierarchyLayout explicite + same_layout_or_throw, point 1 et 2 du
// design docs/AMR_MULTIBLOCK_DESIGN.md).
//
// L'aux est PARTAGE par niveau : tous les blocs doivent vivre sur EXACTEMENT la meme grille
// par niveau (BoxArray boites ET ordre, DistributionMapping, dx/dy, nombre de niveaux). L'ancien
// controle ne comparait que le NOMBRE de boites (.size()) ; ce test verifie la comparaison EXACTE.
//
// Partie A : le ctor JETTE sur tout layout incoherent entre blocs (boites differentes, ORDRE
//   different, dmap differente, dx different, nombre de niveaux different).
// Partie B : le ctor PASSE sur un layout strictement identique entre blocs.
// Partie C : le chemin MONO-BLOC est BIT-IDENTIQUE (dmax == 0) : un seul bloc concorde
//   trivialement avec lui-meme (boucle inter-blocs vide), donc le garde-fou ne change RIEN a
//   l'avance d'un bloc unique sur AMR.
// Partie D : AmrHierarchyLayout::from_levels extrait bien la grille (boites, dmap, dx/dy) qui
//   sert au garde-fou.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/amr_system_coupler.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>  // norm_inf, lincomb
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/layout/refinement.hpp>  // coarsen_index

#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace adc;

// modele d'advection scalaire minimal (calque de test_amr_system_coupler.cpp).
struct AdvectX {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

// remplit U (composante 0) par une fonction de l'indice GROSSIER (le fin echantillonne la meme
// fonction via coarsen_index) -> grossier et fin coherents a l'init.
template <class F>
static void fill_by_coarse_i(MultiFab& U, int ratio, F f) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D g = U.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const int ci = (ratio == 1) ? i : coarsen_index(i, ratio);
        a(i, j, 0) = f(ci);
      }
  }
}

// True si l'appel jette une std::runtime_error (et donc rejette le layout).
template <class F>
static bool throws_runtime(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

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
  const Real dxc = geom.dx(), dyc = geom.dy();

  // grille GROSSIERE de reference : domaine decoupe en 2 boites (pour pouvoir permuter l'ordre
  // et changer une dmap dans les cas de mismatch).
  const Box2D cb0{{0, 0}, {7, 15}}, cb1{{8, 0}, {15, 15}};
  const BoxArray ba_coarse(std::vector<Box2D>{cb0, cb1});
  const DistributionMapping dm(ba_coarse.size(), n_ranks());

  // patch fin sur les cellules grossieres [4..11]^2 -> box fine {{8,8},{23,23}}.
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_fine(std::vector<Box2D>{fbox});
  const DistributionMapping dm_fine(ba_fine.size(), n_ranks());

  auto pat = [](int ci) { return ci < 8 ? Real(1) : Real(-1); };
  auto ne_fn = [&](int ci) { return Real(1) + Real(0.25) * pat(ci); };
  auto ni_fn = [&](int ci) { return Real(1) - Real(0.25) * pat(ci); };

  using Blk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

  // Construit deux niveaux (grossier + fin) sur les BoxArray/dmap/dx donnes, remplis ne/ni.
  auto make_two_level_block = [&](const BoxArray& cba, const DistributionMapping& cdm, Real cdx,
                                  Real cdy, const BoxArray& fba, const DistributionMapping& fdm,
                                  Real fdx, Real fdy, bool is_e) {
    MultiFab Uc(cba, cdm, 1, 2), Uf(fba, fdm, 1, 2);
    auto fn = [&](int ci) { return is_e ? ne_fn(ci) : ni_fn(ci); };
    fill_by_coarse_i(Uc, 1, fn);
    fill_by_coarse_i(Uf, 2, fn);
    std::vector<AmrLevelMP> levels;
    levels.push_back(AmrLevelMP{std::move(Uc), nullptr, cdx, cdy});
    levels.push_back(AmrLevelMP{std::move(Uf), nullptr, fdx, fdy});
    return levels;
  };

  // Fabrique d'un coupleur a DEUX blocs ; le 2e bloc recoit la grille passee (pour fabriquer un
  // mismatch). Renvoie une lambda 0-arg a executer dans throws_runtime / directement.
  auto build_two_block = [&](const BoxArray& cba2, const DistributionMapping& cdm2, Real cdx2,
                             Real cdy2, const BoxArray& fba2, const DistributionMapping& fdm2,
                             Real fdx2, Real fdy2, bool second_has_fine = true) {
    // bloc 0 : grille de REFERENCE.
    auto e_levels = make_two_level_block(ba_coarse, dm, dxc, dyc, ba_fine, dm_fine, dxc / 2,
                                         dyc / 2, /*is_e=*/true);
    // bloc 1 : grille passee en argument.
    MultiFab Uic(cba2, cdm2, 1, 2);
    fill_by_coarse_i(Uic, 1, ni_fn);
    std::vector<AmrLevelMP> i_levels;
    i_levels.push_back(AmrLevelMP{std::move(Uic), nullptr, cdx2, cdy2});
    if (second_has_fine) {
      MultiFab Uif(fba2, fdm2, 1, 2);
      fill_by_coarse_i(Uif, 2, ni_fn);
      i_levels.push_back(AmrLevelMP{std::move(Uif), nullptr, fdx2, fdy2});
    }

    // state pointe (provisoirement) sur le grossier de chaque bloc ; recable par le coupleur.
    MultiFab& e_coarse = e_levels[0].U;
    MultiFab& i_coarse = i_levels[0].U;
    Blk e{"electrons", AdvectX{Real(1)}, e_coarse, BCRec{}};
    Blk ion{"ions", AdvectX{Real(1)}, i_coarse, BCRec{}};
    CoupledSystem system{e, ion};

    std::vector<std::vector<AmrLevelMP>> bl;
    bl.push_back(std::move(e_levels));
    bl.push_back(std::move(i_levels));
    ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};
    // Construit (et detruit aussitot) : la VERIFICATION de layout a lieu au ctor.
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, charge, std::move(bl));
    (void)sim;
  };

  // --- Partie A : le garde-fou JETTE sur tout mismatch ---

  // A.1 boites DIFFERENTES sur le grossier (une seule boite couvre tout le domaine au lieu de 2).
  {
    const BoxArray cba_one(std::vector<Box2D>{dom});
    const DistributionMapping cdm_one(cba_one.size(), n_ranks());
    chk(throws_runtime([&] {
          build_two_block(cba_one, cdm_one, dxc, dyc, ba_fine, dm_fine, dxc / 2, dyc / 2);
        }),
        "guard_throws_on_box_set_mismatch");
  }

  // A.2 memes boites mais ORDRE different (cb1, cb0) : l'aux partage exige le MEME ordre (fab(li)
  // <-> meme region). L'ancien controle .size() ne voyait PAS ce mismatch.
  {
    const BoxArray cba_swapped(std::vector<Box2D>{cb1, cb0});
    const DistributionMapping cdm_swapped(cba_swapped.size(), n_ranks());
    chk(throws_runtime([&] {
          build_two_block(cba_swapped, cdm_swapped, dxc, dyc, ba_fine, dm_fine, dxc / 2, dyc / 2);
        }),
        "guard_throws_on_box_order_mismatch");
  }

  // A.3 memes boites, DistributionMapping different (rang 1 -> rang 0 sur la 2e boite). Pertinent
  // surtout sous MPI ; en serie tous les rangs valent 0 -> on FORCE une dmap explicite differente.
  {
    const DistributionMapping cdm_alt(std::vector<int>{0, 7});  // rang 7 inexistant en serie
    chk(throws_runtime([&] {
          build_two_block(ba_coarse, cdm_alt, dxc, dyc, ba_fine, dm_fine, dxc / 2, dyc / 2);
        }),
        "guard_throws_on_dmap_mismatch");
  }

  // A.4 dx DIFFERENT (geometrie mal cablee : meme grille d'indices mais pas d'espace different).
  {
    chk(throws_runtime([&] {
          build_two_block(ba_coarse, dm, dxc * Real(2), dyc, ba_fine, dm_fine, dxc / 2, dyc / 2);
        }),
        "guard_throws_on_dx_mismatch");
  }

  // A.5 NOMBRE DE NIVEAUX different (le 2e bloc n'a que le grossier).
  {
    chk(throws_runtime([&] {
          build_two_block(ba_coarse, dm, dxc, dyc, ba_fine, dm_fine, dxc / 2, dyc / 2,
                          /*second_has_fine=*/false);
        }),
        "guard_throws_on_nlevels_mismatch");
  }

  // --- Partie B : le garde-fou PASSE sur un layout strictement identique ---
  {
    chk(!throws_runtime(
            [&] { build_two_block(ba_coarse, dm, dxc, dyc, ba_fine, dm_fine, dxc / 2, dyc / 2); }),
        "guard_passes_on_matching_layout");
  }

  // --- Partie C : chemin MONO-BLOC bit-identique (dmax == 0) ---
  // Deux coupleurs mono-bloc construits a l'IDENTIQUE et avances d'un pas : le garde-fou (boucle
  // inter-blocs vide pour un seul bloc) ne change RIEN, donc les champs grossiers coincident au
  // bit pres apres l'avance AMR (reflux + average_down inclus).
  {
    auto build_single = [&]() {
      MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm_fine, 1, 2);
      fill_by_coarse_i(Uc, 1, ne_fn);
      fill_by_coarse_i(Uf, 2, ne_fn);
      std::vector<AmrLevelMP> levels;
      levels.push_back(AmrLevelMP{std::move(Uc), nullptr, dxc, dyc});
      levels.push_back(AmrLevelMP{std::move(Uf), nullptr, dxc / 2, dyc / 2});
      MultiFab& coarse = levels[0].U;
      Blk b{"only", AdvectX{Real(1)}, coarse, BCRec{}};
      CoupledSystem system{b};
      std::vector<std::vector<AmrLevelMP>> bl;
      bl.push_back(std::move(levels));
      return AmrSystemCoupler(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl));
    };
    auto a = build_single();
    auto b = build_single();
    a.step(Real(0.01));
    b.step(Real(0.01));
    // dmax sur le grossier (composante 0) : difference STRICTEMENT nulle.
    MultiFab diff(ba_coarse, dm, 1, 0);
    lincomb(diff, Real(1), a.levels(0)[0].U, Real(-1), b.levels(0)[0].U);
    chk(norm_inf(diff) == Real(0), "single_block_bit_identical_dmax_zero");
  }

  // --- Partie D : AmrHierarchyLayout::from_levels extrait la grille ---
  {
    std::vector<AmrLevelMP> levels = make_two_level_block(ba_coarse, dm, dxc, dyc, ba_fine, dm_fine,
                                                          dxc / 2, dyc / 2, /*is_e=*/true);
    const AmrHierarchyLayout L = AmrHierarchyLayout::from_levels(levels);
    chk(L.nlev() == 2, "layout_nlev");
    chk(L.ba[0].boxes() == ba_coarse.boxes(), "layout_coarse_boxes");
    chk(L.ba[1].boxes() == ba_fine.boxes(), "layout_fine_boxes");
    chk(L.dm[0].ranks() == dm.ranks(), "layout_coarse_dmap");
    chk(L.dx[0] == dxc && L.dy[0] == dyc, "layout_coarse_dxdy");
    chk(L.dx[1] == dxc / 2 && L.dy[1] == dyc / 2, "layout_fine_dxdy");
  }

  if (fails == 0)
    std::printf("OK test_amr_layout_guard\n");
  return fails == 0 ? 0 : 1;
}
