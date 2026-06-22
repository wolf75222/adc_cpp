// AMR MULTI-BLOCS REGRID D'UNION DES TAGS (capstone Phase 2, C.6 ; docs/AMR_REGRID_UNION_TAGS_DESIGN.md).
//
// Le moteur multi-blocs RUNTIME (AmrRuntime) tournait jusqu'ici a hierarchie FIGEE (pas de regrid).
// Cette PR DEVERROUILLE le regrid pilote par l'UNION des tags : tous les @p regrid_every macro-pas, la
// hierarchie partagee est re-grillee a partir de l'UNION (OU cellule a cellule) des tags de TOUS les
// blocs (predicat PAR BLOC, D1) + des tags de phi (sur |grad phi|, D4), suivie d'UN clustering
// Berger-Rigoutsos -> UN nouveau layout fin applique a TOUS les blocs (y compris ceux tenus par leur
// stride, D3) ET a l'aux partage, en maintenant same_layout_or_throw apres regrid. v1 a 2 niveaux (D5).
//
// Ce que ce test verrouille (les cas demandes a-e) :
//   (a) HIERARCHIE QUI EVOLUE : avec regrid_every > 0, le layout fin CHANGE quand la structure taguee
//       se deplace (la BoxArray fine n'est plus celle du build initial central fixe).
//   (b)+(c) UNION DES TAGS : deux blocs taguant des regions DISJOINTES (bloc A a gauche, bloc B a
//       droite) -> le layout d'union COUVRE LES DEUX regions (bounding box du fin enjambe gauche ET
//       droite). Et un raffinement declenche par phi seul (|grad phi|) ajoute des patchs la ou ni A ni B
//       ne taguent : l'union est bien A OU B OU |grad phi|.
//   (d) BLOC STRIDE-TENU RE-GRILLE : un bloc tenu par son stride (stride=4, non avance au macro-pas de
//       regrid) est NEANMOINS re-grille sur le layout d'union (sa BoxArray fine == layout partage, pas
//       l'ancienne) -> same_layout_or_throw passe (sinon le ctor / le regrid aurait leve), et son fin
//       porte des donnees finies (report + interp), pas un fab non initialise sur l'ancienne grille.
//   (e) regrid_every == 0 BIT-IDENTIQUE : multi-blocs fige reste STRICTEMENT le comportement actuel
//       (regrid jamais appele) -> meme cas joue deux fois donne dmax == 0, et le layout fin ne bouge pas.
//
// CHOIX DE COMPILABILITE (nvcc-safe, comme test_amr_coupled_source_role_strict / test_amr_multiblock_
// compiled) : on construit l'AmrRuntime DIRECTEMENT via detail::make_shared_amr_layout +
// detail::dispatch_amr_block (le noyau AMR reste capture par une fonction template NOMMEE, pas une
// lambda etendue cross-TU). Les PREDICATS DE TAG sont des FONCTEURS NOMMES (structs concrets), jamais
// des lambdas generiques sous concept : ils sont evalues dans la boucle HOTE de tag_cells (pas sur
// device), donc une std::function les capturant est licite et compile partout (CPU + Kokkos).

#include <adc/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <adc/runtime/amr/amr_runtime.hpp>    // AmrRuntime, AmrRuntimeBlock
#include <adc/runtime/amr_system.hpp>  // facade AmrSystem (deverrouillage multi-blocs + regrid_every>0)
#include <adc/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <adc/runtime/config/model_spec.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Spec ExB scalaire (1 var, role density) a charge q. Transport ExB : la densite advecte le long du
// champ ExB, donc une structure se DEPLACE -> la region taguee bouge -> le layout fin change (cas a).
static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// ------------------------------------------------------------------------------------------------
// PREDICATS DE TAG : FONCTEURS NOMMES (structs concrets), nvcc-safe (evalues cote hote dans tag_cells).
// ------------------------------------------------------------------------------------------------

// Tag si la densite (composante 0) du bloc depasse un seuil. Critere PAR BLOC de base.
struct TagDensityAbove {
  Real thr;
  bool operator()(const ConstArray4& a, int i, int j) const { return a(i, j, 0) > thr; }
};

// Tag si |grad phi| > seuil, lu sur l'aux partage (composantes 1,2 = grad phi en x,y). Critere de phi
// SEPARE (D4) : on tague le BORD des structures (le gradient du potentiel), pas le potentiel lui-meme.
struct TagGradPhiAbove {
  Real thr;
  bool operator()(const ConstArray4& a, int i, int j) const {
    const Real gx = a(i, j, 1), gy = a(i, j, 2);
    return std::sqrt(gx * gx + gy * gy) > thr;
  }
};

// ------------------------------------------------------------------------------------------------
// densites initiales : un disque gaussien centre en (cx, cy) du domaine [0,1]^2, amplitude amp + base.
// Moyenne ajustee a base pour la solvabilite periodique du Poisson (on travaille relativement a base).
// ------------------------------------------------------------------------------------------------
static std::vector<double> blob(int n, double cx, double cy, double amp, double base,
                                double width) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      rho[static_cast<std::size_t>(j) * n + i] = base + amp * std::exp(-r2 / (width * width));
    }
  return rho;
}

// densite uniforme (bloc neutre / fond) ; ne tague rien tant qu'on n'enregistre pas son predicat.
static std::vector<double> flat(int n, double v) {
  return std::vector<double>(static_cast<std::size_t>(n) * n, v);
}

static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

template <class F>
static bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// Bounding box (coords du niveau FIN) de la BoxArray fine du bloc 0 (layout partage : identique pour
// tous les blocs). Permet de verifier la couverture spatiale du layout d'union (cas b/c).
static Box2D fine_bbox(AmrRuntime& rt) {
  const std::vector<AmrLevelMP>& L = rt.levels(0);
  return L[1].U.box_array().bounding_box();
}

// Vecteur des boites fines (coords fin) du bloc 0, ordonne, pour comparer deux layouts (cas a/e).
static std::vector<Box2D> fine_boxes(AmrRuntime& rt) {
  return rt.levels(0)[1].U.box_array().boxes();
}

static bool same_box_list(const std::vector<Box2D>& a, const std::vector<Box2D>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k].lo[0] != b[k].lo[0] || a[k].lo[1] != b[k].lo[1] || a[k].hi[0] != b[k].hi[0] ||
        a[k].hi[1] != b[k].hi[1])
      return false;
  return true;
}

// Construit un AmrRuntime a deux blocs ExB scalaires sur une hierarchie 2 niveaux N x N (un patch fin
// central seed). Densites initiales fournies. q0/q1 : charges (signe inclus) pour le Poisson somme.
static AmrRuntime make_two_block(int N, double L, double B0, double q0, double q1,
                                 const std::vector<double>& rho0, const std::vector<double>& rho1,
                                 int stride1 = 1) {
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every =
      0;  // le runtime porte sa propre cadence via set_regrid (la facade ne pilote pas ici)
  bp.poisson_bc = BCRec{};  // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(q0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "a", rho0,
                                                /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  detail::dispatch_model(exb_charge(q1, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "b", rho1,
                                                /*has_density=*/true, 1.4, 1, false, false,
                                                stride1));
  });
  return AmrRuntime(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("  [%s] %s\n", c ? "OK " : "XX ", w);
    if (!c)
      ++fails;
  };

  const int N = 32;
  const double L = 1.0, B0 = 1.0;

  // ============================================================================================
  // (e) NON-REGRESSION FIGEE : regrid_every == 0 -> regrid JAMAIS appele -> bit-identique + layout fin
  //     inchange. On le teste D'ABORD pour ancrer le comportement de reference (la hierarchie figee).
  // ============================================================================================
  {
    auto run_frozen = [&]() {
      AmrRuntime rt = make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.35, 0.5, 0.8, 1.0, 0.10),
                                     blob(N, 0.65, 0.5, 0.8, 1.0, 0.10));
      // set_regrid(0) explicite : meme avec des predicats enregistres, regrid_every_==0 -> figee.
      rt.set_regrid(0);
      rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.3)});
      rt.set_block_tag_predicate(1, TagDensityAbove{Real(1.3)});
      const std::vector<Box2D> fb_before = fine_boxes(rt);
      for (int s = 0; s < 8; ++s)
        rt.step(Real(0.01));
      const std::vector<Box2D> fb_after = fine_boxes(rt);
      return std::make_pair(rt.density(0), same_box_list(fb_before, fb_after));
    };
    const auto a = run_frozen();
    const auto b = run_frozen();
    chk(a.second, "e_frozen_fine_layout_unchanged");  // la grille n'a pas bouge
    chk(dmax_field(a.first, b.first) == 0.0, "e_frozen_bit_identical_dmax0");

    AmrRuntime rt = make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.35, 0.5, 0.8, 1.0, 0.10),
                                   blob(N, 0.65, 0.5, 0.8, 1.0, 0.10));
    rt.set_regrid(0);
    rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.3)});
    for (int s = 0; s < 8; ++s)
      rt.step(Real(0.01));
    chk(rt.regrid_count() == 0, "e_regrid_count_zero_when_frozen");
  }

  // ============================================================================================
  // (a) HIERARCHIE QUI EVOLUE : regrid_every > 0 -> le layout fin CHANGE par rapport au seed central
  //     fixe du build (l'union des tags des deux blobs n'est PAS le patch central [n/4..3n/4]^2).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.30, 0.5, 1.0, 1.0, 0.07),
                                   blob(N, 0.70, 0.5, 1.0, 1.0, 0.07));
    rt.set_regrid(/*every=*/2, /*grow=*/2, /*margin=*/2);
    rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.5)});
    rt.set_block_tag_predicate(1, TagDensityAbove{Real(1.5)});
    const std::vector<Box2D> fb_seed = fine_boxes(rt);            // patch central fixe du build
    const double m0_before = rt.mass(0), m1_before = rt.mass(1);  // (V1) snapshot avant la sequence
    for (int s = 0; s < 6; ++s)
      rt.step(Real(0.01));
    chk(rt.regrid_count() >= 1, "a_regrid_was_called");
    const std::vector<Box2D> fb_now = fine_boxes(rt);
    chk(!same_box_list(fb_seed, fb_now), "a_fine_layout_evolved_from_seed");
    chk(all_finite(rt.density(0)) && all_finite(rt.density(1)), "a_state_finite_after_regrid");
    chk(rt.n_patches() >= 1, "a_hierarchy_still_has_fine_patches");
    // (V1) CONSERVATION PAR BLOC a travers les regrids : le report fin (exact) + l'interp parent
    // piecewise-constant (conservative au sens integral) redistribuent sans creer ni detruire de masse ;
    // le transport ExB periodique + reflux conserve la masse grossiere. Verifie POUR CHAQUE bloc.
    chk(std::fabs(rt.mass(0) - m0_before) < 1e-9, "a_block0_mass_conserved_across_regrid");
    chk(std::fabs(rt.mass(1) - m1_before) < 1e-9, "a_block1_mass_conserved_across_regrid");
  }

  // ============================================================================================
  // (b)+(c) UNION DES TAGS. Bloc A tague une region a GAUCHE, bloc B une region a DROITE (disjointes).
  //     Le layout d'union doit COUVRIR LES DEUX (bounding box du fin enjambe gauche ET droite). Puis,
  //     en n'activant que le predicat phi (sur |grad phi|), un raffinement est declenche par phi SEUL.
  // ============================================================================================
  {
    // Bloc A : blob a gauche (cx=0.25). Bloc B : blob a droite (cx=0.75). Charges opposees -> phi non
    // trivial (Poisson somme). Predicats par bloc seulement (phi non enregistre) : union = A OU B.
    AmrRuntime rt = make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.25, 0.5, 1.2, 1.0, 0.06),
                                   blob(N, 0.75, 0.5, 1.2, 1.0, 0.06));
    rt.set_regrid(/*every=*/1, /*grow=*/1, /*margin=*/1);
    rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.6)});  // tague le blob gauche (A)
    rt.set_block_tag_predicate(1, TagDensityAbove{Real(1.6)});  // tague le blob droit (B)
    // phi NON enregistre ici : on isole l'union A OU B. Le premier step (macro_step_=0) ne regrid PAS
    // (la grille est fraichement construite, convention mono-bloc) ; le 2e step (macro_step_=1, every=1)
    // declenche le regrid d'union.
    const std::vector<Box2D> fb_seed = fine_boxes(rt);
    rt.step(Real(0.005));
    rt.step(Real(0.005));
    chk(rt.regrid_count() >= 1, "bc_union_regrid_called");
    const Box2D bb = fine_bbox(rt);
    // coords du niveau fin = 2 x coords grossieres. Gauche ~ cellule grossiere 8 (x=0.25*32) -> fin ~16 ;
    // droite ~ cellule grossiere 24 -> fin ~48. L'union doit enjamber le milieu (fin ~32) : lo a gauche
    // du milieu, hi a droite du milieu -> couvre les DEUX regions, pas une seule.
    const int mid_fine = N;  // milieu du domaine en coords fin (2 * N/2)
    chk(!bb.empty(), "bc_union_layout_nonempty");
    chk(bb.lo[0] < mid_fine && bb.hi[0] > mid_fine, "bc_union_covers_both_left_and_right");
    // Le layout d'union DIFFERE du seed central fixe : la couverture des DEUX regions est bien le
    // produit du regrid d'union, pas un artefact du patch central initial (qui couvre deja le milieu).
    chk(!same_box_list(fb_seed, fine_boxes(rt)), "bc_union_layout_differs_from_seed");
    chk(all_finite(rt.density(0)) && all_finite(rt.density(1)), "bc_state_finite");

    // (c) UNION PAR PHI SEUL : nouveau runtime, AUCUN predicat de bloc, seulement le predicat phi sur
    //     |grad phi|. Un raffinement est alors declenche PAR PHI (preuve que phi entre dans l'union,
    //     independamment des criteres de bloc, D4). On choisit un seuil bas pour garantir des tags. La
    //     comparaison au seed prouve que le layout fin est REELLEMENT celui calcule par le regrid phi.
    AmrRuntime rtp =
        make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.5, 0.5, 1.5, 1.0, 0.06), flat(N, 1.0));
    rtp.set_regrid(/*every=*/1, /*grow=*/1, /*margin=*/1);
    // AUCUN set_block_tag_predicate : les blocs ne taguent rien de leur cote -> seul phi pilote l'union.
    rtp.set_phi_tag_predicate(
        TagGradPhiAbove{Real(1e-6)});  // |grad phi| > epsilon -> tague le bord
    const std::vector<Box2D> fb_seed_phi = fine_boxes(rtp);
    rtp.step(Real(0.005));
    rtp.step(Real(0.005));
    chk(rtp.regrid_count() >= 1, "c_phi_only_regrid_called");
    chk(rtp.n_patches() >= 1, "c_phi_only_triggers_refinement");
    chk(!same_box_list(fb_seed_phi, fine_boxes(rtp)), "c_phi_only_layout_from_regrid_not_seed");
    chk(all_finite(rtp.density(0)), "c_phi_only_state_finite");
  }

  // ============================================================================================
  // (d) BLOC STRIDE-TENU RE-GRILLE. Bloc B a stride=4 : il est TENU (non avance) aux macro-pas 0,1,2
  //     puis rattrape au pas 3. Un regrid au pas 2 (every=2) tombe sur un macro-pas ou B est TENU : B
  //     doit NEANMOINS etre re-grille sur le layout d'union (sa BoxArray fine == celle du bloc A, pas
  //     l'ancienne) -> same_layout_or_throw passe (sinon le regrid aurait leve) et son fin porte des
  //     donnees finies (report + interp), pas un fab non initialise sur l'ancienne grille.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, +1.0, -1.0, blob(N, 0.30, 0.5, 1.0, 1.0, 0.07),
                                   blob(N, 0.70, 0.5, 1.0, 1.0, 0.07), /*stride1=*/4);
    rt.set_regrid(/*every=*/2, /*grow=*/2, /*margin=*/2);
    rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.5)});  // bloc A (rapide) tague
    rt.set_block_tag_predicate(1, TagDensityAbove{Real(1.5)});  // bloc B (stride=4) tague aussi
    // Avance jusqu'a un macro-pas de regrid (macro_step_=2, every=2) ou B est TENU ((2+1)%4 != 0).
    for (int s = 0; s < 3; ++s)
      rt.step(Real(0.01));
    chk(rt.regrid_count() >= 1, "d_regrid_called_with_strided_block");
    // Le bloc B (stride-tenu) partage EXACTEMENT le layout fin du bloc A apres regrid (sinon le
    // same_layout_or_throw interne au regrid aurait leve avant d'arriver ici).
    const std::vector<Box2D> fa = rt.levels(0)[1].U.box_array().boxes();
    const std::vector<Box2D> fb = rt.levels(1)[1].U.box_array().boxes();
    chk(same_box_list(fa, fb), "d_strided_block_on_union_layout_not_stale");
    // Son fin porte des donnees finies (report + interp du regrid), pas un fab non initialise.
    chk(all_finite(rt.density(1)), "d_strided_block_state_finite");
    // Et le bloc B a bien ete re-grille hors de l'ancien seed central : son fin a evolue.
    chk(rt.n_patches() >= 1, "d_strided_block_has_fine_patches");
  }

  // ============================================================================================
  // (T7) DEVERROUILLAGE FACADE : AmrSystem (facade runtime) accepte desormais multi-blocs +
  //      regrid_every > 0 (l'ancien refus de python/amr_system.cpp est leve). On verifie :
  //        - multi-blocs + regrid_every > 0 NE LEVE PLUS (ensure_built reussit, le step tourne) ;
  //        - la hierarchie BOUGE (n_patches/layout evoluent) quand un seuil de raffinement est pose ;
  //        - regrid_every == 0 reste FIGE et BIT-IDENTIQUE (meme cas joue deux fois -> dmax == 0).
  // ============================================================================================
  {
    auto exb_spec = [](double q, double B0) {
      ModelSpec s;
      s.transport = "exb";
      s.source = "none";
      s.elliptic = "charge";
      s.q = q;
      s.B0 = B0;
      return s;
    };
    const std::vector<double> r0 = blob(N, 0.30, 0.5, 1.0, 1.0, 0.07);
    const std::vector<double> r1 = blob(N, 0.70, 0.5, 1.0, 1.0, 0.07);

    // (T7-a) multi-blocs + regrid_every > 0 NE LEVE PLUS + la hierarchie bouge.
    const bool unlocked_no_throw = !raises([&] {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 2;  // AVANT cette PR : ensure_built LEVAIT en multi-blocs
      AmrSystem sim(cfg);
      sim.add_block("a", exb_spec(+1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.add_block("b", exb_spec(-1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_refinement(1.5);  // tag density > 1.5 (union des deux blobs)
      sim.set_density("a", r0);
      sim.set_density("b", r1);
      sim.advance(0.01, 6);
      if (!all_finite(sim.density("a")) || !all_finite(sim.density("b")))
        throw std::runtime_error("etat non fini");
    });
    chk(unlocked_no_throw, "T7_facade_multiblock_regrid_every_positive_no_longer_throws");

    // (T7-b) regrid_every == 0 reste FIGE et BIT-IDENTIQUE a la facade.
    auto run_facade_frozen = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;  // hierarchie figee
      AmrSystem sim(cfg);
      sim.add_block("a", exb_spec(+1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.add_block("b", exb_spec(-1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_refinement(1.5);
      sim.set_density("a", r0);
      sim.set_density("b", r1);
      sim.advance(0.01, 6);
      return sim.density("a");
    };
    const std::vector<double> fa = run_facade_frozen();
    const std::vector<double> fb = run_facade_frozen();
    chk(dmax_field(fa, fb) == 0.0, "T7_facade_frozen_regrid_every_zero_bit_identical_dmax0");
  }

  if (fails == 0)
    std::printf("OK test_amr_multiblock_regrid_union\n");
  else
    std::printf("FAIL test_amr_multiblock_regrid_union : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
