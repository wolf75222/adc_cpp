#pragma once

#include <adc/core/state.hpp>  // kAuxBaseComps
#include <adc/coupling/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb (injection aux coarse->fine)
#include <adc/coupling/amr_system_coupler.hpp>  // detail::same_layout_or_throw (garde de layout partage)
#include <adc/coupling/aux_fill.hpp>        // detail::derive_aux_bc (CL du canal aux)
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, mf_average_down_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <algorithm>  // std::max (pas CFL substeps/stride-aware)
#include <cmath>      // std::isfinite (rejet d'un dt degenere)
#include <cstddef>
#include <functional>
#include <limits>     // std::numeric_limits (dt initial = +inf, min sur les blocs)
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief Moteur multi-blocs AMR a l'EXECUTION (registre type-erase par nom).
///
/// Pendant raffine de System::Impl (python/system.cpp) : la ou System type-erase les especes
/// (struct Species) sur une grille MONO-NIVEAU, AmrRuntime type-erase N blocs sur une hierarchie
/// AMR PARTAGEE. Il reproduit FIDELEMENT l'algorithme de AmrSystemCoupler::solve_fields / step
/// (include/adc/coupling/amr_system_coupler.hpp), mais sur des fermetures type-erasees (le facade
/// runtime ne connait pas les types Model/Limiter/Flux des blocs a la compilation) plutot que sur
/// un CoupledSystem<Blocks...> compile-time.
///
/// INVARIANTS (capstone multi-blocs, docs/AMR_MULTIBLOCK_DESIGN.md) :
///  - UNE seule hierarchie AMR partagee (AmrHierarchyLayout, garde same_layout_or_throw) : tous les
///    blocs vivent sur EXACTEMENT le meme BoxArray + DistributionMapping + dx/dy par niveau ;
///  - TOUS les blocs vivent sur TOUS les patchs (jamais d'absence spatiale locale d'un bloc) ;
///  - Poisson de SYSTEME a second membre SOMME et CO-LOCALISE : rhs[grossier] = Sum_b
///    elliptic_rhs_b(U_b) lu aux MEMES cellules du grossier partage ;
///  - aux PARTAGE par niveau (phi, grad phi) ; un seul solve Poisson grossier puis injection
///    coarse->fine (coupler_inject_aux_mb), exactement comme AmrSystemCoupler ;
///  - conservation PAR BLOC (reflux + average_down du moteur AMR, dans la fermeture advance).
///
/// PERIMETRE (capstone). On porte des blocs EXPLICITES a schemas spatiaux potentiellement DIFFERENTS
/// sur la hierarchie FIGEE (pas de regrid : AmrSystemCoupler n'en a pas), avec le MULTIRATE par bloc :
/// substeps (sous-pas explicites) et stride (cadence hold-then-catch-up), honores dans step() en
/// mirroir de AmrSystemCoupler::step (#140). Les sources couplees, l'IMEX multi-bloc et le regrid
/// d'union des tags restent des PR ULTERIEURES. Le facade runtime (AmrSystem) REFUSE explicitement
/// multi-blocs + regrid_every > 0 tant que le regrid d'union n'existe pas.

namespace adc {

/// Fermetures type-erasees d'UN bloc AMR, posees sur la hierarchie partagee. Calque AMR de la
/// struct Species de System::Impl : un nom + sa pile de niveaux (sur le layout partage) + ses
/// fermetures (advance / elliptic-rhs / max_speed / mass / density). Les fermetures capturent le
/// Model/Limiter/Flux CONCRETS du bloc (resolus au build) : le noyau reste COMPILE, seule la liste
/// de blocs est type-erasee. Produites par detail::build_amr_block (amr_dsl_block.hpp).
struct AmrRuntimeBlock {
  std::string name;
  int ncomp = 1;
  double gamma = 1.4;
  /// Sous-pas EXPLICITES du bloc dans SON macro-pas effectif : le pas effectif (stride * dt) est
  /// decoupe en substeps morceaux egaux et chaque morceau est avance par UN advance_amr (cf.
  /// AmrRuntime::step). substeps=1 => un seul advance_amr de tout le pas effectif (bit-identique).
  int substeps = 1;
  /// Cadence HOLD-THEN-CATCH-UP du bloc (multirate). stride=1 (defaut) : le bloc avance a CHAQUE
  /// macro-pas (bit-identique). stride=M>1 : le bloc est TENU aux macro-pas 0..M-2 (non avance) puis
  /// RATTRAPE au macro-pas M-1, ou (macro_step+1)%M==0, d'un pas effectif M*dt. Memes semantiques que
  /// block_stride_v / AmrSystemCoupler::step (#140). L'INVARIANT du rattrapage en FIN de fenetre :
  /// au macro-pas k le temps systeme est (k+1)*dt et le bloc qui rattrape a alors cumule (k+1)*dt, il
  /// reste donc temporellement COHERENT avec les blocs rapides (jamais "dans le futur"), ce qui garde
  /// le couplage Poisson (RHS somme) sense : un bloc tenu y contribue avec son etat FIGE (sa derniere
  /// avance), pas avec un etat anticipe qui fausserait q_b n_b dans la somme.
  int stride = 1;
  /// Largeur du canal aux LUE par le modele du bloc (aux_comps<Model>() ; >= kAuxBaseComps). Le
  /// canal aux PARTAGE par niveau est dimensionne au MAX de cette largeur sur tous les blocs, pour
  /// qu'un bloc lisant un champ extra (B_z, T_e ; n_aux > 3) ne lise jamais hors borne.
  int aux_ncomp = kAuxBaseComps;

  /// Pile de niveaux du bloc (niveau 0 = grossier, > 0 = patchs fins), SUR le layout partage. Le
  /// pointeur aux de chaque AmrLevelMP est (re)cable par AmrRuntime vers l'aux PARTAGE du niveau.
  /// shared_ptr : AmrRuntimeBlock reste MOVABLE (un std::vector<AmrLevelMP> est lourd a deplacer
  /// dans un std::function, et le ctor du moteur a besoin d'une adresse stable pour les fermetures).
  std::shared_ptr<std::vector<AmrLevelMP>> levels;

  /// Avance le bloc d'UN sous-pas de taille dt : transport AMR (Berger-Oliger + reflux +
  /// average_down conservatifs) sur la pile de niveaux du bloc, avec SON schema spatial (Limiter,
  /// Flux). Capture advance_amr<Limiter, Flux> sur le Model concret. La boucle de sous-pas et la
  /// cadence stride sont portees par AmrRuntime::step (pendant runtime de AmrSystemCoupler::step) :
  /// la fermeture fait UN advance_amr, le moteur l'appelle substeps fois (dt = pas effectif/substeps).
  /// La signature passe le domaine de base + periodicite + politique d'ownership grossier, recables
  /// par le moteur.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> advance;

  /// Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs_b(U_b) sur le grossier.
  /// CO-LOCALISE : la boucle lit U_b et ecrit rhs AUX MEMES cellules (meme BoxArray grossier
  /// partage). La SOMME des contributions de tous les blocs forme le RHS du Poisson de systeme.
  std::function<void(const MultiFab&, MultiFab&)> add_elliptic_rhs;

  /// Vitesse d'onde max du bloc sur le grossier (pour le pas CFL substeps-aware d'une PR future).
  std::function<Real(const MultiFab&, const MultiFab&)> max_speed;

  /// Masse de la composante 0 du grossier du bloc (somme u*dV ; reduite cross-rang si reparti).
  std::function<Real()> mass;

  /// Densite grossiere (composante 0) du bloc en champ n*n row-major global (diagnostic).
  std::function<std::vector<double>()> density;

  /// Potentiel grossier lu depuis l'aux partage (composante 0) en champ n*n row-major (diagnostic).
  /// Identique pour tous les blocs (aux partage) ; porte par bloc pour la symetrie d'API.
  std::function<std::vector<double>(const MultiFab&)> potential;
};

/// Moteur multi-blocs AMR a l'execution. Detient l'aux PARTAGE par niveau, le Poisson grossier
/// (GeometricMG), la geometrie + CL, et le REGISTRE de blocs type-erases. Reproduit l'algorithme
/// de AmrSystemCoupler (solve_fields + step) sur des fermetures plutot qu'un CoupledSystem.
class AmrRuntime {
 public:
  /// @param geom        geometrie du niveau grossier (domaine + extents physiques).
  /// @param ba_coarse   BoxArray du grossier (le Poisson grossier vit dessus).
  /// @param bcPhi       CL du Poisson grossier.
  /// @param blocks      registre des blocs (>= 1), tous sur le MEME layout (garde au ctor).
  /// @param base_per    periodicite du domaine de base (transport).
  /// @param replicated_coarse  ownership du niveau 0 (replique mono-box, ou reparti multi-box).
  /// @param active      predicat paroi conductrice (passe au MG ; vide = aucune).
  AmrRuntime(const Geometry& geom, const BoxArray& ba_coarse, const BCRec& bcPhi,
             std::vector<AmrRuntimeBlock> blocks, Periodicity base_per = Periodicity{true, true},
             bool replicated_coarse = true, std::function<bool(Real, Real)> active = {})
      : geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        mg_(geom, ba_coarse, bcPhi, std::move(active), replicated_coarse),
        blocks_(std::move(blocks)) {
    if (blocks_.empty())
      throw std::runtime_error("AmrRuntime : au moins un bloc requis");
    for (const auto& b : blocks_)
      if (!b.levels || b.levels->empty())
        throw std::runtime_error("AmrRuntime : chaque bloc doit porter au moins un niveau "
                                 "(grossier) sur le layout partage");
    nlev_ = static_cast<int>(blocks_[0].levels->size());

    // Coherence de layout EXACTE entre blocs (l'aux est partage par niveau) : meme nombre de
    // niveaux, et par niveau meme BoxArray (boites ET ordre), meme DistributionMapping, meme
    // dx/dy. MEME garde-fou que AmrSystemCoupler (detail::same_layout_or_throw) : tous les blocs
    // vivent sur TOUS les patchs de l'UNIQUE hierarchie partagee. Un seul bloc concorde
    // trivialement avec lui-meme (la boucle sur les autres blocs est vide).
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_) ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // Largeur du canal aux PARTAGE : max des aux_comps des blocs (>= kAuxBaseComps). Calque de
    // AmrSystemCoupler::system_aux_comps : un bloc lisant un champ extra (B_z, T_e) dispose de la
    // place a chaque niveau, un bloc de base ignore les composantes extra. PR1 ne PEUPLE pas B_z
    // multi-bloc (pas de bz_ ici), mais on dimensionne quand meme le canal au plus large pour que
    // load_aux<aux_comps<Model>> ne lise jamais hors borne. Sans bloc a champ extra -> kAuxBaseComps
    // (3) -> allocation strictement identique au cas de base.
    aux_ncomp_ = kAuxBaseComps;
    for (const auto& b : blocks_)
      if (b.aux_ncomp > aux_ncomp_) aux_ncomp_ = b.aux_ncomp;

    // aux PARTAGE : un MultiFab (phi, grad phi) par niveau, sur la grille commune. Dimensionne une
    // seule fois -> adresses stables pour les pointeurs aux des blocs. Le layout partage est celui
    // du bloc 0 (garde same_layout_or_throw : identique pour tous).
    aux_.resize(nlev_);
    const auto& L0 = *blocks_[0].levels;
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(L0[k].U.box_array(), L0[k].U.dmap(), aux_ncomp_, 1);
    for (auto& b : blocks_)
      for (int k = 0; k < nlev_; ++k) (*b.levels)[k].aux = &aux_[k];
  }

  int nlev() const { return nlev_; }
  std::size_t n_blocks() const { return blocks_.size(); }
  MultiFab& phi() { return mg_.phi(); }
  // Second membre du Poisson de systeme apres le dernier solve_fields : f = Sum_b elliptic_rhs_b(U_b)
  // sur le grossier partage. Expose pour verifier la SOMME CO-LOCALISEE (test PR1) ; meme grille que
  // le grossier (les contributions des blocs y sont accumulees aux memes cellules).
  MultiFab& poisson_rhs() { return mg_.rhs(); }
  const MultiFab& aux(int k) const { return aux_[k]; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return *blocks_[b].levels; }
  Real mass(std::size_t b) const { return blocks_[b].mass(); }
  std::vector<double> density(std::size_t b) const { return blocks_[b].density(); }
  int solve_count() const { return solve_count_; }

  /// sync_down (par bloc) + Poisson grossier de systeme (RHS SOMME co-localise) + aux grossier +
  /// injection fine. Reproduit AmrSystemCoupler::solve_fields a l'identique, mais le RHS de systeme
  /// est assemble par les fermetures add_elliptic_rhs des blocs (Sum_b elliptic_rhs_b(U_b)) au lieu
  /// d'un RhsAssembler compile-time.
  void solve_fields() {
    ++solve_count_;
    // 1. average_down par bloc (fin -> grossier) sur toute la hierarchie.
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      for (int k = nlev_ - 1; k >= 1; --k) mf_average_down_mb(L[k].U, L[k - 1].U);
    }

    // 2. RHS de systeme SOMME et CO-LOCALISE : f = Sum_b elliptic_rhs_b(U_b) sur le grossier. On
    // remet a zero puis chaque bloc ACCUMULE (+=) sa contribution sur les MEMES cellules du
    // grossier partage (mg_.rhs() partage le layout du grossier).
    mg_.rhs().set_val(Real(0));
    for (auto& b : blocks_) b.add_elliptic_rhs((*b.levels)[0].U, mg_.rhs());
    mg_.solve();

    // 3. aux grossier = (phi, grad phi) via le MEME chemin propre que AmrSystemCoupler : remplir
    // les ghosts de phi selon bcPhi_, field_postprocess (phi + grad), remplir les ghosts d'aux
    // selon aux_bc_ (derive de bcPhi_). Gere le non-periodique (Foextrap).
    fill_ghosts(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_[0], dom_, aux_bc_);
    // 4. injection coarse->fine de l'aux (parent replique seulement au niveau 1 si grossier replique).
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  /// Avance le systeme d'un macro-pas dt. Tous les blocs sont EXPLICITES (l'IMEX multi-bloc est une
  /// PR ulterieure). On resout d'abord les champs (Poisson somme co-localise, UNE fois par macro-pas
  /// : cadence OncePerStep), puis chaque bloc avance sur SA pile de niveaux avec SON schema, en
  /// honorant sa cadence stride et ses substeps. Pendant runtime de AmrSystemCoupler::step (cas
  /// tout-explicite, OncePerStep) : la fermeture advance fait UN advance_amr, le moteur porte ici la
  /// boucle de sous-pas et le filtre stride (la version compile-time les a dans block_substeps_v /
  /// block_stride_v).
  void step(Real dt) {
    solve_count_ = 0;
    // Poisson de systeme resolu UNE fois sur l'etat courant (cadence OncePerStep). Un bloc TENU
    // (stride > 1, hors fin de fenetre) y a contribue avec son etat FIGE depuis sa derniere avance :
    // couplage lache assume du multirate, exactement comme System::step / AmrSystemCoupler en
    // OncePerStep. phi reste gele pendant l'avance des blocs (pas de re-solve par sous-pas ici).
    solve_fields();
    for (auto& b : blocks_) {
      // Cadence HOLD-THEN-CATCH-UP (cf. AmrRuntimeBlock::stride, #140) : le bloc est TENU tant que
      // (macro_step_+1) % stride != 0, puis RATTRAPE en fin de fenetre d'un pas effectif stride*dt.
      // Le rattrapage en FIN de fenetre garde le bloc temporellement coherent avec les rapides au
      // point de couplage (jamais dans le futur). stride=1 : toujours vrai -> chaque pas, bit-identique.
      if ((macro_step_ + 1) % b.stride != 0) continue;
      const Real bdt = dt * static_cast<Real>(b.stride);  // catch-up : pas effectif stride*dt
      // substeps sous-pas EXPLICITES egaux de bdt/substeps. La fermeture advance fait UN advance_amr
      // par appel ; substeps=1 -> un seul advance_amr de bdt (bit-identique au cas mono-substep).
      const Real h = bdt / static_cast<Real>(b.substeps);
      for (int s = 0; s < b.substeps; ++s)
        b.advance(*b.levels, dom_, h, base_per_, replicated_coarse_);
    }
    ++macro_step_;
  }

  /// Pas CFL substeps/stride-aware (pendant runtime de System::step_cfl, mirroir EXACT de sa
  /// formule). Un bloc de cadence stride avance d'un pas effectif stride*dt en substeps sous-pas,
  /// donc chaque sous-pas vaut stride*dt/substeps ; la condition de stabilite par sous-pas
  /// stride*dt/substeps <= cfl*h/w_b donne dt <= cfl*h*substeps_b/(stride_b*w_b). Le dt GLOBAL est le
  /// min sur les blocs (le plus contraignant). On resout d'abord les champs (max_speed par bloc exige
  /// l'aux a jour), on calcule dt, puis on avance d'un step(dt). @p h = pas d'espace du grossier
  /// (dx_coarse). Renvoie le dt utilise. Mono-bloc (un seul bloc, stride=1) : si w_b est le seul
  /// contraignant, dt = cfl*h*substeps/w (identique a System::step_cfl mono-bloc).
  Real step_cfl(Real cfl, Real h) {
    solve_fields();  // aux a jour : max_speed de chaque bloc le lit sur le grossier courant
    Real dt = std::numeric_limits<Real>::infinity();
    for (auto& b : blocks_) {
      const Real w = std::max(b.max_speed((*b.levels)[0].U, aux_[0]), Real(1e-30));
      const Real dt_b = cfl * h * static_cast<Real>(b.substeps) /
                        (static_cast<Real>(b.stride) * w);
      if (dt_b < dt) dt = dt_b;
    }
    if (!std::isfinite(dt)) dt = cfl * h / Real(1e-30);  // garde-fou (aucun bloc : impossible ici)
    step(dt);
    return dt;
  }

  /// Potentiel grossier (composante 0 de l'aux partage) en champ n*n row-major. Resout les champs
  /// si besoin (pendant de AmrSystem::potential), puis lit aux(0). Identique pour tous les blocs.
  std::vector<double> potential() {
    solve_fields();
    return blocks_[0].potential(aux_[0]);
  }

  /// Vitesse d'onde max du SYSTEME (max sur les blocs) sur le grossier courant. Exige aux a jour.
  Real max_speed() {
    solve_fields();
    Real w = Real(1e-12);
    for (auto& b : blocks_) {
      const Real wb = b.max_speed((*b.levels)[0].U, aux_[0]);
      if (wb > w) w = wb;
    }
    return w;
  }

  int n_patches() const {
    const auto& L = *blocks_[0].levels;
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  }

 private:
  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  GeometricMG mg_;
  std::vector<AmrRuntimeBlock> blocks_;
  std::vector<MultiFab> aux_;  // [niveau], partage par tous les blocs
  int aux_ncomp_ = kAuxBaseComps;
  int nlev_ = 0;
  int macro_step_ = 0;
  mutable int solve_count_ = 0;
};

}  // namespace adc
