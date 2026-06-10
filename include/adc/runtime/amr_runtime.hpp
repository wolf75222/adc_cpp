#pragma once

#include <adc/amr/regrid.hpp>   // tag_cells, grow_tags (tags par bloc + phi pour le regrid d'union)
#include <adc/amr/tag_box.hpp>  // TagBox, tag_union (OU cellule a cellule des tags de tous les blocs)
#include <adc/core/state.hpp>  // kAuxBaseComps
#include <adc/core/variables.hpp>  // VariableSet, VariableRole, role_from_name (role -> composante des sources couplees)
#include <adc/coupling/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb (injection aux coarse->fine)
#include <adc/coupling/amr_regrid_coupler.hpp>  // regrid_compute_fine_layout + regrid_field_on_layout (briques scindees)
#include <adc/coupling/amr_system_coupler.hpp>  // detail::same_layout_or_throw (garde de layout partage)
#include <adc/coupling/aux_fill.hpp>        // detail::derive_aux_bc (CL du canal aux)
#include <adc/coupling/coupled_source_program.hpp>  // CoupledSourceKernel + CsProgram (ABI plate, bytecode P5)
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, mf_average_down_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/patch_box.hpp>  // PatchBox : empreinte index-space d'un patch fin (patch_boxes())
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
/// PERIMETRE (capstone). On porte des blocs a schemas spatiaux potentiellement DIFFERENTS sur la
/// hierarchie FIGEE (pas de regrid : AmrSystemCoupler n'en a pas), avec le MULTIRATE par bloc :
/// substeps (sous-pas explicites) et stride (cadence hold-then-catch-up), honores dans step() en
/// mirroir de AmrSystemCoupler::step (#140). Le TRAITEMENT TEMPOREL est PAR BLOC : explicite (source
/// en Euler avant, portee par le pas AMR) OU IMEX (source raide traitee en IMPLICITE par
/// backward_euler_source, le transport restant explicite ; capstone vii), selectionne dans step().
///
/// SEMANTIQUE IMEX SOUS substeps (decision d'integration, suite revue #184). A substeps=1 ET stride=1
/// la branche IMEX du runtime COINCIDE avec la branche IMEX du moteur compile-time AmrSystemCoupler::step
/// (un transport SOURCE-FREE + un backward_euler_source sur le pas effectif). POUR substeps>1 les deux
/// chemins DIVERGENT DELIBEREMENT :
///   - le moteur COMPILE-TIME IGNORE substeps sur la branche IMEX : il fait UN seul transport source-free
///     puis UN seul implicit_advance sur tout le pas effectif bdt (cf. amr_system_coupler.hpp : la boucle
///     de sous-pas n'existe que dans la branche Explicit) ;
///   - le RUNTIME SOUS-CYCLE le splitting IMEX : il applique imex_advance K=substeps fois, chacune sur
///     bdt/K, soit K pas de Lie [transport(dt/K) ; source implicite(dt/K)].
/// Ce choix est ASSUME et SAIN (ce n'est PAS un bug) : (a) le transport explicite source-free devient
/// PLUS SUR EN CFL (chaque sous-pas porte dt/K, donc une vitesse d'onde K fois plus grande reste
/// admissible) ; (b) le backward-Euler est INCONDITIONNELLEMENT STABLE quel que soit le pas, donc le
/// sous-cyclage ne destabilise jamais la source ; (c) raffiner le pas du backward-Euler RAPPROCHE la
/// relaxation raide de sa trajectoire continue (erreur de splitting et erreur temporelle implicite en
/// O(dt) toutes deux reduites). Le runtime ne mirroite donc PAS le compile-time bit-a-bit des que
/// substeps>1 ; il honore substeps de facon COHERENTE avec la branche explicite (meme decoupage en K
/// sous-pas egaux), ce qui est le comportement attendu d'un utilisateur reglant substeps. Verrou de
/// non-regression : test_amr_multiblock_imex compare une trajectoire substeps=4 a substeps=1 et exige
/// qu'elles DIFFERENT (le sous-cyclage est intentionnel, pas accidentel).
///
/// Le regrid d'union des tags et le DSL production multi-bloc compile restent des PR ULTERIEURES. Le
/// facade runtime (AmrSystem) REFUSE explicitement multi-blocs + regrid_every > 0 tant que le regrid
/// d'union n'existe pas.

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

  /// Descripteur des variables CONSERVATIVES du modele (noms + ROLES physiques, Model::conservative_vars()).
  /// Source unique de verite pour resoudre un role (Density, MomentumX, ...) -> indice de composante dans
  /// add_coupled_source, comme System::add_coupled_source lit Species::cons_vars. La resolution est STRICTE
  /// (#181) : si le bloc n'expose PAS le role canonique demande (index_of < 0), add_coupled_source LEVE au
  /// lieu de retomber sur la composante 0 (un repli silencieux appliquerait la source au mauvais champ).
  VariableSet cons_vars;

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

  /// TRAITEMENT TEMPOREL du bloc : false (defaut) = EXPLICITE (source en Euler avant, dans advance) ;
  /// true = IMEX (source raide traitee en IMPLICITE par backward_euler_source). Le facade (AmrSystem)
  /// le fige depuis time="imex". Selectionne EXPLICITEMENT dans AmrRuntime::step (pendant runtime du
  /// branchement constexpr block_time_treatment_v de AmrSystemCoupler::step) : un bloc explicite passe
  /// par advance, un bloc IMEX par imex_advance. false partout -> trajectoire bit-identique a l'historique.
  bool imex = false;

  /// Avance IMEX du bloc d'UN sous-pas de taille dt : (1) TRANSPORT EXPLICITE sur le modele SOURCE-FREE
  /// (-div F seul, SourceFreeModel<Model>) par le moteur AMR (Berger-Oliger + reflux + average_down
  /// conservatifs), puis (2) SOURCE RAIDE IMPLICITE backward_euler_source A CHAQUE NIVEAU (Newton local,
  /// jacobienne par differences finies ; masque implicite PORTE PAR LE BLOC pour l'IMEX partiel), suivie
  /// d'une cascade fin -> grossier (mf_average_down_mb). UN appel = UN pas de Lie [transport ; source
  /// implicite] sur dt. La SEMANTIQUE de ce splitting (transport source-free puis backward-Euler) calque
  /// la branche IMEX de AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper) ; A substeps=1
  /// elle lui est IDENTIQUE. Mais step() appelle CETTE fermeture substeps fois (sur dt = pas effectif /
  /// substeps), donc pour substeps>1 le runtime SOUS-CYCLE le splitting IMEX la ou le compile-time
  /// l'applique une seule fois sur tout le pas effectif : divergence ASSUMEE (cf. SEMANTIQUE IMEX SOUS
  /// substeps, en-tete du fichier). Capture le Model/Limiter/Flux CONCRETS + le masque (build_amr_block) ;
  /// le noyau reste COMPILE, seul le registre de blocs est type-erase. INVARIANT DE CONSERVATION (source
  /// LOCALE) : la source est cellule-locale (hors flux de face), donc HORS des registres de reflux -> la
  /// conservation aux interfaces grossier-fin reste intacte ; une cellule grossiere COUVERTE redevient la
  /// moyenne 2x2 de ses enfants par la cascade finale (sinon le diagnostic de masse, somme du seul
  /// grossier, compterait une source fantome). Vide pour un bloc explicite (imex == false) : step() ne
  /// l'appelle jamais.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> imex_advance;

  /// Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs_b(U_b) sur le grossier.
  /// CO-LOCALISE : la boucle lit U_b et ecrit rhs AUX MEMES cellules (meme BoxArray grossier
  /// partage). La SOMME des contributions de tous les blocs forme le RHS du Poisson de systeme.
  std::function<void(const MultiFab&, MultiFab&)> add_elliptic_rhs;

  /// Vitesse pilotant la CFL du bloc sur le grossier. Par defaut max_wave_speed (historique) ;
  /// quand le modele declare le trait HasStabilitySpeed, c'est lambda* (stability_speed) que la
  /// fermeture reduit -- MEME politique que System (make_max_speed), cf. build_amr_block.
  std::function<Real(const MultiFab&, const MultiFab&)> max_speed;

  /// BORNES DE PAS OPTIONNELLES du bloc (StabilityPolicy AMR, audit 2026-06) : evaluees sur le
  /// GROSSIER (niveau 0, la ou vit la CFL AMR -- cf. step_cfl : h = dx_coarse). VIDES (defaut) ->
  /// step_cfl garde la borne transport seule, bit-identique. Remplies par build_amr_block /
  /// build_amr_compiled quand le modele declare HasSourceFrequency / HasStabilityDt (memes
  /// semantiques que System : mu en 1/s -> dt <= cfl*substeps/(stride*mu), sans h ; pas admissible
  /// direct -> dt <= dt_adm*substeps/stride, sans cfl).
  std::function<Real(const MultiFab&, const MultiFab&)> source_frequency;
  std::function<Real(const MultiFab&, const MultiFab&)> stability_dt;

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

    // Predicats de tag du regrid d'union : un slot vide par bloc (set_block_tag_predicate les remplit).
    // Vide par defaut -> aucun tag -> hierarchie figee (le regrid n'est de toute facon pas appele tant
    // que set_regrid n'a pas active regrid_every_ > 0).
    block_tag_.resize(blocks_.size());
  }

  int nlev() const { return nlev_; }
  std::size_t n_blocks() const { return blocks_.size(); }
  std::size_t n_coupled_sources() const { return coupled_sources_.size(); }
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
  int regrid_count() const { return regrid_count_; }

  /// Predicat de tag du regrid d'union : (ConstArray4 du champ lu, i, j) -> doit-on raffiner ? Type
  /// HOTE (evalue dans la boucle hote de tag_cells, jamais sur device) : une std::function capturant un
  /// foncteur concret est licite (nvcc-safe -- le predicat n'entre pas dans un kernel). On l'utilise pour
  /// le critere PAR BLOC (lu sur la densite/U du bloc, composante 0) et pour le critere de phi (lu sur
  /// l'aux partage). docs/AMR_REGRID_UNION_TAGS_DESIGN.md (D1, D4).
  using TagPredicate = std::function<bool(const ConstArray4&, int, int)>;

  /// Active le REGRID D'UNION DES TAGS a la cadence @p every (en macro-pas) : tous les @p every
  /// macro-pas, AVANT le step(dt) du macro-pas (D2, coherent avec le mono-bloc amr_dsl_block.hpp:104),
  /// la hierarchie partagee est re-grillee a partir de l'UNION des tags de tous les blocs + phi.
  /// @p every == 0 (DEFAUT) -> hierarchie FIGEE, regrid jamais appele -> trajectoire BIT-IDENTIQUE a
  /// l'historique (la feature est opt-in). @p grow : dilatation des tags (nesting + anticipation) ;
  /// @p margin : nesting (clamp des patchs aux bords). Doit etre appele AVANT le premier step.
  void set_regrid(int every, int grow = 2, int margin = 2) {
    if (every < 0) throw std::runtime_error("AmrRuntime::set_regrid : regrid_every >= 0");
    regrid_every_ = every;
    regrid_grow_ = grow;
    regrid_margin_ = margin;
  }

  /// Enregistre le PREDICAT DE TAG du bloc @p b (D1 : critere d'union PAR BLOC). Le predicat est evalue
  /// sur U du bloc (composante 0 = densite, ou un gradient discret a la charge de l'appelant) au niveau
  /// PARENT pendant le regrid ; l'UNION (OU) des predicats de tous les blocs + le critere phi pilote le
  /// clustering. Un bloc SANS predicat enregistre ne tague rien de SON cote (il reste re-grille comme
  /// fond, present partout, par l'union des autres criteres). @throws si @p b est hors bornes.
  void set_block_tag_predicate(std::size_t b, TagPredicate crit) {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::set_block_tag_predicate : indice de bloc hors bornes");
    block_tag_[b] = std::move(crit);
  }

  /// Enregistre le PREDICAT DE TAG de PHI (D4 : critere de phi SEPARE, sur |grad phi|). Le predicat est
  /// evalue sur l'aux partage du niveau parent (composantes 1,2 = grad phi en x,y) pendant le regrid ;
  /// il s'ajoute a l'union des tags des blocs. Non enregistre -> phi ne contribue pas a l'union.
  void set_phi_tag_predicate(TagPredicate crit) { phi_tag_ = std::move(crit); }

  /// Enregistre une SOURCE COUPLEE inter-especes (DSL CoupledSource, bytecode P5) sur la facade
  /// runtime, pendant raffine de System::add_coupled_source. L'ABI est PLATE (bytecode postfixe) : on
  /// resout chaque (bloc, role) en (indice de bloc, composante) puis on stocke une fermeture qui, a
  /// chaque macro-pas APRES le transport, applique la source par splitting forward-Euler additif via
  /// coupled_source_step. Le couplage est ENTIEREMENT cuit en machine a pile (foncteur device-clean
  /// CoupledSourceKernel) : AUCUN callback Python par cellule dans le chemin chaud.
  ///
  /// CONSERVATION (echange conservatif) : avec une construction add_pair (un terme +expr sur un bloc,
  /// -expr exactement sur l'autre, MEME cellule), les deux contributions par cellule sont opposees au
  /// signe pres, donc n_a + n_b est conserve PAR CELLULE (et globalement) a la precision machine,
  /// independamment de dt et de l'etat. Le moteur ne l'impose pas (une ionisation creant une paire est
  /// licite) : la conservation est une propriete du couplage construit, controlee cote test.
  ///
  /// @param in_blocks/in_roles  champs LUS (un registre par (bloc, role)), dans l'ordre des registres.
  /// @param consts              constantes (parametres), chargees dans les registres apres les entrees.
  /// @param out_blocks/out_roles cible (bloc, role) de chaque terme de source.
  /// @param prog_ops/prog_args  bytecode postfixe CONCATENE de tous les termes (decoupe par prog_lens).
  /// @param prog_lens           longueur du programme de chaque terme (taille == out_blocks).
  /// @throws std::runtime_error sur une forme incoherente, un role inconnu, un bloc inconnu, un opcode
  ///         ou un registre hors bornes, ou un programme trop long (memes gardes que System).
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens) {
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    const int n_terms = static_cast<int>(out_blocks.size());
    // --- validation de forme (avant tout pas, erreurs EXPLICITES) ; mirroir de System::add_coupled_source.
    if (n_terms == 0)
      throw std::runtime_error("AmrRuntime::add_coupled_source : aucun terme de source (out_blocks vide)");
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error("AmrRuntime::add_coupled_source : in_blocks / in_roles de tailles differentes");
    if (static_cast<int>(out_roles.size()) != n_terms || static_cast<int>(prog_lens.size()) != n_terms)
      throw std::runtime_error("AmrRuntime::add_coupled_source : out_blocks / out_roles / prog_lens de "
                               "tailles differentes");
    if (prog_ops.size() != prog_args.size())
      throw std::runtime_error("AmrRuntime::add_coupled_source : prog_ops / prog_args de tailles differentes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error("AmrRuntime::add_coupled_source : trop de registres (entrees + constantes > " +
                               std::to_string(kCsMaxReg) + ")");
    if (n_terms > kCsMaxTerms)
      throw std::runtime_error("AmrRuntime::add_coupled_source : trop de termes de source (> " +
                               std::to_string(kCsMaxTerms) + ")");
    // Resout (bloc, role) -> (indice de bloc, composante) par le descripteur CONSERVATIF du bloc, comme
    // System (#181). Un bloc inconnu leve immediatement ; un role inconnu (non canonique) aussi.
    auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_source : aucun bloc nomme '" + block + "'");
      const VariableRole r = role_from_name(role);
      if (r == VariableRole::Custom)
        throw std::runtime_error("AmrRuntime::add_coupled_source : role '" + role + "' inconnu (bloc '" +
                                 block + "')");
      // STRICT (pas de repli silencieux ; mirroir de System::add_coupled_source #181) : une source couplee
      // DSL vise un (bloc, role) EXPLICITEMENT demande par l'utilisateur. Si le bloc n'expose PAS ce role
      // (canonique mais absent de cons_vars), un repli sur la composante 0 appliquerait la source au mauvais
      // champ EN SILENCE (le faux-positif identifie a la revue Lot E). On leve. Distinct des couplages NOMMES
      // (add_collision/add_pair cote System) qui assument volontairement la disposition canonique via
      // role_index(..., fallback) et restent inchanges (ils ne passent pas par ce chemin runtime AMR).
      const int comp = blocks_[static_cast<std::size_t>(b)].cons_vars.index_of(r);
      if (comp < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_source : le bloc '" + block +
                                 "' n'expose pas le role '" + role +
                                 "' (pas de repli silencieux sur la composante 0)");
      return {b, comp};
    };
    // Entrees : (bloc, composante) lues par cellule. Capturees par INDICE -> on reconstruit les Array4 a
    // CHAQUE application (les fabs vivent dans la pile de niveaux, repointees par niveau dans le splitting).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      auto [b, comp] = resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    std::vector<CsRef> outs(static_cast<std::size_t>(n_terms));
    int off = 0;
    for (int t = 0; t < n_terms; ++t) {
      auto [b, comp] = resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
      const int len = prog_lens[static_cast<std::size_t>(t)];
      if (len < 0 || len > kCsMaxProg)
        throw std::runtime_error("AmrRuntime::add_coupled_source : programme du terme " +
                                 std::to_string(t) + " trop long (> " + std::to_string(kCsMaxProg) + ")");
      if (off + len > static_cast<int>(prog_ops.size()))
        throw std::runtime_error("AmrRuntime::add_coupled_source : prog_lens incoherent avec prog_ops");
      CsProgram pg;
      pg.len = len;
      for (int k = 0; k < len; ++k) {
        const int opc = prog_ops[static_cast<std::size_t>(off + k)];
        const int a = prog_args[static_cast<std::size_t>(off + k)];
        if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
          throw std::runtime_error("AmrRuntime::add_coupled_source : opcode invalide");
        if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
          throw std::runtime_error("AmrRuntime::add_coupled_source : registre hors bornes dans le programme");
        pg.op[k] = opc;
        pg.arg[k] = a;
      }
      outs[static_cast<std::size_t>(t)] = {b, comp, pg};
      off += len;
    }
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_sources_.push_back(CoupledSourceSpec{std::move(ins), std::move(outs), std::move(kconsts),
                                                 n_in, n_const, n_terms});
  }

  /// Applique TOUTES les sources couplees enregistrees d'un pas dt, par splitting forward-Euler.
  /// Pendant runtime de AmrSystemCoupler::coupled_source_step : on rafraichit les champs (aux par
  /// niveau) puis, source par source, on applique le bytecode INDEPENDAMMENT A CHAQUE NIVEAU de la
  /// hierarchie partagee (les blocs vivent sur TOUS les niveaux), suivi d'une cascade fin -> grossier.
  ///
  /// INVARIANT DE COUVERTURE (#169) : la source a ete appliquee independamment sur CHAQUE niveau, donc
  /// une cellule grossiere COUVERTE par un patch fin porterait sinon sa propre source grossiere, sans
  /// rapport avec la source vue par ses enfants fins. Une cellule grossiere couverte DOIT etre la
  /// moyenne 2x2 de ses enfants (elle ne represente pas de matiere a elle seule). On restaure cette
  /// coherence par la MEME cascade fin -> grossier (mf_average_down_mb) que solve_fields et le moteur
  /// compile-time : sans elle, le diagnostic de masse (somme du seul grossier) compterait une source
  /// grossiere fantome sous le patch. Hierarchie mono-niveau : aucune cellule couverte, les boucles de
  /// cascade ne s'executent pas -> bit-identique au cas sans patch.
  ///
  /// CONSERVATION PAR CELLULE : a un niveau donne, chaque terme ecrit out(i,j,comp) += dt * S(reg(i,j))
  /// sur la MEME cellule (i,j) lue par les entrees ; un echange add_pair pose +S sur un bloc et -S sur
  /// l'autre AU MEME (i,j), donc la somme des deux blocs est inchangee cellule par cellule. Sans source
  /// enregistree (coupled_sources_ vide) : no-op total -> trajectoire bit-identique a l'historique.
  void coupled_source_step(Real dt) {
    if (coupled_sources_.empty()) return;  // opt-in : aucune source -> chemin bit-identique
    solve_fields();  // aux par niveau a jour (un terme peut lire phi/grad via une entree future)
    for (const auto& cs : coupled_sources_) {
      // Application PAR NIVEAU : a chaque niveau k, les blocs partagent EXACTEMENT le meme layout
      // (garde same_layout_or_throw), donc meme local_size() et meme indexation locale -> on itere en
      // parallele sur les fabs locaux. local_size()==0 sur un rang sans boite -> boucle vide (MPI-safe).
      for (int k = 0; k < nlev_; ++k) {
        const int sref = cs.n_in > 0 ? cs.ins[0].block : cs.outs[0].block;
        MultiFab& Uref = (*blocks_[static_cast<std::size_t>(sref)].levels)[k].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = cs.n_in;
          kern.n_const = cs.n_const;
          kern.n_terms = cs.n_terms;
          for (int c = 0; c < cs.n_in; ++c) {
            kern.in[c] = (*blocks_[static_cast<std::size_t>(cs.ins[static_cast<std::size_t>(c)].block)]
                               .levels)[k].U.fab(li).array();
            kern.in_comp[c] = cs.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < cs.n_const; ++c) kern.consts[c] = cs.kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < cs.n_terms; ++t) {
            kern.out[t] = (*blocks_[static_cast<std::size_t>(cs.outs[static_cast<std::size_t>(t)].block)]
                                .levels)[k].U.fab(li).array();
            kern.out_comp[t] = cs.outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = cs.outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li), kern);  // foncteur NOMME (device-clean), additif forward-Euler
        }
      }
      // Restaure la coherence des cellules grossieres couvertes (cf. INVARIANT DE COUVERTURE ci-dessus).
      for (auto& b : blocks_)
        for (int k = nlev_ - 1; k >= 1; --k) mf_average_down_mb((*b.levels)[k].U, (*b.levels)[k - 1].U);
    }
  }

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

  /// REGRID D'UNION DES TAGS (capstone Phase 2, C.6 ; docs/AMR_REGRID_UNION_TAGS_DESIGN.md, etapes
  /// R0-R8). Re-grille la hierarchie PARTAGEE a partir de l'UNION (OU cellule a cellule) des tags de
  /// TOUS les blocs (predicat par bloc, D1) + des tags de phi (sur |grad phi|, D4), suivie d'UN SEUL
  /// clustering Berger-Rigoutsos -> UN SEUL nouveau layout fin applique a TOUS les blocs (y compris
  /// ceux tenus par leur stride, D3) ET a l'aux partage. Maintient la PRECONDITION de layout partage
  /// (same_layout_or_throw) apres le regrid. v1 a 2 NIVEAUX (grossier + 1 fin, D5) : no-op si nlev < 2.
  /// No-op (grille inchangee) si l'union des tags est vide (rien a raffiner).
  void regrid() {
    if (nlev_ < 2) return;  // 2 niveaux requis (D5) : rien a re-griller en mono-niveau
    const int fk = nlev_ - 1, pk = fk - 1;  // fin + son parent (pk == 0 en v1 a 2 niveaux)

    // (R0) PRECONDITION : champs a jour (aux par niveau, pour le critere |grad phi|). Le snapshot de
    // masse par bloc n'est PAS necessaire au moteur (la conservation est verifiee cote test V1).
    solve_fields();

    // (R1)+(R2) TAGS PAR BLOC (sur U du bloc au niveau parent) + TAGS DE PHI (sur l'aux partage).
    const int PNX = dom_.nx() << pk, PNY = dom_.ny() << pk;
    const Box2D pdom = Box2D::from_extents(PNX, PNY);
    std::vector<TagBox> parts;
    parts.reserve(blocks_.size() + 1);
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      const TagPredicate& crit = block_tag_[b];
      if (!crit) continue;  // bloc sans critere : ne tague rien de son cote (re-grille comme fond)
      parts.push_back(tag_cells((*blocks_[b].levels)[pk].U, pdom, crit));
    }
    if (phi_tag_) parts.push_back(tag_cells(aux_[pk], pdom, phi_tag_));
    if (parts.empty()) return;  // aucun critere actif -> aucune cellule taguee -> grille inchangee

    // (R3) UNION (OU) des tags + dilatation (nesting + anticipation du deplacement des structures).
    TagBox grown = grow_tags(tag_union(parts), regrid_grow_, pdom);

    // (R4)+(R5) reduction collective cross-rang (si grossier reparti) + clustering UNIQUE -> layout fin
    // PARTAGE. all_reduce_or_inplace est appele DANS regrid_compute_fine_layout pour pk==0 reparti :
    // tous les rangs partent de la MEME grille de tags -> fb/dmap IDENTIQUES par rang (sinon MPI desync).
    auto [fb, dmap] = regrid_compute_fine_layout(std::move(grown), pdom, pk, regrid_margin_,
                                                 replicated_coarse_);
    if (fb.size() == 0) return;  // rien a raffiner : on garde la grille courante (no-op)

    // (R6) PROLONG / RESTRICT COHERENT de TOUS les blocs sur le MEME fb/dmap (y compris les blocs tenus
    // par leur stride : leur etat fige est present partout et contribue au Poisson, D3). La largeur de
    // ghost est HERITEE par bloc (un bloc MUSCL ordre 2 porte 2 ghosts ; un bloc Minmod et un VanLeer
    // peuvent differer), donc le schema ne lit pas hors bornes au pas suivant (V2 / risque X4).
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      const int ngf = L[fk].U.n_grow();
      L[fk].U = regrid_field_on_layout(fb, dmap, L[pk].U, L[fk].U, pk, ngf, replicated_coarse_);
    }

    // (R7) REBUILD DE L'AUX PARTAGE (un seul, largeur aux_ncomp_) sur le nouveau layout + RE-CABLAGE
    // du pointeur aux de CHAQUE bloc. L'adresse &aux_[fk] reste stable (reallocation en place du
    // MultiFab dans le std::vector existant) -> les pointeurs des autres niveaux ne bougent pas.
    aux_[fk] = MultiFab(fb, dmap, aux_ncomp_, 1);
    for (auto& b : blocks_)
      (*b.levels)[fk].aux = &aux_[fk];

    // (V3) INVARIANT DE LAYOUT PARTAGE : tous les blocs DOIVENT vivre sur EXACTEMENT le meme fb/dmap
    // (boites, ordre, rang par boite) apres le regrid. Garde-fou collectif (cross-bloc) ; attrape toute
    // reconstruction incoherente avant qu'elle ne corrompe l'aux partage / le Poisson somme.
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_) ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // (R8) RESTAURATION DE L'INVARIANT DE COUVERTURE : re-solve pour que phi / grad phi soient
    // coherents avec la nouvelle grille ET pour declencher la cascade fin -> grossier (mf_average_down_mb,
    // dans solve_fields) qui restaure les cellules grossieres couvertes (sinon un diagnostic de masse,
    // somme du seul grossier, compterait une valeur grossiere fantome sous le nouveau patch, X5).
    solve_fields();
    ++regrid_count_;
  }

  /// Avance le systeme d'un macro-pas dt. On resout d'abord les champs (Poisson somme co-localise, UNE
  /// fois par macro-pas : cadence OncePerStep), puis chaque bloc avance sur SA pile de niveaux avec SON
  /// schema, en honorant sa cadence stride et ses substeps, et SON traitement temporel. Pendant runtime
  /// de AmrSystemCoupler::step (OncePerStep) : la version compile-time porte substeps/stride dans
  /// block_substeps_v / block_stride_v et choisit le traitement par le constexpr block_time_treatment_v ;
  /// ici le moteur porte la boucle de sous-pas, le filtre stride ET la selection IMEX-vs-explicite.
  ///
  /// SELECTION DU TRAITEMENT (capstone vii) :
  ///  - bloc EXPLICITE (b.imex == false) : la fermeture advance fait UN advance_amr (transport + source
  ///    en Euler avant), appelee substeps fois ;
  ///  - bloc IMEX (b.imex == true) : la fermeture imex_advance fait UN advance_amr SOURCE-FREE puis la
  ///    source raide IMPLICITE backward_euler_source par niveau + cascade (cf. AmrRuntimeBlock::imex_advance),
  ///    appelee substeps fois. Inconditionnellement stable sur une relaxation raide (la ou l'explicite,
  ///    de facteur |1 - dt/eps|, DIVERGE des que dt > 2 eps).
  /// La boucle de sous-pas est COMMUNE aux deux traitements (substeps applications de h = bdt/substeps),
  /// donc le runtime SOUS-CYCLE aussi le splitting IMEX. A substeps=1 ce sous-cyclage est un no-op et le
  /// chemin IMEX coincide avec la branche IMEX du moteur compile-time AmrSystemCoupler::step ; pour
  /// substeps>1 il DIVERGE deliberement de ce moteur (qui, lui, ignore substeps sur sa branche IMEX) :
  /// voir SEMANTIQUE IMEX SOUS substeps en en-tete (CFL-safe sur le transport, backward-Euler stable a
  /// tout pas, relaxation raide plus precise). imex == false partout -> chemin advance seul ->
  /// trajectoire bit-identique a l'historique (l'IMEX est opt-in).
  void step(Real dt) {
    solve_count_ = 0;
    // REGRID D'UNION DES TAGS (capstone Phase 2, C.6 ; D2 : AVANT le step du macro-pas, coherent avec
    // le mono-bloc amr_dsl_block.hpp:108). Cadence regrid_every_ en MACRO-PAS, HORS des boucles de
    // substeps et des fenetres de stride (granularite macro-pas SEULEMENT, D3). regrid_every_ == 0 ->
    // hierarchie FIGEE, regrid jamais appele -> trajectoire BIT-IDENTIQUE a l'historique. Le garde
    // macro_step_ > 0 (comme le mono-bloc) evite un regrid au tout premier pas (la grille initiale est
    // deja celle du build). Le regrid se place AVANT solve_fields ci-dessous : il fait son propre
    // solve_fields (R0/R8), puis le solve_fields du step recalcule phi sur la grille re-grillee.
    if (regrid_every_ > 0 && macro_step_ > 0 && macro_step_ % regrid_every_ == 0) regrid();
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
      // substeps sous-pas egaux de bdt/substeps. La fermeture choisie fait UN advance par appel ;
      // substeps=1 -> un seul advance de bdt (bit-identique au cas mono-substep). SELECTION du
      // traitement par bloc : IMEX (transport source-free + source raide implicite, calque la branche
      // IMEX de AmrSystemCoupler::step) si b.imex, sinon EXPLICITE (transport + source Euler avant). Le
      // test est PAR BLOC et stable : un seul bloc IMEX ne change rien aux blocs explicites voisins.
      // NOTE substeps>1 : la boucle ci-dessous appelle step_block substeps fois pour LES DEUX
      // traitements, donc le splitting IMEX est SOUS-CYCLE (K pas de Lie sur bdt/K). Le compile-time, lui,
      // n'applique son IMEX qu'une fois sur bdt (il ignore substeps sur sa branche IMEX) : divergence
      // ASSUMEE et saine pour substeps>1 (cf. SEMANTIQUE IMEX SOUS substeps en en-tete du fichier).
      const Real h = bdt / static_cast<Real>(b.substeps);
      auto& step_block = b.imex ? b.imex_advance : b.advance;
      for (int s = 0; s < b.substeps; ++s)
        step_block(*b.levels, dom_, h, base_per_, replicated_coarse_);
    }
    // Sources couplees inter-especes APRES le transport (meme ordre que AmrSystemCoupler : transport
    // puis coupled_source_step), par splitting forward-Euler. No-op si aucune source enregistree ->
    // trajectoire bit-identique a l'historique (la feature est opt-in).
    coupled_source_step(dt);
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
    last_dt_reason_ = "degenerate";
    for (auto& b : blocks_) {
      const Real w = std::max(b.max_speed((*b.levels)[0].U, aux_[0]), kCflSpeedFloor);
      Real dt_b = cfl * h * static_cast<Real>(b.substeps) /
                  (static_cast<Real>(b.stride) * w);
      const char* why = "transport";
      // BORNES OPTIONNELLES du bloc (StabilityPolicy AMR, audit 2026-06) : memes formules
      // substeps/stride que SystemStepper::step_cfl, evaluees sur le GROSSIER. Fermetures vides
      // (modele sans trait) -> non interrogees, borne transport seule (bit-identique).
      if (b.source_frequency) {
        const Real mu = b.source_frequency((*b.levels)[0].U, aux_[0]);
        if (mu > Real(0)) {
          const Real dt_src = cfl * static_cast<Real>(b.substeps) /
                              (static_cast<Real>(b.stride) * mu);
          if (dt_src < dt_b) { dt_b = dt_src; why = "source_frequency"; }
        }
      }
      if (b.stability_dt) {
        const Real db = b.stability_dt((*b.levels)[0].U, aux_[0]);
        if (db > Real(0)) {
          const Real dt_adm = db * static_cast<Real>(b.substeps) / static_cast<Real>(b.stride);
          if (dt_adm < dt_b) { dt_b = dt_adm; why = "stability_dt"; }
        }
      }
      if (dt_b < dt) { dt = dt_b; last_dt_reason_ = std::string(why) + ":" + b.name; }
    }
    // Bornes GLOBALES (AmrRuntime::add_dt_bound, parite avec System::add_dt_bound) : evaluees PAR
    // RANG puis reduites all_reduce_min (dt identique sur tous les rangs ; <= 0/non finie = inerte).
    for (const auto& g : dt_bounds_) {
      if (!g.fn) continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v)) v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (static_cast<Real>(v) < dt) { dt = static_cast<Real>(v); last_dt_reason_ = "global:" + g.label; }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * h / kCflSpeedFloor;  // garde-fou (aucun bloc : impossible ici)
      last_dt_reason_ = "degenerate";
    }
    step(dt);
    return dt;
  }

  /// Borne GLOBALE de pas (pendant AMR de System::add_dt_bound) : fn() evaluee une fois par
  /// step_cfl, all_reduce_min, <= 0/non finie = inerte. Pour couplage/scheduler/politiques user.
  void add_dt_bound(const std::string& label, std::function<double()> fn) {
    dt_bounds_.push_back(GlobalDtBound{label, std::move(fn)});
  }

  /// Borne ACTIVE du dernier step_cfl ("transport:<bloc>" / "source_frequency:<bloc>" /
  /// "stability_dt:<bloc>" / "global:<label>" / "degenerate" / "" avant le premier pas).
  const std::string& last_dt_bound() const { return last_dt_reason_; }

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

  // Empreintes index-space des patchs fins (level + coins lo/hi inclusifs), pour TOUS les niveaux
  // fins. Lecture seule du BoxArray GLOBAL (toutes boites/tous rangs) deja stocke -> rank-independent,
  // zero communication, AUCUN cout chemin chaud (query entre les pas). Mirroir de n_patches() : la
  // meme box_array() qui donne le COMPTE donne les BOITES. Bloc 0 representatif (layout PARTAGE, garde
  // same_layout_or_throw). Boucle k = 1..nlev-1 : un seul niveau fin aujourd'hui (ratio 2), correct si
  // un futur ajoute des niveaux (le champ level desambiguise le pas dx = L / (n << level) cote Python).
  std::vector<PatchBox> patch_boxes() const {
    const auto& L = *blocks_[0].levels;
    std::vector<PatchBox> out;
    for (int k = 1; k < static_cast<int>(L.size()); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const Box2D& b : bxs)
        out.push_back(PatchBox{k, b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  }

 private:
  // Index du bloc nomme @p name dans le registre (-1 si absent). Pendant de AmrSystem::Impl::block_index
  // (la facade nomme les blocs ; les sources couplees les ciblent par nom, resolu une fois a l'enregistrement).
  int block_index(const std::string& name) const {
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (blocks_[i].name == name) return static_cast<int>(i);
    return -1;
  }

  // Reference resolue d'un champ d'une source couplee : (indice de bloc, composante) + le programme
  // bytecode du terme (vide pour une entree). Les entrees ne portent que block/comp ; les sorties
  // portent en plus le programme postfixe evalue par cellule. On capture l'INDICE de bloc (pas un
  // pointeur de fab) : les Array4 sont reconstruits a chaque application, par niveau.
  struct CsRef {
    int block;
    int comp;
    CsProgram prog;  // sorties : programme du terme ; entrees : inutilise (CsProgram{})
  };
  // Une source couplee enregistree : ses entrees, ses termes de sortie et ses constantes, pretes a etre
  // marshalees dans un CoupledSourceKernel par niveau / par fab a l'application.
  struct CoupledSourceSpec {
    std::vector<CsRef> ins;
    std::vector<CsRef> outs;
    std::vector<Real> kconsts;
    int n_in = 0;
    int n_const = 0;
    int n_terms = 0;
  };

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  GeometricMG mg_;
  std::vector<AmrRuntimeBlock> blocks_;
  // Bornes GLOBALES de pas (add_dt_bound, parite System) + borne ACTIVE du dernier step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  std::string last_dt_reason_;
  std::vector<MultiFab> aux_;  // [niveau], partage par tous les blocs
  std::vector<CoupledSourceSpec> coupled_sources_;  // sources couplees enregistrees (appliquees apres transport)
  // REGRID D'UNION DES TAGS (capstone Phase 2, C.6). regrid_every_ == 0 -> hierarchie FIGEE (defaut,
  // bit-identique). block_tag_ : predicat de tag PAR BLOC (D1 ; meme taille que blocks_, vide = ce
  // bloc ne tague rien de son cote). phi_tag_ : predicat de tag de phi sur |grad phi| (D4 ; vide = phi
  // ne contribue pas a l'union).
  std::vector<TagPredicate> block_tag_;
  TagPredicate phi_tag_;
  int regrid_every_ = 0;
  int regrid_grow_ = 2;
  int regrid_margin_ = 2;
  int aux_ncomp_ = kAuxBaseComps;
  int nlev_ = 0;
  int macro_step_ = 0;
  mutable int solve_count_ = 0;
  int regrid_count_ = 0;
};

}  // namespace adc
