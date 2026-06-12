#pragma once

#include <adc/core/state.hpp>       // kAuxBaseComps (canal aux par defaut de l'etage Schur : B_z)
#include <adc/core/types.hpp>       // Real
#include <adc/core/variables.hpp>   // VariableSet (descripteur a roles porte par chaque bloc)
#include <adc/mesh/box2d.hpp>       // Box2D
#include <adc/mesh/for_each.hpp>    // device_fence (le marshaling synchronise le device avant de lire l'hote)
#include <adc/mesh/multifab.hpp>    // MultiFab, Array4, ConstArray4

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief SystemBlockStore : la responsabilite GESTION DE BLOCS extraite du god-class System::Impl
///        (audit Lot B.3, derniere extraction P0 ; suite de SystemFieldSolver #176 et SystemStepper).
///        Extrait VERBATIM de python/system.cpp : aucune modification de numerique, de layout, d'ordre
///        d'iteration, d'indice, ni de message d'erreur. STRICTEMENT bit-identique -- le code est
///        deplace tel quel.
///
/// CONTRAT / INVARIANTS
/// - POSSEDE : la struct BlockState (ex-Species : etat U + descripteurs + fermetures du bloc) et le
///   registre ordonne des blocs (vector<BlockState>), UNIQUE source de verite peuplee par tous les
///   chemins d'ajout (add_block / install_block ; add_dynamic_block / add_compiled_block / add_native_block
///   via native_loader).
/// - EXPOSE : index(name) (indice 0-base ou erreur), find(name) (const + non const, reference au bloc ou
///   erreur), et les helpers de MARSHALING d'etat copy_comp0 / copy_state / write_state (recopie hote <->
///   MultiFab, device_fence inclus). L'ORDRE d'insertion fixe l'indexation et donc l'iteration : il est
///   PRESERVE (push_back en queue, jamais de tri ni de remaniement).
/// - MESSAGES D'ERREUR INCHANGES : "System : bloc inconnu '...'" (index/find) et
///   "System::set_state : taille != ncomp*n*n" (write_state) sont repris VERBATIM (la non-regression des
///   tests de facade en depend).
/// - NE POSSEDE PAS : le domaine (ba/dm/dom/geom/pgeom_), l'aux et sa largeur, le Poisson/elliptique, les
///   couplages, t/macro_step_. Ces concerns restent dans System::Impl (ou dans SystemFieldSolver /
///   SystemStepper) ; le store n'en sait rien.
///
/// Le registre `blocks` est PUBLIC : System::Impl l'expose tel quel via un membre reference `sp` (alias),
/// de sorte que les gabarits en-tete deja extraits (SystemFieldSolver, SystemStepper, native_loader) qui
/// iterent `owner_->sp` / `P->sp` et nomment `Impl::Species` restent INCHANGES et bit-identiques. La
/// struct est nommee BlockState (sens plus clair que l'historique "Species") ; Impl en garde l'alias
/// `using Species = SystemBlockStore::BlockState;` pour la compatibilite des gabarits.

namespace adc {

/// Forward-declaration : BlockState porte un shared_ptr<CondensedSchurSourceStepper> (etage source
/// condense OPT-IN). Le pointeur seul suffit ici ; la definition vit dans le header de couplage, inclus
/// la ou l'etage est reellement construit (python/system.cpp::set_source_stage).
class CondensedSchurSourceStepper;
/// Forward-declaration : pendant POLAIRE de l'etage source condense (anneau (r, theta)). Un BlockState
/// porte AU PLUS UN des deux steppers (cartesien OU polaire), selon la geometrie du System, choisi par
/// set_source_stage. Le pointeur seul suffit ici (cf. PolarCondensedSchurSourceStepper, Voie A etape 2c).
class PolarCondensedSchurSourceStepper;

/// Registre ORDONNE des blocs du System + helpers de marshaling d'etat. Voir contrat ci-dessus (POSSEDE
/// BlockState + le vector ; EXPOSE index/find + copy/write_state ; NE POSSEDE PAS le domaine/aux/Poisson).
class SystemBlockStore {
 public:
  /// Type-erase de la conversion PONCTUELLE (une cellule) cons <-> prim d'un bloc : in/out sont des
  /// tableaux de ncomp doubles. MEME type que System::CellConvert (std::function identique) : l'assignation
  /// depuis set_block_conversion / native_loader reste un move trivial.
  using CellConvert = std::function<void(const double* in, double* out)>;

  /// Fermetures compilees figees a l'ajout du bloc (modele compose + schema spatial + temps).
  /// Type-erased SEULEMENT au niveau de la liste de blocs ; le noyau reste compile.
  /// ORDRE DES MEMBRES FIGE : install_block (python/system.cpp) et native_loader (push_dynamic /
  /// add_compiled_model) initialisent ce struct par AGREGAT positionnel
  /// {name, U, ncomp, substeps, evolve, stride, gamma, advance, rhs_into, max_speed, add_poisson_rhs} ;
  /// ne pas reordonner ces membres ni en inserer avant add_poisson_rhs.
  struct BlockState {
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;                                             // sous-pas statiques (add_block)
    bool evolve;                                              // false = espece gelee (fond fixe non avance)
    int stride = 1;                                           // cadence : avance 1 fois tous les stride macro-pas
    double gamma;                                             // pour l'energie au repos (4 var)
    std::function<void(MultiFab&, Real, int)> advance;        // (U, dt, n) : n sous-pas de dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;        // R <- -div F + S (Poisson fige)
    std::function<Real(const MultiFab&)> max_speed;           // max |vitesse d'onde| du bloc
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
    // Descripteur des variables conservatives / primitives (noms + ROLES physiques) du bloc.
    // Les roles (fournis par M::conservative_vars()) permettent aux couplages inter-especes de
    // viser une composante par son SENS (qte de mvt, energie) au lieu d'un indice u[1]/u[3] en dur.
    VariableSet cons_vars, prim_vars;
    // Conversions PONCTUELLES cons <-> prim DU MODELE du bloc (une cellule, ncomp doubles in/out).
    // Posees a l'ajout (install_block / push_dynamic) depuis le modele reel ; vides -> identite (le
    // modele n'expose pas de conversion, p.ex. scalaire pur ou .so genere avant ce chantier).
    // Consommees par set_primitive_state / get_primitive_state (init/diagnostic en primitif).
    CellConvert prim_to_cons, cons_to_prim;
    // ETAGE SOURCE condense par Schur (OPT-IN, adc.Split(source=CondensedSchur), cf. set_source_stage).
    // nullptr (defaut) = pas d'etage source condense : le bloc avance EXACTEMENT comme avant
    // (bit-identique). Non nul = apres le transport hyperbolique, le pas joue l'etage source autonome
    // (CondensedSchurSourceStepper, #126) en lieu et place de la source explicite / IMEX. shared_ptr :
    // garde BlockState MOVABLE (le stepper porte un GeometricMG, ni copiable ni movable simplement).
    std::shared_ptr<CondensedSchurSourceStepper> schur;
    // Pendant POLAIRE de l'etage source condense (anneau (r, theta), Voie A etape 2c). Exclusif avec
    // schur ci-dessus : set_source_stage construit l'UN ou l'AUTRE selon la geometrie du System (polaire
    // -> schur_polar, cartesien -> schur). run_source_stage dispatche sur celui qui est non nul. Le
    // chemin cartesien reste BIT-IDENTIQUE (schur_polar == nullptr quand le System est cartesien).
    std::shared_ptr<PolarCondensedSchurSourceStepper> schur_polar;
    double schur_theta = 0.5;  // theta-schema de l'etage source (0.5 = Crank-Nicolson)
    // Composante du canal aux lue comme champ magnetique Omega = B_z par l'etage Schur (audit
    // vague 2 : champ TRANSPORTE dans l'ABI au lieu du litteral kAuxBaseComps fige). Defaut =
    // kAuxBaseComps (canal canonique B_z), bit-identique ; set_source_stage peut le rediriger.
    int schur_bz_comp = kAuxBaseComps;
    // ETAGE SOURCE GENERIQUE (optionnel) : un callable (U, dt) qui avance EN PLACE l'etage source du
    // bloc. nullptr (defaut) = aucun etage source generique (chemin BIT-IDENTIQUE). run_source_stage le
    // joue UNIQUEMENT si aucun etage Schur condense (schur / schur_polar) n'est cable, donc il ne change
    // RIEN au chemin Schur de production. Sert au splitting generique (adc.Strang sur un etage source
    // arbitraire) et aux tests d'ordre temporel du stepper (operateurs jouets non commutants). Trailing
    // + defaut nullptr : l'init par agregat positionnel des autres membres reste inchangee.
    std::function<void(MultiFab&, Real)> source_step;
    // AVANCES DE TRANSPORT DISQUE (chantier T5-PR3, OPT-IN). Vides (defaut) -> aucun routage disque :
    // le stepper avance via `advance` (chemin plein cartesien, BIT-IDENTIQUE). Non vides, elles MIMENT
    // `advance` (meme schema RK / IMEX, meme limiteur / flux) mais aiguillent le residu de transport
    // vers l'operateur disque, et ne sont SELECTIONNEES que si le System est en mode Staircase (resp.
    // CutCell) ET qu'un disque est fixe (set_disc_domain). Construites a l'ajout du bloc (build_block)
    // EN MEME TEMPS que `advance`, elles lisent le masque / level set du System par pointeur au pas
    // (adresse stable) : l'ordre add_block / set_disc_domain est indifferent. Trailing + defaut vide :
    // l'init par agregat positionnel des autres membres reste inchangee.
    std::function<void(MultiFab&, Real, int)> advance_masked;  // residu via assemble_rhs_masked (Staircase)
    std::function<void(MultiFab&, Real, int)> advance_eb;      // residu via assemble_rhs_eb (CutCell)
    // DIAGNOSTIC dt_hotspot (ADC-182) : (U, w, i, j) -> cellule GLOBALE dominant la borne CFL de
    // transport du bloc + sa vitesse w = max(wx, wy). A LA DEMANDE seulement (System::dt_hotspot) :
    // jamais interrogee par step/step_cfl (chemin chaud bit-identique). Trailing + defaut vide.
    std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
    // BORNES DE PAS OPTIONNELLES du bloc (audit 2026-06, chantier step_cfl). VIDES (defaut) -> le
    // stepper ne les interroge pas : politique de pas STRICTEMENT historique (transport seul,
    // bit-identique). Posees par set_block_dt_bounds quand le modele declare les traits
    // HasSourceFrequency / HasStabilityDt (cf. core/physical_model.hpp pour la semantique).
    // Trailing + defaut vide : l'init par agregat positionnel des autres membres reste inchangee.
    std::function<Real(const MultiFab&)> source_frequency;  // max cellules de mu [1/s] (0 = ne contraint pas)
    std::function<Real(const MultiFab&)> stability_dt;      // min cellules du pas admissible (0 = ne contraint pas)
  };

  /// Registre ORDONNE des blocs (UNIQUE source de verite). PUBLIC : Impl l'aliase en `sp` pour les
  /// gabarits deja extraits (SystemFieldSolver / SystemStepper / native_loader) qui iterent owner_->sp.
  std::vector<BlockState> blocks;

  // --- acces par NOM (indexation 0-base, ordre d'insertion) ------------------------------------------
  /// Reference au bloc @p name (en ecriture). @throws std::runtime_error "System : bloc inconnu '...'".
  BlockState& find(const std::string& name) {
    for (auto& s : blocks)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// Reference au bloc @p name (en lecture). @throws std::runtime_error "System : bloc inconnu '...'".
  const BlockState& find(const std::string& name) const {
    for (auto& s : blocks)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// Indice 0-base du bloc @p name (ordre d'insertion). @throws std::runtime_error si inconnu.
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < blocks.size(); ++k)
      if (blocks[k].name == name) return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  /// Nombre de blocs enregistres.
  int size() const { return static_cast<int>(blocks.size()); }
  /// Noms des blocs, dans l'ordre d'insertion (UNIQUE source : tous les chemins d'ajout y figurent).
  std::vector<std::string> names() const {
    // Lit le registre de blocs UNIQUE, peuple par tous les chemins d'ajout : un bloc charge via
    // add_dynamic_block / add_compiled_block (.so) y figure au meme titre qu'un add_block.
    std::vector<std::string> out;
    out.reserve(blocks.size());
    for (const auto& s : blocks) out.push_back(s.name);
    return out;
  }

  // --- marshaling d'etat (recopie hote <-> MultiFab ; device_fence inclus) ---------------------------
  /// Recopie la composante 0 du fab(0) (densite) en row-major (j lent, i rapide). device_fence prealable :
  /// le marshaling lit l'hote, il faut que le device ait fini d'ecrire U.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    // MPI mono-box : la box vit sur le rang proprietaire (rang 0). Un rang sans box (local_size()==0)
    // n'a PAS de fab(0) -> retour VIDE plutot qu'un acces HORS BORNES (UB). Mono-rang : local_size()
    // vaut toujours 1, comportement INCHANGE. Pour le champ global multi-rangs : System::density_global.
    if (mf.local_size() == 0) return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }
  /// Recopie les ncomp composantes du fab(0) en layout composante-majeur (c lent, puis j, puis i).
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    // Rang sans box (MPI mono-box, non proprietaire) : retour VIDE (pas de fab(0)). Cf. copy_comp0 ;
    // le champ global multi-rangs passe par System::state_global (gather collectif).
    if (mf.local_size() == 0) return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, c));
    return out;
  }
  /// Ecrit les ncomp composantes dans le fab(0) depuis un tampon composante-majeur (meme layout que
  /// copy_state). @throws std::runtime_error "System::set_state : taille != ncomp*n*n" si la taille
  /// ne correspond pas a ncomp*nx*ny (message inchange).
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    // Rang sans box (MPI mono-box, non proprietaire) : NO-OP (pas de fab(0) a ecrire). Permet a
    // sim.set_state / sim.restart d'etre appeles sur TOUS les rangs avec le champ GLOBAL : seul le
    // rang proprietaire (rang 0, box = domaine complet) ecrit, les autres ne font rien. Mono-rang :
    // local_size()==1, validation + ecriture INCHANGEES (bit-identique).
    if (mf.local_size() == 0) return;
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) u(i, j, c) = in[k++];
  }
};

}  // namespace adc
