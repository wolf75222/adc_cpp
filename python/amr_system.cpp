#include <adc/runtime/amr_system.hpp>

#include <adc/runtime/abi_key.hpp>         // detail::abi_key_string : cle d'ABI (header-only), comparee a celle du loader
#include <adc/runtime/amr_dsl_block.hpp>   // detail::dispatch_amr_compiled + build_amr_compiled (chemin partage)
#include <adc/runtime/amr_runtime.hpp>     // AmrRuntime + AmrRuntimeBlock (moteur multi-blocs runtime)
#include <adc/runtime/model_factory.hpp>   // detail::dispatch_model + briques compilees
#include <adc/runtime/wall_predicate.hpp>  // detail::wall_predicate (paroi partagee System/AmrSystem)

#include <cmath>
#include <cstddef>
#include <dlfcn.h>  // dlopen/dlsym : chargement du loader natif AMR genere (.so)
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace adc {

struct AmrSystem::Impl {
  AmrSystemConfig cfg;

  // Specification d'UN bloc (figee a add_block, materialisee au build paresseux). Le facade detient
  // un REGISTRE PAR NOM de blocs (cf. design AMR_MULTIBLOCK_DESIGN.md section 0/3) : le chemin
  // mono-bloc passe par AmrCouplerMP (intouche, bit-identique) ; deux blocs ou plus passent par le
  // moteur runtime multi-blocs AmrRuntime (hierarchie partagee, Poisson somme co-localise).
  struct BlockSpec {
    std::string name;
    // Chemin ModelSpec natif (briques composees) OU chemin compile (.so / add_compiled_model).
    bool is_compiled = false;
    ModelSpec spec;                 // chemin ModelSpec (is_compiled == false)
    std::string limiter = "minmod", riemann = "rusanov";
    bool recon_prim = false;        // recon == "primitive"
    bool imex = false;              // time == "imex" : source raide implicite
    int substeps = 1;
    int stride = 1;                 // cadence hold-then-catch-up (multi-blocs ; cf. AmrRuntimeBlock)
    double gamma = 1.4;
    // Chemin compile MONO-BLOC : builder type-erase (AmrCompiledHooks d'un AmrCouplerMP concret),
    // invoque au build paresseux. Le multi-blocs compile (DSL production) est une PR ulterieure.
    std::function<AmrCompiledHooks(const AmrBuildParams&)> compiled_hooks_builder;
    // Densite initiale du bloc (composante 0), n*n row-major ; ciblee par set_density(name, rho).
    bool has_density = false;
    std::vector<double> density;
  };

  std::vector<BlockSpec> blocks;
  double gamma = 1.4;  // gamma du PREMIER bloc (compat : lu par le chemin mono-bloc)

  double refine_threshold = 1e30;  // 1e30 => aucun raffinement par defaut

  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto",
              p_wall = "none";
  double p_wall_radius = 0.0;

  bool built = false;
  // --- chemin mono-bloc (AmrCouplerMP, intouche : bit-identique a l'historique) ---
  std::shared_ptr<void> coupler_holder;  // maintient en vie l'AmrCouplerMP<Model> des hooks
  std::function<void(double)> step_fn;
  std::function<double()> max_speed_fn;
  std::function<double()> mass_fn;
  std::function<int()> n_patches_fn;
  std::function<std::vector<double>()> density_fn;
  std::function<std::vector<double>()> potential_fn;
  // --- chemin multi-blocs (AmrRuntime, hierarchie partagee + Poisson somme) ---
  std::shared_ptr<adc::AmrRuntime> runtime;
  double t = 0;

  explicit Impl(const AmrSystemConfig& c) : cfg(c) {}

  // Index du bloc nomme @p name dans le registre (-1 si absent). Nom vide -> premier bloc (compat
  // mono-bloc : le nom etait cosmetique). Plusieurs blocs sans nom n'est pas ambigu a l'ajout (un
  // nom vide cible toujours le bloc 0), mais set_density(name) le resout precisement.
  int block_index(const std::string& name) const {
    if (name.empty()) return blocks.empty() ? -1 : 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
      if (blocks[i].name == name) return static_cast<int>(i);
    return -1;
  }

  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic") return b;
    if (mode == "dirichlet") { b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet; return b; }
    if (mode == "neumann") { b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap; return b; }
    throw std::runtime_error("AmrSystem::set_poisson : bc inconnu '" + mode + "'");
  }
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, cfg.L, "AmrSystem::set_poisson");
  }

  // Materialise les parametres figes du build paresseux du chemin MONO-BLOC (AmrCouplerMP) a partir
  // de la config + des choix refine/poisson/density + du SEUL bloc (blocks[0]). Communs aux deux
  // chemins mono-bloc (ModelSpec natif et add_compiled_model) : les deux instancient le MEME builder
  // partage (detail::build_amr_compiled). INTOUCHE par le multi-blocs -> mono-bloc bit-identique.
  AmrBuildParams make_build_params() {
    const BlockSpec& b = blocks[0];
    AmrBuildParams bp;
    bp.n = cfg.n;
    bp.L = cfg.L;
    bp.regrid_every = cfg.regrid_every;
    bp.gamma = b.gamma;
    bp.substeps = b.substeps;
    bp.recon_prim = b.recon_prim;
    bp.imex = b.imex;
    bp.refine_threshold = refine_threshold;
    bp.poisson_bc = poisson_bc();
    bp.wall = wall_active();
    bp.has_density = b.has_density;
    bp.density = b.density;
    bp.distribute_coarse = cfg.distribute_coarse;
    bp.coarse_max_grid = cfg.coarse_max_grid;
    return bp;
  }

  // Installe les fermetures type-erased du chemin MONO-BLOC (AmrCouplerMP).
  void install(AmrCompiledHooks&& h) {
    coupler_holder = std::move(h.coupler_holder);
    step_fn = std::move(h.step);
    max_speed_fn = std::move(h.max_speed);
    mass_fn = std::move(h.mass);
    n_patches_fn = std::move(h.n_patches);
    density_fn = std::move(h.density);
    potential_fn = std::move(h.potential);
    built = true;
  }

  bool multi_block() const { return blocks.size() >= 2; }

  // Construit le moteur runtime MULTI-BLOCS (AmrRuntime) : un SharedAmrLayout commun (hierarchie
  // partagee, figee), puis CHAQUE bloc materialise dessus son AmrRuntimeBlock type-erase (via son
  // block_builder, qui capture le Model/Limiter/Flux concret). Le Poisson grossier est SOMME et
  // CO-LOCALISE (Sum_b elliptic_rhs_b(U_b) lu aux memes cellules du grossier partage).
  void build_multi() {
    AmrBuildParams bp = make_build_params();  // geometrie + poisson_bc + wall + ownership communs
    const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
    std::vector<adc::AmrRuntimeBlock> rblocks;
    rblocks.reserve(blocks.size());
    for (auto& b : blocks) {
      if (b.is_compiled) {
        // chemin compile multi-blocs : non cable dans PR1 (bloc compile = mono-bloc AmrCouplerMP).
        throw std::runtime_error(
            "AmrSystem : un bloc compile (.so / add_compiled_model) en MULTI-BLOCS n'est pas "
            "cable dans cette version (PR1 : blocs natifs add_block ; DSL production multi-bloc = "
            "PR ulterieure)");
      }
      // Chemin ModelSpec natif : dispatch du modele -> type concret, puis dispatch schema spatial
      // -> build_amr_block (alloue la pile de niveaux du bloc sur le layout PARTAGE + fermetures).
      // La densite du bloc est portee par le BlockSpec (set_density(name) la cible).
      detail::dispatch_model(b.spec, [&](auto m) {
        rblocks.push_back(detail::dispatch_amr_block(m, b.limiter, b.riemann, S, b.name, b.density,
                                                     b.has_density, b.gamma, b.substeps,
                                                     b.recon_prim, b.imex, b.stride));
      });
    }
    runtime = std::make_shared<adc::AmrRuntime>(S.geom, S.ba_coarse, S.poisson_bc,
                                                std::move(rblocks), S.base_per, S.replicated_coarse,
                                                S.wall);
    built = true;
  }

  void ensure_built() {
    if (built) return;
    if (blocks.empty())
      throw std::runtime_error("AmrSystem : appeler add_block d'abord");
    // CONTRAINTE DURE PR1 (design section 0/4) : AmrSystemCoupler / AmrRuntime n'ont AUCUN regrid
    // (hierarchie FIGEE). Autoriser regrid_every > 0 en multi-blocs ferait croire a un AMR dynamique
    // inexistant (la grille ne bouge jamais) -> REFUS explicite. Le mono-bloc garde son regrid
    // (chemin AmrCouplerMP intouche). Le regrid d'union des tags multi-blocs est une PR ulterieure.
    if (multi_block() && cfg.regrid_every > 0)
      throw std::runtime_error(
          "AmrSystem : multi-blocs (>= 2 blocs) + regrid_every > 0 n'est pas supporte (la "
          "hierarchie multi-blocs est FIGEE a la construction ; le regrid d'union des tags est une "
          "etape ulterieure). Mettre regrid_every=0 pour une hierarchie multi-blocs figee, ou "
          "utiliser un seul bloc pour l'AMR dynamique.");
    if (multi_block()) { build_multi(); return; }

    // --- chemin MONO-BLOC (AmrCouplerMP, intouche : bit-identique a l'historique) ---
    const BlockSpec& b = blocks[0];
    const AmrBuildParams bp = make_build_params();
    if (b.is_compiled) {  // chemin compile : le builder fige les types (Model, Limiter, Flux)
      install(b.compiled_hooks_builder(bp));
      return;
    }
    // Chemin ModelSpec natif : la dispatch de modele resout le type concret, puis le MEME
    // dispatch schema spatial + build du coupleur que add_compiled_model (detail, partage).
    detail::dispatch_model(b.spec, [&](auto m) {
      install(detail::dispatch_amr_compiled(m, b.limiter, b.riemann, bp));
    });
  }
};

AmrSystem::AmrSystem(const AmrSystemConfig& c) : p_(std::make_unique<Impl>(c)) {}
AmrSystem::~AmrSystem() = default;
AmrSystem::AmrSystem(AmrSystem&&) noexcept = default;
AmrSystem& AmrSystem::operator=(AmrSystem&&) noexcept = default;

void AmrSystem::add_block(const std::string& name, const ModelSpec& model,
                          const std::string& limiter, const std::string& riemann,
                          const std::string& recon, const std::string& time, int substeps,
                          int stride) {
  if (p_->built)
    throw std::runtime_error("AmrSystem::add_block : le systeme est deja construit (appeler "
                             "add_block avant tout step/mass/density)");
  if (substeps < 1) throw std::runtime_error("AmrSystem::add_block : substeps >= 1");
  if (stride < 1) throw std::runtime_error("AmrSystem::add_block : stride >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("AmrSystem : time '" + time +
                             "' inconnu sur AMR (explicit|imex)");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem : recon inconnu '" + recon +
                             "' (conservative|primitive)");
  // MULTI-BLOCS PR1 : un 2e bloc (ou plus) bascule sur le moteur runtime AmrRuntime (hierarchie
  // partagee, Poisson somme co-localise). Le mono-bloc reste sur AmrCouplerMP (bit-identique).
  // Un bloc deja COMPILE (set_compiled_block) ne se melange pas a un bloc natif en multi-blocs
  // dans PR1 (le DSL production multi-bloc est une PR ulterieure).
  if (!p_->blocks.empty() && p_->blocks[0].is_compiled)
    throw std::runtime_error(
        "AmrSystem::add_block : melanger un bloc compile (.so / add_compiled_model) et un bloc "
        "natif (add_block) en multi-blocs n'est pas cable (PR1). Utiliser des blocs natifs "
        "(add_block) pour le multi-blocs.");
  // MULTI-BLOCS : le nom INDEXE le bloc (set_density/mass/density) -> il doit etre UNIQUE et non
  // vide des qu'il y a (ou qu'on cree) plus d'un bloc. Mono-bloc : le nom reste cosmetique.
  if (!p_->blocks.empty()) {
    if (name.empty())
      throw std::runtime_error("AmrSystem::add_block : en multi-blocs chaque bloc doit avoir un nom "
                               "NON vide (le nom indexe le bloc : set_density/mass/density)");
    if (p_->block_index(name) >= 0)
      throw std::runtime_error("AmrSystem::add_block : nom de bloc deja utilise '" + name +
                               "' (les noms de blocs doivent etre uniques en multi-blocs)");
  }
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = false;
  b.spec = model;
  b.limiter = limiter;
  b.riemann = riemann;
  b.recon_prim = (recon == "primitive");
  b.imex = (time == "imex");
  b.substeps = substeps;
  b.stride = stride;
  b.gamma = model.gamma;  // indice adiabatique du bloc (Euler), lu par coupler_write_coarse
  if (p_->blocks.empty()) p_->gamma = model.gamma;  // compat : gamma du 1er bloc (chemin mono-bloc)
  p_->blocks.push_back(std::move(b));
}

void AmrSystem::set_compiled_block(int ncomp, double gamma, int substeps,
                                   std::function<AmrCompiledHooks(const AmrBuildParams&)> builder) {
  (void)ncomp;  // le nombre de variables est porte par le Model concret (Model::n_vars) dans le
                // builder type-erase ; le parametre reste pour la symetrie d'API avec System.
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_compiled_block : le systeme est deja construit");
  // Un bloc compile en MULTI-BLOCS n'est pas cable dans PR1 (DSL production multi-bloc = PR
  // ulterieure). On reste donc MONO-BLOC compile (chemin AmrCouplerMP, intouche).
  if (!p_->blocks.empty())
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : un seul bloc compile (.so / add_compiled_model) est "
        "supporte (le multi-blocs compile / DSL production est une PR ulterieure ; pour le "
        "multi-blocs, utiliser des blocs natifs add_block)");
  p_->gamma = gamma;
  Impl::BlockSpec b;
  b.is_compiled = true;
  b.gamma = gamma;
  b.substeps = substeps;
  b.compiled_hooks_builder = std::move(builder);
  p_->blocks.push_back(std::move(b));
}

namespace {
// Ancre de module pour dladdr : son ADRESSE vit dans l'image qui contient amr_system.cpp (le module
// _adc, ou le binaire de test). add_native_block s'en sert pour localiser le module et le promouvoir
// en portee globale (RTLD_NOLOAD). Une fonction locale a la TU suffit (pas besoin d'export) et evite
// de dependre d'un symbole defini ailleurs (adc::abi_key, system.cpp).
void amr_native_anchor() {}
}  // namespace

void AmrSystem::add_native_block(const std::string& name, const std::string& so_path,
                                 const std::string& limiter, const std::string& riemann,
                                 const std::string& recon, const std::string& time, double gamma,
                                 int substeps) {
  if (substeps < 1) throw std::runtime_error("AmrSystem::add_native_block : substeps >= 1");
  // Validation AMONT du schema (comme add_block) : add_compiled_model(AmrSystem&) rejette deja
  // time hors {explicit, imex} et recon hors {conservative, primitive}, mais on diagnostique ICI une
  // faute de frappe avant la frontiere C++. time == "imex" => source raide traitee en IMPLICITE
  // (backward_euler_source), transport explicite porte par le reflux. limiter (dont weno5, cable #105)
  // et riemann (dont hllc/roe, cables a parite #113) sont valides par dispatch_amr_compiled dans le
  // loader (exception claire).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem::add_native_block : recon 'conservative' | 'primitive' "
                             "(recu '" + recon + "')");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("AmrSystem::add_native_block : time 'explicit' | 'imex' sur AMR (recu '" +
                             time + "')");
  // Chemin "production" du DSL cote AMR : le loader .so genere (emit_cpp_native_loader avec
  // target="amr_system") inline le gabarit en-tete add_compiled_model(AmrSystem&, ...), qui
  // materialise un AmrCouplerMP<Model> concret au build paresseux et installe ses hooks via
  // set_compiled_block -- chemin NATIF, MEME hierarchie AMR que add_block (reflux, regrid). Le loader
  // appelle donc set_compiled_block (methode hors-ligne de adc::AmrSystem) DEFINIE dans CE module ;
  // il faut la resoudre a travers le dlopen contre le module _adc deja charge.
  // PORTABILITE ELF (Linux) : CPython charge _adc en RTLD_LOCAL, donc ses symboles ne sont PAS dans
  // la portee globale. On PROMEUT le module courant en portee globale (RTLD_NOLOAD = sans le
  // recharger ; RTLD_GLOBAL OR'e dans les flags de l'objet deja charge), localise par dladdr sur une
  // ADRESSE de CE module : amr_native_anchor (fonction locale a cette TU). On evite ainsi de dependre
  // de adc::abi_key (defini dans system.cpp) -- ce qui couplerait au link tout test compilant
  // amr_system.cpp seul. Sur macOS, inoffensif (le loader resout par dynamic_lookup).
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&amr_native_anchor), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // RTLD_GLOBAL : place les symboles du loader dans la portee globale ET autorise le loader a resoudre
  // ses indefinis (set_compiled_block exportee ADC_EXPORT) contre les images deja chargees. RTLD_NOW :
  // resolution immediate -> un symbole AmrSystem manquant echoue ICI, pas en vol.
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error("AmrSystem::add_native_block : dlopen('" + so_path + "') : " +
                             std::string(e ? e : "?") +
                             " (le symbole adc::AmrSystem::set_compiled_block doit etre exporte ET le "
                             "module _adc charge globalement ; cf. ADC_EXPORT)");
  }
  // GARDE-FOU ABI EXPLICITE : la cle baked dans le loader (a SA compilation) doit egaler la cle du
  // module. Un ecart = en-tetes / compilateur / standard divergents -> agencement memoire de
  // AmrSystem/AmrBuildParams/AmrCompiledHooks potentiellement different a la frontiere -> UB. On leve
  // une erreur CLAIRE plutot que de laisser passer un loader incompatible. MEME symbole de cle que le
  // chemin System (adc_native_abi_key) : seul l'installateur (adc_install_native_amr) differe.
  auto key_fn = reinterpret_cast<const char* (*)()>(dlsym(h, "adc_native_abi_key"));
  if (!key_fn) {
    dlclose(h);
    throw std::runtime_error("AmrSystem::add_native_block : adc_native_abi_key absent du .so "
                             "(regenerer via dsl.compile_native(target='amr_system') / "
                             "compile(backend='production', target='amr_system'))");
  }
  const std::string loader_key = key_fn();
  // Cle du module = MEME calcul que adc::abi_key() (header-only detail::abi_key_string()) : evite la
  // dependance au symbole hors-ligne adc::abi_key (system.cpp). Le loader bake la sienne a SA compil.
  const std::string module_key = detail::abi_key_string();
  if (loader_key != module_key) {
    dlclose(h);
    throw std::runtime_error("AmrSystem::add_native_block : ABI incompatible -- cle du loader '" +
                             loader_key + "' != cle du module '" + module_key +
                             "'. Recompiler le loader avec le MEME compilateur, standard C++ et "
                             "en-tetes adc que le module _adc.");
  }
  // Installateur natif AMR du loader : reinterpret_cast<AmrSystem*>(this) puis
  // add_compiled_model<ProdModel>(*amrsys, ...). Schema marshale en arguments plats extern "C". Pas
  // de parametre evolve (AMR mono-bloc, pas de bloc fond gele comme System). SYMBOLE DISTINCT
  // (adc_install_native_amr, vs adc_install_native cote System) : un loader genere pour System ne
  // l'exporte PAS, donc le brancher ici echoue clairement au lieu d'un cast incoherent.
  using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int);
  auto install = reinterpret_cast<install_fn_t>(dlsym(h, "adc_install_native_amr"));
  if (!install) {
    dlclose(h);
    throw std::runtime_error("AmrSystem::add_native_block : adc_install_native_amr absent du .so "
                             "(loader genere pour System, ou regenerer via "
                             "dsl.compile_native(target='amr_system'))");
  }
  install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps);
  // Le .so reste charge (RTLD_GLOBAL) pour la duree du process : le builder type-erase installe par
  // set_compiled_block capture du code (gabarit en-tete) qui y vit. On NE le ferme PAS.
}

void AmrSystem::set_refinement(double threshold) { p_->refine_threshold = threshold; }

void AmrSystem::set_poisson(const std::string& rhs, const std::string& solver,
                            const std::string& bc, const std::string& wall,
                            double wall_radius) {
  // CONTRAT mono-bloc/explicite (cf. set_compiled_block) : l'AMR cable un SEUL solveur
  // elliptique (GeometricMG, le defaut template d'AmrCouplerMP) et un SEUL second membre
  // (f = model.elliptic_rhs(U), assemble par coupler_eval_rhs). On REFUSE donc explicitement
  // toute valeur de rhs/solver hors du domaine reellement cable, au lieu de la stocker en
  // silence (le no-op historique laissait croire que solver='fft' marchait sur la hierarchie).
  // bc/wall sont reellement consommes par poisson_bc()/wall_active() : valides la-bas.
  if (rhs != "charge_density" && rhs != "composite")
    throw std::runtime_error("AmrSystem::set_poisson : rhs '" + rhs +
                             "' inconnu (charge_density|composite ; le second membre = somme des "
                             "briques elliptiques du bloc)");
  if (solver != "geometric_mg")
    throw std::runtime_error("AmrSystem::set_poisson : solver '" + solver +
                             "' non supporte sur AMR (seul 'geometric_mg' est cable sur la "
                             "hierarchie ; 'fft' n'existe que sur grille mono-niveau, cf. System)");
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
}

void AmrSystem::set_density(const std::string& name, const std::vector<double>& rho) {
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_density : le systeme est deja construit (poser la "
                             "densite avant tout step/mass/density)");
  if (p_->blocks.empty())
    throw std::runtime_error("AmrSystem::set_density : appeler add_block d'abord");
  // MONO-BLOC : le nom est COSMETIQUE (compat historique, chemins add_compiled_model /
  // add_native_block qui ne nomment pas le BlockSpec) -> la densite vise l'unique bloc, quel que
  // soit le nom passe. MULTI-BLOCS : le nom INDEXE le bloc (chaque bloc a SA densite initiale, pour
  // le RHS Poisson somme Sum_b q_b n_b) ; un nom inconnu est une erreur explicite.
  std::size_t idx = 0;
  if (p_->blocks.size() >= 2) {
    const int i = p_->block_index(name);
    if (i < 0)
      throw std::runtime_error("AmrSystem::set_density : aucun bloc nomme '" + name +
                               "' (multi-blocs : le nom indexe le bloc)");
    idx = static_cast<std::size_t>(i);
  }
  p_->blocks[idx].density = rho;
  p_->blocks[idx].has_density = true;
}

void AmrSystem::step(double dt) {
  p_->ensure_built();
  if (p_->runtime) p_->runtime->step(static_cast<Real>(dt));
  else p_->step_fn(dt);
  p_->t += dt;
}
void AmrSystem::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double AmrSystem::step_cfl(double cfl) {
  p_->ensure_built();
  const double hx = p_->cfg.L / p_->cfg.n;  // pas d'espace du grossier (dx_coarse)
  if (p_->runtime) {
    // MULTI-BLOCS : pas CFL SUBSTEPS/STRIDE-AWARE, mirroir EXACT de System::step_cfl. Un bloc de
    // cadence stride avance d'un pas effectif stride*dt en substeps sous-pas, donc chaque sous-pas
    // vaut stride*dt/substeps ; la condition de stabilite par sous-pas donne le dt par bloc
    // dt_b = cfl*h*substeps_b/(stride_b*w_b), et le dt GLOBAL est le min sur les blocs (le plus
    // contraignant). AmrRuntime::step_cfl porte la formule (les w_b/substeps/stride y vivent), puis
    // avance d'un step(dt) qui re-applique stride et substeps. RETRO-COMPAT : avec substeps=1 et
    // stride=1 partout, dt = cfl*h*min_b(1/w_b) = cfl*h/w_max, identique a l'ancienne formule
    // (max_speed = max_b w_b) -> facade multi-blocs substeps=1/stride=1 bit-identique.
    const double dt = static_cast<double>(
        p_->runtime->step_cfl(static_cast<Real>(cfl), static_cast<Real>(hx)));
    p_->t += dt;
    return dt;
  }
  // MONO-BLOC (AmrCouplerMP, intouche) : formule historique dt = cfl*h/w_max (bit-identique).
  const double h = cfl * hx / p_->max_speed_fn();
  p_->step_fn(h);
  p_->t += h;
  return h;
}

int AmrSystem::nx() const { return p_->cfg.n; }
double AmrSystem::time() const { return p_->t; }
int AmrSystem::n_blocks() const { return static_cast<int>(p_->blocks.size()); }
int AmrSystem::n_patches() {
  p_->ensure_built();
  if (p_->runtime) return p_->runtime->n_patches();
  return p_->n_patches_fn();
}
double AmrSystem::mass() { return mass(std::string()); }
double AmrSystem::mass(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime) return p_->mass_fn();  // MONO-BLOC : nom cosmetique -> unique bloc
  const int idx = p_->block_index(name);   // MULTI-BLOCS : le nom indexe le bloc
  if (idx < 0)
    throw std::runtime_error("AmrSystem::mass : aucun bloc nomme '" + name + "'");
  return static_cast<double>(p_->runtime->mass(static_cast<std::size_t>(idx)));
}
std::vector<double> AmrSystem::density() { return density(std::string()); }
std::vector<double> AmrSystem::density(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime) return p_->density_fn();  // MONO-BLOC : nom cosmetique -> unique bloc
  const int idx = p_->block_index(name);      // MULTI-BLOCS : le nom indexe le bloc
  if (idx < 0)
    throw std::runtime_error("AmrSystem::density : aucun bloc nomme '" + name + "'");
  return p_->runtime->density(static_cast<std::size_t>(idx));
}
std::vector<double> AmrSystem::potential() {
  p_->ensure_built();
  if (p_->runtime) return p_->runtime->potential();  // aux partage (commun a tous les blocs)
  return p_->potential_fn();
}

}  // namespace adc
