#include <adc/runtime/amr_system.hpp>

#include <adc/runtime/abi_key.hpp>         // detail::abi_key_string : cle d'ABI (header-only), comparee a celle du loader
#include <adc/runtime/amr_dsl_block.hpp>   // detail::dispatch_amr_compiled + build_amr_compiled (chemin partage)
#include <adc/runtime/model_factory.hpp>   // detail::dispatch_model + briques compilees
#include <adc/runtime/wall_predicate.hpp>  // detail::wall_predicate (paroi partagee System/AmrSystem)

#include <cmath>
#include <dlfcn.h>  // dlopen/dlsym : chargement du loader natif AMR genere (.so)
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace adc {

struct AmrSystem::Impl {
  AmrSystemConfig cfg;

  // Specification du bloc (figee a add_block, materialisee au build paresseux).
  bool has_block = false;
  ModelSpec b_spec;
  std::string b_limiter = "minmod", b_riemann = "rusanov";
  bool b_recon_prim = false;  // recon == "primitive" (fige a add_block)
  int b_substeps = 1;
  double gamma = 1.4;

  double refine_threshold = 1e30;  // 1e30 => aucun raffinement par defaut

  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto",
              p_wall = "none";
  double p_wall_radius = 0.0;

  std::vector<double> pending_density;
  bool has_density = false;

  // Chemin compile (add_compiled_model, header amr_dsl_block.hpp) : builder type-erase d'un
  // AmrCouplerMP<Model> concret, invoque au build paresseux avec les AmrBuildParams figes. Exclusif
  // de has_block (un seul bloc, comme le chemin ModelSpec).
  bool has_compiled = false;
  std::function<AmrCompiledHooks(const AmrBuildParams&)> compiled_builder;

  bool built = false;
  std::shared_ptr<void> coupler_holder;  // maintient en vie l'AmrCouplerMP<Model> des hooks
  std::function<void(double)> step_fn;
  std::function<double()> max_speed_fn;
  std::function<double()> mass_fn;
  std::function<int()> n_patches_fn;
  std::function<std::vector<double>()> density_fn;
  std::function<std::vector<double>()> potential_fn;
  double t = 0;

  explicit Impl(const AmrSystemConfig& c) : cfg(c) {}

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

  // Materialise les parametres figes du build paresseux a partir de la config + des choix
  // refine/poisson/density. Communs aux deux chemins (ModelSpec natif et add_compiled_model) :
  // les deux instancient le MEME builder partage (detail::build_amr_compiled, amr_dsl_block.hpp).
  AmrBuildParams make_build_params() {
    AmrBuildParams bp;
    bp.n = cfg.n;
    bp.L = cfg.L;
    bp.regrid_every = cfg.regrid_every;
    bp.gamma = gamma;
    bp.substeps = b_substeps;
    bp.recon_prim = b_recon_prim;
    bp.refine_threshold = refine_threshold;
    bp.poisson_bc = poisson_bc();
    bp.wall = wall_active();
    bp.has_density = has_density;
    bp.density = pending_density;
    bp.distribute_coarse = cfg.distribute_coarse;
    bp.coarse_max_grid = cfg.coarse_max_grid;
    return bp;
  }

  // Installe les fermetures type-erased produites par le builder partage.
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

  void ensure_built() {
    if (built) return;
    if (!has_block && !has_compiled)
      throw std::runtime_error("AmrSystem : appeler add_block d'abord");
    const AmrBuildParams bp = make_build_params();
    if (has_compiled) {  // chemin compile : le builder fige les types (Model, Limiter, Flux)
      install(compiled_builder(bp));
      return;
    }
    // Chemin ModelSpec natif : la dispatch de modele resout le type concret, puis le MEME
    // dispatch schema spatial + build du coupleur que add_compiled_model (detail, partage).
    detail::dispatch_model(b_spec, [&](auto m) {
      install(detail::dispatch_amr_compiled(m, b_limiter, b_riemann, bp));
    });
  }
};

AmrSystem::AmrSystem(const AmrSystemConfig& c) : p_(std::make_unique<Impl>(c)) {}
AmrSystem::~AmrSystem() = default;
AmrSystem::AmrSystem(AmrSystem&&) noexcept = default;
AmrSystem& AmrSystem::operator=(AmrSystem&&) noexcept = default;

void AmrSystem::add_block(const std::string& name, const ModelSpec& model,
                          const std::string& limiter, const std::string& riemann,
                          const std::string& recon, const std::string& time, int substeps) {
  (void)name;  // AMR mono-bloc : le nom est cosmetique, il n'indexe rien (cf. header).
  if (p_->has_block || p_->has_compiled)
    throw std::runtime_error("AmrSystem : un seul bloc (AMR mono-modele)");
  if (substeps < 1) throw std::runtime_error("AmrSystem::add_block : substeps >= 1");
  if (time != "explicit")
    throw std::runtime_error("AmrSystem : seul time='explicit' est supporte sur AMR");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem : recon inconnu '" + recon +
                             "' (conservative|primitive)");
  p_->b_spec = model;
  p_->b_limiter = limiter;
  p_->b_riemann = riemann;
  p_->b_recon_prim = (recon == "primitive");
  p_->b_substeps = substeps;
  p_->gamma = model.gamma;  // indice adiabatique du bloc (Euler), lu par coupler_write_coarse
  p_->has_block = true;
}

void AmrSystem::set_compiled_block(int ncomp, double gamma, int substeps,
                                   std::function<AmrCompiledHooks(const AmrBuildParams&)> builder) {
  (void)ncomp;  // le nombre de variables est porte par le Model concret (Model::n_vars) dans le
                // builder type-erase ; le parametre reste pour la symetrie d'API avec System.
  if (p_->has_block || p_->has_compiled)
    throw std::runtime_error("AmrSystem : un seul bloc (AMR mono-modele)");
  p_->gamma = gamma;
  p_->b_substeps = substeps;
  p_->compiled_builder = std::move(builder);
  p_->has_compiled = true;
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
  // time != "explicit" et recon hors {conservative, primitive}, mais on diagnostique ICI une faute
  // de frappe avant la frontiere C++. RESPECT des LIMITES AMR (non-parite avec System) : seul
  // time == "explicit" est cable sur la hierarchie (pas d'IMEX). limiter/riemann sont valides par
  // dispatch_amr_compiled dans le loader (exception claire, ABI partagee verifiee plus bas).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem::add_native_block : recon 'conservative' | 'primitive' "
                             "(recu '" + recon + "')");
  if (time != "explicit")
    throw std::runtime_error("AmrSystem::add_native_block : seul time='explicit' sur AMR (recu '" +
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
  (void)name;  // AMR mono-bloc : la densite vise l'unique bloc, le nom est cosmetique.
  p_->pending_density = rho;
  p_->has_density = true;
}

void AmrSystem::step(double dt) {
  p_->ensure_built();
  p_->step_fn(dt);
  p_->t += dt;
}
void AmrSystem::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double AmrSystem::step_cfl(double cfl) {
  p_->ensure_built();
  const double h = cfl * (p_->cfg.L / p_->cfg.n) / p_->max_speed_fn();
  p_->step_fn(h);
  p_->t += h;
  return h;
}

int AmrSystem::nx() const { return p_->cfg.n; }
double AmrSystem::time() const { return p_->t; }
int AmrSystem::n_patches() {
  p_->ensure_built();
  return p_->n_patches_fn();
}
double AmrSystem::mass() {
  p_->ensure_built();
  return p_->mass_fn();
}
std::vector<double> AmrSystem::density() {
  p_->ensure_built();
  return p_->density_fn();
}
std::vector<double> AmrSystem::potential() {
  p_->ensure_built();
  return p_->potential_fn();
}

}  // namespace adc
