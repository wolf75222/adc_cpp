// Bindings pybind11 de la LIB adc_cpp : le module compile `_adc`. On expose la
// facade de composition a l'execution `System` (le "coupleur / systeme" du
// tuteur) + sa config. Python compose QUOI assembler (modele + schema spatial +
// traitement temporel + sous-pas par bloc, Poisson de systeme) ; tout le calcul
// cellule par cellule reste dans la lib compilee. Le sucre lisible (Spatial,
// Explicit, IMEX, System) est ajoute par le paquet Python adc/__init__.py.
// Construit seulement avec -DADC_BUILD_PYTHON=ON.

#include <pybind11/functional.h>  // std::function<double()> <- callable Python (add_dt_bound)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/core/kokkos_env.hpp>  // Kokkos_Core sous ADC_HAS_KOKKOS (kokkos_is_initialized)
#include <adc/parallel/comm.hpp>    // adc::my_rank / n_ranks : garde rang-0 de la facade IO multi-rangs
#include <adc/runtime/abi_key.hpp>  // adc::abi_key : cle d'ABI exposee au DSL (chemin "production")
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/system.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>  // std::tuple : argument de AmrSystem.set_hierarchy (boites patch_boxes) (ADC-65)
#include <vector>

namespace py = pybind11;
using namespace adc;

// champ (ny*nx row-major, j lent / i rapide) -> tableau numpy (ny, nx) (copie). On dimensionne le tampon
// avec les DEUX extents reels du domaine d'indices (rows = ny, cols = nx) : carre n x n en cartesien
// (INCHANGE), mais nr x ntheta en polaire ou nr != ntheta. Un remodelage carre (n, n) y allouait nx^2
// cases pour ny*nx valeurs -> memcpy deborde le tampon numpy (heap overflow, crash au teardown). On
// VERIFIE l'accord taille du tampon == taille de la source avant le memcpy (garde-fou explicite).
static py::array_t<double> to_2d(const std::vector<double>& v, int rows, int cols) {
  py::array_t<double> a({rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings) : taille du champ (" + std::to_string(v.size()) +
                             ") != rows*cols (" + std::to_string(rows) + "*" + std::to_string(cols) +
                             ") ; remodelage 2D incoherent");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
// etat (ncomp*ny*nx, ordre composante-majeur, j lent / i rapide) -> tableau numpy (ncomp, ny, nx).
// Meme garde-fou que to_2d : rows = ny, cols = nx (carre en cartesien, nr x ntheta en polaire).
static py::array_t<double> to_3d(const std::vector<double>& v, int ncomp, int rows, int cols) {
  py::array_t<double> a({ncomp, rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings) : taille de l'etat (" + std::to_string(v.size()) +
                             ") != ncomp*rows*cols (" + std::to_string(ncomp) + "*" +
                             std::to_string(rows) + "*" + std::to_string(cols) +
                             ") ; remodelage 3D incoherent");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
static std::vector<double> flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

PYBIND11_MODULE(_adc, m) {
  m.doc() =
      "adc_cpp (lib) : composition multi-especes a l'execution. System compose un "
      "systeme bloc par bloc ; le calcul reste C++ compile.";

  // Cle d'ABI du module (compilateur + standard C++ + signature des en-tetes adc). Le DSL la
  // consulte (diagnostic) ; add_native_block la compare a la cle baked dans un loader natif.
  m.def("abi_key", &adc::abi_key,
        "Cle d'ABI du module (compilateur, standard C++, signature des en-tetes adc).");

  // Rang / nombre de rangs MPI du communicateur (0 / 1 en serie ou MPI non initialise, cf.
  // adc/parallel/comm.hpp). Exposes pour que la facade IO (sim.write / sim.checkpoint) n'ecrive le
  // fichier que sur le rang 0 apres un gather collectif (state_global / potential_global).
  m.def("my_rank", &adc::my_rank, "Rang MPI du processus (0 en serie).");
  m.def("n_ranks", &adc::n_ranks, "Nombre de rangs MPI (1 en serie).");

  // Norme C++ du LOADER (ADC_CXX_STD injecte par le build : 20 sous Kokkos, 23 sinon). Le DSL
  // backend="production" DOIT compiler le modele natif avec cette MEME norme, sinon __cplusplus
  // diverge -> cle d'ABI differente -> add_native_block leve "ABI incompatible". On l'expose comme
  // entier (20/23) ; dsl.compile en derive le flag -std=c++NN au lieu de figer c++23.
#ifdef ADC_CXX_STD
  m.attr("__cxx_std__") = static_cast<int>(ADC_CXX_STD);
#else
  // Build manuel sans -DADC_CXX_STD : on retombe sur __cplusplus pour rester coherent avec la cle
  // d'ABI (qui, elle, encode toujours __cplusplus). 202002L -> 20, au-dela -> 23.
  m.attr("__cxx_std__") = static_cast<int>(__cplusplus > 202002L ? 23 : 20);
#endif

  // Backend de calcul COMPILE dans le module : True si _adc a ete construit avec Kokkos
  // (-DADC_USE_KOKKOS=ON -> ADC_HAS_KOKKOS), donc capable de multi-thread (device OpenMP) / GPU.
  // adc.set_threads / adc.parallel_info s'en servent pour avertir qu'un module SERIE ignore le
  // reglage de threads. Un build serie expose False ; pas de faux negatif.
#ifdef ADC_HAS_KOKKOS
  m.attr("__has_kokkos__") = true;
#else
  m.attr("__has_kokkos__") = false;
#endif

  // Chemin du COMPILATEUR qui a construit ce module (ADC_CXX_COMPILER, injecte par CMake). La cle
  // d'ABI encodant __VERSION__, le DSL "production" DOIT recompiler ses loaders avec CE compilateur :
  // dsl.py le prefere au `which c++` du PATH (qui, dans un env conda, designe souvent un autre
  // compilateur -> "-std=c++23 invalide" ou rejet ABI). Build manuel sans -D : chaine vide, dsl.py
  // retombe alors sur sa detection historique.
#ifdef ADC_CXX_COMPILER
  m.attr("__cxx_compiler__") = ADC_CXX_COMPILER;
#else
  m.attr("__cxx_compiler__") = "";
#endif

  // Version du projet (ADC_VERSION = PROJECT_VERSION CMake, source unique). Reexposee en
  // adc.__version__ par le paquet ; "unknown" sur un build manuel sans -D.
#ifdef ADC_VERSION
  m.attr("__version__") = ADC_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

  // Etat REEL de l'init Kokkos (lazy : 1re allocation de Fab, par N'IMPORTE quel chemin --
  // System, AmrSystem, .so DSL...). adc.set_threads s'appuie dessus plutot que sur un drapeau
  // Python qui ne voyait que System/AmrSystem : le warning "trop tard" devient fiable.
  // Build serie : toujours False (rien a initialiser, le reglage de threads est sans objet).
  m.def("kokkos_is_initialized", []() {
#ifdef ADC_HAS_KOKKOS
    return Kokkos::is_initialized();
#else
    return false;
#endif
  }, "True si le runtime Kokkos du module est deja initialise (set_threads arrive alors trop tard).");

  py::class_<SystemConfig>(m, "SystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &SystemConfig::n)
      .def_readwrite("L", &SystemConfig::L)
      .def_readwrite("periodic", &SystemConfig::periodic)
      // Geometrie opt-in (chantier "grille polaire", Phase 1). "cartesian" (defaut) = bit-identique ;
      // "polar" = anneau global porte par adc.PolarMesh. Champs polaires ignores si geometry=="cartesian".
      .def_readwrite("geometry", &SystemConfig::geometry)
      .def_readwrite("nr", &SystemConfig::nr)
      .def_readwrite("ntheta", &SystemConfig::ntheta)
      .def_readwrite("r_min", &SystemConfig::r_min)
      .def_readwrite("r_max", &SystemConfig::r_max)
      .def_readwrite("theta_boxes", &SystemConfig::theta_boxes);

  // ModelSpec : composition de briques generiques (transport/source/elliptic + parametres).
  // Aucun scenario nomme ; le sucre adc.Model(...) cote Python remplit ces champs.
  py::class_<ModelSpec>(m, "ModelSpec")
      .def(py::init<>())
      .def_readwrite("transport", &ModelSpec::transport)
      .def_readwrite("source", &ModelSpec::source)
      .def_readwrite("elliptic", &ModelSpec::elliptic)
      .def_readwrite("B0", &ModelSpec::B0)
      .def_readwrite("gamma", &ModelSpec::gamma)
      .def_readwrite("cs2", &ModelSpec::cs2)
      .def_readwrite("qom", &ModelSpec::qom)
      .def_readwrite("q", &ModelSpec::q)
      .def_readwrite("alpha", &ModelSpec::alpha)
      .def_readwrite("n0", &ModelSpec::n0)
      .def_readwrite("sign", &ModelSpec::sign)
      .def_readwrite("four_pi_G", &ModelSpec::four_pi_G)
      .def_readwrite("rho0", &ModelSpec::rho0);

  py::class_<System>(m, "System")
      .def(py::init<const SystemConfig&>())
      // Composition par bloc : modele (briques) + schema spatial (limiter/riemann) + temps
      // (explicit/imex) + sous-pas. Python dit QUOI, le C++ compile fait le calcul.
      .def("add_block", &System::add_block, py::arg("name"), py::arg("model"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative",
           py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("evolve") = true, py::arg("stride") = 1,
           // Masque implicite PORTE PAR LE BLOC (IMEX) : variables conservees traitees en implicite par
           // NOM (implicit_vars) ou par ROLE physique (implicit_roles). Vides (defaut) -> defaut modele,
           // bit-identique. Resolus cote C++ contre les noms/roles du bloc (erreur sur un nom/role absent).
           py::arg("implicit_vars") = std::vector<std::string>{},
           py::arg("implicit_roles") = std::vector<std::string>{},
           // Options du Newton de la source implicite IMEX (defauts = constantes historiques 2 / 1e-7,
           // bit-identique). newton_diagnostics=True active le rapport (newton_report(name)).
           py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
           py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
           py::arg("newton_diagnostics") = false, py::arg("newton_damping") = 1.0,
           py::arg("newton_fail_policy") = "none",
           // Limiteur de POSITIVITE Zhang-Shu (ADC-76) : plancher de densite des etats de face
           // reconstruits (scaling conservatif vers la moyenne de cellule). 0 (defaut) = inactif,
           // chemin bit-identique. Exige un modele exposant le role Density.
           py::arg("positivity_floor") = 0.0)
      // Rapport Newton (diagnostics IMEX OPT-IN) : dict {enabled, converged, max_residual,
      // max_iters_used, n_failed, failed_cell, failed_component}, agrege sur les sous-pas de la
      // DERNIERE avance du bloc. failed_cell = (i, j) d'UNE cellule fautive ou None.
      .def("newton_report",
           [](const System& s, const std::string& name) {
             const System::SourceNewtonReport r = s.newton_report(name);
             py::dict d;
             d["enabled"] = r.enabled;
             d["converged"] = r.converged;
             d["max_residual"] = r.max_residual;
             d["max_iters_used"] = r.max_iters_used;
             d["n_failed"] = r.n_failed;
             if (r.failed_i >= 0)
               d["failed_cell"] = py::make_tuple(static_cast<int>(r.failed_i),
                                                 static_cast<int>(r.failed_j));
             else
               d["failed_cell"] = py::none();
             d["failed_component"] = static_cast<int>(r.failed_comp);
             return d;
           },
           py::arg("name"))
      // Bloc dont le modele est charge a l'execution depuis un .so genere par le DSL (chemin hote).
      .def("add_dynamic_block", &System::add_dynamic_block, py::arg("name"), py::arg("so_path"),
           py::arg("substeps") = 1, py::arg("names") = std::vector<std::string>{},
           py::arg("recon") = "none")
      .def("add_compiled_block", &System::add_compiled_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("names") = std::vector<std::string>{}, py::arg("positivity_floor") = 0.0)
      // P7-b : change les parametres RUNTIME d'un bloc AOT SANS recompiler le .so. values = bloc
      // complet (ordre trie des noms cote DSL). cf. System::set_block_params.
      .def("set_block_params", &System::set_block_params, py::arg("name"), py::arg("values"))
      // Bloc NATIF charge depuis un loader .so genere par le DSL (backend "production",
      // dsl.compile_native) : le .so inline add_compiled_model<ProdModel> -> bloc zero-copie sur le
      // contexte reel du System, cle d'ABI verifiee. cf. System::add_native_block.
      .def("add_native_block", &System::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1, py::arg("evolve") = true, py::arg("stride") = 1,
           py::arg("positivity_floor") = 0.0)
      .def("add_ionization", &System::add_ionization, py::arg("electron"), py::arg("ion"),
           py::arg("neutral"), py::arg("rate"))
      .def("add_collision", &System::add_collision, py::arg("a"), py::arg("b"), py::arg("rate"))
      .def("add_thermal_exchange", &System::add_thermal_exchange, py::arg("a"), py::arg("b"),
           py::arg("rate"))
      // Etage source condense par Schur (OPT-IN, adc.Split(source=adc.CondensedSchur(...))) : remplace
      // la source explicite / IMEX du bloc par l'etage condense C++ (CondensedSchurSourceStepper, #126)
      // apres le transport hyperbolique. kind='electrostatic_lorentz'. Defaut (sans appel) inchange.
      .def("set_source_stage", &System::set_source_stage, py::arg("name"), py::arg("kind"),
           py::arg("theta"), py::arg("alpha"),
           // Tolerance / budget du solve Krylov de l'etage (audit 2026-06) : <= 0 = defauts
           // historiques du stepper (1e-10 ; 400 cartesien / 600 polaire).
           py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0,
           // Descripteurs des champs (audit vague 2 : roles transportes dans l'ABI) : "" = role
           // canonique (bit-identique) ; sinon nom de role stable ou de variable du bloc.
           // bz_aux_component < 0 = canal canonique B_z. Honore en cartesien comme en polaire.
           py::arg("density") = "", py::arg("momentum_x") = "", py::arg("momentum_y") = "",
           py::arg("energy") = "", py::arg("bz_aux_component") = -1)
      // Politique de splitting en temps : "lie" (defaut, bit-identique) ou "strang" (H(dt/2) S(dt)
      // H(dt/2), 2e ordre). Cf. System::set_time_scheme / SystemStepper::step_strang.
      .def("set_time_scheme", &System::set_time_scheme, py::arg("scheme"))
      // (System) -- voir aussi AmrSystem.add_coupled_source plus bas pour le pendant AMR.
      // Borne GLOBALE de pas de temps (audit step_cfl) : fn() evaluee UNE fois par pas (hote) par
      // step_cfl / step_adaptive ; dt <= fn() quand fn() > 0 et fini. Crochet des contraintes non
      // locales-cellule (couplage, Schur/Poisson, scheduler, rampe utilisateur). Une callback
      // Python est acceptable ici (jamais par cellule).
      .def("add_dt_bound", &System::add_dt_bound, py::arg("label"), py::arg("fn"))
      // Borne ACTIVE du dernier step_cfl : "transport:<bloc>" | "source_frequency:<bloc>" |
      // "stability_dt:<bloc>" | "global:<label>" | "degenerate" | "" (aucun pas CFL encore).
      .def("last_dt_bound", &System::last_dt_bound)
      // Horloge (IO v1) : macro_step expose + restauration (t, macro_step) pour le restart -- la
      // cadence stride depend de macro_step % stride, pas seulement de t.
      .def("macro_step", &System::macro_step)
      .def("set_clock", &System::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("set_potential", &System::set_potential, py::arg("phi"))
      // Politique de la loi de Gauss (R0, repro Hoffart) : "restart" (defaut, re-resout Poisson chaque
      // pas, bit-identique) ou "evolve" (apres phi^0, plus de re-solve ; l'etage Schur fait evoluer phi
      // sans restart, comme le papier). Cf. System::set_gauss_policy.
      .def("set_gauss_policy", &System::set_gauss_policy, py::arg("policy"))
      // Source COUPLEE generique (adc.dsl.CoupledSource, P5) : ABI plate (bytecode postfixe). Lit des
      // champs (bloc, role) et ecrit des termes de source compiles en machine a pile, appliques par
      // splitting explicite apres le transport (meme seam que add_ionization). Sans appel, inchange.
      .def("add_coupled_source", &System::add_coupled_source, py::arg("in_blocks"),
           py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"), py::arg("out_roles"),
           py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
           // Frequence CONSTANTE declaree mu du couplage (CoupledSource.frequency, vague 3) : borne de
           // pas dt <= cfl/mu sur le macro-pas ; <= 0 = pas de borne (historique).
           py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
           // Frequence PAR CELLULE optionnelle mu(U) : programme bytecode (meme machine a pile / table
           // de registres que les termes). VIDES (defaut) = frequence constante seule, bit-identique.
           py::arg("freq_prog_ops") = std::vector<int>{},
           py::arg("freq_prog_args") = std::vector<int>{})
      .def("variable_names", &System::variable_names, py::arg("name"),
           py::arg("kind") = "conservative")
      .def("variable_roles", &System::variable_roles, py::arg("name"),
           py::arg("kind") = "conservative")
      .def("block_gamma", &System::block_gamma, py::arg("name"))
      .def("set_poisson", &System::set_poisson, py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0, py::arg("epsilon") = 1.0,
           py::arg("abs_tol") = 0.0)
      // Domaine de transport DISQUE (chantiers T2 / T5-PR3) : materialise un masque 0/1 cellule-centre
      // (cellule active si son centre est dans hypot(x-cx, y-cy) - R < 0) et CABLE le transport selon
      // mode= : 'none' (defaut, transport plein cartesien, bit-identique meme avec le disque pose),
      // 'staircase' (assemble_rhs_masked, porte de face 0/1), 'cutcell' (assemble_rhs_eb, cut-cell EB,
      // apertures + kappa). Honore sous Lie ET Strang (set_time_scheme). cf. System::set_disc_domain.
      .def("set_disc_domain", &System::set_disc_domain, py::arg("cx"), py::arg("cy"), py::arg("R"),
           py::arg("mode") = "none")
      // Bascule SEULE du mode de transport disque ('none'|'staircase'|'cutcell') sans (re)definir le
      // disque. Un mode != 'none' exige un disque deja fixe (set_disc_domain) -> erreur sinon.
      .def("set_geometry_mode", &System::set_geometry_mode, py::arg("mode"))
      // Masque de domaine 0/1 (ny, nx) row-major (diagnostic / verification du contrat). Tout 1.0 sans
      // set_disc_domain.
      .def("disc_mask", [](const System& s) { return to_2d(s.disc_mask(), s.ny(), s.nx()); })
      .def("set_epsilon_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_epsilon_field(flat(arr));
           },
           py::arg("eps"))
      .def("set_epsilon_anisotropic_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> eps_x,
              py::array_t<double, py::array::c_style | py::array::forcecast> eps_y) {
             s.set_epsilon_anisotropic_field(flat(eps_x), flat(eps_y));
           },
           py::arg("eps_x"), py::arg("eps_y"))
      .def("set_reaction_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_reaction_field(flat(arr));
           },
           py::arg("kappa"))
      .def("set_magnetic_field",
           [](System& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_magnetic_field(flat(arr));
           },
           py::arg("bz"))
      // Champs aux NOMMES (ADC-70 phase 1) : par COMPOSANTE canonique (>= 5). La resolution nom ->
      // comp vit dans la facade Python (adc.System.set_aux_field), qui appelle ces deux methodes.
      .def("set_aux_field_component",
           [](System& s, int comp,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_aux_field_component(comp, flat(arr));
           },
           py::arg("comp"), py::arg("field"))
      .def("aux_field_component",
           [](const System& s, int comp) {
             return to_2d(s.aux_field_component(comp), s.ny(), s.nx());
           },
           py::arg("comp"))
      .def("set_electron_temperature_from", &System::set_electron_temperature_from,
           py::arg("name"))
      .def("set_density",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      // Init depuis les PRIMITIVES : prim = tableau (ncomp, n, n) composante-majeur dans l'ordre de
      // primitive_vars(name) ; converti en conservatif par le modele du bloc. La facade Python
      // (adc.System.set_primitive_state(**prims)) assemble ce tableau a partir des kwargs nommes.
      .def("set_primitive_state",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_primitive_state(name, flat(arr));
           },
           py::arg("name"), py::arg("prim"))
      // Diagnostic : etat conservatif -> primitif (ncomp, n, n), ordre de primitive_vars(name).
      .def("get_primitive_state",
           [](System& s, const std::string& name) {
             return to_3d(s.get_primitive_state(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("solve_fields", &System::solve_fields)
      .def("step", &System::step, py::arg("dt"))
      .def("advance", &System::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &System::step_cfl, py::arg("cfl"))
      .def("dt_hotspot", &System::dt_hotspot, py::arg("name"))
      .def("step_adaptive", &System::step_adaptive, py::arg("cfl"))
      // Primitives pour un integrateur temporel CUSTOM en Python (take_step) :
      .def("eval_rhs",
           [](System& s, const std::string& name) {
             return to_3d(s.eval_rhs(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("get_state",
           [](System& s, const std::string& name) {
             return to_3d(s.get_state(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("set_state",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_state(name, flat(arr));
           },
           py::arg("name"), py::arg("u"))
      .def("n_vars", &System::n_vars, py::arg("name"))
      .def("nx", &System::nx)
      .def("ny", &System::ny)
      .def("time", &System::time)
      .def("n_species", &System::n_species)
      .def("block_names", &System::block_names)
      .def("mass", &System::mass, py::arg("name"))
      .def("density",
           [](const System& s, const std::string& name) {
             return to_2d(s.density(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("potential", [](System& s) { return to_2d(s.potential(), s.ny(), s.nx()); })
      // Accesseurs GLOBAUX (collectifs MPI-safe) : sorties / checkpoint multi-rangs (IO v1). Chaque
      // rang DOIT les appeler (all_reduce interne) ; ils rendent le champ COMPLET (gather rang-0
      // implicite par all_reduce_sum) -- mono-rang : bit-identique a density / get_state / potential.
      // La facade sim.write / sim.checkpoint les utilise puis n'ecrit le fichier que sur le rang 0.
      .def("density_global",
           [](const System& s, const std::string& name) {
             return to_2d(s.density_global(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("state_global",
           [](const System& s, const std::string& name) {
             return to_3d(s.state_global(name), s.n_vars(name), s.ny(), s.nx());
           },
           py::arg("name"))
      .def("potential_global", [](System& s) { return to_2d(s.potential_global(), s.ny(), s.nx()); })
      // Accesseurs LOCAUX par fab (NON collectifs) : ecriture HDF5 PARALLELE par hyperslabs (PR-IO-3,
      // sim.write(format='hdf5', parallel=True)). local_boxes rend la liste des boites locales
      // (ilo, jlo, ihi, jhi) en indices GLOBAUX ; local_state rend l'etat du fab li remodele
      // (n_vars, bny, bnx) pour un hyperslab dset[:, jlo:jhi+1, ilo:ihi+1]. Un rang sans box rend une
      // liste vide. Le System etant mono-box, le vrai parallelisme n'apparait que sur une geometrie
      // multi-box (cf. AMR) ; l'API reste correcte dans le cas general.
      .def("local_boxes", &System::local_boxes, py::arg("name"))
      .def("local_state",
           [](const System& s, const std::string& name, int li) {
             const auto boxes = s.local_boxes(name);
             if (li < 0 || li >= static_cast<int>(boxes.size()))
               throw std::out_of_range("System.local_state : indice de fab local hors bornes");
             const int bnx = boxes[li][2] - boxes[li][0] + 1;  // ihi - ilo + 1
             const int bny = boxes[li][3] - boxes[li][1] + 1;  // jhi - jlo + 1
             return to_3d(s.local_state(name, li), s.n_vars(name), bny, bnx);
           },
           py::arg("name"), py::arg("li"))
      .def_static("abi_key", &System::abi_key,
                  "Cle d'ABI du module (cf. adc.abi_key) ; comparee a celle d'un loader natif.");

  // --- AMR : composition mono-espece sur AMR multi-patch (brique generique composable) ---
  // adc_cases la PILOTE depuis Python (pas de C++ cote cases) au meme titre que System.
  //
  // NB : l'integrateur AP deux-fluides (schema asymptotic-preserving SUR MESURE, non composable
  // bloc a bloc) a quitte le coeur : ce n'est pas une brique generique mais un SCENARIO. Il vit
  // desormais dans adc_cases (cf. adc_cases/two_fluid_ap/), compile a la volee contre les
  // en-tetes generiques d'adc_cpp ; il n'est plus expose par le module _adc.

  // AmrSystem : composition mono-espece generique sur AMR.
  py::class_<AmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &AmrSystemConfig::n)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("periodic", &AmrSystemConfig::periodic)
      .def_readwrite("distribute_coarse", &AmrSystemConfig::distribute_coarse)
      .def_readwrite("coarse_max_grid", &AmrSystemConfig::coarse_max_grid);

  py::class_<AmrSystem>(m, "AmrSystem")
      .def(py::init<const AmrSystemConfig&>())
      .def("add_block", &AmrSystem::add_block, py::arg("name"), py::arg("model"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit",
           py::arg("substeps") = 1, py::arg("stride") = 1,
           // Masque IMEX partiel PORTE PAR LE BLOC (capstone vii) : variables conservees traitees en
           // implicite par NOM (implicit_vars) ou par ROLE physique (implicit_roles). Vides (defaut)
           // -> backward-Euler plein. N'ont de sens qu'en time="imex" et en MULTI-BLOCS (cf. add_block).
           py::arg("implicit_vars") = std::vector<std::string>{},
           py::arg("implicit_roles") = std::vector<std::string>{},
           // Options Newton IMEX (vague 3, parite System) : OPTIONS cablees en MONO-BLOC (coupleur)
           // ET MULTI-BLOCS (moteur). newton_diagnostics (rapport newton_report) : MULTI-BLOCS natif
           // seulement (mono-bloc rejete au build ; loaders .so rejetes a la facade Python).
           py::arg("newton_max_iters") = 2, py::arg("newton_rel_tol") = 0.0,
           py::arg("newton_abs_tol") = 0.0, py::arg("newton_fd_eps") = 1e-7,
           py::arg("newton_damping") = 1.0, py::arg("newton_fail_policy") = "none",
           py::arg("newton_diagnostics") = false)
      // Rapport Newton (diagnostics IMEX OPT-IN, MULTI-BLOCS natif) : dict {enabled, converged,
      // max_residual, max_iters_used, n_failed, failed_cell, failed_component}, agrege sur les
      // niveaux/sous-pas de la DERNIERE avance du bloc. failed_cell = (i, j) ou None. Forme EXACTE du
      // binding System.newton_report (parite, y compris failed_cell tuple/None).
      .def("newton_report",
           [](AmrSystem& s, const std::string& name) {
             const AmrSystem::SourceNewtonReport r = s.newton_report(name);
             py::dict d;
             d["enabled"] = r.enabled;
             d["converged"] = r.converged;
             d["max_residual"] = r.max_residual;
             d["max_iters_used"] = r.max_iters_used;
             d["n_failed"] = r.n_failed;
             if (r.failed_i >= 0)
               d["failed_cell"] = py::make_tuple(static_cast<int>(r.failed_i),
                                                 static_cast<int>(r.failed_j));
             else
               d["failed_cell"] = py::none();
             d["failed_component"] = static_cast<int>(r.failed_comp);
             return d;
           },
           py::arg("name"))
      // Bloc NATIF AMR charge depuis un loader .so genere par le DSL (backend "production",
      // target="amr_system") : le .so inline add_compiled_model(AmrSystem&) -> bloc natif sur la
      // hierarchie AMR (reflux, regrid), cle d'ABI verifiee. cf. AmrSystem::add_native_block. PAS de
      // evolve (AMR mono-bloc). Les LIMITES AMR (primitive/roe/hllc/weno5) sont gardees cote facade
      // Python (AmrSystem.add_equation) avant ce binding.
      .def("add_native_block", &AmrSystem::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1)
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"))
      // Tag de PHI sur |grad phi| (D4) ajoute a l'union des tags du regrid : raffine aussi la ou la
      // norme du gradient du potentiel depasse grad_threshold (bord d'anneau du diocotron). MULTI-BLOCS
      // + regrid_every > 0. <= 0 (defaut) -> phi DESACTIVE (bit-identique). cf. AmrSystem::set_phi_refinement.
      .def("set_phi_refinement", &AmrSystem::set_phi_refinement, py::arg("grad_threshold"))
      .def("set_poisson", &AmrSystem::set_poisson, py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      // Borne GLOBALE de pas + borne ACTIVE (StabilityPolicy AMR, parite System.add_dt_bound).
      .def("add_dt_bound", &AmrSystem::add_dt_bound, py::arg("label"), py::arg("fn"))
      .def("last_dt_bound", &AmrSystem::last_dt_bound)
      // CHEMIN amr-schur (pendant AMR de System.set_magnetic_field / set_source_stage / set_time_scheme).
      // Etage source condense par Schur GLOBAL (electrostatique/Lorentz) sur la hierarchie mono-bloc, au
      // lieu de la source IMEX locale. B_z (terme de Lorentz) accepte un numpy (n, n) aplati.
      .def("set_magnetic_field",
           [](AmrSystem& s,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_magnetic_field(flat(arr));
           },
           py::arg("bz"))
      .def("set_source_stage", &AmrSystem::set_source_stage, py::arg("name"), py::arg("kind"),
           py::arg("theta"), py::arg("alpha"),
           // Reglages transportes (vague 3, parite System) : tolerances Krylov du solve grossier
           // (<= 0 = defauts 1e-10/400) + descripteurs de champs ("" = role canonique).
           py::arg("krylov_tol") = 0.0, py::arg("krylov_max_iters") = 0,
           py::arg("density") = "", py::arg("momentum_x") = "", py::arg("momentum_y") = "",
           py::arg("energy") = "")
      .def("set_time_scheme", &AmrSystem::set_time_scheme, py::arg("scheme"))
      .def("set_density",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      // Etat conservatif initial COMPLET (ncomp, n, n) -> demarre l'AMR depuis l'etat de derive du
      // papier (rho, rho*u, rho*v) au lieu de m=0. Garde ndim==3 EXPLICITE : flat() applatit
      // n'importe quel tableau C-contigu, donc une densite 2D (n, n) passee par erreur deviendrait un
      // etat a 1 composante (compo 0 = densite, qty de mvt laissee a 0) -- une mascarade de densite
      // silencieuse a la mauvaise physique. On exige (ncomp, n, n). flat() applatit ensuite en
      // composante-majeur c*n*n + j*n + i (meme convention que to_3d / set_state).
      .def("set_conservative_state",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             if (arr.ndim() != 3)
               throw std::runtime_error(
                   "AmrSystem.set_conservative_state : etat attendu de forme (ncomp, n, n) ; recu "
                   "un tableau " + std::to_string(arr.ndim()) + "D (une densite 2D ? utiliser "
                   "set_density)");
             s.set_conservative_state(name, flat(arr));
           },
           py::arg("name"), py::arg("U"))
      // Source COUPLEE inter-especes (adc.dsl.CoupledSource compilee, P5 bytecode), MULTI-BLOCS sur la
      // hierarchie AMR PARTAGEE : appliquee apres le transport a chaque macro-pas, par splitting
      // explicite, niveau par niveau + cascade fin -> grossier (cellules couvertes coherentes). MEME
      // ABI plate que System.add_coupled_source. Sans appel, inchange. cf. AmrSystem::add_coupled_source.
      .def("add_coupled_source", &AmrSystem::add_coupled_source, py::arg("in_blocks"),
           py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"), py::arg("out_roles"),
           py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
           py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
           // Frequence PAR CELLULE optionnelle mu(U) : evaluee sur le grossier (cf. System).
           py::arg("freq_prog_ops") = std::vector<int>{},
           py::arg("freq_prog_args") = std::vector<int>{})
      .def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl, py::arg("cfl"))
      .def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      // Horloge AMR (IO v1, parite System) : compteur de macro-pas + restauration (t, macro_step) ->
      // la cadence regrid/stride reprend exactement apres un set_clock. Prerequis PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("n_blocks", &AmrSystem::n_blocks)
      .def("block_names", &AmrSystem::block_names)
      .def("n_patches", &AmrSystem::n_patches)
      // Empreintes index-space des patchs fins : liste de tuples (level, ilo, jlo, ihi, jhi), coins
      // INCLUSIFS, dans l'espace d'indices du niveau (n << level cellules/direction, ratio 2). MEME
      // source que n_patches() (le BoxArray fin GLOBAL) -> rank-independent, MPI-safe. Query entre les
      // pas, zero cout chemin chaud. Le wrapper Python convertit en [0, L]^2 (il connait n via nx() et
      // L) ; cf. AmrSystem.patch_rectangles() cote facade.
      .def("patch_boxes",
           [](AmrSystem& s) {
             py::list out;
             for (const adc::PatchBox& b : s.patch_boxes())
               out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
             return out;
           })
      // mass / density : surcharge par NOM de bloc (multi-blocs ; nom vide -> 1er bloc, compat
      // mono-bloc ou nom cosmetique). Le nom INDEXE le bloc en multi-blocs (chaque bloc a sa masse /
      // densite, conservees PAR BLOC au reflux). Sans argument -> 1er bloc (retro-compat mono-bloc).
      .def("mass", [](AmrSystem& s) { return s.mass(); })
      .def("mass", [](AmrSystem& s, const std::string& name) { return s.mass(name); },
           py::arg("name"))
      // AMR : domaine CARRE (n x n), aucune geometrie polaire -> rows == cols == nx() (inchange).
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.nx(), s.nx()); })
      .def("density",
           [](AmrSystem& s, const std::string& name) {
             return to_2d(s.density(name), s.nx(), s.nx());
           },
           py::arg("name"))
      // phi du niveau grossier (base), (n, n). MEME observable que System.potential() : le niveau 0
      // couvre tout le domaine -> suffit a echantillonner un cercle median (FFT azimutale). En
      // multi-blocs, phi resulte du Poisson de SYSTEME (Sum_b q_b n_b co-localise), partage par tous.
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.nx(), s.nx()); })
      // CHECKPOINT / RESTART AMR mono-rang (ADC-65) : etat conservatif COMPLET par niveau + phi
      // (warm-start) + imposition de la hierarchie fine sauvee. MONO-BLOC SERIE (multi-blocs : rejet
      // C++ ; np>1 : rejet facade -- gather par niveau = suite). level_state / level_potential rendent
      // des champs PLATS (c*nf*nf + j*nf + i / nf*nf, nf = nx << k) ; la facade reshape. set_*
      // applatissent n'importe quel tableau C-contigu (flat). set_hierarchy : liste de tuples
      // (level, ilo, jlo, ihi, jhi) comme patch_boxes() (le coupleur filtre le niveau 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("n_vars", [](AmrSystem& s) { return s.n_vars(); })
      .def("level_state", [](AmrSystem& s, int k) { return s.level_state(k); }, py::arg("k"))
      .def("set_level_state",
           [](AmrSystem& s, int k,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_level_state(k, flat(arr));
           },
           py::arg("k"), py::arg("state"))
      .def("level_potential", [](AmrSystem& s, int k) { return s.level_potential(k); }, py::arg("k"))
      .def("set_level_potential",
           [](AmrSystem& s, int k,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_level_potential(k, flat(arr));
           },
           py::arg("k"), py::arg("phi"))
      .def("set_hierarchy",
           [](AmrSystem& s, const std::vector<std::tuple<int, int, int, int, int>>& boxes) {
             std::vector<adc::PatchBox> bx;
             bx.reserve(boxes.size());
             for (const auto& b : boxes)
               bx.push_back(adc::PatchBox{std::get<0>(b), std::get<1>(b), std::get<2>(b),
                                          std::get<3>(b), std::get<4>(b)});
             s.set_hierarchy(bx);
           },
           py::arg("boxes"));
}
