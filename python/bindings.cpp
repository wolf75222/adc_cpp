// Bindings pybind11 de la LIB adc_cpp : le module compile `_adc`. On expose la
// facade de composition a l'execution `System` (le "coupleur / systeme" du
// tuteur) + sa config. Python compose QUOI assembler (modele + schema spatial +
// traitement temporel + sous-pas par bloc, Poisson de systeme) ; tout le calcul
// cellule par cellule reste dans la lib compilee. Le sucre lisible (Spatial,
// Explicit, IMEX, System) est ajoute par le paquet Python adc/__init__.py.
// Construit seulement avec -DADC_BUILD_PYTHON=ON.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/runtime/abi_key.hpp>  // adc::abi_key : cle d'ABI exposee au DSL (chemin "production")
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/system.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
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
      .def_readwrite("r_max", &SystemConfig::r_max);

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
           py::arg("implicit_roles") = std::vector<std::string>{})
      // Bloc dont le modele est charge a l'execution depuis un .so genere par le DSL (chemin hote).
      .def("add_dynamic_block", &System::add_dynamic_block, py::arg("name"), py::arg("so_path"),
           py::arg("substeps") = 1, py::arg("names") = std::vector<std::string>{},
           py::arg("recon") = "none")
      .def("add_compiled_block", &System::add_compiled_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("substeps") = 1,
           py::arg("names") = std::vector<std::string>{})
      // P7-b : change les parametres RUNTIME d'un bloc AOT SANS recompiler le .so. values = bloc
      // complet (ordre trie des noms cote DSL). cf. System::set_block_params.
      .def("set_block_params", &System::set_block_params, py::arg("name"), py::arg("values"))
      // Bloc NATIF charge depuis un loader .so genere par le DSL (backend "production",
      // dsl.compile_native) : le .so inline add_compiled_model<ProdModel> -> bloc zero-copie sur le
      // contexte reel du System, cle d'ABI verifiee. cf. System::add_native_block.
      .def("add_native_block", &System::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit", py::arg("gamma") = 1.4,
           py::arg("substeps") = 1, py::arg("evolve") = true, py::arg("stride") = 1)
      .def("add_ionization", &System::add_ionization, py::arg("electron"), py::arg("ion"),
           py::arg("neutral"), py::arg("rate"))
      .def("add_collision", &System::add_collision, py::arg("a"), py::arg("b"), py::arg("rate"))
      .def("add_thermal_exchange", &System::add_thermal_exchange, py::arg("a"), py::arg("b"),
           py::arg("rate"))
      // Etage source condense par Schur (OPT-IN, adc.Split(source=adc.CondensedSchur(...))) : remplace
      // la source explicite / IMEX du bloc par l'etage condense C++ (CondensedSchurSourceStepper, #126)
      // apres le transport hyperbolique. kind='electrostatic_lorentz'. Defaut (sans appel) inchange.
      .def("set_source_stage", &System::set_source_stage, py::arg("name"), py::arg("kind"),
           py::arg("theta"), py::arg("alpha"))
      // Politique de splitting en temps : "lie" (defaut, bit-identique) ou "strang" (H(dt/2) S(dt)
      // H(dt/2), 2e ordre). Cf. System::set_time_scheme / SystemStepper::step_strang.
      .def("set_time_scheme", &System::set_time_scheme, py::arg("scheme"))
      // Source COUPLEE generique (adc.dsl.CoupledSource, P5) : ABI plate (bytecode postfixe). Lit des
      // champs (bloc, role) et ecrit des termes de source compiles en machine a pile, appliques par
      // splitting explicite apres le transport (meme seam que add_ionization). Sans appel, inchange.
      .def("add_coupled_source", &System::add_coupled_source, py::arg("in_blocks"),
           py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"), py::arg("out_roles"),
           py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"))
      .def("variable_names", &System::variable_names, py::arg("name"),
           py::arg("kind") = "conservative")
      .def("variable_roles", &System::variable_roles, py::arg("name"),
           py::arg("kind") = "conservative")
      .def("block_gamma", &System::block_gamma, py::arg("name"))
      .def("set_poisson", &System::set_poisson, py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0, py::arg("epsilon") = 1.0)
      // Domaine de transport DISQUE (chantier T2, CONTRAT inerte par defaut) : materialise un masque
      // 0/1 cellule-centre (cellule active si son centre est dans hypot(x-cx, y-cy) - R < 0). Sans cet
      // appel, le masque est tout actif et le chemin de transport reste bit-identique. cf.
      // System::set_disc_domain.
      .def("set_disc_domain", &System::set_disc_domain, py::arg("cx"), py::arg("cy"), py::arg("R"))
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
           py::arg("implicit_roles") = std::vector<std::string>{})
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
      .def("set_density",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      // Source COUPLEE inter-especes (adc.dsl.CoupledSource compilee, P5 bytecode), MULTI-BLOCS sur la
      // hierarchie AMR PARTAGEE : appliquee apres le transport a chaque macro-pas, par splitting
      // explicite, niveau par niveau + cascade fin -> grossier (cellules couvertes coherentes). MEME
      // ABI plate que System.add_coupled_source. Sans appel, inchange. cf. AmrSystem::add_coupled_source.
      .def("add_coupled_source", &AmrSystem::add_coupled_source, py::arg("in_blocks"),
           py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"), py::arg("out_roles"),
           py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"))
      .def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl, py::arg("cfl"))
      .def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      .def("n_blocks", &AmrSystem::n_blocks)
      .def("n_patches", &AmrSystem::n_patches)
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
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.nx(), s.nx()); });
}
