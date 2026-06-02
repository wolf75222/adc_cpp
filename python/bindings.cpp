// Bindings pybind11 de la LIB adc_cpp : le module compile `_adc`. On expose la
// facade de composition a l'execution `System` (le « coupleur / systeme » du
// tuteur) + sa config. Python compose QUOI assembler (modele + schema spatial +
// traitement temporel + sous-pas par bloc, Poisson de systeme) ; tout le calcul
// cellule par cellule reste dans la lib compilee. Le sucre lisible (Spatial,
// Explicit, IMEX, System) est ajoute par le paquet Python adc/__init__.py.
// Construit seulement avec -DADC_BUILD_PYTHON=ON.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/system.hpp>
#include <adc/solver/two_fluid_ap_solver.hpp>

#include <cstring>
#include <vector>

namespace py = pybind11;
using namespace adc;

// champ (n*n row-major) -> tableau numpy (n, n) (copie).
static py::array_t<double> to_2d(const std::vector<double>& v, int n) {
  py::array_t<double> a({n, n});
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
// etat (ncomp*n*n, ordre composante-majeur) -> tableau numpy (ncomp, n, n).
static py::array_t<double> to_3d(const std::vector<double>& v, int ncomp, int n) {
  py::array_t<double> a({ncomp, n, n});
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

  py::class_<SystemConfig>(m, "SystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &SystemConfig::n)
      .def_readwrite("L", &SystemConfig::L)
      .def_readwrite("periodic", &SystemConfig::periodic);

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
           py::arg("time") = "explicit", py::arg("substeps") = 1)
      .def("set_poisson", &System::set_poisson, py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      .def("set_density",
           [](System& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      .def("solve_fields", &System::solve_fields)
      .def("step", &System::step, py::arg("dt"))
      .def("advance", &System::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &System::step_cfl, py::arg("cfl"))
      .def("step_adaptive", &System::step_adaptive, py::arg("cfl"))
      // Primitives pour un integrateur temporel CUSTOM en Python (take_step) :
      .def("eval_rhs",
           [](System& s, const std::string& name) {
             return to_3d(s.eval_rhs(name), s.n_vars(name), s.nx());
           },
           py::arg("name"))
      .def("get_state",
           [](System& s, const std::string& name) {
             return to_3d(s.get_state(name), s.n_vars(name), s.nx());
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
      .def("time", &System::time)
      .def("n_species", &System::n_species)
      .def("mass", &System::mass, py::arg("name"))
      .def("density",
           [](const System& s, const std::string& name) {
             return to_2d(s.density(name), s.nx());
           },
           py::arg("name"))
      .def("potential", [](System& s) { return to_2d(s.potential(), s.nx()); });

  // --- Solveurs SPECIALISES (integrateurs sur mesure non composables bloc-a-bloc) ---
  // Exposes comme facades de la lib : un schema asymptotic-preserving deux-fluides, et
  // une composition mono-espece sur AMR multi-patch. adc_cases les PILOTE depuis Python (pas de C++ cote
  // cases) au meme titre que System.

  py::class_<TwoFluidAPConfig>(m, "TwoFluidAPConfig")
      .def(py::init<>())
      .def_readwrite("n", &TwoFluidAPConfig::n)
      .def_readwrite("L", &TwoFluidAPConfig::L)
      .def_readwrite("cse2", &TwoFluidAPConfig::cse2)
      .def_readwrite("csi2", &TwoFluidAPConfig::csi2)
      .def_readwrite("omega_pe", &TwoFluidAPConfig::omega_pe)
      .def_readwrite("omega_pi", &TwoFluidAPConfig::omega_pi)
      .def_readwrite("stabilize", &TwoFluidAPConfig::stabilize)
      .def_readwrite("eps", &TwoFluidAPConfig::eps)
      .def_readwrite("upwind_continuity", &TwoFluidAPConfig::upwind_continuity)
      .def_readwrite("omega_ce", &TwoFluidAPConfig::omega_ce)
      .def_readwrite("omega_ci", &TwoFluidAPConfig::omega_ci);

  py::class_<TwoFluidAPSolver>(m, "TwoFluidAP")
      .def(py::init<const TwoFluidAPConfig&>(), py::arg("cfg") = TwoFluidAPConfig{})
      .def("step", &TwoFluidAPSolver::step, py::arg("dt"))
      .def("advance", &TwoFluidAPSolver::advance, py::arg("dt"), py::arg("nsteps"))
      .def("nx", &TwoFluidAPSolver::nx)
      .def("mass_e", &TwoFluidAPSolver::mass_e)
      .def("mass_i", &TwoFluidAPSolver::mass_i)
      .def("max_charge", &TwoFluidAPSolver::max_charge)
      .def("max_dev", &TwoFluidAPSolver::max_dev)
      .def("density_e",
           [](const TwoFluidAPSolver& s) { return to_2d(s.density_e(), s.nx()); })
      .def("density_i",
           [](const TwoFluidAPSolver& s) { return to_2d(s.density_i(), s.nx()); });

  // AmrSystem : composition mono-espece generique sur AMR.
  py::class_<AmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &AmrSystemConfig::n)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("periodic", &AmrSystemConfig::periodic);

  py::class_<AmrSystem>(m, "AmrSystem")
      .def(py::init<const AmrSystemConfig&>())
      .def("add_block", &AmrSystem::add_block, py::arg("name"), py::arg("model"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("time") = "explicit", py::arg("substeps") = 1)
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"))
      .def("set_poisson", &AmrSystem::set_poisson, py::arg("rhs") = "charge_density",
           py::arg("solver") = "geometric_mg", py::arg("bc") = "auto",
           py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      .def("set_density",
           [](AmrSystem& s, const std::string& name,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             s.set_density(name, flat(arr));
           },
           py::arg("name"), py::arg("rho"))
      .def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl, py::arg("cfl"))
      .def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      .def("n_patches", &AmrSystem::n_patches)
      .def("mass", &AmrSystem::mass)
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.nx()); });
}
