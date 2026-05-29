// Bindings pybind11 de la FACADE compilee (libadc). On expose les solveurs CONCRETS
// et sans template (DiocotronSolver, EulerPoissonSolver, TwoFluidAPSolver) : c'est
// tout l'interet du src/ -> une surface stable et bindable, jamais Coupler<Model,...>.
// Le backend (serie/OpenMP/Kokkos) est celui avec lequel libadc a ete compilee.
// Construit seulement avec -DADC_BUILD_PYTHON=ON.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/solver/diocotron_solver.hpp>
#include <adc/solver/euler_poisson_solver.hpp>
#include <adc/solver/two_fluid_ap_solver.hpp>

#include <cstring>
#include <vector>

namespace py = pybind11;
using namespace adc;

// champ (n*n row-major) -> tableau numpy (ny, nx) (copie).
static py::array_t<double> to_2d(const std::vector<double>& v, int n) {
  py::array_t<double> a({n, n});
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}

PYBIND11_MODULE(adc, m) {
  m.doc() =
      "adc_cpp : solveurs hyperbolique-elliptique sur AMR (facade compilee libadc). "
      "Le coeur generique reste C++ template ; ici les solveurs concrets.";

  // --- Diocotron (derive E x B) ---
  py::enum_<DiocotronIC>(m, "DiocotronIC", "Condition initiale du diocotron")
      .value("Smooth", DiocotronIC::Smooth)
      .value("Band", DiocotronIC::Band)
      .value("Ring", DiocotronIC::Ring);

  py::class_<DiocotronConfig>(m, "DiocotronConfig")
      .def(py::init<>())
      .def_readwrite("n", &DiocotronConfig::n)
      .def_readwrite("L", &DiocotronConfig::L)
      .def_readwrite("B0", &DiocotronConfig::B0)
      .def_readwrite("n_i0", &DiocotronConfig::n_i0)
      .def_readwrite("alpha", &DiocotronConfig::alpha)
      .def_readwrite("eps", &DiocotronConfig::eps)
      .def_readwrite("poisson_per_stage", &DiocotronConfig::poisson_per_stage)
      .def_readwrite("ic", &DiocotronConfig::ic)
      .def_readwrite("band_amp", &DiocotronConfig::band_amp)
      .def_readwrite("band_width", &DiocotronConfig::band_width)
      .def_readwrite("band_mode", &DiocotronConfig::band_mode)
      .def_readwrite("band_disp", &DiocotronConfig::band_disp)
      .def_readwrite("ring_r0", &DiocotronConfig::ring_r0)
      .def_readwrite("ring_r1", &DiocotronConfig::ring_r1)
      .def_readwrite("ring_delta", &DiocotronConfig::ring_delta)
      .def_readwrite("ring_mode", &DiocotronConfig::ring_mode)
      .def_readwrite("ring_floor", &DiocotronConfig::ring_floor)
      .def_readwrite("wall_radius", &DiocotronConfig::wall_radius);

  py::class_<DiocotronSolver>(m, "DiocotronSolver")
      .def(py::init<const DiocotronConfig&>())
      .def("step", &DiocotronSolver::step, py::arg("dt"))
      .def("step_cfl", &DiocotronSolver::step_cfl, py::arg("cfl"))
      .def("max_drift_speed", &DiocotronSolver::max_drift_speed)
      .def("dx", &DiocotronSolver::dx)
      .def("mass", &DiocotronSolver::mass)
      .def("time", &DiocotronSolver::time)
      .def("nx", &DiocotronSolver::nx)
      .def("density",
           [](const DiocotronSolver& s) { return to_2d(s.density(), s.nx()); })
      .def("potential",
           [](const DiocotronSolver& s) { return to_2d(s.potential(), s.nx()); });

  // --- Euler-Poisson auto-gravitant ---
  py::class_<EulerPoissonConfig>(m, "EulerPoissonConfig")
      .def(py::init<>())
      .def_readwrite("n", &EulerPoissonConfig::n)
      .def_readwrite("L", &EulerPoissonConfig::L)
      .def_readwrite("gamma", &EulerPoissonConfig::gamma)
      .def_readwrite("four_pi_G", &EulerPoissonConfig::four_pi_G)
      .def_readwrite("rho0", &EulerPoissonConfig::rho0)
      .def_readwrite("p0", &EulerPoissonConfig::p0)
      .def_readwrite("eps", &EulerPoissonConfig::eps)
      .def_readwrite("poisson_per_stage", &EulerPoissonConfig::poisson_per_stage)
      .def_readwrite("use_fft", &EulerPoissonConfig::use_fft);

  py::class_<EulerPoissonSolver>(m, "EulerPoissonSolver")
      .def(py::init<const EulerPoissonConfig&>())
      .def("step", &EulerPoissonSolver::step, py::arg("dt"))
      .def("mass", &EulerPoissonSolver::mass)
      .def("energy", &EulerPoissonSolver::energy)
      .def("total_momentum", &EulerPoissonSolver::total_momentum, py::arg("dir"))
      .def("time", &EulerPoissonSolver::time)
      .def("nx", &EulerPoissonSolver::nx)
      .def("density",
           [](const EulerPoissonSolver& s) { return to_2d(s.density(), s.nx()); });

  // --- deux-fluides isotherme 2D asymptotic-preserving ---
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

  py::class_<TwoFluidAPSolver>(m, "TwoFluidAPSolver")
      .def(py::init<const TwoFluidAPConfig&>())
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
}
