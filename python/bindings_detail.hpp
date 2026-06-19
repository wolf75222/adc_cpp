#pragma once
// Shared surface for the split pybind11 bindings of `_adc` (ADC-365). bindings.cpp is the thin
// PYBIND11_MODULE that calls init_core / init_system / init_amr; each lives in its own TU so the
// py::class_/.def template instantiations compile in parallel (better incremental, lower peak pybind
// memory per TU). This header carries the common includes, the small array/POD helpers (moved verbatim
// from the old monolithic bindings.cpp), and the init_* declarations.

#include <pybind11/functional.h>  // std::function<double()> <- Python callable (add_dt_bound)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/core/kokkos_env.hpp>  // Kokkos_Core under ADC_HAS_KOKKOS (kokkos_is_initialized)
#include <adc/parallel/comm.hpp>  // adc::my_rank / n_ranks: rank-0 guard of the multi-rank IO facade
#include <adc/runtime/abi_key.hpp>  // adc::abi_key: ABI key exposed to the DSL ("production" path)
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/system.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>  // std::tuple: argument of AmrSystem.set_hierarchy (patch_boxes boxes) (ADC-65)
#include <vector>

namespace py = pybind11;
using namespace adc;

// field (ny*nx row-major, j slow / i fast) -> numpy array (ny, nx) (copy). We size the buffer
// with BOTH real extents of the index domain (rows = ny, cols = nx): square n x n in Cartesian
// (UNCHANGED), but nr x ntheta in polar where nr != ntheta. A square reshape (n, n) would allocate nx^2
// slots for ny*nx values -> memcpy overflows the numpy buffer (heap overflow, crash at teardown). We
// CHECK buffer size == source size before the memcpy (explicit guard).
inline py::array_t<double> to_2d(const std::vector<double>& v, int rows, int cols) {
  py::array_t<double> a({rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings): field size (" + std::to_string(v.size()) +
                             ") != rows*cols (" + std::to_string(rows) + "*" +
                             std::to_string(cols) + "); inconsistent 2D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
// state (ncomp*ny*nx, component-major order, j slow / i fast) -> numpy array (ncomp, ny, nx).
// Same guard as to_2d: rows = ny, cols = nx (square in Cartesian, nr x ntheta in polar).
inline py::array_t<double> to_3d(const std::vector<double>& v, int ncomp, int rows, int cols) {
  py::array_t<double> a({ncomp, rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("adc (bindings): state size (" + std::to_string(v.size()) +
                             ") != ncomp*rows*cols (" + std::to_string(ncomp) + "*" +
                             std::to_string(rows) + "*" + std::to_string(cols) +
                             "); inconsistent 3D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
inline std::vector<double> flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

// ADC-214: the Python SURFACE keeps the newton_fail_policy kwarg as a STRING ("none"/"warn"/"throw");
// the NewtonOptions POD carries an integer (NewtonOptions::kFail*). This conversion table therefore
// lives in the bindings (where the flat kwargs are assembled into a POD), with the SAME explicit error
// message as before this work. @p where names the calling method in the message.
inline int newton_fail_policy_from_string(const std::string& policy, const char* where) {
  if (policy == "none")
    return NewtonOptions::kFailNone;
  if (policy == "warn")
    return NewtonOptions::kFailWarn;
  if (policy == "throw")
    return NewtonOptions::kFailThrow;
  throw std::runtime_error(std::string(where) +
                           ": newton_fail_policy 'none'|'warn'|'throw' (got '" + policy + "')");
}

// Per-area binding registration, each defined in its own TU (init_core.cpp / init_system.cpp /
// init_amr.cpp). bindings.cpp calls them in this order: init_core registers SystemConfig / ModelSpec
// (used by System / AmrSystem signatures) before init_system / init_amr run.
void init_core(py::module_& m);
void init_system(py::module_& m);
void init_amr(py::module_& m);
