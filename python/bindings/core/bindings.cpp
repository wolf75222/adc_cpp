// pybind11 bindings of the adc_cpp LIB: compiles the `_pops` module. Exposes the
// runtime composition facade `System` (the tutor's "coupler / system") + its
// config. Python composes WHAT to assemble (model + spatial scheme + temporal
// treatment + per-block substeps, system Poisson); all the cell-by-cell compute
// stays in the compiled lib. The readable sugar (Spatial, Explicit, IMEX,
// System) is added by the Python package adc/__init__.py.
// Built only with -DADC_BUILD_PYTHON=ON.
//
// ADC-365: the py::class_/.def surface is split across init_core / init_system / init_amr (each its own
// TU, declared in bindings_detail.hpp) so they compile in parallel with lower peak pybind memory. This
// file is the thin PYBIND11_MODULE that calls them in order (init_core first: it registers SystemConfig
// and ModelSpec, which the System / AmrSystem signatures reference).

#include "bindings_detail.hpp"

PYBIND11_MODULE(_pops, m) {
  init_core(m);
  init_system(m);
  init_amr(m);
}
