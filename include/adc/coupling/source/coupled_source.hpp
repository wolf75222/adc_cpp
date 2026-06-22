/// @file
/// @brief CoupledSourceFor / NoCoupledSource: contract of an inter-species COUPLING source.
///
/// model.source(U, aux) is PURELY LOCAL (state of a single block only). A multi-species plasma exchanges
/// between species (collisions, charge transfer, friction): S_e depends on U_i, etc. This term does NOT
/// belong in the local PhysicalModel: it is a system-level responsibility. A CoupledSource reads the state
/// of MULTIPLE blocks (+ aux = phi, grad phi) and updates the blocks over a step dt, via the minimal
/// contract apply(system, aux, dt). The skeleton applies it by splitting (additive forward-Euler,
/// SystemCoupler::coupled_source_step). The concrete sources live in adc_cases / the tests (the core stays
/// model-free).

#pragma once

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/storage/multifab.hpp>

namespace adc {

// Contract of a coupling source for a given system.
/// Concept: C is a valid coupling source for System if System is a CoupledSystem and if C
/// exposes apply(System&, const MultiFab& aux, Real dt) (updates the blocks over the step dt).
template <class C, class System>
concept CoupledSourceFor =
    CoupledSystemLike<System> &&
    requires(const C c, System& s, const MultiFab& aux, Real dt) { c.apply(s, aux, dt); };

// Default: no coupling source. Single-species case, or coupling provided
// solely by Poisson (the field). No-op, zero cost.
/// NULL coupling source (default): apply() is a no-op. Single-species case or coupling provided
/// solely by the field (Poisson). Zero cost.
struct NoCoupledSource {
  template <CoupledSystemLike System>
  void apply(System&, const MultiFab&, Real) const {}
};

}  // namespace adc
