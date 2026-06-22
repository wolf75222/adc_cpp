#pragma once

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/spatial_operator.hpp>  // load_state

#include <cstddef>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Elliptic (Poisson) RIGHT-HAND-SIDE assemblers: single-model and N-species.
///
/// model.elliptic_rhs(U) covers the SINGLE-block case (SingleModelEllipticRhs). For several species the
/// right-hand side is a SYSTEM quantity (f = Sum_s q_s n_s) reading several MultiFab. This header isolates
/// that responsibility from the Coupler. ChargeDensityRhs (N species) REQUIRES one entry per block
/// (species.size() == n_blocks): a forgotten block does not silently disappear from the right-hand side;
/// a neutral species is declared explicitly with charge = 0. add_scaled_component (rhs += q * U) is the
/// accumulation brick. The kernels assume identical layouts between U and rhs.

namespace adc {

namespace detail {

/// NAMED functor (not an ADC_HD lambda) of the single-model RHS: f(i,j,0) = model.elliptic_rhs(U).
template <class Model>
struct SingleModelEllipticRhsKernel {
  Model m;
  ConstArray4 s;
  Array4 f;
  ADC_HD void operator()(int i, int j) const {
    f(i, j, 0) = m.elliptic_rhs(load_state<Model>(s, i, j));
  }
};

}  // namespace detail

/// SINGLE-model RHS assembler: rhs(.,.,0) = model.elliptic_rhs(U) over the valid cells.
/// @tparam Model PhysicalModel exposing elliptic_rhs(State).
template <class Model>
struct SingleModelEllipticRhs {
  Model model;

  /// rhs <- elliptic_rhs(state) cell by cell (identical layouts between state and rhs).
  void operator()(const MultiFab& state, MultiFab& rhs) const {
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      Array4 f = rhs.fab(li).array();
      const Box2D v = rhs.box(li);
      const Model m = model;
      for_each_cell(v, detail::SingleModelEllipticRhsKernel<Model>{m, s, f});
    }
  }
};

namespace detail {

/// NAMED functor (not an ADC_HD lambda) of the two-field RHS: r(i,j,0) = a0 u0 + a1 u1.
struct TwoFieldChargeDensityRhsKernel {
  Array4 r;
  ConstArray4 u0, u1;
  Real a0, a1;
  int c0, c1;
  ADC_HD void operator()(int i, int j) const { r(i, j, 0) = a0 * u0(i, j, c0) + a1 * u1(i, j, c1); }
};

}  // namespace detail

/// Two-field RHS: rhs = q0 * U0(.,.,comp0) + q1 * U1(.,.,comp1) (two-species charge density).
struct TwoFieldChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  /// rhs <- q0 U0 + q1 U1 over the valid cells (identical layouts).
  void operator()(const MultiFab& U0, const MultiFab& U1, MultiFab& rhs) const {
    for (int li = 0; li < rhs.local_size(); ++li) {
      const ConstArray4 u0 = U0.fab(li).const_array();
      const ConstArray4 u1 = U1.fab(li).const_array();
      Array4 r = rhs.fab(li).array();
      const Box2D b = rhs.box(li);
      const Real a0 = q0, a1 = q1;
      const int c0 = comp0, c1 = comp1;
      for_each_cell(b, detail::TwoFieldChargeDensityRhsKernel{r, u0, u1, a0, a1, c0, c1});
    }
  }
};

/// Two-block RHS: same computation as TwoFieldChargeDensityRhs but reads blocks 0 and 1 of a
/// CoupledSystem (q0 n0 + q1 n1).
struct TwoBlockChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  /// rhs <- q0 n_0 + q1 n_1 from blocks 0 and 1 of the system.
  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab& rhs) const {
    TwoFieldChargeDensityRhs two;
    two.q0 = q0;
    two.q1 = q1;
    two.comp0 = comp0;
    two.comp1 = comp1;
    two(system.template block<0>().U(), system.template block<1>().U(), rhs);
  }
};

/// Charge (with sign) and density component of a species for the elliptic RHS assembly. Default
/// charge = 0: a neutral species (or a forgotten block) does NOT contribute to Poisson.
/// charge includes the sign (q_e < 0, q_i > 0); comp locates the density component n_s in the
/// block MultiFab (0 for a scalar, the rho index for a conservative Euler state).
struct SpeciesCharge {
  Real charge = Real(0);
  int comp = 0;
};

namespace detail {

/// NAMED functor (not an ADC_HD lambda) of the accumulation: r(i,j,0) += a * u(i,j,c).
struct AddScaledComponentKernel {
  Array4 r;
  ConstArray4 u;
  Real a;
  int c;
  ADC_HD void operator()(int i, int j) const { r(i, j, 0) += a * u(i, j, c); }
};

}  // namespace detail

/// rhs(.,.,0) += q * U(.,.,comp) over the valid cells. Accumulation brick of the N-species elliptic
/// RHS (identical layouts between U and rhs).
inline void add_scaled_component(const MultiFab& U, Real q, int comp, MultiFab& rhs) {
  for (int li = 0; li < rhs.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    Array4 r = rhs.fab(li).array();
    const Box2D b = rhs.box(li);
    const Real a = q;
    const int c = comp;
    for_each_cell(b, detail::AddScaledComponentKernel{r, u, a, c});
  }
}

/// N-species Poisson RHS: f = Sum_s q_s n_s over ALL blocks of the system. species[k] describes
/// block k in the CoupledSystem order; REQUIRES species.size() == n_blocks (otherwise an exception).
///
/// Two-fluid example "rhs = n_i - n_e":
///   ChargeDensityRhs{{ {.charge=-1, .comp=0},   // block 0: electrons
///                      {.charge=+1, .comp=0} }} // block 1: ions
struct ChargeDensityRhs {
  std::vector<SpeciesCharge> species;

  /// rhs <- 0 then += q_s n_s for each block. Throws if species.size() != n_blocks.
  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab& rhs) const {
    if (species.size() != System::n_blocks)
      throw std::runtime_error(
          "ChargeDensityRhs: exactly one SpeciesCharge per block is required "
          "(a neutral species is declared with charge = 0)");
    rhs.set_val(Real(0));
    std::size_t k = 0;
    system.for_each_block([&](const auto& block) {
      const SpeciesCharge sc = species[k++];
      add_scaled_component(block.U(), sc.charge, sc.comp, rhs);
    });
  }
};

}  // namespace adc
