#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/operator/spatial_operator.hpp>  // load_state

#include <cstddef>
#include <vector>

// Assemblage du second membre elliptique.
//
// Le PhysicalModel historique expose model.elliptic_rhs(U) pour le cas mono-bloc.
// Pour plusieurs especes, le second membre de Poisson est une quantite de systeme
// (par exemple q_i n_i + q_e n_e) : il doit pouvoir lire plusieurs MultiFab.
//
// Ce header isole cette responsabilite du Coupler. Les coupleurs existants passent
// par SingleModelEllipticRhs ; les futurs SystemCoupler pourront utiliser un
// assembleur qui lit tous les EquationBlock.

namespace adc {

template <class Model>
struct SingleModelEllipticRhs {
  Model model;

  void operator()(const MultiFab& state, MultiFab& rhs) const {
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      Array4 f = rhs.fab(li).array();
      const Box2D v = rhs.box(li);
      const Model m = model;
      for_each_cell(v, [=] ADC_HD(int i, int j) {
        f(i, j, 0) = m.elliptic_rhs(load_state<Model>(s, i, j));
      });
    }
  }
};

struct TwoFieldChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  void operator()(const MultiFab& U0, const MultiFab& U1, MultiFab& rhs) const {
    for (int li = 0; li < rhs.local_size(); ++li) {
      const ConstArray4 u0 = U0.fab(li).const_array();
      const ConstArray4 u1 = U1.fab(li).const_array();
      Array4 r = rhs.fab(li).array();
      const Box2D b = rhs.box(li);
      const Real a0 = q0, a1 = q1;
      const int c0 = comp0, c1 = comp1;
      for_each_cell(b, [=] ADC_HD(int i, int j) {
        r(i, j, 0) = a0 * u0(i, j, c0) + a1 * u1(i, j, c1);
      });
    }
  }
};

struct TwoBlockChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

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

// Charge (ou masse) portee par une espece, pour l'assemblage du second membre
// elliptique. `charge` inclut le signe (q_e < 0, q_i > 0) ; `comp` repere la
// composante densite n_s dans le MultiFab du bloc (0 pour un scalaire, l'indice
// rho pour un etat conserve Euler).
struct SpeciesCharge {
  Real charge = Real(1);
  int comp = 0;
};

// rhs += q * U(.,.,comp) sur les cellules valides. Brique d'accumulation du RHS
// elliptique a N especes (suppose layouts identiques entre U et rhs).
inline void add_scaled_component(const MultiFab& U, Real q, int comp,
                                 MultiFab& rhs) {
  for (int li = 0; li < rhs.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    Array4 r = rhs.fab(li).array();
    const Box2D b = rhs.box(li);
    const Real a = q;
    const int c = comp;
    for_each_cell(b, [=] ADC_HD(int i, int j) { r(i, j, 0) += a * u(i, j, c); });
  }
}

// Second membre de Poisson a N especes : f = Sum_s q_s n_s, somme sur TOUS les
// blocs du systeme (generalise TwoBlockChargeDensityRhs aux N especes demandees
// par le tuteur). `species[k]` decrit le bloc k dans l'ordre de CoupledSystem ;
// un bloc sans entree (k >= species.size()) est traite avec la charge par defaut
// (q = +1, comp = 0) plutot que d'echouer silencieusement.
//
// Exemple deux fluides "rhs = n_i - n_e" :
//   ChargeDensityRhs{{ {.charge=-1, .comp=0},   // bloc 0 : electrons
//                      {.charge=+1, .comp=0} }} // bloc 1 : ions
struct ChargeDensityRhs {
  std::vector<SpeciesCharge> species;

  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab& rhs) const {
    rhs.set_val(Real(0));
    std::size_t k = 0;
    system.for_each_block([&](const auto& block) {
      const SpeciesCharge sc =
          (k < species.size()) ? species[k] : SpeciesCharge{};
      ++k;
      add_scaled_component(block.U(), sc.charge, sc.comp, rhs);
    });
  }
};

}  // namespace adc
