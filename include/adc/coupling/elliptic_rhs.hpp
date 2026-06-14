#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/spatial_operator.hpp>  // load_state

#include <cstddef>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Assembleurs du SECOND MEMBRE elliptique (Poisson) : mono-modele et a N especes.
///
/// model.elliptic_rhs(U) couvre le cas MONO-bloc (SingleModelEllipticRhs). Pour plusieurs especes le
/// second membre est une quantite de SYSTEME (f = Sum_s q_s n_s) lisant plusieurs MultiFab. Ce header
/// isole cette responsabilite du Coupler. ChargeDensityRhs (N especes) EXIGE une entree par bloc
/// (species.size() == n_blocks) : un bloc oublie ne disparait pas silencieusement du second membre ;
/// une espece neutre se declare explicitement avec charge = 0. add_scaled_component (rhs += q * U)
/// est la brique d'accumulation. Les kernels supposent des layouts identiques entre U et rhs.

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

namespace detail {

/// Foncteur NOMME (et non lambda ADC_HD) du RHS mono-modele : f(i,j,0) = model.elliptic_rhs(U).
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

/// Assembleur de RHS MONO-modele : rhs(.,.,0) = model.elliptic_rhs(U) sur les cellules valides.
/// @tparam Model : PhysicalModel exposant elliptic_rhs(State).
template <class Model>
struct SingleModelEllipticRhs {
  Model model;

  /// rhs <- elliptic_rhs(state) cellule par cellule (layouts identiques entre state et rhs).
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

/// Foncteur NOMME (et non lambda ADC_HD) du RHS deux champs : r(i,j,0) = a0 u0 + a1 u1.
struct TwoFieldChargeDensityRhsKernel {
  Array4 r;
  ConstArray4 u0, u1;
  Real a0, a1;
  int c0, c1;
  ADC_HD void operator()(int i, int j) const {
    r(i, j, 0) = a0 * u0(i, j, c0) + a1 * u1(i, j, c1);
  }
};

}  // namespace detail

/// RHS deux champs : rhs = q0 * U0(.,.,comp0) + q1 * U1(.,.,comp1) (densite de charge a deux especes).
struct TwoFieldChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  /// rhs <- q0 U0 + q1 U1 sur les cellules valides (layouts identiques).
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

/// RHS deux blocs : meme calcul que TwoFieldChargeDensityRhs mais lit les blocs 0 et 1 d'un
/// CoupledSystem (q0 n0 + q1 n1).
struct TwoBlockChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  /// rhs <- q0 n_0 + q1 n_1 depuis les blocs 0 et 1 du systeme.
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
//
// Defaut `charge = 0` (revue Codex) : une espece NEUTRE, ou un bloc dont l'entree
// serait oubliee, ne contribue PAS a Poisson (au lieu d'y verser q=+1 par accident).
/// Charge (avec signe) et composante densite d'une espece pour l'assemblage du RHS elliptique. Defaut
/// charge = 0 : une espece neutre (ou un bloc oublie) ne contribue PAS a Poisson.
struct SpeciesCharge {
  Real charge = Real(0);
  int comp = 0;
};

namespace detail {

/// Foncteur NOMME (et non lambda ADC_HD) de l'accumulation : r(i,j,0) += a * u(i,j,c).
struct AddScaledComponentKernel {
  Array4 r;
  ConstArray4 u;
  Real a;
  int c;
  ADC_HD void operator()(int i, int j) const { r(i, j, 0) += a * u(i, j, c); }
};

}  // namespace detail

// rhs += q * U(.,.,comp) sur les cellules valides. Brique d'accumulation du RHS
// elliptique a N especes (suppose layouts identiques entre U et rhs).
/// rhs(.,.,0) += q * U(.,.,comp) sur les cellules valides. Brique d'accumulation du RHS a N especes
/// (layouts identiques entre U et rhs).
inline void add_scaled_component(const MultiFab& U, Real q, int comp,
                                 MultiFab& rhs) {
  for (int li = 0; li < rhs.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    Array4 r = rhs.fab(li).array();
    const Box2D b = rhs.box(li);
    const Real a = q;
    const int c = comp;
    for_each_cell(b, detail::AddScaledComponentKernel{r, u, a, c});
  }
}

// Second membre de Poisson a N especes : f = Sum_s q_s n_s, somme sur TOUS les
// blocs du systeme (generalise TwoBlockChargeDensityRhs aux N especes demandees
// par le tuteur). `species[k]` decrit le bloc k dans l'ordre de CoupledSystem.
//
// Contrat (revue Codex) : EXIGER une entree par bloc (species.size() == n_blocks),
// pour qu'un bloc oublie ne soit pas silencieusement absent du second membre. Une
// espece neutre se declare explicitement avec `charge = 0`.
//
// Exemple deux fluides "rhs = n_i - n_e" :
//   ChargeDensityRhs{{ {.charge=-1, .comp=0},   // bloc 0 : electrons
//                      {.charge=+1, .comp=0} }} // bloc 1 : ions
/// RHS de Poisson a N especes : f = Sum_s q_s n_s sur TOUS les blocs du systeme. species[k] decrit
/// le bloc k dans l'ordre de CoupledSystem ; EXIGE species.size() == n_blocks (sinon exception).
struct ChargeDensityRhs {
  std::vector<SpeciesCharge> species;

  /// rhs <- 0 puis += q_s n_s pour chaque bloc. Jette si species.size() != n_blocks.
  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab& rhs) const {
    if (species.size() != System::n_blocks)
      throw std::runtime_error(
          "ChargeDensityRhs : il faut exactement une SpeciesCharge par bloc "
          "(une espece neutre se declare avec charge = 0)");
    rhs.set_val(Real(0));
    std::size_t k = 0;
    system.for_each_block([&](const auto& block) {
      const SpeciesCharge sc = species[k++];
      add_scaled_component(block.U(), sc.charge, sc.comp, rhs);
    });
  }
};

}  // namespace adc
