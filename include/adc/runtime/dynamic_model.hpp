#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <memory>

/// @file
/// @brief Interface de modele TYPE-ERASED : dispatch d'un modele a l'EXECUTION (via vtable).
///
/// Le solveur de production est TEMPLATE : `CompositeModel<Hyperbolic, Source, Elliptic>` est connu a
/// la compilation, ce qui permet l'inlining et l'execution GPU (Kokkos). Mais un modele GENERE a
/// l'execution (brique du DSL compilee en .so puis chargee) n'a pas de type connu a la compilation.
/// `IModel<NV>` fournit le pont : une interface virtuelle (flux + max_wave_speed) qu'un modele
/// statique quelconque satisfait via `ModelAdapter`.
///
/// CHEMIN HOTE / PROTOTYPAGE uniquement : les appels virtuels ne passent PAS dans un kernel GPU et
/// coutent un saut indirect par cellule. C'est le pendant COMPILE de `adc.PythonFlux` (un cran plus
/// rapide, sans GIL). La production GPU/MPI reste le chemin template. Ne pas utiliser dans la boucle
/// chaude des cas a haute performance.

namespace adc {

/// Modele hyperbolique vu derriere une interface virtuelle (dispatch a l'execution).
template <int NV>
struct IModel {
  using State = StateVec<NV>;
  static constexpr int n_vars = NV;

  virtual ~IModel() = default;
  virtual State flux(const State& u, const Aux& a, int dir) const = 0;
  virtual Real max_wave_speed(const State& u, const Aux& a, int dir) const = 0;
};

/// Adapte un modele STATIQUE M (expose flux / max_wave_speed) en IModel<M::n_vars>.
/// M peut etre une brique compilee (Euler, IsothermalFlux...) ou une brique GENEREE par le DSL.
template <class M>
struct ModelAdapter final : IModel<M::n_vars> {
  using State = StateVec<M::n_vars>;
  M model{};

  ModelAdapter() = default;
  explicit ModelAdapter(M m) : model(m) {}

  State flux(const State& u, const Aux& a, int dir) const override { return model.flux(u, a, dir); }
  Real max_wave_speed(const State& u, const Aux& a, int dir) const override {
    return model.max_wave_speed(u, a, dir);
  }
};

/// Fabrique : enrobe un modele statique dans un IModel possede (unique_ptr).
template <class M>
std::unique_ptr<IModel<M::n_vars>> make_dynamic(M model = {}) {
  return std::make_unique<ModelAdapter<M>>(model);
}

}  // namespace adc
