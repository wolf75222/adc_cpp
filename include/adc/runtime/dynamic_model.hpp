#pragma once

#include <adc/core/physical_model.hpp>  // aux_comps<M> : largeur aux du modele enrobe
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
  /// Terme source S(U, aux) (defaut : zero, modele sans source).
  virtual State source(const State&, const Aux&) const { return State{}; }
  /// Second membre elliptique f(U) du Poisson de systeme (defaut : zero, pas de couplage).
  virtual Real elliptic_rhs(const State&) const { return Real(0); }
  /// Conversion conservatif -> primitif (P = M.to_primitive(U)). Le DSL/CompositeModel l'expose
  /// (brique hyperbolique) ; un modele sans conversion (scalaire pur) la laisse a l'IDENTITE
  /// (prim == cons), ce qui est exact pour un transport scalaire. Sert au marshaling hote de
  /// System::get_primitive_state. Prim partage la meme largeur NV que State (cf. concept).
  virtual State to_primitive(const State& u) const { return u; }
  /// Conversion primitif -> conservatif (U = M.to_conservative(P)). Pendant de to_primitive ;
  /// sert a System::set_primitive_state (init depuis les primitives). Identite par defaut.
  virtual State to_conservative(const State& p) const { return p; }
  /// Largeur du canal aux que le modele LIT (cf. aux_comps). Permet au runtime System de
  /// dimensionner et de marshaler le bon nombre de composantes vers le chemin hote (B_z...).
  /// Defaut : contrat de base (phi/grad), pour un modele qui ne lit pas de champ extra.
  virtual int n_aux() const { return kAuxBaseComps; }
};

/// Adapte un modele STATIQUE M en IModel<M::n_vars>. M peut etre une brique hyperbolique (flux +
/// max_wave_speed) ou un CompositeModel complet (flux + source + elliptic_rhs) : source / elliptic_rhs
/// sont forwardes QUAND M les expose (sinon valeur par defaut). Permet de charger a l'execution un
/// modele GENERE par le DSL (CompositeModel<GenHyp, GenSrc, GenEll>) comme un vrai bloc couple.
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
  State source(const State& u, const Aux& a) const override {
    if constexpr (requires(const M& mm, const State& s, const Aux& aa) { mm.source(s, aa); })
      return model.source(u, a);
    else
      return State{};
  }
  Real elliptic_rhs(const State& u) const override {
    if constexpr (requires(const M& mm, const State& s) { mm.elliptic_rhs(s); })
      return model.elliptic_rhs(u);
    else
      return Real(0);
  }
  // Conversions cons <-> prim forwardees QUAND M les expose (brique hyperbolique : to_primitive /
  // to_conservative). M::Prim partage la largeur NV de State (contrat HyperbolicPhysicalModel), donc
  // le retour s'aligne directement sur State. Modele sans conversion (scalaire) -> identite (defaut),
  // exact pour un transport scalaire (prim == cons).
  State to_primitive(const State& u) const override {
    if constexpr (requires(const M& mm, const State& s) { mm.to_primitive(s); })
      return model.to_primitive(u);
    else
      return u;
  }
  State to_conservative(const State& p) const override {
    if constexpr (requires(const M& mm, const State& s) { mm.to_conservative(s); })
      return model.to_conservative(p);
    else
      return p;
  }
  int n_aux() const override { return aux_comps<M>(); }  // p.ex. 4 si une brique lit B_z
};

/// Fabrique : enrobe un modele statique dans un IModel possede (unique_ptr).
template <class M>
std::unique_ptr<IModel<M::n_vars>> make_dynamic(M model = {}) {
  return std::make_unique<ModelAdapter<M>>(model);
}

}  // namespace adc
