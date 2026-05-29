#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/coupling_policy.hpp>
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <functional>
#include <type_traits>
#include <utility>

// Coupleur hyperbolique-elliptique : ferme la boucle Poisson -> aux -> advance.
//
// A chaque etage de l'integrateur (couplage stade par stade) :
//   1. second membre f = model.elliptic_rhs(U)
//   2. resolution lap(phi) = f par la multigrille geometrique (warm start)
//   3. aux = (phi, grad phi) par differences centrees
//   4. assemblage du residu hyperbolique avec ce aux
//
// Pour diocotron aux entre par le flux (derive E x B) ; pour Euler-Poisson il
// entrerait par la source. Le coupleur est identique, seul le modele change.

namespace adc {

namespace detail {
// Helpers a portee de namespace : un lambda etendu __host__ __device__ ne peut
// PAS etre defini dans une methode privee/protegee (restriction nvcc), d'ou
// l'extraction hors de la classe Coupler.

// f = model.elliptic_rhs(U) sur les cellules valides.
template <class Model>
inline void coupler_eval_rhs(const MultiFab& state, MultiFab& rhs,
                             const Model& model) {
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

// aux = (phi, d phi/dx, d phi/dy) par differences centrees.
inline void coupler_grad_phi(const MultiFab& phi, MultiFab& aux, Real cx,
                             Real cy) {
  for (int li = 0; li < aux.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 a = aux.fab(li).array();
    const Box2D v = aux.box(li);
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      a(i, j, 0) = p(i, j);
      a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) * cx;
      a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) * cy;
    });
  }
}
}  // namespace detail

template <class Model, class Elliptic = GeometricMG>
class Coupler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique du Coupler doit modeler EllipticSolver");

 public:
  // active : predicat optionnel "interieur du conducteur" (paroi embedded pour
  // le solveur de Poisson). Vide => pas de paroi interne.
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba,
          const BCRec& bcU, const BCRec& bcPhi,
          std::function<bool(Real, Real)> active = {})
      : model_(model),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcU_(bcU),
        bcPhi_(bcPhi),
        aux_bc_(derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_(ba, dm_, 3, 1) {}

  // SSPRK2 couple. Limiteur (reconstruction) et politique de couplage temporel
  // sont des parametres de template ; U doit avoir au moins Limiter::n_ghost
  // ghosts. Policy = PerStageCoupling (defaut) recalcule phi a chaque etage ;
  // OncePerStepCoupling le resout une fois par pas (aux gele).
  template <class Limiter = NoSlope, class Policy = PerStageCoupling>
  void advance(MultiFab& U, Real dt) {
    static_assert(std::is_same_v<Policy, PerStageCoupling> ||
                      std::is_same_v<Policy, OncePerStepCoupling>,
                  "Policy doit etre PerStageCoupling ou OncePerStepCoupling");
    MultiFab R(ba_, dm_, Model::n_vars, 0);

    update_aux(U);
    fill_ghosts(U, geom_.domain, bcU_);
    assemble_rhs<Limiter>(model_, U, aux_, geom_, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);

    // PerStage : phi recalcule pour l'etat intermediaire U1 (plus precis).
    // OncePerStep : on reutilise le aux du debut de pas (un seul solve elliptique).
    if constexpr (std::is_same_v<Policy, PerStageCoupling>) update_aux(U1);
    fill_ghosts(U1, geom_.domain, bcU_);
    assemble_rhs<Limiter>(model_, U1, aux_, geom_, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }

  // Resout phi et derive aux pour un etat donne, sans avancer en temps
  // (utile pour estimer la vitesse E x B avant de fixer le pas de temps).
  void solve_fields(const MultiFab& U) { update_aux(U); }

  MultiFab& phi() { return mg_.phi(); }
  const MultiFab& aux() const { return aux_; }

 private:
  static BCRec derive_aux_bc(const BCRec& b) {
    auto t = [](BCType x) {
      return x == BCType::Periodic ? BCType::Periodic : BCType::Foextrap;
    };
    BCRec a;
    a.xlo = t(b.xlo);
    a.xhi = t(b.xhi);
    a.ylo = t(b.ylo);
    a.yhi = t(b.yhi);
    return a;
  }

  void update_aux(const MultiFab& state) {
    detail::coupler_eval_rhs(state, mg_.rhs(), model_);
    mg_.solve();  // interface du concept EllipticSolver (backend-agnostique)
    derive_aux();
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    detail::coupler_grad_phi(mg_.phi(), aux_, cx, cy);
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  Model model_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcU_, bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
};

// Le backend elliptique du coupleur respecte le contrat commun : echanger
// GeometricMG contre un autre solveur conforme (FFT enveloppe, PETSc) ne demandera
// que de changer le type du membre, pas la logique de couplage.
static_assert(EllipticSolver<GeometricMG>,
              "GeometricMG doit modeler le concept EllipticSolver");

}  // namespace adc
