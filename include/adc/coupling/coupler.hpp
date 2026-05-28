#pragma once

#include <adc/core/types.hpp>
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

template <class Model>
class Coupler {
 public:
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba,
          const BCRec& bcU, const BCRec& bcPhi)
      : model_(model),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcU_(bcU),
        bcPhi_(bcPhi),
        aux_bc_(derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi),
        aux_(ba, dm_, 3, 1) {}

  // SSPRK2 couple (phi recalcule a chaque etage). Le limiteur (reconstruction)
  // est un parametre de template ; U doit avoir au moins Limiter::n_ghost ghosts.
  template <class Limiter = NoSlope>
  void advance(MultiFab& U, Real dt, Real mg_tol = 1e-8, int mg_maxc = 30) {
    MultiFab R(ba_, dm_, Model::n_vars, 0);

    update_aux(U, mg_tol, mg_maxc);
    fill_ghosts(U, geom_.domain, bcU_);
    assemble_rhs<Limiter>(model_, U, aux_, geom_, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);

    update_aux(U1, mg_tol, mg_maxc);
    fill_ghosts(U1, geom_.domain, bcU_);
    assemble_rhs<Limiter>(model_, U1, aux_, geom_, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }

  // Resout phi et derive aux pour un etat donne, sans avancer en temps
  // (utile pour estimer la vitesse E x B avant de fixer le pas de temps).
  void solve_fields(const MultiFab& U, Real mg_tol = 1e-8, int mg_maxc = 30) {
    update_aux(U, mg_tol, mg_maxc);
  }

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

  void update_aux(const MultiFab& state, Real tol, int maxc) {
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      Array4 f = mg_.rhs().fab(li).array();
      const Box2D v = mg_.rhs().box(li);
      const Model m = model_;
      for_each_cell(v, [=](int i, int j) {
        f(i, j, 0) = m.elliptic_rhs(load_state<Model>(s, i, j));
      });
    }
    mg_.solve(tol, maxc);
    derive_aux();
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    for (int li = 0; li < aux_.local_size(); ++li) {
      const ConstArray4 p = mg_.phi().fab(li).const_array();
      Array4 a = aux_.fab(li).array();
      const Box2D v = aux_.box(li);
      for_each_cell(v, [=](int i, int j) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) * cx;
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) * cy;
      });
    }
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  Model model_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcU_, bcPhi_, aux_bc_;
  GeometricMG mg_;
  MultiFab aux_;
};

}  // namespace adc
