#pragma once

#include <adc/core/types.hpp>
#include <adc/elliptic/elliptic_solver.hpp>  // concept EllipticSolver
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_multilevel.hpp>  // AmrLevel, amr_step_multilevel, average_down_fab
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

// Stepper couple AMR E x B (couplage decouple), PRET A L'EMPLOI. Encapsule la
// boucle qui etait reecrite dans diocotron_amr / diocotron_amr3 :
//
//   sync_down (fin -> grossier) -> Poisson grossier (multigrille) -> aux = grad
//   phi (grossier) + injection vers les niveaux fins -> amr_step_multilevel
//   (sous-cyclage Berger-Oliger + reflux, N niveaux).
//
// Le stepper possede la hierarchie (vector<AmrLevel>), la multigrille grossiere
// et le stockage aux (redimensionne automatiquement quand un regrid change la
// box d'un niveau). Le critere de raffinement / regrid reste a l'appelant (il
// est specifique au probleme, comme un critere de tag) : il manipule levels()
// pour reconstruire les box fines, le stepper resynchronise aux tout seul.
//
// Generique sur le modele (Diocotron ; tout modele a alpha, n_i0, B0, flux E x B).
// Pendant AMR du SpectralCoupler distribue.

namespace adc {

template <class Model, class Elliptic = GeometricMG>
class AmrCoupler {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

 public:
  AmrCoupler(const Model& model, const Geometry& geom,
             const BoxArray& ba_coarse, const BCRec& bc,
             std::vector<AmrLevel> levels)
      : model_(model),
        geom_(geom),
        dom_(geom.domain),
        mg_(geom, ba_coarse, bc),
        L_(std::move(levels)) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = Fab2D(L_[k].U.box(), 3, 1);
      L_[k].aux = &aux_[k];  // adresse stable : aux_ n'est jamais redimensionne
    }
  }

  std::vector<AmrLevel>& levels() { return L_; }            // pour le regrid
  const std::vector<AmrLevel>& levels() const { return L_; }
  Fab2D& coarse() { return L_[0].U; }
  const Fab2D& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }

  // moyenne fin -> grossier sur toute la hierarchie (avant le solve Poisson).
  void sync_down() {
    for (int k = nlev_ - 1; k >= 1; --k)
      average_down_fab(L_[k].U, L_[k - 1].U, L_[k - 1].rCI0, L_[k - 1].rCI1,
                       L_[k - 1].rCJ0, L_[k - 1].rCJ1);
  }

  // Poisson grossier + aux = grad phi + injection vers les fins. Redimensionne
  // aux si un regrid a change une box (les pointeurs aux restent valides).
  void compute_aux() {
    for (int k = 0; k < nlev_; ++k)
      if (!(aux_[k].box() == L_[k].U.box())) aux_[k] = Fab2D(L_[k].U.box(), 3, 1);

    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dx = geom_.dx(), dy = geom_.dy();
    {
      Array4 f = mg_.rhs().fab(0).array();
      const ConstArray4 u0 = L_[0].U.const_array();
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          f(i, j) = model_.alpha * (u0(i, j) - model_.n_i0);
    }
    mg_.solve();
    const ConstArray4 p = mg_.phi().fab(0).const_array();
    Array4 a0 = aux_[0].array();
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        a0(i, j, 0) = p(i, j);
        a0(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a0(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    fill_periodic_mc(aux_[0], dom_);
    for (int k = 1; k < nlev_; ++k) inject_aux(aux_[k - 1], aux_[k]);
  }

  // resout les champs (sync + Poisson + aux), sans avancer en temps.
  void update() {
    sync_down();
    compute_aux();
  }

  // un pas couple complet.
  void step(Real dt) {
    update();
    amr_step_multilevel(model_, L_, dom_, dt);
  }

  // vitesse de derive max sur le grossier, pour la CFL.
  Real max_drift_speed() const {
    const ConstArray4 a = aux_[0].const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    Real v = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model_.B0);
    return std::max(v, Real(1e-12));
  }

  // masse totale (sur le grossier).
  Real mass() const {
    const ConstArray4 u = L_[0].U.const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) M += u(i, j) * dV;
    return M;
  }

 private:
  static int crsn(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

  // injection piecewise-constante de aux (3 comp) parent -> enfant (valides +
  // ghosts), ratio 2 : aux est un champ grossier (grad phi connu au niveau 0).
  static void inject_aux(const Fab2D& parent, Fab2D& child) {
    const ConstArray4 pp = parent.const_array();
    Array4 c = child.array();
    const Box2D g = child.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        for (int k = 0; k < 3; ++k) c(i, j, k) = pp(crsn(i), crsn(j), k);
  }

  // ghosts periodiques multi-composantes.
  static void fill_periodic_mc(Fab2D& F, const Box2D& dom) {
    const int ng = F.n_ghost(), nc = F.ncomp();
    for (int c = 0; c < nc; ++c) {
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int g = 1; g <= ng; ++g) {
          F(dom.lo[0] - g, j, c) = F(dom.hi[0] - g + 1, j, c);
          F(dom.hi[0] + g, j, c) = F(dom.lo[0] + g - 1, j, c);
        }
      for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
        for (int g = 1; g <= ng; ++g) {
          F(i, dom.lo[1] - g, c) = F(i, dom.hi[1] - g + 1, c);
          F(i, dom.hi[1] + g, c) = F(i, dom.lo[1] + g - 1, c);
        }
    }
  }

  Model model_;
  Geometry geom_;
  Box2D dom_;
  Elliptic mg_;
  std::vector<AmrLevel> L_;
  std::vector<Fab2D> aux_;
  int nlev_ = 0;
};

}  // namespace adc
