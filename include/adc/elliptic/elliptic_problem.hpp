#pragma once

#include <adc/core/types.hpp>
#include <adc/elliptic/geometric_mg.hpp>  // homogeneous(const BCRec&)
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <utility>

// Types DESCRIPTIFS de l'etage elliptique. Ils NOMMENT des valeurs et des
// conventions deja codees aujourd'hui sans modifier une seule operation
// flottante : ce header est un refactor structurel bit-identique.
//
// EllipticProblem rassemble ce qui definit le probleme lap(eps phi) = f resolu
// par GeometricMG ou PoissonFFTSolver :
//   eps             coefficient constant du Laplacien. Aujourd'hui le stencil
//                   ecrit lap = somme/dx2 SANS facteur, c'est-a-dire eps = 1
//                   implicite (cf. apply_laplacian / poisson_residual / gs_color
//                   dans elliptic/poisson_operator.hpp). Le champ est purement
//                   DESCRIPTIF a cette etape : le stencil ne le lit PAS. Le
//                   brancher (multiplier le Laplacien par eps) changerait les
//                   valeurs des que eps != 1, donc hors-perimetre ici.
//   bc              la BCRec physique deja passee partout (Dirichlet, Foextrap,
//                   periodique).
//   nullspace_const la solution n'est definie qu'a une constante additive pres
//                   et cette constante est projetee. Cas periodique :
//                   PoissonFFTSolver met le mode k=0 a zero (phi de moyenne
//                   nulle, cf. poisson_fft_solver.hpp), les tests de GeometricMG
//                   font un demean. C'est une ETIQUETTE : elle ne change aucune
//                   instruction, elle nomme une operation deja effective.
//
// FieldPostProcess nomme la derivation E = -grad phi et la convention de signe.
// Le coupler stocke aux = (phi, +d phi/dx, +d phi/dy) : le signe physique
// E = -grad phi est porte plus loin par diocotron::drift_velocity qui lit
// aux.grad_x / aux.grad_y (cf. model/diocotron.hpp). C'est GradSign::Plus.
// two_fluid_ap::tfap_efield stocke directement E = -grad phi : GradSign::Minus.
// FieldPostProcess::apply reproduit caractere pour caractere l'expression de
// detail::coupler_grad_phi (meme ordre, meme facteur multiplicatif *cx / *cy).

namespace adc {

// (a) Probleme elliptique : coefficient, CL physiques, nullspace.
struct EllipticProblem {
  Real eps = 1;                 // descriptif : etat actuel du stencil (eps = 1)
  BCRec bc{};                   // CL physiques deja propagees
  bool nullspace_const = false; // solution a une constante additive pres
};

// BCRec homogene associee au probleme : delegue a homogeneous(const BCRec&)
// deja dans geometric_mg.hpp (correction multigrille a CL homogenes).
inline BCRec homogeneous_bc(const EllipticProblem& p) {
  return homogeneous(p.bc);
}

// Fabrique additive : construit un EllipticSolver (GeometricMG, PoissonFFTSolver)
// a partir d'un EllipticProblem nomme. Delegue au constructeur (geom, ba, bc,
// active, ...) existant en extrayant problem.bc. eps et nullspace_const sont
// descriptifs a cette etape (eps = 1 deja dans le stencil ; nullspace gere par
// le bottom-solve + demean cote MG, par le mode k=0 a zero cote FFT) : aucun
// appelant existant n'est touche, aucune valeur numerique ne change. Template
// pour eviter un cycle d'inclusion (ce header inclut deja geometric_mg.hpp).
template <class Solver, class... Args>
inline Solver make_elliptic_solver(const Geometry& geom, const BoxArray& ba,
                                   const EllipticProblem& problem,
                                   Args&&... args) {
  return Solver(geom, ba, problem.bc, std::forward<Args>(args)...);
}

// (b) Post-traitement du champ : convention de derivation E = -grad phi.
struct FieldPostProcess {
  enum class GradSign { Plus, Minus };
  GradSign sign = GradSign::Plus;  // +grad (coupler) ou -grad (two_fluid)
  bool store_phi = true;           // phi en composante 0 (convention coupler)
};

// Derive le champ a partir de phi selon spec. Le corps reproduit EXACTEMENT
// detail::coupler_grad_phi (coupler.hpp) : meme ordre d'operations, meme facteur
// multiplicatif *cx / *cy, le signe s = +1 (Plus) ou -1 (Minus) etant le seul
// degre de liberte. cx, cy restent les facteurs centres deja calcules par
// l'appelant (1/(2 dx), 1/(2 dy)).
inline void field_postprocess(const MultiFab& phi, MultiFab& out, Real cx,
                              Real cy, FieldPostProcess spec) {
  const Real s = (spec.sign == FieldPostProcess::GradSign::Plus) ? Real(1)
                                                                 : Real(-1);
  const bool store_phi = spec.store_phi;
  for (int li = 0; li < out.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 a = out.fab(li).array();
    const Box2D v = out.box(li);
    const int gx = store_phi ? 1 : 0;  // decalage de composante si phi stocke
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      if (store_phi) a(i, j, 0) = p(i, j);
      a(i, j, gx) = s * (p(i + 1, j) - p(i - 1, j)) * cx;
      a(i, j, gx + 1) = s * (p(i, j + 1) - p(i, j - 1)) * cy;
    });
  }
}

}  // namespace adc
