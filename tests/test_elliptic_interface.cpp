// Conformite des classes elliptiques EXISTANTES aux concepts communs formalises
// dans elliptic_interface.hpp (audit D.1). Le test est ESSENTIELLEMENT statique : les
// static_assert ci-dessous echouent A LA COMPILATION si une classe cesse de modeler son
// concept. Le main() runtime existe parce que pops_add_test enregistre un binaire ctest ;
// il exerce en plus field_postprocess A TRAVERS le concept FieldPostProcessor (preuve que
// la contrainte est appelable, pas seulement bien-formee) et revalide quelques bits.
//
// AUCUNE classe elliptique n'est modifiee : ce fichier OBSERVE les contrats deja codes.
// Les concepts sont de la metaprogrammation hote (pas de kernel) : zero incidence device,
// la pile elliptique device-validee reste bit-identique.

#include <pops/numerics/elliptic/interface/elliptic_interface.hpp>

#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>      // field_postprocess, FieldPostProcess
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>       // EllipticSolver
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>          // GeometricMG
#include <pops/numerics/elliptic/linear/krylov_solver.hpp>         // TensorKrylovSolver, KrylovResult
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>    // PoissonFFTSolver, DistributedFFTSolver
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>  // PolarPoissonSolver, PolarEllipticSolver

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// =====================================================================================
// (1) EllipticOperator : role d'operateur (coefficients + geom + bc). GeometricMG le porte.
static_assert(EllipticOperator<GeometricMG>,
              "GeometricMG doit modeler EllipticOperator (role operateur : op_eps/op_coef/op_kappa/"
              "op_eps_y/op_a_xy/op_a_yx/op_mask + geom + bc)");

// GAP DOCUMENTE : les solveurs DIRECTS (FFT, polaire) n'exposent PAS de coefficients
// d'operateur (pas de matvec matrice-libre) -> ils ne modelent PAS EllipticOperator.
// C'est le comportement attendu : seul l'operateur MG porte ce role aujourd'hui.
static_assert(
    !EllipticOperator<PoissonFFTSolver>,
    "PoissonFFTSolver (solveur direct) n'a PAS de role operateur a coefficients : attendu");
static_assert(
    !EllipticOperator<PolarPoissonSolver>,
    "PolarPoissonSolver (solveur direct) n'a PAS de role operateur a coefficients : attendu");

// =====================================================================================
// (2) LinearSolver : solveur ITERATIF a solve(rel_tol, max_iters) -> resultat non void.
static_assert(LinearSolver<GeometricMG>,
              "GeometricMG doit modeler LinearSolver (solve(rel_tol, max_cycles) -> int)");
static_assert(
    LinearSolver<TensorKrylovSolver>,
    "TensorKrylovSolver doit modeler LinearSolver (solve(rel_tol, max_iters) -> KrylovResult)");

// Le contrat de socle (rhs/phi/solve()/residual/geom) reste EllipticSolver : tout
// LinearSolver l'est. On le reverifie pour les deux solveurs iteratifs.
static_assert(EllipticSolver<GeometricMG>, "GeometricMG modele EllipticSolver");
static_assert(EllipticSolver<TensorKrylovSolver>, "TensorKrylovSolver modele EllipticSolver");

// GAP DOCUMENTE : les solveurs DIRECTS resolvent en une passe, sans tolerance iterative.
// Ils modelent EllipticSolver (cartesien) ou PolarEllipticSolver (polaire) mais PAS
// LinearSolver. On le PROUVE pour verrouiller la frontiere du concept.
static_assert(EllipticSolver<PoissonFFTSolver>,
              "PoissonFFTSolver modele EllipticSolver (cartesien)");
static_assert(EllipticSolver<DistributedFFTSolver>,
              "DistributedFFTSolver modele EllipticSolver (cartesien)");
static_assert(PolarEllipticSolver<PolarPoissonSolver>,
              "PolarPoissonSolver modele PolarEllipticSolver (polaire)");
static_assert(!LinearSolver<PoissonFFTSolver>,
              "PoissonFFTSolver est DIRECT (pas de solve(tol, iters)) : non-LinearSolver attendu");
static_assert(!LinearSolver<DistributedFFTSolver>,
              "DistributedFFTSolver est DIRECT : non-LinearSolver attendu");
static_assert(!LinearSolver<PolarPoissonSolver>,
              "PolarPoissonSolver est DIRECT : non-LinearSolver attendu");

// Le resultat d'arret est bien NON void pour chaque solveur iteratif (l'invariant commun).
static_assert(!std::is_same_v<decltype(std::declval<GeometricMG&>().solve(Real(1e-8), 1)), void>,
              "GeometricMG::solve(tol, iters) rend un compte rendu (int), pas void");
static_assert(
    !std::is_same_v<decltype(std::declval<TensorKrylovSolver&>().solve(Real(1e-8), 1)), void>,
    "TensorKrylovSolver::solve(tol, iters) rend un compte rendu (KrylovResult), pas void");

// =====================================================================================
// (3) FieldPostProcessor : phi -> aux/grad. field_postprocess (fonction libre) le modele.
// On capture le pointeur de fonction dans un type pour la verification du concept.
using FieldPostProcessFn = void (*)(const MultiFab&, MultiFab&, Real, Real, FieldPostProcess);
static_assert(FieldPostProcessor<FieldPostProcessFn>,
              "field_postprocess (signature (phi, out, cx, cy, spec) -> void) doit modeler "
              "FieldPostProcessor");

// Helper generique CONTRAINT par le concept : ne compile que si pp est un FieldPostProcessor.
// Sert a prouver que le concept est utilisable comme contrainte (pas seulement un predicat).
template <FieldPostProcessor PP>
void apply_pp(PP pp, const MultiFab& phi, MultiFab& out, Real cx, Real cy, FieldPostProcess spec) {
  pp(phi, out, cx, cy, spec);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int N = 32;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // periodique

  auto fr = [&](int i, int j) {
    return std::sin(2 * kPi * geom.x_cell(i)) * std::sin(2 * kPi * geom.y_cell(j));
  };

  // phi connu (1 ghost) periodique, identique au temoin de test_elliptic_problem.
  MultiFab phi(ba, dm, 1, 1);
  {
    Array4 p = phi.fab(0).array();
    const Box2D v = phi.box(0);
    for_each_cell(v, [=] POPS_HD(int i, int j) { p(i, j) = fr(i, j); });
    fill_ghosts(phi, dom, bc);
  }
  const Real cx = Real(1) / (2 * geom.dx());
  const Real cy = Real(1) / (2 * geom.dy());

  // Appel DIRECT vs appel via le helper contraint par FieldPostProcessor : memes bits.
  // Prouve que la fonction libre traverse la contrainte de concept sans rien changer.
  const FieldPostProcess spec{FieldPostProcess::GradSign::Plus, true};
  MultiFab direct(ba, dm, 3, 1), via_concept(ba, dm, 3, 1);
  field_postprocess(phi, direct, cx, cy, spec);
  apply_pp(&field_postprocess, phi, via_concept, cx, cy, spec);

  bool bit_eq = true;
  {
    const ConstArray4 ad = direct.fab(0).const_array();
    const ConstArray4 ac = via_concept.fab(0).const_array();
    const Box2D v = direct.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < 3; ++c)
          if (ad(i, j, c) != ac(i, j, c))
            bit_eq = false;
  }
  chk(bit_eq, "FieldPostProcessor_via_concept_bit_identique");

  // Verification runtime legere : GeometricMG (LinearSolver iteratif) resout et son
  // compte rendu d'arret est un int positif borne par max_cycles. On ne valide pas la
  // physique (couverte ailleurs), seulement le CONTRAT de retour du concept.
  {
    GeometricMG mg(geom, ba, bc);
    Array4 f = mg.rhs().fab(0).array();
    const Box2D v = mg.rhs().box(0);
    for_each_cell(v, [=] POPS_HD(int i, int j) { f(i, j) = fr(i, j); });
    mg.phi().set_val(0.0);
    const int cycles = mg.solve(Real(1e-8), 50);  // variante LinearSolver (tol, iters)
    chk(cycles >= 0 && cycles <= 50, "LinearSolver_GeometricMG_compte_rendu_borne");
    static_assert(std::is_same_v<decltype(cycles), const int>,
                  "GeometricMG::solve(tol, iters) rend int (nombre de V-cycles)");
  }

  if (fails == 0)
    std::printf("OK test_elliptic_interface\n");
  return fails == 0 ? 0 : 1;
}
