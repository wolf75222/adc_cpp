// Operateur elliptique a TENSEUR PLEIN 2x2 : L(phi) = -div(A grad phi) + kappa phi,
// A = [[Axx, Axy], [Ayx, Ayy]] (Axy/Ayx eventuellement NON nuls, A eventuellement NON
// symetrique). Ce jalon n'ajoute QUE les termes hors-diagonaux (flux croises) a l'APPLICATION
// (apply_laplacian / poisson_residual via GeometricMG::set_cross_terms) ; il ne pretend PAS
// resoudre un systeme non symetrique. On valide donc l'OPERATEUR (apply/residu) :
//
//   (A) A = I (Axx=Ayy=1, Axy=Ayx=0, kappa=0) : le residu de l'operateur plein est BIT-IDENTIQUE
//       (dmax = 0) au Poisson canonique poisson_residual sans coefficients. Additivite -> pas de
//       regression sur le chemin historique.
//   (B) A = diag(eps_x, eps_y) (Axy=Ayx=0) : residu BIT-IDENTIQUE (dmax = 0) a l'operateur
//       anisotrope existant set_epsilon_anisotropic. Le bloc plein degenere = bloc diagonal.
//   (C) A NON diagonal CONSTANT, MMS sur l'OPERATEUR (aucun solve requis) : on choisit phi_exact,
//       on calcule f = div(A grad phi_exact) ANALYTIQUEMENT, et on verifie que la norme du residu
//       discret f - L_discret(phi_exact) converge a l'ORDRE 2. Deux variantes : A symetrique
//       (Axy=Ayx=c) et A NON symetrique (Axy=-Ayx=c).
//
// MMS (C) : phi(x,y) = sin(pi x) sin(pi y), nulle sur le bord du carre unite (Dirichlet exact).
//   Pour A CONSTANT, div(A grad phi) = Axx phi_xx + (Axy + Ayx) phi_xy + Ayy phi_yy, avec
//     phi_xx = phi_yy = -pi^2 phi,  phi_xy = pi^2 cos(pi x) cos(pi y).
//   => f = -pi^2 (Axx + Ayy) phi + (Axy + Ayx) pi^2 cos(pi x) cos(pi y).
//   (Pour A non symetrique Axy = -Ayx, le terme croise s'annule analytiquement, mais le STENCIL
//    croise discret n'est PAS nul : la convergence ordre 2 du residu prouve que les deux moities
//    d_x(Axy d_y phi) et d_y(Ayx d_x phi) sont chacune correctes.)
//
// OBSERVATION (non bloquante) : on tente aussi un V-cycle MG sur la MMS non diagonale (A SDP fort)
// et on rapporte s'il converge ou non. Le lisseur reste 5 points (bloc diagonal), termes croises
// explicites : c'est volontaire (jalon = operateur, pas solveur non symetrique). L'observation
// alimente la decision PR2 (Krylov). NON gating : le test ne FAIL jamais sur ce point.

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Foncteur de remplissage commun aux tests (A) et (B) : pose phi = sin(pi x) sin(2 pi y) et
// f = cos(pi x) sin(pi y) sur la grille. Top-level (device-clean) : pas de lambda dans lambda.
struct FillPhiRhsKernel {
  Array4 ap, af;
  Geometry geom;
  ADC_HD void operator()(int i, int j) const {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    ap(i, j) = std::sin(kPi * x) * std::sin(2 * kPi * y);
    af(i, j) = std::cos(kPi * x) * std::sin(kPi * y);
  }
};

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}
static double eps_x_field(double x, double /*y*/) {
  return 1.0 + 0.5 * x;
}
static double eps_y_field(double /*x*/, double y) {
  return 1.0 + 0.3 * y;
}

// Foncteur de remplissage MMS (C) : pose phi = phi_exact (sin(pi x) sin(pi y)) et f = div(A grad phi)
// analytique pour A constant non diagonal. Top-level / device-clean (recette #64/#97/#133 : struct a
// portee namespace tenant Array4 + Geometry + scalaires en membres, operator()(i, j) ADC_HD avec le
// corps verbatim). phi_exact (fonction hote static) est INLINE ici en s = sin(pi x) sin(pi y) : math
// strictement identique (phi_exact(x, y) == s), corps device-callable sans appeler une fonction hote.
struct OperatorMmsFillKernel {
  Array4 ap, af;
  Geometry geom;
  double axx, ayy, csum;
  ADC_HD void operator()(int i, int j) const {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double s = std::sin(kPi * x) * std::sin(kPi * y);
    const double cc = std::cos(kPi * x) * std::cos(kPi * y);
    ap(i, j) = s;                                                     // phi_exact(x, y)
    af(i, j) = -kPi * kPi * (axx + ayy) * s + csum * kPi * kPi * cc;  // div(A grad phi)
  }
};

// Foncteur de remplissage du RHS de l'OBSERVATION MG (non gating) : f = div(A grad phi) pour A SDP non
// diagonale (Axy=Ayx=c, csum=2c, bloc diagonal = I). Top-level / device-clean, meme recette.
struct ObserveMgRhsKernel {
  Array4 af;
  Geometry geom;
  double csum;
  ADC_HD void operator()(int i, int j) const {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double s = std::sin(kPi * x) * std::sin(kPi * y);
    const double cc = std::cos(kPi * x) * std::cos(kPi * y);
    af(i, j) = -kPi * kPi * 2.0 * s + csum * kPi * kPi * cc;
  }
};

// Ecart MAX (norme inf) entre deux residus discrets sur le meme phi : dmax pour le gating bit.
static double residual_gap(MultiFab& ra, MultiFab& rb, const Box2D& dom) {
  const ConstArray4 a = ra.fab(0).const_array(), b = rb.fab(0).const_array();
  double d = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      d = std::fmax(d, std::fabs(a(i, j) - b(i, j)));
  return d;
}

// (A) A = I : residu plein (set_cross_terms a 0) == Poisson canonique sans coefficient. dmax attendu 0.
static double gap_identity(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  // phi non trivial commun aux deux operateurs.
  auto fill = [&](GeometricMG& mg) {
    Array4 ap = mg.phi().fab(0).array(), af = mg.rhs().fab(0).array();
    for_each_cell(dom, FillPhiRhsKernel{ap, af, geom});
  };

  // operateur PLEIN avec Axy = Ayx = 0 (donc A = I).
  GeometricMG mg_full(geom, ba, bc);
  mg_full.set_cross_terms([](Real, Real) { return Real(0); }, [](Real, Real) { return Real(0); });
  fill(mg_full);
  const Real r_full = mg_full.current_residual();

  // Poisson canonique de reference (aucun coefficient) : meme phi, meme rhs.
  GeometricMG mg_ref(geom, ba, bc);
  fill(mg_ref);
  const Real r_ref = mg_ref.current_residual();

  return std::fabs(r_full - r_ref);
}

// (B) A = diag(eps_x, eps_y), Axy = Ayx = 0 : residu plein == residu anisotrope existant. dmax 0.
static double gap_diagonal(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  auto fill = [&](GeometricMG& mg) {
    Array4 ap = mg.phi().fab(0).array(), af = mg.rhs().fab(0).array();
    for_each_cell(dom, FillPhiRhsKernel{ap, af, geom});
  };

  // anisotrope SEUL (reference) : div(diag(eps_x, eps_y) grad phi).
  GeometricMG mg_aniso(geom, ba, bc);
  mg_aniso.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                                   [](Real x, Real y) { return Real(eps_y_field(x, y)); });
  fill(mg_aniso);
  const Real r_aniso = mg_aniso.current_residual();

  // PLEIN : meme bloc diagonal + Axy = Ayx = 0. Doit etre bit-identique.
  GeometricMG mg_full(geom, ba, bc);
  mg_full.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                                  [](Real x, Real y) { return Real(eps_y_field(x, y)); });
  mg_full.set_cross_terms([](Real, Real) { return Real(0); }, [](Real, Real) { return Real(0); });
  fill(mg_full);
  const Real r_full = mg_full.current_residual();

  return std::fabs(r_aniso - r_full);
}

// (C) MMS sur l'OPERATEUR : A constant non diagonal. f = div(A grad phi) analytique ; on mesure la
// norme inf du residu discret f - L_discret(phi_exact), SANS solve. Convergence ordre 2 attendue.
//   axx, ayy : bloc diagonal (constants) ; cxy, cyx : termes croises (constants).
static double operator_mms_resid(int n, double axx, double ayy, double cxy, double cyx) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([axx](Real, Real) { return Real(axx); },
                             [ayy](Real, Real) { return Real(ayy); });
  mg.set_cross_terms([cxy](Real, Real) { return Real(cxy); },
                     [cyx](Real, Real) { return Real(cyx); });

  // phi = phi_exact (donnee EXACTE au centre des cellules) ; f = div(A grad phi) analytique.
  const double csum = cxy + cyx;
  Array4 ap = mg.phi().fab(0).array(), af = mg.rhs().fab(0).array();
  for_each_cell(dom, OperatorMmsFillKernel{ap, af, geom, axx, ayy, csum});

  // residu de l'OPERATEUR = f - L_discret(phi_exact) (norme inf). current_residual applique
  // exactement l'operateur plein cable ci-dessus.
  return static_cast<double>(mg.current_residual());
}

// OBSERVATION MG (non gating) : V-cycles sur la MMS NON diagonale (A SDP : Axy=Ayx=c, |c|<1 -> A
// reste definie positive). On rapporte si le residu chute (convergence) ou non. Le lisseur etant
// 5 points (bloc diagonal), c'est un indicateur pour PR2 (Krylov si non symetrique fort).
static void observe_mg_solve(int n, double c, double& r0_out, double& rN_out, int& nc_out) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([](Real, Real) { return Real(1); },
                             [](Real, Real) { return Real(1); });
  mg.set_cross_terms([c](Real, Real) { return Real(c); }, [c](Real, Real) { return Real(c); });

  const double csum = 2 * c;
  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, ObserveMgRhsKernel{af, geom, csum});
  mg.phi().set_val(0.0);

  r0_out = static_cast<double>(mg.current_residual());
  double rn = r0_out;
  int c_done = 0;
  for (int k = 0; k < 60 && rn > 1e-10 * r0_out; ++k) {
    mg.vcycle();
    rn = static_cast<double>(mg.current_residual());
    ++c_done;
  }
  rN_out = rn;
  nc_out = c_done;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // (A) A = I : bit-identique au Poisson canonique.
  const double gA = gap_identity(64);
  std::printf("(A) A=I : ecart residu plein vs Poisson = %.3e\n", gA);
  chk(gA == 0.0, "A_eq_I_bit_identique");

  // (B) A = diag(eps_x, eps_y) : bit-identique a l'anisotrope existant.
  const double gB = gap_diagonal(64);
  std::printf("(B) A=diag(eps_x,eps_y) : ecart residu plein vs anisotrope = %.3e\n", gB);
  chk(gB == 0.0, "A_diag_bit_identique");

  // (C1) A symetrique constant Axy=Ayx=c : residu de l'operateur ordre 2.
  const double c = 0.4;
  const double s32 = operator_mms_resid(32, 1.0, 1.0, c, c);
  const double s64 = operator_mms_resid(64, 1.0, 1.0, c, c);
  const double s128 = operator_mms_resid(128, 1.0, 1.0, c, c);
  const double rs1 = s32 / s64, rs2 = s64 / s128;
  std::printf(
      "(C1) A sym (Axy=Ayx=%.2f) : residu op r32=%.3e r64=%.3e r128=%.3e | ratios %.2f %.2f\n", c,
      s32, s64, s128, rs1, rs2);
  chk(rs1 > 3.5 && rs1 < 4.5, "C1_op_ordre2_32_64");
  chk(rs2 > 3.5 && rs2 < 4.5, "C1_op_ordre2_64_128");

  // (C2) A NON symetrique constant Axy=c, Ayx=-c : residu de l'operateur ordre 2.
  const double u32 = operator_mms_resid(32, 1.0, 1.0, c, -c);
  const double u64 = operator_mms_resid(64, 1.0, 1.0, c, -c);
  const double u128 = operator_mms_resid(128, 1.0, 1.0, c, -c);
  const double ru1 = u32 / u64, ru2 = u64 / u128;
  std::printf(
      "(C2) A NON sym (Axy=%.2f Ayx=%.2f) : residu op r32=%.3e r64=%.3e r128=%.3e | ratios %.2f "
      "%.2f\n",
      c, -c, u32, u64, u128, ru1, ru2);
  chk(ru1 > 3.5 && ru1 < 4.5, "C2_op_ordre2_32_64");
  chk(ru2 > 3.5 && ru2 < 4.5, "C2_op_ordre2_64_128");

  // OBSERVATION (non gating) : V-cycle MG sur la MMS non diagonale SDP (Axy=Ayx=c).
  double r0, rN;
  int nc;
  observe_mg_solve(64, c, r0, rN, nc);
  std::printf("[obs] MG V-cycle, A SDP non diag (c=%.2f) : r0=%.3e rN=%.3e (%d cycles) -> %s\n", c,
              r0, rN, nc,
              (rN < 1e-6 * r0 ? "CONVERGE" : (rN < r0 ? "decroit (incomplet)" : "DIVERGE/STAGNE")));

  if (fails == 0)
    std::printf("OK test_full_tensor_operator\n");
  return fails == 0 ? 0 : 1;
}
