// MMS de l'operateur elliptique POLAIRE TENSORIEL iteratif (PolarTensorKrylovSolver), Voie A etape 2a
// (brique foundational vers le Schur polaire). Resout sur un anneau (r, theta), r in [r_min, r_max] > 0
// (AUCUNE singularite r=0), theta PERIODIQUE :
//   L(phi) = div(A grad phi),  A = [[a_rr, a_rt], [a_tr, a_tt]]  (eventuellement NON symetrique)
// par BiCGStab matrice-libre preconditionne JACOBI (diagonal). PAS de V-cycle MG (stagnation 1/r^2).
//
// On verifie quatre proprietes sur des solutions MANUFACTUREES (phi_exact lisse, periodique en theta,
// f = div(A grad phi_exact) en forme close) :
//
//   (A) CONVERGENCE O(2) ISOTROPE (A = I, Dirichlet radial). Sanity check : l'operateur tensoriel
//       degenere EXACTEMENT en le Laplacien polaire scalaire quand A = I (a_rr=a_tt=1, a_rt=a_tr=0).
//       On raffine (nr, nth) ensemble et on observe l'ordre 2 (stencil FV conservatif radial + FD
//       azimutal 2 points, tous deux O(h^2)). Une metrique r erronee donnerait un ordre 0.
//
//   (B) CONVERGENCE O(2) TENSEUR NON SYMETRIQUE (termes croises a_rt != a_tr, constants). C'est le
//       coeur de l'etape 2a : un tenseur PLEIN non symetrique (comme la rotation B^{-1} du Schur).
//       Le source analytique a une part cos(m theta) ET sin(m theta) (le terme croise (a_rt+a_tr)
//       phi_rt/r couple r et theta). On verifie l'ordre 2 ET la CONVERGENCE de BiCGStab (rel < tol).
//
//   (C) CONVERGENCE BiCGStab + COMPORTEMENT (pas de stagnation) sur le cas tenseur : on rapporte le
//       nombre d'iterations et le residu relatif a chaque raffinement, et on exige converged==true.
//       C'est le test de la question ouverte du scoping (anisotropie 1/r^2 + termes croises).
//
//   (D) CONSISTANCE ISOTROPE vs PolarPoissonSolver DIRECT. Sur le MEME probleme scalaire (A = I,
//       Dirichlet), l'iteratif tensoriel et le direct (FFT-en-theta + tridiag-en-r) doivent donner la
//       MEME solution a la tolerance pres -- a la difference de stencil azimutal pres (FD 2 points
//       O(dtheta^2) cote iteratif vs spectral -k^2 cote direct), donc l'ecart decroit avec nth. On
//       prend nth grand pour que les deux stencils azimutaux coincident a O(dtheta^2).
//
//   (E) CHOIX DU PRECONDITIONNEUR (documente). On compare le nombre d'iterations BiCGStab de Jacobi
//       (diagonal, iterations ~ 1/h^2, plafonne a grille fine) vs RadialLine (Thomas radial par ligne
//       theta, iterations faibles a croissance moderee). Justifie le DEFAUT RadialLine, sans MG.
//
//   (F) NEUMANN homogene aux DEUX bords radiaux (operateur SINGULIER, constante dans le noyau) +
//       PINNING DE JAUGE iteratif (projection de moyenne FV nulle). Sans pinning BiCGStab DIVERGE ;
//       avec pinning il converge et l'erreur (modulo la jauge) est O(2). Pendant iteratif du pinning de
//       mode 0 du solveur direct.
//
// Host / Serial-safe : UNE box, n_ranks()==1 (solveur mono-rang, non enregistre MPI a l'etape 2a).

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/polar/polar_poisson_solver.hpp>  // (D) reference directe
#include <adc/numerics/elliptic/polar/polar_tensor_operator.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;

// ------------------------------------------------------------------------------------------------
// Solution manufacturee : phi(r, theta) = S(r) + H(r) cos(m theta).
//   S(r) = 1 + 0.5 (r - r_min)            (part m=0, porte la donnee Dirichlet radiale non triviale)
//   H(r) = sin(a (r - r_min)), a = pi/(r_max - r_min)  -> H(r_min) = H(r_max) = 0 (face nulle pour les
//          modes m != 0 ; la reflexion Dirichlet HOMOGENE des modes non nuls est donc exacte).
// ------------------------------------------------------------------------------------------------
static double aS() {
  return kPiL / (kRmax - kRmin);
}
static double S(double r) {
  return 1.0 + 0.5 * (r - kRmin);
}
static double Sp(double /*r*/) {
  return 0.5;
}
static double Spp(double /*r*/) {
  return 0.0;
}
static double H(double r) {
  return std::sin(aS() * (r - kRmin));
}
static double Hp(double r) {
  return aS() * std::cos(aS() * (r - kRmin));
}
static double Hpp(double r) {
  return -aS() * aS() * std::sin(aS() * (r - kRmin));
}

static double phi_exact(double r, double th, int m) {
  return S(r) + H(r) * std::cos(m * th);
}

// Source analytique f = div(A grad phi) pour un tenseur A CONSTANT (a_rr, a_rt, a_tr, a_tt) :
//   div(A grad phi) = a_rr (phi_rr + phi_r/r) + a_tt phi_tt / r^2 + (a_rt + a_tr) phi_rt / r
// avec phi = S + H cos(m th) :
//   phi_rr + phi_r/r = (S'' + S'/r) + (H'' + H'/r) cos(m th)
//   phi_tt / r^2     = -m^2 H cos(m th) / r^2
//   phi_rt / r       = -m H' sin(m th) / r
// (le terme a_rt phi_t/r^2 du flux radial s'annule EXACTEMENT avec un terme egal et oppose de la
//  divergence en coordonnees polaires -> seul (a_rt + a_tr) phi_rt/r subsiste, cf. entete du header).
static double f_tensor(double r, double th, int m, double arr, double art, double atr, double att) {
  const double rad = Spp(r) + Sp(r) / r + (Hpp(r) + Hp(r) / r) * std::cos(m * th);
  const double azi = -(double)m * m * H(r) * std::cos(m * th) / (r * r);
  const double cross = -(double)m * Hp(r) * std::sin(m * th) / r;
  return arr * rad + att * azi + (art + atr) * cross;
}

// ------------------------------------------------------------------------------------------------
// (F) Solution manufacturee NEUMANN homogene aux DEUX bords radiaux (d_r phi = 0 en r_min/r_max) ->
//   operateur SINGULIER (constante dans le noyau), defini modulo une constante (jauge). On valide le
//   PINNING DE JAUGE iteratif (projection de moyenne nulle). A = I (scalaire) suffit a exercer le
//   chemin singulier (les termes croises n'y changent pas la nature du noyau).
//   phi(r, theta) = G(r) + K(r) cos(m theta) avec G'(r_min)=G'(r_max)=0 et K'(r_min)=K'(r_max)=0.
// ------------------------------------------------------------------------------------------------
static double bN() {
  return kPiL / (kRmax - kRmin);
}
static double G(double r) {
  return std::cos(bN() * (r - kRmin));
}
static double Gp(double r) {
  return -bN() * std::sin(bN() * (r - kRmin));
}
static double Gpp(double r) {
  return -bN() * bN() * std::cos(bN() * (r - kRmin));
}
static double Kf(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return u * u * w * w;
}
static double Kp(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return 2 * u * w * w + 2 * u * u * w;
}
static double Kpp(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return 2 * w * w + 8 * u * w + 2 * u * u;
}
static double phi_neu(double r, double th, int m) {
  return G(r) + Kf(r) * std::cos(m * th);
}
static double f_neu(double r, double th, int m) {  // A = I
  const double rad = Gpp(r) + Gp(r) / r + (Kpp(r) + Kp(r) / r) * std::cos(m * th);
  const double azi = -(double)m * m * Kf(r) * std::cos(m * th) / (r * r);
  return rad + azi;
}

// Erreur L2 (ponderee volume r dr dtheta) entre phi numerique et phi exact.
struct ErrL2 {
  double l2;
  double linf;
};
static ErrL2 err_vs_exact(const MultiFab& phi, const PolarGeometry& g, const Box2D& dom, int m) {
  const ConstArray4 p = phi.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double l2 = 0, vol = 0, linf = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double w = g.r_cell(i) * dr * dth;
      const double e = p(i, j, 0) - phi_exact(g.r_cell(i), g.theta_cell(j), m);
      l2 += e * e * w;
      vol += w;
      if (std::fabs(e) > linf)
        linf = std::fabs(e);
    }
  return {std::sqrt(l2 / vol), linf};
}

// Remplit un champ scalaire CONSTANT sur les cellules valides (boucle hote ; les ghosts sont remplis
// par set_coefficients via fill_ghosts).
static void fill_const(MultiFab& mf, const Box2D& dom, double val) {
  Array4 a = mf.fab(0).array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      a(i, j, 0) = val;
}

// DistributionMapping helper (boite unique mono-rang) pour les MultiFab de coefficient du test.
static DistributionMapping solver_dm(const BoxArray& ba) {
  return DistributionMapping(ba.size(), n_ranks());
}

// Resout le cas tenseur (arr, art, atr, att) sur (nr x nth), mode m, BC Dirichlet, et rend (erreur,
// resultat BiCGStab). Le RHS est rempli cote HOTE (f_tensor = fonction hote ; un pointeur de fonction
// hote n'est pas appelable depuis un kernel device, cf. test_polar_poisson_mms).
static PolarKrylovResult solve_tensor(int nr, int nth, int m, double arr, double art, double atr,
                                      double att, ErrL2& err,
                                      PolarPrecond pc = PolarPrecond::RadialLine) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});

  BCRec bc;
  bc.xlo = bc.xhi = BCType::Dirichlet;
  bc.ylo = bc.yhi = BCType::Periodic;  // theta periodique
  bc.xlo_val = S(kRmin);               // donnee Dirichlet = part m=0 (H s'annule aux bords)
  bc.xhi_val = S(kRmax);

  PolarTensorKrylovSolver solver(g, ba, bc, pc);

  // Coefficients du tenseur (champs au centre, constants ici). a_rr/a_tt fournis ; a_rt/a_tr seulement
  // s'ils sont non nuls (sinon on laisse l'operateur en mode diagonal).
  MultiFab arr_mf(ba, solver_dm(ba), 1, 1), att_mf(ba, solver_dm(ba), 1, 1);
  MultiFab art_mf(ba, solver_dm(ba), 1, 1), atr_mf(ba, solver_dm(ba), 1, 1);
  fill_const(arr_mf, dom, arr);
  fill_const(att_mf, dom, att);
  const bool cross = (art != 0.0) || (atr != 0.0);
  if (cross) {
    fill_const(art_mf, dom, art);
    fill_const(atr_mf, dom, atr);
    solver.set_coefficients(&arr_mf, &att_mf, &art_mf, &atr_mf);
  } else {
    solver.set_coefficients(&arr_mf, &att_mf);
  }

  Array4 rhs = solver.rhs().fab(0).array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      rhs(i, j, 0) = f_tensor(g.r_cell(i), g.theta_cell(j), m, arr, art, atr, att);

  solver.phi().set_val(0.0);  // depart froid
  PolarKrylovResult kr = solver.solve(1e-11, 4000);
  err = err_vs_exact(solver.phi(), g, dom, m);
  return kr;
}

int main() {
  std::printf(
      "=== MMS de l'operateur elliptique POLAIRE TENSORIEL iteratif (Voie A etape 2a) ===\n");
  std::printf(
      "Anneau r in [%.2f, %.2f] (r_min > 0), theta in [0, 2pi). BiCGStab + precond RadialLine.\n",
      kRmin, kRmax);
  bool ok = true;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("  ECHEC %s\n", w);
      ok = false;
    }
  };

  const int m = 3;
  const int nrs[3] = {32, 64, 128};

  // ---------------------------------------------------------------------------------------------
  // (A) CONVERGENCE O(2) ISOTROPE (A = I). nr = nth raffines ensemble (erreur radiale ET azimutale O(2)).
  // ---------------------------------------------------------------------------------------------
  std::printf("\n--- (A) Convergence O(2) ISOTROPE (A = I), mode m=%d ---\n", m);
  ErrL2 eA[3];
  for (int k = 0; k < 3; ++k) {
    PolarKrylovResult kr = solve_tensor(nrs[k], nrs[k], m, 1.0, 0.0, 0.0, 1.0, eA[k]);
    std::printf("  n=%-4d : L2=%.4e  Linf=%.4e  [BiCGStab iters=%d rel=%.2e conv=%d]\n", nrs[k],
                eA[k].l2, eA[k].linf, kr.iters, kr.rel_residual, (int)kr.converged);
    chk(kr.converged, "A_bicgstab_converge");
  }
  const double pA1 = std::log2(eA[0].l2 / eA[1].l2);
  const double pA2 = std::log2(eA[1].l2 / eA[2].l2);
  std::printf("  ordre observe (L2) : %.2f (32->64), %.2f (64->128)\n", pA1, pA2);
  chk(pA1 >= 1.7 && pA1 <= 2.3, "ordreA_32_64_dans_[1.7,2.3]");
  chk(pA2 >= 1.7 && pA2 <= 2.3, "ordreA_64_128_dans_[1.7,2.3]");

  // ---------------------------------------------------------------------------------------------
  // (B)/(C) CONVERGENCE O(2) TENSEUR NON SYMETRIQUE + BiCGStab converge. a_rr=1.4, a_tt=0.8,
  //   a_rt=0.5, a_tr=-0.3 (a_rt != a_tr -> A NON symetrique, comme la rotation B^{-1} du Schur).
  //   A reste DEFINI POSITIF en partie symetrique (a_rr a_tt > ((a_rt+a_tr)/2)^2 = 0.01 ; 1.12 >> 0.01)
  //   donc l'operateur diagonal-dominant est inversible et BiCGStab+Jacobi doit converger.
  // ---------------------------------------------------------------------------------------------
  std::printf("\n--- (B)/(C) Convergence O(2) TENSEUR NON SYMETRIQUE (a_rt=0.5, a_tr=-0.3) ---\n");
  const double arr = 1.4, att = 0.8, art = 0.5, atr = -0.3;
  ErrL2 eB[3];
  for (int k = 0; k < 3; ++k) {
    PolarKrylovResult kr = solve_tensor(nrs[k], nrs[k], m, arr, art, atr, att, eB[k]);
    std::printf("  n=%-4d : L2=%.4e  Linf=%.4e  [BiCGStab iters=%d rel=%.2e conv=%d]\n", nrs[k],
                eB[k].l2, eB[k].linf, kr.iters, kr.rel_residual, (int)kr.converged);
    chk(kr.converged, "B_bicgstab_converge");  // (C) pas de stagnation
  }
  const double pB1 = std::log2(eB[0].l2 / eB[1].l2);
  const double pB2 = std::log2(eB[1].l2 / eB[2].l2);
  std::printf("  ordre observe (L2) : %.2f (32->64), %.2f (64->128)\n", pB1, pB2);
  chk(pB1 >= 1.7 && pB1 <= 2.3, "ordreB_32_64_dans_[1.7,2.3]");
  chk(pB2 >= 1.7 && pB2 <= 2.3, "ordreB_64_128_dans_[1.7,2.3]");

  // ---------------------------------------------------------------------------------------------
  // (D) CONSISTANCE ISOTROPE vs PolarPoissonSolver DIRECT. Meme probleme scalaire (A = I, Dirichlet),
  //   nth grand (=256) pour que le stencil azimutal FD 2 points (iteratif) coincide avec le spectral
  //   -k^2 (direct) a O(dtheta^2). On compare les deux solutions point a point (ecart L2 relatif petit).
  // ---------------------------------------------------------------------------------------------
  std::printf(
      "\n--- (D) Consistance ISOTROPE iteratif vs PolarPoissonSolver direct (nr=128, nth=256) "
      "---\n");
  {
    const int nr = 128, nth = 256;
    Box2D dom = Box2D::from_extents(nr, nth);
    PolarGeometry g{dom, kRmin, kRmax};
    BoxArray ba(std::vector<Box2D>{dom});
    BCRec bc;
    bc.xlo = bc.xhi = BCType::Dirichlet;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.xlo_val = S(kRmin);
    bc.xhi_val = S(kRmax);

    // Iteratif (A = I).
    PolarTensorKrylovSolver itr(g, ba, bc);
    MultiFab one_rr(ba, solver_dm(ba), 1, 1), one_tt(ba, solver_dm(ba), 1, 1);
    fill_const(one_rr, dom, 1.0);
    fill_const(one_tt, dom, 1.0);
    itr.set_coefficients(&one_rr, &one_tt);
    {
      Array4 rhs = itr.rhs().fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          rhs(i, j, 0) = f_tensor(g.r_cell(i), g.theta_cell(j), m, 1.0, 0.0, 0.0, 1.0);
    }
    itr.phi().set_val(0.0);
    PolarKrylovResult kr = itr.solve(1e-11, 2000);
    chk(kr.converged, "D_bicgstab_converge");

    // Direct.
    PolarPoissonSolver dir(g, ba, bc);
    {
      Array4 rhs = dir.rhs().fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          rhs(i, j, 0) = f_tensor(g.r_cell(i), g.theta_cell(j), m, 1.0, 0.0, 0.0, 1.0);
    }
    dir.solve();

    const ConstArray4 pi = itr.phi().fab(0).const_array();
    const ConstArray4 pd = dir.phi().fab(0).const_array();
    const double dr = g.dr(), dth = g.dtheta();
    double diff = 0, ref = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double w = g.r_cell(i) * dr * dth;
        const double e = pi(i, j, 0) - pd(i, j, 0);
        diff += e * e * w;
        ref += pd(i, j, 0) * pd(i, j, 0) * w;
      }
    const double rel = std::sqrt(diff / ref);
    std::printf("  iteratif iters=%d rel=%.2e ; ecart L2 relatif iteratif/direct = %.3e\n",
                kr.iters, kr.rel_residual, rel);
    // Les deux solveurs partagent le stencil radial FV ; ils ne different qu'en theta (FD 2 points vs
    // spectral), ecart O(dtheta^2). A nth=256, dtheta ~ 0.0245, dtheta^2 ~ 6e-4 : ecart bien < 1e-2.
    chk(rel < 1e-2, "D_consistance_iteratif_vs_direct");
  }

  // ---------------------------------------------------------------------------------------------
  // (E) CHOIX DU PRECONDITIONNEUR (documente, pas un critere d'echec). On compare le nombre
  //   d'iterations BiCGStab de Jacobi (diagonal) vs RadialLine (Thomas radial par ligne theta) sur le
  //   cas tenseur, a deux finesses. Jacobi : iterations ~ 1/h^2 (mauvais conditionnement Laplacien) ->
  //   stagne/plafonne a grille fine. RadialLine : iterations QUASI INDEPENDANTES de h (le couplage
  //   radial fort est inverse exactement, l'anisotropie 1/r^2 est dans la diagonale lumpee). C'est la
  //   raison du DEFAUT RadialLine. On exige seulement que RadialLine fasse STRICTEMENT moins
  //   d'iterations que Jacobi a la grille fine (preuve quantitative du gain ; pas de MG requis).
  // ---------------------------------------------------------------------------------------------
  std::printf(
      "\n--- (E) Preconditionneur Jacobi vs RadialLine (cas tenseur, iterations BiCGStab) ---\n");
  {
    // solve_tensor utilise un plafond large (4000 iters) : Jacobi a une vraie chance de converger a la
    // grille fine, ce qui rend la comparaison honnete (il plafonne quand meme a n=96).
    for (int n : {32, 96}) {
      ErrL2 ej, el;
      PolarKrylovResult krj = solve_tensor(n, n, m, arr, art, atr, att, ej, PolarPrecond::Jacobi);
      PolarKrylovResult krl =
          solve_tensor(n, n, m, arr, art, atr, att, el, PolarPrecond::RadialLine);
      std::printf("  n=%-4d : Jacobi iters=%-5d (conv=%d) | RadialLine iters=%-4d (conv=%d)\n", n,
                  krj.iters, (int)krj.converged, krl.iters, (int)krl.converged);
      if (n == 96) {
        chk(krl.converged, "E_radialline_converge_grille_fine");
        chk(krl.iters < krj.iters || !krj.converged,
            "E_radialline_moins_iters_que_jacobi");  // gain quantitatif du precond ligne
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // (F) NEUMANN homogene aux DEUX bords (operateur SINGULIER) + PINNING DE JAUGE iteratif. Sans le
  //   pinning (projection de moyenne nulle), BiCGStab DIVERGE (la constante du noyau n'est pas amortie).
  //   Avec le pinning, il converge et l'erreur (modulo la jauge = moyenne FV retiree) est O(2). On
  //   raffine (nr=nth) et on observe l'ordre 2 + convergence. Mode m=2.
  // ---------------------------------------------------------------------------------------------
  std::printf(
      "\n--- (F) Neumann homogene 2 bords (operateur singulier) + pinning de jauge, mode m=2 "
      "---\n");
  {
    const int mN = 2;
    double pF1 = 0, pF2 = 0;
    double l2s[3];
    for (int k = 0; k < 3; ++k) {
      const int n = nrs[k];
      Box2D dom = Box2D::from_extents(n, n);
      PolarGeometry g{dom, kRmin, kRmax};
      BoxArray ba(std::vector<Box2D>{dom});
      BCRec bc;
      bc.xlo = bc.xhi = BCType::Foextrap;  // Neumann homogene (flux radial nul)
      bc.ylo = bc.yhi = BCType::Periodic;
      PolarTensorKrylovSolver solver(g, ba, bc);
      MultiFab one_rr(ba, solver_dm(ba), 1, 1), one_tt(ba, solver_dm(ba), 1, 1);
      fill_const(one_rr, dom, 1.0);
      fill_const(one_tt, dom, 1.0);
      solver.set_coefficients(&one_rr, &one_tt);
      Array4 rhs = solver.rhs().fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          rhs(i, j, 0) = f_neu(g.r_cell(i), g.theta_cell(j), mN);
      solver.phi().set_val(0.0);
      PolarKrylovResult kr = solver.solve(1e-10, 4000);
      chk(kr.converged, "F_neumann_pinning_converge");
      // erreur L2 ponderee, jauge (moyenne FV) retiree des deux champs.
      const ConstArray4 p = solver.phi().fab(0).const_array();
      const double dr = g.dr(), dth = g.dtheta();
      double mn = 0, me = 0, vol = 0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double w = g.r_cell(i) * dr * dth;
          mn += p(i, j, 0) * w;
          me += phi_neu(g.r_cell(i), g.theta_cell(j), mN) * w;
          vol += w;
        }
      mn /= vol;
      me /= vol;
      double l2 = 0, v2 = 0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double w = g.r_cell(i) * dr * dth;
          const double e = (p(i, j, 0) - mn) - (phi_neu(g.r_cell(i), g.theta_cell(j), mN) - me);
          l2 += e * e * w;
          v2 += w;
        }
      l2s[k] = std::sqrt(l2 / v2);
      std::printf("  n=%-4d : L2(jauge)=%.4e  [BiCGStab iters=%d rel=%.2e conv=%d]\n", n, l2s[k],
                  kr.iters, kr.rel_residual, (int)kr.converged);
    }
    pF1 = std::log2(l2s[0] / l2s[1]);
    pF2 = std::log2(l2s[1] / l2s[2]);
    std::printf("  ordre observe (L2, jauge) : %.2f (32->64), %.2f (64->128)\n", pF1, pF2);
    chk(pF1 >= 1.6 && pF1 <= 2.4, "ordreF_32_64_dans_[1.6,2.4]");
    chk(pF2 >= 1.6 && pF2 <= 2.4, "ordreF_64_128_dans_[1.6,2.4]");
  }

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  if (ok)
    std::printf("OK test_polar_tensor_elliptic_mms\n");
  return ok ? 0 : 1;
}
