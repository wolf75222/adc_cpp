// MMS du solveur de Poisson POLAIRE direct (PolarPoissonSolver), chantier "grille polaire diocotron"
// Phase 2a. Resout sur un anneau (r, theta), r in [r_min, r_max] > 0 (AUCUNE singularite r = 0),
// theta periodique :
//   (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi = f
// par FFT-en-theta (decouple les modes azimutaux m) + une tridiagonale (Thomas) en r par mode.
//
// On verifie quatre proprietes sur des solutions MANUFACTUREES (phi_exact lisse, periodique en theta,
// f = Laplacien polaire(phi_exact) en forme close) :
//
//   (A) CONVERGENCE EN r a l'ORDRE 2 (Dirichlet aux deux bords). La discretisation radiale est un
//       schema volumes finis conservatif d'ordre 2 (comme assemble_rhs_polar) ; on raffine nr a nth
//       FIXE assez grand pour que l'erreur azimutale (spectrale) soit negligeable -> l'erreur observee
//       est purement radiale et doit decroitre en O(dr^2). Une metrique r erronee donnerait un ordre 0.
//
//   (B) PRECISION SPECTRALE EN theta. La solution manufacturee est un mode azimutal PUR cos(m theta) :
//       la FFT le represente EXACTEMENT (a l'arrondi). A nr grand fixe, augmenter nth ne change plus
//       l'erreur (deja au plancher radial), et meme un nth modeste (>= 2(m+1)) atteint ce plancher.
//       On verifie que l'erreur a nth=16 et nth=64 (mode m=3, plancher radial commun) coincide.
//
//   (C) RESIDU DISCRET ~ARRONDI. Le solveur est DIRECT (FFT + Thomas exacts par mode) : le residu
//       ||L_h phi - f|| du stencil discret doit etre au niveau de l'arrondi, pas une tolerance
//       iterative. C'est le test propre du "solve exact" (pendant polaire de test_poisson_fft).
//
//   (D) BC de NEUMANN homogene + jauge mode 0. Avec deux bords Neumann (Foextrap, flux radial nul a la
//       paroi) l'operateur radial du mode m=0 a la constante pour noyau : le solveur epingle
//       phi_hat(0, 0) = 0. On choisit une solution manufacturee a flux radial nul aux deux bords
//       (d_r phi(r_min) = d_r phi(r_max) = 0) et on verifie la convergence O(2) MODULO une constante
//       additive (jauge), + residu ~arrondi.
//
// Host / Serial-safe : UNE box, n_ranks()==1 dans les 3 jobs CI (test non enregistre MPI ; le solveur
// leve proprement sous MPI, hors scope Phase 2a).

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/polar/polar_poisson_solver.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;

// =================================================================================================
// (A)/(B)/(C) Solution manufacturee DIRICHLET : mode azimutal pur m, profil radial non polynomial.
//   phi(r, theta) = S(r) + H(r) cos(m theta),  S(r) = 1 + 0.5 (r - r_min),  H(r) = sin(a (r - r_min))
//   a = pi / (r_max - r_min).
// CHOIX DE BC : le solveur Phase 2a impose une valeur Dirichlet CONSTANTE en theta (= la part m=0 de
// la donnee de bord) ; les modes m != 0 recoivent une reflexion HOMOGENE (phi=0 a la face). On choisit
// donc H qui s'ANNULE aux deux bords (sin(0)=sin(pi)=0) -> la part cos(m theta) a une valeur de face
// NULLE, exactement representee par la reflexion homogene. La part m=0 = S(r) porte une valeur de face
// CONSTANTE non triviale (S(r_min), S(r_max)), injectee dans le mode 0. La solution exerce donc les
// DEUX chemins de BC (valeur constante non nulle ET reflexion homogene). Laplacien polaire par terme :
//   L[S]            = S'' + S'/r   (S lineaire -> S''=0, S'/r non nul : terme radial non trivial)
//   L[H cos(m th)]  = (H'' + H'/r - m^2 H / r^2) cos(m theta)
// =================================================================================================
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

static double phi_dir(double r, double th, int m) {
  return S(r) + H(r) * std::cos(m * th);
}
static double f_dir(double r, double th, int m) {
  const double lS = Spp(r) + Sp(r) / r;
  const double lH = (Hpp(r) + Hp(r) / r - (double)m * m * H(r) / (r * r)) * std::cos(m * th);
  return lS + lH;
}

// =================================================================================================
// (D) Solution manufacturee NEUMANN homogene aux deux bords (d_r phi = 0 en r_min et r_max).
//   phi(r, theta) = G(r) + K(r) cos(m theta)
//   G(r) = cos(b (r - r_min)),  b = pi / (r_max - r_min)  -> G'(r_min) = G'(r_max) = 0 (sin(0)=sin(pi)=0)
//   K(r) = (r - r_min)^2 (r - r_max)^2  -> K'(r_min) = K'(r_max) = 0 (racine double a chaque bord)
// =================================================================================================
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
static double K(double r) {
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
  return G(r) + K(r) * std::cos(m * th);
}
static double f_neu(double r, double th, int m) {
  const double lG = Gpp(r) + Gp(r) / r;
  const double lK = (Kpp(r) + Kp(r) / r - (double)m * m * K(r) / (r * r)) * std::cos(m * th);
  return lG + lK;
}

// Erreur L2 (ponderee volume r dr dtheta) entre phi numerique et phi exact. @p subtract_mean :
// retire la moyenne (jauge) des deux champs avant de comparer (cas Neumann pur, defini modulo cste).
struct ErrL2 {
  double l2;
  double linf;
};
template <class PhiExact>
static ErrL2 err_vs_exact(const MultiFab& phi, const PolarGeometry& g, const Box2D& dom,
                          PhiExact phi_exact, bool subtract_mean) {
  const ConstArray4 p = phi.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double mean_num = 0, mean_ex = 0, vol = 0;
  if (subtract_mean) {
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double w = g.r_cell(i) * dr * dth;
        mean_num += p(i, j, 0) * w;
        mean_ex += phi_exact(g.r_cell(i), g.theta_cell(j)) * w;
        vol += w;
      }
    mean_num /= vol;
    mean_ex /= vol;
  }
  double l2 = 0, vol2 = 0, linf = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double w = g.r_cell(i) * dr * dth;
      const double e =
          (p(i, j, 0) - mean_num) - (phi_exact(g.r_cell(i), g.theta_cell(j)) - mean_ex);
      l2 += e * e * w;
      vol2 += w;
      if (std::fabs(e) > linf)
        linf = std::fabs(e);
    }
  return {std::sqrt(l2 / vol2), linf};
}

// Resout un cas (Dirichlet OU Neumann) sur (nr x nth) et rend (erreur vs exact, residu discret).
//
// REMPLISSAGE DU RHS = BOUCLE HOTE (pas for_each_cell). La donnee manufacturee f_exact est une
// FONCTION HOTE (f_dir / f_neu, static dans cette TU) passee par pointeur de fonction. Un pointeur de
// fonction HOTE n'est PAS appelable depuis un kernel device (Kokkos::Cuda) : l'appel dans la
// MDRangePolicy est une instruction illegale (cudaErrorIllegalInstruction au prochain
// cudaDeviceSynchronize). On remplit donc le RHS cote HOTE, comme le sibling device-propre
// test_polar_transport_mms (fill_exact = boucle hote, jamais for_each_cell pour des MMS a fonctions
// hote). Le stockage Fab est en memoire UNIFIEE : la donnee ecrite cote hote est lue par solve()
// (algorithme hote) sans copie. PolarPoissonSolver::solve()/residual() font un sync_host() en entree
// (cf. polar_poisson_solver.hpp) : coherence garantie quel que soit le producteur du RHS. Sous
// Serie/OpenMP rien ne change ; sous Kokkos Cuda on evite l'appel device invalide.
template <class PhiExact, class FExact>
static void solve_case(int nr, int nth, int m, const BCRec& bc, PhiExact phi_exact, FExact f_exact,
                       bool subtract_mean, ErrL2& err, double& res) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});

  PolarPoissonSolver solver(g, ba, bc);
  Array4 rhs = solver.rhs().fab(0).array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      rhs(i, j, 0) = f_exact(g.r_cell(i), g.theta_cell(j), m);
  solver.solve();
  err = err_vs_exact(
      solver.phi(), g, dom, [phi_exact, m](double r, double th) { return phi_exact(r, th, m); },
      subtract_mean);
  res = solver.residual();
}

int main() {
  std::printf("=== MMS du solveur de Poisson POLAIRE direct (PolarPoissonSolver), Phase 2a ===\n");
  std::printf("Anneau r in [%.2f, %.2f] (r_min > 0, AUCUNE singularite), theta in [0, 2pi)\n",
              kRmin, kRmax);
  bool ok = true;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("  ECHEC %s\n", w);
      ok = false;
    }
  };

  // ---------------------------------------------------------------------------------------------
  // (A) CONVERGENCE EN r a l'ORDRE 2 (Dirichlet), mode azimutal m=3, nth FIXE = 256 (>> mode -> la
  //     part azimutale spectrale est exacte, l'erreur est purement radiale -> O(dr^2)).
  // ---------------------------------------------------------------------------------------------
  std::printf("\n--- (A) Convergence radiale O(2), Dirichlet, mode m=3 (nth=256 fixe) ---\n");
  BCRec bcd;
  bcd.xlo = bcd.xhi = BCType::Dirichlet;
  // Valeurs Dirichlet aux faces r_min / r_max : la part m=0 (constante en theta) = S(r_face). La part
  // H(r_face) cos(m theta) est NULLE (H s'annule aux bords) -> la reflexion homogene des modes m != 0
  // est exacte. Seule la constante S(r_face) est injectee dans le mode 0 (cf. PolarPoissonSolver).
  bcd.xlo_val = S(kRmin);  // S(r_min) = 1
  bcd.xhi_val = S(kRmax);  // S(r_max) = 1 + 0.5 (r_max - r_min)
  const int m = 3;
  const int nth_fix = 256;
  const int nrs[3] = {32, 64, 128};
  ErrL2 eA[3];
  double resA[3];
  for (int k = 0; k < 3; ++k) {
    solve_case(nrs[k], nth_fix, m, bcd, phi_dir, f_dir, /*subtract_mean=*/false, eA[k], resA[k]);
    std::printf("  nr=%-4d nth=%-4d : L2=%.4e  Linf=%.4e  residu=%.3e\n", nrs[k], nth_fix, eA[k].l2,
                eA[k].linf, resA[k]);
  }
  const double pA1 = std::log2(eA[0].l2 / eA[1].l2);
  const double pA2 = std::log2(eA[1].l2 / eA[2].l2);
  std::printf("  ordre observe (L2) : %.2f (32->64), %.2f (64->128)\n", pA1, pA2);
  chk(pA1 >= 1.7 && pA1 <= 2.3, "ordreA_32_64_dans_[1.7,2.3]");
  chk(pA2 >= 1.7 && pA2 <= 2.3, "ordreA_64_128_dans_[1.7,2.3]");
  chk(resA[2] < 1e-9, "residuA_arrondi");  // (C) residu discret ~arrondi

  // ---------------------------------------------------------------------------------------------
  // (B) PRECISION SPECTRALE EN theta : mode azimutal pur m=3, nr grand FIXE = 512 (plancher radial).
  //     A nth=16 (>= 2(m+1)=8) et nth=64 l'erreur est IDENTIQUE (le mode est deja represente
  //     exactement) -> aucune amelioration en raffinant theta : signature du spectral.
  // ---------------------------------------------------------------------------------------------
  std::printf("\n--- (B) Precision spectrale en theta (mode m=3 pur, nr=512 fixe) ---\n");
  ErrL2 eB16, eB64;
  double rB16, rB64;
  solve_case(512, 16, m, bcd, phi_dir, f_dir, false, eB16, rB16);
  solve_case(512, 64, m, bcd, phi_dir, f_dir, false, eB64, rB64);
  std::printf("  nth=16 : L2=%.6e   nth=64 : L2=%.6e   ecart relatif=%.3e\n", eB16.l2, eB64.l2,
              std::fabs(eB16.l2 - eB64.l2) / eB64.l2);
  // Spectral : l'erreur azimutale est nulle, l'erreur totale = plancher radial commun -> ecart << 1e-3.
  chk(std::fabs(eB16.l2 - eB64.l2) / eB64.l2 < 1e-3, "spectral_theta_nth16_eq_nth64");

  // ---------------------------------------------------------------------------------------------
  // (D) NEUMANN homogene aux deux bords (flux radial nul), jauge mode 0 epinglee. Solution a flux
  //     radial nul aux bords -> O(2) MODULO une constante additive (on retire la moyenne). Mode m=2.
  // ---------------------------------------------------------------------------------------------
  std::printf("\n--- (D) Neumann homogene (2 bords) + jauge mode 0, mode m=2 (nth=256 fixe) ---\n");
  BCRec bcn;
  bcn.xlo = bcn.xhi = BCType::Foextrap;  // Neumann homogene (flux radial nul)
  bcn.ylo = bcn.yhi = BCType::Periodic;
  const int mN = 2;
  ErrL2 eD[3];
  double resD[3];
  for (int k = 0; k < 3; ++k) {
    solve_case(nrs[k], nth_fix, mN, bcn, phi_neu, f_neu, /*subtract_mean=*/true, eD[k], resD[k]);
    std::printf("  nr=%-4d nth=%-4d : L2(jauge)=%.4e  Linf=%.4e  residu=%.3e\n", nrs[k], nth_fix,
                eD[k].l2, eD[k].linf, resD[k]);
  }
  const double pD1 = std::log2(eD[0].l2 / eD[1].l2);
  const double pD2 = std::log2(eD[1].l2 / eD[2].l2);
  std::printf("  ordre observe (L2, jauge) : %.2f (32->64), %.2f (64->128)\n", pD1, pD2);
  chk(pD1 >= 1.6 && pD1 <= 2.4, "ordreD_32_64_dans_[1.6,2.4]");
  chk(pD2 >= 1.6 && pD2 <= 2.4, "ordreD_64_128_dans_[1.6,2.4]");
  chk(resD[2] < 1e-9, "residuD_arrondi");

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  if (ok)
    std::printf("OK test_polar_poisson_mms\n");
  return ok ? 0 : 1;
}
