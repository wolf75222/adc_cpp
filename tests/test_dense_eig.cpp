// adc::real_eig_minmax : extremes du spectre (parties reelles) d'un petit bloc dense.
//
// References EXACTES sans LAPACK : matrices compagnon (racines prescrites d'un polynome),
// matrices triangulaires (spectre = diagonale), similarites entieres S D S^-1 (spectre = D),
// rotations (paire complexe pure). Cas adverses : racines quasi-degenerees (conditionnement
// eps^(1/m), tolerance adaptee -- limite du PROBLEME, documentee dans l'en-tete), matrice
// fortement non normale (triangulaire a grand hors-diagonale). Contrat de repli : cap
// d'iterations a 0 -> bornes de Gershgorin (converged = false) qui ENCADRENT le vrai spectre.

#include <adc/numerics/dense_eig.hpp>

#include <cmath>
#include <cstdio>

using adc::Real;
using adc::EigBounds;
using adc::real_eig_minmax;

static int fails = 0;

static void chk(bool ok, const char* label) {
  std::printf("  [%s] %s\n", ok ? "OK " : "XX ", label);
  if (!ok) ++fails;
}

static bool close_rel(Real a, Real b, Real rtol, Real atol = Real(1e-12)) {
  const Real d = std::fabs(a - b);
  const Real s = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
  return d <= rtol * s + atol;
}

/// Matrice compagnon (premiere ligne = -coefficients) du polynome unitaire de racines @p roots :
/// spectre exactement {roots}. p(x) = prod (x - r_k) developpe par produits successifs.
template <int N>
static void companion(const Real (&roots)[N], Real (&A)[N][N]) {
  Real c[N + 1];  // coefficients de p, c[0] = terme dominant 1
  c[0] = Real(1);
  for (int k = 0; k < N; ++k) c[k + 1] = Real(0);
  for (int k = 0; k < N; ++k)
    for (int j = k + 1; j >= 1; --j) c[j] -= roots[k] * c[j - 1];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) A[i][j] = Real(0);
  for (int j = 0; j < N; ++j) A[0][j] = -c[j + 1];
  for (int i = 1; i < N; ++i) A[i][i - 1] = Real(1);
}

int main() {
  std::printf("== formes fermees N = 1, 2 ==\n");
  {
    const Real A1[1][1] = {{Real(-3.5)}};
    const EigBounds b = real_eig_minmax(A1);
    chk(b.converged && b.lmin == Real(-3.5) && b.lmax == Real(-3.5) && b.max_im == Real(0),
        "N=1 : spectre trivial");
  }
  {
    const Real A2[2][2] = {{Real(1), Real(2)}, {Real(3), Real(0)}};  // lambda = 3, -2
    const EigBounds b = real_eig_minmax(A2);
    chk(b.converged && close_rel(b.lmin, Real(-2), 1e-14) && close_rel(b.lmax, Real(3), 1e-14),
        "N=2 reel : {-2, 3} exact");
  }
  {
    const Real R[2][2] = {{Real(0), Real(2)}, {Real(-2), Real(0)}};  // lambda = +-2i
    const EigBounds b = real_eig_minmax(R);
    chk(b.converged && close_rel(b.lmin, Real(0), 1e-14) && close_rel(b.lmax, Real(0), 1e-14)
            && close_rel(b.max_im, Real(2), 1e-14),
        "N=2 rotation : Re = 0, max_im = 2 (indicateur d'hyperbolicite)");
  }

  std::printf("== N = 3, 4, 5 : racines prescrites (compagnon), rtol 1e-10 ==\n");
  {
    const Real roots[3] = {Real(1), Real(2), Real(3)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(1), 1e-10) && close_rel(b.lmax, Real(3), 1e-10)
            && b.max_im < Real(1e-8),
        "N=3 compagnon {1,2,3}");
  }
  {
    const Real roots[4] = {Real(-2), Real(-0.5), Real(1), Real(4)};
    Real A[4][4];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-2), 1e-10) && close_rel(b.lmax, Real(4), 1e-10),
        "N=4 compagnon {-2,-0.5,1,4}");
  }
  {
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real A[5][5];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-3), 1e-10) && close_rel(b.lmax, Real(5), 1e-10),
        "N=5 compagnon {-3,-1,0,2,5} (taille des blocs HyQMOM, sans en dependre)");
  }

  std::printf("== similarite dense S D S^-1 (spectre exact, matrice pleine) ==\n");
  {
    // S unitriangulaire entiere, D = diag(-1, 0.5, 2, 7) : A = S D S^-1 est PLEINE et son
    // spectre est exactement D (similarite). Calcul de A a la main : S D S^-1 avec
    // S = [[1,0,0,0],[1,1,0,0],[0,1,1,0],[1,0,1,1]], S^-1 entiere aussi (det 1).
    const Real D[4] = {Real(-1), Real(0.5), Real(2), Real(7)};
    const Real S[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 1, 0}, {1, 0, 1, 1}};
    const Real Sinv[4][4] = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {-2, 1, -1, 1}};
    Real A[4][4];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A[i][j] = Real(0);
        for (int k = 0; k < 4; ++k) A[i][j] += S[i][k] * D[k] * Sinv[k][j];
      }
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-1), 1e-10) && close_rel(b.lmax, Real(7), 1e-10),
        "N=4 similarite dense : {-1, 7}");
  }

  std::printf("== N = 5 PLEINE par conjugaison orthogonale (exerce hessenberg_reduce) ==\n");
  {
    // P = I - 2 v v^T / (v^T v) (reflecteur, P = P^-1 exactement) ; A = P C P a le spectre de C
    // mais est DENSE : la reduction de Householder travaille sur toutes ses colonnes (la matrice
    // compagnon, deja Hessenberg, ne l'exercait qu'a vide).
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real C[5][5];
    companion(roots, C);
    const Real v[5] = {1, 2, -1, 3, 1};
    Real vv = 0;
    for (int i = 0; i < 5; ++i) vv += v[i] * v[i];
    Real P[5][5], T[5][5], A[5][5];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) P[i][j] = (i == j ? Real(1) : Real(0)) - 2 * v[i] * v[j] / vv;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        T[i][j] = 0;
        for (int k = 0; k < 5; ++k) T[i][j] += P[i][k] * C[k][j];
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        A[i][j] = 0;
        for (int k = 0; k < 5; ++k) A[i][j] += T[i][k] * P[k][j];
      }
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-3), 1e-10) && close_rel(b.lmax, Real(5), 1e-10),
        "N=5 dense (P C P) : spectre {-3..5} conserve, Householder multi-etapes exerce");
  }

  std::printf("== N = 8 : genericite au-dela des tailles HyQMOM ==\n");
  {
    const Real roots[8] = {Real(-7), Real(-5), Real(-2), Real(-1),
                           Real(1), Real(3), Real(6), Real(9)};
    Real A[8][8];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-7), 1e-9) && close_rel(b.lmax, Real(9), 1e-9),
        "N=8 compagnon {-7..9} (rtol 1e-9 : compagnon de degre 8)");
  }

  std::printf("== spectre mixte reel + paire complexe ==\n");
  {
    // Bloc diagonal : rotation 2x2 (lambda = +-2i) et diag(-1, 3), plonge dans une similarite.
    Real A[4][4] = {{0, 2, 1, 0}, {-2, 0, 0, 1}, {0, 0, -1, 5}, {0, 0, 0, 3}};
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-1), 1e-10) && close_rel(b.lmax, Real(3), 1e-10)
            && close_rel(b.max_im, Real(2), 1e-10),
        "N=4 mixte : Re dans {-1, 0, 3}, max_im = 2");
  }

  std::printf("== cas adverses ==\n");
  {
    // Racines GROUPEES {1-1e-3, 1, 1+1e-3} (separees, pas une vraie multiplicite : l'argument
    // eps^(1/m) ne s'applique qu'a la limite confondue ; ici l'erreur observee est bien moindre).
    // Tolerance ABSOLUE 1e-5 : marge volontaire couvrant la degradation a l'approche du groupe.
    const Real roots[3] = {Real(1) - Real(1e-3), Real(1), Real(1) + Real(1e-3)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && std::fabs(b.lmin - (Real(1) - Real(1e-3))) < Real(1e-5)
            && std::fabs(b.lmax - (Real(1) + Real(1e-3))) < Real(1e-5),
        "N=3 racines groupees a 1e-3 (tolerance large volontaire)");
  }
  {
    // Fortement non normale : triangulaire a hors-diagonale 1e6 -- spectre = diagonale, exact.
    const Real A[3][3] = {{Real(1), Real(1e6), Real(1e6)},
                          {Real(0), Real(2), Real(1e6)},
                          {Real(0), Real(0), Real(-3)}};
    const EigBounds b = real_eig_minmax(A);
    chk(b.converged && close_rel(b.lmin, Real(-3), 1e-10) && close_rel(b.lmax, Real(2), 1e-10),
        "N=3 non normale (hors-diagonale 1e6) : spectre = diagonale");
  }

  std::printf("== bloc compagnon quasi-degenere (cap defaut converge, pas de repli) ==\n");
  {
    // Bloc compagnon 5x5 reel d'un cas HyQMOM : superdiagonale de 1, derniere ligne = coefficients.
    // Spectre quasi-double (paires ~+-1.7326 et ~+-1.7527 + ~0.01) : la deflation QR rampe et
    // demande ~42 iterations. Sous l'ancien cap 30 ce bloc repliait en silence (Gershgorin ~+-15.6,
    // vitesse d'onde sur-estimee ~9x) ; le defaut a 100 le fait converger avec marge.
    // Reference numpy (np.linalg.eigvals) : min Re = -1.732589689893011, max Re = 1.752707143107345.
    Real A[5][5];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) A[i][j] = Real(0);
    for (int i = 0; i < 4; ++i) A[i][i + 1] = Real(1);
    const Real last[5] = {Real(0.0927583829495191), Real(-9.220453484757002),
                          Real(-0.18326928704092538), Real(6.072635227251581),
                          Real(0.05029363303583967)};
    for (int j = 0; j < 5; ++j) A[4][j] = last[j];
    bool fb = true;  // doit etre remis a false : aucun repli attendu au cap defaut
    const EigBounds b = real_eig_minmax(A, /*max_iter_per_eig=*/100, &fb);
    // Tolerance ABSOLUE 1e-6 (et non 1e-9) : la paire superieure est QUASI-DOUBLE, son
    // conditionnement non symetrique est ~eps^(1/2) (~1.5e-8) -- exiger 1e-9 contredirait le
    // contrat documente dans l'en-tete. 1e-6 reste a 7 ordres de grandeur du repli (~+-15.6).
    chk(b.converged && !fb, "compagnon quasi-degenere : converge au cap defaut, fallback = false");
    chk(std::fabs(b.lmin - Real(-1.732589689893011)) < Real(1e-6)
            && std::fabs(b.lmax - Real(1.752707143107345)) < Real(1e-6),
        "min/max corrects (vs numpy) : pas le repli Gershgorin");
    chk(b.max_im < Real(1e-6), "spectre essentiellement reel (max_im ~ 0)");
    // Verrou du DEFAUT : meme bloc appele SANS cap explicite (donc avec le defaut de la signature).
    // Ce bloc demande ~42 iterations ; si le defaut regressait sous ce seuil (p.ex. l'ancien 30) il
    // replirait en silence et bdef.converged passerait a false. Epingle le defaut a >= 42.
    const EigBounds bdef = real_eig_minmax(A);
    chk(bdef.converged && std::fabs(bdef.lmin - Real(-1.732589689893011)) < Real(1e-6)
            && std::fabs(bdef.lmax - Real(1.752707143107345)) < Real(1e-6),
        "cap par DEFAUT suffit a converger (une regression 100->30 ferait echouer ce test)");
  }

  std::printf("== contrat de repli (cap = 0 -> Gershgorin) ==\n");
  {
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real A[5][5];
    companion(roots, A);
    bool fb = false;  // doit passer a true : le parametre de sortie reporte le repli
    const EigBounds b = real_eig_minmax(A, /*max_iter_per_eig=*/0, &fb);
    // Gershgorin ENCADRE le vrai spectre [-3, 5] et le flag dit la verite.
    chk(!b.converged && fb && b.lmin <= Real(-3) && b.lmax >= Real(5),
        "cap 0 : converged = false, fallback = true, bornes de Gershgorin englobantes");
    Real glo, ghi;
    adc::detail::gershgorin_bounds(A, glo, ghi);
    chk(b.lmin == glo && b.lmax == ghi, "le repli EST la borne de Gershgorin (contrat documente)");
  }

  std::printf("FAILS = %d\n", fails);
  return fails ? 1 : 0;
}
