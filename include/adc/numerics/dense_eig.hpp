#pragma once
/// @file
/// @brief Extremes du spectre (parties reelles) d'une petite matrice dense : bornes de vitesses
/// d'onde signees fournies par un modele (HLL exact via jacobien de flux).
///
/// Utilitaire GENERIQUE : le coeur ne sait rien du modele appelant -- il recoit un bloc dense
/// Real[N][N] et rend min/max des parties reelles de son spectre. Consommateur vise : le codegen
/// DSL wave_speeds_from_jacobian (le modele fournit dF/dU, eventuellement par blocs diagonaux ;
/// chaque bloc passe ici). Tout autre usage "petites valeurs propres sur la pile" est legitime.
///
/// Algorithme (valeurs propres SEULES, jamais les vecteurs) :
///   N == 1, 2 : forme fermee (trace/determinant) ;
///   N >= 3    : reduction de Hessenberg (Householder, sans accumulation) puis iteration QR a
///               DOUBLE SHIFT implicite de Francis avec deflation (formulation EISPACK/hqr) --
///               les paires complexes conjuguees restent en arithmetique reelle (blocs 2x2),
///               shifts exceptionnels apres 10 et 20 iterations sur un meme bloc.
///
/// CONTRAT DE ROBUSTESSE : si un bloc ne converge pas sous le cap d'iterations, le resultat est
/// le REPLI de Gershgorin sur la matrice ENTIERE (converged = false) : un encadrement EXTERNE
/// toujours valide de toutes les parties reelles (sL <= toutes les vitesses <= sR, donc un flux
/// HLL stable, simplement plus diffusif). Si le repli se declenche, lmin/lmax ne sont PAS les
/// valeurs propres et max_im vaut 0 par CONVENTION (rien n'a ete calcule, ce n'est PAS un signal
/// de spectre reel) : tout bit-match contre une reference eig est alors caduc. converged (ou le
/// parametre de sortie fallback) tranche : ne jamais lire lmin/lmax/max_im sans l'avoir consulte.
/// Le cap par defaut (100) est dimensionne pour que les blocs compagnons quasi-degeneres usuels
/// convergent ; le repli reste le filet de securite pour les cas hors gabarit.
///
/// HYPERBOLICITE : max_im rend le plus grand |Im(lambda)| rencontre. Un systeme hyperbolique a
/// un spectre reel (max_im ~ 0) ; un modele qui perd l'hyperbolicite ne recoit pas une vitesse
/// plausible-mais-fausse en silence -- l'appelant decide (assert, warning, clamp).
///
/// PRECISION : valeurs propres simples et separees -> precision machine (testee a rtol 1e-10).
/// Valeurs propres GROUPEES d'une matrice non symetrique : conditionnement ~ eps^(1/m) pour une
/// quasi-multiplicite m (limite du probleme, pas de l'algorithme) -- ne pas exiger 1e-10 sur un
/// triple point. Aucune revendication d'exactitude n'est faite sous le cap d'iterations : le cap
/// borne le COUT, le repli borne le RESULTAT.
///
/// Device : ADC_HD, tampons sur la pile (O(N^2)), zero allocation, zero conteneur std::, boucles
/// bornees, pas de recursion -- seuls std::sqrt / std::fabs (resolus device sous nvcc/Kokkos,
/// comme sur le chemin de flux).

#include <cmath>
#include <limits>

#include <adc/core/types.hpp>

namespace adc {

/// Resultat de real_eig_minmax : extremes des parties reelles + diagnostic. Le consommateur
/// (codegen DSL wave_speeds_from_jacobian, ADC-87) recoit la structure ENTIERE : converged et
/// max_im font partie du contrat de securite, aucune surcharge ne les jette en silence.
struct EigBounds {
  Real lmin;      ///< plus petite partie reelle (ou borne basse de Gershgorin si !converged)
  Real lmax;      ///< plus grande partie reelle (ou borne haute de Gershgorin si !converged)
  Real max_im;    ///< plus grand |Im(lambda)| rencontre (0 = spectre reel : hyperbolique). N'a ce
                  ///< sens QUE si converged : sous repli il vaut 0 par CONVENTION (le spectre n'est
                  ///< pas calcule), surtout pas un signal d'hyperbolicite -- lire converged d'abord.
  bool converged; ///< false -> repli Gershgorin (encadrement externe valide, PAS le spectre)
};

namespace detail {

/// Encadrement de Gershgorin des PARTIES REELLES : tout lambda du spectre verifie
/// lo <= Re(lambda) <= hi (disques centres a_ii de rayon somme des |hors-diagonale| de la
/// ligne). Borne externe sure pour HLL, atteinte seulement si la matrice est diagonale.
template <int N>
ADC_HD inline void gershgorin_bounds(const Real (&A)[N][N], Real& lo, Real& hi) {
  for (int i = 0; i < N; ++i) {
    Real r = Real(0);
    for (int j = 0; j < N; ++j)
      if (j != i) r += std::fabs(A[i][j]);
    const Real l = A[i][i] - r, h = A[i][i] + r;
    if (i == 0 || l < lo) lo = l;
    if (i == 0 || h > hi) hi = h;
  }
}

/// Reduction de Hessenberg superieure par reflexions de Householder, EN PLACE, sans accumulation
/// des transformations (valeurs propres seules). Stable inconditionnellement.
template <int N>
ADC_HD inline void hessenberg_reduce(Real (&H)[N][N]) {
  Real v[N];  // vecteur de Householder de l'etape courante (composantes k..N-1)
  for (int k = 1; k <= N - 2; ++k) {
    Real scale = Real(0);
    for (int i = k; i < N; ++i) scale += std::fabs(H[i][k - 1]);
    if (scale == Real(0)) continue;  // colonne deja nulle sous la sous-diagonale
    Real h = Real(0);
    for (int i = k; i < N; ++i) {
      v[i] = H[i][k - 1] / scale;
      h += v[i] * v[i];
    }
    Real g = std::sqrt(h);
    if (v[k] > Real(0)) g = -g;
    h -= v[k] * g;       // h = v.v / 2 apres mise a jour de v[k]
    v[k] -= g;
    if (h == Real(0)) continue;
    // P = I - v v^T / h ; H <- P H P (colonne k-1 fixee explicitement, zeros exacts dessous)
    for (int j = k; j < N; ++j) {  // P * H sur les lignes k..N-1
      Real f = Real(0);
      for (int i = k; i < N; ++i) f += v[i] * H[i][j];
      f /= h;
      for (int i = k; i < N; ++i) H[i][j] -= f * v[i];
    }
    for (int i = 0; i < N; ++i) {  // H * P sur les colonnes k..N-1
      Real f = Real(0);
      for (int j = k; j < N; ++j) f += H[i][j] * v[j];
      f /= h;
      for (int j = k; j < N; ++j) H[i][j] -= f * v[j];
    }
    H[k][k - 1] = scale * g;
    for (int i = k + 1; i < N; ++i) H[i][k - 1] = Real(0);
  }
}

ADC_HD inline Real hqr_copysign(Real mag, Real sgn) {
  return sgn >= Real(0) ? std::fabs(mag) : -std::fabs(mag);
}

/// Accumule une valeur propre (re, im) dans les extremes courants. Fonction nommee plutot qu'une
/// lambda locale : prudence device (nvcc et les lambdas dans du code __host__ __device__).
ADC_HD inline void record_eig(Real re, Real im, Real& lmin, Real& lmax, Real& max_im,
                              bool& first) {
  if (first || re < lmin) lmin = re;
  if (first || re > lmax) lmax = re;
  const Real ai = std::fabs(im);
  if (first || ai > max_im) max_im = ai;
  first = false;
}

/// Iteration QR a double shift implicite de Francis sur une matrice de Hessenberg (formulation
/// EISPACK/hqr, valeurs propres seules, blocs traites du bas vers le haut avec deflation).
/// Accumule directement min/max des parties reelles et max|Im|. @return true si TOUT le spectre
/// est extrait sous le cap (@p max_iter_per_eig iterations par bloc actif), false sinon.
template <int N>
ADC_HD inline bool hqr_minmax(Real (&H)[N][N], Real& lmin, Real& lmax, Real& max_im,
                              int max_iter_per_eig) {
  constexpr Real kEps = std::numeric_limits<Real>::epsilon();  // suit le type Real
  Real anorm = Real(0);  // norme de la partie Hessenberg (critere de deflation des cas s == 0)
  for (int i = 0; i < N; ++i)
    for (int j = (i > 0 ? i - 1 : 0); j < N; ++j) anorm += std::fabs(H[i][j]);
  if (anorm == Real(0)) {  // matrice nulle : spectre {0}
    lmin = lmax = max_im = Real(0);
    return true;
  }

  bool first = true;

  int nn = N - 1;     // indice haut du bloc actif
  Real t = Real(0);   // shift cumule (shifts exceptionnels)
  Real p = Real(0), q = Real(0), r = Real(0), x, y, z, w, s;
  while (nn >= 0) {
    int its = 0;
    int l;
    do {
      // deflation : plus petit l tel que H[l][l-1] soit negligeable
      for (l = nn; l >= 1; --l) {
        s = std::fabs(H[l - 1][l - 1]) + std::fabs(H[l][l]);
        if (s == Real(0)) s = anorm;
        if (std::fabs(H[l][l - 1]) <= kEps * s) {
          H[l][l - 1] = Real(0);
          break;
        }
      }
      x = H[nn][nn];
      if (l == nn) {  // valeur propre reelle 1x1
        record_eig(x + t, Real(0), lmin, lmax, max_im, first);
        --nn;
      } else {
        y = H[nn - 1][nn - 1];
        w = H[nn][nn - 1] * H[nn - 1][nn];
        if (l == nn - 1) {  // bloc 2x2 : paire reelle ou complexe conjuguee
          p = Real(0.5) * (y - x);
          q = p * p + w;
          z = std::sqrt(std::fabs(q));
          x += t;
          if (q >= Real(0)) {  // deux valeurs reelles
            z = p + hqr_copysign(z, p);
            record_eig(x + z, Real(0), lmin, lmax, max_im, first);
            record_eig(z != Real(0) ? x - w / z : x + z, Real(0), lmin, lmax, max_im, first);
          } else {             // paire complexe : Re = x + p, |Im| = z
            record_eig(x + p, z, lmin, lmax, max_im, first);
            record_eig(x + p, -z, lmin, lmax, max_im, first);
          }
          nn -= 2;
        } else {  // bloc > 2 : une iteration de double shift de Francis
          if (its == max_iter_per_eig) return false;  // cap atteint -> repli appelant
          if (its == 10 || its == 20) {  // shift exceptionnel (cycles lents)
            t += x;
            for (int i = 0; i <= nn; ++i) H[i][i] -= x;
            s = std::fabs(H[nn][nn - 1]) + std::fabs(H[nn - 1][nn - 2]);
            y = x = Real(0.75) * s;
            w = Real(-0.4375) * s * s;
          }
          ++its;
          int m;
          for (m = nn - 2; m >= l; --m) {  // deux sous-diagonales consecutives petites
            z = H[m][m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
            q = H[m + 1][m + 1] - z - r - s;
            r = H[m + 2][m + 1];
            s = std::fabs(p) + std::fabs(q) + std::fabs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m == l) break;
            const Real u = std::fabs(H[m][m - 1]) * (std::fabs(q) + std::fabs(r));
            const Real v = std::fabs(p) * (std::fabs(H[m - 1][m - 1]) + std::fabs(z)
                                           + std::fabs(H[m + 1][m + 1]));
            if (u <= kEps * v) break;
          }
          for (int i = m + 2; i <= nn; ++i) {
            H[i][i - 2] = Real(0);
            if (i > m + 2) H[i][i - 3] = Real(0);
          }
          for (int k = m; k <= nn - 1; ++k) {  // balayage QR double shift sur les colonnes m..nn-1
            if (k != m) {
              p = H[k][k - 1];
              q = H[k + 1][k - 1];
              r = (k != nn - 1) ? H[k + 2][k - 1] : Real(0);
              x = std::fabs(p) + std::fabs(q) + std::fabs(r);
              if (x != Real(0)) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            s = hqr_copysign(std::sqrt(p * p + q * q + r * r), p);
            if (s == Real(0)) continue;
            if (k == m) {
              if (l != m) H[k][k - 1] = -H[k][k - 1];
            } else {
              H[k][k - 1] = -s * x;
            }
            p += s;
            x = p / s;
            y = q / s;
            z = r / s;
            q /= p;
            r /= p;
            for (int j = k; j <= nn; ++j) {  // transformation des lignes k..k+2
              p = H[k][j] + q * H[k + 1][j];
              if (k != nn - 1) {
                p += r * H[k + 2][j];
                H[k + 2][j] -= p * z;
              }
              H[k + 1][j] -= p * y;
              H[k][j] -= p * x;
            }
            const int mmin = (nn < k + 3) ? nn : k + 3;
            for (int i = l; i <= mmin; ++i) {  // transformation des colonnes k..k+2
              p = x * H[i][k] + y * H[i][k + 1];
              if (k != nn - 1) {
                p += z * H[i][k + 2];
                H[i][k + 2] -= p * r;
              }
              H[i][k + 1] -= p * q;
              H[i][k] -= p;
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return true;
}

}  // namespace detail

/// Extremes des PARTIES REELLES du spectre d'un petit bloc dense @p A, plus grand |Im| rencontre
/// et indicateur de convergence (cf. l'en-tete du fichier pour le contrat complet : repli de
/// Gershgorin sur non-convergence, max_im comme detecteur de perte d'hyperbolicite).
/// @p max_iter_per_eig : cap d'iterations QR par bloc actif (defaut 100). L'heuristique EISPACK
/// historique (30) ne suffit pas sur les blocs compagnons quasi-degeneres (valeurs propres
/// quasi-doubles) ou la deflation rampe : un tel bloc 5x5 demande ~42 iterations, sous 30 il
/// repliait en silence (vitesse d'onde sur-estimee ~9x). 100 laisse plus du double de marge ; le
/// surcout n'est paye QUE par les blocs pathologiques (les cas sains convergent en quelques
/// iterations). 0 force le repli DES QU'UN bloc actif >= 3 existe (utile pour tester le contrat de
/// l'appelant) ; une matrice qui se deflate entierement en blocs 1x1 / 2x2 (quasi-triangulaire)
/// n'itere jamais et converge meme a cap 0.
/// @p fallback : si non nul, recoit true quand le repli de Gershgorin s'est declenche (spectre NON
/// calcule), false sinon. Defaut nullptr -> comportement inchange pour tout appelant existant ;
/// miroir de !EigBounds::converged, pour qui ne veut que le drapeau (ex. OR sur plusieurs blocs).
template <int N>
ADC_HD inline EigBounds real_eig_minmax(const Real (&A)[N][N], int max_iter_per_eig = 100,
                                        bool* fallback = nullptr) {
  static_assert(N >= 1, "real_eig_minmax : N >= 1");
  static_assert(N <= 16, "real_eig_minmax : bloc limite a 16x16 (tampon pile O(N^2) par thread "
                         "device, ~2 Ko ; au-dela, un solveur dense avec allocation est plus "
                         "indique que ce chemin)");
  EigBounds b{Real(0), Real(0), Real(0), true};
  if constexpr (N == 1) {
    b.lmin = b.lmax = A[0][0];
  } else if constexpr (N == 2) {  // forme fermee : trace / determinant
    const Real tr2 = Real(0.5) * (A[0][0] + A[1][1]);
    const Real disc = Real(0.25) * (A[0][0] - A[1][1]) * (A[0][0] - A[1][1]) + A[0][1] * A[1][0];
    if (disc >= Real(0)) {
      const Real z = std::sqrt(disc);
      b.lmin = tr2 - z;
      b.lmax = tr2 + z;
    } else {
      b.lmin = b.lmax = tr2;
      b.max_im = std::sqrt(-disc);
    }
  } else {
    Real H[N][N];  // copie de travail (A n'est pas modifiee)
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) H[i][j] = A[i][j];
    detail::hessenberg_reduce(H);
    if (!detail::hqr_minmax(H, b.lmin, b.lmax, b.max_im, max_iter_per_eig)) {
      // non-convergence : encadrement externe de Gershgorin sur la matrice D'ORIGINE (la copie de
      // travail est dans un etat intermediaire) -- borne sure, pas le spectre. max_im force a 0
      // par CONVENTION (rien n'a ete calcule), jamais a interpreter comme un spectre reel.
      detail::gershgorin_bounds(A, b.lmin, b.lmax);
      b.max_im = Real(0);
      b.converged = false;
    }
  }
  if (fallback) *fallback = !b.converged;
  return b;
}

}  // namespace adc
