#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/poisson_fft.hpp>  // fft1d (DFT/FFT 1D reutilisee verbatim)
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Solveur de Poisson POLAIRE direct sur un anneau (PolarGeometry), Phase 2a.
///
/// Resout l'equation de Poisson polaire sur le maillage ANNULAIRE (r, theta) du chantier
/// "grille polaire diocotron" (Phase 1, PolarGeometry) :
///   (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi = f
/// avec theta PERIODIQUE (l'anneau couvre [0, 2pi)) et une BC PHYSIQUE en r_min / r_max
/// (Dirichlet ou Neumann homogene). L'anneau EXCLUT r = 0 (r_min > 0) : AUCUNE singularite
/// de coordonnee, contrairement au disque plein.
///
/// METHODE (FFT-en-theta + tridiagonale-en-r, robuste, directe -- PAS de multigrille).
/// Le scoping a signale que le V-cycle MG peut STAGNER sur l'operateur 1/r^2 en polaire. La
/// methode classique robuste : theta etant periodique a coefficient constant, une FFT en theta
/// DECOUPLE les modes azimutaux m (le stencil 1/r^2 d_theta^2 se diagonalise sous la DFT). Pour
/// chaque mode m on resout alors une EDO radiale 1D
///   (1/r) d_r(r d_r phi_m) - (lambda_theta(m)/r^2) phi_m = f_m
/// par une tridiagonale (Thomas) en r. C'est EXACT par mode, robuste (aucun probleme de
/// convergence iterative) et bon marche (O(ntheta log ntheta) en theta + O(nr) par mode).
///
/// La FFT 1D (fft1d / dft1d_direct) est REUTILISEE verbatim depuis poisson_fft.hpp (radix-2 sur
/// les puissances de 2, repli DFT directe O(n^2) sinon -> ntheta quelconque accepte).
///
/// VALEUR PROPRE AZIMUTALE SPECTRALE : la base de Fourier diagonalise EXACTEMENT d_theta^2. Le mode
/// d'indice DFT m (m in [0, ntheta)) correspond au nombre d'onde signe k(m) = m si m <= ntheta/2,
/// sinon m - ntheta (repliement) ; d_theta^2 e^{i k theta} = -k^2 e^{i k theta}, donc la valeur propre
/// est -k(m)^2 (et NON le stencil 2 points (2cos-2)/dtheta^2, qui n'est qu'une approximation O(dtheta^2)
/// de -k^2). Employer -k(m)^2 rend la direction theta SPECTRALE (exacte pour une donnee a bande
/// limitee, comme le diocotron : un petit nombre de modes azimutaux). Le terme azimutal en cellule
/// (i, m) est donc (-k(m)^2 / r_i^2) phi_hat(i, m), DIAGONAL en m (1/r_i^2 local a la ligne i).
///
/// DISCRETISATION RADIALE (volumes finis conservatifs, ordre 2, comme assemble_rhs_polar) :
///   (1/r_i) [ r_{i+1/2}(phi_{i+1}-phi_i)/dr - r_{i-1/2}(phi_i - phi_{i-1})/dr ] / dr
/// soit, par mode m, le systeme tridiagonal (en phi_hat(., m)) :
///   sous-diag   a_i = (1/r_i) r_{i-1/2} / dr^2
///   sur-diag    c_i = (1/r_i) r_{i+1/2} / dr^2
///   diag        b_i = -(a_i + c_i) - k(m)^2 / r_i^2
/// r_i = r_cell(i), r_{i+/-1/2} = r_face(i+1)/r_face(i) (tous > 0 sur l'anneau).
///
/// CONDITIONS AUX LIMITES EN r (Dirichlet ou Neumann homogene, par BCRec.xlo / .xhi) :
///   - Dirichlet (valeur v a la face) : ghost de reflexion phi_{-1} = 2 v - phi_0. Le coefficient
///     a_0 plie alors dans la diagonale (b_0 -= a_0) et 2 a_0 v passe au second membre. Idem en
///     r_max avec c_{nr-1}, v = xhi_val.
///   - Neumann homogene (Foextrap, gradient nul a la face) : ghost phi_{-1} = phi_0 -> a_0 plie en
///     b_0 += a_0 (le flux a la paroi est nul). Idem c_{nr-1} en r_max.
/// theta reste PERIODIQUE (gere par la FFT, aucun ghost azimutal).
///
/// MODE m = 0 + DEUX bords Neumann : l'operateur radial pur a un noyau (la constante), donc la
/// tridiagonale est SINGULIERE. On fixe la jauge en epinglant phi_hat(0, 0) = 0 (phi de moyenne
/// radiale nulle pour le mode constant), comme le solveur FFT epingle le mode k = 0. Avec au moins
/// un bord Dirichlet l'operateur est inversible pour tout m (pas de jauge a fixer).
///
/// CONTRAT EllipticSolver (rhs()/phi()/solve()/residual()/geom()) : MODELE adapte au polaire. geom()
/// rend la PolarGeometry (et non Geometry) -- ce solveur n'est PAS encore branche dans System.step
/// (Phase 2b) ; il compose seulement comme un solveur elliptique polaire autonome. ADDITIF : aucun
/// chemin cartesien (geometric_mg, PoissonFFTSolver) n'est touche.
///
/// PORTEE : mono-rang, boite unique couvrant l'anneau (comme PoissonFFTSolver). La FFT-en-theta +
/// tridiag-en-r exige la ligne theta complete ET la colonne radiale complete sur un rang ; le
/// distribue imposerait une transposee parallele (hors scope Phase 2a). Garde-fou DUR (actif en
/// Release) si n_ranks() > 1 ou ba.size() != 1, leve sur TOUS les rangs (pas d'interblocage) ;
/// solve()/residual() sont local_size()==0-safe.

namespace adc {

// Contrat des solveurs elliptiques POLAIRES : meme forme que EllipticSolver (elliptic_solver.hpp)
// mais geom() rend une PolarGeometry (anneau (r, theta)) au lieu d'une Geometry cartesienne. Pendant
// polaire du concept cartesien : un futur coupleur polaire (Phase 2b) dependrait de ce CONCEPT, pas
// de PolarPoissonSolver en dur, exactement comme Coupler depend de EllipticSolver. Sibling NON
// intrusif : aucune modification d'elliptic_solver.hpp ni du chemin cartesien.
template <class S>
concept PolarEllipticSolver = requires(S s) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const PolarGeometry&>;
};

class PolarPoissonSolver {
 public:
  /// @param geom anneau (r, theta), domain.nx() = nr cellules radiales, ny() = ntheta azimutales.
  /// @param ba   BoxArray a UNE box couvrant tout l'anneau (mono-rang).
  /// @param bc   BC radiale : xlo/xhi (Dirichlet -> xlo_val/xhi_val ; Foextrap -> Neumann homogene).
  ///             theta (ylo/yhi) est traite comme PERIODIQUE quelle que soit la valeur (FFT).
  PolarPoissonSolver(const PolarGeometry& geom, const BoxArray& ba, const BCRec& bc = BCRec{})
      : geom_(geom),
        bc_(bc),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 0),
        rhs_(ba, dm_, 1, 0) {
    if (n_ranks() != 1)
      throw std::runtime_error(
          "PolarPoissonSolver : non supporte en MPI (n_ranks>1) ; la FFT-en-theta + tridiag-en-r "
          "exige ligne theta + colonne radiale completes sur un rang (transposee parallele = hors "
          "scope Phase 2a)");
    if (ba.size() != 1)
      throw std::runtime_error(
          "PolarPoissonSolver : boite unique requise (ba.size()==1) couvrant tout l'anneau");
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const PolarGeometry& geom() const { return geom_; }

  /// Resout (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi = rhs en place, par FFT-en-theta puis une
  /// tridiagonale (Thomas) par mode azimutal. phi() contient la solution apres l'appel.
  void solve() {
    if (phi_.local_size() == 0) return;  // rang sans box locale (MPI) : rien a faire (cf. garde-fou)
    // COHERENCE HOTE/DEVICE : solve() est un algorithme HOTE (std::vector / std::complex / fft1d /
    // Thomas) qui lit le RHS via des pointeurs hote (f = rhs_.fab(0).const_array() ci-dessous). Le RHS
    // a pu etre rempli par un kernel device (for_each_cell) reste eventuellement en vol. On rend la
    // residence hote du RHS valide AVANT toute lecture hote, exactement comme les autres lecteurs
    // hote du repo (cf. le seam sync_host()/device_fence() de for_each.hpp / MultiFab). Sous Kokkos
    // Cuda = un device_fence() cible (sans quoi : donnee perimee -> cudaErrorIllegalInstruction). Sous
    // Kokkos Serial/OpenMP (execution hote, memoire unifiee) = fence sans effet observable (aucun kernel
    // device en vol a drainer).
    rhs_.sync_host();
    const int nr = geom_.domain.nx();
    const int nth = geom_.domain.ny();
    const Real dr = geom_.dr();  // theta est SPECTRAL (valeur propre -k^2) : dtheta n'intervient pas
    const Box2D v = rhs_.box(0);  // cellules valides : i in [v.lo[0]..], j in [v.lo[1]..]
    const ConstArray4 f = rhs_.fab(0).const_array();
    Array4 p = phi_.fab(0).array();

    // --- 1) FFT en theta, ligne radiale par ligne radiale : f(i, .) -> f_hat(i, m) ---
    // On stocke par ligne radiale i un vecteur de ntheta complexes (le spectre azimutal de la ligne).
    std::vector<std::vector<cplx>> fhat(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      std::vector<cplx>& row = fhat[static_cast<std::size_t>(i)];
      row.resize(static_cast<std::size_t>(nth));
      for (int j = 0; j < nth; ++j)
        row[static_cast<std::size_t>(j)] = cplx(f(v.lo[0] + i, v.lo[1] + j), 0.0);
      fft1d(row.data(), nth, /*inv=*/false);
    }

    // --- 2) Coefficients radiaux INDEPENDANTS du mode (geometrie pure) ---
    // a_i (sous-diag), c_i (sur-diag), et la part radiale de la diagonale d_rad_i = -(a_i + c_i).
    std::vector<Real> a(static_cast<std::size_t>(nr)), c(static_cast<std::size_t>(nr)),
        d_rad(static_cast<std::size_t>(nr)), inv_r2(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      const Real ri = geom_.r_cell(i);        // r au centre de la cellule i (> 0)
      const Real rm = geom_.r_face(i);         // r_{i-1/2} (face basse)
      const Real rp = geom_.r_face(i + 1);     // r_{i+1/2} (face haute)
      const Real inv_r_dr2 = Real(1) / (ri * dr * dr);
      a[static_cast<std::size_t>(i)] = rm * inv_r_dr2;
      c[static_cast<std::size_t>(i)] = rp * inv_r_dr2;
      d_rad[static_cast<std::size_t>(i)] = -(a[static_cast<std::size_t>(i)] + c[static_cast<std::size_t>(i)]);
      inv_r2[static_cast<std::size_t>(i)] = Real(1) / (ri * ri);
    }

    // BC radiale : Dirichlet (face) -> reflexion ghost = 2 v - interne ; Foextrap -> Neumann homogene
    // (ghost = interne). On replie le coefficient de bord dans la diagonale et, pour Dirichlet, on
    // injecte 2 a/c * valeur au second membre (mode m = 0 SEULEMENT : la BC est reelle et constante
    // en theta, donc ne contribue qu'au mode azimutal moyen).
    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    const Real vlo = bc_.xlo_val, vhi = bc_.xhi_val;
    const bool neumann_both = !dir_lo && !dir_hi;  // jauge a fixer pour le mode m = 0

    // --- 3) Une tridiagonale (Thomas) par mode m, sur le vecteur complexe phi_hat(., m) ---
    // Le systeme est REEL (a, b, c reels) ; on resout les parties reelle et imaginaire avec la MEME
    // matrice (Thomas sur des complexes : a/b/c reels, second membre complexe). phat[i][m] recoit le
    // spectre radial du mode m. On boucle m a l'exterieur (matrice fixe pour ce m), i a l'interieur.
    std::vector<std::vector<cplx>> phat(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) phat[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));

    std::vector<cplx> rhs_m(static_cast<std::size_t>(nr)), sol_m(static_cast<std::size_t>(nr));
    std::vector<Real> bdiag(static_cast<std::size_t>(nr));
    for (int m = 0; m < nth; ++m) {
      // Nombre d'onde SIGNE (repliement DFT) : k = m pour m <= nth/2, sinon m - nth. Valeur propre
      // SPECTRALE de d_theta^2 = -k^2 (exacte, pas le stencil 2 points). dtheta n'intervient pas ici.
      const int kw = (m <= nth / 2) ? m : m - nth;
      const Real lam_th = -static_cast<Real>(kw) * static_cast<Real>(kw);
      // Diagonale complete du mode m : radiale + azimutale (lam_th/r_i^2 = -k^2/r_i^2).
      for (int i = 0; i < nr; ++i)
        bdiag[static_cast<std::size_t>(i)] =
            d_rad[static_cast<std::size_t>(i)] + lam_th * inv_r2[static_cast<std::size_t>(i)];
      for (int i = 0; i < nr; ++i) rhs_m[static_cast<std::size_t>(i)] = fhat[static_cast<std::size_t>(i)][static_cast<std::size_t>(m)];

      // Repli des BC radiales dans la diagonale / le second membre (du mode m).
      // r_min (i = 0) : ghost phi_{-1}. Dirichlet -> phi_{-1} = 2 vlo - phi_0 : b_0 -= a_0,
      // rhs_0 -= 2 a_0 vlo (mode 0). Neumann -> phi_{-1} = phi_0 : b_0 += a_0.
      if (dir_lo) {
        bdiag[0] -= a[0];
        if (m == 0) rhs_m[0] -= cplx(Real(2) * a[0] * vlo * static_cast<Real>(nth), 0.0);
      } else {
        bdiag[0] += a[0];
      }
      // r_max (i = nr-1) : ghost phi_{nr}. Dirichlet -> phi_{nr} = 2 vhi - phi_{nr-1}, Neumann -> = phi_{nr-1}.
      const std::size_t last = static_cast<std::size_t>(nr - 1);
      if (dir_hi) {
        bdiag[last] -= c[last];
        if (m == 0) rhs_m[last] -= cplx(Real(2) * c[last] * vhi * static_cast<Real>(nth), 0.0);
      } else {
        bdiag[last] += c[last];
      }

      // Mode m = 0 avec DEUX bords Neumann : operateur radial singulier (noyau = constante). On epingle
      // phi_hat(0, 0) = 0 (jauge : moyenne radiale nulle du mode constant) en forcant la 1ere ligne a
      // l'identite (diag 1, sur-diag 0, rhs 0). Sans cela Thomas divise par un pivot nul.
      const bool pin0 = neumann_both && m == 0;

      thomas_solve(a, bdiag, c, rhs_m, sol_m, nr, pin0);
      for (int i = 0; i < nr; ++i) phat[static_cast<std::size_t>(i)][static_cast<std::size_t>(m)] = sol_m[static_cast<std::size_t>(i)];
    }

    // --- 4) FFT inverse en theta : phi_hat(i, m) -> phi(i, theta) (la partie reelle, l'imaginaire ~ arrondi) ---
    for (int i = 0; i < nr; ++i) {
      std::vector<cplx>& row = phat[static_cast<std::size_t>(i)];
      fft1d(row.data(), nth, /*inv=*/true);
      for (int j = 0; j < nth; ++j) p(v.lo[0] + i, v.lo[1] + j) = row[static_cast<std::size_t>(j)].real();
    }
  }

  /// Residu discret ||L_h phi - rhs|| (norme inf), reduit sur les rangs. L_h est l'operateur EXACT que
  /// solve() inverse : FFT en theta (mode m, valeur propre spectrale -k(m)^2/r^2) + stencil radial
  /// conservatif d'ordre 2, BC radiale repliee (Dirichlet : valeur constante = part m=0 ; Neumann :
  /// reflexion homogene). On l'evalue MODE PAR MODE dans l'espace de Fourier (la ou la diagonalisation
  /// theta est exacte), donc le residu reflete la SEULE qualite du solve (et pas l'ecart spectral vs
  /// differences finies d'un stencil theta physique). Pour ce solveur DIRECT, ~arrondi. Le mode m=0
  /// double-Neumann est EXCLU (sa ligne 0 est epinglee = jauge : equation identite triviale).
  Real residual() {
    if (phi_.local_size() == 0) return static_cast<Real>(all_reduce_max(0.0));
    // Comme solve() : evaluation HOTE (FFT + stencil sur des pointeurs hote). On rend la residence hote
    // du RHS valide avant lecture (kernel device eventuellement en vol). phi_ est ecrit cote hote par
    // solve(), mais on fence aussi par symetrie/robustesse si residual() est appele independamment. Sous
    // Kokkos Serial/OpenMP (execution hote) = fence sans effet observable ; sous Kokkos Cuda = device_fence() cible.
    rhs_.sync_host();
    phi_.sync_host();
    const int nr = geom_.domain.nx();
    const int nth = geom_.domain.ny();
    const Real dr = geom_.dr();
    const Box2D v = rhs_.box(0);
    const ConstArray4 p = phi_.fab(0).const_array();
    const ConstArray4 f = rhs_.fab(0).const_array();

    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    const bool neumann_both = !dir_lo && !dir_hi;

    // Coefficients radiaux (memes que solve()).
    std::vector<Real> a(static_cast<std::size_t>(nr)), c(static_cast<std::size_t>(nr)),
        d_rad(static_cast<std::size_t>(nr)), inv_r2(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      const Real ri = geom_.r_cell(i);
      const Real inv_r_dr2 = Real(1) / (ri * dr * dr);
      a[static_cast<std::size_t>(i)] = geom_.r_face(i) * inv_r_dr2;
      c[static_cast<std::size_t>(i)] = geom_.r_face(i + 1) * inv_r_dr2;
      d_rad[static_cast<std::size_t>(i)] = -(a[static_cast<std::size_t>(i)] + c[static_cast<std::size_t>(i)]);
      inv_r2[static_cast<std::size_t>(i)] = Real(1) / (ri * ri);
    }

    // FFT de phi et de f, ligne radiale par ligne radiale.
    std::vector<std::vector<cplx>> ph(static_cast<std::size_t>(nr)), fh(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      ph[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));
      fh[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(nth));
      for (int j = 0; j < nth; ++j) {
        ph[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cplx(p(v.lo[0] + i, v.lo[1] + j), 0.0);
        fh[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cplx(f(v.lo[0] + i, v.lo[1] + j), 0.0);
      }
      fft1d(ph[static_cast<std::size_t>(i)].data(), nth, false);
      fft1d(fh[static_cast<std::size_t>(i)].data(), nth, false);
    }

    Real rmax = 0;
    for (int m = 0; m < nth; ++m) {
      const int kw = (m <= nth / 2) ? m : m - nth;
      const Real lam_th = -static_cast<Real>(kw) * static_cast<Real>(kw);
      const bool pin0 = neumann_both && m == 0;  // mode jauge : ligne 0 epinglee (exclue du residu)
      for (int i = 0; i < nr; ++i) {
        if (pin0 && i == 0) continue;  // ligne identite epinglee : pas une equation physique
        const std::size_t I = static_cast<std::size_t>(i);
        Real bdiag = d_rad[I] + lam_th * inv_r2[I];
        cplx lhs(0.0, 0.0);
        cplx rm = fh[I][static_cast<std::size_t>(m)];
        // Bord bas (i==0) : repli de a_0 (Dirichlet : b-=a, rhs-=2 a vlo*nth en m=0 ; Neumann : b+=a).
        if (i == 0) {
          if (dir_lo) {
            bdiag -= a[0];
            if (m == 0) rm -= cplx(Real(2) * a[0] * bc_.xlo_val * static_cast<Real>(nth), 0.0);
          } else {
            bdiag += a[0];
          }
        } else {
          lhs += cplx(a[I], 0.0) * ph[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(m)];
        }
        // Bord haut (i==nr-1) : repli de c.
        if (i == nr - 1) {
          if (dir_hi) {
            bdiag -= c[I];
            if (m == 0) rm -= cplx(Real(2) * c[I] * bc_.xhi_val * static_cast<Real>(nth), 0.0);
          } else {
            bdiag += c[I];
          }
        } else {
          lhs += cplx(c[I], 0.0) * ph[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(m)];
        }
        lhs += cplx(bdiag, 0.0) * ph[I][static_cast<std::size_t>(m)];
        // Residu de ce mode (normalise par nth : la DFT non normalisee echelonne f_hat par nth ; on
        // compare a la MEME echelle, donc le ratio est independant de la convention).
        const cplx r_m = (lhs - rm) / static_cast<double>(nth);
        rmax = std::max(rmax, static_cast<Real>(std::abs(r_m)));
      }
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(rmax)));
  }

 private:
  // Algorithme de Thomas (elimination tridiagonale) sur des matrices a coefficients a/b/c REELS et un
  // second membre COMPLEXE. a[i] = sous-diag (couple i-1), b[i] = diag, c[i] = sur-diag (couple i+1).
  // a[0] et c[n-1] ne sont PAS utilises (bords). @p pin0 : si vrai, epingle x[0] = 0 (jauge mode 0,
  // double Neumann) en remplacant la 1ere ligne par l'identite (b_0 = 1, c_0 = 0, rhs_0 = 0). On
  // travaille sur des copies LOCALES de la diagonale / sur-diagonale / rhs pour ne pas alterer les
  // tableaux de l'appelant. Pas de pivotage : la matrice est diagonale-dominante (terme azimutal <= 0,
  // BC repliees) -> Thomas stable sans permutation.
  void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b, const std::vector<Real>& c,
                    const std::vector<cplx>& rhs, std::vector<cplx>& x, int n, bool pin0) const {
    const std::size_t N = static_cast<std::size_t>(n);
    bloc_.assign(N, 0.0);   // diagonale de travail
    cloc_.assign(N, 0.0);   // sur-diagonale de travail
    rloc_.assign(N, cplx(0.0, 0.0));  // second membre de travail
    for (std::size_t i = 0; i < N; ++i) {
      bloc_[i] = b[i];
      cloc_[i] = c[i];
      rloc_[i] = rhs[i];
    }
    if (pin0) {  // ligne 0 -> identite : x[0] = 0, decouplee du reste
      bloc_[0] = Real(1);
      cloc_[0] = Real(0);
      rloc_[0] = cplx(0.0, 0.0);
    }
    cgamma_.assign(N, cplx(0.0, 0.0));
    Real beta = bloc_[0];
    x[0] = rloc_[0] / cplx(beta, 0.0);
    for (std::size_t i = 1; i < N; ++i) {
      cgamma_[i] = cplx(cloc_[i - 1] / beta, 0.0);
      beta = bloc_[i] - a[i] * cloc_[i - 1] / beta;
      x[i] = (rloc_[i] - cplx(a[i], 0.0) * x[i - 1]) / cplx(beta, 0.0);
    }
    for (int i = n - 2; i >= 0; --i)
      x[static_cast<std::size_t>(i)] -= cgamma_[static_cast<std::size_t>(i + 1)] * x[static_cast<std::size_t>(i + 1)];
  }

  PolarGeometry geom_;
  BCRec bc_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_;
  // Buffers reutilises par Thomas (evitent les allocations par mode). mutable : thomas_solve est const.
  mutable std::vector<cplx> cgamma_, rloc_;
  mutable std::vector<Real> bloc_, cloc_;
};

static_assert(PolarEllipticSolver<PolarPoissonSolver>,
              "PolarPoissonSolver doit modeler PolarEllipticSolver");

}  // namespace adc
