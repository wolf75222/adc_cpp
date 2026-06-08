#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Operateur elliptique POLAIRE a coefficient TENSORIEL anisotrope (termes croises), iteratif.
///        Voie A etape 2a (brique foundational vers le Schur polaire).
///
/// CONTEXTE. Pour faire le systeme Euler-Poisson COMPLET RAIDE en geometrie POLAIRE (figure 2D
/// diocotron a omega_c eleve), il faut l'etage Schur condense (CondensedSchurSourceStepper, niveau 4
/// du splitting Hoffart et al., arXiv:2510.11808) sur le polaire. Le verrou est l'OPERATEUR elliptique :
/// le Schur condense un operateur TENSORIEL PLEIN A = I + c rho B^{-1} ou B^{-1} est la rotation de
/// Lorentz, qui injecte des termes CROISES a_rt / a_tr (et un coefficient dependant de theta des que
/// rho ou B_z varient en theta). Le PolarPoissonSolver DIRECT existant (FFT-en-theta + tridiag-en-r,
/// polar_poisson_solver.hpp) diagonalise theta par la FFT : il EXIGE un operateur a coefficient CONSTANT
/// en theta et SANS couplage croise. Il est donc structurellement INCOMPATIBLE avec l'operateur Schur.
/// Cet operateur-ci est le pendant ITERATIF (matrice-libre) qui accepte le tenseur plein.
///
/// EQUATION. Sur un anneau (r, theta), r in [r_min, r_max] > 0 (AUCUNE singularite r=0), theta
/// PERIODIQUE, on resout l'operateur sous forme DIVERGENCE avec metrique polaire et tenseur A dans le
/// repere physique ORTHONORME (e_r, e_theta) :
///   L(phi) = div(A grad phi),   grad phi = (d_r phi, (1/r) d_theta phi),
///   div F  = (1/r) d_r(r F_r) + (1/r) d_theta(F_theta),
///   A = [[a_rr, a_rt], [a_tr, a_tt]]  (eventuellement NON symetrique : a_rt != a_tr).
/// Le flux est F = A grad phi :
///   F_r     = a_rr d_r phi + a_rt (1/r) d_theta phi
///   F_theta = a_tr d_r phi + a_tt (1/r) d_theta phi
/// Cas ISOTROPE a_rr=a_tt=1, a_rt=a_tr=0 :
///   L = (1/r) d_r(r d_r phi) + (1/r^2) d_theta^2 phi   (Laplacien polaire scalaire, cf. PolarPoissonSolver).
///
/// DISCRETISATION VOLUMES FINIS CONSERVATIVE (ordre 2, comme assemble_rhs_polar / PolarPoissonSolver).
/// Cellule (i, j) de poids volume r_i dr dtheta. On integre div(F) sur la cellule et on divise par le
/// poids, ce qui donne sur chaque face le flux porte par sa LONGUEUR FV :
///   - faces RADIALES i+-1/2 (rayon r_face, poids de face r_face dtheta) :
///       terme radial DIAGONAL : (1/r_i) [ r_{i+1/2} a_rr^{i+1/2} (phi_{i+1,j}-phi_{i,j})/dr
///                                        - r_{i-1/2} a_rr^{i-1/2} (phi_{i,j}-phi_{i-1,j})/dr ] / dr ;
///       terme radial CROISE a_rt (1/r) d_theta phi a la face i+-1/2 (d_theta moyenne sur 4 coins,
///       1/r_{face} local a la face) ;
///   - faces AZIMUTALES j+-1/2 (rayon r_i, poids de face dr) :
///       terme azimutal DIAGONAL : a_tt^{j+-1/2} (phi_{i,j+1}-2 phi+phi_{i,j-1}) / (r_i^2 dtheta^2) ;
///       terme azimutal CROISE a_tr d_r phi a la face j+-1/2 (d_r moyenne sur 4 coins).
/// Coefficient de FACE = moyenne ARITHMETIQUE des deux centres adjacents (ordre 2 ; pour le terme
/// diagonal radial/azimutal d'un milieu lisse l'arithmetique suffit ; le terme croise n'est pas un
/// flux normal -> arithmetique aussi, cf. poisson_operator.hpp cartesien). Les coefficients a_rr/a_tt/
/// a_rt/a_tr sont AU CENTRE des cellules (1 composante chacun, ghosts remplis par l'appelant).
///
/// VALIDITE de la metrique : les faces radiales utilisent r_face(i) > 0 (anneau, r_min > 0) ; aucune
/// division par r=0. La diagonale du stencil est SOMMEE des poids de face (pour le preconditionneur
/// Jacobi) ; en isotrope elle redonne EXACTEMENT le stencil polaire scalaire.
///
/// CONDITIONS AUX LIMITES. theta PERIODIQUE (ghosts periodiques, fill_ghosts avec ylo/yhi=Periodic).
/// r_min/r_max : BC PHYSIQUE Dirichlet (valeur a la face, ghost de reflexion 2 v - interne) ou Neumann
/// homogene (Foextrap, ghost = interne). C'est exactement le fill_ghosts cartesien applique en (r,
/// theta) (direction d'indice 0 = radiale, 1 = azimutale ; cf. convention PolarGeometry). La donnee
/// Dirichlet est repliee dans le residu (operateur AFFINE) pour le residu vrai r0, et LINEARISEE
/// (offset c_bc retranche) pour les matvec en boucle (directions de correction) : meme mecanique que
/// TensorKrylovSolver cartesien (krylov_solver.hpp).
///
/// OPERATEUR SINGULIER (pure Neumann radial + theta periodique, aucune reaction) : la CONSTANTE est dans
/// le noyau de L_int ; BiCGStab diverge sans traitement. On FIXE LA JAUGE par PROJECTION sur le
/// sous-espace de MOYENNE FV NULLE (moyenne ponderee r dr dtheta retiree du residu initial, de la
/// solution, et de chaque direction de correction preconditionnee). C'est le pendant ITERATIF du pinning
/// de mode 0 du PolarPoissonSolver direct, sans perturber le stencil. Detecte automatiquement (les deux
/// bords radiaux non-Dirichlet) ; au moins UN bord Dirichlet => operateur inversible, AUCUN pinning,
/// chemin Dirichlet bit-identique. La solution est alors definie modulo une constante.
///
/// SOLVEUR : Krylov BiCGStab MATRICE-LIBRE (gere le NON symetrique du terme croise), PRECONDITIONNE par
/// un preconditionneur SIMPLE (PAS de V-cycle MG : le scoping, avertissement polar_poisson_solver.hpp,
/// signale que le V-cycle MG STAGNE sur l'anisotropie 1/r^2 polaire ; on s'en tient a Krylov + precond
/// simple comme demande). Deux preconditionneurs (cf. enum PolarPrecond) :
///   - Jacobi (diagonal) : le plus simple, mais le nombre d'iterations CROIT comme 1/h^2 (Laplacien mal
///     conditionne). A grille fine il PLAFONNE sans converger (cf. test, bloc E). Sanity check / repli.
///   - RadialLine (DEFAUT) : inverse EXACTEMENT la tridiagonale RADIALE du bloc diagonal par ligne theta
///     (Thomas), diagonale azimutale lumpee. Attaque le couplage radial fort + l'anisotropie 1/r^2 ;
///     nombre d'iterations FAIBLE et a croissance MODEREE (mesure : isotrope ~ x2 par doublement de
///     grille, tenseur ~ x2.4). Reste "simple" (aucune hierarchie MG, aucune grille grossiere).
/// LIMITE HONNETE : le terme CROISE et le couplage AZIMUTAL ne sont PAS dans le preconditionneur (qui
/// n'inverse que le radial) -> le nombre d'iterations du cas tenseur croit plus vite que l'isotrope.
/// C'est acceptable pour la brique foundational (convergence SOLIDE, aucune stagnation observee) ; un
/// preconditionneur azimutal (ligne theta) ou un MG-line robuste serait un raffinement ULTERIEUR.
/// BiCGStab a une empreinte memoire FIXE (r, rhat, p, v, s, t, + les preconditionnes phat/shat) et ne
/// stocke pas de base de Krylov croissante.
///
/// PORTEE : MULTI-RANG MPI par decoupage AZIMUTAL (theta seul). Les produits scalaires (dot/norme L2)
/// sont collectifs (all_reduce via mf_arith::dot) et local_size()==0-safe ; la matvec passe par
/// fill_ghosts (echange de halos MPI + CL physique) ; la projection de jauge project_mean est all_reduit
/// sur tous les rangs (numerateur + denominateur). Le preconditionneur RadialLine (Thomas en r) exige
/// qu'une LIGNE RADIALE entiere (colonne i=0..nr-1 a theta fixe) tienne dans UNE box (sinon le sweep
/// sequentiel en r franchirait une frontiere de box/rang) : on impose donc que CHAQUE box couvre la
/// PLAGE RADIALE COMPLETE du domaine (decoupage en theta uniquement). C'est le compromis le plus SIMPLE
/// et CORRECT (mirroir de l'exigence "colonne complete" du solveur FFT direct) ; un decoupage qui coupe
/// r est REFUSE par un garde-fou explicite (au lieu d'un resultat faux silencieux). Le decoupage theta
/// est legitime : theta est la direction PERIODIQUE et le mode azimutal porteur du diocotron ; couper r
/// (peu de cellules, fort couplage radial) serait de toute facon contre-productif. Si un decoupage qui
/// coupe r est requis, le repli Jacobi (PolarPrecond::Jacobi, par cellule, MPI-trivial) reste valide
/// sans contrainte de layout. MONO-RANG / BOITE UNIQUE : chemin BIT-IDENTIQUE (all_reduce = identite en
/// serie ; la boucle local_size() = la boucle fab(0) quand il n'y a qu'une box couvrant tout l'anneau).
///
/// ADDITIF : aucun chemin existant n'est touche. Le PolarPoissonSolver scalaire DIRECT reste INTOUCHE
/// (chemin separe) ; le Schur CARTESIEN reste BIT-IDENTIQUE. Ce header est OPT-IN (etape 2b = wiring
/// d'un SchurReconstructKernel polaire + d'un coupleur, hors scope ici).

namespace adc {

namespace detail {

/// Contribution de FACE RADIALE au stencil diagonal + croise, en (i, j). Renvoie L_int local SANS le
/// terme azimutal (calcule a part). Foncteur libre device-clean (ADC_HD). Coefficients de face =
/// moyenne arithmetique ; metrique r_face/r_i ; terme croise a_rt (1/r_face) (d_theta phi)_face.
/// Termes croises absents (hrt=false) -> seul le terme radial diagonal contribue, bit-identique au
/// stencil polaire scalaire avec a_rr=1.
ADC_HD inline Real polar_radial_div(const ConstArray4& p, const ConstArray4& arr,
                                    bool hrt, const ConstArray4& art, int i, int j, Real ri,
                                    Real rfm, Real rfp, Real idr, Real idth) {
  // Coefficients a_rr de face (moyenne arithmetique des centres adjacents).
  const Real arr_p = Real(0.5) * (arr(i, j) + arr(i + 1, j));
  const Real arr_m = Real(0.5) * (arr(i, j) + arr(i - 1, j));
  // Terme radial DIAGONAL : (1/r_i) [ r_{i+1/2} a_rr_p (phi_{i+1}-phi_i)/dr - r_{i-1/2} a_rr_m (phi_i-phi_{i-1})/dr ] / dr.
  Real out = (rfp * arr_p * (p(i + 1, j) - p(i, j)) - rfm * arr_m * (p(i, j) - p(i - 1, j))) *
             (idr * idr / ri);
  if (hrt) {  // Terme radial CROISE a_rt (1/r_face) d_theta phi a la face i+-1/2 (d_theta moyenne 4 coins).
    const Real art_p = Real(0.5) * (art(i, j) + art(i + 1, j));
    const Real art_m = Real(0.5) * (art(i, j) + art(i - 1, j));
    // (d_theta phi)_face moyenne sur les 4 coins de la face : 1/(r_face) car grad_theta = (1/r) d_theta phi.
    const Real dth_p = (p(i, j + 1) + p(i + 1, j + 1) - p(i, j - 1) - p(i + 1, j - 1)) * (Real(0.25) * idth);
    const Real dth_m = (p(i - 1, j + 1) + p(i, j + 1) - p(i - 1, j - 1) - p(i, j - 1)) * (Real(0.25) * idth);
    // (1/r_i)(1/dr) [ r_{i+1/2} F_cross_p - r_{i-1/2} F_cross_m ], F_cross = a_rt (1/r_face) d_theta phi.
    // r_face * (1/r_face) = 1 : la metrique r_face annule le 1/r_face du gradient azimutal -> facteur 1.
    out += (art_p * dth_p - art_m * dth_m) * (idr / ri);
  }
  return out;
}

/// Contribution de FACE AZIMUTALE au stencil diagonal + croise, en (i, j). Renvoie L_int local du
/// terme azimutal. Coefficients a_tt de face (arithmetique) ; metrique 1/(r_i^2) ; terme croise a_tr
/// d_r phi a la face j+-1/2 (d_r moyenne 4 coins).
ADC_HD inline Real polar_azimuthal_div(const ConstArray4& p, const ConstArray4& att,
                                       bool htr, const ConstArray4& atr, int i, int j, Real ri,
                                       Real idr, Real idth) {
  const Real inv_r2 = Real(1) / (ri * ri);
  const Real att_p = Real(0.5) * (att(i, j) + att(i, j + 1));
  const Real att_m = Real(0.5) * (att(i, j) + att(i, j - 1));
  // Terme azimutal DIAGONAL : a_tt^{j+-1/2} (phi_{j+1}-2 phi+phi_{j-1}) / (r_i^2 dtheta^2), forme flux.
  Real out = (att_p * (p(i, j + 1) - p(i, j)) - att_m * (p(i, j) - p(i, j - 1))) * (idth * idth * inv_r2);
  if (htr) {  // Terme azimutal CROISE a_tr d_r phi a la face j+-1/2 (d_r moyenne 4 coins).
    const Real atr_p = Real(0.5) * (atr(i, j) + atr(i, j + 1));
    const Real atr_m = Real(0.5) * (atr(i, j) + atr(i, j - 1));
    const Real dr_p = (p(i + 1, j) + p(i + 1, j + 1) - p(i - 1, j) - p(i - 1, j + 1)) * (Real(0.25) * idr);
    const Real dr_m = (p(i + 1, j - 1) + p(i + 1, j) - p(i - 1, j - 1) - p(i - 1, j)) * (Real(0.25) * idr);
    // (1/r_i)(1/dtheta) [ F_cross_p - F_cross_m ], F_cross = a_tr d_r phi. Metrique 1/r_i (poids face dr).
    out += (atr_p * dr_p - atr_m * dr_m) * (idth / ri);
  }
  return out;
}

/// Diagonale (coefficient de phi_{i,j}) du stencil POLAIRE diagonal (radial + azimutal), pour le
/// preconditionneur Jacobi. Termes croises EXCLUS de la diagonale (ils ne touchent pas phi_{i,j} :
/// les coins i+-1, j+-1 sont hors-diagonaux). Renvoie la valeur (NEGATIVE) du coefficient diagonal de
/// L_int (somme des -poids de face), comme le stencil scalaire (diag < 0).
ADC_HD inline Real polar_diag(const ConstArray4& arr, const ConstArray4& att, int i, int j, Real ri,
                              Real rfm, Real rfp, Real idr, Real idth) {
  const Real arr_p = Real(0.5) * (arr(i, j) + arr(i + 1, j));
  const Real arr_m = Real(0.5) * (arr(i, j) + arr(i - 1, j));
  const Real att_p = Real(0.5) * (att(i, j) + att(i, j + 1));
  const Real att_m = Real(0.5) * (att(i, j) + att(i, j - 1));
  const Real inv_r2 = Real(1) / (ri * ri);
  const Real rad = (rfp * arr_p + rfm * arr_m) * (idr * idr / ri);
  const Real azi = (att_p + att_m) * (idth * idth * inv_r2);
  return -(rad + azi);  // coefficient diagonal de L_int (les voisins ont des coefficients positifs)
}

/// L_int(phi) = div(A grad phi) en polaire (apply). Foncteur NOMME device-clean (recette #93 : kernel
/// premiere-instancie cross-TU -> pas de lambda etendue sous nvcc). arr/att toujours fournis (au moins
/// a_rr=a_tt=1) ; art/atr optionnels (termes croises). r_min/dr/dtheta passes en scalaires (PolarGeometry
/// accesseurs ADC_HD recalcules en kernel).
struct PolarApplyKernel {
  ConstArray4 p;
  Array4 L;
  ConstArray4 arr, att;
  bool hrt, htr;
  ConstArray4 art, atr;
  Real r_min, dr, idr, idth;
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;     // r_cell(i)
    const Real rfm = r_min + i * dr;                  // r_face(i)   = r_{i-1/2}
    const Real rfp = r_min + (i + 1) * dr;            // r_face(i+1) = r_{i+1/2}
    L(i, j) = polar_radial_div(p, arr, hrt, art, i, j, ri, rfm, rfp, idr, idth) +
              polar_azimuthal_div(p, att, htr, atr, i, j, ri, idr, idth);
  }
};

/// out = (f - L0 phi) / |diag| -- une iteration de Jacobi (relaxation point a point) sur le stencil
/// DIAGONAL polaire (termes croises exclus du splitting Jacobi : ils restent au second membre via le
/// residu). out est ECRIT. Foncteur NOMME device-clean. Sert au preconditionneur Jacobi : applique a un
/// residu z, rend M^{-1} z = diag^{-1} z (forme simple, un balayage). idiag = 1/diag stocke a part.
struct PolarJacobiApplyKernel {
  ConstArray4 z;
  Array4 out;
  ConstArray4 idiag;  // 1 / coefficient diagonal de L_int (negatif), precalcule
  ADC_HD void operator()(int i, int j) const { out(i, j) = z(i, j) * idiag(i, j); }
};

/// Copie composante 0 (dst <- src). Foncteur NOMME device-clean local (le header n'inclut pas
/// geometric_mg.hpp : on ne depend PAS du V-cycle MG, ecarte volontairement en polaire).
struct PolarCopyKernel {
  Array4 d;
  ConstArray4 s;
  ADC_HD void operator()(int i, int j) const { d(i, j) = s(i, j, 0); }
};

/// Calcule idiag = 1 / diag du stencil polaire diagonal (pour Jacobi). diag = polar_diag (< 0).
struct PolarInvDiagKernel {
  ConstArray4 arr, att;
  Array4 idiag;
  Real r_min, dr, idr, idth;
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real rfm = r_min + i * dr;
    const Real rfp = r_min + (i + 1) * dr;
    const Real d = polar_diag(arr, att, i, j, ri, rfm, rfp, idr, idth);
    idiag(i, j) = d != Real(0) ? Real(1) / d : Real(0);
  }
};

}  // namespace detail

/// Applique L_int(phi) = div(A grad phi) en polaire sur tout le MultiFab. Ghosts de @p phi supposes
/// remplis (theta periodique, r physique). a_rr/a_tt : coefficients diagonaux (1 composante, centres ;
/// nullptr -> coefficient 1 uniforme, isotrope). a_rt/a_tr : termes croises (nullptr -> absents).
inline void apply_polar_tensor(const MultiFab& phi, const PolarGeometry& geom, MultiFab& lap,
                               const MultiFab* a_rr, const MultiFab* a_tt, const MultiFab* a_rt,
                               const MultiFab* a_tr) {
  // CONTRAT : a_rr/a_tt requis (les coefficients diagonaux du stencil sont toujours lus). Un cas isotrope
  // fournit des champs CONSTANTS a 1 (PolarTensorKrylovSolver le fait via ses stores internes). a_rt/a_tr
  // optionnels (termes croises). On ne peut pas deref un nullptr en kernel -> garde-fou a l'entree.
  assert(a_rr && a_tt && "apply_polar_tensor : a_rr et a_tt requis (champs a 1 si isotrope)");
  const Real dr = geom.dr();
  const Real idr = Real(1) / dr;
  const Real idth = Real(1) / geom.dtheta();
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    const ConstArray4 arr = a_rr->fab(li).const_array();
    const ConstArray4 att = a_tt->fab(li).const_array();
    const bool hrt = a_rt != nullptr;
    const bool htr = a_tr != nullptr;
    const ConstArray4 art = hrt ? a_rt->fab(li).const_array() : ConstArray4{};
    const ConstArray4 atr = htr ? a_tr->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, detail::PolarApplyKernel{p, L, arr, att, hrt, htr, art, atr, geom.r_min, dr, idr, idth});
  }
}

/// Contrat des operateurs elliptiques POLAIRES iteratifs : meme forme que PolarEllipticSolver (cf.
/// polar_poisson_solver.hpp) + variante a tolerance solve(rel_tol, max_iters) (pendant polaire de
/// LinearSolver cartesien). On ne redefinit PAS PolarEllipticSolver (defini avec PolarPoissonSolver) :
/// on l'inclut indirectement via les memes membres. Type de retour de solve(tol,it) NON void.
template <class S>
concept PolarLinearSolver = requires(S s, Real tol, int it) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const PolarGeometry&>;
  s.solve(tol, it);
  requires !std::same_as<decltype(s.solve(tol, it)), void>;
};

/// Resultat d'un solve BiCGStab polaire : iterations, residu relatif, convergence. (Meme forme que
/// KrylovResult cartesien -- on reutilise le type si deja inclus ; sinon on declare le pendant local.)
struct PolarKrylovResult {
  int iters = 0;
  Real rel_residual = 0;
  bool converged = false;
};

/// Choix du PRECONDITIONNEUR SIMPLE de BiCGStab (PAS de V-cycle MG -- stagnation sur 1/r^2 polaire,
/// cf. entete). Deux options, toutes deux "simples" (sans hierarchie multigrille) :
///   - Jacobi : diagonal pur M^{-1} = diag^{-1}. Le plus simple, mais le nombre d'iterations CROIT
///     comme 1/h^2 (Laplacien mal conditionne) : utile en sanity check, peu performant a grille fine.
///   - RadialLine : inverse EXACTEMENT la tridiagonale RADIALE du bloc diagonal par ligne theta
///     (Thomas, comme le solveur direct), la diagonale du bloc azimutal etant lumpee. Attaque la
///     direction radiale FORTEMENT couplee (et l'anisotropie 1/r^2 via la diagonale lumpee), laissant
///     a Krylov le seul couplage azimutal residuel : nombre d'iterations QUASI INDEPENDANT de h.
///     Reste "simple" (pas de MG, pas de grille grossiere). DEFAUT.
enum class PolarPrecond { Jacobi, RadialLine };

/// Solveur de Krylov BiCGStab MATRICE-LIBRE pour l'operateur elliptique POLAIRE a tenseur PLEIN
/// L_int(phi) = div(A grad phi), A = [[a_rr, a_rt], [a_tr, a_tt]] eventuellement NON symetrique.
/// Preconditionneur SIMPLE (Jacobi diagonal ou RadialLine = Thomas radial par ligne theta). PAS de
/// V-cycle MG (stagnation sur 1/r^2 polaire, cf. entete et PolarPrecond).
///
/// CYCLE DE VIE : construit sur une PolarGeometry + BoxArray + BCRec (radiale). En MPI multi-rang, le
/// BoxArray doit etre decoupe en THETA SEULEMENT (chaque box couvre la plage radiale complete) pour le
/// preconditionneur RadialLine (cf. portee + check_radial_columns) ; le repli Jacobi n'a pas cette
/// contrainte. rhs()/phi() sont les champs du solve (warm start sur phi()). set_coefficients(...) fixe le
/// tenseur (pointeurs vers des champs de coefficient AU CENTRE, ghosts remplis par l'appelant avant
/// solve). En isotrope, appeler sans a_rt/a_tr et fournir a_rr/a_tt = champs a 1 (helper fill_one).
class PolarTensorKrylovSolver {
 public:
  /// @param geom anneau (r, theta) ; @param ba BoxArray (boite unique ou decoupage THETA en MPI/multi-box) ;
  /// @param bc BC radiale (xlo/xhi), theta periodique.
  /// @param precond preconditionneur simple (RadialLine par defaut ; Jacobi en repli/sanity check).
  PolarTensorKrylovSolver(const PolarGeometry& geom, const BoxArray& ba, const BCRec& bc,
                          PolarPrecond precond = PolarPrecond::RadialLine)
      : geom_(geom),
        bc_(bc),
        precond_(precond),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 1), rhs_(ba, dm_, 1, 0),
        r_(ba, dm_, 1, 0), rhat_(ba, dm_, 1, 0), p_(ba, dm_, 1, 0),
        v_(ba, dm_, 1, 0), s_(ba, dm_, 1, 0), t_(ba, dm_, 1, 0),
        phat_(ba, dm_, 1, 1), shat_(ba, dm_, 1, 1),
        idiag_(ba, dm_, 1, 0), op_offset_(ba, dm_, 1, 0),
        a_rr_store_(ba, dm_, 1, 1), a_tt_store_(ba, dm_, 1, 1) {
    // GARDE-FOU DE LAYOUT (remplace le garde-fou mono-rang) : le preconditionneur RadialLine resout une
    // tridiagonale RADIALE par ligne theta via un sweep Thomas SEQUENTIEL en r. Ce sweep doit rester
    // LOCAL a une box (il ne peut pas franchir une frontiere de box/rang). On EXIGE donc que chaque box
    // du BoxArray couvre la PLAGE RADIALE COMPLETE [domain.lo[0], domain.hi[0]] (decoupage en theta
    // seulement). Le repli Jacobi (par cellule) n'a pas cette contrainte -> aucun check. Mono-rang boite
    // unique : la box couvre tout l'anneau, le check passe trivialement (chemin inchange).
    if (precond_ == PolarPrecond::RadialLine) check_radial_columns(ba);
    // Coefficients diagonaux par defaut = 1 (isotrope) : champs internes a 1, ghosts remplis Foextrap.
    a_rr_store_.set_val(Real(1));
    a_tt_store_.set_val(Real(1));
    a_rr_ = &a_rr_store_;
    a_tt_ = &a_tt_store_;
  }

  // --- contrat PolarEllipticSolver / PolarLinearSolver ---
  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const PolarGeometry& geom() const { return geom_; }

  /// Fixe les coefficients du tenseur A. a_rr/a_tt AU CENTRE (defaut interne = 1 si non appele) ; a_rt/
  /// a_tr termes croises (nullptr -> absents). Les ghosts des champs fournis sont remplis ICI (coeff_bc :
  /// periodique theta conserve, radial Foextrap). L'appelant garde la propriete des champs (pointeurs).
  void set_coefficients(MultiFab* a_rr, MultiFab* a_tt, MultiFab* a_rt = nullptr,
                        MultiFab* a_tr = nullptr) {
    a_rr_ = a_rr ? a_rr : &a_rr_store_;
    a_tt_ = a_tt ? a_tt : &a_tt_store_;
    a_rt_ = a_rt;
    a_tr_ = a_tr;
    fill_coeff_ghosts();
    coeffs_ready_ = false;  // idiag a recalculer
  }

  /// Residu L2 GLOBAL courant ||rhs - L_int(phi)|| (collectif). prepare une fois si besoin.
  Real residual() {
    ensure_coeffs();
    apply_operator(phi_, r_);
    lincomb(r_, Real(1), rhs_, Real(-1), r_);
    return l2_norm(r_);
  }

  void solve() { solve(Real(1e-10), 400); }

  /// BiCGStab preconditionne Jacobi. phi() = inconnue (warm start), rhs() = second membre. Rend
  /// iterations + residu relatif + convergence.
  PolarKrylovResult solve(Real rel_tol, int max_iters) {
    ensure_coeffs();
    prepare_offset();  // c_bc = apply_operator(0) (part inhomogene de bord Dirichlet) une fois
    PolarKrylovResult res;
    // MULTI-RANG MPI : TOUS les rangs (y compris ceux sans box, local_size()==0) executent le MEME corps
    // BiCGStab. Les operations par fab (lincomb/saxpy/copy/apply_precond/apply_polar_tensor) sont des
    // no-op sur un rang vide ; les COLLECTIFS (dot/l2_norm/project_mean -> all_reduce_sum) sont appeles
    // par TOUS les rangs dans le MEME ordre (un rang vide contribue 0). Les criteres d'arret reposent
    // uniquement sur ces scalaires GLOBAUX (identiques partout) -> aucune desynchronisation, aucun
    // chemin "run_empty_rank" separe a maintenir en parallele. fill_ghosts (fill_boundary) est un
    // echange PAIRWISE (Isend/Irecv apparies, pas un collectif) : un rang vide ne poste rien et personne
    // ne lui envoie -> pas d'interblocage.
    // r0 = rhs - L_int(phi) (operateur AFFINE : la donnee de Dirichlet est repliee dans le residu).
    apply_operator(phi_, v_);
    lincomb(r_, Real(1), rhs_, Real(-1), v_);
    // OPERATEUR SINGULIER (pure Neumann radial + theta periodique, aucune reaction) : la constante est
    // dans le noyau de L_int. BiCGStab diverge alors (le residu garde une composante constante non
    // amortie). On FIXE LA JAUGE par projection sur le sous-espace de MOYENNE NULLE : on retire la
    // moyenne de r0 (le RHS doit etre compatible : moyenne nulle au sens FV) et de phi (jauge), puis de
    // chaque direction de correction preconditionnee dans la boucle. C'est le pendant iteratif du
    // pinning de mode 0 du PolarPoissonSolver direct, sans perturber le stencil. Cas Dirichlet (>= un
    // bord) : pin_gauge_ reste false -> CHEMIN INCHANGE, bit-identique.
    if (pin_gauge_) { project_mean(r_); project_mean(phi_); }
    const Real bnorm = l2_norm(rhs_);
    const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);
    Real rnorm = l2_norm(r_);
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) { res.converged = true; return res; }

    copy_into(rhat_, r_);
    p_.set_val(Real(0));
    v_.set_val(Real(0));
    Real rho_prev = Real(1), alpha = Real(1), omega = Real(1);

    for (int k = 1; k <= max_iters; ++k) {
      const Real rho = dot(rhat_, r_);  // COLLECTIF
      if (std::fabs(rho) < kTiny || std::fabs(omega) < kTiny) {
        res.iters = k - 1; res.rel_residual = rnorm / norm0; return res;
      }
      const Real beta = (rho / rho_prev) * (alpha / omega);
      lincomb(p_, Real(1), p_, -omega, v_);  // p <- p - omega v
      lincomb(p_, beta, p_, Real(1), r_);    // p <- r + beta p
      apply_precond(p_, phat_);              // phat = M^{-1} p
      if (pin_gauge_) project_mean(phat_);   // jauge : direction de correction a moyenne nulle
      apply_operator_lin(phat_, v_);         // v = L_lin(phat) (matvec LINEAIRE)
      const Real rhat_dot_v = dot(rhat_, v_);
      if (std::fabs(rhat_dot_v) < kTiny) { res.iters = k - 1; res.rel_residual = rnorm / norm0; return res; }
      alpha = rho / rhat_dot_v;
      lincomb(s_, Real(1), r_, -alpha, v_);  // s <- r - alpha v
      saxpy(phi_, alpha, phat_);             // phi <- phi + alpha phat
      const Real snorm = l2_norm(s_);
      if (snorm <= rel_tol * norm0) {
        rnorm = snorm; res.iters = k; res.rel_residual = rnorm / norm0; res.converged = true; return res;
      }
      apply_precond(s_, shat_);              // shat = M^{-1} s
      if (pin_gauge_) project_mean(shat_);   // jauge : direction de correction a moyenne nulle
      apply_operator_lin(shat_, t_);         // t = L_lin(shat)
      const Real tt = dot(t_, t_);
      omega = tt > kTiny ? dot(t_, s_) / tt : Real(0);
      saxpy(phi_, omega, shat_);             // phi <- phi + omega shat
      lincomb(r_, Real(1), s_, -omega, t_);  // r <- s - omega t
      rnorm = l2_norm(r_);
      res.iters = k; res.rel_residual = rnorm / norm0;
      if (rnorm <= rel_tol * norm0) { res.converged = true; return res; }
      rho_prev = rho;
    }
    return res;  // max_iters atteint : meilleur effort (converged=false)
  }

 private:
  static constexpr Real kTiny = Real(1e-300);

  /// Garde-fou de LAYOUT du preconditionneur RadialLine : chaque box doit couvrir la plage radiale
  /// COMPLETE du domaine (decoupage en THETA seulement), faute de quoi le sweep Thomas en r franchirait
  /// une frontiere de box. Leve sur TOUS les rangs (la BoxArray est repliquee, cf. DistributionMapping)
  /// -> pas d'interblocage. Mono-rang boite unique : la box == domaine, check trivialement vrai.
  void check_radial_columns(const BoxArray& ba) const {
    const Box2D dom = geom_.domain;
    for (int b = 0; b < ba.size(); ++b) {
      if (ba[b].lo[0] != dom.lo[0] || ba[b].hi[0] != dom.hi[0])
        throw std::runtime_error(
            "PolarTensorKrylovSolver (precond RadialLine) : le BoxArray doit etre decoupe en THETA "
            "seulement (chaque box couvre la plage radiale complete). Le sweep Thomas en r ne peut pas "
            "franchir une frontiere de box. Utiliser un decoupage azimutal, ou le repli "
            "PolarPrecond::Jacobi (par cellule, sans contrainte de layout) si r doit etre coupe.");
    }
  }

  /// CL des champs de coefficient : periodique conserve (theta), bord physique radial -> Foextrap.
  BCRec coeff_bc() const {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc_.xlo); b.xhi = fo(bc_.xhi);
    b.ylo = BCType::Periodic; b.yhi = BCType::Periodic;  // theta toujours periodique
    return b;
  }

  void fill_coeff_ghosts() {
    const BCRec eb = coeff_bc();
    device_fence();
    fill_ghosts(*a_rr_, geom_.domain, eb);
    fill_ghosts(*a_tt_, geom_.domain, eb);
    if (a_rt_) fill_ghosts(*a_rt_, geom_.domain, eb);
    if (a_tr_) fill_ghosts(*a_tr_, geom_.domain, eb);
  }

  void ensure_coeffs() {
    if (coeffs_ready_) return;
    // idiag = 1/diag du stencil diagonal (pour Jacobi).
    const Real dr = geom_.dr();
    const Real idr = Real(1) / dr;
    const Real idth = Real(1) / geom_.dtheta();
    for (int li = 0; li < idiag_.local_size(); ++li) {
      for_each_cell(idiag_.box(li),
                    detail::PolarInvDiagKernel{a_rr_->fab(li).const_array(), a_tt_->fab(li).const_array(),
                                               idiag_.fab(li).array(), geom_.r_min, dr, idr, idth});
    }
    if (precond_ == PolarPrecond::RadialLine) build_radial_lines();
    coeffs_ready_ = true;
  }

  /// Precalcule, pour le preconditionneur RadialLine, les coefficients de la tridiagonale RADIALE du
  /// bloc DIAGONAL (par ligne theta j et par rayon i). C'est HOTE (les sweeps Thomas du precond lisent
  /// ces tableaux cote hote, comme PolarPoissonSolver). Le coefficient a_rr est moyenne en r (les deux
  /// faces radiales i+-1/2) ; le terme azimutal du bloc diagonal est LUMPE dans la diagonale b_ij (pour
  /// que M approche le diag complet de L_int). BC radiale repliee HOMOGENE (le precond agit sur des
  /// directions/residus) : Dirichlet b -= a/c, Neumann b += a/c.
  /// MULTI-BOX / MPI (decoupage THETA seul) : on boucle sur les boxes LOCALES (local_size()). Chaque box
  /// couvre la plage radiale COMPLETE (garde-fou check_radial_columns), donc nr est global et le sweep
  /// Thomas reste box-local. Les tableaux sont ranges PAR BOX LOCALE : line_b_[li] indexe [jl*nr + i] ou
  /// jl est l'indice theta LOCAL a la box. Mono-rang boite unique : une seule box -> identique a l'ancien
  /// rangement [j*nr + i] sur tout le domaine (chemin bit-identique).
  void build_radial_lines() {
    a_rr_->sync_host(); a_tt_->sync_host();
    const int nr = geom_.domain.nx();
    const Box2D dom = geom_.domain;
    const Real dr = geom_.dr();
    const Real idr2 = Real(1) / (dr * dr);
    const Real idth2 = Real(1) / (geom_.dtheta() * geom_.dtheta());
    const bool dir_lo = bc_.xlo == BCType::Dirichlet;
    const bool dir_hi = bc_.xhi == BCType::Dirichlet;
    nr_ = nr;
    const int nloc = idiag_.local_size();
    line_b_.resize(static_cast<std::size_t>(nloc));
    line_sub_.resize(static_cast<std::size_t>(nloc));
    line_sup_.resize(static_cast<std::size_t>(nloc));
    for (int li = 0; li < nloc; ++li) {
      const Box2D vb = idiag_.box(li);           // box VALIDE locale (plage r complete, sous-plage theta)
      const int nth_l = vb.ny();                 // nombre de lignes theta LOCALES a cette box
      const ConstArray4 arr = a_rr_->fab(li).const_array();
      const ConstArray4 att = a_tt_->fab(li).const_array();
      std::vector<Real>& lb = line_b_[li];
      std::vector<Real>& lsub = line_sub_[li];
      std::vector<Real>& lsup = line_sup_[li];
      lb.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      lsub.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      lsup.assign(static_cast<std::size_t>(nth_l) * static_cast<std::size_t>(nr), Real(0));
      for (int jl = 0; jl < nth_l; ++jl) {
        const int jg = vb.lo[1] + jl;            // indice theta GLOBAL
        for (int i = 0; i < nr; ++i) {
          const int ig = dom.lo[0] + i;
          const Real ri = geom_.r_cell(i);
          const Real rfm = geom_.r_face(i);
          const Real rfp = geom_.r_face(i + 1);
          const Real arr_p = Real(0.5) * (arr(ig, jg) + arr(ig + 1, jg));
          const Real arr_m = Real(0.5) * (arr(ig, jg) + arr(ig - 1, jg));
          const Real att_p = Real(0.5) * (att(ig, jg) + att(ig, jg + 1));
          const Real att_m = Real(0.5) * (att(ig, jg) + att(ig, jg - 1));
          const Real ai = rfm * arr_m * (idr2 / ri);   // coeff de p_{i-1} dans L_int
          const Real ci = rfp * arr_p * (idr2 / ri);    // coeff de p_{i+1}
          const Real azi_diag = (att_p + att_m) * (idth2 / (ri * ri));  // part azimutale du -diag (lumpee)
          Real bi = -(ai + ci) - azi_diag;              // diagonale COMPLETE de L_int (radiale + azimutale)
          Real sub = ai, sup = ci;
          if (i == 0) {  // repli BC bas HOMOGENE : Dirichlet b -= a, Neumann b += a ; sous-diag annulee
            bi += dir_lo ? -ai : ai;
            sub = Real(0);
          }
          if (i == nr - 1) {  // repli BC haut HOMOGENE
            bi += dir_hi ? -ci : ci;
            sup = Real(0);
          }
          const std::size_t idx = static_cast<std::size_t>(jl) * static_cast<std::size_t>(nr) + static_cast<std::size_t>(i);
          lb[idx] = bi;
          lsub[idx] = sub;
          lsup[idx] = sup;
        }
      }
    }
  }

  /// matvec MATRICE-LIBRE INHOMOGENE : out = L_int(in), ghosts de in remplis avec bc_ (CL ENTIERE).
  /// AFFINE en in sous Dirichlet non nul (terme constant c_bc replie). Utilise pour r0 / residual().
  void apply_operator(MultiFab& in, MultiFab& out) {
    device_fence();
    fill_ghosts(in, geom_.domain, bc_);
    apply_polar_tensor(in, geom_, out, a_rr_, a_tt_, a_rt_, a_tr_);
  }

  /// matvec MATRICE-LIBRE LINEAIRE : out = L_int(in) - c_bc. BiCGStab applique la matvec a des
  /// DIRECTIONS de correction -> l'operateur doit y etre lineaire (on retranche l'offset de bord).
  void apply_operator_lin(MultiFab& in, MultiFab& out) {
    apply_operator(in, out);
    if (has_op_offset_) lincomb(out, Real(1), out, Real(-1), op_offset_);
  }

  /// preconditionneur SIMPLE : Jacobi (diagonal) ou RadialLine (Thomas radial par ligne theta).
  /// LINEAIRE, sans CL inhomogene (agit sur des directions/residus). Pendant SIMPLE du M^{-1} du Krylov
  /// cartesien (qui, lui, fait N V-cycles MG ; ici PAS de MG : stagnation 1/r^2 polaire).
  void apply_precond(MultiFab& in, MultiFab& out) {
    if (precond_ == PolarPrecond::Jacobi) {
      for (int li = 0; li < out.local_size(); ++li)
        for_each_cell(out.box(li),
                      detail::PolarJacobiApplyKernel{in.fab(li).const_array(), out.fab(li).array(),
                                                     idiag_.fab(li).const_array()});
      return;
    }
    apply_precond_radial_line(in, out);
  }

  /// RadialLine : pour chaque ligne theta j, resout la tridiagonale radiale (sub/diag/sup precalculee
  /// dans build_radial_lines) par Thomas, second membre = in(., j). out(., j) recoit la correction.
  /// HOTE (Thomas sequentiel en r) comme PolarPoissonSolver ; sync_host avant lecture de in.
  /// MULTI-BOX / MPI : boucle sur les boxes LOCALES (chaque box couvre la plage r complete -> le sweep
  /// reste box-local) ; line_b_/sub_/sup_[li] sont indexes par l'indice theta LOCAL a la box. Aucun
  /// echange MPI (M^{-1} est BLOC-DIAGONAL en box, par construction du decoupage theta) -> pas
  /// d'interblocage. Mono-rang boite unique : une box -> chemin bit-identique a l'ancien fab(0).
  void apply_precond_radial_line(MultiFab& in, MultiFab& out) {
    if (in.local_size() == 0) return;
    in.sync_host();
    const int r_lo = geom_.domain.lo[0];
    const std::size_t N = static_cast<std::size_t>(nr_);
    cthom_.assign(N, Real(0));  // sur-diag de travail (Thomas)
    xthom_.assign(N, Real(0));  // solution de travail
    for (int li = 0; li < in.local_size(); ++li) {
      const Box2D vb = in.box(li);
      const ConstArray4 z = in.fab(li).const_array();
      Array4 o = out.fab(li).array();
      const std::vector<Real>& lb = line_b_[li];
      const std::vector<Real>& lsub = line_sub_[li];
      const std::vector<Real>& lsup = line_sup_[li];
      const int nth_l = vb.ny();
      for (int jl = 0; jl < nth_l; ++jl) {
        const int jg = vb.lo[1] + jl;
        const std::size_t base = static_cast<std::size_t>(jl) * N;
        // Thomas en r : a = lsub, b = lb, c = lsup (base + i), rhs = z(., jg).
        Real beta = lb[base + 0];
        xthom_[0] = (beta != Real(0) ? z(r_lo, jg) / beta : Real(0));
        for (std::size_t i = 1; i < N; ++i) {
          cthom_[i] = lsup[base + i - 1] / beta;
          beta = lb[base + i] - lsub[base + i] * cthom_[i];
          const Real zi = z(r_lo + static_cast<int>(i), jg);
          xthom_[i] = (beta != Real(0) ? (zi - lsub[base + i] * xthom_[i - 1]) / beta : Real(0));
        }
        for (int i = static_cast<int>(N) - 2; i >= 0; --i)
          xthom_[static_cast<std::size_t>(i)] -= cthom_[static_cast<std::size_t>(i + 1)] * xthom_[static_cast<std::size_t>(i + 1)];
        for (std::size_t i = 0; i < N; ++i) o(r_lo + static_cast<int>(i), jg) = xthom_[i];
      }
    }
  }

  /// Prepare l'offset c_bc = apply_operator(0) une fois par solve. CL Dirichlet nulle -> offset nul
  /// (has_op_offset_ = false), chemin inchange. Detecte aussi l'operateur SINGULIER (pure Neumann radial
  /// : aucun bord Dirichlet) -> pin_gauge_ : la constante est dans le noyau, on fixera la jauge par
  /// projection de moyenne nulle (cf. solve). Au moins un bord Dirichlet => inversible => pin_gauge_=false.
  void prepare_offset() {
    has_op_offset_ = (bc_.xlo == BCType::Dirichlet && bc_.xlo_val != Real(0)) ||
                     (bc_.xhi == BCType::Dirichlet && bc_.xhi_val != Real(0));
    pin_gauge_ = (bc_.xlo != BCType::Dirichlet) && (bc_.xhi != BCType::Dirichlet);
    if (has_op_offset_) {
      // Appele sur TOUS les rangs (apply_operator est no-op sur un rang vide hormis fill_ghosts pairwise) :
      // pas de branche local_size() qui desapparierait l'echange de halos cross-rang.
      phat_.set_val(Real(0));
      apply_operator(phat_, op_offset_);  // op_offset_ <- L_int(0) = c_bc
    }
  }

  /// Retire la moyenne FV (ponderee volume r_i dr dtheta) de @p x (projection sur le sous-espace de
  /// moyenne nulle, orthogonal au noyau constant de L_int). Brique du pinning de jauge (operateur
  /// singulier pure Neumann). HOTE (la moyenne ponderee est un scalaire global).
  /// MULTI-RANG MPI : le numerateur (sum) et le denominateur (vol) sont d'abord accumules sur les
  /// cellules LOCALES (toutes les boxes de ce rang), puis all_reduit (MPI_SUM) -> la MEME moyenne
  /// GLOBALE sur chaque rang ; on la retranche ensuite sur toutes les cellules locales. COLLECTIF :
  /// all_reduce_sum est appele sur CHAQUE rang (y compris un rang sans box, qui contribue 0) -> pas
  /// d'interblocage. Mono-rang : all_reduce_sum = identite, une seule box -> chemin bit-identique.
  void project_mean(MultiFab& x) {
    x.sync_host();
    const int r_lo = geom_.domain.lo[0];
    const Real dr = geom_.dr(), dth = geom_.dtheta();
    Real sum = 0, vol = 0;
    for (int li = 0; li < x.local_size(); ++li) {
      const Box2D vb = x.box(li);
      const ConstArray4 a = x.fab(li).const_array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          const Real w = geom_.r_cell(i - r_lo) * dr * dth;  // i - r_lo = offset radial 0-base
          sum += a(i, j) * w;
          vol += w;
        }
    }
    // all-reduce COLLECTIF (sur tous les rangs, y compris vides) : moyenne GLOBALE identique partout.
    sum = static_cast<Real>(all_reduce_sum(static_cast<double>(sum)));
    vol = static_cast<Real>(all_reduce_sum(static_cast<double>(vol)));
    const Real mean = vol > Real(0) ? sum / vol : Real(0);
    for (int li = 0; li < x.local_size(); ++li) {
      const Box2D vb = x.box(li);
      Array4 a = x.fab(li).array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) a(i, j) -= mean;
    }
  }

  Real l2_norm(const MultiFab& x) { return std::sqrt(dot(x, x)); }

  void copy_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      for_each_cell(dst.box(li), detail::PolarCopyKernel{d, s});
    }
  }

  PolarGeometry geom_;
  BCRec bc_;
  PolarPrecond precond_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_;
  MultiFab r_, rhat_, p_, v_, s_, t_;
  MultiFab phat_, shat_;
  MultiFab idiag_;        ///< 1/diag (Jacobi), recalcule a chaque set_coefficients
  MultiFab op_offset_;    ///< c_bc = apply_operator(0) (part inhomogene de bord Dirichlet)
  MultiFab a_rr_store_, a_tt_store_;  ///< coefficients diagonaux par defaut (= 1, isotrope)
  MultiFab* a_rr_ = nullptr;          ///< coefficients courants (pointe sur le store ou l'externe)
  MultiFab* a_tt_ = nullptr;
  MultiFab* a_rt_ = nullptr;          ///< termes croises (nullptr -> absents)
  MultiFab* a_tr_ = nullptr;
  bool coeffs_ready_ = false;
  bool has_op_offset_ = false;
  bool pin_gauge_ = false;  ///< operateur singulier (pure Neumann) : fixe la jauge (projection moyenne nulle)
  // Preconditionneur RadialLine : tridiagonale radiale du bloc diagonal par ligne theta. PAR BOX LOCALE
  // (line_b_/sub_/sup_[li]), chacun range [jl*nr + i] ou jl = indice theta LOCAL a la box, i = rayon
  // global 0..nr-1 (a_rr peut dependre de theta -> coeffs par (i, jl)). Le decoupage etant en THETA seul,
  // M^{-1} est bloc-diagonal en box (aucun couplage radial cross-box) -> chaque box est inversee
  // independamment. cthom_/xthom_ = buffers de travail Thomas (reutilises, longueur nr). HOTE.
  std::vector<std::vector<Real>> line_b_, line_sub_, line_sup_;  ///< tridiag par box locale [li][jl*nr+i]
  mutable std::vector<Real> cthom_, xthom_;    ///< buffers Thomas du precond (longueur nr)
  int nr_ = 0;
};

static_assert(PolarLinearSolver<PolarTensorKrylovSolver>,
              "PolarTensorKrylovSolver doit modeler PolarLinearSolver");

}  // namespace adc
