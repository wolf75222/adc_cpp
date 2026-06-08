#pragma once

#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/polar_tensor_operator.hpp>  // PolarTensorKrylovSolver, apply_polar_tensor (#210)
#include <adc/numerics/lorentz_eliminator.hpp>               // B^{-1} ferme (#118)
#include <adc/parallel/comm.hpp>

#include <stdexcept>

/// @file
/// @brief PolarCondensedSchurSourceStepper : ETAGE SOURCE condense par Schur en geometrie POLAIRE
///        (anneau (r, theta)). Pendant POLAIRE du CondensedSchurSourceStepper cartesien (#126,
///        condensed_schur_source_stepper.hpp), pour la source implicite couplee potentiel / vitesse /
///        Lorentz (Hoffart et al., arXiv:2510.11808) d'un fluide magnetise RAIDE sur un maillage
///        annulaire. C'est un etage SOURCE AUTONOME (transport gele) ; il NE remplace PAS System::step
///        et ne s'y branche PAS (cablage facade = ulterieur).
///
/// CHEMIN SEPARE, additif (Voie A etape 2b). Le Schur CARTESIEN reste BIT-IDENTIQUE
/// (condensed_schur_source_stepper.hpp INTOUCHE) ; ce header ne touche aucun chemin existant. Il COMPOSE
/// les briques deja sur master :
///   - PolarTensorKrylovSolver (polar_tensor_operator.hpp, #210) : BiCGStab matrice-libre pour
///     L_int(phi) = div(A grad phi) en METRIQUE POLAIRE, tenseur PLEIN A = [[a_rr, a_rt], [a_tr, a_tt]]
///     non symetrique (preconditionneur RadialLine) ;
///   - LorentzEliminator (lorentz_eliminator.hpp, #118) : B^{-1} ferme (rotation-dilatation 2x2).
///
/// POURQUOI LE MEME B^{-1} 2x2 QU'EN CARTESIEN. La vitesse polaire (v_r, v_theta) est exprimee dans le
/// repere LOCAL ORTHONORME (e_r, e_theta). Le champ magnetique est porte par z (B = B_z z_hat,
/// orthogonal au plan (r, theta)). La force de Lorentz v x B est une ROTATION dans le plan, INDEPENDANTE
/// de l'orientation du repere orthonorme local : (v x B)_r = +B_z v_theta, (v x B)_theta = -B_z v_r,
/// exactement la forme cartesienne (v x B)_x = +B_z v_y, (v x B)_y = -B_z v_x. Donc le MEME
/// LorentzEliminator s'applique a (v_r, v_theta), et le tenseur condense A = I + c rho B^{-1} (c =
/// theta^2 dt^2 alpha) a les memes entrees, simplement RE-ETIQUETEES (x -> r, y -> theta) :
///   a_rr = 1 + c rho binv_11      a_rt = c rho binv_12
///   a_tr = c rho binv_21          a_tt = 1 + c rho binv_22
/// avec B^{-1} = (1/det)[[1, w], [-w, 1]], w = theta dt B_z, det = 1 + w^2. C'est EXACTEMENT le tenseur
/// plein non symetrique que PolarTensorKrylovSolver (#210) accepte : le terme croise a_rt/a_tr EST la
/// rotation de Lorentz. La METRIQUE POLAIRE (1/r, 1/r^2) est portee par l'operateur du solveur ; le
/// stepper n'assemble QUE les coefficients AU CENTRE et le second membre.
///
/// SEQUENCE (une fois par etage source), pendant polaire de docs/SCHUR_CONDENSATION_DESIGN.md niveau 4 :
///   1. ASSEMBLER A = I + c rho^n B^{-1} (4 champs de coefficient au centre) et le second membre
///      condense rhs_polar = Lap_polar phi^n + theta dt alpha div_polar(rho^n B^{-1} v^n)            ;
///   2. RESOUDRE L_int(phi) = div(A grad phi) = rhs_polar via PolarTensorKrylovSolver                ;
///   3. RECONSTRUIRE v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}), mom = rho^n v    ;
///   4. ENERGIE (si role Energy) : E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2)                 ;
///   5. EXTRAPOLER U^{n+theta} -> U^{n+1} : f^{n+1} = f^n + (1/theta)(f^{n+theta} - f^n)             ;
///   6. REMPLIR les ghosts (theta periodique, r physique) avant de rendre la main.
///
/// CONVENTION DE SIGNE DU SOLVE (a lire avant toute modification). Le batisseur cartesien (#124) ecrit
///   l'etage L_schur(phi) = rhs_schur avec L_schur(phi) = -div(A grad phi), et
///       rhs_schur = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n).
///   Le PolarTensorKrylovSolver (#210), lui, resout L_int(phi) = +div(A grad phi) = rhs (signe POSITIF,
///   cf. test_polar_tensor_elliptic_mms : le RHS y vaut directement f = div(A grad phi_exact)). On a
///   L_schur = -L_int, donc resoudre L_schur(phi) = rhs_schur revient a resoudre L_int(phi) = -rhs_schur.
///   Le second membre POLAIRE fourni au solveur est donc -rhs_schur :
///       rhs_polar = -rhs_schur = +Lap_polar phi^n + theta dt alpha div_polar(rho B^{-1} v^n).
///   Garde-fou : c = 0 et B_z = 0 -> A = I, le solve devient Lap_polar phi^{n+theta} = Lap_polar phi^n
///   -> phi^{n+theta} = phi^n (a la jauge pres), et la reconstruction degenere en la poussee
///   electrostatique explicite v^{n+theta} = v^n - theta dt grad_polar phi^n (cas B = 0).
///
/// DISCRETISATION DU SECOND MEMBRE (coherente avec l'operateur polaire #210 et assemble_rhs_polar).
///   - Lap_polar phi^n : applique apply_polar_tensor avec A = I (coefficients a 1) = (1/r) d_r(r d_r phi)
///     + (1/r^2) d_theta^2 phi, stencil FV conservatif radial + FD azimutal 2 points (le MEME que
///     l'operateur du solve, garantissant la degenerescence exacte en c=0,B_z=0).
///   - div_polar(F), F = rho B^{-1} v^n = B^{-1}(mr, mtheta) au centre : divergence polaire FV centree
///       div F (i,j) = (1/r_i) (r_{i+1/2} - r_{i-1/2} repartis par diff centree) ... discretisee par
///       div F = (1/r_i)(d_r(r F_r))_centre + (1/r_i) (d_theta F_theta)_centre, soit
///       div F = (1/r_i) [ (r_{i+1} F_r(i+1) - r_{i-1} F_r(i-1)) / (2 dr) + (F_th(i,j+1) - F_th(i,j-1)) / (2 dtheta) ].
///     Centree d'ordre 2 (pendant polaire de la divergence centree cartesienne du batisseur #124).
///
/// GRADIENT POLAIRE (reconstruction). grad_polar phi = (d_r phi, (1/r) d_theta phi), differences CENTREES :
///       (grad phi)_r     = (phi(i+1,j) - phi(i-1,j)) / (2 dr)
///       (grad phi)_theta = (phi(i,j+1) - phi(i,j-1)) / (2 r_i dtheta).
///   Memes differences centrees que la divergence du RHS (coherence terme a terme a la precision du
///   solve, comme le SchurReconstructKernel cartesien utilise grad centre).
///
/// PORTEE : MULTI-RANG MPI par decoupage AZIMUTAL (theta seul), comme PolarTensorKrylovSolver (#210).
///   Le corps de step() itere deja sur local_size() partout et passe par fill_ghosts (echange de halos
///   MPI + CL physique) pour tous les champs (bz_/a_rr_/a_tt_/a_rt_/a_tr_/fr_/ft_/phi/state) : il est
///   structurellement distribue. Le solve elliptique (PolarTensorKrylovSolver) supporte le multi-rang
///   sous la contrainte de decoupage theta (chaque box couvre la plage radiale complete pour le
///   preconditionneur RadialLine ; le garde-fou check_radial_columns du solveur l'impose). Mono-rang /
///   boite unique : chemin BIT-IDENTIQUE. Multi-box / device = differe (Extend).
///
/// DEVICE. Tous les kernels sont des FONCTEURS NOMMES device-clean (recette #93 : pas de lambda etendue
///   premiere-instanciee cross-TU, limite nvcc #64/#97). Les tampons sont ALLOUES UNE FOIS a la
///   construction et REUTILISES a chaque step(). Le solveur de Krylov est cree par appel (tampons fixes).

namespace adc {

namespace detail {

/// Coefficients du tenseur condense POLAIRE A = I + c rho B^{-1} au centre des cellules. Re-etiquetage
/// polaire des coefficients cartesiens (#124, SchurOperatorCoeffKernel) : x -> r, y -> theta. Le MEME
/// B^{-1} 2x2 (rotation de Lorentz) car la force v x B est independante de l'orientation du repere
/// orthonorme local (cf. entete). Foncteur NOMME device-clean.
struct PolarSchurOperatorCoeffKernel {
  ConstArray4 s;     ///< etat fluide (lecture rho)
  ConstArray4 bz;    ///< champ B_z au centre
  Array4 arr, att;   ///< sortie : a_rr, a_tt (diagonale de A)
  Array4 art, atr;   ///< sortie : termes croises a_rt, a_tr
  Real c;            ///< c = theta^2 dt^2 alpha
  Real th_dt;        ///< theta * dt (w = th_dt * B_z, binv ne depend que de w)
  int c_rho;         ///< composante Density
  ADC_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    const Real cr = c * rho;
    arr(i, j, 0) = Real(1) + cr * le.binv_11();
    att(i, j, 0) = Real(1) + cr * le.binv_22();
    art(i, j, 0) = cr * le.binv_12();
    atr(i, j, 0) = cr * le.binv_21();
  }
};

/// Flux EXPLICITE F = rho B^{-1} v^n = B^{-1}(mr, mtheta) au centre (composantes physiques (e_r,
/// e_theta)). On applique B^{-1} DIRECTEMENT a la quantite de mouvement (evite la division par rho).
/// Foncteur NOMME device-clean. Re-etiquetage polaire de SchurExplicitFluxKernel (#124).
struct PolarSchurExplicitFluxKernel {
  ConstArray4 s;     ///< etat fluide (mr, mtheta lus aux composantes c_mx, c_my)
  ConstArray4 bz;    ///< champ B_z au centre
  Array4 fr, ft;     ///< sortie : F_r, F_theta = B^{-1}(mr, mtheta)
  Real th_dt;        ///< theta * dt (w = th_dt * B_z)
  int c_mx, c_my;    ///< composantes MomentumX (= radiale), MomentumY (= azimutale)
  ADC_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    Real Fr, Ft;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fr, Ft);  // B^{-1}(mr, mtheta) = rho B^{-1} v
    fr(i, j, 0) = Fr;
    ft(i, j, 0) = Ft;
  }
};

/// rhs_polar(i,j) = lap_polar(i,j) (= Lap_polar phi^n) + g * div_polar F, divergence POLAIRE centree
/// d'ordre 2 d'un champ vecteur F = (F_r, F_theta) au centre (ghosts remplis). g = theta dt alpha.
///   div_polar F = (1/r_i) [ (r_{i+1} F_r(i+1) - r_{i-1} F_r(i-1)) / (2 dr)
///                          + (F_th(i,j+1) - F_th(i,j-1)) / (2 dtheta) ].
/// Le signe POSITIF de div (et de lap) est l'opposee de la convention schur cartesienne : on resout
/// L_int = +div(A grad phi) = rhs_polar = -rhs_schur (cf. entete, convention de signe). Foncteur NOMME.
struct PolarSchurRhsAssembleKernel {
  ConstArray4 lap;      ///< Lap_polar phi^n (signe positif, A=I)
  ConstArray4 fr, ft;   ///< flux F au centre (ghosts remplis)
  Array4 rhs;           ///< sortie : second membre condense (signe L_int)
  Real g;               ///< theta dt alpha
  Real half_idr, half_idth;  ///< 1/(2 dr), 1/(2 dtheta)
  Real r_min, dr;       ///< pour r_cell(i), r_cell(i+-1)
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real rip = r_min + (i + Real(1.5)) * dr;  // r_cell(i+1)
    const Real rim = r_min + (i - Real(0.5)) * dr;  // r_cell(i-1)
    const Real inv_r = Real(1) / ri;
    const Real div_r = (rip * fr(i + 1, j, 0) - rim * fr(i - 1, j, 0)) * half_idr;  // d_r(r F_r) centre
    const Real div_t = (ft(i, j + 1, 0) - ft(i, j - 1, 0)) * half_idth;             // d_theta(F_th) centre
    const Real divF = inv_r * (div_r + div_t);
    rhs(i, j, 0) = lap(i, j, 0) + g * divF;
  }
};

/// Reconstruit v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}) et ecrit mom = rho^n
/// v^{n+theta}. grad_polar phi est la difference CENTREE : (d_r phi, (1/r) d_theta phi). rho lu dans
/// l'etat (role Density), GELE. Foncteur NOMME device-clean. Pendant polaire de SchurReconstructKernel.
struct PolarSchurReconstructKernel {
  ConstArray4 phi;     ///< phi^{n+theta} (ghosts remplis : grad centre lit i+-1, j+-1)
  ConstArray4 vr, vt;  ///< v^n (composantes 0 : vitesse, PAS quantite de mouvement)
  ConstArray4 bz;      ///< champ B_z au centre
  Array4 st;           ///< etat fluide (ECRITURE de mr, mtheta ; LECTURE de rho)
  Array4 nvr, nvt;     ///< sortie : v^{n+theta} (composante 0) pour l'energie / le diagnostic
  Real th_dt;          ///< theta * dt (w = th_dt * B_z, et facteur du gradient)
  Real half_idr, half_idth;  ///< 1/(2 dr), 1/(2 dtheta)
  Real r_min, dr;      ///< pour r_cell(i) (metrique azimutale 1/r)
  int c_rho, c_mx, c_my;     ///< composantes Density / MomentumX (radial) / MomentumY (azimutal)
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real gr = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idr;             // d_r phi
    const Real gt = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * (half_idth / ri);     // (1/r) d_theta phi
    const Real rhsr = vr(i, j, 0) - th_dt * gr;  // (v^n - theta dt grad_polar phi)_r
    const Real rhst = vt(i, j, 0) - th_dt * gt;
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    Real nr, nt;
    le.apply_Binv(rhsr, rhst, nr, nt);  // v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi)
    nvr(i, j, 0) = nr;
    nvt(i, j, 0) = nt;
    const Real rho = st(i, j, c_rho);   // rho^n (gele dans la source)
    st(i, j, c_mx) = rho * nr;          // mom^{n+theta} = rho^n v^{n+theta}
    st(i, j, c_my) = rho * nt;
  }
};

/// Extrapolation lineaire d'un champ SCALAIRE du theta-stage au pas plein : f^{n+1} = f^n + (1/theta)
/// (f^{n+theta} - f^n). theta = 1 -> identite. Foncteur NOMME device-clean. (Local au header polaire
/// pour ne pas dependre du chemin cartesien.)
struct PolarSchurExtrapolateScalarKernel {
  ConstArray4 f_n;     ///< f^n
  Array4 f;            ///< IN: f^{n+theta} ; OUT: f^{n+1}
  Real inv_theta;      ///< 1 / theta
  ADC_HD void operator()(int i, int j) const {
    f(i, j, 0) = f_n(i, j, 0) + inv_theta * (f(i, j, 0) - f_n(i, j, 0));
  }
};

/// Extrapolation lineaire de la VITESSE (vr, vtheta) du theta-stage au pas plein, puis recompose
/// mom = rho^n v^{n+1}. rho gelee. Foncteur NOMME device-clean.
struct PolarSchurExtrapolateVelocityKernel {
  ConstArray4 vr_n, vt_n;  ///< v^n
  Array4 vr, vt;           ///< IN: v^{n+theta} ; OUT: v^{n+1}
  Array4 st;               ///< etat (ECRITURE mr, mtheta ; LECTURE rho)
  Real inv_theta;          ///< 1 / theta
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real nr = vr_n(i, j, 0) + inv_theta * (vr(i, j, 0) - vr_n(i, j, 0));
    const Real nt = vt_n(i, j, 0) + inv_theta * (vt(i, j, 0) - vt_n(i, j, 0));
    vr(i, j, 0) = nr;
    vt(i, j, 0) = nt;
    const Real rho = st(i, j, c_rho);
    st(i, j, c_mx) = rho * nr;
    st(i, j, c_my) = rho * nt;
  }
};

/// Mise a jour de l'energie : E^{n+1} = E^n + (1/2) rho^n (|v^{n+1}|^2 - |v^n|^2). Applique seulement
/// quand le role Energy est present (garde cote hote). Foncteur NOMME device-clean.
struct PolarSchurEnergyKernel {
  ConstArray4 vr_n, vt_n;  ///< v^n
  ConstArray4 vr, vt;      ///< v^{n+1}
  Array4 st;               ///< etat (ECRITURE E ; LECTURE rho)
  int c_rho, c_E;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real ke_old = Real(0.5) * rho * (vr_n(i, j, 0) * vr_n(i, j, 0) + vt_n(i, j, 0) * vt_n(i, j, 0));
    const Real ke_new = Real(0.5) * rho * (vr(i, j, 0) * vr(i, j, 0) + vt(i, j, 0) * vt(i, j, 0));
    st(i, j, c_E) += ke_new - ke_old;
  }
};

/// Extrait la vitesse v = (mr, mtheta) / rho de l'etat vers deux champs scalaires vr, vt. rho = 0 ->
/// vitesse 0 (garde-fou ; la source gele rho > 0). Foncteur NOMME device-clean.
struct PolarExtractVelocityKernel {
  ConstArray4 st;
  Array4 vr, vt;
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv = rho != Real(0) ? Real(1) / rho : Real(0);
    vr(i, j, 0) = st(i, j, c_mx) * inv;
    vt(i, j, 0) = st(i, j, c_my) * inv;
  }
};

/// Copie le champ B_z (canal aux) vers un MultiFab scalaire interne. Foncteur NOMME device-clean.
struct PolarCopyBzKernel {
  ConstArray4 bz_src;
  Array4 bz_dst;
  int c_bz;
  ADC_HD void operator()(int i, int j) const { bz_dst(i, j, 0) = bz_src(i, j, c_bz); }
};

/// dst <- src (composante 0). Foncteur NOMME device-clean (local : le header polaire ne depend pas de
/// geometric_mg.hpp, comme polar_tensor_operator.hpp).
struct PolarSchurCopyComp0Kernel {
  Array4 d;
  ConstArray4 s;
  ADC_HD void operator()(int i, int j) const { d(i, j, 0) = s(i, j, 0); }
};

}  // namespace detail

/// ETAGE SOURCE condense par Schur en geometrie POLAIRE, AUTONOME (transport gele), GENERIQUE sur tout
/// bloc fluide polaire qui expose les roles Density / MomentumX (radial) / MomentumY (azimutal)
/// (+ Energy optionnel). Pendant polaire de CondensedSchurSourceStepper ; chemin SEPARE, le cartesien
/// reste BIT-IDENTIQUE.
///
/// CONVENTION DE ROLES EN POLAIRE : la quantite de mouvement RADIALE (rho v_r) porte le role MomentumX,
/// la quantite de mouvement AZIMUTALE (rho v_theta) porte le role MomentumY (cf. IsothermalFluxPolar :
/// composante 1 = radiale, composante 2 = azimutale, base locale orthonormee (e_r, e_theta)).
///
/// CYCLE DE VIE : construit UNE fois sur (PolarGeometry + BoxArray) fixe ; alloue ses tampons a la
/// construction ; step() les REUTILISE. theta/dt peuvent changer entre appels.
class PolarCondensedSchurSourceStepper {
 public:
  /// @p vars  : descripteur du bloc fluide ; DOIT exposer Density / MomentumX / MomentumY (Energy
  ///            optionnel). Contrat valide ICI (hote), avant tout kernel.
  /// @p geom  : geometrie POLAIRE (anneau) ; @p ba : boite UNIQUE ; @p bcPhi : CL radiale du potentiel
  ///            (xlo/xhi : Dirichlet ou Foextrap ; theta toujours periodique cote solveur).
  /// @p alpha : constante de couplage electrostatique.
  /// @p precond : preconditionneur du PolarTensorKrylovSolver (RadialLine par defaut).
  PolarCondensedSchurSourceStepper(const VariableSet& vars, const PolarGeometry& geom,
                                   const BoxArray& ba, const BCRec& bcPhi, Real alpha,
                                   PolarPrecond precond = PolarPrecond::RadialLine)
      : vars_(vars),
        c_rho_(vars.index_of(VariableRole::Density)),
        c_mx_(vars.index_of(VariableRole::MomentumX)),
        c_my_(vars.index_of(VariableRole::MomentumY)),
        c_E_(vars.index_of(VariableRole::Energy)),
        alpha_(alpha),
        geom_(geom),
        bcPhi_(bcPhi),
        precond_(precond),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        // coefficients du tenseur condense A = I + c rho B^{-1} (1 ghost : faces de l'operateur)
        a_rr_(ba, dm_, 1, 1), a_tt_(ba, dm_, 1, 1), a_rt_(ba, dm_, 1, 1), a_tr_(ba, dm_, 1, 1),
        // tampons du RHS condense
        lap_(ba, dm_, 1, 0), rhs_(ba, dm_, 1, 0),
        fr_(ba, dm_, 1, 1), ft_(ba, dm_, 1, 1),
        // un MultiFab a 1 (A=I) pour le Lap_polar scalaire du RHS
        one_rr_(ba, dm_, 1, 1), one_tt_(ba, dm_, 1, 1),
        bz_(ba, dm_, 1, 1),
        phi_n_(ba, dm_, 1, 1),
        vr_n_(ba, dm_, 1, 0), vt_n_(ba, dm_, 1, 0),
        vr_t_(ba, dm_, 1, 0), vt_t_(ba, dm_, 1, 0) {
    // MULTI-RANG MPI (decoupage theta seul) : plus de garde-fou mono-rang. La contrainte de layout
    // (chaque box couvre la plage radiale complete, exigee par le preconditionneur RadialLine du solve
    // elliptique) est verifiee par le PolarTensorKrylovSolver construit dans step() (check_radial_columns)
    // -> une erreur claire est levee sur tous les rangs si le decoupage coupe r. Mono-rang / boite unique :
    // chemin inchange (le check passe trivialement, all_reduce = identite en serie).
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "PolarCondensedSchurSourceStepper : le bloc fluide doit exposer les roles Density, MomentumX "
          "(radial) et MomentumY (azimutal).");
    one_rr_.set_val(Real(1));
    one_tt_.set_val(Real(1));
  }

  /// true si le modele porte un role Energy (mise a jour d'energie active).
  bool has_energy() const { return c_E_ >= 0; }

  /// ETAGE SOURCE condense POLAIRE, IN-PLACE sur @p state et @p phi.
  ///   @p state : etat fluide (rho, mom_r, mom_theta [, E]) ; ECRIT mom (+ E) ; rho GELEE.
  ///   @p phi   : potentiel ; ENTREE phi^n (warm start du solve) ; SORTIE phi^{n+1}.
  ///   @p bz_field : champ B_z (canal aux), composante @p c_bz lue au centre. theta/dt : theta-schema.
  void step(MultiFab& state, MultiFab& phi, const MultiFab& bz_field, int c_bz, Real theta, Real dt) {
    const Real th_dt = theta * dt;
    const Real c = theta * theta * dt * dt * alpha_;  // c = theta^2 dt^2 alpha
    const Real g = theta * dt * alpha_;               // theta dt alpha
    const Real dr = geom_.dr();
    const Real dth = geom_.dtheta();
    const Real half_idr = Real(1) / (Real(2) * dr);
    const Real half_idth = Real(1) / (Real(2) * dth);

    // -1) figer phi^n (extrapolation finale).
    copy_comp0(phi_n_, phi);

    // 0) extraire v^n = (mr, mtheta)/rho et copier B_z dans le tampon interne.
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      for_each_cell(state.box(li),
                    detail::PolarExtractVelocityKernel{s, vr_n_.fab(li).array(), vt_n_.fab(li).array(),
                                                       c_rho_, c_mx_, c_my_});
      for_each_cell(bz_.box(li),
                    detail::PolarCopyBzKernel{bz_field.fab(li).const_array(), bz_.fab(li).array(), c_bz});
    }
    const BCRec ebc = coeff_bc(bcPhi_);
    device_fence();
    fill_ghosts(bz_, geom_.domain, ebc);

    // 1a) ASSEMBLER les coefficients A = I + c rho B^{-1} au centre (4 champs).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz_.fab(li).const_array();
      for_each_cell(a_rr_.box(li),
                    detail::PolarSchurOperatorCoeffKernel{s, b, a_rr_.fab(li).array(),
                                                          a_tt_.fab(li).array(), a_rt_.fab(li).array(),
                                                          a_tr_.fab(li).array(), c, th_dt, c_rho_});
    }
    // ghosts des coefficients (la moyenne de face de l'operateur lit le voisin a +-1) : periodique
    // theta, radial Foextrap. PolarTensorKrylovSolver::set_coefficients remplit aussi ces ghosts, mais
    // on les remplit ici par robustesse (assemble_rhs lit fr/ft -> coherence).
    device_fence();
    fill_ghosts(a_rr_, geom_.domain, ebc);
    fill_ghosts(a_tt_, geom_.domain, ebc);
    fill_ghosts(a_rt_, geom_.domain, ebc);
    fill_ghosts(a_tr_, geom_.domain, ebc);

    // 1b) ASSEMBLER le second membre condense rhs_polar = Lap_polar phi^n + g div_polar(rho B^{-1} v^n).
    //     Lap_polar phi^n : apply_polar_tensor avec A = I (coefficients a 1), MEME stencil que le solve.
    device_fence();
    fill_ghosts(phi, geom_.domain, bcPhi_);  // ghosts de phi^n pour le Laplacien de bord
    apply_polar_tensor(phi, geom_, lap_, &one_rr_, &one_tt_, nullptr, nullptr);  // lap_ = Lap_polar phi^n
    // flux explicite F = B^{-1}(mr, mtheta) au centre (1 ghost pour la div centree).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz_.fab(li).const_array();
      for_each_cell(fr_.box(li),
                    detail::PolarSchurExplicitFluxKernel{s, b, fr_.fab(li).array(),
                                                         ft_.fab(li).array(), th_dt, c_mx_, c_my_});
    }
    device_fence();
    fill_ghosts(fr_, geom_.domain, ebc);  // div centree lit F(i+-1), F(j+-1)
    fill_ghosts(ft_, geom_.domain, ebc);
    // rhs_polar = Lap_polar phi^n + g div_polar F (signe L_int = -rhs_schur).
    for (int li = 0; li < rhs_.local_size(); ++li)
      for_each_cell(rhs_.box(li),
                    detail::PolarSchurRhsAssembleKernel{lap_.fab(li).const_array(),
                                                        fr_.fab(li).const_array(),
                                                        ft_.fab(li).const_array(), rhs_.fab(li).array(),
                                                        g, half_idr, half_idth, geom_.r_min, dr});

    // 2) RESOUDRE L_int(phi) = div(A grad phi) = rhs_polar via PolarTensorKrylovSolver.
    PolarTensorKrylovSolver kry(geom_, ba_, bcPhi_, precond_);
    const bool cross = (c != Real(0));  // termes croises non triviaux des que c != 0 (B_z != 0 -> binv_12 != 0)
    if (cross)
      kry.set_coefficients(&a_rr_, &a_tt_, &a_rt_, &a_tr_);
    else
      kry.set_coefficients(&a_rr_, &a_tt_);
    copy_comp0(kry.phi(), phi);  // warm start : phi^n -> kry.phi()
    copy_comp0(kry.rhs(), rhs_);
    last_result_ = kry.solve(Real(1e-10), 600);
    copy_comp0(phi, kry.phi());  // phi <- phi^{n+theta}

    // 3) RECONSTRUIRE v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}) ; mom = rho v.
    device_fence();
    fill_ghosts(phi, geom_.domain, bcPhi_);  // grad_polar centre lit phi(i+-1), phi(j+-1)
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::PolarSchurReconstructKernel{
                        phi.fab(li).const_array(), vr_n_.fab(li).const_array(),
                        vt_n_.fab(li).const_array(), bz_.fab(li).const_array(), state.fab(li).array(),
                        vr_t_.fab(li).array(), vt_t_.fab(li).array(), th_dt, half_idr, half_idth,
                        geom_.r_min, dr, c_rho_, c_mx_, c_my_});
    // vr_t_/vt_t_ portent v^{n+theta}.

    // 5) EXTRAPOLER phi et v du theta-stage au pas plein : f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n).
    const Real inv_theta = Real(1) / theta;
    for (int li = 0; li < phi.local_size(); ++li)
      for_each_cell(phi.box(li),
                    detail::PolarSchurExtrapolateScalarKernel{phi_n_.fab(li).const_array(),
                                                              phi.fab(li).array(), inv_theta});
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::PolarSchurExtrapolateVelocityKernel{
                        vr_n_.fab(li).const_array(), vt_n_.fab(li).const_array(),
                        vr_t_.fab(li).array(), vt_t_.fab(li).array(), state.fab(li).array(),
                        inv_theta, c_rho_, c_mx_, c_my_});

    // 4) ENERGIE (si role present) : E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2).
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li),
                      detail::PolarSchurEnergyKernel{vr_n_.fab(li).const_array(),
                                                     vt_n_.fab(li).const_array(),
                                                     vr_t_.fab(li).const_array(),
                                                     vt_t_.fab(li).const_array(), state.fab(li).array(),
                                                     c_rho_, c_E_});

    // 6) REMPLIR les ghosts de l'etat et du potentiel avant de rendre la main.
    device_fence();
    fill_ghosts(state, geom_.domain, bcU_default());
    fill_ghosts(phi, geom_.domain, bcPhi_);
  }

  /// Diagnostic du dernier solve (iterations BiCGStab, residu relatif, convergence).
  const PolarKrylovResult& last_solve() const { return last_result_; }

  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }
  int energy_comp() const { return c_E_; }

 private:
  /// CL par defaut de l'etat pour le remplissage final des ghosts : periodique theta conserve, radial
  /// Foextrap. L'etage source est local en espace ; la CL de l'etat n'influe que sur les ghosts publies.
  BCRec bcU_default() const { return coeff_bc(bcPhi_); }

  /// CL des champs de coefficient / flux : periodique theta conserve, bord physique radial -> Foextrap.
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo); b.xhi = fo(bc.xhi);
    b.ylo = BCType::Periodic; b.yhi = BCType::Periodic;  // theta toujours periodique
    return b;
  }

  /// dst <- src (composante 0, cellules valides). Foncteur NOMME local (decouple du cartesien).
  void copy_comp0(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::PolarSchurCopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }

  VariableSet vars_;
  int c_rho_, c_mx_, c_my_, c_E_;
  Real alpha_;
  PolarGeometry geom_;
  BCRec bcPhi_;
  PolarPrecond precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab a_rr_, a_tt_, a_rt_, a_tr_;  ///< coefficients A = I + c rho B^{-1} (reutilises)
  MultiFab lap_, rhs_;                  ///< Lap_polar phi^n + RHS condense
  MultiFab fr_, ft_;                    ///< flux explicite F = B^{-1}(mr, mtheta) au centre
  MultiFab one_rr_, one_tt_;            ///< champs a 1 (A=I) pour le Lap_polar du RHS
  MultiFab bz_;                         ///< B_z au centre (tampon interne, 1 ghost)
  MultiFab phi_n_;                      ///< phi^n fige (extrapolation finale)
  MultiFab vr_n_, vt_n_;                ///< v^n (extrait au debut de step)
  MultiFab vr_t_, vt_t_;                ///< v^{n+theta} puis v^{n+1}
  PolarKrylovResult last_result_;       ///< diagnostic du dernier solve
};

}  // namespace adc
