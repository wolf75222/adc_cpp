#pragma once

#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/coupling/schur_condensation.hpp>            // ElectrostaticLorentzCondensation (batisseur #124)
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>          // operateur + preconditionneur (#120)
#include <adc/numerics/elliptic/krylov_solver.hpp>          // TensorKrylovSolver (BiCGStab, #122)
#include <adc/numerics/lorentz_eliminator.hpp>              // B^{-1} ferme (#118)

#include <stdexcept>

/// @file
/// @brief CondensedSchurSourceStepper : ETAGE SOURCE condense par Schur (niveau 4 de
///        docs/SCHUR_CONDENSATION_DESIGN.md) pour la source implicite couplee potentiel / vitesse /
///        Lorentz (Hoffart et al., arXiv:2510.11808). C'est un etage SOURCE AUTONOME (transport gele) :
///        il NE remplace PAS le chemin System::step et ne s'y branche PAS (cablage facade = PR5). Il
///        COMPOSE les briques deja sur master :
///          - ElectrostaticLorentzCondensation (schur_condensation.hpp, #124) : assemble A_op = I + c
///            rho B^{-1} (eps_x/eps_y diag, a_xy/a_yx croises) + le RHS condense ;
///          - TensorKrylovSolver (krylov_solver.hpp, #122) : BiCGStab matrice-libre preconditionne par
///            le V-cycle GeometricMG sur la partie SYMETRIQUE (l'operateur est NON auto-adjoint des que
///            B_z != 0) ;
///          - LorentzEliminator (lorentz_eliminator.hpp, #118) : B^{-1} ferme pour la reconstruction.
///
/// SEQUENCE (docs/SCHUR_CONDENSATION_DESIGN.md section 5, niveau 4), une fois par etage source :
///   1. ASSEMBLER A_op = I + c rho^n B^{-1} (c = theta^2 dt^2 alpha) et le RHS condense
///      rhs_schur = -Lap phi^n - theta dt alpha div(rho^n B^{-1} v^n)         [#124]
///   2. RESOUDRE L_schur(phi^{n+theta}) = rhs_schur, L_schur = -div(A_op grad phi)              [#122]
///   3. RECONSTRUIRE v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}), mom = rho^n v     [#118]
///   4. ENERGIE (si role Energy present) : voir CHOIX ci-dessous
///   5. EXTRAPOLER U^{n+theta} -> U^{n+1} : voir CHOIX ci-dessous
///   6. REMPLIR les ghosts (fill_boundary, halos MPI) et publier phi^{n+1} + grad phi^{n+1}.
///
/// CONVENTION DE SIGNE DU SOLVE (a lire avant toute modification). Le batisseur #124 ecrit l'etage sous
///   la forme L_schur(phi) = rhs_schur avec L_schur(phi) = -div(A_op grad phi) (cf. l'entete de
///   schur_condensation.hpp). Le TensorKrylovSolver, lui, resout L_int(phi) = rhs_kry avec la convention
///   poisson_operator L_int = div(A grad phi) - kappa phi (kappa = 0 ici). Donc L_schur = -L_int, et
///   resoudre L_schur(phi) = rhs_schur revient a resoudre L_int(phi) = -rhs_schur. On passe donc
///   rhs_kry = -rhs_schur au solveur (champ rhs() negue). Garde-fou : c = 0 et B_z = 0 -> A = I, le
///   solve devient Lap phi^{n+theta} = Lap phi^n -> phi^{n+theta} = phi^n (a la constante pres), et la
///   reconstruction degenere en la poussee electrostatique explicite v^{n+theta} = v^n - theta dt grad
///   phi^n (cas B = 0). Le batisseur teste deja ce garde-fou cote assemblage (test C2).
///
/// CHOIX EXTRAPOLATION (etape 5), DOCUMENTE. theta-stage -> n+1 par EXTRAPOLATION LINEAIRE :
///       U^{n+1} = U^n + (1/theta) (U^{n+theta} - U^n).
///   C'est la forme citee par docs/SCHUR_CONDENSATION_DESIGN.md (section 5, niveau 4, etape 5). En
///   theta = 1 (implicite pur) l'etage atterrit DIRECTEMENT en n+1 et l'extrapolation est l'identite
///   (facteur 1/theta = 1). En theta = 1/2 (Crank-Nicolson) le facteur vaut 2 : l'etat a mi-pas est
///   extrapole au pas plein. Appliquee a phi et v (et donc mom = rho v) ; rho est GELEE dans la source
///   (tout son transport est dans l'etage hyperbolique), donc rho^{n+1} = rho^n.
///
/// CHOIX ENERGIE (etape 4), DOCUMENTE. La source d_t v = -grad phi + v x Omega fait un TRAVAIL sur le
///   fluide via la force electrostatique -grad phi (la rotation de Lorentz v x Omega ne travaille pas :
///   |B v|^2 = |v|^2 a une dilatation det pres ; en l'absence de dissipation l'energie cinetique ne
///   change que par le champ). rho etant gelee, l'energie INTERNE est conservee dans la source ; seule
///   l'energie CINETIQUE varie. On met donc a jour l'energie totale par l'INCREMENT d'energie cinetique :
///       E^{n+1} = E^n + (1/2) rho^n ( |v^{n+1}|^2 - |v^n|^2 ).
///   Coherent avec une energie totale E = E_interne + (1/2) rho |v|^2 ou seul le terme cinetique bouge
///   sous la source. Applique UNIQUEMENT si le role Energy est present ; sinon l'energie est ignoree
///   (modele isotherme / sans equation d'energie), comme le batisseur #124 qui ignore deja Energy.
///
/// DEVICE / MPI (docs/GPU_RUNTIME_PORT.md). Tous les kernels de cet etage sont des FONCTEURS NOMMES
///   device-clean (pas de lambda etendue premiere-instanciee cross-TU, limite nvcc #64/#97). Les
///   tampons MultiFab (operateur A_op, RHS, phi^n, grad phi, v^n) sont ALLOUES UNE FOIS a la
///   construction et REUTILISES a chaque appel step() (la conception #124 signale le cout d'une
///   reallocation par pas). Toutes les boucles iterent sur local_size() : un rang SANS box
///   (local_size()==0, decoupage MPI) n'execute aucun kernel et ne deref jamais fab(0). Le solve de
///   Krylov est COLLECTIF (dot/all_reduce sur tous les rangs, y compris vides) : pas d'interblocage.

namespace adc {

namespace detail {

/// Reconstruit v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) et ecrit mom = rho^n v^{n+theta}
/// dans l'etat. grad phi est la difference CENTREE (meme discretisation que la divergence du RHS
/// condense #124 et que field_postprocess) : la relation implicite B v = v^n - theta dt grad phi tient
/// alors a la precision du SOLVE, terme a terme. rho est lue dans l'etat (role Density) et GELEE.
/// Foncteur NOMME device-clean : capture des handles Array4 (POD) + scalaires.
struct SchurReconstructKernel {
  ConstArray4 phi;     ///< phi^{n+theta} (ghosts remplis : grad centre lit i+-1, j+-1)
  ConstArray4 vx, vy;  ///< v^n (composantes 0 de leurs MultiFab : vitesse, PAS quantite de mouvement)
  ConstArray4 bz;      ///< champ B_z au centre
  Array4 st;           ///< etat fluide (ECRITURE de mx, my ; LECTURE de rho)
  Array4 nvx, nvy;     ///< sortie : v^{n+theta} (composante 0) pour l'energie / le diagnostic
  Real th_dt;          ///< theta * dt (w = th_dt * B_z, et facteur du gradient)
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy) (gradient centre)
  int c_rho, c_mx, c_my;    ///< composantes Density / MomentumX / MomentumY
  ADC_HD void operator()(int i, int j) const {
    const Real gx = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idx;  // d_x phi^{n+theta}
    const Real gy = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * half_idy;  // d_y phi^{n+theta}
    const Real rhsx = vx(i, j, 0) - th_dt * gx;  // (v^n - theta dt grad phi)_x
    const Real rhsy = vy(i, j, 0) - th_dt * gy;
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    Real nx, ny;
    le.apply_Binv(rhsx, rhsy, nx, ny);  // v^{n+theta} = B^{-1}(v^n - theta dt grad phi)
    nvx(i, j, 0) = nx;
    nvy(i, j, 0) = ny;
    const Real rho = st(i, j, c_rho);   // rho^n (gele dans la source)
    st(i, j, c_mx) = rho * nx;          // mom^{n+theta} = rho^n v^{n+theta}
    st(i, j, c_my) = rho * ny;
  }
};

/// Extrapolation lineaire d'un champ SCALAIRE du theta-stage au pas plein : f^{n+1} = f^n + (1/theta)
/// (f^{n+theta} - f^n). theta = 1 -> identite (inv_theta = 1). Foncteur NOMME device-clean.
struct SchurExtrapolateScalarKernel {
  ConstArray4 f_n;     ///< f^n
  Array4 f;            ///< IN: f^{n+theta} ; OUT: f^{n+1}
  Real inv_theta;      ///< 1 / theta
  ADC_HD void operator()(int i, int j) const {
    f(i, j, 0) = f_n(i, j, 0) + inv_theta * (f(i, j, 0) - f_n(i, j, 0));
  }
};

/// Extrapolation lineaire de la VITESSE (vx, vy) du theta-stage au pas plein, puis recompose
/// mom = rho^n v^{n+1} dans l'etat. rho gelee. Foncteur NOMME device-clean.
struct SchurExtrapolateVelocityKernel {
  ConstArray4 vx_n, vy_n;  ///< v^n
  Array4 vx, vy;           ///< IN: v^{n+theta} ; OUT: v^{n+1}
  Array4 st;               ///< etat (ECRITURE mx, my ; LECTURE rho)
  Real inv_theta;          ///< 1 / theta
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real nx = vx_n(i, j, 0) + inv_theta * (vx(i, j, 0) - vx_n(i, j, 0));
    const Real ny = vy_n(i, j, 0) + inv_theta * (vy(i, j, 0) - vy_n(i, j, 0));
    vx(i, j, 0) = nx;
    vy(i, j, 0) = ny;
    const Real rho = st(i, j, c_rho);
    st(i, j, c_mx) = rho * nx;
    st(i, j, c_my) = rho * ny;
  }
};

/// Mise a jour de l'energie : E^{n+1} = E^n + (1/2) rho^n (|v^{n+1}|^2 - |v^n|^2). Foncteur NOMME
/// device-clean. Applique seulement quand le role Energy est present (garde cote hote).
struct SchurEnergyKernel {
  ConstArray4 vx_n, vy_n;  ///< v^n
  ConstArray4 vx, vy;      ///< v^{n+1}
  Array4 st;               ///< etat (ECRITURE E ; LECTURE rho)
  int c_rho, c_E;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real ke_old = Real(0.5) * rho * (vx_n(i, j, 0) * vx_n(i, j, 0) + vy_n(i, j, 0) * vy_n(i, j, 0));
    const Real ke_new = Real(0.5) * rho * (vx(i, j, 0) * vx(i, j, 0) + vy(i, j, 0) * vy(i, j, 0));
    st(i, j, c_E) += ke_new - ke_old;
  }
};

/// Extrait la vitesse v = (mx, my) / rho de l'etat (role Density / MomentumX / MomentumY) vers deux
/// champs scalaires vx, vy. rho = 0 -> vitesse 0 (garde-fou ; la source gele rho > 0 en pratique).
/// Foncteur NOMME device-clean.
struct ExtractVelocityKernel {
  ConstArray4 st;
  Array4 vx, vy;
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv = rho != Real(0) ? Real(1) / rho : Real(0);
    vx(i, j, 0) = st(i, j, c_mx) * inv;
    vy(i, j, 0) = st(i, j, c_my) * inv;
  }
};

/// Copie le champ B_z (canal aux) vers un MultiFab scalaire interne (0 ghost suffit, lu en (i,j)).
/// Foncteur NOMME device-clean.
struct CopyBzKernel {
  ConstArray4 bz_src;
  Array4 bz_dst;
  int c_bz;
  ADC_HD void operator()(int i, int j) const { bz_dst(i, j, 0) = bz_src(i, j, c_bz); }
};

}  // namespace detail

/// ETAGE SOURCE condense par Schur, AUTONOME (transport gele), GENERIQUE sur tout bloc fluide qui
/// expose les roles Density / MomentumX / MomentumY (+ Energy optionnel). Ne nomme aucun scenario :
/// un modele anneau polaire a derive ExB n'est qu'UN client, un modele a deux especes magnetisees en serait un autre.
///
/// CYCLE DE VIE : construit UNE fois sur un layout (BoxArray + DistributionMapping + Geometry) fixe ;
/// alloue tous ses tampons a la construction (operateur GeometricMG + preconditionneur + champs A_op,
/// RHS, phi^n, v^n, grad, + le solveur de Krylov kry_ et sa poignee de MultiFab). step() les REUTILISE
/// sans reallocation : kry_.solve() recalcule la TOTALITE de son etat par-solve (prepare_solve, r0,
/// directions de descente), donc le tampon persistant est bit-identique a l'ancienne construction d'un
/// TensorKrylovSolver par appel. theta/dt peuvent changer entre appels (parametres de step()).
class CondensedSchurSourceStepper {
 public:
  /// @p vars  : descripteur du bloc fluide ; DOIT exposer Density / MomentumX / MomentumY (Energy
  ///            optionnel). Le contrat est valide ICI (hote), avant tout kernel.
  /// @p geom  : geometrie (cartesienne) ; @p ba : decoupage ; @p bcPhi : CL du potentiel phi.
  /// @p alpha : constante de couplage electrostatique.
  /// @p n_precond_vcycles : N V-cycles MG par application du preconditionneur BiCGStab (1 ou 2).
  CondensedSchurSourceStepper(const VariableSet& vars, const Geometry& geom, const BoxArray& ba,
                              const BCRec& bcPhi, Real alpha, int n_precond_vcycles = 1)
      : CondensedSchurSourceStepper(vars, vars.index_of(VariableRole::Density),
                                    vars.index_of(VariableRole::MomentumX),
                                    vars.index_of(VariableRole::MomentumY),
                                    vars.index_of(VariableRole::Energy), geom, ba, bcPhi, alpha,
                                    n_precond_vcycles) {}

  /// Variante a COMPOSANTES EXPLICITES (audit 2026-06, vague 2 : roles/champs transportes dans
  /// l'ABI). L'appelant DESIGNE les composantes (rho, mx, my[, E]) au lieu de laisser le stepper
  /// resoudre les roles canoniques -- pour un bloc dont le descripteur n'expose pas Density/
  /// MomentumX/MomentumY (noms libres, roles Custom) ou range ses champs ailleurs. @p c_E < 0 =
  /// pas d'energie. Le ctor canonique ci-dessus DELEGUE ici (resolution par roles inchangee,
  /// bit-identique).
  CondensedSchurSourceStepper(const VariableSet& vars, int c_rho, int c_mx, int c_my, int c_E,
                              const Geometry& geom, const BoxArray& ba, const BCRec& bcPhi,
                              Real alpha, int n_precond_vcycles = 1)
      : vars_(vars),
        c_rho_(c_rho),
        c_mx_(c_mx),
        c_my_(c_my),
        c_E_(c_E),
        alpha_(alpha),
        geom_(geom),
        bcPhi_(bcPhi),
        n_precond_(n_precond_vcycles),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        // operateur PLEIN (matvec BiCGStab) + preconditionneur SYMETRIQUE (sans termes croises).
        op_(geom, ba, bcPhi),
        precond_(geom, ba, bcPhi),
        // tampons reutilises (alloues UNE fois) :
        eps_x_(ba, dm_, 1, 1), eps_y_(ba, dm_, 1, 1), a_xy_(ba, dm_, 1, 1), a_yx_(ba, dm_, 1, 1),
        rhs_schur_(ba, dm_, 1, 0),
        bz_(ba, dm_, 1, 1),
        vx_n_(ba, dm_, 1, 0), vy_n_(ba, dm_, 1, 0),
        vx_t_(ba, dm_, 1, 0), vy_t_(ba, dm_, 1, 0),
        phi_n_(ba, dm_, 1, 1),
        kry_(op_, precond_, n_precond_) {
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "CondensedSchurSourceStepper : le bloc fluide doit exposer les roles Density, MomentumX "
          "et MomentumY (VariableSet.roles renseignes).");
  }

  /// true si le modele porte un role Energy (mise a jour d'energie active).
  bool has_energy() const { return c_E_ >= 0; }

  /// ETAGE SOURCE condense, IN-PLACE sur @p state et @p phi.
  ///   @p state : etat fluide (rho, mom_x, mom_y [, E]) ; ECRIT mom (+ E) ; rho GELEE.
  ///   @p phi   : potentiel ; ENTREE phi^n (warm start du solve) ; SORTIE phi^{n+1}.
  ///   @p bz_field : champ B_z (canal aux), composante @p c_bz lue au centre. theta/dt : theta-schema.
  /// Pas de transport : c'est l'etage SOURCE seul (Hoffart et al., operateur (2)-(3) du splitting).
  void step(MultiFab& state, MultiFab& phi, const MultiFab& bz_field, int c_bz, Real theta, Real dt) {
    const Real th_dt = theta * dt;

    // -1) figer phi^n (pour l'extrapolation finale ; phi() de op_ sera ecrase par le solve).
    copy_comp0(phi_n_, phi);

    // 0) extraire v^n = (mx, my)/rho et copier B_z dans le tampon interne (1 ghost rempli ensuite).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{s, vx_n_.fab(li).array(), vy_n_.fab(li).array(),
                                                  c_rho_, c_mx_, c_my_});
      for_each_cell(bz_.box(li),
                    detail::CopyBzKernel{bz_field.fab(li).const_array(), bz_.fab(li).array(), c_bz});
    }
    const BCRec ebc = coeff_bc(bcPhi_);
    fill_ghosts(bz_, geom_.domain, ebc);  // B_z lu en (i,j) seulement ici, mais 1 ghost = MPI-propre

    // 1) ASSEMBLER A_op = I + c rho B^{-1} + RHS condense (#124). phi voit ses ghosts remplis (CL).
    ElectrostaticLorentzCondensation builder(vars_, alpha_, theta, dt);
    builder.assemble_operator(state, bz_, geom_, bcPhi_, eps_x_, eps_y_, a_xy_, a_yx_);
    builder.assemble_rhs(phi, state, bz_, geom_, bcPhi_, rhs_schur_);

    // 2) configurer l'operateur PLEIN (op_) et le preconditionneur SYMETRIQUE (precond_), puis
    //    RESOUDRE L_int(phi) = -rhs_schur (cf. convention de signe de l'entete) par BiCGStab.
    op_.set_epsilon_anisotropic(eps_x_, eps_y_);
    op_.set_cross_terms(a_xy_, a_yx_);
    precond_.set_epsilon_anisotropic(eps_x_, eps_y_);  // partie symetrique : termes croises LARGUES.
    // warm start : phi^n -> op_.phi() ; rhs = -rhs_schur.
    copy_comp0(op_.phi(), phi);
    negate_into(op_.rhs(), rhs_schur_);
    // kry_ est un MEMBRE persistant (ses ~10 MultiFab sont alloues UNE fois a la construction, plus
    // par appel) : solve() recalcule la totalite de son etat par-solve (prepare_solve, r0, directions),
    // donc le resultat est BIT-IDENTIQUE a l'ancienne construction d'un TensorKrylovSolver local ici.
    // Tolerances configurables via set_krylov (audit 2026-06) ; defauts = 1e-10/400 historiques.
    last_result_ = kry_.solve(krylov_tol_, krylov_max_iters_);
    copy_comp0(phi, op_.phi());  // phi <- phi^{n+theta}

    // 3) RECONSTRUIRE v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) ; mom = rho v.
    device_fence();
    fill_ghosts(phi, geom_.domain, bcPhi_);  // grad centre lit phi(i+-1), phi(j+-1)
    const Real half_idx = Real(1) / (Real(2) * geom_.dx());
    const Real half_idy = Real(1) / (Real(2) * geom_.dy());
    for (int li = 0; li < state.local_size(); ++li) {
      for_each_cell(state.box(li),
                    detail::SchurReconstructKernel{
                        phi.fab(li).const_array(), vx_n_.fab(li).const_array(),
                        vy_n_.fab(li).const_array(), bz_.fab(li).const_array(),
                        state.fab(li).array(), vx_t_.fab(li).array(), vy_t_.fab(li).array(), th_dt,
                        half_idx, half_idy, c_rho_, c_mx_, c_my_});
    }
    // vx_t_/vy_t_ portent maintenant v^{n+theta}.

    // 5) EXTRAPOLER phi et v du theta-stage au pas plein : f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n).
    //    theta = 1 -> identite. phi^n est fige dans phi_n_ (etape -1), v^n dans vx_n_/vy_n_.
    const Real inv_theta = Real(1) / theta;
    for (int li = 0; li < phi.local_size(); ++li)
      for_each_cell(phi.box(li),
                    detail::SchurExtrapolateScalarKernel{phi_n_.fab(li).const_array(),
                                                         phi.fab(li).array(), inv_theta});
    // v + mom : extrapolation lineaire, recompose mom = rho v^{n+1}. vx_t_/vy_t_ portent ensuite v^{n+1}.
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::SchurExtrapolateVelocityKernel{
                        vx_n_.fab(li).const_array(), vy_n_.fab(li).const_array(),
                        vx_t_.fab(li).array(), vy_t_.fab(li).array(), state.fab(li).array(),
                        inv_theta, c_rho_, c_mx_, c_my_});

    // 4) ENERGIE (si role present) : E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2).
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li),
                      detail::SchurEnergyKernel{vx_n_.fab(li).const_array(),
                                                vy_n_.fab(li).const_array(),
                                                vx_t_.fab(li).const_array(),
                                                vy_t_.fab(li).const_array(), state.fab(li).array(),
                                                c_rho_, c_E_});

    // 6) REMPLIR les ghosts de l'etat et du potentiel (halos MPI / CL physiques) avant de rendre la main.
    device_fence();
    fill_ghosts(state, geom_.domain, bcU_default());
    fill_ghosts(phi, geom_.domain, bcPhi_);
  }

  /// Diagnostic du dernier solve (iterations BiCGStab, residu relatif, convergence).
  const KrylovResult& last_solve() const { return last_result_; }

  /// Tolerance / budget d'iterations du solve Krylov de l'etage (BiCGStab). DEFAUTS = constantes
  /// historiques (1e-10, 400), rendues configurables par l'audit 2026-06 (constantes numeriques
  /// explicites). @throws std::invalid_argument hors domaine.
  void set_krylov(Real tol, int max_iters) {
    if (!(tol > Real(0)) || max_iters < 1)
      throw std::invalid_argument("CondensedSchurSourceStepper::set_krylov : tol > 0, max_iters >= 1");
    krylov_tol_ = tol;
    krylov_max_iters_ = max_iters;
  }

  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }
  int energy_comp() const { return c_E_; }

 private:
  /// CL par defaut de l'etat pour le remplissage final des ghosts : Foextrap partout sauf periodique
  /// conserve (gradient-nul, coherent avec le coeff_bc des coefficients). L'etage source est local en
  /// espace ; la CL de l'etat n'influe que sur les ghosts publies pour l'etage transport suivant.
  BCRec bcU_default() const { return coeff_bc(bcPhi_); }

  /// CL des champs de coefficient : periodique conserve, bord physique -> gradient-nul (Foextrap).
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo); b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo); b.yhi = fo(bc.yhi);
    return b;
  }

  /// dst <- src (composante 0, cellules valides). Foncteur NOMME (reutilise CopyComp0Kernel #93).
  void copy_comp0(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }

  /// dst <- -src (composante 0). Foncteur NOMME (reutilise NegateKernel de schur_condensation.hpp).
  void negate_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::NegateKernel{src.fab(li).const_array(), dst.fab(li).array()});
  }

  VariableSet vars_;               ///< descripteur du bloc fluide (roles) passe au batisseur #124
  int c_rho_, c_mx_, c_my_, c_E_;  ///< composantes des roles (c_E_ < 0 si Energy absent)
  Real alpha_;
  Geometry geom_;
  BCRec bcPhi_;
  int n_precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  GeometricMG op_, precond_;  ///< operateur plein (matvec) + preconditionneur symetrique
  MultiFab eps_x_, eps_y_, a_xy_, a_yx_;  ///< coefficients A_op = I + c rho B^{-1} (reutilises)
  MultiFab rhs_schur_;        ///< RHS condense (signe schur ; negue avant le solve)
  MultiFab bz_;               ///< B_z au centre (tampon interne, 1 ghost)
  MultiFab vx_n_, vy_n_;      ///< v^n (extrait au debut de step)
  MultiFab vx_t_, vy_t_;      ///< v^{n+theta} puis v^{n+1} (reconstruction + extrapolation)
  MultiFab phi_n_;            ///< phi^n fige (extrapolation) ; alloue au premier advance_source
  KrylovResult last_result_;  ///< diagnostic du dernier solve
  Real krylov_tol_ = Real(1e-10);  ///< tolerance du solve (defaut historique, cf. set_krylov)
  int krylov_max_iters_ = 400;     ///< budget d'iterations (defaut historique, cf. set_krylov)
  // Solveur de Krylov PERSISTANT (BiCGStab + precond MG). Alloue ses tampons (r/rhat/p/v/s/t/phat/
  // shat + offsets de CL) UNE fois ; step() reutilise kry_ sans reallocation. DOIT etre declare APRES
  // op_/precond_/n_precond_ (qu'il reference) : ordre d'init (apres eux) et de destruction (avant eux)
  // corrects. Bit-identique a l'ancienne construction par appel (solve() reinitialise tout son etat).
  TensorKrylovSolver kry_;
};

}  // namespace adc
