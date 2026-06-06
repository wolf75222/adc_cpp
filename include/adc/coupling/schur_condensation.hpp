#pragma once

#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>  // apply_laplacian (Lap phi^n)
#include <adc/numerics/lorentz_eliminator.hpp>          // B^{-1} 2x2 ferme

#include <stdexcept>

/// @file
/// @brief BATISSEUR (et NON solveur) de l'etage source condense par Schur de la source implicite
///        couplee potentiel / vitesse / Lorentz (Hoffart et al., arXiv:2510.11808 ;
///        docs/SCHUR_CONDENSATION_DESIGN.md). Ce header N'ASSEMBLE QUE les coefficients de l'operateur
///        elliptique tensoriel A_op et le second membre condense ; il NE resout PAS et NE reconstruit
///        PAS la vitesse (etage PR4, qui appellera le concept EllipticSolver, p.ex. TensorKrylovSolver).
///
/// CONVENTION DE SIGNE FIGEE (docs/SCHUR_CONDENSATION_DESIGN.md section 2.1) :
///   On theta-discretise la source. L'inconnue est phi^{n+theta}. L'operateur condense est
///       L_schur(phi) = -Lap phi - theta^2 dt^2 alpha div( rho B^{-1} grad phi )
///   et son second membre
///       RHS = -Lap phi^n - theta dt alpha div( rho B^{-1} v^n ),  v^n = (mx, my) / rho.
///   On note c = theta^2 dt^2 alpha (coefficient du terme condense).
///
///   ECRITURE COMME OPERATEUR A TENSEUR PLEIN (poisson_operator.hpp). Le chemin elliptique resout
///   L(phi) = -div(A grad phi) + kappa phi, A = [[eps_x, a_xy], [a_yx, eps_y]]. On identifie
///       -Lap phi - c div(rho B^{-1} grad phi) = -div( (I + c rho B^{-1}) grad phi ),
///   donc A = I + c rho B^{-1}, ce qui donne PAR CELLULE :
///       eps_x = 1 + c rho binv_11      a_xy = c rho binv_12
///       a_yx  = c rho binv_21          eps_y = 1 + c rho binv_22
///   avec B^{-1} = (1/det) [[1, w], [-w, 1]], w = theta dt B_z, det = 1 + w^2 (LorentzEliminator).
///   En B_z = 0 : w = 0, binv = I, donc a_xy = a_yx = 0 et eps_x = eps_y = 1 + c rho. Si de plus
///   c = 0 (theta=0 ou dt=0 ou alpha=0), A = I et L_schur degenere EXACTEMENT en le Laplacien
///   canonique (Poisson standard). kappa reste NUL (pas de terme de masse dans la condensation).
///
///   DISCRETISATION DU SECOND MEMBRE. -Lap phi^n est calcule par apply_laplacian SANS coefficient
///   (Laplacien 5 points canonique = div(grad phi^n)), puis negue. La divergence du flux EXPLICITE
///   F = rho B^{-1} v^n (champ de vecteur AU CENTRE des cellules) est une divergence FV centree :
///       div F (i,j) = (Fx(i+1,j) - Fx(i-1,j)) / (2 dx) + (Fy(i,j+1) - Fy(i,j-1)) / (2 dy),
///   d'ordre 2, coherente avec la discretisation centree de l'operateur (poisson_operator.hpp).
///
/// GENERICITE (docs/SCHUR_CONDENSATION_DESIGN.md section 4). Ce batisseur ne nomme AUCUN scenario :
///   il lit les ROLES Density / MomentumX / MomentumY (et Energy ignore ici) d'un VariableSet, plus un
///   champ aux B_z, alpha, theta, dt. Tout bloc fluide qui expose ces roles est eligible (anneau polaire a derive ExB OU
///   multi-especes magnetisees) sans une ligne de code Schur supplementaire. Le contrat est valide a
///   l'assemblage (roles presents, sinon exception explicite cote HOTE, avant tout kernel).
///
/// DEVICE / MPI (docs/GPU_RUNTIME_PORT.md). Tous les kernels sont des FONCTEURS NOMMES device-clean
///   (pas de lambda etendue premiere-instanciee cross-TU, limite nvcc #64/#97). Les coefficients vivent
///   en MultiFab (un champ par entree). Toutes les boucles iterent sur local_size() : un rang SANS box
///   (local_size()==0, decoupage MPI) n'execute aucun kernel et ne deref jamais fab(0) -> MPI-propre.

namespace adc {

namespace detail {

/// Coefficients de l'operateur tensoriel A_op = I + c rho B^{-1} ASSEMBLES par cellule depuis l'etat
/// fluide et le champ B_z. Ecrit eps_x, eps_y (diagonale) et a_xy, a_yx (croises) en composante 0 de
/// leurs MultiFab respectifs. Foncteur NOMME (device-clean) : capture des handles Array4 (POD) + scalaires.
///   s    : etat fluide (lecture rho = s(.,.,c_rho)) ;
///   bz   : champ B_z au centre (lecture B_z = bz(.,.,0)) ;
///   c_rho: composante du role Density dans l'etat ;
///   c    : theta^2 dt^2 alpha ; th_dt : theta dt (pour w = theta dt B_z).
struct SchurOperatorCoeffKernel {
  ConstArray4 s;     ///< etat fluide
  ConstArray4 bz;    ///< champ B_z au centre
  Array4 ex, ey;     ///< sortie : eps_x, eps_y (diagonale de A)
  Array4 axy, ayx;   ///< sortie : termes croises a_xy, a_yx
  Real c;            ///< c = theta^2 dt^2 alpha
  Real th_dt;        ///< theta * dt (pour LorentzEliminator : w = th_dt * B_z, et binv ne depend que de w)
  int c_rho;         ///< composante Density
  ADC_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    // B^{-1} ferme (rotation-dilatation 2x2) : seul w = theta dt B_z entre dans binv. On reconstruit
    // l'eliminateur via (theta=th_dt, dt=1, B_z) pour obtenir w = th_dt * B_z (memes binv).
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    const Real cr = c * rho;  // c rho : facteur commun des 4 entrees de A - I
    ex(i, j, 0)  = Real(1) + cr * le.binv_11();
    ey(i, j, 0)  = Real(1) + cr * le.binv_22();
    axy(i, j, 0) = cr * le.binv_12();
    ayx(i, j, 0) = cr * le.binv_21();
  }
};

/// Flux EXPLICITE F = rho B^{-1} v^n (v = (mx,my)/rho) ASSEMBLE par cellule, ECRIT au centre dans fx/fy.
/// rho B^{-1} v = B^{-1} (rho v) = B^{-1} (mx, my) : on applique B^{-1} DIRECTEMENT a la quantite de
/// mouvement (mx, my), ce qui evite la division par rho (et son cas rho=0). Foncteur NOMME device-clean.
struct SchurExplicitFluxKernel {
  ConstArray4 s;     ///< etat fluide (mx, my lus aux composantes c_mx, c_my)
  ConstArray4 bz;    ///< champ B_z au centre
  Array4 fx, fy;     ///< sortie : composantes du flux F = B^{-1} (mx, my)
  Real th_dt;        ///< theta * dt (w = th_dt * B_z)
  int c_mx, c_my;    ///< composantes MomentumX / MomentumY
  ADC_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    Real Fx, Fy;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fx, Fy);  // B^{-1} (mx, my) = rho B^{-1} v
    fx(i, j, 0) = Fx;
    fy(i, j, 0) = Fy;
  }
};

/// rhs(i,j) = lap(i,j) (= -Lap phi^n, deja negue par l'appelant) - g * div F, divergence centree d'ordre 2
/// d'un flux F au centre (fx, fy, ghosts remplis). g = theta dt alpha. Foncteur NOMME device-clean.
struct SchurRhsAssembleKernel {
  ConstArray4 neg_lap;  ///< -Lap phi^n (deja negue)
  ConstArray4 fx, fy;   ///< flux F au centre (ghosts remplis)
  Array4 rhs;           ///< sortie : second membre condense
  Real g;               ///< theta dt alpha
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy)
  ADC_HD void operator()(int i, int j) const {
    const Real divF = (fx(i + 1, j, 0) - fx(i - 1, j, 0)) * half_idx +
                      (fy(i, j + 1, 0) - fy(i, j - 1, 0)) * half_idy;
    rhs(i, j, 0) = neg_lap(i, j, 0) - g * divF;
  }
};

/// neg(i,j) = -src(i,j) (negation de la composante 0). Foncteur NOMME device-clean.
struct NegateKernel {
  ConstArray4 src;
  Array4 dst;
  ADC_HD void operator()(int i, int j) const { dst(i, j, 0) = -src(i, j, 0); }
};

}  // namespace detail

/// Resultat de l'assemblage Schur : les MultiFab de coefficients de l'operateur tensoriel A_op et le
/// second membre condense. Possede ses propres champs (move-only par std::vector interne du MultiFab) :
/// l'appelant (PR4) les passe ensuite a GeometricMG::set_epsilon_anisotropic(eps_x, eps_y) +
/// set_cross_terms-par-champ / au TensorKrylovSolver, et resout L_schur(phi) = rhs.
///
/// kappa reste ABSENT (la condensation ne produit pas de terme de masse) : l'operateur est
/// -div(A grad phi), pas Helmholtz. eps_x == eps_y == (1 + c rho) et a_xy == a_yx == 0 quand B_z == 0.
struct SchurCondensationOperator {
  MultiFab eps_x;  ///< diagonale A_xx = 1 + c rho binv_11
  MultiFab eps_y;  ///< diagonale A_yy = 1 + c rho binv_22
  MultiFab a_xy;   ///< croise A_xy = c rho binv_12
  MultiFab a_yx;   ///< croise A_yx = c rho binv_21
  MultiFab rhs;    ///< second membre condense -Lap phi^n - theta dt alpha div(rho B^{-1} v^n)
};

/// Batisseur GENERIQUE de l'etage source condense pour la source electrostatique + Lorentz
/// (kind="electrostatic_lorentz" cote facade Python future, PR5). Lit les roles d'un bloc fluide, NE
/// resout PAS. C'est l'objet "niveau 1+4 partiel" de docs/SCHUR_CONDENSATION_DESIGN.md restreint a
/// l'ASSEMBLAGE (operateur + RHS), sans le solve ni la reconstruction de vitesse.
class ElectrostaticLorentzCondensation {
 public:
  /// @p vars  : descripteur du bloc fluide ; DOIT exposer les roles Density / MomentumX / MomentumY.
  /// @p alpha : constante de couplage (electrostatique).
  /// @p theta : implicite du theta-schema (0.5 = Crank-Nicolson).
  /// @p dt    : pas de temps de l'etage source.
  /// Le contrat de roles est valide ICI (hote, avant tout kernel) : roles manquants -> exception claire.
  ElectrostaticLorentzCondensation(const VariableSet& vars, Real alpha, Real theta, Real dt)
      : c_rho_(vars.index_of(VariableRole::Density)),
        c_mx_(vars.index_of(VariableRole::MomentumX)),
        c_my_(vars.index_of(VariableRole::MomentumY)),
        alpha_(alpha), theta_(theta), dt_(dt) {
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "ElectrostaticLorentzCondensation : le bloc fluide doit exposer les roles Density, "
          "MomentumX et MomentumY (VariableSet.roles renseignes).");
  }

  Real c_coeff() const { return theta_ * theta_ * dt_ * dt_ * alpha_; }  ///< c = theta^2 dt^2 alpha
  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }

  /// Assemble UNIQUEMENT les coefficients de l'operateur tensoriel A_op = I + c rho B^{-1} dans des
  /// MultiFab eps_x/eps_y (1 ghost, pour la moyenne harmonique de face de l'operateur) et a_xy/a_yx
  /// (1 ghost, pour la moyenne arithmetique de face des flux croises). Les ghosts sont remplis par la
  /// CL @p bc (extrapolation gradient-nul sur les bords physiques, comme le cablage eps existant).
  ///   @p state : etat fluide (lecture rho au centre) ;
  ///   @p bz    : champ B_z au centre (1 ghost minimum recommande ; lu en (i,j) seulement ici).
  /// state/bz/eps_x/... partagent BoxArray + DistributionMapping (meme decoupage). MPI-propre.
  void assemble_operator(const MultiFab& state, const MultiFab& bz, const Geometry& geom,
                         const BCRec& bc, MultiFab& eps_x, MultiFab& eps_y, MultiFab& a_xy,
                         MultiFab& a_yx) const {
    const Real c = c_coeff();
    const Real th_dt = theta_ * dt_;
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz.fab(li).const_array();
      Array4 ex = eps_x.fab(li).array();
      Array4 ey = eps_y.fab(li).array();
      Array4 fxy = a_xy.fab(li).array();
      Array4 fyx = a_yx.fab(li).array();
      for_each_cell(eps_x.box(li),
                    detail::SchurOperatorCoeffKernel{s, b, ex, ey, fxy, fyx, c, th_dt, c_rho_});
    }
    // ghosts des coefficients : la moyenne de face de l'operateur lit le voisin a +-1 (bord de box ET
    // bord physique). On etend par gradient-nul sur les bords physiques (Foextrap), periodique sinon
    // (eps_bc de GeometricMG). a_xy/a_yx : moyenne arithmetique de face -> meme extension.
    const BCRec ebc = coeff_bc(bc);
    fill_ghosts(eps_x, geom.domain, ebc);
    fill_ghosts(eps_y, geom.domain, ebc);
    fill_ghosts(a_xy, geom.domain, ebc);
    fill_ghosts(a_yx, geom.domain, ebc);
  }

  /// Assemble UNIQUEMENT le second membre condense -Lap phi^n - theta dt alpha div(rho B^{-1} v^n).
  ///   @p phi_n : potentiel courant phi^n (ghosts remplis par la CL @p bc avant le Laplacien) ;
  ///   @p state : etat fluide (lecture mx, my) ; @p bz : champ B_z au centre ;
  ///   @p rhs   : sortie (0 ghost suffit).
  /// Tampons internes (Lap phi^n, flux F) alloues sur le layout de @p rhs : memoire transitoire, mais
  /// l'API reste pure-assemblage (aucun etat persistant entre appels). MPI-propre (boucles local_size()).
  void assemble_rhs(MultiFab& phi_n, const MultiFab& state, const MultiFab& bz,
                    const Geometry& geom, const BCRec& bc, MultiFab& rhs) const {
    const Real th_dt = theta_ * dt_;
    const Real g = theta_ * dt_ * alpha_;  // theta dt alpha (coefficient du terme div(rho B^{-1} v))
    const BoxArray& ba = rhs.box_array();
    const DistributionMapping& dm = rhs.dmap();

    // 1) -Lap phi^n : apply_laplacian SANS coefficient = div(grad phi^n) (Laplacien canonique), negue.
    //    On remplit d'abord les ghosts de phi^n (CL physique) pour que le stencil de bord soit correct.
    device_fence();
    fill_ghosts(phi_n, geom.domain, bc);
    MultiFab lap(ba, dm, 1, 0);
    apply_laplacian(phi_n, geom, lap);  // lap = Lap phi^n (A=I, kappa=0)
    MultiFab neg_lap(ba, dm, 1, 0);
    for (int li = 0; li < neg_lap.local_size(); ++li)
      for_each_cell(neg_lap.box(li),
                    detail::NegateKernel{lap.fab(li).const_array(), neg_lap.fab(li).array()});

    // 2) flux EXPLICITE F = rho B^{-1} v^n = B^{-1} (mx, my), au centre (1 ghost pour la div centree).
    MultiFab fx(ba, dm, 1, 1), fy(ba, dm, 1, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz.fab(li).const_array();
      for_each_cell(fx.box(li), detail::SchurExplicitFluxKernel{
                                    s, b, fx.fab(li).array(), fy.fab(li).array(), th_dt, c_mx_, c_my_});
    }
    // ghosts de F : la divergence centree lit Fx(i+-1), Fy(j+-1) (bord de box ET bord physique).
    const BCRec ebc = coeff_bc(bc);
    fill_ghosts(fx, geom.domain, ebc);
    fill_ghosts(fy, geom.domain, ebc);

    // 3) rhs = -Lap phi^n - g div F (divergence centree d'ordre 2).
    const Real half_idx = Real(1) / (Real(2) * geom.dx());
    const Real half_idy = Real(1) / (Real(2) * geom.dy());
    for (int li = 0; li < rhs.local_size(); ++li)
      for_each_cell(rhs.box(li), detail::SchurRhsAssembleKernel{
                                     neg_lap.fab(li).const_array(), fx.fab(li).const_array(),
                                     fy.fab(li).const_array(), rhs.fab(li).array(), g, half_idx,
                                     half_idy});
  }

  /// Assemblage COMPLET (operateur + RHS) en un objet SchurCondensationOperator alloue sur le layout de
  /// @p state. Confort pour l'appelant PR4 ; les coefficients portent 1 ghost (faces de l'operateur),
  /// le RHS 0 ghost. @p phi_n est modifie (ghosts remplis).
  SchurCondensationOperator assemble(MultiFab& phi_n, const MultiFab& state, const MultiFab& bz,
                                     const Geometry& geom, const BCRec& bc) const {
    const BoxArray& ba = state.box_array();
    const DistributionMapping& dm = state.dmap();
    SchurCondensationOperator op{MultiFab(ba, dm, 1, 1), MultiFab(ba, dm, 1, 1),
                                 MultiFab(ba, dm, 1, 1), MultiFab(ba, dm, 1, 1),
                                 MultiFab(ba, dm, 1, 0)};
    assemble_operator(state, bz, geom, bc, op.eps_x, op.eps_y, op.a_xy, op.a_yx);
    assemble_rhs(phi_n, state, bz, geom, bc, op.rhs);
    return op;
  }

 private:
  /// CL des champs de coefficient / flux : periodique conserve, bord physique -> gradient-nul
  /// (Foextrap). Identique a GeometricMG::eps_bc : la valeur de face au bord du domaine vaut la valeur
  /// interieure (coefficient continu au contour), coherent avec l'operateur elliptique.
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo); b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo); b.yhi = fo(bc.yhi);
    return b;
  }

  int c_rho_, c_mx_, c_my_;  ///< composantes des roles Density / MomentumX / MomentumY
  Real alpha_, theta_, dt_;
};

}  // namespace adc
