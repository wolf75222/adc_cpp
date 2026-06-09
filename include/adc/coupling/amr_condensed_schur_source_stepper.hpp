#pragma once

#include <adc/coupling/condensed_schur_source_stepper.hpp>  // CondensedSchurSourceStepper (#126) + detail kernels
#include <adc/coupling/schur_condensation.hpp>              // ElectrostaticLorentzCondensation (assemble par niveau)
#include <adc/numerics/elliptic/composite_fac_poisson.hpp>  // CompositeFacPoisson (solve elliptique composite FAC)
#include <adc/numerics/time/amr_reflux_mf.hpp>              // mf_average_down_mb (cascade fin -> grossier)
#include <adc/numerics/time/amr_subcycling.hpp>              // AmrLevelMP (hierarchie multi-patch)

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief AmrCondensedSchurSourceStepper : pendant AMR de l'etage SOURCE condense par Schur
///        (CondensedSchurSourceStepper, #126), porte sur une HIERARCHIE de niveaux (AmrLevelMP) plutot
///        que sur une grille uniforme. C'est l'etage source GLOBAL electrostatique/Lorentz du chemin
///        "amr-schur" -- l'equivalent raffine du chemin uniforme
///          System(...).add_equation(time=Strang(hyperbolic=Explicit(ssprk3),
///                                                source=CondensedSchur(theta, alpha)))
///        et NON une source locale cellule-par-cellule (cf. l'IMEX local backward_euler_source du
///        chemin amr-imex, qui n'est PAS quantitativement comparable au papier Hoffart arXiv:2510.11808).
///
/// STRATEGIE (option A, miroir du Poisson AMR existant compute_aux/solve_fields). Le solveur elliptique
/// AMR de ce code resout le Poisson sur le NIVEAU GROSSIER puis injecte grad phi aux niveaux fins (les
/// patchs fins raffinent le TRANSPORT, pas la resolution elliptique). L'etage source condense suit la
/// MEME approche : il assemble et resout l'operateur condense A_op = I + theta^2 dt^2 alpha rho B^{-1}
/// sur le grossier (en COMPOSANT l'etage uniforme #126, bit-pour-bit), puis -- pour une hierarchie
/// multi-niveau -- injecte grad phi^{n+theta} aux fins et y reconstruit les vitesses, en terminant par
/// la cascade fin -> grossier (average_down) qui retablit la coherence des cellules grossieres
/// couvertes (invariant #169). Un etat constant en espace (mono-niveau) degenere EXACTEMENT en l'etage
/// uniforme : c'est le critere de parite (Etape 2).
///
/// PERIMETRE DE CETTE VERSION (Etape 2, parite System d'abord). Le chemin MONO-NIVEAU est complet et
/// bit-identique a l'etage uniforme. Le chemin MULTI-NIVEAU (reconstruction des vitesses fines a partir
/// du grad injecte + cascade average_down) est un suivi dedie (Etape 4) : step() le REFUSE
/// explicitement (erreur claire) plutot que d'appliquer la source au seul grossier en silence (les
/// cellules fines ne sentiraient pas la source -> faux). On valide d'abord la parite mono-niveau contre
/// l'etage uniforme #126, comme demande.
///
/// CYCLE DE VIE / DEVICE / MPI. Construit UNE fois sur le layout GROSSIER (BoxArray + Geometry + CL
/// Poisson) ; tous les tampons de l'etage uniforme grossier sont alloues a la construction et reutilises
/// par step(). Le solve de Krylov grossier est COLLECTIF (dot/all_reduce sur tous les rangs, y compris
/// vides) -- comme l'etage uniforme : pas d'interblocage. theta/dt peuvent changer entre appels.

namespace adc {

/// ETAGE SOURCE condense par Schur sur une hierarchie AMR. GENERIQUE sur tout bloc fluide qui expose
/// les roles Density / MomentumX / MomentumY (+ Energy optionnel), exactement comme l'etage uniforme.
class AmrCondensedSchurSourceStepper {
 public:
  /// @p vars  : descripteur du bloc fluide (DOIT exposer Density / MomentumX / MomentumY ; Energy
  ///            optionnel). Valide ICI (hote) par le ctor de l'etage uniforme grossier.
  /// @p coarse_geom : geometrie du NIVEAU GROSSIER (cartesienne).
  /// @p coarse_ba   : decoupage du niveau grossier (mono-box replique ou multi-box reparti).
  /// @p bcPhi : CL du potentiel phi (memes que le Poisson grossier).
  /// @p alpha : constante de couplage electrostatique.
  /// @p n_precond_vcycles : N V-cycles MG par application du preconditionneur BiCGStab (1 ou 2).
  AmrCondensedSchurSourceStepper(const VariableSet& vars, const Geometry& coarse_geom,
                                 const BoxArray& coarse_ba, const BCRec& bcPhi, Real alpha,
                                 int n_precond_vcycles = 1)
      : vars_(vars),
        coarse_geom_(coarse_geom),
        coarse_ba_(coarse_ba),
        bcPhi_(bcPhi),
        alpha_(alpha),
        c_rho_(vars.index_of(VariableRole::Density)),
        c_mx_(vars.index_of(VariableRole::MomentumX)),
        c_my_(vars.index_of(VariableRole::MomentumY)),
        c_E_(vars.index_of(VariableRole::Energy)),
        coarse_(vars, coarse_geom, coarse_ba, bcPhi, alpha, n_precond_vcycles) {}

  /// true si le modele porte un role Energy (mise a jour d'energie active dans l'etage grossier).
  bool has_energy() const { return coarse_.energy_comp() >= 0; }

  /// ETAGE SOURCE condense, IN-PLACE sur la hierarchie @p levels et le potentiel grossier @p coarse_phi.
  ///   @p levels    : hierarchie multi-patch ; levels[0] = GROSSIER (level 0), levels[k>=1] = FIN
  ///                  (ratio 2). L'etat conservatif de chaque niveau est levels[k].U (rho GELEE,
  ///                  mom/E mis a jour ; meme convention que l'etage uniforme).
  ///   @p coarse_phi: potentiel du niveau grossier. ENTREE phi^n (warm start du solve) ; SORTIE
  ///                  phi^{n+1}. Meme objet que le Poisson grossier (mg_.phi() du coupleur) cote facade.
  ///   @p coarse_bz : champ B_z du niveau grossier (canal aux), composante @p c_bz lue au centre.
  ///   @p theta / @p dt : theta-schema (theta dans (0, 1]) ; dt = pas effectif (facteur stride inclus
  ///                  par l'appelant, comme s.advance / run_source_stage du chemin uniforme).
  void step(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi, const MultiFab& coarse_bz,
            int c_bz, Real theta, Real dt) {
    if (levels.empty()) return;
    // Un niveau fin EFFECTIVEMENT PEUPLE (>= un patch) signale une hierarchie multi-niveau. NB : le
    // chemin compile (build_amr_compiled) alloue TOUJOURS un niveau fin seed, VIDE apres regrid quand
    // aucun raffinement n'est demande (refine_threshold desactive) -> levels.size() vaut 2 mais la
    // hierarchie est EFFECTIVEMENT mono-niveau. On garde donc sur le NOMBRE DE PATCHS fins, pas sur
    // levels.size(), pour ne pas refuser le cas mono-niveau a niveau fin alloue mais vide.
    int n_fine_patches = 0;
    for (std::size_t k = 1; k < levels.size(); ++k)
      n_fine_patches += static_cast<int>(levels[k].U.box_array().size());
    if (n_fine_patches == 0) {
      // MONO-NIVEAU (aucun patch fin) : etage uniforme COMPLET sur le grossier (assemble + solve +
      // reconstruction + extrapolation + energie + ghosts), bit-pour-bit identique a #126.
      coarse_.step(levels[0].U, coarse_phi, coarse_bz, c_bz, theta, dt);
      return;
    }
    // MULTI-NIVEAU (Phase 3c) : etage source condense COMPOSITE -- le patch fin raffine VRAIMENT
    // l'elliptique (operateur tensoriel Schur resolu par FAC sur grossier + fin), puis reconstruction
    // des vitesses PAR NIVEAU et cascade average_down. Cadre : 2 niveaux, UN patch fin mono-box,
    // grossier replique (mono-rang). Au-dela -> erreur claire (multi-patch / MPI = Phase 4).
    if (levels.size() != 2 || levels[1].U.box_array().size() != 1)
      throw std::runtime_error(
          "AmrCondensedSchurSourceStepper : etage source condense COMPOSITE cable pour 2 niveaux + UN "
          "patch fin mono-box (mono-rang) ; multi-patch / >2 niveaux / MPI = Phase 4.");
    step_multilevel(levels, coarse_phi, coarse_bz, c_bz, theta, dt);
  }

  /// Diagnostic du dernier solve de l'etage grossier (iterations BiCGStab, residu relatif, convergence).
  const KrylovResult& last_solve() const { return coarse_.last_solve(); }

  int density_comp() const { return coarse_.density_comp(); }
  int momentum_x_comp() const { return coarse_.momentum_x_comp(); }
  int momentum_y_comp() const { return coarse_.momentum_y_comp(); }
  int energy_comp() const { return coarse_.energy_comp(); }

 private:
  /// ETAGE SOURCE condense COMPOSITE 2 niveaux (1 patch fin mono-box). Assemble l'operateur condense de
  /// Schur (A = I + c rho B^{-1}, tenseur plein) + le RHS condense PAR NIVEAU (ElectrostaticLorentzCondensation),
  /// resout l'elliptique COMPOSITE (CompositeFacPoisson : le patch fin raffine l'elliptique), reconstruit
  /// la vitesse PAR NIVEAU (v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})), extrapole phi/v au
  /// pas plein, met a jour l'energie, puis cascade fin -> grossier (average_down, cellules couvertes).
  void step_multilevel(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi,
                       const MultiFab& coarse_bz, int c_bz, Real theta, Real dt) {
    const Box2D fine_box = levels[1].U.box_array()[0];
    ensure_fac(fine_box);
    const Geometry geom_c = coarse_geom_;
    const Geometry geom_f = coarse_geom_.refine(2);
    ElectrostaticLorentzCondensation builder(vars_, alpha_, theta, dt);

    MultiFab& Uc = levels[0].U;
    MultiFab& Uf = levels[1].U;
    const BoxArray bac = Uc.box_array();
    const DistributionMapping dmc = Uc.dmap();
    const BoxArray baf = Uf.box_array();
    const DistributionMapping dmf = Uf.dmap();

    // --- B_z 1-composante par niveau (grossier : extrait de coarse_bz ; fin : bilerp du grossier) ---
    MultiFab bz_c(bac, dmc, 1, 1), bz_f(baf, dmf, 1, 1);
    copy_comp(bz_c, coarse_bz, c_bz);
    device_fence();
    fill_ghosts(bz_c, geom_c.domain, coeff_bc(bcPhi_));
    bilerp_coarse_to_fine(bz_f, bz_c);  // B_z fin depuis le grossier (B0 uniforme -> exact)

    // --- phi^n par niveau (grossier = coarse_phi ; fin = aux injecte, levels[1].aux comp 0) ---
    MultiFab phi_n_c(bac, dmc, 1, 1), phi_n_f(baf, dmf, 1, 1);
    copy0(phi_n_c, coarse_phi);
    copy0(phi_n_f, *levels[1].aux);

    // --- v^n par niveau (avant le solve : la reconstruction ecrase mom) ---
    MultiFab vx_n_c(bac, dmc, 1, 0), vy_n_c(bac, dmc, 1, 0);
    MultiFab vx_n_f(baf, dmf, 1, 0), vy_n_f(baf, dmf, 1, 0);
    extract_v(Uc, vx_n_c, vy_n_c);
    extract_v(Uf, vx_n_f, vy_n_f);

    // --- assemblage operateur + RHS condense PAR NIVEAU, dans les champs du solveur composite ---
    // eps_x == eps_y pour le Schur (A_xx = A_yy = 1 + c rho/det) : on ecrit eps_x dans l'eps unique du
    // composite et eps_y dans un scratch jete. f_composite = -rhs_schur (convention de signe #126).
    MultiFab eps_y_c(bac, dmc, 1, 1), eps_y_f(baf, dmf, 1, 1);
    MultiFab rhs_c(bac, dmc, 1, 0), rhs_f(baf, dmf, 1, 0);
    builder.assemble_operator(Uc, bz_c, geom_c, bcPhi_, fac_->eps_coarse(), eps_y_c,
                              fac_->a_xy_coarse(), fac_->a_yx_coarse());
    builder.assemble_operator(Uf, bz_f, geom_f, bcPhi_, fac_->eps_fine(), eps_y_f,
                              fac_->a_xy_fine(), fac_->a_yx_fine());
    { MultiFab pn(bac, dmc, 1, 1); copy0(pn, phi_n_c);
      builder.assemble_rhs(pn, Uc, bz_c, geom_c, bcPhi_, rhs_c);
      negate_into(fac_->rhs_coarse(), rhs_c); }
    { MultiFab pn(baf, dmf, 1, 1); copy0(pn, phi_n_f);
      builder.assemble_rhs(pn, Uf, bz_f, geom_f, bcPhi_, rhs_f);
      negate_into(fac_->rhs_fine(), rhs_f); }

    // --- SOLVE COMPOSITE : phi^{n+theta} par niveau (le patch fin raffine l'elliptique) ---
    fac_->use_variable_coefficient(true);
    fac_->use_cross_terms(true);
    fac_->solve();

    // --- reconstruction des vitesses + extrapolation phi/v + energie, PAR NIVEAU ---
    reconstruct_level(Uc, fac_->phi_coarse(), phi_n_c, bz_c, vx_n_c, vy_n_c, geom_c, theta, dt,
                      /*fill_phi_ghosts=*/true);
    reconstruct_level(Uf, fac_->phi_fine(), phi_n_f, bz_f, vx_n_f, vy_n_f, geom_f, theta, dt,
                      /*fill_phi_ghosts=*/false);  // ghosts C-F deja poses par le solve composite

    // phi^{n+1} grossier (extrapole en place dans fac_->phi_coarse()) -> publie dans coarse_phi.
    copy0(coarse_phi, fac_->phi_coarse());

    // --- cascade fin -> grossier : les cellules grossieres COUVERTES = moyenne 2x2 des fines (#169) ---
    device_fence();
    mf_average_down_mb(Uf, Uc);
    device_fence();
    fill_ghosts(coarse_phi, geom_c.domain, bcPhi_);
  }

  /// Reconstruit v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) (grad CENTRE), ecrit mom = rho v ;
  /// extrapole phi et v du theta-stage au pas plein (f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n)) ; met a
  /// jour l'energie (si role present). @p fill_phi_ghosts : remplir les ghosts physiques de phi (grossier)
  /// ; false pour le fin (les ghosts C-F sont deja poses par le solve composite -- ne PAS les ecraser).
  void reconstruct_level(MultiFab& state, MultiFab& phi_nt, const MultiFab& phi_n, const MultiFab& bz,
                         const MultiFab& vx_n, const MultiFab& vy_n, const Geometry& geom, Real theta,
                         Real dt, bool fill_phi_ghosts) {
    const Real th_dt = theta * dt, inv_theta = Real(1) / theta;
    const Real half_idx = Real(1) / (Real(2) * geom.dx());
    const Real half_idy = Real(1) / (Real(2) * geom.dy());
    device_fence();
    if (fill_phi_ghosts) fill_ghosts(phi_nt, geom.domain, bcPhi_);
    device_fence();
    MultiFab vx_t(state.box_array(), state.dmap(), 1, 0), vy_t(state.box_array(), state.dmap(), 1, 0);
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::SchurReconstructKernel{
                        phi_nt.fab(li).const_array(), vx_n.fab(li).const_array(),
                        vy_n.fab(li).const_array(), bz.fab(li).const_array(), state.fab(li).array(),
                        vx_t.fab(li).array(), vy_t.fab(li).array(), th_dt, half_idx, half_idy, c_rho_,
                        c_mx_, c_my_});
    for (int li = 0; li < phi_nt.local_size(); ++li)
      for_each_cell(phi_nt.box(li), detail::SchurExtrapolateScalarKernel{
                                        phi_n.fab(li).const_array(), phi_nt.fab(li).array(), inv_theta});
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::SchurExtrapolateVelocityKernel{
                        vx_n.fab(li).const_array(), vy_n.fab(li).const_array(), vx_t.fab(li).array(),
                        vy_t.fab(li).array(), state.fab(li).array(), inv_theta, c_rho_, c_mx_, c_my_});
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li), detail::SchurEnergyKernel{
                                         vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
                                         vx_t.fab(li).const_array(), vy_t.fab(li).const_array(),
                                         state.fab(li).array(), c_rho_, c_E_});
    device_fence();
    fill_ghosts(state, geom.domain, coeff_bc(bcPhi_));
  }

  /// Construit (ou reconstruit si le patch change) le solveur elliptique composite sur le patch fin.
  void ensure_fac(const Box2D& fine_box) {
    if (fac_ && fac_fine_box_lo0_ == fine_box.lo[0] && fac_fine_box_lo1_ == fine_box.lo[1] &&
        fac_fine_box_hi0_ == fine_box.hi[0] && fac_fine_box_hi1_ == fine_box.hi[1])
      return;
    fac_ = std::make_unique<CompositeFacPoisson>(coarse_geom_, coarse_ba_, bcPhi_, fine_box, 2);
    fac_fine_box_lo0_ = fine_box.lo[0]; fac_fine_box_lo1_ = fine_box.lo[1];
    fac_fine_box_hi0_ = fine_box.hi[0]; fac_fine_box_hi1_ = fine_box.hi[1];
  }

  /// CL des coefficients (eps/B_z) et de l'etat publie : periodique conserve, bord physique gradient-nul.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo); c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo); c.yhi = fo(b.yhi);
    return c;
  }

  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li), detail::CopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }
  void copy_comp(MultiFab& dst, const MultiFab& src, int c) {  // dst comp0 <- src comp c
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyBzKernel{src.fab(li).const_array(), dst.fab(li).array(), c});
  }
  void negate_into(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li), detail::NegateKernel{src.fab(li).const_array(), dst.fab(li).array()});
  }
  void extract_v(const MultiFab& state, MultiFab& vx, MultiFab& vy) {
    device_fence();
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{state.fab(li).const_array(), vx.fab(li).array(),
                                                  vy.fab(li).array(), c_rho_, c_mx_, c_my_});
  }
  /// Remplit valides + ghosts du champ fin @p fine par bilerp du champ grossier @p coarse (B_z, etc.).
  void bilerp_coarse_to_fine(MultiFab& fine, const MultiFab& coarse) {
    device_fence();
    const ConstArray4 C = coarse.fab(0).const_array();
    const int ng = fine.n_grow();
    Array4 F = fine.fab(0).array();
    const Box2D vb = fine.box(0);
    for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
      for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i)
        F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, 2);
  }

  VariableSet vars_;
  Geometry coarse_geom_;
  BoxArray coarse_ba_;
  BCRec bcPhi_;
  Real alpha_;
  int c_rho_, c_mx_, c_my_, c_E_;
  /// Etage source condense uniforme porte sur le NIVEAU GROSSIER (chemin MONO-NIVEAU, parite #126).
  CondensedSchurSourceStepper coarse_;
  /// Solveur elliptique composite (chemin MULTI-NIVEAU), construit paresseusement sur le patch fin.
  std::unique_ptr<CompositeFacPoisson> fac_;
  int fac_fine_box_lo0_ = 0, fac_fine_box_lo1_ = 0, fac_fine_box_hi0_ = -1, fac_fine_box_hi1_ = -1;
};

}  // namespace adc
