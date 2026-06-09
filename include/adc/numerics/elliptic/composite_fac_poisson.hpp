#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>                 // average_down, coarsen_index
#include <adc/numerics/elliptic/geometric_mg.hpp>  // solveur grossier (multigrille geometrique)
#include <adc/numerics/elliptic/poisson_operator.hpp>  // apply_laplacian (residu, lit les ghosts deja remplis)
#include <adc/numerics/time/amr_patch_range.hpp>   // PatchRange, CoverageMask (empreinte grossiere d'un patch)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

/// @file
/// @brief CompositeFacPoisson : solveur elliptique COMPOSITE AMR a 2 niveaux (Fast Adaptive Composite,
///        FAC) pour le Poisson SCALAIRE Lap phi = f sur un niveau grossier + UN patch fin (ratio 2).
///
/// MOTIVATION (chemin amr-schur). L'actuel Poisson AMR (Option A) resout l'elliptique sur le SEUL
/// niveau grossier puis injecte grad phi (constant par morceaux) aux patchs fins : les patchs raffinent
/// le TRANSPORT mais PAS le couplage elliptique. Un solveur COMPOSITE fait que le patch fin RAFFINE
/// VRAIMENT la solution elliptique (phi/grad phi plus precis pres du patch). C'est le verrou de fidelite
/// AMR (cf. le TODO historique amr_reflux.hpp:20 "couplage Poisson composite (FAC) viendront ensuite").
///
/// ALGORITHME FAC 2 niveaux (McCormick), un patch fin INTERIEUR au domaine grossier. Solution composite
/// phi = phi_f sur le patch, phi_c ailleurs :
///   0. solve grossier initial : GeometricMG(Lap phi_c = f_c, Dirichlet) ;
///   it. repeter :
///      1. ghosts C-F : remplir l'anneau de ghosts du patch par INTERPOLATION BILINEAIRE de phi_c (ordre
///         2 vs l'injection constante d'Option A) -> condition de Dirichlet C-F cellule-centre ;
///      2. solve fin : GS rouge-noir sur le patch a ghosts GELES (Lap phi_f = f_f) ;
///      3. average_down phi_f -> phi_c sur les cellules grossieres COUVERTES (coherence) ;
///      4. residu composite grossier : r_c = f_c - Lap phi_c (cellules NON couvertes), 0 sur les couvertes,
///         + CORRECTION DE FLUX C-F : sur les cellules grossieres BORDANT le patch, le flux a travers la
///         face C-F est remplace par le flux FIN (somme conservative des 2 faces fines) -> two-way coupling ;
///      5. correction grossiere : GeometricMG(Lap e_c = r_c, Dirichlet homogene) ; phi_c += e_c (non couvertes) ;
///   jusqu'a ||r_c|| (norme du residu composite) sous tolerance.
///
/// PERIMETRE (Phase 1). Cartesien, 2 niveaux, UN patch fin mono-box strictement interieur, ratio 2,
/// grossier MONO-BOX REPLIQUE (serie / mono-rang). Le Schur condense (operateur tenseur A=I+c rho B^-1),
/// le multi-patch, et MPI sont des phases ulterieures. Le test MMS valide que le patch fin REDUIT
/// l'erreur elliptique pres du patch vs coarse-only.

namespace adc {

namespace detail {

/// Interpolation BILINEAIRE du potentiel grossier (cellule-centre, @p C avec ghosts) au CENTRE de la
/// cellule fine (i, j). Ratio @p r. Le centre fin a l'abscisse (i+0.5)/r en unites de pas grossier, soit
/// l'indice-centre grossier fx = (i+0.5)/r - 0.5 ; on interpole les 4 centres grossiers entourants.
/// Patch INTERIEUR -> Ic, Ic+1, Jc, Jc+1 sont dans le domaine grossier (ghosts inclus).
ADC_HD inline Real fac_bilerp_coarse(const ConstArray4& C, int i, int j, int r) {
  const Real fx = (Real(i) + Real(0.5)) / Real(r) - Real(0.5);
  const Real fy = (Real(j) + Real(0.5)) / Real(r) - Real(0.5);
  const int Ic = static_cast<int>(std::floor(fx));
  const int Jc = static_cast<int>(std::floor(fy));
  const Real tx = fx - Real(Ic), ty = fy - Real(Jc);
  const Real c00 = C(Ic, Jc, 0), c10 = C(Ic + 1, Jc, 0);
  const Real c01 = C(Ic, Jc + 1, 0), c11 = C(Ic + 1, Jc + 1, 0);
  return (Real(1) - tx) * (Real(1) - ty) * c00 + tx * (Real(1) - ty) * c10 +
         (Real(1) - tx) * ty * c01 + tx * ty * c11;
}

}  // namespace detail

/// Solveur Poisson COMPOSITE FAC 2 niveaux (scalaire). Construit sur le layout grossier (mono-box
/// replique) + le patch fin (mono-box). Le caller fournit f_c (grossier) et f_f (fin) ; le solveur rend
/// phi_c (grossier, couvert = average_down du fin) et phi_f (fin).
class CompositeFacPoisson {
 public:
  /// @p geom_c : geometrie grossiere (domaine entier). @p ba_c : BoxArray grossier (mono-box couvrant
  ///             le domaine). @p bc : CL du domaine (Dirichlet pour ce jalon). @p fine_box : box du
  ///             patch fin (espace d'indices FIN, ratio 2, strictement interieur). @p ratio : 2.
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const Box2D& fine_box, int ratio = 2)
      : geom_c_(geom_c),
        geom_f_(geom_c.refine(ratio)),
        ba_c_(ba_c),
        dm_c_(ba_c.size(), n_ranks()),
        bc_(bc),
        ratio_(ratio),
        fine_box_(fine_box),
        ba_f_(std::vector<Box2D>{fine_box}),
        dm_f_(1, n_ranks()),
        mg_(geom_c, ba_c, bc, {}, /*replicated=*/true),
        phi_c_(ba_c, dm_c_, 1, 1),
        phi_f_(ba_f_, dm_f_, 1, 1),
        f_c_(ba_c, dm_c_, 1, 0),
        f_f_(ba_f_, dm_f_, 1, 0),
        res_c_(ba_c, dm_c_, 1, 0),
        eps_c_(ba_c, dm_c_, 1, 1),
        eps_f_(ba_f_, dm_f_, 1, 1),
        axy_c_(ba_c, dm_c_, 1, 1),
        ayx_c_(ba_c, dm_c_, 1, 1),
        axy_f_(ba_f_, dm_f_, 1, 1),
        ayx_f_(ba_f_, dm_f_, 1, 1),
        // empreinte grossiere du patch (cellules couvertes) : PatchRange (lo/2 .. (hi-1)/2).
        patch_coarse_(PatchRange(fine_box).box()),
        cov_(Box2D::from_extents(geom_c.domain.nx(), geom_c.domain.ny())) {
    cov_.mark(patch_coarse_);
    phi_c_.set_val(Real(0));
    phi_f_.set_val(Real(0));
    eps_c_.set_val(Real(1));  // permittivite par defaut 1 -> operateur = Laplacien (scalaire)
    eps_f_.set_val(Real(1));
    axy_c_.set_val(Real(0));  // termes croises par defaut 0 -> bloc diagonal seul
    ayx_c_.set_val(Real(0));
    axy_f_.set_val(Real(0));
    ayx_f_.set_val(Real(0));
  }

  MultiFab& rhs_coarse() { return f_c_; }   ///< second membre grossier f_c (div(eps grad phi_c) = f_c)
  MultiFab& rhs_fine() { return f_f_; }     ///< second membre fin f_f (div(eps grad phi_f) = f_f)
  MultiFab& phi_coarse() { return phi_c_; }
  MultiFab& phi_fine() { return phi_f_; }
  /// Permittivite VARIABLE eps (au centre des cellules) PAR NIVEAU. A remplir + use_variable_coefficient(true)
  /// pour passer de Lap phi = f a div(eps grad phi) = f -- l'operateur condense de Schur a B_z = 0
  /// (eps = 1 + theta^2 dt^2 alpha rho). eps non rempli / non active -> scalaire (Phase 1), bit-identique.
  MultiFab& eps_coarse() { return eps_c_; }
  MultiFab& eps_fine() { return eps_f_; }
  void use_variable_coefficient(bool v) { has_eps_ = v; }
  /// Termes croises a_xy / a_yx (au centre des cellules) PAR NIVEAU : tenseur PLEIN A = diag(eps,eps) +
  /// [[0,a_xy],[a_yx,0]]. C'est l'operateur condense de Schur a B_z != 0 (a_xy = c rho w/det,
  /// a_yx = -a_xy, w = theta dt B_z) -- antisymetrique, NON auto-adjoint. Petit pour le pas Schur
  /// (c = theta^2 dt^2 alpha) -> SOR/V-cycle convergent (termes croises EXPLICITES). Non actives -> bloc
  /// diagonal seul (Phase 3a/1), bit-identique. Exige use_variable_coefficient(true) (le bloc diagonal).
  MultiFab& a_xy_coarse() { return axy_c_; }
  MultiFab& a_yx_coarse() { return ayx_c_; }
  MultiFab& a_xy_fine() { return axy_f_; }
  MultiFab& a_yx_fine() { return ayx_f_; }
  void use_cross_terms(bool v) { has_cross_ = v; }
  const Box2D& patch_coarse() const { return patch_coarse_; }

  void set_verbose(bool v) { verbose_ = v; }
  /// true : iterer le couplage two-way FAC (correction de flux C-F + correction grossiere). false :
  /// chemin ONE-WAY (solve grossier + solve fin a ghosts C-F bilineaires) -- le patch raffine localement.
  void set_two_way(bool v) { two_way_ = v; }

  /// Resout le systeme composite. @return le residu composite max final.
  /// @p max_iters iterations FAC (two-way) ; @p fine_sweeps balayages SOR par solve fin ; @p tol tolerance.
  Real solve(int max_iters = 30, int fine_sweeps = 400, Real tol = 1e-9) {
    const Real omega = sor_omega();
    // COEFFICIENT VARIABLE (operateur condense Schur B_z=0) : pose eps sur le solveur grossier et
    // remplit les ghosts de eps PAR NIVEAU. eps_c ghosts = gradient-nul (coeff_bc Foextrap, comme le
    // batisseur Schur) ; eps_f ghosts C-F = bilerp de eps_c (coherence du flux de coefficient a travers
    // l'interface). Sans coefficient variable -> operateur Laplacien scalaire (Phase 1, bit-identique).
    if (has_eps_) {
      device_fence();
      fill_ghosts(eps_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(eps_c_, eps_f_);
      mg_.set_epsilon(eps_c_);
    }
    if (has_cross_) {  // tenseur PLEIN (Schur B_z != 0) : termes croises sur les 2 solveurs + ghosts.
      device_fence();
      fill_ghosts(axy_c_, geom_c_.domain, coeff_bc(bc_));
      fill_ghosts(ayx_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(axy_c_, axy_f_);
      fill_cf_coarse_to_fine(ayx_c_, ayx_f_);
      mg_.set_cross_terms(axy_c_, ayx_c_);
    }
    // 0) solve grossier initial (donne un phi_c pour le 1er ghost C-F).
    copy0(mg_.rhs(), f_c_);
    mg_.phi().set_val(Real(0));
    mg_.solve(Real(1e-12), 100);
    copy0(phi_c_, mg_.phi());

    // 1) ghosts C-F bilineaires + solve fin (ONE-WAY de base).
    refresh_fine(fine_sweeps, omega);

    Real rnorm = composite_coarse_residual();
    if (verbose_ && my_rank() == 0) std::fprintf(stderr, "[FAC] init r_c=%.4e\n", rnorm);
    if (!two_way_) { last_residual_ = rnorm; return rnorm; }

    // 2) iterations FAC two-way : correction grossiere (flux C-F) puis re-solve fin.
    for (int it = 0; it < max_iters; ++it) {
      if (rnorm < tol) break;
      // correction grossiere : Lap e_c = r_c (Dirichlet homogene), phi_c += e_c (non couvertes).
      copy0(mg_.rhs(), res_c_);
      mg_.phi().set_val(Real(0));
      mg_.solve(Real(1e-12), 100);
      add_uncovered(phi_c_, mg_.phi());
      // re-ghost + re-solve fin sur le phi_c corrige.
      refresh_fine(fine_sweeps, omega);
      rnorm = composite_coarse_residual();
      if (verbose_ && my_rank() == 0) std::fprintf(stderr, "[FAC] it=%d r_c=%.4e\n", it, rnorm);
    }
    last_residual_ = rnorm;
    return rnorm;
  }

  Real last_residual() const { return last_residual_; }

 private:
  /// dst <- src (composante 0, cellules valides).
  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      const Box2D b = dst.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) d(i, j, 0) = s(i, j, 0);
    }
  }

  /// phi_c += e_c sur les cellules NON couvertes (la correction ne touche pas le couvert = average_down).
  void add_uncovered(MultiFab& phi, const MultiFab& e) {
    device_fence();
    for (int li = 0; li < phi.local_size(); ++li) {
      Array4 p = phi.fab(li).array();
      const ConstArray4 ec = e.fab(li).const_array();
      const Box2D b = phi.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          if (!cov_.covered(i, j)) p(i, j, 0) += ec(i, j, 0);
    }
  }

  /// Remplit l'anneau de ghosts du patch fin par bilerp de phi_c (Dirichlet C-F cellule-centre).
  void fill_cf_ghosts() {
    const ConstArray4 C = phi_c_.fab(0).const_array();  // grossier mono-box replique
    const int ng = phi_f_.n_grow();
    Array4 F = phi_f_.fab(0).array();
    const Box2D vb = phi_f_.box(0);
    for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
      for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
        const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
        if (inside) continue;  // ghosts seulement
        F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
      }
  }

  /// Remplit les ghosts d'un champ de COEFFICIENT fin (@p fine) par bilerp du champ grossier (@p coarse) :
  /// coherence du coefficient a l'interface C-F (la face de coefficient au bord du patch melange le coeff
  /// fin interieur et le coeff grossier injecte). Generique (eps, a_xy, a_yx).
  void fill_cf_coarse_to_fine(const MultiFab& coarse, MultiFab& fine) {
    const ConstArray4 C = coarse.fab(0).const_array();
    const int ng = fine.n_grow();
    Array4 F = fine.fab(0).array();
    const Box2D vb = fine.box(0);
    for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
      for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
        const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
        if (inside) continue;
        F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
      }
  }

  /// CL des coefficients (eps) : periodique conserve, bord physique -> gradient-nul (Foextrap), comme
  /// le batisseur Schur (coeff_bc) -- le coefficient ne porte pas de Dirichlet.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo); c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo); c.yhi = fo(b.yhi);
    return c;
  }

  /// Facteur de sur-relaxation SOR ~ optimal pour le patch (2/(1+sin(pi/N))) -> convergence O(N) sweeps
  /// au lieu de O(N^2) pour GS. N = plus grand cote du patch fin.
  Real sor_omega() const {
    const int N = std::max(fine_box_.nx(), fine_box_.ny());
    return Real(2) / (Real(1) + std::sin(Real(kPi_) / Real(N)));
  }

  /// Re-remplit les ghosts C-F bilineaires depuis phi_c puis relaxe le patch fin (SOR) a ghosts GELES.
  void refresh_fine(int sweeps, Real omega) {
    device_fence();
    fill_ghosts(phi_c_, geom_c_.domain, bc_);  // phi_c ghosts physiques (le bilerp lit jusqu'au bord)
    fill_cf_ghosts();
    fine_sor(sweeps, omega);
    average_down(phi_f_, phi_c_, ratio_);  // coherence : couvert grossier = moyenne fine
  }

  /// SOR rouge-noir sur le patch fin : div(eps grad phi_f) = f_f (eps = harmonique de face), ghosts
  /// GELES (pas de re-remplissage). eps == 1 partout (scalaire) -> Laplacien, bit-identique a Phase 1.
  void fine_sor(int sweeps, Real omega) {
    const Real idx2 = Real(1) / (geom_f_.dx() * geom_f_.dx());
    const Real idy2 = Real(1) / (geom_f_.dy() * geom_f_.dy());
    const Box2D vb = phi_f_.box(0);
    Array4 P = phi_f_.fab(0).array();
    const ConstArray4 Pc = phi_f_.fab(0).const_array();  // vue const (meme memoire) pour le stencil croise
    const ConstArray4 F = f_f_.fab(0).const_array();
    const ConstArray4 E = eps_f_.fab(0).const_array();
    const ConstArray4 AXY = axy_f_.fab(0).const_array();
    const ConstArray4 AYX = ayx_f_.fab(0).const_array();
    const bool he = has_eps_;
    const bool hc = has_cross_;
    const Real idx = Real(1) / geom_f_.dx(), idy = Real(1) / geom_f_.dy();  // cross_div : 1/dx, 1/dy
    for (int s = 0; s < sweeps; ++s)
      for (int color = 0; color < 2; ++color)
        for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
          for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
            if (((i + j) & 1) != color) continue;
            // permittivites de FACE (moyenne harmonique des 2 centres) ; eps==1 -> faces == 1.
            const Real exm = he ? eps_harmonic(E(i, j, 0), E(i - 1, j, 0)) : Real(1);
            const Real exp = he ? eps_harmonic(E(i, j, 0), E(i + 1, j, 0)) : Real(1);
            const Real eym = he ? eps_harmonic(E(i, j, 0), E(i, j - 1, 0)) : Real(1);
            const Real eyp = he ? eps_harmonic(E(i, j, 0), E(i, j + 1, 0)) : Real(1);
            const Real diag = (exm + exp) * idx2 + (eym + eyp) * idy2;
            const Real nb = (exm * P(i - 1, j, 0) + exp * P(i + 1, j, 0)) * idx2 +
                            (eym * P(i, j - 1, 0) + eyp * P(i, j + 1, 0)) * idy2;
            // termes croises EXPLICITES (9 points, lus a partir du P courant) : div(A grad phi) =
            // bloc_diag + cross. On resout bloc_diag(P) + cross(P) = f -> P = (nb + cross - f)/diag.
            const Real cross =
                hc ? detail::cross_div(Pc, true, AXY, true, AYX, i, j, idx, idy) : Real(0);
            const Real pgs = (nb + cross - F(i, j, 0)) / diag;
            P(i, j, 0) = (Real(1) - omega) * P(i, j, 0) + omega * pgs;  // sur-relaxation (sous-relax si fort)
          }
  }

  /// Residu composite grossier : r_c = f_c - div(eps grad phi_c) (non couvert), 0 (couvert), + correction
  /// de FLUX C-F sur les cellules bordant le patch. @return ||r_c||_inf (cellules NON couvertes).
  Real composite_coarse_residual() {
    device_fence();
    fill_ghosts(phi_c_, geom_c_.domain, bc_);
    // r_c = f_c - div(A grad phi_c) (apply_laplacian lit les ghosts deja remplis ; eps + croises si actifs).
    // Les croises sont lus aussi sur les cellules COUVERTES (= moyenne fine apres average_down) -> le
    // stencil 9 points reste coherent a l'interface ; seul le flux NORMAL est explicitement raccorde C-F
    // (le flux croise, tangentiel et petit pour le pas Schur, est porte par le stencil de volume).
    MultiFab lap(ba_c_, dm_c_, 1, 0);
    apply_laplacian(phi_c_, geom_c_, lap, /*coef=*/nullptr, has_eps_ ? &eps_c_ : nullptr,
                    /*kappa=*/nullptr, /*eps_y=*/nullptr, has_cross_ ? &axy_c_ : nullptr,
                    has_cross_ ? &ayx_c_ : nullptr);
    device_fence();
    Array4 R = res_c_.fab(0).array();
    const ConstArray4 LAP = lap.fab(0).const_array();
    const ConstArray4 FC = f_c_.fab(0).const_array();
    const Box2D b = res_c_.box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        R(i, j, 0) = cov_.covered(i, j) ? Real(0) : (FC(i, j, 0) - LAP(i, j, 0));

    // CORRECTION DE FLUX C-F. Sur chaque cellule grossiere BORDANT le patch (non couverte, voisin
    // couvert), on REMPLACE la contribution de la face C-F dans div(eps grad phi_c) par la contribution
    // FINE (somme conservative des r faces fines, eps de face harmonique). r_c += (grossiere - fine).
    const ConstArray4 PC = phi_c_.fab(0).const_array();
    const ConstArray4 PF = phi_f_.fab(0).const_array();
    const ConstArray4 EC = eps_c_.fab(0).const_array();
    const ConstArray4 EF = eps_f_.fab(0).const_array();
    const bool he = has_eps_;
    const Real idx2 = Real(1) / (geom_c_.dx() * geom_c_.dx());
    const Real idy2 = Real(1) / (geom_c_.dy() * geom_c_.dy());
    const int Ic0 = patch_coarse_.lo[0], Ic1 = patch_coarse_.hi[0];
    const int Jc0 = patch_coarse_.lo[1], Jc1 = patch_coarse_.hi[1];
    const int r = ratio_;
    // Faces NORMALES A X : colonnes bordantes I = Ic0-1 (face +x couverte) et I = Ic1+1 (face -x).
    for (int J = Jc0; J <= Jc1; ++J) {
      {  // gauche : cellule (Ic0-1, J), face grossiere entre (Ic0-1) et (Ic0) ; faces fines a i = r*Ic0.
        const int I = Ic0 - 1;
        const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I + 1, J, 0)) : Real(1);
        const Real coarse_c = efc * (PC(I + 1, J, 0) - PC(I, J, 0)) * idx2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int jf = r * J + t;
          const Real eff = he ? eps_harmonic(EF(r * Ic0 - 1, jf, 0), EF(r * Ic0, jf, 0)) : Real(1);
          fine_sum += eff * (PF(r * Ic0, jf, 0) - PF(r * Ic0 - 1, jf, 0));  // interieur - ghost
        }
        R(I, J, 0) += coarse_c - fine_sum * idx2;
      }
      {  // droite : cellule (Ic1+1, J), face entre (Ic1) et (Ic1+1) ; faces fines a i = r*Ic1+r.
        const int I = Ic1 + 1;
        const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I - 1, J, 0)) : Real(1);
        const Real coarse_c = efc * (PC(I - 1, J, 0) - PC(I, J, 0)) * idx2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int jf = r * J + t;
          const Real eff =
              he ? eps_harmonic(EF(r * Ic1 + r - 1, jf, 0), EF(r * Ic1 + r, jf, 0)) : Real(1);
          fine_sum += eff * (PF(r * Ic1 + r - 1, jf, 0) - PF(r * Ic1 + r, jf, 0));
        }
        R(I, J, 0) += coarse_c - fine_sum * idx2;
      }
    }
    // Faces NORMALES A Y : rangees bordantes J = Jc0-1 (face +y) et J = Jc1+1 (face -y).
    for (int I = Ic0; I <= Ic1; ++I) {
      {
        const int J = Jc0 - 1;
        const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I, J + 1, 0)) : Real(1);
        const Real coarse_c = efc * (PC(I, J + 1, 0) - PC(I, J, 0)) * idy2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int iff = r * I + t;
          const Real eff = he ? eps_harmonic(EF(iff, r * Jc0 - 1, 0), EF(iff, r * Jc0, 0)) : Real(1);
          fine_sum += eff * (PF(iff, r * Jc0, 0) - PF(iff, r * Jc0 - 1, 0));
        }
        R(I, J, 0) += coarse_c - fine_sum * idy2;
      }
      {
        const int J = Jc1 + 1;
        const Real efc = he ? eps_harmonic(EC(I, J, 0), EC(I, J - 1, 0)) : Real(1);
        const Real coarse_c = efc * (PC(I, J - 1, 0) - PC(I, J, 0)) * idy2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int iff = r * I + t;
          const Real eff =
              he ? eps_harmonic(EF(iff, r * Jc1 + r - 1, 0), EF(iff, r * Jc1 + r, 0)) : Real(1);
          fine_sum += eff * (PF(iff, r * Jc1 + r - 1, 0) - PF(iff, r * Jc1 + r, 0));
        }
        R(I, J, 0) += coarse_c - fine_sum * idy2;
      }
    }

    // norme inf du residu sur les cellules NON couvertes.
    Real nrm = Real(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (!cov_.covered(i, j)) nrm = std::fmax(nrm, std::fabs(R(i, j, 0)));
    return nrm;
  }

  Geometry geom_c_, geom_f_;
  BoxArray ba_c_;
  DistributionMapping dm_c_;
  BCRec bc_;
  int ratio_;
  Box2D fine_box_;
  BoxArray ba_f_;
  DistributionMapping dm_f_;
  GeometricMG mg_;  ///< solveur grossier (initial + corrections), Dirichlet homogene
  MultiFab phi_c_, phi_f_, f_c_, f_f_, res_c_;
  MultiFab eps_c_, eps_f_;  ///< permittivite variable par niveau (operateur condense Schur B_z=0)
  MultiFab axy_c_, ayx_c_, axy_f_, ayx_f_;  ///< termes croises par niveau (tenseur plein, Schur B_z!=0)
  Box2D patch_coarse_;
  CoverageMask cov_;
  Real last_residual_ = 0;
  bool has_eps_ = false;   ///< true : operateur div(eps grad phi) ; false : Laplacien scalaire (Phase 1)
  bool has_cross_ = false; ///< true : ajoute les termes croises a_xy/a_yx (tenseur plein, Schur B_z!=0)
  bool verbose_ = false;
  bool two_way_ = true;
  static constexpr Real kPi_ = Real(3.14159265358979323846);
};

}  // namespace adc
