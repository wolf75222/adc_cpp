#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

// Operateur de Poisson 5 points et lisseur Gauss-Seidel red-black, briques de
// la multigrille geometrique maison.
//
// Convention : on resout l'operateur elliptique div(eps grad phi) = f. Le second
// membre vient du modele, f = model.elliptic_rhs (p.ex. une densite de charge
// signee). Par defaut eps == nullptr : permittivite uniforme eps=1 et l'operateur
// degenere EXACTEMENT en lap(phi) = f (chemin historique, bit-identique).
//
// Masque optionnel (embedded boundary) : un MultiFab mask (1 = actif, 0 =
// conducteur) fige phi=0 dans les cellules conductrices. Le lisseur et le residu
// sautent ces cellules ; les cellules actives voisines lisent phi=0 chez le
// conducteur, ce qui impose la condition de Dirichlet sur la frontiere.
//
// Coefficients CUT-CELL optionnels (coef) : par defaut le stencil est uniforme
// (diag = 2/dx^2 + 2/dy^2), ce qui place la CL Dirichlet sur le contour en ESCALIER
// des centres de cellules (erreur O(1) au bord embedded). Si un champ coef a 5
// composantes est fourni (w_xm, w_xp, w_ym, w_yp, w_diag, par cellule active), le
// stencil les utilise : ce sont les poids de Shortley-Weller calcules a partir des
// DISTANCES reelles au bord (cellules coupees), imposant phi=0 sur le CERCLE et non
// l'escalier (ordre 2 jusqu'au bord). Loin du bord (theta=1 partout) les coef valent
// 1/dx^2, 1/dy^2, 2/dx^2+2/dy^2 -> identiques au stencil uniforme. coef==nullptr
// redonne EXACTEMENT le chemin historique (bit-identique).
//
// PERMITTIVITE VARIABLE eps(x) optionnelle (eps) : un MultiFab a 1 composante,
// eps AU CENTRE des cellules, dont les ghosts doivent etre deja remplis (les
// voisins de bord de box sont lus). On discretise div(eps grad phi) par volumes
// finis : le flux sur chaque face vaut eps_face (phi_voisin - phi_centre)/h, et
// l'operateur est la divergence discrete sum_faces eps_face (phi_voisin -
// phi_centre)/h^2. La permittivite de FACE est la MOYENNE HARMONIQUE des deux
// centres adjacents :
//     eps_face = 2 eps_c eps_v / (eps_c + eps_v).
// Choix HARMONIQUE (et non arithmetique) : c'est la moyenne qui rend le FLUX NORMAL
// continu a l'interface (resistances en serie), donc la discretisation conservative
// reste correcte meme pour un eps DISCONTINU (saut de milieu) ; pour un eps lisse
// elle reste d'ordre 2 comme l'arithmetique. Avec eps uniforme, eps_face == eps et
// l'on retombe sur le Laplacien constant (a l'echelle eps pres). eps==nullptr
// redonne EXACTEMENT le chemin historique (bit-identique).
//
// PERMITTIVITE ANISOTROPE optionnelle (eps_y) : second champ AU CENTRE des cellules
// (memes conventions que eps : 1 composante, ghosts remplis). L'operateur passe de
// div(eps grad phi) (eps scalaire) a div(diag(eps_x, eps_y) grad phi) : le coefficient
// scalaire eps devient alors eps_x et porte UNIQUEMENT les faces NORMALES A X (exm, exp),
// tandis que eps_y porte les faces NORMALES A Y (eym, eyp). Chaque face reste la moyenne
// HARMONIQUE des deux centres adjacents de SON champ (eps_x pour x, eps_y pour y).
// Cas d'usage : milieu/maillage anisotrope (permittivite tensorielle diagonale).
// eps_y==nullptr => ISOTROPE : eps_y = eps (eps_x), donc faces x et y partagent le meme
// champ, comportement actuel STRICTEMENT bit-identique. Le terme kappa est inchange.
//
// Combinaison cut-cell + eps : chaque poids de face Shortley-Weller est multiplie
// par sa permittivite de face, et la diagonale est la somme des poids de face (eps
// inclus). Avec eps uniforme=1 on retrouve les poids de face cut-cell d'origine.
//
// Le balayage red-black est parallelisable : avec le stencil 5 points, une
// cellule rouge ne depend que de cellules noires.

namespace adc {

// Moyenne harmonique de deux permittivites de centre -> permittivite de face.
// Garde-fou anti division par 0 si les deux centres sont nuls (cellule inactive).
ADC_HD inline Real eps_harmonic(Real ec, Real ev) {
  const Real s = ec + ev;
  return s > Real(0) ? Real(2) * ec * ev / s : Real(0);
}

namespace detail {
// FONCTEURS NOMMES (et non lambdas ADC_HD) pour les kernels de l'operateur de Poisson et du lisseur
// Gauss-Seidel. Memes raisons que le reste du chemin elliptique (#93, recette #64) : ces kernels sont
// premiere-instancies depuis le V-cycle MG tire d'une TU externe (harness / loader natif) ; une lambda
// etendue y fait buter l'emission du kernel device sous nvcc (kernel-stub nul -> segfault Cuda en
// Release -O sans -g). Corps STRICTEMENT identique aux anciennes lambdas (memes branches he/hc/hk,
// meme stencil) -> residu et potentiel bit-identiques sur CPU et device.

// L = div(eps grad phi) - kappa phi (apply_laplacian). cf/ep/ey/ka inutilises si le flag est faux.
struct ApplyLaplacianKernel {
  ConstArray4 p;
  Array4 L;
  Real idx2, idy2;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  ADC_HD void operator()(int i, int j) const {
    if (he) {  // permittivite de face (harmonique), avec ou sans cut-cell
      const Real ec = ep(i, j);    // eps_x au centre (faces x)
      const Real ecy = ey(i, j);   // eps_y au centre (faces y) ; == ec en isotrope
      const Real exm = eps_harmonic(ec, ep(i - 1, j));
      const Real exp = eps_harmonic(ec, ep(i + 1, j));
      const Real eym = eps_harmonic(ecy, ey(i, j - 1));
      const Real eyp = eps_harmonic(ecy, ey(i, j + 1));
      Real wxm, wxp, wym, wyp;
      if (hc) {  // cut-cell : eps_face multiplie chaque poids Shortley-Weller
        wxm = cf(i, j, 0) * exm; wxp = cf(i, j, 1) * exp;
        wym = cf(i, j, 2) * eym; wyp = cf(i, j, 3) * eyp;
      } else {   // stencil 5 points a coefficient de face variable
        wxm = exm * idx2; wxp = exp * idx2;
        wym = eym * idy2; wyp = eyp * idy2;
      }
      L(i, j) = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) +
                wym * p(i, j - 1) - (wxm + wxp + wym + wyp) * p(i, j);
    } else if (hc)
      L(i, j) = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
                cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1) -
                cf(i, j, 4) * p(i, j);
    else
      L(i, j) = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    // operateur Helmholtz / ecrante : L phi = div(eps grad phi) - kappa phi.
    if (hk) L(i, j) -= ka(i, j) * p(i, j);
  }
};

// res = f - L phi sur les cellules actives, 0 sur les conductrices (poisson_residual).
struct PoissonResidualKernel {
  ConstArray4 p, ff;
  Array4 r;
  Real idx2, idy2;
  bool hm;
  ConstArray4 mk;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  ADC_HD void operator()(int i, int j) const {
    if (hm && mk(i, j) == Real(0)) {
      r(i, j) = 0;
      return;
    }
    Real lap;
    if (he) {  // permittivite de face (harmonique), avec ou sans cut-cell
      const Real ec = ep(i, j);    // eps_x au centre (faces x)
      const Real ecy = ey(i, j);   // eps_y au centre (faces y) ; == ec en isotrope
      const Real exm = eps_harmonic(ec, ep(i - 1, j));
      const Real exp = eps_harmonic(ec, ep(i + 1, j));
      const Real eym = eps_harmonic(ecy, ey(i, j - 1));
      const Real eyp = eps_harmonic(ecy, ey(i, j + 1));
      Real wxm, wxp, wym, wyp;
      if (hc) {
        wxm = cf(i, j, 0) * exm; wxp = cf(i, j, 1) * exp;
        wym = cf(i, j, 2) * eym; wyp = cf(i, j, 3) * eyp;
      } else {
        wxm = exm * idx2; wxp = exp * idx2;
        wym = eym * idy2; wyp = eyp * idy2;
      }
      lap = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) +
            wym * p(i, j - 1) - (wxm + wxp + wym + wyp) * p(i, j);
    } else if (hc)
      lap = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
            cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1) -
            cf(i, j, 4) * p(i, j);
    else
      lap = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
            (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
    // res = f - L phi, L phi = div(eps grad phi) - kappa phi = lap - kappa phi.
    r(i, j) = ff(i, j) - lap + (hk ? ka(i, j) * p(i, j) : Real(0));
  }
};
}  // namespace detail

inline void apply_laplacian(const MultiFab& phi, const Geometry& geom,
                            MultiFab& lap, const MultiFab* coef = nullptr,
                            const MultiFab* eps = nullptr,
                            const MultiFab* kappa = nullptr,
                            const MultiFab* eps_y = nullptr) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 L = lap.fab(li).array();
    const Box2D v = lap.box(li);
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotrope : faces y lisent le meme champ que les faces x (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // terme de reaction -kappa phi
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, detail::ApplyLaplacianKernel{p, L, idx2, idy2, hc, cf, he, ep, ey, hk, ka});
  }
}

// res = f - div(eps grad phi) sur les cellules actives, 0 sur les conductrices.
inline void poisson_residual(MultiFab& phi, const MultiFab& f,
                             const Geometry& geom, const BCRec& bc,
                             MultiFab& res, const MultiFab* mask = nullptr,
                             const MultiFab* coef = nullptr,
                             const MultiFab* eps = nullptr,
                             const MultiFab* kappa = nullptr,
                             const MultiFab* eps_y = nullptr) {
  device_fence();  // GPU : phi a pu etre ecrit par un kernel (lisseur) ; on
                   // attend avant la lecture hote de fill_ghosts.
  fill_ghosts(phi, geom.domain, bc);
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  for (int li = 0; li < phi.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    const ConstArray4 ff = f.fab(li).const_array();
    Array4 r = res.fab(li).array();
    const Box2D v = res.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotrope : faces y lisent le meme champ que les faces x (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // terme de reaction -kappa phi
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, detail::PoissonResidualKernel{p, ff, r, idx2, idy2, hm, mk, hc, cf,
                                                   he, ep, ey, hk, ka});
  }
}

namespace detail {
// Lisseur Gauss-Seidel red-black sur une couleur (gs_color). p est ECRIT en place. Corps identique a
// l'ancienne lambda -> bit-identique. Voir le commentaire des autres kernels (#93) pour la motivation
// du foncteur nomme.
struct GsColorKernel {
  Array4 p;
  ConstArray4 ff;
  Real idx2, idy2, diag0;
  int color;
  bool hm;
  ConstArray4 mk;
  bool hc;
  ConstArray4 cf;
  bool he;
  ConstArray4 ep, ey;
  bool hk;
  ConstArray4 ka;
  ADC_HD void operator()(int i, int j) const {
    if (((i + j) & 1) != color) return;
    if (hm && mk(i, j) == Real(0)) return;  // conducteur : fige phi=0
    Real off, diag;
    if (he) {  // permittivite de face (harmonique), avec ou sans cut-cell
      const Real ec = ep(i, j);    // eps_x au centre (faces x)
      const Real ecy = ey(i, j);   // eps_y au centre (faces y) ; == ec en isotrope
      const Real exm = eps_harmonic(ec, ep(i - 1, j));
      const Real exp = eps_harmonic(ec, ep(i + 1, j));
      const Real eym = eps_harmonic(ecy, ey(i, j - 1));
      const Real eyp = eps_harmonic(ecy, ey(i, j + 1));
      Real wxm, wxp, wym, wyp;
      if (hc) {
        wxm = cf(i, j, 0) * exm; wxp = cf(i, j, 1) * exp;
        wym = cf(i, j, 2) * eym; wyp = cf(i, j, 3) * eyp;
      } else {
        wxm = exm * idx2; wxp = exp * idx2;
        wym = eym * idy2; wyp = eyp * idy2;
      }
      off = wxp * p(i + 1, j) + wxm * p(i - 1, j) + wyp * p(i, j + 1) +
            wym * p(i, j - 1);
      diag = wxm + wxp + wym + wyp;
    } else if (hc) {  // stencil cut-cell (Shortley-Weller) ; voisin conducteur = phi=0 sur le cercle
      off = cf(i, j, 1) * p(i + 1, j) + cf(i, j, 0) * p(i - 1, j) +
            cf(i, j, 3) * p(i, j + 1) + cf(i, j, 2) * p(i, j - 1);
      diag = cf(i, j, 4);
    } else {
      off = (p(i + 1, j) + p(i - 1, j)) * idx2 +
            (p(i, j + 1) + p(i, j - 1)) * idy2;
      diag = diag0;
    }
    // Terme de reaction : l'operateur devient div(eps grad phi) - kappa phi, donc la
    // diagonale gagne +kappa (kappa >= 0 => plus diagonalement dominant, MG converge mieux).
    p(i, j) = (off - ff(i, j)) / (diag + (hk ? ka(i, j) : Real(0)));
  }
};

inline void gs_color(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                     int color, const MultiFab* mask, const MultiFab* coef,
                     const MultiFab* eps, const MultiFab* kappa = nullptr,
                     const MultiFab* eps_y = nullptr) {
  const Real idx2 = Real(1) / (geom.dx() * geom.dx());
  const Real idy2 = Real(1) / (geom.dy() * geom.dy());
  const Real diag0 = 2 * idx2 + 2 * idy2;
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ff = f.fab(li).const_array();
    const Box2D v = phi.box(li);
    const bool hm = mask != nullptr;
    const ConstArray4 mk = hm ? mask->fab(li).const_array() : ConstArray4{};
    const bool hc = coef != nullptr;
    const ConstArray4 cf = hc ? coef->fab(li).const_array() : ConstArray4{};
    const bool he = eps != nullptr;
    const ConstArray4 ep = he ? eps->fab(li).const_array() : ConstArray4{};
    // eps_y==nullptr => isotrope : faces y lisent le meme champ que les faces x (eps_x).
    const ConstArray4 ey = (he && eps_y) ? eps_y->fab(li).const_array() : ep;
    const bool hk = kappa != nullptr;  // terme de reaction -kappa phi (Helmholtz / ecrante)
    const ConstArray4 ka = hk ? kappa->fab(li).const_array() : ConstArray4{};
    for_each_cell(v, GsColorKernel{p, ff, idx2, idy2, diag0, color, hm, mk, hc, cf,
                                   he, ep, ey, hk, ka});
  }
}
}  // namespace detail

inline void gs_rb_sweep(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                        const BCRec& bc, const MultiFab* mask = nullptr,
                        const MultiFab* coef = nullptr,
                        const MultiFab* eps = nullptr,
                        const MultiFab* kappa = nullptr,
                        const MultiFab* eps_y = nullptr) {
  device_fence();  // attend le kernel precedent avant la lecture hote des halos
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 0, mask, coef, eps, kappa, eps_y);  // rouge (kernel GPU)
  device_fence();  // le balayage noir lit les valeurs rouges via fill_ghosts hote
  fill_ghosts(phi, geom.domain, bc);
  detail::gs_color(phi, f, geom, 1, mask, coef, eps, kappa, eps_y);  // noir
}

inline void gs_smooth(MultiFab& phi, const MultiFab& f, const Geometry& geom,
                      const BCRec& bc, int nsweeps, const MultiFab* mask = nullptr,
                      const MultiFab* coef = nullptr, const MultiFab* eps = nullptr,
                      const MultiFab* kappa = nullptr, const MultiFab* eps_y = nullptr) {
  for (int s = 0; s < nsweeps; ++s) gs_rb_sweep(phi, f, geom, bc, mask, coef, eps, kappa, eps_y);
}

namespace detail {
// Fige phi=0 dans les cellules conductrices (mask==0). Foncteur nomme (#93) ; corps identique.
struct ZeroConductorKernel {
  Array4 p;
  ConstArray4 mk;
  ADC_HD void operator()(int i, int j) const {
    if (mk(i, j) == Real(0)) p(i, j) = 0;
  }
};
}  // namespace detail

// Force phi=0 dans les cellules conductrices (mask==0).
inline void zero_conductor(MultiFab& phi, const MultiFab& mask) {
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 mk = mask.fab(li).const_array();
    const Box2D v = phi.box(li);
    for_each_cell(v, detail::ZeroConductorKernel{p, mk});
  }
}

}  // namespace adc
