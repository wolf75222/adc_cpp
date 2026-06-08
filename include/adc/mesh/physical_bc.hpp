/// @file
/// @brief Conditions aux limites PHYSIQUES au bord du domaine (BCType, BCRec, fill_physical_bc,
///        fill_ghosts).
///
/// fill_boundary remplit deja les ghosts INTERIEURS et periodiques ; ici on remplit les ghosts qui
/// tombent HORS du domaine sur les faces non periodiques. Foextrap : extrapolation d'ordre 0
/// (gradient nul), ghost = cellule interne miroir (outflow / mur ordre 0). Dirichlet : valeur
/// imposee a la FACE, ghost = 2 v - interne miroir (la moyenne ghost/interne vaut v sur la face).
/// fill_ghosts compose les deux dans le bon ordre (interieur/periodique PUIS bord physique) et
/// remplit les coins via faces-x puis faces-y sur l'extension complete. Les kernels de bord sont des
/// FONCTEURS NOMMES device-clean (limite nvcc).

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>

// Conditions aux limites physiques au bord du domaine. fill_boundary remplit
// deja les ghosts interieurs et periodiques ; ici on remplit les ghosts qui
// tombent hors du domaine sur les faces non periodiques.
//
//   Foextrap  : extrapolation d'ordre 0 (gradient nul), ghost = cellule interne
//               la plus proche. Sert d'outflow et de mur a l'ordre 0.
//   Dirichlet : valeur imposee a la face, par reflexion ghost = 2 v - interne
//               miroir (la moyenne ghost/interne vaut v sur la face).
//
// fill_ghosts compose les deux dans le bon ordre (interieur/periodique puis
// bord physique), et remplit les coins via l'ordre faces-x puis faces-y sur
// l'extension complete.

namespace adc {

/// Type de condition au bord d'une face : Periodic (gere par fill_boundary), Foextrap (gradient nul,
/// outflow/mur ordre 0), Dirichlet (valeur imposee a la face par reflexion).
enum class BCType { Periodic, Foextrap, Dirichlet };

/// Conditions aux limites des QUATRE faces du domaine (type + valeur Dirichlet associee). Defaut
/// tout periodique (xlo_val... ignores pour les faces non Dirichlet).
struct BCRec {
  BCType xlo = BCType::Periodic, xhi = BCType::Periodic;
  BCType ylo = BCType::Periodic, yhi = BCType::Periodic;
  Real xlo_val = 0, xhi_val = 0, ylo_val = 0, yhi_val = 0;
};

namespace detail {
// FONCTEURS NOMMES (et non lambdas ADC_HD) pour les conditions aux limites physiques. Memes raisons
// que le reste du chemin elliptique/maillage (#93, recette #64) : fill_physical_bc est appele depuis
// fill_ghosts, lui-meme tire du V-cycle MG premiere-instancie depuis une TU externe ; une lambda
// etendue y fait buter l'emission du kernel device sous nvcc. Corps identique aux anciennes lambdas
// (Foextrap : copie de la cellule miroir interne ; Dirichlet : 2 v - reflexion) -> bit-identique.
// Face x bas : i = lo - k (k = lo - i), miroir Dirichlet en 2 lo - i - 1.
struct BCFaceXLoKernel {
  Array4 a;
  int nc, lo;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(lo, j, c) : 2 * val - a(2 * lo - i - 1, j, c);
  }
};
struct BCFaceXHiKernel {
  Array4 a;
  int nc, hi;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(hi, j, c) : 2 * val - a(2 * hi - i + 1, j, c);
  }
};
struct BCFaceYLoKernel {
  Array4 a;
  int nc, lo;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(i, lo, c) : 2 * val - a(i, 2 * lo - j - 1, c);
  }
};
struct BCFaceYHiKernel {
  Array4 a;
  int nc, hi;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(i, hi, c) : 2 * val - a(i, 2 * hi - j + 1, c);
  }
};
}  // namespace detail

/// Remplit les ghosts HORS domaine des faces NON periodiques de @p mf selon @p bc (Foextrap ou
/// Dirichlet), sur toutes les composantes. No-op si pas de ghost ou tout periodique. PRECONDITION :
/// fill_boundary a deja rempli l'interieur/periodique (les faces x lisent les ghosts y/theta deja
/// remplis pour etendre la CL radiale dans la halo, et les faces y lisent les ghosts x pour les coins).
/// COINS du stencil a 9 points : la CL des faces x est etendue a la plage j ETENDUE (ghosts y/theta
/// inclus), de sorte que le coin (x-physique CROISE y-periodique/voisin) -- lu par les termes croises
/// d'un operateur a 9 points (ex. PolarTensorKrylovSolver) -- soit correct meme en MULTI-BOX.
inline void fill_physical_bc(MultiFab& mf, const Box2D& domain,
                             const BCRec& bc) {
  const int ng = mf.n_grow();
  if (ng == 0) return;
  // Tout periodique : fill_boundary a deja tout fait, rien a lire/ecrire ici (et on
  // evite une barriere inutile sur le chemin chaud de la multigrille periodique).
  if (bc.xlo == BCType::Periodic && bc.xhi == BCType::Periodic &&
      bc.ylo == BCType::Periodic && bc.yhi == BCType::Periodic)
    return;
  // Bords physiques sur DEVICE (for_each_cell -> kernel) : ghost = cellule miroir (Foextrap : copie
  // de la 1ere interne ; Dirichlet : 2 v - reflexion). Indice ghost <-> couche : pour x bas, i = lo-k
  // donc le miroir Dirichlet est 2 lo - i - 1 (k = lo - i). Plus de device_fence ni d'acces hote : ces
  // kernels s'ordonnent apres copy_shifted (meme espace d'execution), et les faces y (i ETENDU pour
  // les coins) s'ordonnent apres les faces x sur le meme flux.
  const int nc = mf.ncomp();
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D v = F.box();
    Array4 a = F.array();

    // --- faces x, sur la plage j ETENDUE (j-ghosts inclus) ---
    // On etend la plage j aux GHOSTS en y/theta (j de v.lo[1]-ng a v.hi[1]+ng) au lieu de la seule
    // plage VALIDE. Raison (coin du stencil a 9 points, multi-box) : un terme CROISE (a_rt/a_tr de
    // l'operateur polaire) lit les voisins DIAGONAUX p(i+-1, j+-1) -> le ghost de COIN (x-physique
    // CROISE y-ghost) doit etre rempli. Quand y/theta est PERIODIQUE ou borde une box VOISINE,
    // fill_boundary a deja rempli la ligne j-ghost pour les colonnes x INTERIEURES ; la reflexion
    // x-physique (qui lit a(lo, j) / a(2 lo - i - 1, j) a la MEME j) etend donc correctement le bord
    // radial dans la halo y. Sans cette extension, le coin (x-ghost, y-ghost) reste a 0 et le terme
    // croise est FAUX au bord de box (divergence multi-box, cf. test_polar_schur_multibox). La plage
    // VALIDE seule suffisait en 5-points (pas de lecture diagonale) ; ce coin n'etait jamais lu.
    // NOTE : un coin DOUBLE-physique (x ET y non periodiques) est ensuite ECRASE par la passe y (i
    // etendu, plus bas, qui s'execute APRES) -> comportement cartesien inchange (y l'emporte). En
    // y-physique on lit ici a(lo, j-ghost) potentiellement non rempli, mais le resultat est ecrase :
    // sans effet sur la valeur finale du coin. Mono-box theta periodique : seuls les coins
    // (precedemment a 0, jamais lus en 5-points) changent -> bit-identique pour tout stencil <=
    // 9-points dont le 5-points cartesien (la nouvelle valeur de coin n'est lue que par un 9-points).
    const int jglo = v.lo[1] - ng, jghi = v.hi[1] + ng;
    if (bc.xlo != BCType::Periodic && v.lo[0] == domain.lo[0]) {
      const int lo = domain.lo[0];
      const bool foe = bc.xlo == BCType::Foextrap;
      const Real val = bc.xlo_val;
      for_each_cell(Box2D{{lo - ng, jglo}, {lo - 1, jghi}},
                    detail::BCFaceXLoKernel{a, nc, lo, foe, val});
    }
    if (bc.xhi != BCType::Periodic && v.hi[0] == domain.hi[0]) {
      const int hi = domain.hi[0];
      const bool foe = bc.xhi == BCType::Foextrap;
      const Real val = bc.xhi_val;
      for_each_cell(Box2D{{hi + 1, jglo}, {hi + ng, jghi}},
                    detail::BCFaceXHiKernel{a, nc, hi, foe, val});
    }

    // --- faces y, sur la plage i ETENDUE (coins via les ghosts-x deja remplis) ---
    const int iglo = v.lo[0] - ng, ighi = v.hi[0] + ng;
    if (bc.ylo != BCType::Periodic && v.lo[1] == domain.lo[1]) {
      const int lo = domain.lo[1];
      const bool foe = bc.ylo == BCType::Foextrap;
      const Real val = bc.ylo_val;
      for_each_cell(Box2D{{iglo, lo - ng}, {ighi, lo - 1}},
                    detail::BCFaceYLoKernel{a, nc, lo, foe, val});
    }
    if (bc.yhi != BCType::Periodic && v.hi[1] == domain.hi[1]) {
      const int hi = domain.hi[1];
      const bool foe = bc.yhi == BCType::Foextrap;
      const Real val = bc.yhi_val;
      for_each_cell(Box2D{{iglo, hi + 1}, {ighi, hi + ng}},
                    detail::BCFaceYHiKernel{a, nc, hi, foe, val});
    }
  }
}

// Remplissage complet des ghosts : interieur + periodique, puis bord physique.
/// Remplissage COMPLET des ghosts : fill_boundary (interieur + periodique, periodicite deduite de
/// @p bc) PUIS fill_physical_bc (bords physiques). Point d'entree usuel avant un assemblage de residu.
inline void fill_ghosts(MultiFab& mf, const Box2D& domain, const BCRec& bc) {
  Periodicity per{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  fill_boundary(mf, domain, per);
  fill_physical_bc(mf, domain, bc);
}

}  // namespace adc
