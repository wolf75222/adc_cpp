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

enum class BCType { Periodic, Foextrap, Dirichlet };

struct BCRec {
  BCType xlo = BCType::Periodic, xhi = BCType::Periodic;
  BCType ylo = BCType::Periodic, yhi = BCType::Periodic;
  Real xlo_val = 0, xhi_val = 0, ylo_val = 0, yhi_val = 0;
};

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

    // --- faces x, sur la plage j valide ---
    if (bc.xlo != BCType::Periodic && v.lo[0] == domain.lo[0]) {
      const int lo = domain.lo[0];
      const bool foe = bc.xlo == BCType::Foextrap;
      const Real val = bc.xlo_val;
      for_each_cell(Box2D{{lo - ng, v.lo[1]}, {lo - 1, v.hi[1]}}, [=] ADC_HD(int i, int j) {
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = foe ? a(lo, j, c) : 2 * val - a(2 * lo - i - 1, j, c);
      });
    }
    if (bc.xhi != BCType::Periodic && v.hi[0] == domain.hi[0]) {
      const int hi = domain.hi[0];
      const bool foe = bc.xhi == BCType::Foextrap;
      const Real val = bc.xhi_val;
      for_each_cell(Box2D{{hi + 1, v.lo[1]}, {hi + ng, v.hi[1]}}, [=] ADC_HD(int i, int j) {
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = foe ? a(hi, j, c) : 2 * val - a(2 * hi - i + 1, j, c);
      });
    }

    // --- faces y, sur la plage i ETENDUE (coins via les ghosts-x deja remplis) ---
    const int iglo = v.lo[0] - ng, ighi = v.hi[0] + ng;
    if (bc.ylo != BCType::Periodic && v.lo[1] == domain.lo[1]) {
      const int lo = domain.lo[1];
      const bool foe = bc.ylo == BCType::Foextrap;
      const Real val = bc.ylo_val;
      for_each_cell(Box2D{{iglo, lo - ng}, {ighi, lo - 1}}, [=] ADC_HD(int i, int j) {
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = foe ? a(i, lo, c) : 2 * val - a(i, 2 * lo - j - 1, c);
      });
    }
    if (bc.yhi != BCType::Periodic && v.hi[1] == domain.hi[1]) {
      const int hi = domain.hi[1];
      const bool foe = bc.yhi == BCType::Foextrap;
      const Real val = bc.yhi_val;
      for_each_cell(Box2D{{iglo, hi + 1}, {ighi, hi + ng}}, [=] ADC_HD(int i, int j) {
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = foe ? a(i, hi, c) : 2 * val - a(i, 2 * hi - j + 1, c);
      });
    }
  }
}

// Remplissage complet des ghosts : interieur + periodique, puis bord physique.
inline void fill_ghosts(MultiFab& mf, const Box2D& domain, const BCRec& bc) {
  Periodicity per{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  fill_boundary(mf, domain, per);
  fill_physical_bc(mf, domain, bc);
}

}  // namespace adc
