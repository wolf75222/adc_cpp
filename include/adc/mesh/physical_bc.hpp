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
  // GPU : ces boucles HOTE lisent/ecrivent les ghosts que copy_shifted (kernel
  // device, dans fill_boundary juste avant) vient potentiellement d'ecrire ->
  // barriere obligatoire avant tout acces hote a la memoire unifiee (no-op hors GPU).
  device_fence();
  const int nc = mf.ncomp();

  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D v = F.box();
    Array4 a = F.array();

    // --- faces x, sur la plage j valide ---
    if (bc.xlo != BCType::Periodic && v.lo[0] == domain.lo[0]) {
      for (int c = 0; c < nc; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int k = 1; k <= ng; ++k) {
            int g = domain.lo[0] - k;
            a(g, j, c) = (bc.xlo == BCType::Foextrap)
                             ? a(domain.lo[0], j, c)
                             : 2 * bc.xlo_val - a(domain.lo[0] + k - 1, j, c);
          }
    }
    if (bc.xhi != BCType::Periodic && v.hi[0] == domain.hi[0]) {
      for (int c = 0; c < nc; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int k = 1; k <= ng; ++k) {
            int g = domain.hi[0] + k;
            a(g, j, c) = (bc.xhi == BCType::Foextrap)
                             ? a(domain.hi[0], j, c)
                             : 2 * bc.xhi_val - a(domain.hi[0] - k + 1, j, c);
          }
    }

    // --- faces y, sur la plage i ETENDUE (coins via les ghosts-x deja remplis) ---
    const int iglo = v.lo[0] - ng, ighi = v.hi[0] + ng;
    if (bc.ylo != BCType::Periodic && v.lo[1] == domain.lo[1]) {
      for (int c = 0; c < nc; ++c)
        for (int i = iglo; i <= ighi; ++i)
          for (int k = 1; k <= ng; ++k) {
            int g = domain.lo[1] - k;
            a(i, g, c) = (bc.ylo == BCType::Foextrap)
                             ? a(i, domain.lo[1], c)
                             : 2 * bc.ylo_val - a(i, domain.lo[1] + k - 1, c);
          }
    }
    if (bc.yhi != BCType::Periodic && v.hi[1] == domain.hi[1]) {
      for (int c = 0; c < nc; ++c)
        for (int i = iglo; i <= ighi; ++i)
          for (int k = 1; k <= ng; ++k) {
            int g = domain.hi[1] + k;
            a(i, g, c) = (bc.yhi == BCType::Foextrap)
                             ? a(i, domain.hi[1], c)
                             : 2 * bc.yhi_val - a(i, domain.hi[1] - k + 1, c);
          }
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
