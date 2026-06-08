#pragma once

namespace adc {

/// Empreinte INDEX-SPACE d'un patch fin AMR, exposee a Python par AmrSystem::patch_boxes().
///
/// (level, ilo, jlo, ihi, jhi) : le niveau (0 = grossier ; >= 1 = fin) et les coins lo/hi de la box
/// dans l'espace d'indices du niveau (n << level cellules par direction, ratio 2 par niveau). Coins
/// INCLUSIFS (convention Box2D / AMReX) : la box couvre (ihi - ilo + 1) x (jhi - jlo + 1) cellules.
///
/// POD trivial : c'est une vue de lecture sur les boites deja stockees (le meme BoxArray que lit
/// n_patches()), recoltee entre les pas (query) -> aucun cout sur le chemin chaud. La conversion en
/// coordonnees physiques [0, L]^2 se fait cote Python (qui connait n via nx() et L) : dx = L / (n <<
/// level), x0 = ilo * dx, largeur = (ihi - ilo + 1) * dx.
struct PatchBox {
  int level;
  int ilo;
  int jlo;
  int ihi;
  int jhi;
};

}  // namespace adc
