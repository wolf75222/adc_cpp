/// @file
/// @brief Regrid dynamique : (re)construit un niveau fin a partir du tagging d'un niveau grossier.
///
/// Couche : `include/adc/amr` (primitives geometriques AMR).
/// Role : tague un niveau, regroupe les cellules taguees en boxes (Berger-Rigoutsos), raffine en
/// BoxArray du niveau suivant, reconstruit son MultiFab et le remplit (interpolation depuis le
/// grossier, puis copie de l'ancien fin la ou il existait pour preserver la precision).
/// Contrat : le critere de tagging est un predicat generique sur (ConstArray4, i, j) ; on reste
/// agnostique de la physique. Pour un critere a gradient, l'appelant remplit les ghosts avant.
///
/// Invariants :
/// - regrid conservatif = hierarchie commune, cellules co-localisees, regrid par union des tags ;
/// - le niveau fin est dans l'espace d'indices raffine du niveau grossier (refine(ref_ratio)) ;
/// - sans tag, le niveau fin (et les plus fins) est supprime.

#pragma once

#include <adc/amr/amr_hierarchy.hpp>
#include <adc/amr/cluster.hpp>
#include <adc/amr/tag_box.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>
#include <adc/parallel/comm.hpp>

#include <utility>
#include <vector>

// Regrid dynamique : tague un niveau, regroupe les cellules taguees en boxes
// (Berger-Rigoutsos), raffine en BoxArray du niveau suivant, reconstruit son
// MultiFab et le remplit (interpolation depuis le grossier, puis copie de
// l'ancien fin la ou il existait pour preserver la precision).
//
// Le critere de tagging est un predicat generique sur (ConstArray4, i, j) : on
// reste agnostique de la physique. Pour un critere a gradient, l'appelant
// remplit les ghosts avant.

namespace adc {

/// Marque les cellules valides ou le predicat est vrai, sur une TagBox couvrant le domaine.
/// @tparam Crit predicat (ConstArray4, i, j) -> bool, evalue sur les cellules valides de chaque fab.
/// @param mf champ source (local : ne parcourt que les fabs locaux du rang).
/// @param domain domaine couvert par la TagBox retournee (espace d'indices du niveau).
/// @return TagBox sur domain, marquee la ou crit est vrai.
template <class Crit>
TagBox tag_cells(const MultiFab& mf, const Box2D& domain, Crit crit) {
  TagBox tb(domain);
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const ConstArray4 a = f.const_array();
    const Box2D v = f.box();
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        if (crit(a, i, j)) tb(i, j) = 1;
  }
  return tb;
}

/// Dilate les tags de n cellules (voisinage carre), en restant dans le domaine.
/// @param n rayon de dilatation (buffer) ; sert au nesting et a anticiper le deplacement des structures.
/// @param domain borne le voisinage : aucun tag n'est pose hors du domaine.
/// @return nouvelle TagBox sur in.box, marquee sur l'union des voisinages carres des cellules taguees.
inline TagBox grow_tags(const TagBox& in, int n, const Box2D& domain) {
  TagBox out(in.box);
  const Box2D& b = in.box;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      if (in(i, j))
        for (int dj = -n; dj <= n; ++dj)
          for (int di = -n; di <= n; ++di) {
            const int ii = i + di, jj = j + dj;
            if (b.contains(ii, jj) && domain.contains(ii, jj)) out(ii, jj) = 1;
          }
  return out;
}

/// Parametres du regrid (objet de configuration).
struct RegridParams {
  int n_buffer = 1;        ///< rayon de dilatation des tags (grow_tags) avant clustering.
  ClusterParams cluster{};  ///< parametres du clustering Berger-Rigoutsos.
};

/// (Re)construit le niveau coarse_lev+1 a partir du tagging du niveau coarse_lev.
/// @tparam Crit predicat de tagging (ConstArray4, i, j) -> bool.
/// @param h hierarchie modifiee en place (niveau fin installe, ou niveaux plus fins supprimes si aucun tag).
/// @param coarse_lev niveau grossier source du tagging ; le niveau fin construit est coarse_lev+1.
/// @param rp parametres (buffer de tags, clustering).
/// Etapes : tag -> grow -> Berger-Rigoutsos -> refine(ref_ratio) -> interpolation depuis le grossier,
/// puis copie de l'ancien fin la ou il existait pour preserver la precision.
template <class Crit>
void regrid_level(AmrHierarchy& h, int coarse_lev, Crit crit,
                  const RegridParams& rp = {}) {
  const Box2D cdom = h.domain(coarse_lev);
  TagBox tags = tag_cells(h.data(coarse_lev), cdom, crit);
  TagBox grown = grow_tags(tags, rp.n_buffer, cdom);

  if (grown.count() == 0) {  // plus rien a raffiner
    h.clear_above(coarse_lev);
    return;
  }

  // boxes grossieres -> boxes fines (espace d'indices du niveau coarse_lev+1)
  std::vector<Box2D> cboxes = berger_rigoutsos(grown, rp.cluster);
  std::vector<Box2D> fboxes;
  fboxes.reserve(cboxes.size());
  for (const auto& b : cboxes) fboxes.push_back(b.refine(h.ref_ratio()));
  BoxArray fba(std::move(fboxes));

  MultiFab newfine(fba, DistributionMapping(fba.size(), n_ranks()), h.ncomp(),
                   h.n_grow());
  interpolate(h.data(coarse_lev), newfine, h.ref_ratio());
  if (h.num_levels() > coarse_lev + 1)
    parallel_copy(newfine, h.data(coarse_lev + 1));  // preserver l'ancien fin

  h.install_level(coarse_lev + 1, fba, std::move(newfine));
}

}  // namespace adc
