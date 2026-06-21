#pragma once

#include <adc/amr/cluster.hpp>  // berger_rigoutsos, ClusterParams
#include <adc/amr/refinement_ratio.hpp>
#include <adc/amr/regrid.hpp>   // tag_cells, grow_tags
#include <adc/amr/tag_box.hpp>  // TagBox
#include <adc/core/types.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, mf_find_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence (barrier after async parallel_copy under Cuda)
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/parallel/comm.hpp>    // n_ranks (explicit include, no longer an indirect path)

#include <algorithm>
#include <utility>
#include <vector>

/// @file
/// @brief amr_regrid_finest: Berger-Rigoutsos regrid of the finest level (responsibility b).
///
/// Free function template on the criterion, modeled on the STYLE of amr/regrid.hpp (regrid_level) but
/// NOT merged: different invariants (level fk coords = parent x2, nesting margin clamp,
/// carry-over of the old fine). Body moved as-is from AmrCouplerMP::regrid: same tagging,
/// clustering, clamp, parent interp then fine carry-over, swap + aux realloc. Does not assume single-rank
/// (DistributionMapping built with n_ranks()). Under a DISTRIBUTED coarse level, the global OR of the tags
/// (all_reduce_or) guarantees IDENTICAL fine patches on all ranks (otherwise incompatible dmaps).

namespace adc {

// ------------------------------------------------------------------------------------------------
// REFACTOR (docs/AMR_REGRID_UNION_TAGS_DESIGN.md section 6): amr_regrid_finest split into TWO
// responsibilities so the multi-block UNION regrid (AmrRuntime::regrid) can re-grid
// SEVERAL fields on a SINGLE layout imposed from outside (the same one for all blocks):
//   (1) regrid_compute_fine_layout: tags of a parent -> grow -> all_reduce_or (if distributed) ->
//       berger_rigoutsos -> nesting clamp -> (fine BoxArray, DistributionMapping). This is the
//       COMPUTATION of the layout, done ONCE on the union of the tags by the multi-block path.
//   (2) regrid_field_on_layout: takes an IMPOSED (fb, dmap) and rebuilds ONE fine MultiFab (parent
//       interp + fine carry-over), exactly the BODY of the old amr_regrid_finest, but without the
//       layout computation. Called PER BLOCK on the same union layout.
// amr_regrid_finest stays the CHAINING of (1) then (2) on a SINGLE block -> the single-block path
// (AmrCouplerMP::regrid) stays BIT-IDENTICAL (same operations, same order).
// ------------------------------------------------------------------------------------------------

/// Compute the fine layout (BoxArray + DistributionMapping) of a Berger-Rigoutsos regrid from the
/// @p grown tags ALREADY dilated (grow_tags) on the PARENT domain @p pdom. @p pk: parent level (the
/// cross-rank MPI reduction only happens for pk==0 distributed); @p margin: nesting; @p coarse_replicated:
/// ownership policy of level 0. Fine level coords = parent x2. Returns an EMPTY BoxArray if there is nothing
/// to refine (the caller then keeps the current grid). MPI-safe: under a distributed coarse level, the global OR
/// of the tags (all_reduce_or) guarantees IDENTICAL patches on all ranks (otherwise incompatible dmaps).
inline std::pair<BoxArray, DistributionMapping> regrid_compute_fine_layout(
    TagBox grown, const Box2D& pdom, int pk, int margin, bool coarse_replicated = true) {
  const int PNX = pdom.nx(), PNY = pdom.ny();
  // DISTRIBUTED COARSE (pk == 0 && !coarse_replicated): each rank only tagged its LOCAL boxes.
  // Global OR before the clustering -> all ranks start from the SAME tag grid (otherwise the
  // fine BoxArray would differ per rank -> incompatible dmaps, MPI desynchronized). Replicated: the
  // grid is already complete on each rank (no-op, all_reduce_or would be the identity).
  if (pk == 0 && !coarse_replicated)
    all_reduce_or_inplace(grown.t.data(), static_cast<int>(grown.t.size()));
  std::vector<Box2D> cl = berger_rigoutsos(grown, ClusterParams{});
  std::vector<Box2D> fb;  // fine patches (fine level coords = parent x2)
  for (Box2D b : cl) {
    b.lo[0] = std::max(b.lo[0], margin); b.lo[1] = std::max(b.lo[1], margin);
    b.hi[0] = std::min(b.hi[0], PNX - 1 - margin); b.hi[1] = std::min(b.hi[1], PNY - 1 - margin);
    if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
    fb.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
  }
  if (fb.empty()) return {BoxArray{}, DistributionMapping{}};  // nothing to refine
  BoxArray ba(fb);
  return {ba, DistributionMapping(static_cast<int>(ba.size()), n_ranks())};
}

/// Rebuild ONE fine MultiFab on the IMPOSED layout @p fb / @p dmap (the same one for all blocks in
/// multi-block): (a) piecewise-constant interpolation from the parent @p par where the new patch
/// is not covered by the old fine, (b) carry-over of the existing fine data @p old where the old
/// patch covers the new one. @p ngf: fine ghost width (inherited from the old level being replaced);
/// @p coarse_replicated: ownership policy of level 0 (distributed parent -> parallel_copy + fence).
/// This is the BODY of the old amr_regrid_finest, without the layout computation. The parent's pk is passed
/// to decide whether the parent is replicated (pk != 0 -> always distributed). Returns the new MultiFab.
inline MultiFab regrid_field_on_layout(const BoxArray& fb, const DistributionMapping& dmap,
                                       const MultiFab& par, const MultiFab& old, int pk, int ngf,
                                       bool coarse_replicated = true) {
  MultiFab nU(fb, dmap, old.ncomp(), ngf);
  const int ncf = nU.ncomp();
  // DISTRIBUTED parent (de-replicated coarse): par.fab only holds the LOCAL boxes, so
  // mf_find_box would return -1 for a coarse cell owned by a REMOTE rank and the patch
  // would stay uninitialized there. We bring the needed parent regions onto a LOCAL
  // child-coarsen grid (coarsen of the fine BoxArray) via parallel_copy, then interpolate from
  // it. Replicated parent: par is entirely local, direct read via mf_find_box.
  const bool par_replicated = (pk != 0) || coarse_replicated;
  MultiFab parloc;
  if (!par_replicated) {
    parloc = MultiFab(coarsen(nU.box_array(), kAmrRefRatio), nU.dmap(), par.ncomp(), 0);
    parallel_copy(parloc, par);
    // parallel_copy launches async kernels under Cuda and, at np=1, returns WITHOUT a fence: without this
    // fence the read of parloc below would read device memory not yet written -> NaN.
    device_fence();
  }
  for (int li = 0; li < nU.local_size(); ++li) {
    Array4 a = nU.fab(li).array();
    const Box2D nb = nU.box(li);
    if (par_replicated) {
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)  // 1) interp from the parent (local)
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i) {
          const int pb = mf_find_box(par, coarsen_index(i, kAmrRefRatio), coarsen_index(j, kAmrRefRatio));
          if (pb < 0) continue;
          const ConstArray4 pp = par.fab(pb).const_array();
          for (int k = 0; k < ncf; ++k)
            a(i, j, k) = pp(coarsen_index(i, kAmrRefRatio), coarsen_index(j, kAmrRefRatio), k);
        }
    } else {
      const ConstArray4 pp = parloc.fab(li).const_array();  // local child-coarsen grid
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i)
          for (int k = 0; k < ncf; ++k)
            a(i, j, k) = pp(coarsen_index(i, kAmrRefRatio), coarsen_index(j, kAmrRefRatio), k);
    }
    for (int ol = 0; ol < old.local_size(); ++ol) {  // 2) carry-over of the fine data
      const ConstArray4 o = old.fab(ol).const_array();
      const Box2D inter = nb.intersect(old.box(ol));
      if (inter.empty()) continue;
      for (int j = inter.lo[1]; j <= inter.hi[1]; ++j)
        for (int i = inter.lo[0]; i <= inter.hi[0]; ++i)
          for (int k = 0; k < ncf; ++k) a(i, j, k) = o(i, j, k);
    }
  }
  return nU;
}

/// Regrid the finest level (L.back()) by Berger-Rigoutsos on the criterion @p crit applied to the
/// parent: rebuilds the patches (fine data carry-over otherwise parent interp) + the aux. @p grow:
/// tag dilation; @p margin: nesting; @p aux_ncomp: rebuilt aux width;
/// @p coarse_replicated: ownership policy of level 0. NO-OP if < 2 levels or no patch.
///
/// aux_ncomp: width of the rebuilt aux channel (default kAuxBaseComps = 3). The coupler,
/// which knows the Model, propagates aux_comps<Model>() so that a model reading extra
/// fields (B_z, ...) keeps the room after regrid. Since the Model is not in scope here (free
/// function on the criterion only), the width is PROPAGATED as a parameter; default 3 ->
/// MultiFab(..., 3, 1) allocation strictly bit-identical to the historical one.
template <class Crit>
void amr_regrid_finest(std::vector<AmrLevelMP>& L, std::vector<MultiFab>& aux,
                       const Box2D& dom, Crit crit, int grow, int margin,
                       int aux_ncomp = kAuxBaseComps, bool coarse_replicated = true) {
  const int nlev = static_cast<int>(L.size());
  if (nlev < 2) return;
  const int fk = nlev - 1, pk = fk - 1;  // fine and its parent
  const int PNX = dom.nx() << pk, PNY = dom.ny() << pk;
  const Box2D pdom = Box2D::from_extents(PNX, PNY);
  TagBox tags = tag_cells(L[pk].U, pdom, crit);
  TagBox grown = grow_tags(tags, grow, pdom);
  // (1) Compute the fine layout (tags -> grow [already done] -> all_reduce_or -> clustering -> clamp).
  auto [fb, dmap] = regrid_compute_fine_layout(std::move(grown), pdom, pk, margin, coarse_replicated);
  if (fb.size() == 0) return;  // nothing to refine: keep the current grid
  // The new patches INHERIT the ghost width of the level being replaced (not a frozen 1): a
  // level rebuilt in 2nd-order MUSCL (Minmod / VanLeer) carries 2 ghosts, which the regrid must
  // preserve, otherwise the reconstruction would read out of bounds after re-refinement.
  const int ngf = L[fk].U.n_grow();
  // (2) Re-grid the field U on this layout (parent interp + fine carry-over): SAME body as before,
  // so the single-block path stays BIT-IDENTICAL (chaining of (1) then (2) on a single block).
  L[fk].U = regrid_field_on_layout(fb, dmap, L[pk].U, L[fk].U, pk, ngf, coarse_replicated);
  aux[fk] = MultiFab(L[fk].U.box_array(), L[fk].U.dmap(), aux_ncomp, 1);  // stable address
  L[fk].aux = &aux[fk];
}

}  // namespace adc
