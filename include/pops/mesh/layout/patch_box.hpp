#pragma once

/// @file
/// @brief PatchBox: index-space footprint of an AMR fine patch, exposed read-only to Python.
///
/// Layer: `include/pops/mesh`.
/// Role: trivial POD (level, ilo, jlo, ihi, jhi) describing the position of a box in the index
/// space of its level; a read-only view onto the boxes already stored, harvested between steps
/// by AmrSystem::patch_boxes() (no cost on the hot path).
/// Contract: INCLUSIVE corners (Box2D / AMReX convention); the conversion to physical coordinates
/// [0, L]^2 happens on the Python side, which knows n and L.
///
/// Invariants:
/// - level 0 = coarse, level >= 1 = fine (ratio 2 per level, n << level cells per direction);
/// - the box covers (ihi - ilo + 1) x (jhi - jlo + 1) cells.

namespace pops {

/// INDEX-SPACE footprint of an AMR fine patch, exposed to Python by AmrSystem::patch_boxes().
///
/// (level, ilo, jlo, ihi, jhi): the level (0 = coarse; >= 1 = fine) and the lo/hi corners of the
/// box in the index space of the level (n << level cells per direction, ratio 2 per level).
/// INCLUSIVE corners (Box2D / AMReX convention): the box covers (ihi - ilo + 1) x (jhi - jlo + 1)
/// cells.
///
/// Trivial POD: a read-only view onto the boxes already stored (the same BoxArray that n_patches()
/// reads), harvested between steps (query) -> no cost on the hot path. The conversion to physical
/// coordinates [0, L]^2 happens on the Python side (which knows n via nx() and L): dx = L / (n <<
/// level), x0 = ilo * dx, width = (ihi - ilo + 1) * dx.
struct PatchBox {
  int level;
  int ilo;
  int jlo;
  int ihi;
  int jhi;
};

}  // namespace pops
