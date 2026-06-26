#!/usr/bin/env python3
"""patch_boxes() : la geometrie index-space des patchs fins AMR, expose a Python.

Verifie que :
  1. patch_boxes() renvoie une liste de tuples (level, ilo, jlo, ihi, jhi) ;
  2. son cardinal == n_patches() (le COMPTE et les BOITES viennent du MEME box_array fin) ;
  3. chaque box est coherente : level >= 1, coins inclusifs ilo <= ihi, jlo <= jhi, dans les bornes
     de l'espace d'indices du niveau (0 .. n*2^level - 1) ;
  4. patch_rectangles() (facade) convertit en [0, L]^2 sans deborder ;
  5. mono-bloc (AmrCouplerMP) ET multi-blocs (AmrRuntime) repondent tous deux ;
  6. la query ne perturbe pas l'integration (densite/n_patches inchanges apres l'appel).

Lancement : PYTHONPATH=<build>/python python3 python/tests/test_amr_patch_boxes.py
"""
import sys

import numpy as np

import pops


def _band(n, L=1.0, width=0.06, floor=1.0, amp=1.0):
    xs = (np.arange(n) + 0.5) / n * L
    X, Y = np.meshgrid(xs, xs, indexing="xy")  # field[j, i] : X varie selon i, Y selon j
    y0 = 0.5 * L + 0.04 * np.cos(2.0 * np.pi * 2.0 * X / L)
    return np.ascontiguousarray(floor + amp * np.exp(-((Y - y0) ** 2) / width ** 2))


def _check_boxes(sim, n, L, tag):
    boxes = sim.patch_boxes()
    npatch = sim.n_patches()
    assert isinstance(boxes, list), f"{tag} : patch_boxes() doit renvoyer une liste"
    assert len(boxes) == npatch, f"{tag} : len(patch_boxes)={len(boxes)} != n_patches()={npatch}"
    for b in boxes:
        assert len(b) == 5, f"{tag} : box mal formee {b!r} (attendu (level,ilo,jlo,ihi,jhi))"
        level, ilo, jlo, ihi, jhi = b
        assert level >= 1, f"{tag} : un patch fin a level={level} (attendu >= 1)"
        assert ilo <= ihi and jlo <= jhi, f"{tag} : coins non ordonnes {b!r}"
        lim = n << level  # n * 2^level cellules par direction au niveau fin
        assert 0 <= ilo and ihi < lim, f"{tag} : i hors bornes [0,{lim}) : {b!r}"
        assert 0 <= jlo and jhi < lim, f"{tag} : j hors bornes [0,{lim}) : {b!r}"
    # patch_rectangles : conversion physique, dans [0, L]^2 (tolerance flottante)
    rects = sim.patch_rectangles()
    assert len(rects) == npatch, f"{tag} : len(patch_rectangles)={len(rects)} != n_patches()={npatch}"
    eps = 1e-9
    for (x0, y0, w, h) in rects:
        assert w > 0 and h > 0, f"{tag} : rectangle degenere ({x0},{y0},{w},{h})"
        assert -eps <= x0 and x0 + w <= L + eps, f"{tag} : x deborde [0,L] : ({x0},{w})"
        assert -eps <= y0 and y0 + h <= L + eps, f"{tag} : y deborde [0,L] : ({y0},{h})"
    return boxes, npatch


def main():
    n, L = 64, 1.0
    ne0 = _band(n, L)

    # --- MONO-BLOC : AmrCouplerMP ---
    mono = pops.AmrSystem(n=n, L=L, regrid_every=10, periodic=True)
    mono.add_block("ne",
                   model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                                   source=pops.NoSource(),
                                   elliptic=pops.BackgroundDensity(alpha=1.0, n0=float(ne0.mean()))),
                   spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    mono.set_refinement(threshold=0.05)
    mono.set_poisson(rhs="charge_density", solver="geometric_mg")
    mono.set_density("ne", ne0)
    for _ in range(8):
        mono.step_cfl(0.4)
    boxes_m, np_m = _check_boxes(mono, n, L, "mono-bloc")
    assert np_m >= 1, "mono-bloc : raffinement inactif (n_patches == 0)"
    # idempotence : l'appel query ne perturbe pas l'etat (densite identique avant/apres)
    d_before = np.asarray(mono.density("ne")).copy()
    _ = mono.patch_boxes()
    d_after = np.asarray(mono.density("ne"))
    assert np.array_equal(d_before, d_after), "mono-bloc : patch_boxes() a perturbe la densite"

    # --- MULTI-BLOCS : AmrRuntime (>= 2 add_block, hierarchie figee regrid_every=0) ---
    ne1 = _band(n, L)
    ne2 = _band(n, L) * 0.5 + 0.5
    multi = pops.AmrSystem(n=n, L=L, regrid_every=0, periodic=True)
    for nm, arr in (("a", ne1), ("b", ne2)):
        multi.add_block(nm,
                        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                                        source=pops.NoSource(),
                                        elliptic=pops.BackgroundDensity(alpha=1.0, n0=float(arr.mean()))),
                        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    multi.set_refinement(threshold=0.05)
    multi.set_poisson(rhs="charge_density", solver="geometric_mg")
    multi.set_density("a", ne1)
    multi.set_density("b", ne2)
    for _ in range(4):
        multi.step_cfl(0.4)
    _check_boxes(multi, n, L, "multi-blocs")

    print("OK test_amr_patch_boxes (mono=%d patchs, boxes coherentes index+physique)" % np_m)


if __name__ == "__main__":
    sys.exit(main())
