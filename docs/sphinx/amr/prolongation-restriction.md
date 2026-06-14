# Prolongation and restriction


The transfers between levels are the two classical conservative operators:

- **Restriction** (`average_down`, fine -> coarse): each coarse cell receives the
  average of the fine cells it covers. This is what keeps the coarse consistent under
  a fine patch (and what prevents a mass diagnostic on the coarse alone from counting a phantom
  value under the patch).
- **Prolongation / injection** (`interpolate`, coarse -> fine): a new fine patch is
  filled by interpolation from the parent where it did not yet exist, and by carrying over
  the existing fine data where the old patch covers the new one. This injection is
  conservative (the average of the reconstructed children equals the parent).

When advancing in time, the coarse takes one step `dt` while the fine level takes `r` substeps of
`dt/r` (Berger-Oliger subcycling), each respecting its own CFL. The coarse-fine ghosts of the
fine level are filled by space-time interpolation from the coarse. Core code:
`mesh/refinement.hpp` (`average_down` / `interpolate`), engine `numerics/time/amr_reflux_mf.hpp`
(`advance_amr`). Detail: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 13.
