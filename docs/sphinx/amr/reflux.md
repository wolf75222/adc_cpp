# Reflux


At the interface between a fine level and the coarse, the two levels compute different face fluxes:
conservation would be broken if no care were taken. The reflux corrects the
coarse cell by the difference between the fine flux (integrated over the `r` substeps) and the coarse
flux:

$$U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_s \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

The reflux is coverage-aware: a coverage mask (`CoverageMask`) built on the global BoxArray
avoids the double correction at the joint between two neighboring fine patches (fine-fine interface, where
it must not reflux) and directs the correction to the right parent box when the coarse
is itself multi-box. In multi-block, each block has its own flux registers: conservation
is verified block by block. Under MPI, the flux register is gathered by
`all_reduce_sum` and the remote reflux goes through `parallel_copy`. Roles promoted to types:
`FluxRegister` (accumulation of face fluxes), `CoverageMask` (covered cells). Detail:
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 13-14, [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8.
