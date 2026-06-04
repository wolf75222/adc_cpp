# Runs ROMEO (GH200 + EPYC)

Reproduction du taux de croissance diocotron (Hoffart, arXiv:2510.11808, mode 4, cible
analytique 0.911) sur le supercalculateur ROMEO (URCA). Journal des jobs SLURM :
[`romeo/HERO_RESULTS.md`](../romeo/HERO_RESULTS.md). Récit complet :
[`tutorials/10_diocotron_reproduction.md`](../tutorials/10_diocotron_reproduction.md).

## Convergence du taux (WENO5-Z + SSPRK3, EPYC)

| Figure | Source | Conclusion |
|---|---|---|
| ![convergence WENO5](romeo_highorder_convergence.png) | job 613961, WENO5-Z + SSPRK3, modes 3/4/5 × eff 256/512/1024 | sur-tir ~+8 % **plat en résolution** : limite géométrique (bord cartésien), pas numérique |
| ![croissance mode 4](romeo_growth_mode4.png) | `ring_amp.csv` réels, mode 4 | eff 512 et 1024 **confondues** : taux convergé en résolution effective |
| ![efficacité AMR](romeo_amr_efficiency.png) | job 613945, colonne AMR vs uniforme | même taux pour **~40 % des cellules** (AMR multi-niveau VanLeer) |

La colonne sur AMR avec Poisson multi-niveau égale l'uniforme à résolution effective égale
pour ~41-44 % des cellules (taux mode 4 normalisé `0.42`/`0.526`/`0.563`/`0.592` aux eff
192/256/320/448). La cible analytique `0.911` n'est pas atteinte : le sur-tir est plat en
résolution, ce qui pointe la géométrie cartésienne en escalier (anneau et paroi circulaires
sur grille carrée), pas la diffusion numérique. Prochaine étape : cut-cell ou coordonnées
polaires.

## GPU : bit-identique CPU/GPU

Sur 1 GH200 (`armgpu`, Kokkos/CUDA 12.6), le pas couplé complet + l'AMR multi-patch
tournent sur GPU :

- checksum `diocotron_amr_kokkos` = **4394594.404318** exactement égal au CPU → bit-identique ;
- dérive de masse AMR sur GPU : `2.2e-16` ;
- garde-fou `romeo/sanitizer.sbatch` (compute-sanitizer memcheck/initcheck/synccheck) : 0 erreur.
