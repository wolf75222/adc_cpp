# Prise en main

Cette section est le guide utilisateur de `adc_cpp` : elle part de zero (qu'est-ce que
c'est, comment le construire) et conduit jusqu'a une simulation diocotron complete,
reproductible de bout en bout. Elle ne presuppose ni connaissance du coeur C++, ni du DSL.

Le parcours conseille, dans l'ordre :

- [Presentation](presentation.md) : ce qu'est `adc_cpp`, son perimetre honnete, ses couches.
- [Installation](installation.md) : build CMake du coeur, du module Python, et le piege de
  l'interpreteur (le `.so` compile est lie a une version de Python).
- [Premier run](first_run.md) : le plus petit exemple Python, copiable tel quel.
- [Verifier son backend](backend.md) : savoir quel backend parallele tourne
  (serie par defaut ; Kokkos / MPI sont des configs de build ; GPU = ROMEO uniquement).
- [Organisation des depots](organisation.md) : `adc_cpp` (la lib, agnostique) vs `adc_cases`
  (les scenarios nommes : diocotron, euler_poisson...).
- [Tutoriel A->Z](tutorial.md) : le tutoriel canonique en 18 etapes, de `git clone` a la
  comparaison uniforme/AMR, avec figures et GIF generes.
- [Windows (WSL2)](windows_wsl2.md) : guide de zero pour un PC Windows 11, du `wsl --install`
  a un cas `adc_cases`, avec les ecarts par rapport au chemin Linux.
- [Windows natif](windows_native.md) : statut et feuille de route du portage natif (sans WSL2) ;
  perimetre v1/v2 et etat reel.

Pour la reference de conception (contributeurs), chaque page renvoie aux documents `docs/*.md`
du depot ; cette section les resume pour un nouvel utilisateur sans les dupliquer.

```{toctree}
:maxdepth: 1
:hidden:

presentation
installation
first_run
backend
organisation
tutorial
windows_wsl2
windows_native
```
