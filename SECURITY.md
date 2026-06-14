# Politique de securite

`adc_cpp` est un code de recherche scientifique (solveur numerique). Il n'a pas vocation a
traiter des donnees sensibles ni a etre expose en service, mais les signalements de
vulnerabilite ou de probleme de surete (lecture hors bornes, UB, deni de service par
entree malformee...) sont les bienvenus.

## Signaler

- De preference via le **private vulnerability reporting** de GitHub
  (onglet *Security* du depot, *Report a vulnerability*).
- A defaut, ouvrir une issue ; pour un sujet sensible, indiquer seulement le type de
  probleme et demander un canal prive avant de donner les details.

Suivi **best-effort** : projet maintenu par une personne, sans engagement de delai. Les
corrections sont publiees comme des correctifs ordinaires sur `master`.

## Perimetre

Le coeur numerique, les bindings Python et le chemin DSL (compilation de code genere a
l'execution). Les dependances tierces (Kokkos, OpenMPI, pybind11...) relevent de leurs
propres mainteneurs.
