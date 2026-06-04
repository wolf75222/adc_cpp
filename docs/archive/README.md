# Archive documentaire

Documents conserves pour la trace mais HORS de la navigation principale : ce sont
soit des notes de planification et de vision (l'etat reel est decrit par
[`../ARCHITECTURE.md`](../ARCHITECTURE.md), [`../ALGORITHMS.md`](../ALGORITHMS.md) et
[`../GPU_RUNTIME_PORT.md`](../GPU_RUNTIME_PORT.md)), soit des notes applicatives dont le
contenu execute vit maintenant dans le depot [`adc_cases`](https://github.com/wolf75222/adc_cases).

Ils ne sont pas tenus a jour. En cas de divergence, la documentation hors archive fait foi.

## Planification et vision

| Fichier | Contenu |
|---|---|
| `ROADMAP.md` | liste vivante des intentions, fait / a faire |
| `TODO.md` | taches du chantier `PhysicalModel` -> systeme multi-especes |
| `WORK_TODO.md` | liste de travail du chantier aux extensible / parite AMR / runtime |
| `ETAT_DES_LIEUX.md` | synthese d'audits croises (architecture, numerique, robustesse, perf, HPC) |
| `ARCHITECTURE_CIBLE.md` | doc de vision (north star), anterieure a l'etat actuel |
| `DESIGN_MULTISPECIES.md` | conception du cap multi-especes (session tableau) |
| `PLAN_VARIABLES_EPM.md` | plan du chantier niveau Variables + EPM |

## Notes applicatives (scenarios, runs)

Le coeur `adc_cpp` est agnostique au modele : ces notes decrivent des SCENARIOS ou des
campagnes de mesure qui vivent cote application.

| Fichier | Contenu |
|---|---|
| `two_fluid_ap.md` | note de methode du schema deux-fluides AP (scenario, vit dans `adc_cases/two_fluid_ap/`) |
| `DIOCOTRON_GROWTH_RATE.md` | reproduction du taux de croissance diocotron vs Hoffart arXiv:2510.11808 |
| `HERO_RUN_AMR.md` | conception du hero-run AMR distribue sur ROMEO |
| `ROMEO.md` | journal des runs ROMEO (GH200 + EPYC) |
