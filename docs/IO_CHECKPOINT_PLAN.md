# Sorties / checkpoint / restart -- plan de design (audit 2026-06, chantier 7)

Statut : **PLAN** (API cible + contraintes HPC + decoupage en PR). Rien n'est cable dans ce
document ; il fixe le contrat AVANT l'implementation pour eviter une API de sortie ad hoc par cas
(aujourd'hui chaque cas adc_cases ecrit ses .npy / .png a la main via `density()` / `get_state()`).

## API utilisateur cible

```python
sim.write("out/run", format="hdf5", step=k)     # un dump de visualisation (champ + metadonnees)
sim.write("out/run", format="vtk", step=k)      # idem, lisible par ParaView/VisIt sans h5py
sim.checkpoint("out/chk_000400")                # etat COMPLET redemarrable
sim.restart("out/chk_000400")                   # reprend exactement (memes blocs deja composes)
```

- `write` = SORTIE DE VISUALISATION : perte acceptable (sous-ensemble de champs, eventuellement
  sous-echantillonne), formats ouverts (HDF5/XDMF ou VTK), UNE cadence utilisateur (`step=` nomme le
  fichier, l'utilisateur pilote la frequence).
- `checkpoint` / `restart` = REPRISE EXACTE : tout ce qui est necessaire pour reprendre le run
  bit-pour-bit (modulo reductions non deterministes du backend), AUCUNE perte.

## Contenu minimal d'un checkpoint

| Element | Source actuelle | Note |
|---|---|---|
| temps `t` | `System::time()` | double exact |
| compteur `macro_step_` | Impl | OBLIGATOIRE : la cadence stride (hold-then-catch-up) en depend |
| grille (n, L, geometry, nr/ntheta, r_min/r_max, periodic) | SystemConfig | re-validee au restart (mismatch = erreur explicite) |
| blocs : nom, ncomp, substeps, stride, evolve, gamma | BlockState | l'ORDRE d'insertion doit etre conserve (indexation) |
| etat U de chaque bloc (toutes composantes, cellules valides) | MultiFab | les ghosts se reconstruisent (fill_boundary) |
| aux partage (phi, grad, B_z, T_e, largeur) | Impl::aux + bz_field_ | B_z est une ENTREE (pas re-derivable) ; phi re-derivable mais le sauver preserve le warm start (gauss_policy="evolve" : OBLIGATOIRE, phi n'est plus re-derive !) |
| parametres modele (ModelSpec / runtime params des .so) | spec + block_params_ | les .so eux-memes ne sont PAS embarques : on sauve model_hash + so_path pour verification |
| politique temporelle (scheme lie/strang, gauss_policy) | stepper/fields | |
| options Newton / bornes dt globales | add_block / add_dt_bound | les callbacks Python `add_dt_bound` ne sont PAS serialisables : documenter "a re-poser apres restart" |

Decision assumee : `restart` NE RECONSTRUIT PAS la composition (les `add_block` / `set_poisson` /
couplages restent du ressort du script utilisateur, qui rejoue sa composition puis appelle
`restart`). Le checkpoint VERIFIE la coherence (memes blocs, memes tailles, meme model_hash si
disponible) et leve une erreur explicite sinon. C'est le contrat le plus simple qui evite de
serialiser des fermetures C++/Python.

## Contraintes HPC

1. **Pas un fichier par processus sur les gros runs.** Trois niveaux, du plus simple au plus
   scalable :
   - V1 (mono-rang / petits runs) : un fichier unique ecrit par le rang 0 apres gather
     (reutilise le marshaling `copy_state` existant). Suffisant pour les cas locaux actuels
     (System = une box).
   - V2 (MPI moyen) : HDF5 sequentiel par AGREGATION (rang 0 collecte par morceaux, ecrit en
     streaming) -- pas de dependance MPI-IO, memoire bornee.
   - V3 (production GPFS/Lustre, cf. ROMEO) : HDF5 PARALLELE (h5py-mpi / HDF5 natif) avec un
     dataset GLOBAL par bloc, hyperslabs par rang. **FAIT cote System (write) : ADC-66 /
     PR-IO-3.** OPT-IN par `sim.write(format="hdf5", parallel=True)` : ouverture collective
     `h5py.File(driver="mpio", comm=COMM_WORLD)`, datasets globaux `(ncomp, ny, nx)` crees
     collectivement, chaque rang ecrit SES boites en hyperslabs. PAS de flag CMake ni de dependance
     C++ HDF5 : tout passe par h5py-mpi cote Python (absent / sans MPI -> erreur CLAIRE avec remede,
     jamais d'ecriture silencieuse) ; le seul ajout C++ est une paire d'accesseurs minimaux NON
     collectifs `System::local_boxes` / `System::local_state`. `parallel=False` (defaut) = chemin V1
     gather rang-0 INCHANGE. **Le System cartesien etant MONO-BOX** (une box, rang 0), le vrai
     parallelisme par hyperslabs n'apparait que sur une geometrie MULTI-BOX -- l'AMR (un
     groupe/dataset par niveau + boites) reste a faire (ADC-65).
2. **Layout** : datasets `(ncomp, ny, nx)` (composante-majeur, coherent avec `get_state`),
   attributs HDF5 pour les metadonnees (t, macro_step, config, noms/roles de variables).
   AMR : un groupe par niveau + boites (format inspire des plotfiles AMReX, mais HDF5).
3. **Atomicite** : ecrire dans `<path>.tmp` puis rename (un checkpoint corrompu par un crash en
   cours d'ecriture ne doit pas ecraser le precedent).
4. **VTK** : format `.vti` (ImageData) suffisant en cartesien uniforme ; polaire -> `.vts`
   (StructuredGrid r/theta). Ecrit cote Python (numpy -> binaire), sans dependance lourde.

## Decoupage en PR

1. **PR-IO-1** : `sim.write(path, format="vtk"|"npz", step=)` cote PYTHON pur (lit get_state /
   potential / variable_names ; zero changement C++). Donne tout de suite une sortie visualisable
   uniforme aux cas adc_cases. `format="npz"` = checkpoint-lite mono-rang de facto.
2. **PR-IO-2** : `sim.checkpoint(path)` / `sim.restart(path)` Python (npz ou hdf5 si h5py present),
   contrat de verification ci-dessus. EXIGE d'exposer `macro_step()` cote bindings (trivial) et un
   `set_time(t, macro_step)` controle (restaure la cadence stride).
3. **PR-IO-3** : HDF5 agrege/parallele + AMR (multi-niveau) + gros runs ROMEO. HDF5 agrege (V1) =
   FAIT (PR-IO-2). HDF5 PARALLELE par hyperslabs cote System `write` = **FAIT (ADC-66)** :
   `sim.write(format="hdf5", parallel=True)` (h5py mpio + mpi4py, opt-in ; accesseurs C++ minimaux
   `local_boxes`/`local_state`). RESTE : AMR multi-niveau (ADC-65), et un CHECKPOINT redemarrable
   HDF5 parallele (le checkpoint reste npz gather-rang-0 ; `checkpoint(parallel=True)` leve).

## Non-objectifs (explicites)

- Pas de serialisation des compositions (blocs/couplages) ni des callbacks Python.
- Pas de format proprietaire : HDF5/VTK/NPZ seulement.
- Le restart inter-version (en-tetes adc differents) n'est PAS garanti : le checkpoint embarque
  `abi_key`/`model_hash` pour le DETECTER, pas pour le convertir.
