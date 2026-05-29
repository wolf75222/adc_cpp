# Schema deux-fluides isotherme asymptotic-preserving

Note de methode sur le solveur deux-fluides de `adc_cpp` : modele, raideur, schema AP,
discretisation du transport, et enveloppe de robustesse mesuree. Le code est dans
[`include/adc/integrator/two_fluid_ap.hpp`](../include/adc/integrator/two_fluid_ap.hpp)
(coeur portable GPU) et la facade dans
[`include/adc/solver/two_fluid_ap_solver.hpp`](../include/adc/solver/two_fluid_ap_solver.hpp).
Reference physique : Hoffart, arXiv:2510.11808.

## 1. Modele

Deux especes s (electrons e, ions i), densite `n_s`, quantite de mouvement `m_s = n_s u_s`,
vitesse du son isotherme `c_s`. Charge `z_e = -1`, `z_i = +1`. Champ electrostatique
`E = -grad phi`, frequences plasma `omega_ps`. Pour chaque espece :

```
d_t n_s + div(m_s)                              = 0
d_t m_s + div(m_s o m_s / n_s + c_s^2 n_s I)    = z_s omega_ps^2 n_s E
```

couplees par le Poisson `lap(phi) = z_e n_e + z_i n_i = n_i - n_e` (n_0 = 1).

## 2. La raideur

La frequence plasma `omega_pe` est la frequence des oscillations de Langmuir. Un schema
explicite est stable seulement si `omega_pe * dt < O(1)`. En regime quasi-neutre
`omega_pe` est grand : resoudre l'oscillation impose un pas minuscule alors que la
dynamique d'interet (transport) est lente. Un schema explicite "explose" des que
`omega_pe * dt` depasse l'ordre 1 (verifie par le test : le mode non stabilise diverge).

## 3. Schema asymptotic-preserving

Le terme de Lorentz est traite implicitement. En reportant la quantite de mouvement
implicite `m^{n+1} = m* + dt z omega_ps^2 E` dans la continuite puis dans le Poisson, le
champ obeit a une equation elliptique reformulee. A `beta0` constant (n_0 = 1) :

```
beta0 = dt^2 (omega_pe^2 + omega_pi^2)
lap(phi) = (n_e* - n_i*) / (1 + beta0)
```

Le facteur `1 / (1 + beta0)` est ce qui rend le schema AP : quand `omega_pe -> inf`
(`beta0 -> inf`), le RHS tend vers 0, donc `n_e -> n_i` (quasi-neutralite) et le pas reste
stable independamment de `omega_pe * dt`. C'est l'option `stabilize` (defaut).

Un pas (operateur scinde) :

```
1. m*   = m  - dt div(F_mom)         flux Rusanov (tfap_mstar)
2. n*   = n  - dt div(m*)            continuite (centree ou upwind)   -> RHS Poisson
3. phi  : lap(phi) = (n_e* - n_i*) / (1 + beta0)   multigrille (GeometricMG)
4. E    = -grad phi
5. m^{n+1} = m* + dt z omega_ps^2 E  Lorentz implicite
6. n^{n+1} = n  - dt div(m^{n+1})    continuite
```

L'elliptique est delegue a un `EllipticSolver` template (`PoissonFFTSolver` en CPU,
`GeometricMG` entierement on-device pour le GPU GH200).

## 4. Discretisation du transport

La quantite de mouvement (etape 1) utilise un flux **Rusanov** dimensionnellement scinde
(vitesse d'onde `a = |u| + c_s`), ordre 1. La continuite (etapes 2 et 6) propose deux
schemas, selectionnes par `upwind_continuity` :

- **centree** (defaut) : `div(m)` par differences centrees. C'est un flux de masse central
  pur `F = 0.5 (m_i + m_{i+1})`, sans dissipation. Ordre 2, mais dispersif sur les fronts
  raides.
- **upwind MUSCL** : flux de masse Rusanov coherent avec la quantite de mouvement,
  `F = 0.5 (m_i + m_{i+1}) - 0.5 a (n_R - n_L)`, ou `n_L, n_R` sont reconstruits a la face
  avec une pente **minmod**. La dissipation est en `O(dx^2)` en regime lisse (pas de
  sur-diffusion) et retombe a l'ordre 1 aux extrema (monotone). Necessite 2 ghosts sur la
  densite. C'est un upwind ordre 2.

## 5. Enveloppe de robustesse mesuree

Mesuree par [`tests/test_two_fluid_ap_amplitude.cpp`](../tests/test_two_fluid_ap_amplitude.cpp).

**Perturbation lisse** (mode cosinus, regime raide AP `dt * omega_pe = 5`) : le schema est
robuste jusqu'a une grande amplitude. A `eps = 0.8` la densite reste positive
(`min n_e = 0.89`), bornee, quasi-neutre, masse conservee a l'arrondi. C'est le regime de
conception du schema.

**Front raide** (bosse acoustique etroite, couplage faible) : la continuite centree
montre un sous-depassement de 17.6%. Point important verifie par l'upwind MUSCL : ce
sous-depassement est presque **identique** (17.0%) avec le schema precis MUSCL. Les deux
schemas precis s'accordent, donc ce sous-depassement est surtout la **rarefaction
acoustique physique**, pas une oscillation de Gibbs numerique. Un flux de masse Rusanov de
1er ordre le reduisait a 13.4%, mais c'etait de la sur-diffusion (perte de precision), pas
une correction. Sur une bosse lisse bien resolue, MUSCL ne perd que 0.4% du pic par
rapport au schema centre : c'est un upwind ordre 2 precis, c'est son apport reel.

Conclusion pratique : la continuite centree suffit pour le regime AP lisse vise ; l'option
upwind MUSCL est un transport ordre 2 precis pour les cas a forts gradients, sans
sur-diffusion.

## 6. Validation

- Dispersion isotrope sur mode diagonal : ecart theorie/mesure 3.1%.
- Borne AP et quasi-neutralite a `omega_pe = 1e3`, `dt * omega_pe = 5`, la ou l'explicite
  explose.
- Conservation de la masse par espece a l'arrondi (`~1e-11`), centree et upwind.
- Portable GPU GH200 (memes kernels `for_each_cell` + `ADC_HD`, multigrille on-device),
  bit-identique au CPU.
- Expose jusqu'a Python : `TwoFluidAPConfig` (dont `upwind_continuity`), `TwoFluidAPSolver`.
