#!/usr/bin/env python3
"""check_model : encadrement NON CIRCULAIRE du spectre par le jacobien dense (ADC-196).

Motivation : avec set_wave_speeds_from_jacobian(blocks=...), max_wave_speed ET la coherence
wave_speeds<->max_wave_speed derivent TOUS DEUX de la MEME partition de sous-blocs. Une partition
qui n'est pas reellement bloc-triangulaire (les extremes des sous-blocs ne bornent pas le spectre)
passe alors check_model en silence : mws sous-estime, dt CFL non sur. check_model calcule desormais
le rayon spectral du jacobien DENSE complet du flux (differences finies centrales, independant de
la partition) et exige max_wave_speed >= ce rayon.

On verifie :
 (a) repro K-swap (Fx = [K*b, K*a], spectre +-K) avec partition fausse blocks=[[0],[1]] : DETECTEE
     (ok=False, l'echec mentionne la partition) ;
 (b) modele 3-var bloc-triangulaire LEGITIME (partition vraie) : passe (ok=True) ;
 (c) doublon intra-bloc (blocks=[[0,0,1],[2]]) : message correct (meme bloc, PAS "deux blocs") ;
 (d) K-swap avec partition CORRECTE blocks=[[0,1]] (bloc plein 2x2) : passe (ok=True).
Pur Python (interprete CPU) : aucun compilateur requis.
"""
import sys

import numpy as np

from adc import dsl

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def kswap_model(blocks):
    """2 var, flux x = y = [K*b, K*a] : jacobien antidiagonal, spectre exact {-K, +K}.
    Les sous-blocs diagonaux 1x1 sont NULS -> blocks=[[0],[1]] (faux) borne le spectre a 0."""
    m = dsl.Model("kswap")
    a, b = m.conservative_vars("a", "b")
    K = 2.0
    m.flux(x=[K * b, K * a], y=[K * b, K * a])
    m.wave_speeds_from_jacobian(blocks=blocks)
    m.primitive_vars(a, b)
    m.conservative_from([a, b])
    return m


def triangular_model():
    """3 var, jacobiens x et y bloc-INFERIEUR-triangulaires sous la partition [[0], [1, 2]] :
    q0 autonome, (q1, q2) recoivent de q0 mais q0 ne recoit pas d'eux. Les extremes des
    sous-blocs == spectre vrai -> partition legitime, check_model doit passer."""
    m = dsl.Model("tri3")
    q0, q1, q2 = m.conservative_vars("q0", "q1", "q2")
    m.flux(x=[2.0 * q0, 3.0 * q1 + 0.5 * q0, -1.0 * q2 + 0.2 * q1],
           y=[1.0 * q0, 2.0 * q1 + 0.3 * q0, 0.5 * q2 - 0.1 * q1])
    m.wave_speeds_from_jacobian(blocks=[[0], [1, 2]])
    m.primitive_vars(q0, q1, q2)
    m.conservative_from([q0, q1, q2])
    return m


def check_a_kswap_wrong_partition_detected():
    print("== (a) K-swap, partition fausse [[0],[1]] : DETECTEE ==")
    rep = kswap_model([[0], [1]]).check_model(raise_on_error=False)
    chk(not rep["ok"], "ok == False (la verification circulaire seule passerait)")
    part = [f for f in rep["failures"] if "partition" in f]
    chk(bool(part), f"un echec mentionne la partition : {part[:1]}")
    chk(any("jacobien" in f for f in part),
        "l'echec nomme le rayon spectral du jacobien (borne non circulaire)")
    # garde-fou : le chemin CIRCULAIRE (coherence ws<->mws) ne suffisait pas a detecter.
    circ = [f for f in rep["failures"] if "incoherentes avec max_wave_speed" in f]
    chk(not circ, "la coherence ws<->mws (circulaire) reste muette : seul le jacobien tranche")


def check_b_triangular_partition_passes():
    print("== (b) modele bloc-triangulaire legitime : passe ==")
    rep = triangular_model().check_model(raise_on_error=False)
    chk(rep["ok"], f"ok == True (extremes des sous-blocs == spectre) ; failures={rep['failures']}")


def check_c_intra_block_message():
    print("== (c) doublon intra-bloc : message correct ==")
    m = dsl.Model("intra")
    a, b, c = m.conservative_vars("a", "b", "c")
    m.flux(x=[a, b, c], y=[a, b, c])
    msg = err_msg(lambda: m.wave_speeds_from_jacobian(blocks=[[0, 0, 1], [2]]))
    chk("meme bloc" in msg, f"message dit 'meme bloc' (intra) : {msg}")
    chk("deux blocs" not in msg, "message NE dit PAS 'deux blocs' (ce serait trompeur, doublon intra)")
    # inter-blocs : le message historique 'deux blocs' reste, lui, correct.
    msg2 = err_msg(lambda: m.wave_speeds_from_jacobian(blocks=[[0], [0, 1], [2]]))
    chk("deux blocs" in msg2, f"doublon inter-blocs : message dit bien 'deux blocs' : {msg2}")


def check_d_kswap_full_block_passes():
    print("== (d) K-swap, partition correcte [[0,1]] (bloc plein) : passe ==")
    rep = kswap_model([[0, 1]]).check_model(raise_on_error=False)
    chk(rep["ok"], f"ok == True (le bloc plein 2x2 encadre le spectre +-K) ; failures={rep['failures']}")


def main():
    check_a_kswap_wrong_partition_detected()
    check_b_triangular_partition_passes()
    check_c_intra_block_message()
    check_d_kswap_full_block_passes()
    if fails:
        print(f"FAIL test_check_model_partition : {fails} echec(s)")
        sys.exit(1)
    print("OK test_check_model_partition")


if __name__ == "__main__":
    main()
