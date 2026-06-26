"""Hook generique de PROJECTION PONCTUELLE post-pas (ADC-177) : U <- P(U, aux) applique par le
System a la FIN de chaque macro-pas ENTIER (jamais par etage RK), par bloc, sur les cellules
valides. La projection est EMISE PAR LA DSL (m.projection -> trait C++ HasPointwiseProjection,
compile comme flux/source) : elle remplace le callback Python par cellule (serie) par un chemin
compile -- prerequis du Ma eleve, de la relaxation sur AMR, MPI et GPU. Les formules de
realisabilite restent cote cas ; seul le hook est coeur (zero physique hyqmom dans adc_cpp).

Modele jouet : transport lineaire 2 variables (q0, q1) a vitesses constantes ; projection de
reference = clamp de positivite q0 <- (q0 + |q0|)/2 et masque q1 <- q1 * (sign(q0) + 1)/2
(les branches par cellule s'ecrivent en max/min/sign SANS if, exactement la doctrine ADC-177).

On verifie :
 (1) sign() symbolique : eval numpy (-1/0/1), codegen sans branche, derivee nulle (dsl.diff) ;
 (2) NO-DEFAULT-CHANGE (backend production, add_native_block) : un modele SANS projection et le
     MEME modele avec projection IDENTITE donnent des trajectoires BIT-IDENTIQUES (le hook tourne
     mais ne change rien) ;
 (3) SEMANTIQUE POST-PAS (production) : avec le clamp, l'etat apres CHAQUE pas == la reference
     Python "transport d'un pas SANS hook puis projection numpy" (np.allclose atol=1e-15, et en
     pratique bit-exact) -- 3 pas -> 3 applications, chacune APRES le pas entier ;
 (4) MEME semantique sur le backend aot (add_compiled_block, ABI .so marshalee
     pops_compiled_has_projection / pops_compiled_project_p) ;
 (5) GARDES : backend 'prototype' (JIT) rejette m.projection (ValueError explicite) ; un loader
     production target='amr_system' avec projection est desormais ACCEPTE par add_native_block
     (AmrSystem) : la projection ponctuelle est cablee PAR NIVEAU apres le reflux (ADC-312) ;
 (6) PROJECTION LISANT L'AUX (production + aot) : un plancher par cellule q1 <- max(q1, w) ou w est un
     champ aux NOMME (set_aux_field) -- l'etat post-pas == transport puis projection numpy AVEC le meme
     w (egalite bit-exacte, W asymetrique => valide la convention d'index) ; un plancher 5.0 force
     q1==5 partout (preuve que l'aux marshalee est CONSOMMEE, pas un canal a vide). Exerce les deux
     sites de marshaling aux : compiled_block_abi::pointwise_project (aot) et la fermeture project du
     native_loader (production).
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
A_X, A_Y = 1.0, 0.5  # vitesses d'advection constantes du transport jouet
N, L, DT, NSTEPS = 32, 1.0, 1e-3, 3

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build_model(tag, projection):
    """Transport jouet 2 variables. projection : None (aucun hook), 'identity' ou 'clamp'."""
    m = dsl.HyperbolicModel("toyproj_" + tag)
    q0, q1 = m.conservative_vars("q0", "q1")
    m.set_flux(x=[A_X * q0, A_X * q1], y=[A_Y * q0, A_Y * q1])
    m.set_eigenvalues(x=[dsl.Const(A_X)], y=[dsl.Const(A_Y)])
    m.set_primitive_state("q0", "q1")  # primitives = conservatives (transport pur)
    m.set_conservative_from([q0, q1])
    if projection == "identity":
        m.projection([q0, q1])
    elif projection == "clamp":
        # clamp de positivite + masque par signe : branches par cellule SANS if (abs / sign).
        m.projection([(q0 + dsl.abs_(q0)) / 2.0,
                      q1 * 0.5 * (dsl.sign(q0) + 1.0)])
    return m


def build_aux_model(tag):
    """Transport jouet 2 variables AVEC une projection qui LIT l'aux : plancher par cellule
    q1 <- max(q1, w), ou w est un champ aux NOMME (aux_field) fixe par bloc via set_aux_field.
    max(a, b) = (a + b + |a - b|) / 2 -> branche par cellule SANS if (idempotent : max(max,w)=max).
    Exerce le marshaling aux des DEUX chemins compiles (compiled_block_abi::pointwise_project en aot,
    fermeture project du native_loader en production) : sans aux lue, ces branches tournaient a vide."""
    m = dsl.HyperbolicModel("toyauxproj_" + tag)
    q0, q1 = m.conservative_vars("q0", "q1")
    m.set_flux(x=[A_X * q0, A_X * q1], y=[A_Y * q0, A_Y * q1])
    m.set_eigenvalues(x=[dsl.Const(A_X)], y=[dsl.Const(A_Y)])
    m.set_primitive_state("q0", "q1")  # primitives = conservatives (transport pur)
    m.set_conservative_from([q0, q1])
    w = m.aux_field("wfloor")  # champ aux NOMME -> composante dsl.AUX_NAMED_BASE, fixee par set_aux_field
    floored = (q1 + w + dsl.abs_(q1 - w)) / 2.0  # max(q1, w) sans branche
    m.projection([q0, floored])
    return m


def initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    q0 = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # change de signe : le clamp est ACTIF
    q1 = 0.7 + 0.2 * np.sin(2 * np.pi * Y)
    return np.stack([q0, q1])


def make_sys(so, adder):
    s = pops.System(n=N, L=L, periodic=True)
    if adder == "native":
        s._s.add_native_block("toy", so, limiter="minmod", riemann="rusanov",
                              recon="conservative", time="explicit", gamma=1.4, substeps=1)
    else:
        s._s.add_compiled_block("toy", so, limiter="minmod", riemann="rusanov",
                                recon="conservative", time="explicit", substeps=1)
    s.set_state("toy", initial_state(N))
    return s


def run_states(so, adder, nsteps=NSTEPS):
    """Etats apres chaque pas (liste de (2, N, N))."""
    s = make_sys(so, adder)
    out = []
    for _ in range(nsteps):
        s.step(DT)
        out.append(np.array(s.get_state("toy")).reshape(2, N, N))
    return out


def reference_states(so_nohook, adder, m_clamp, nsteps=NSTEPS):
    """Reference POST-PAS : depuis l'etat projete courant, UN pas du MEME transport SANS hook,
    puis projection numpy (m.projection_value, miroir exact du project(U, aux) emis)."""
    s = make_sys(so_nohook, adder)
    cur = initial_state(N)
    out = []
    for _ in range(nsteps):
        s.set_state("toy", cur)
        s.step(DT)
        cur = m_clamp.projection_value(np.array(s.get_state("toy")).reshape(2, N, N))
        out.append(cur)
    return out


def run_states_aux(so, adder, w, nsteps=NSTEPS):
    """Comme run_states mais fixe d'abord le champ aux NOMME 'wfloor' (composante AUX_NAMED_BASE).
    Le champ est STATIQUE et PERSISTE d'un pas a l'autre : un seul set_aux_field_component suffit."""
    s = make_sys(so, adder)
    s._s.set_aux_field_component(dsl.AUX_NAMED_BASE, np.asarray(w, dtype=float).reshape(-1))
    out = []
    for _ in range(nsteps):
        s.step(DT)
        out.append(np.array(s.get_state("toy")).reshape(2, N, N))
    return out


def reference_states_aux(so_nohook, adder, m_aux, w, nsteps=NSTEPS):
    """Reference POST-PAS pour une projection LISANT l'aux : transport SANS hook (so_nohook ne lit pas
    w -> identique), puis projection numpy avec le MEME champ aux (aux={'wfloor': w}). Le meme tableau
    w sert ici et a set_aux_field_component : meme convention [ligne=y, colonne=x] des deux cotes."""
    s = make_sys(so_nohook, adder)
    cur = initial_state(N)
    out = []
    for _ in range(nsteps):
        s.set_state("toy", cur)
        s.step(DT)
        cur = m_aux.projection_value(np.array(s.get_state("toy")).reshape(2, N, N),
                                     aux={"wfloor": np.asarray(w, dtype=float)})
        out.append(cur)
    return out


def err_msg(fn):
    try:
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def main():
    print("== (1) sign() symbolique ==")
    x = dsl.Var("x", "cons")
    sg = dsl.sign(x)
    chk(np.array_equal(sg.eval({"x": np.array([-2.0, 0.0, 3.0])}),
                       np.array([-1.0, 0.0, 1.0])), "eval numpy = -1/0/1")
    cpp = sg.to_cpp()
    chk("> 0" in cpp and "< 0" in cpp and "?" not in cpp,
        "codegen sans branche : %s" % cpp)
    d = dsl.diff(dsl.sign(x) * x, "x")
    chk(abs(float(d.eval({"x": 5.0})) - 1.0) < 1e-15,
        "diff : d(sign(x)*x)/dx = sign(x) (derivee de sign nulle p.p.)")

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_projection_hook : OK (rien a compiler)")
        sys.exit(1 if fails else 0)

    tmp = tempfile.mkdtemp()
    try:
        m_none = build_model("none", None)
        m_id = build_model("ident", "identity")
        m_clamp = build_model("clamp", "clamp")

        print("== (2) no-default-change (production) : sans hook == hook identite ==")
        so_none = m_none.compile(os.path.join(tmp, "toy_none.so"), INCLUDE, backend="production")
        so_id = m_id.compile(os.path.join(tmp, "toy_id.so"), INCLUDE, backend="production")
        st_none = run_states(so_none, "native")
        st_id = run_states(so_id, "native")
        d = max(float(np.max(np.abs(a - b))) for a, b in zip(st_none, st_id))
        chk(d == 0.0, "trajectoires BIT-IDENTIQUES sur %d pas (ecart max %.2e)" % (NSTEPS, d))

        print("== (3) semantique post-pas (production) : clamp == reference numpy ==")
        so_clamp = m_clamp.compile(os.path.join(tmp, "toy_clamp.so"), INCLUDE,
                                   backend="production")
        st_clamp = run_states(so_clamp, "native")
        st_ref = reference_states(so_none, "native", m_clamp)
        for k, (a, b) in enumerate(zip(st_clamp, st_ref)):
            dk = float(np.max(np.abs(a - b)))
            chk(np.allclose(a, b, rtol=0.0, atol=1e-15),
                "pas %d : etat == transport-sans-hook puis projection (ecart max %.2e)"
                % (k + 1, dk))
        # la projection est ACTIVE (le clamp change reellement l'etat) et appliquee a CHAQUE pas
        # (2 pas != 1 application : l'etat 2-pas projete differe du 2-pas sans hook projete une fois).
        d_active = float(np.max(np.abs(st_clamp[0] - st_none[0])))
        chk(d_active > 1e-6, "le clamp modifie l'etat (ecart %d-eme pas %.2e)" % (1, d_active))
        once = m_clamp.projection_value(st_none[1])
        chk(float(np.max(np.abs(st_clamp[1] - once))) > 1e-9,
            "2 pas -> 2 applications (!= projeter une seule fois a la fin)")
        chk(float(st_clamp[-1][0].min()) >= 0.0, "positivite q0 >= 0 apres le dernier pas")

        print("== (4) meme semantique sur le backend aot (add_compiled_block) ==")
        so_none_a = m_none.compile(os.path.join(tmp, "toy_none_aot.so"), INCLUDE, backend="aot")
        so_clamp_a = m_clamp.compile(os.path.join(tmp, "toy_clamp_aot.so"), INCLUDE, backend="aot")
        st_clamp_a = run_states(so_clamp_a, "aot")
        st_ref_a = reference_states(so_none_a, "aot", m_clamp)
        d = max(float(np.max(np.abs(a - b))) for a, b in zip(st_clamp_a, st_ref_a))
        chk(all(np.allclose(a, b, rtol=0.0, atol=1e-15) for a, b in zip(st_clamp_a, st_ref_a)),
            "aot : etat post-pas == reference numpy a chaque pas (ecart max %.2e)" % d)

        print("== (5) gardes ==")
        msg = err_msg(lambda: m_clamp.compile(os.path.join(tmp, "toy_proto.so"), INCLUDE,
                                              backend="prototype"))
        chk("projection" in msg and "prototype" in msg,
            "backend prototype rejete : %s" % msg[:80])
        # ADC-312 : la projection ponctuelle est desormais cablee sur AmrSystem (helper par niveau
        # apres reflux). add_native_block d'un loader avec projection n'est donc PLUS rejete.
        so_amr = m_clamp.compile(os.path.join(tmp, "toy_amr.so"), INCLUDE,
                                 backend="production", target="amr_system")
        s_amr = pops.AmrSystem(n=N, L=L, periodic=True)
        amr_err = err_msg(lambda: s_amr._s.add_native_block(
            "toy", so_amr, limiter="minmod", riemann="rusanov", recon="conservative",
            time="explicit", gamma=1.4, substeps=1))
        chk(amr_err == "",
            "target amr_system : projection cablee (ADC-312), add_native_block accepte%s"
            % ("" if amr_err == "" else " -- a leve : %s" % amr_err[:80]))

        print("== (6) projection LISANT l'aux : champ aux marshale ET consomme (production + aot) ==")
        m_aux = build_aux_model("floor")
        # Plancher par cellule q1 <- max(q1, W). W ASYMETRIQUE (ligne != colonne) dans [0.3, 0.9] :
        # actif la ou W > q1 (q1 dans [0.5, 0.9]), inactif ailleurs -> exerce LES DEUX branches du max
        # et VALIDE la convention d'index (un transpose aux casserait l'egalite bit-exacte).
        ix = np.arange(N)
        JJ, II = np.meshgrid(ix, ix, indexing="ij")           # JJ = ligne (y), II = colonne (x)
        W = 0.6 + 0.3 * np.sin(2 * np.pi * (II + 0.5) / N) * np.cos(2 * np.pi * (JJ + 0.5) / N)
        W_hi = np.full((N, N), 5.0)                            # plancher >> q1 : signature non ambigue
        none_so = {"native": so_none, "aot": so_none_a}        # transport sans hook (ne lit pas W)
        none_st = {"native": st_none, "aot": run_states(so_none_a, "aot")}
        for backend, adder in (("production", "native"), ("aot", "aot")):
            so_aux = m_aux.compile(os.path.join(tmp, "toy_aux_%s.so" % backend), INCLUDE,
                                   backend=backend)
            st_aux = run_states_aux(so_aux, adder, W)
            st_ref = reference_states_aux(none_so[adder], adder, m_aux, W)
            d = max(float(np.max(np.abs(a - b))) for a, b in zip(st_aux, st_ref))
            chk(all(np.allclose(a, b, rtol=0.0, atol=1e-15) for a, b in zip(st_aux, st_ref)),
                "%s : etat post-pas == transport puis plancher(aux) numpy (ecart max %.2e)"
                % (backend, d))
            # le plancher LIT reellement W : q1 est releve dans au moins une cellule vs le run sans hook.
            d_active = max(float(np.max(np.abs(a[1] - b[1])))
                           for a, b in zip(st_aux, none_st[adder]))
            chk(d_active > 1e-6, "%s : le plancher aux modifie q1 (ecart %.2e)" % (backend, d_active))
            # ADVERSARIAL : un plancher 5.0 (>> q1) force q1 == 5 partout a chaque pas. Si l'aux etait
            # ignoree (marshaling a vide), q1 resterait le transport (~[0.5, 0.9]) -> ce chk tomberait.
            st_hi = run_states_aux(so_aux, adder, W_hi)
            chk(all(np.allclose(s[1], 5.0, rtol=0.0, atol=1e-12) for s in st_hi),
                "%s : plancher aux=5.0 -> q1==5 a chaque pas (aux lu, jamais ignore)" % backend)
            chk(all(float(np.max(np.abs(a[0] - b[0]))) == 0.0 for a, b in zip(st_hi, none_st[adder])),
                "%s : q0 inchange (projection identite sur q0, aux ne touche que q1)" % backend)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("FAILS =", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
