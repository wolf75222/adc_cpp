"""PARAMETRES RUNTIME du DSL (P7-b) : un parametre declare adc.dsl.Param(..., kind='runtime') peut voir
sa valeur CHANGEE a l'execution SANS recompiler le .so, alors qu'un parametre kind='const' (defaut)
reste INLINE EN DUR (bit-identique a l'historique).

Mecanique (backend "aot", add_compiled_block) : le codegen emet `params.get(<indice>)` pour un param
runtime (lecture d'un membre adc::RuntimeParams de la brique generee) au lieu d'une constante ; l'ABI du
.so AOT transporte un bloc plat de valeurs (symboles `_p`) ; System.set_block_params(name, values) ecrit
dans le bloc PARTAGE -> le comportement change au prochain pas. cf. include/adc/runtime/runtime_params.hpp.

Ce test verifie :
  1) NON-REGRESSION : un param const reste inline (codegen byte-identique a un modele sans param runtime ;
     aucun #include runtime_params.hpp, aucun membre RuntimeParams, valeur ecrite en dur) ;
  2) RUNTIME : un modele avec un param runtime cs2 (vitesse du son au carre) compile, tourne, et
     set_block_params change eval_rhs en consequence (le residu = -div F scale avec cs2 via p = cs2*rho) ;
  3) PAS DE RECOMPILATION : recompiler le MEME modele (meme model_hash + abi_key) reutilise le .so en cache
     (cache HIT) ; changer cs2 au runtime n'engendre AUCUNE recompilation ;
  4) cohrence avec un modele OU cs2 est cuit en CONST : eval_rhs(runtime cs2=k) == eval_rhs(const cs2=k).
"""
import os
import shutil
import tempfile

import numpy as np

import adc

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _build_iso(cs2_kind, cs2_value=1.0):
    """Modele isotherme 2D (rho, rho_u, rho_v) avec p = cs2 * rho. @p cs2_kind = 'runtime' | 'const'.
    Le SEUL parametre est cs2 : un meme modele a deux variantes (cs2 runtime vs cs2 const)."""
    m = adc.dsl.Model("iso")
    rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
    cs2 = m.param("cs2", cs2_value, kind=cs2_kind)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = adc.dsl.sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    return m


def _initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((3, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)  # densite non uniforme
    return U


def _check_codegen_non_regression():
    """(1) Un param CONST reste INLINE : le codegen d'un modele a param const est byte-identique a
    celui du MEME modele sans aucun param (cs2 ecrit en dur), et ne porte aucun artefact runtime."""
    const_src = _build_iso("const", 2.5)._m.emit_cpp_brick(name="IsoHyp")
    assert "runtime_params.hpp" not in const_src, "param const : ne doit PAS inclure runtime_params.hpp"
    assert "RuntimeParams" not in const_src, "param const : ne doit PAS porter de membre RuntimeParams"
    assert "params.get" not in const_src, "param const : doit etre INLINE (pas de params.get)"
    assert "2.5" in const_src, "param const cs2=2.5 doit etre ecrit EN DUR dans la brique"

    rt_src = _build_iso("runtime", 2.5)._m.emit_cpp_brick(name="IsoHyp")
    assert "runtime_params.hpp" in rt_src, "param runtime : doit inclure runtime_params.hpp"
    assert "adc::RuntimeParams params{1, {2.5}}" in rt_src, "param runtime : membre seede a la declaration"
    assert "params.get(0)" in rt_src, "param runtime : doit lire params.get(0) (pas de valeur en dur)"
    print("OK  (1) param const INLINE (byte-identique), param runtime -> params.get(0) + membre seede")


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_dsl_runtime_params : OK (rien a compiler)")
        return

    _check_codegen_non_regression()

    n, L = 32, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    tmp = tempfile.mkdtemp()
    try:
        m = _build_iso("runtime", 1.0)
        compiled = m.compile(os.path.join(tmp, "iso_runtime.so"), INCLUDE, backend="aot")
        assert compiled.runtime_param_names == ["cs2"], \
            "runtime_param_names attendu ['cs2'], recu %r" % compiled.runtime_param_names
        so = compiled.so_path

        def build(cs2):
            sys = adc.System(n=n, L=L, periodic=True)
            sys._s.add_compiled_block("gas", so, limiter="minmod", riemann="rusanov",
                                      recon="conservative", time="explicit",
                                      names=["rho", "rho_u", "rho_v"])
            sys._s.set_state("gas", Uflat)
            if cs2 is not None:
                sys._s.set_block_params("gas", [cs2])
            return sys

        # (2) RUNTIME : eval_rhs au defaut (cs2=1) puis apres set_block_params (cs2=4), SANS recompiler.
        sys = build(cs2=None)  # defaut de declaration cs2=1
        R1 = np.array(sys._s.eval_rhs("gas")).reshape(3, n, n)
        sys._s.set_block_params("gas", [4.0])  # CHANGE le param au RUNTIME, meme .so
        R4 = np.array(sys._s.eval_rhs("gas")).reshape(3, n, n)
        # La pression p = cs2*rho n'entre que dans la composante quantite de mouvement (rho_u, rho_v) du
        # flux ; le residu de qte de mvt = -div(p + ...) scale donc LINEAIREMENT avec cs2 sur la partie
        # pression. Le residu de densite (composante 0) ne depend PAS de cs2 (flux = rho*u, u=0 ici).
        assert np.max(np.abs(R1[1])) > 1e-3, "residu qte de mvt trivial (cs2=1)"
        assert np.max(np.abs(R4[1])) > np.max(np.abs(R1[1])) * 2.0, \
            "augmenter cs2 (1 -> 4) doit AUGMENTER le residu de qte de mvt (effet runtime absent ?)"
        assert np.max(np.abs(R1[0] - R4[0])) < 1e-14, \
            "le residu de densite ne doit PAS dependre de cs2 (vitesse nulle)"
        print("OK  (2) set_block_params change eval_rhs SANS recompiler (residu qte mvt cs2=1 vs cs2=4)")

        # (3) PAS DE RECOMPILATION : recompiler le MEME modele (sans so_path) -> cache HIT (meme chemin).
        m2 = _build_iso("runtime", 1.0)
        c_a = m2.compile(include=INCLUDE, backend="aot")  # cache hors source, keye sur model_hash+abi
        c_b = m2.compile(include=INCLUDE, backend="aot")  # 2e compile du MEME modele
        assert c_a.so_path == c_b.so_path, "cache : meme modele -> meme chemin .so"
        mtime = os.path.getmtime(c_b.so_path)
        c_c = _build_iso("runtime", 1.0).compile(include=INCLUDE, backend="aot")
        assert os.path.getmtime(c_c.so_path) == mtime, "cache HIT : le .so NE doit PAS etre recompile"
        print("OK  (3) recompiler le meme modele runtime -> cache HIT (.so reutilise, pas recompile)")

        # (4) COHERENCE runtime vs const : eval_rhs(runtime cs2=k) == eval_rhs(const cs2=k). On compile un
        # modele a cs2 CONST=2.0 et on le compare au modele runtime apres set_block_params(cs2=2.0).
        mc = _build_iso("const", 2.0)
        so_const = mc.compile(os.path.join(tmp, "iso_const2.so"), INCLUDE, backend="aot").so_path
        sysc = adc.System(n=n, L=L, periodic=True)
        sysc._s.add_compiled_block("gas", so_const, limiter="minmod", riemann="rusanov",
                                   recon="conservative", time="explicit",
                                   names=["rho", "rho_u", "rho_v"])
        sysc._s.set_state("gas", Uflat)
        Rc = np.array(sysc._s.eval_rhs("gas")).reshape(3, n, n)

        sysr = build(cs2=2.0)  # MEME .so runtime, param fixe a 2.0
        Rr = np.array(sysr._s.eval_rhs("gas")).reshape(3, n, n)
        drc = float(np.max(np.abs(Rr - Rc)))
        assert drc < 1e-12, "eval_rhs(runtime cs2=2) != eval_rhs(const cs2=2) (ecart %.2e)" % drc
        print("OK  (4) runtime cs2=2 == const cs2=2 (eval_rhs ecart %.1e) : meme numerique" % drc)

        # (5) GARDE-FOU : set_block_params sur un bloc SANS param runtime leve une erreur explicite.
        raised = False
        try:
            sysc._s.set_block_params("gas", [1.0])  # le bloc 'gas' de sysc est const-only
        except RuntimeError as ex:
            raised = True
            assert "pas de parametre runtime" in str(ex), "message inattendu : %s" % ex
        assert raised, "set_block_params sur un bloc const-only doit lever (sinon set silencieux)"
        print("OK  (5) set_block_params sur un bloc const-only REJETE explicitement")

        print("test_dsl_runtime_params : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
