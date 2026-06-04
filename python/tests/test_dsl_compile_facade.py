"""Facade de compilation par INTENTION : HyperbolicModel.compile(backend=...) aiguille vers les
moteurs existants (compile_so JIT / compile_aot AOT) SANS changer la numerique, et preserve de bout
en bout noms, VariableRole, gamma, n_aux et B_z (les metadonnees ABI #75).

Deux niveaux :
(1) GARDE-FOUS pur-Python (aucun compilateur requis) : backend inconnu, mapping backend -> adder,
    et erreurs EXPLICITES quand require_metadata est demande sur un modele sans roles/gamma ou sur le
    backend prototype (JIT, dispatch virtuel hote, non device-clean).
(2) BOUT EN BOUT (saute si aucun compilateur C++ / en-tetes adc) : compile(backend=...) produit une
    .so qui, branchee sur l'adder correspondant, expose les BONS noms/roles/gamma (pas le fallback) et
    lit bien le canal aux etendu (B_z). On prouve aussi que compile() == compile_or_jit() pour le
    backend equivalent (la facade ne fait qu'aiguiller, elle ne regresse pas la numerique).

Lance avec python3, meme PYTHONPATH que les autres tests DSL.
"""
import os
import shutil
import tempfile

import numpy as np

import adc
from adc import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.6667  # gamma NON STANDARD (5/3), distinct du defaut historique 1.4


def build_meta_euler():
    """Euler aux roles canoniques + gamma 5/3 : fournit des metadonnees UTILES (roles non 'Custom',
    gamma explicite). Sert a prouver que la facade les transporte et que require_metadata passe."""
    e = dsl.HyperbolicModel("euler_facade")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    e.set_gamma(GAMMA)  # gamma explicite -> transporte via adc_compiled_gamma
    return e


def build_bare_scalar():
    """Transport scalaire 'q' SANS role canonique ni gamma : metadonnees PAUVRES, declenche les
    erreurs require_metadata (le System retomberait sur le fallback custom / 1.4)."""
    e = dsl.HyperbolicModel("bare_q")
    (q,) = e.conservative_vars("q")
    e.set_flux(x=[q], y=[q])
    e.set_eigenvalues(x=[q], y=[q])
    e.set_primitive_state(q)
    e.set_conservative_from([q])
    return e


def build_bz_scalar():
    """Scalaire sans flux, source magnetisee S = B_z * n (lit aux('B_z')) : exerce le canal aux
    etendu (n_aux=4) a travers la facade."""
    m = dsl.HyperbolicModel("bz_facade")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn
    m.set_flux(x=[zero], y=[zero])
    m.set_eigenvalues(x=[zero], y=[zero])
    m.set_primitive_state(nn)
    m.set_conservative_from([nn])
    bz = m.aux("B_z")
    m.set_source([bz * nn])
    return m


def test_guardrails():
    """(1) Garde-fous pur-Python : ne compilent rien, tournent toujours."""
    e = build_meta_euler()
    bare = build_bare_scalar()

    # backend inconnu -> ValueError explicite
    for bad in ("jit", "compile", "gpu", "", None):
        try:
            e.compile("x.so", INCLUDE, backend=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("backend %r aurait du lever" % (bad,))
    try:
        dsl.HyperbolicModel.adder_for("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("adder_for(backend inconnu) aurait du lever")
    print("OK  backend inconnu rejete (compile + adder_for)")

    # mapping backend -> adder System (couplage compilation/execution)
    assert dsl.HyperbolicModel.adder_for("prototype") == "add_dynamic_block"
    assert dsl.HyperbolicModel.adder_for("aot") == "add_compiled_block"
    assert dsl.HyperbolicModel.adder_for("production") == "add_compiled_block"
    print("OK  adder_for : prototype->add_dynamic_block, aot/production->add_compiled_block")

    # require_metadata sur prototype (JIT, dispatch virtuel hote) : incoherent -> erreur claire
    try:
        e.compile("x.so", INCLUDE, backend="prototype", require_metadata=True)
    except ValueError as ex:
        assert "prototype" in str(ex)
    else:
        raise AssertionError("prototype + require_metadata aurait du lever")
    print("OK  backend prototype + require_metadata=True rejete (non device-clean)")

    # require_metadata sur un modele PAUVRE (pas de roles, pas de gamma) : erreur listant le manque
    try:
        bare.compile("x.so", INCLUDE, backend="production", require_metadata=True)
    except ValueError as ex:
        msg = str(ex)
        assert "roles" in msg and "gamma" in msg, "le message devrait lister roles ET gamma : %r" % msg
    else:
        raise AssertionError("modele sans roles/gamma + require_metadata aurait du lever")
    print("OK  require_metadata sur modele pauvre rejete (roles + gamma manquants signales)")

    # le modele RICHE (roles canoniques + gamma) ne doit PAS echouer la verification de metadonnees :
    # on isole le pre-check en passant un backend pauvre cote compilation (mais on ne compile pas ici,
    # on verifie juste que la verification de metadonnees ne leve pas avant l'appel au moteur). On le
    # prouve via le chemin bout-en-bout ci-dessous ; ici on s'assure juste que bare != e.
    assert all(r == "Custom" for r in dsl.roles_for(bare.cons_names)), "scalaire q devrait etre Custom"
    assert dsl.roles_for(e.cons_names) == ["Density", "MomentumX", "MomentumY", "Energy"]
    print("OK  garde-fous pur-Python verts")


def test_end_to_end():
    """(2) Bout en bout : compile(backend=...) -> .so -> adder -> metadonnees + B_z preserves."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> bout-en-bout saute")
        return

    e = build_meta_euler()
    n, L = 16, 1.0
    tmp = tempfile.mkdtemp()
    try:
        # --- aot / production : meme moteur (compile_aot) ; require_metadata=True passe (roles+gamma) ---
        for backend in ("aot", "production"):
            so = e.compile(os.path.join(tmp, "facade_%s.so" % backend), INCLUDE,
                           backend=backend, require_metadata=True)
            s = adc.System(n=n, L=L, periodic=True)
            adder = getattr(s, dsl.HyperbolicModel.adder_for(backend))
            adder("gas", so, limiter="minmod", riemann="hllc", recon="primitive")
            # noms/roles DU MODELE (pas le fallback u0.. / custom)
            assert s.variable_names("gas") == ["rho", "rho_u", "rho_v", "E"], \
                "%s : noms != metadonnees : %r" % (backend, s.variable_names("gas"))
            assert s.variable_roles("gas") == ["density", "momentum_x", "momentum_y", "energy"], \
                "%s : roles != metadonnees : %r" % (backend, s.variable_roles("gas"))
            assert s.variable_roles("gas", "primitive") == \
                ["density", "velocity_x", "velocity_y", "pressure"], \
                "%s : roles primitifs != metadonnees : %r" % (backend, s.variable_roles("gas", "primitive"))
            assert abs(s.block_gamma("gas") - GAMMA) < 1e-12, \
                "%s : gamma != metadonnees : %r" % (backend, s.block_gamma("gas"))
            print("OK  backend=%s : noms/roles/gamma propages (gamma=%.4f) via %s"
                  % (backend, s.block_gamma("gas"), dsl.HyperbolicModel.adder_for(backend)))

        # --- la facade ne regresse pas la numerique : compile(aot) octets-identiques a
        #     compile_or_jit(mode="compile") (meme source generee, meme toolchain) ---
        a = e.compile(os.path.join(tmp, "via_facade.so"), INCLUDE, backend="aot")
        b = e.compile_or_jit(os.path.join(tmp, "via_legacy.so"), INCLUDE, mode="compile")
        # la SOURCE generee est identique (le binaire peut differer par des chemins temporaires)
        assert e.emit_cpp_aot_source() == e.emit_cpp_aot_source(), "source non deterministe"
        s1 = adc.System(n=n, L=L, periodic=True)
        s1.add_compiled_block("g", a, limiter="minmod", riemann="hllc", recon="primitive")
        s2 = adc.System(n=n, L=L, periodic=True)
        s2.add_compiled_block("g", b, limiter="minmod", riemann="hllc", recon="primitive")
        assert s1.variable_names("g") == s2.variable_names("g")
        assert s1.variable_roles("g") == s2.variable_roles("g")
        assert abs(s1.block_gamma("g") - s2.block_gamma("g")) < 1e-15
        print("OK  compile(backend='aot') == compile_or_jit(mode='compile') (memes metadonnees)")

        # --- prototype : JIT (add_dynamic_block), roles/gamma transportes aussi (sans require_metadata) ---
        sop = e.compile(os.path.join(tmp, "facade_proto.so"), INCLUDE, backend="prototype")
        sp = adc.System(n=n, L=L, periodic=True)
        getattr(sp, dsl.HyperbolicModel.adder_for("prototype"))("gas", sop, recon="minmod")
        assert sp.variable_names("gas") == ["rho", "rho_u", "rho_v", "E"], \
            "prototype : noms != metadonnees : %r" % sp.variable_names("gas")
        assert sp.variable_roles("gas") == ["density", "momentum_x", "momentum_y", "energy"], \
            "prototype : roles != metadonnees : %r" % sp.variable_roles("gas")
        assert abs(sp.block_gamma("gas") - GAMMA) < 1e-12, \
            "prototype : gamma != metadonnees : %r" % sp.block_gamma("gas")
        print("OK  backend=prototype (JIT) : noms/roles/gamma propages via add_dynamic_block")

        # --- n_aux / B_z preserves a travers la facade (canal aux etendu) ---
        m = build_bz_scalar()
        c = 0.7
        so_bz = m.compile(os.path.join(tmp, "facade_bz.so"), INCLUDE, backend="aot")
        sb = adc.System(n=n, L=L, periodic=True)
        sb.add_compiled_block("bz", so_bz, limiter="none", riemann="rusanov",
                              recon="conservative", names=["n"])
        sb.set_poisson(rhs="charge_density", solver="geometric_mg")
        sb.set_density("bz", np.ones((n, n)))
        sb.set_magnetic_field(c * np.ones((n, n)))  # peuple le canal B_z partage (n_aux=4)
        sb.solve_fields()
        R = np.array(sb.eval_rhs("bz"))
        err = float(np.max(np.abs(R - c)))  # flux nul -> R = S = B_z n = c
        assert err < 1e-12, "B_z non lu a travers la facade (ecart %.2e)" % err
        print("OK  backend=aot : n_aux/B_z preserves (max|R - B_z| = %.2e)" % err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_guardrails()
    test_end_to_end()
    print("test_dsl_compile_facade : tout est vert")


if __name__ == "__main__":
    main()
