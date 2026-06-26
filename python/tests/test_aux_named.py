"""Champs aux NOMMES declares par le modele -- ADC-70 phase 1 (System cartesien).

Un modele DSL peut declarer un champ auxiliaire ARBITRAIRE via m.aux_field("nom") (au-dela des champs
canoniques phi/grad/B_z/T_e). Le k-ieme nom reserve la composante AUX_NAMED_BASE + k (= 5 + k) du canal
aux, lue en C++ via aux.extra_field(k). La FACADE resout nom -> composante par bloc
(System.set_aux_field / aux_field) ; le C++ ne manipule que des indices.

Couvre :
  (forme, sans compilateur) helpers de largeur, emission n_aux + extra_field, retro-compat (modele sans
    champ nomme -> pas de n_aux), rejets DSL (nom canonique, doublon, depassement), rejets FACADE
    (B_z/T_e/canonique rediriges, bloc inconnu) ;
  (bout en bout, saute sans compilateur C++) source S = -kappa*n lisant aux_field("kappa") :
    (a) kappa CONSTANT -> residu == -kappa*n exact ; kappa SPATIAL (gaussien) -> residu suit le champ ;
    (b) PERSISTANCE : le champ nomme survit a plusieurs step() (relecture aux_field) ;
    (c) defaut : un modele simple SANS aux_field garde n_aux=3 ;
    (d) lecture AVANT ecriture -> zeros (documente) ; champ inconnu d'un bloc enregistre -> rejet.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_decay_model():
    """Scalaire 'n' sans flux, source S = -kappa * n ou kappa est un champ aux NOMME (aux_field).
    flux nul -> eval_rhs = source = -kappa * n (verifiable exactement)."""
    m = dsl.Model("kappadecay")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn                      # expression nulle (flux/eig n'enrobent pas un float brut)
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(n=nn)               # layout Prim = [n]
    m.conservative_from([nn])
    kappa = m.aux_field("kappa")         # champ aux NOMME -> composante 5
    m.source([-(kappa * nn)])            # S = -kappa n
    return m


def test_form():
    """Garde-fous PUR-PYTHON (aucun compilateur) : largeurs, emission, retro-compat, rejets DSL."""
    # (1) largeur totale du canal : base seule = 3 ; un champ nomme -> AUX_NAMED_BASE + 1 = 6.
    assert dsl.aux_total_n_aux([], []) == 3
    assert dsl.aux_total_n_aux([], ["kappa"]) == 6
    assert dsl.aux_total_n_aux([], ["kappa", "sigma"]) == 7
    assert dsl.aux_total_n_aux(["B_z"], ["kappa"]) == 6  # max(4, 6)
    assert dsl.AUX_NAMED_BASE == 5
    print("OK  aux_total_n_aux : base=3, 1 nomme=6 (AUX_NAMED_BASE=5), 2 nommes=7")

    # (2) emission : un modele lisant aux_field('kappa') declare n_aux=6 et lit a.extra_field(0).
    m = dsl.HyperbolicModel("decay")
    (nn,) = m.conservative_vars("n")
    kappa = m.aux_field("kappa")
    m.set_source([-(kappa * nn)])
    src = m.emit_cpp_source(name="GenDecaySrc")
    assert "static constexpr int n_aux = 6;" in src, "n_aux=6 absent : %s" % src
    assert "const pops::Real kappa = a.extra_field(0);" in src, "lecture extra_field(0) absente : %s" % src
    print("OK  emit_cpp_source(aux_field) : n_aux=6 + a.extra_field(0)")

    # (3) retro-compat : un modele SANS aux_field n'emet PAS de n_aux (bit-identique a l'historique).
    m2 = dsl.HyperbolicModel("plain")
    (n2,) = m2.conservative_vars("n")
    m2.set_source([0.0 * n2])
    src2 = m2.emit_cpp_source(name="GenPlainSrc")
    assert "n_aux" not in src2, "n_aux ne doit pas etre emis pour un modele sans champ aux : %s" % src2
    assert m2._total_n_aux() == 3, "modele simple : n_aux total doit rester 3"
    print("OK  modele sans aux_field : pas de n_aux emis, largeur 3 (defaut)")

    # (4) rejets DSL : nom canonique, doublon, depassement de la borne kAuxMaxExtra.
    m3 = dsl.HyperbolicModel("rej")
    m3.conservative_vars("n")
    for bad in ("B_z", "T_e", "phi", "grad_x"):
        try:
            m3.aux_field(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("aux_field(%r) aurait du lever (nom canonique)" % bad)
    m3.aux_field("kappa")
    try:
        m3.aux_field("kappa")  # doublon
    except ValueError:
        pass
    else:
        raise AssertionError("aux_field doublon aurait du lever")
    # remplir jusqu'a la borne (kappa deja pose -> 3 de plus = 4 max), le 5e leve.
    m3.aux_field("a")
    m3.aux_field("b")
    m3.aux_field("c")
    try:
        m3.aux_field("d")  # 5e champ : depasse AUX_NAMED_MAX
    except ValueError:
        pass
    else:
        raise AssertionError("aux_field au-dela de AUX_NAMED_MAX aurait du lever")
    print("OK  aux_field rejette : nom canonique, doublon, > %d champs" % dsl.AUX_NAMED_MAX)


def test_facade_rejects():
    """Rejets de la FACADE qui ne demandent aucun bloc compile (resolution avant la table) : B_z / T_e
    rediriges vers leur chemin dedie, nom canonique non fixable, bloc inconnu."""
    sim = pops.System(n=8, L=1.0, periodic=True)
    field = np.ones((8, 8))
    # B_z -> set_magnetic_field (message redirigeant)
    try:
        sim.set_aux_field("blk", "B_z", field)
    except ValueError as ex:
        assert "set_magnetic_field" in str(ex), "le message B_z devrait rediriger : %r" % str(ex)
    else:
        raise AssertionError("set_aux_field('B_z') aurait du lever")
    # T_e -> set_electron_temperature_from
    try:
        sim.set_aux_field("blk", "T_e", field)
    except ValueError as ex:
        assert "set_electron_temperature_from" in str(ex), "le message T_e devrait rediriger : %r" % str(ex)
    else:
        raise AssertionError("set_aux_field('T_e') aurait du lever")
    # autre nom canonique (phi) non fixable
    try:
        sim.set_aux_field("blk", "phi", field)
    except ValueError:
        pass
    else:
        raise AssertionError("set_aux_field('phi') aurait du lever")
    # bloc inconnu (aucun champ nomme enregistre)
    try:
        sim.set_aux_field("inexistant", "kappa", field)
    except ValueError as ex:
        assert "inexistant" in str(ex)
    else:
        raise AssertionError("set_aux_field sur bloc inconnu aurait du lever")
    print("OK  facade : B_z/T_e/phi rediriges, bloc inconnu rejete")


def test_end_to_end():
    """Bout en bout : source lisant aux_field('kappa'), branchee via add_equation (backend AOT)."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> bout-en-bout saute (%s)" % INCLUDE)
        print("test_aux_named : OK (forme seulement)")
        return

    n, L = 16, 1.0
    tmp = tempfile.mkdtemp()
    try:
        m = build_decay_model()
        compiled = m.compile(os.path.join(tmp, "kappadecay.so"), include=INCLUDE, backend="aot")
        assert compiled.aux_extra_names == ["kappa"], "aux_extra_names attendu ['kappa']"
        assert compiled.n_aux == 6, "n_aux=6 attendu (5 + 1 champ nomme)"

        sim = pops.System(n=n, L=L, periodic=True)
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.add_equation("decay", model=compiled,
                         spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                         time=pops.Explicit())
        sim.set_density("decay", np.ones((n, n)))

        # (d) lecture AVANT ecriture : le champ nomme vaut 0 partout (canal initialise a zero).
        before = sim.aux_field("decay", "kappa")
        assert before.shape == (n, n) and float(np.max(np.abs(before))) == 0.0, \
            "kappa avant ecriture devrait etre 0 partout"
        print("OK  lecture avant ecriture : kappa == 0 (documente)")

        # (a1) kappa CONSTANT : eval_rhs = S = -kappa*n = -2 (n=1 partout).
        kc = 2.0
        sim.set_aux_field("decay", "kappa", kc * np.ones((n, n)))
        sim.solve_fields()
        R = np.array(sim.eval_rhs("decay"))
        err = float(np.max(np.abs(R + kc)))  # R = -kappa*n = -2
        assert err < 1e-12, "kappa constant non lu (max|R+kappa| = %.2e)" % err
        # relecture : aux_field rend bien le champ pose.
        rk = sim.aux_field("decay", "kappa")
        assert float(np.max(np.abs(rk - kc))) < 1e-12, "aux_field ne relit pas kappa constant"
        print("OK  kappa constant : eval_rhs == -kappa*n (max ecart %.2e)" % err)

        # (a2) kappa SPATIAL (gaussien) : eval_rhs = -kappa(x)*n suit le champ exactement.
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        ks = 1.0 + 3.0 * np.exp(-30.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        sim.set_aux_field("decay", "kappa", ks)
        sim.solve_fields()
        R2 = np.array(sim.eval_rhs("decay"))
        err2 = float(np.max(np.abs(R2 + ks)))  # n=1 -> R = -kappa(x)
        assert err2 < 1e-12, "kappa spatial non suivi (max|R+kappa| = %.2e)" % err2
        print("OK  kappa spatial (gaussien) : eval_rhs suit -kappa(x) (max ecart %.2e)" % err2)

        # (b) PERSISTANCE : plusieurs step() ; kappa (champ statique) reste inchange.
        for _ in range(5):
            sim.step_cfl(0.4)
        rk2 = sim.aux_field("decay", "kappa")
        errp = float(np.max(np.abs(rk2 - ks)))
        assert errp < 1e-12, "kappa n'a pas persiste apres 5 step (max ecart %.2e)" % errp
        print("OK  persistance : kappa intact apres 5 step (max ecart %.2e)" % errp)

        # (d) champ inconnu d'un bloc ENREGISTRE -> rejet listant les champs connus.
        try:
            sim.set_aux_field("decay", "sigma", np.ones((n, n)))
        except ValueError as ex:
            assert "sigma" in str(ex) and "kappa" in str(ex), "le rejet devrait lister les champs : %r" % str(ex)
        else:
            raise AssertionError("set_aux_field('decay','sigma') aurait du lever (non declare)")
        print("OK  champ aux nomme inconnu d'un bloc enregistre rejete")

        print("test_aux_named : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _have_compiler():
    return bool((shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))
                and os.path.isdir(INCLUDE))


def test_polar_named_aux():
    """ADC-291 phase 2 : un aux NOMME (aux_field) lu en geometrie POLAIRE via pops.System(PolarMesh).
    Avant ADC-291, le chemin polaire n'elargissait pas le canal aux (System::add_block polaire sans
    ensure_aux_width) -> set_aux_field('kappa') aurait leve 'canal a 3 composantes' (ou lu hors borne).
    On verifie set + lecture + eval_rhs = source = -kappa*n exact sur l'anneau."""
    if not _have_compiler():
        print("skip  polar named aux (compilateur/en-tetes absents)")
        return
    tmp = tempfile.mkdtemp()
    try:
        m = build_decay_model()
        compiled = m.compile(os.path.join(tmp, "kpolar.so"), include=INCLUDE, backend="aot")
        nr, nth = 16, 16
        sim = pops.System(mesh=pops.PolarMesh(r_min=0.3, r_max=1.0, nr=nr, ntheta=nth))
        sim.add_equation("decay", model=compiled,
                         spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                         time=pops.Explicit())
        sim.set_density("decay", np.ones((nth, nr)))

        # lecture avant ecriture : 0 partout (le canal s'est bien elargi : pas de rejet, pas d'OOB).
        before = sim.aux_field("decay", "kappa")
        assert before.shape == (nth, nr) and float(np.max(np.abs(before))) == 0.0, \
            "kappa polaire avant ecriture devrait etre 0 (canal elargi par ensure_aux_width)"

        # kappa constant : eval_rhs = S = -kappa*n = -3 (flux nul, n=1).
        kc = 3.0
        sim.set_aux_field("decay", "kappa", kc * np.ones((nth, nr)))
        R = np.array(sim.eval_rhs("decay"))
        err = float(np.max(np.abs(R + kc)))
        assert err < 1e-12, "polar : kappa non lu (max|R+kappa| = %.2e)" % err
        rk = sim.aux_field("decay", "kappa")
        assert float(np.max(np.abs(rk - kc))) < 1e-12, "polar : aux_field ne relit pas kappa"
        print("OK  polar named aux : eval_rhs == -kappa*n (max ecart %.2e)" % err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile_amr_decay(tmp, fname):
    """Decay model compiled for the AMR native path (backend='production', target='amr_system')."""
    return build_decay_model().compile(os.path.join(tmp, fname), include=INCLUDE,
                                       backend="production", target="amr_system")


def _bump_density(n, lo, hi, base, peak):
    """Uniform @p base with a square bump @p peak in [lo, hi)^2 (drives the refinement criterion)."""
    rho = np.full((n, n), float(base))
    rho[lo:hi, lo:hi] = float(peak)
    return rho


def test_amr_named_aux_single_block_regrid():
    """ADC-291 : aux NOMME sur le chemin AMR MONO-BLOC (AmrCouplerMP), avec un VRAI niveau fin + regrid.
    Le modele decay (flux nul, S=-kappa*n) sur une densite a BOSSE centrale > seuil -> set_refinement
    construit un patch fin sur la bosse ; regrid_every=1 RECONSTRUIT (et re-zero) l'aux fin a chaque pas.
    kappa UNIFORME -> chaque cellule (grossiere ET fine) doit decroitre du MEME facteur : si le champ
    nomme n'etait pas re-applique + re-injecte au fin apres regrid, la zone raffinee ne decroitrait PAS
    (ratio ~1 vs ~0.9 ailleurs). On verrouille donc (a) un niveau fin existe, (b) la masse decroit a
    chaque pas (persistance a travers le regrid), (c) la decroissance est ~uniforme (aux nomme present
    au fin). Sans set_aux_field (kappa=0) la masse est inchangee."""
    if not _have_compiler():
        print("skip  AMR named aux single-block (compilateur absent)")
        return
    tmp = tempfile.mkdtemp()
    try:
        n = 24
        sp = pops.FiniteVolume(limiter="none", riemann="rusanov")
        lo, hi = n // 3, 2 * n // 3  # central bump [8, 16)^2

        # (a) reference : SANS set_aux_field -> kappa=0 -> masse inchangee (meme avec raffinement).
        ref = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        ref.add_equation("decay", model=_compile_amr_decay(tmp, "amr0.so"), spatial=sp, time=pops.Explicit())
        ref.set_poisson(rhs="charge_density", solver="geometric_mg")
        ref.set_refinement(2.0)  # refine where density (comp 0) > 2 -> tags the bump
        ref.set_density("decay", _bump_density(n, lo, hi, 1.0, 5.0))
        m0 = ref.mass("decay")
        for _ in range(3):
            ref.step(1e-2)
        assert abs(ref.mass("decay") - m0) < 1e-10, "sans kappa la masse AMR devrait etre inchangee"

        # (b) AVEC kappa uniforme + raffinement + regrid : decroissance persistante ET uniforme.
        sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        sim.add_equation("decay", model=_compile_amr_decay(tmp, "amr1.so"), spatial=sp, time=pops.Explicit())
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.set_refinement(2.0)
        rho0 = _bump_density(n, lo, hi, 1.0, 5.0)
        sim.set_density("decay", rho0)
        sim.set_aux_field("decay", "kappa", 2.0 * np.ones((n, n)))
        masses = [sim.mass("decay")]
        for _ in range(5):  # regrid_every=1 -> the fine aux is rebuilt (zeroed) + re-injected each step
            sim.step(1e-2)
            masses.append(sim.mass("decay"))
        assert sim.n_patches() > 0, "a fine level must exist (regrid path exercised)"
        assert all(masses[i + 1] < masses[i] - 1e-9 for i in range(len(masses) - 1)), \
            "decay must persist across regrid (kappa re-applied each solve), got %r" % masses
        # uniform kappa -> the refined bump region must decay like the rest (named aux reached the fine
        # level). If the fine cells read kappa=0, the bump's ratio would be ~1 while background ~0.9.
        ratio = np.array(sim.density("decay")) / rho0
        assert float(np.std(ratio)) < 1e-2, \
            "named aux must reach the FINE level (uniform decay), std(ratio)=%.3e" % float(np.std(ratio))
        assert float(np.mean(ratio)) < 0.95, "a real decay was expected, mean ratio=%.3f" % float(np.mean(ratio))
        print("OK  AMR single-block named aux + regrid : n_patches=%d, mass %.5f->%.5f, decay std %.2e"
              % (sim.n_patches(), masses[0], masses[-1], float(np.std(ratio))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_const_decay_model(name, c0):
    """Scalaire 'n', flux nul, source CONSTANTE S = -c0*n (c0 fige a la compilation), SANS aux_field
    (n_aux=3) : decroit a un taux FIXE c0, INDEPENDANT du canal aux partage -> temoin d'isolation."""
    m = dsl.Model(name)
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn
    m.flux(x=[zero], y=[zero]); m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(n=nn); m.conservative_from([nn])
    m.source([-(float(c0) * nn)])
    return m


def test_amr_named_aux_multiblock_regrid():
    """ADC-291 : aux NOMME sur le chemin AMR MULTI-BLOCS (AmrRuntime, moteur DIFFERENT du mono-bloc) avec
    regrid actif ET niveau fin. Le canal aux est PARTAGE entre les blocs (comme B_z). Deux blocs :
    'decay' LIT extra_field(0) (kappa GRAND), 'plain' a une source CONSTANTE (taux c0, n_aux=3, ne lit
    pas la composante nommee). Verrouille : (a) 'decay' s'effondre (kappa applique par AmrRuntime + re-
    injecte au fin apres regrid), (b) ISOLATION non triviale : 'plain' decroit SEULEMENT a son taux c0
    (le kappa du canal partage ne fuit PAS dans 'plain') -- si kappa fuyait, 'plain' s'effondrerait aussi."""
    if not _have_compiler():
        print("skip  AMR named aux multi-block (compilateur absent)")
        return
    tmp = tempfile.mkdtemp()
    try:
        n = 24
        sp = pops.FiniteVolume(limiter="none", riemann="rusanov")
        lo, hi = n // 3, 2 * n // 3
        decay_so = _compile_amr_decay(tmp, "amrdecay.so")
        c0 = 1.0
        plain_so = build_const_decay_model("plaindecay", c0).compile(
            os.path.join(tmp, "amrplain.so"), include=INCLUDE, backend="production", target="amr_system")
        sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        sim.add_equation("decay", model=decay_so, spatial=sp, time=pops.Explicit())
        sim.add_equation("plain", model=plain_so, spatial=sp, time=pops.Explicit())
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.set_refinement(2.0)  # refine on the 'decay' bump -> a real fine level + regrid
        sim.set_density("decay", _bump_density(n, lo, hi, 1.0, 5.0))
        sim.set_density("plain", np.ones((n, n)))
        kappa = 50.0  # LARGE: a leak into 'plain' would crash its mass; c0=1 keeps 'plain' mild
        sim.set_aux_field("decay", "kappa", kappa * np.ones((n, n)))
        md0, mp0 = sim.mass("decay"), sim.mass("plain")
        for _ in range(5):
            sim.step(1e-2)
        md1, mp1 = sim.mass("decay"), sim.mass("plain")
        assert sim.n_patches() > 0, "a fine level must exist (multi-block regrid path exercised)"
        assert md1 < 0.5 * md0, \
            "AMR multi-bloc : 'decay' (kappa=50) devrait s'effondrer, got %.6f->%.6f" % (md0, md1)
        # ISOLATION : 'plain' decays ONLY at c0=1 (ratio ~ 0.99^5 ~ 0.95), NOT at kappa. A leak would
        # push it well below 0.9 (toward 'decay's collapse). Lower bound rejects a leak; upper rejects
        # "plain didn't run at all".
        rp = mp1 / mp0
        assert 0.90 < rp < 0.99, \
            "AMR multi-bloc : 'plain' doit decroitre a son taux c0 SEUL (canal partage isole), ratio=%.4f" % rp
        print("OK  AMR multi-block named aux + regrid : decay %.5f->%.5f (collapse), plain ratio %.4f (c0 only)"
              % (md0, md1, rp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_amr_named_aux_rejections():
    """ADC-291 : rejets de la facade AMR set_aux_field (parite avec System) : canal canonique redirige,
    bloc inconnu, champ non declare. Aucun compilateur requis (resolution AVANT le build)."""
    sim = pops.AmrSystem(n=8, L=1.0, periodic=True)
    field = np.ones((8, 8))
    for nm, redirect in (("B_z", "set_magnetic_field"), ("phi", "CANONICAL")):
        try:
            sim.set_aux_field("blk", nm, field)
        except ValueError as ex:
            assert redirect in str(ex), "le message %s devrait mentionner %s : %r" % (nm, redirect, str(ex))
        else:
            raise AssertionError("AmrSystem.set_aux_field(%r) aurait du lever" % nm)
    # bloc inconnu (aucun champ nomme enregistre).
    try:
        sim.set_aux_field("inexistant", "kappa", field)
    except ValueError as ex:
        assert "inexistant" in str(ex)
    else:
        raise AssertionError("AmrSystem.set_aux_field sur bloc inconnu aurait du lever")
    print("OK  AMR facade : B_z/phi rediriges, bloc inconnu rejete")


def main():
    test_form()
    test_facade_rejects()
    test_end_to_end()
    test_polar_named_aux()
    test_amr_named_aux_single_block_regrid()
    test_amr_named_aux_multiblock_regrid()
    test_amr_named_aux_rejections()


if __name__ == "__main__":
    main()
