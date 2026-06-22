"""Temoin de VALEURS PROPRES dans la DSL de projection (ADC-289) : extension ADDITIVE du hook
m.projection (ADC-177). dsl.eig_max_im(rows) / eig_lmin / eig_lmax rendent une valeur scalaire (Expr)
du spectre d'une PETITE matrice dense construite a partir d'expressions de moments, via
``adc::real_eig_minmax`` (dense_eig.hpp, ADC_HD, device-safe, repli Gershgorin). max_im = plus grande
|Im(lambda)| = TEMOIN de VP complexes (0 = spectre reel). Sert la branche "suppression des VP
complexes" de relaxation15 (ADC-275) : "si une matrice de moments a une VP complexe, corriger" s'ecrit
en masque max/min/sign sur max_im, SANS branche dynamique (contrat pointwise / idempotent d'ADC-177).

Codegen device-clean : un FONCTEUR NOMME (methode statique ADC_HD de la brique) remplit M[k][k] et
appelle real_eig_minmax -- jamais de lambda etendue cross-TU (casse nvcc).

On verifie :
 (1) eval numpy de eig_max_im / eig_lmin / eig_lmax == reference np.linalg.eigvals (champ par champ,
     matrices 2x2 et 3x3 ; tolerance 1e-12) ;
 (2) codegen : la brique #include dense_eig.hpp et declare le foncteur nomme adc_eig_max_im_KxK
     (pas de lambda ; appel scalaire compatible CSE) ;
 (3) [compilateur] la brique GENEREE compile contre les en-tetes adc et son hook project(U, aux),
     execute sur un CHAMP de matrices construites des vars, reproduit la reference numpy CELLULE PAR
     CELLULE : projection jouet "si max_im > tol alors q <- cible" branchless == numpy (atol 1e-10) ;
 (4) [_adc] semantique POST-PAS de bout en bout (production + aot) : le hook tourne dans le System et
     l'etat post-pas == transport-sans-hook puis projection numpy (mirroir projection_value) ;
 (5) le hook m.projection sans temoin VP (ADC-177) reste INCHANGE : aucun include dense_eig, aucun
     foncteur emis (extension strictement additive, test_projection_hook reste vert).
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Import DIRECT du module dsl (pur Python) : le temoin VP, son eval numpy et son codegen ne dependent
# pas de l'extension compilee _adc. La partie System (4) est gardee par la disponibilite de _adc.
_DSL_PATH = os.path.join(os.path.dirname(__file__), "..", "adc", "dsl.py")
_spec = importlib.util.spec_from_file_location("adc_dsl_eig", os.path.abspath(_DSL_PATH))
dsl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsl)

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
TOL_EVAL = 1e-12   # eval numpy vs np.linalg.eigvals (meme algebre numpy des deux cotes)
TOL_CPP = 1e-10    # brique C++ (Francis QR sur pile) vs numpy : matrices saines, simples/separees

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def ref_field(M, field):
    """Reference numpy : np.linalg.eigvals sur un champ (..., k, k) -> champ scalaire (...)."""
    ev = np.linalg.eigvals(M)
    if field == "max_im":
        return np.max(np.abs(ev.imag), axis=-1)
    if field == "lmin":
        return np.min(ev.real, axis=-1)
    return np.max(ev.real, axis=-1)


def test_eval_vs_numpy():
    print("== (1) eval numpy eig_max_im / eig_lmin / eig_lmax == np.linalg.eigvals ==")
    rng = np.random.default_rng(289)
    for k in (2, 3):
        # champ de matrices : entrees = vars m_ij, valeurs = champs aleatoires (rotation -> VP complexes)
        names = [["a%d%d" % (i, j) for j in range(k)] for i in range(k)]
        rows = [[dsl.Var(names[i][j], "cons") for j in range(k)] for i in range(k)]
        env, M = {}, np.empty((7, k, k))
        for i in range(k):
            for j in range(k):
                v = rng.standard_normal(7)
                env[names[i][j]] = v
                M[:, i, j] = v
        for field, fn in (("max_im", dsl.eig_max_im), ("lmin", dsl.eig_lmin),
                          ("lmax", dsl.eig_lmax)):
            got = fn(rows).eval(env)
            ref = ref_field(M, field)
            d = float(np.max(np.abs(got - ref)))
            chk(np.allclose(got, ref, rtol=0.0, atol=TOL_EVAL),
                "%dx%d %s == numpy (ecart max %.2e)" % (k, k, field, d))
        # cas scalaire (pas de champ) : matrice de rotation explicite [[c, -s],[s, c]] -> max_im = |s|
        if k == 2:
            c, s = 1.3, 0.8
            wit = dsl.eig_max_im([[dsl.Const(c), dsl.Const(-s)], [dsl.Const(s), dsl.Const(c)]])
            chk(abs(float(wit.eval({})) - abs(s)) < TOL_EVAL,
                "scalaire : rotation [[c,-s],[s,c]] -> max_im = |s| = %.3f" % abs(s))


def build_eig_model(tag):
    """Modele jouet 3 variables (q0, q1, q2). Projection : si la matrice 2x2 [[q0, -q1],[q1, q0]]
    (VP = q0 +- i q1) a une |Im| > tol, mettre q2 a une cible ; sinon q2 inchange. Ecrit en masque
    max/min/sign sur max_im, SANS if : mask = (sign(max_im - tol) + 1)/2 ; q2 <- q2 (1-mask) + cible*mask."""
    tol, target = 0.5, 9.0
    m = dsl.HyperbolicModel("toyeig_" + tag)
    q0, q1, q2 = m.conservative_vars("q0", "q1", "q2")
    m.set_flux(x=[q0, q1, q2], y=[0.5 * q0, 0.5 * q1, 0.5 * q2])
    m.set_eigenvalues(x=[dsl.Const(1.0)], y=[dsl.Const(0.5)])
    m.set_primitive_state("q0", "q1", "q2")
    m.set_conservative_from([q0, q1, q2])
    wit = dsl.eig_max_im([[q0, -q1], [q1, q0]])
    mask = 0.5 * (dsl.sign(wit - tol) + 1.0)  # 1 si max_im > tol, 0 sinon (branchless)
    m.projection([q0, q1, q2 * (1.0 - mask) + target * mask])
    return m, tol, target


def test_codegen():
    print("== (2) codegen : include dense_eig + foncteur nomme (pas de lambda) ==")
    m, _, _ = build_eig_model("cg")
    src = m.emit_cpp_brick(name="ToyEigCg")
    chk("#include <adc/numerics/linalg/dense_eig.hpp>" in src, "brique inclut dense_eig.hpp")
    chk("static ADC_HD adc::Real adc_eig_max_im_2x2(" in src, "foncteur nomme adc_eig_max_im_2x2 declare")
    chk("adc::real_eig_minmax(M).max_im" in src, "le foncteur appelle real_eig_minmax(M).max_im")
    chk("[&]" not in src and "[=]" not in src, "aucune lambda etendue (device-clean)")
    chk("adc_eig_max_im_2x2(" in src.split("State project")[1], "project() appelle le foncteur")


def test_cpp_brick_vs_numpy(cxx, tmp):
    print("== (3) [compilateur] brique generee project(U) == reference numpy (cellule par cellule) ==")
    m, tol, target = build_eig_model("cpp")
    hpp = os.path.join(tmp, "eig_brick.hpp")
    with open(hpp, "w") as f:
        f.write(m.emit_cpp_brick(name="ToyEigCpp"))

    # champ de N cellules : q0, q1 varies -> matrice [[q0,-q1],[q1,q0]], VP q0 +- i q1, max_im = |q1|.
    rng = np.random.default_rng(2890)
    n = 64
    q0 = rng.standard_normal(n)
    q1 = rng.uniform(-1.0, 1.0, n)  # |q1| traverse tol=0.5 -> exerce LES DEUX branches du masque
    q2_in = np.zeros(n)

    main = os.path.join(tmp, "eig_main.cpp")
    with open(main, "w") as f:
        f.write(
            "#include <cstdio>\n"
            "#include <adc/core/foundation/types.hpp>\n"
            "#include <adc/core/state/state.hpp>\n"
            "#include <adc/core/state/variables.hpp>\n"
            '#include "eig_brick.hpp"\n'
            "int main(int argc, char** argv) {\n"
            "  adc_generated::ToyEigCpp m;\n"
            "  adc::Aux a{};\n"
            "  std::FILE* fp = std::fopen(argv[1], \"w\");\n"
            "  for (int i = 2; i < argc; i += 3) {\n"
            "    adc::StateVec<3> U{atof(argv[i]), atof(argv[i+1]), atof(argv[i+2])};\n"
            "    auto P = m.project(U, a);\n"
            "    std::fprintf(fp, \"%.17g\\n\", (double)P[2]);\n"
            "  }\n"
            "  std::fclose(fp);\n"
            "  return 0;\n"
            "}\n")
    exe = os.path.join(tmp, "eig_main")
    cp = subprocess.run([cxx, "-std=c++17", "-I", INCLUDE, main, "-o", exe],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        chk(False, "compilation de la brique generee (voir stderr)")
        print(cp.stderr[:2000])
        return
    chk(True, "la brique generee compile contre les en-tetes adc")

    out = os.path.join(tmp, "q2.txt")
    args = [exe, out]
    for i in range(n):
        args += ["%.17g" % q0[i], "%.17g" % q1[i], "%.17g" % q2_in[i]]
    subprocess.run(args, check=True)
    q2_cpp = np.loadtxt(out)

    # reference numpy : max_im du MEME champ de matrices, puis le MEME masque branchless.
    M = np.zeros((n, 2, 2))
    M[:, 0, 0] = q0; M[:, 0, 1] = -q1; M[:, 1, 0] = q1; M[:, 1, 1] = q0
    max_im = ref_field(M, "max_im")
    mask = 0.5 * (np.sign(max_im - tol) + 1.0)
    q2_ref = q2_in * (1.0 - mask) + target * mask
    d = float(np.max(np.abs(q2_cpp - q2_ref)))
    chk(np.allclose(q2_cpp, q2_ref, rtol=0.0, atol=TOL_CPP),
        "project(U) C++ == numpy sur %d cellules (ecart max %.2e)" % (n, d))
    # les DEUX branches sont exercees (sinon le test est trivial).
    chk(np.any(mask > 0.5) and np.any(mask < 0.5),
        "les deux branches (VP complexe / reelle) sont exercees (%d/%d corrigees)"
        % (int(np.sum(mask > 0.5)), n))
    # la reference numpy max_im == |q1| (rotation) : le foncteur calcule bien la PARTIE IMAGINAIRE.
    chk(np.allclose(max_im, np.abs(q1), atol=TOL_CPP),
        "temoin = |Im| : max_im == |q1| de la rotation (ecart %.2e)"
        % float(np.max(np.abs(max_im - np.abs(q1)))))


def test_additive():
    print("== (5) extension ADDITIVE : projection SANS temoin VP inchangee (ADC-177) ==")
    m = dsl.HyperbolicModel("toyplain")
    q0, q1 = m.conservative_vars("q0", "q1")
    m.set_flux(x=[q0, q1], y=[0.5 * q0, 0.5 * q1])
    m.set_eigenvalues(x=[dsl.Const(1.0)], y=[dsl.Const(0.5)])
    m.set_primitive_state("q0", "q1")
    m.set_conservative_from([q0, q1])
    m.projection([(q0 + dsl.abs_(q0)) / 2.0, q1])  # clamp ADC-177, aucun temoin VP
    src = m.emit_cpp_brick(name="ToyPlain")
    chk("dense_eig.hpp" not in src, "aucun include dense_eig sans temoin VP")
    chk("adc_eig_" not in src, "aucun foncteur eig sans temoin VP (additif)")


def test_system_end_to_end():
    """(4) Bout en bout via _adc : le hook tourne dans le System, etat post-pas == projection numpy.
    Garde sur la disponibilite de l'extension compilee (import adc) ; sinon ignore."""
    print("== (4) [_adc] semantique POST-PAS (production + aot) == reference numpy ==")
    try:
        import adc
        from adc import dsl as dsl_pkg
    except Exception as ex:  # noqa: BLE001
        print("  skip  extension _adc absente (%s) -- (3) couvre deja la numerique compilee"
              % type(ex).__name__)
        return
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        print("  skip  pas de compilateur C++")
        return

    N, L, DT, NSTEPS = 24, 1.0, 1e-3, 3
    tol, target = 0.5, 9.0

    def build_pkg(tag):
        m = dsl_pkg.HyperbolicModel("toyeigsys_" + tag)
        q0, q1, q2 = m.conservative_vars("q0", "q1", "q2")
        m.set_flux(x=[q0, q1, q2], y=[0.5 * q0, 0.5 * q1, 0.5 * q2])
        m.set_eigenvalues(x=[dsl_pkg.Const(1.0)], y=[dsl_pkg.Const(0.5)])
        m.set_primitive_state("q0", "q1", "q2")
        m.set_conservative_from([q0, q1, q2])
        wit = dsl_pkg.eig_max_im([[q0, -q1], [q1, q0]])
        mask = 0.5 * (dsl_pkg.sign(wit - tol) + 1.0)
        m.projection([q0, q1, q2 * (1.0 - mask) + target * mask])
        return m

    def init(n):
        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        q0 = np.sin(2 * np.pi * X)
        q1 = 0.9 * np.cos(2 * np.pi * Y)   # |q1| traverse tol -> les deux branches actives
        q2 = np.zeros((n, n))
        return np.stack([q0, q1, q2])

    def make_sys(so, adder):
        s = adc.System(n=N, L=L, periodic=True)
        if adder == "native":
            s._s.add_native_block("toy", so, limiter="minmod", riemann="rusanov",
                                  recon="conservative", time="explicit", gamma=1.4, substeps=1)
        else:
            s._s.add_compiled_block("toy", so, limiter="minmod", riemann="rusanov",
                                    recon="conservative", time="explicit", substeps=1)
        s.set_state("toy", init(N))
        return s

    tmp = tempfile.mkdtemp()
    try:
        m_eig = build_pkg("e")
        m_none = build_pkg("n")  # meme transport ; on neutralise sa projection pour la reference
        m_none._proj = None
        for backend, adder in (("production", "native"), ("aot", "aot")):
            so = m_eig.compile(os.path.join(tmp, "eig_%s.so" % backend), INCLUDE, backend=backend)
            so_n = m_none.compile(os.path.join(tmp, "none_%s.so" % backend), INCLUDE, backend=backend)
            # run AVEC hook
            s = make_sys(so, adder)
            run = []
            for _ in range(NSTEPS):
                s.step(DT)
                run.append(np.array(s.get_state("toy")).reshape(3, N, N))
            # reference : transport SANS hook un pas, puis projection numpy (mirroir projection_value)
            sr = make_sys(so_n, adder)
            cur, ref = init(N), []
            for _ in range(NSTEPS):
                sr.set_state("toy", cur)
                sr.step(DT)
                cur = m_eig.projection_value(np.array(sr.get_state("toy")).reshape(3, N, N))
                ref.append(cur)
            d = max(float(np.max(np.abs(a - b))) for a, b in zip(run, ref))
            chk(all(np.allclose(a, b, rtol=0.0, atol=1e-10) for a, b in zip(run, ref)),
                "%s : etat post-pas == transport puis projection(temoin VP) numpy (ecart %.2e)"
                % (backend, d))
            # le temoin VP est ACTIF : q2 est mis a la cible dans au moins une cellule.
            chk(any(np.any(np.isclose(a[2], target)) for a in run),
                "%s : la branche VP complexe corrige q2 (cible atteinte)" % backend)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_eval_vs_numpy()
    test_codegen()
    test_additive()

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx and os.path.isdir(INCLUDE):
        tmp = tempfile.mkdtemp()
        try:
            test_cpp_brick_vs_numpy(cxx, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("== (3) skip : compilateur ou en-tetes adc absents ==")

    test_system_end_to_end()

    print("FAILS =", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
