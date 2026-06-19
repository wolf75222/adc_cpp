"""Predicat REEL/COMPLEXE du spectre dans la DSL de projection (ADC-362) : surface DSL du predicat
C++ ``adc::EigBounds::all_real`` ajoute par ADC-276. ``dsl.eig_all_real(rows, im_tol=1e-5)`` rend une
valeur scalaire DSL valant 1.0 si le spectre de la PETITE matrice dense @c rows est REEL (et le bloc a
CONVERGE), 0.0 sinon -- paire complexe OU non-convergence. Compose sans branche dans m.projection
(ADC-177) : "si le bloc a une paire complexe, corriger" s'ecrit ``complexe = 1 - eig_all_real(...)``,
puis un melange max/min sans if.

POURQUOI le predicat et non ``eig_max_im(...) <= tol`` : sous le repli Gershgorin (non-convergence d'un
bloc >= 3), ``real_eig_minmax`` force ``max_im = 0`` PAR CONVENTION (rien n'est calcule) ; un test brut
``max_im <= tol`` lirait donc un repli comme un spectre REEL -- exactement l'ecueil. ``all_real`` est
verrouille sur ``converged`` (``converged && max_im <= im_tol*scale``) : un repli -> false -> 0.0
(= PAS reel), jamais 1.0. Le codegen abaisse donc le predicat sur ``.all_real(im_tol)``, pas sur un
comparatif ``.max_im`` (verrouille en (3)), et la non-convergence reste conservatrice au runtime (4).

SEMANTIQUE HOTE du kUnknown : le miroir numpy (np.linalg.eigvals / LAPACK) CONVERGE toujours, il n'a
donc PAS de kUnknown ; sur matrices saines (cas vise) hote et brique coincident. La seule divergence
possible est une non-convergence cote DEVICE (bloc pathologique sous le cap QR), ou la brique rend 0.0
(PAS reel) tandis que le miroir hote rendrait 1.0 : c'est la direction SURE (le device est conservateur).

On verifie :
 (1) eval numpy de eig_all_real : spectre reel (symetrique / diagonal / triangulaire) -> 1.0 ;
     paire complexe (rotation) -> 0.0 ; champ vectorise + cas scalaire ; coherent avec la formule
     RELATIVE max_im <= im_tol*max(|lmin|,|lmax|,1) ;
 (2) contrat de tolerance RELATIVE + parametre im_tol : une vraie paire complexe sous im_tol*scale est
     rendue reelle PAR DESIGN (asymetrie documentee de all_real), et im_tol fait basculer le verdict ;
 (3) codegen : la brique #include dense_eig.hpp, declare le foncteur adc_eig_all_real_KxK abaisse sur
     ``adc::Real(adc::real_eig_minmax(M).all_real(<im_tol>))`` (verrou de surete : PAS un comparatif
     ``.max_im``), passe im_tol en argument, pas de lambda ; project() appelle le foncteur ;
 (4) [compilateur] non-convergence CONSERVATRICE : ``real_eig_minmax(companion3x3, /*max_iter=*/0)``
     -> converged == false, max_im == 0 (le piege), all_real() == false (0.0, JAMAIS reel) ;
 (5) [compilateur] coherence DSL vs C++ : la brique GENEREE project(U) sur un champ de matrices
     traversant la frontiere reel/complexe == le miroir numpy CELLULE PAR CELLULE (atol 1e-10), les
     deux branches exercees ;
 (6) le hook m.projection sans predicat (ADC-177) reste INCHANGE : aucun include dense_eig, aucun
     foncteur (extension strictement additive) ;
 (7) CSE : deux eig_all_real de la MEME matrice mais im_tol DIFFERENT ne partagent PAS la locale (la
     cle CSE inclut im_tol) ; im_tol identique -> partagee.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Import DIRECT du module dsl (pur Python) : le predicat, son eval numpy et son codegen ne dependent
# pas de l'extension compilee _adc.
_DSL_PATH = os.path.join(os.path.dirname(__file__), "..", "adc", "dsl.py")
_spec = importlib.util.spec_from_file_location("adc_dsl_eig_pred", os.path.abspath(_DSL_PATH))
dsl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsl)

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
TOL_EVAL = 1e-12
TOL_CPP = 1e-10

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def ref_all_real(M, im_tol):
    """Reference numpy de all_real sur un champ (..., k, k) -> champ {0.0, 1.0} (...). MEME formule
    RELATIVE que adc::EigBounds::all_real : max_im <= im_tol*max(|lmin|,|lmax|,1) (hote = toujours
    converge -> pas de kUnknown)."""
    ev = np.linalg.eigvals(M)
    max_im = np.max(np.abs(ev.imag), axis=-1)
    lmin = np.min(ev.real, axis=-1)
    lmax = np.max(ev.real, axis=-1)
    scale = np.maximum(np.maximum(np.abs(lmin), np.abs(lmax)), 1.0)
    return (max_im <= im_tol * scale).astype(float)


def test_eval_real_vs_complex():
    print("== (1) eval numpy eig_all_real : reel -> 1.0, paire complexe -> 0.0 ==")
    # paire complexe : rotation [[c,-s],[s,c]] -> VP c +- i s, |Im| = |s| > 0 -> complexe.
    c, s = 0.3, 0.8
    rot = [[dsl.Const(c), dsl.Const(-s)], [dsl.Const(s), dsl.Const(c)]]
    chk(float(dsl.eig_all_real(rot).eval({})) == 0.0,
        "scalaire : rotation (VP complexes) -> eig_all_real = 0.0")
    # spectre reel : matrice symetrique 2x2 -> VP reelles -> 1.0.
    sym = [[dsl.Const(2.0), dsl.Const(1.0)], [dsl.Const(1.0), dsl.Const(3.0)]]
    chk(float(dsl.eig_all_real(sym).eval({})) == 1.0,
        "scalaire : symetrique (VP reelles) -> eig_all_real = 1.0")
    # spectre reel : triangulaire 3x3 (VP = diagonale, reelles) -> 1.0.
    tri = [[dsl.Const(1.0), dsl.Const(5.0), dsl.Const(-2.0)],
           [dsl.Const(0.0), dsl.Const(2.0), dsl.Const(4.0)],
           [dsl.Const(0.0), dsl.Const(0.0), dsl.Const(3.0)]]
    chk(float(dsl.eig_all_real(tri).eval({})) == 1.0,
        "scalaire : triangulaire 3x3 (VP reelles) -> eig_all_real = 1.0")

    # champ vectorise : [[q0,-q1],[q1,q0]] -> VP q0 +- i|q1| ; reel ssi |q1| <= im_tol*max(|q0|,1).
    # q0 aleatoire (exerce le scale = max(|q0|,1) par cellule) ; q1 alterne minuscule (reel a coup sur)
    # et tres grand (complexe a coup sur, > im_tol*scale) -> les DEUX verdicts apparaissent (non flaky).
    rng = np.random.default_rng(362)
    n, im_tol = 64, 1e-5
    q0 = rng.standard_normal(n)
    q1 = np.where(np.arange(n) % 2 == 0, 1e-9, 10.0 * (1.0 + np.abs(q0)))
    env = {"q0": q0, "q1": q1}
    rows = [[dsl.Var("q0", "cons"), -dsl.Var("q1", "cons")],
            [dsl.Var("q1", "cons"), dsl.Var("q0", "cons")]]
    got = dsl.eig_all_real(rows, im_tol=im_tol).eval(env)
    M = np.zeros((n, 2, 2))
    M[:, 0, 0] = q0; M[:, 0, 1] = -q1; M[:, 1, 0] = q1; M[:, 1, 1] = q0
    ref = ref_all_real(M, im_tol)
    chk(np.array_equal(got, ref),
        "champ 2x2 : eig_all_real == reference numpy all_real (n=%d)" % n)
    chk(np.any(got == 0.0) and np.any(got == 1.0),
        "champ : les deux verdicts (reel / complexe) apparaissent")


def test_im_tol_relative_asymmetry():
    print("== (2) tolerance RELATIVE + parametre im_tol (asymetrie all_real) ==")
    # grande echelle : VP 1e8 +- i, |Im| = 1 <= 1e-5*1e8 = 1e3 -> rendu REEL par design.
    big = [[dsl.Const(1e8), dsl.Const(-1.0)], [dsl.Const(1.0), dsl.Const(1e8)]]
    chk(float(dsl.eig_all_real(big).eval({})) == 1.0,
        "1e8 +- i au seuil 1e-5 : paire complexe sous im_tol*scale -> reel (asymetrie)")
    # petite echelle : VP +- 1e-3 i, scale = 1 ; im_tol = 1e-5 -> 1e-3 > 1e-5 -> complexe (0.0).
    small = [[dsl.Const(0.0), dsl.Const(-1e-3)], [dsl.Const(1e-3), dsl.Const(0.0)]]
    chk(float(dsl.eig_all_real(small, im_tol=1e-5).eval({})) == 0.0,
        "+- 1e-3 i au seuil 1e-5 -> complexe (0.0)")
    # meme matrice, im_tol releve a 1.0 -> 1e-3 <= 1.0 -> reel (1.0) : im_tol fait basculer le verdict.
    chk(float(dsl.eig_all_real(small, im_tol=1.0).eval({})) == 1.0,
        "meme matrice, im_tol=1.0 -> reel (1.0) : im_tol fait basculer le verdict")
    # im_tol invalide -> rejet explicite (pas de seuil <= 0).
    for bad in (0.0, -1e-3):
        try:
            dsl.eig_all_real(small, im_tol=bad)
            chk(False, "im_tol=%r rejete" % bad)
        except ValueError:
            chk(True, "im_tol=%r rejete (ValueError)" % bad)


def build_pred_model(tag, im_tol=0.5, target=9.0):
    """Modele jouet 3 variables. Matrice [[q0,-q1],[q1,q0]] (VP q0 +- i|q1|). Projection branchless :
    si le spectre N'EST PAS reel (paire complexe), mettre q2 a une cible ; sinon q2 inchange.
    complexe = 1 - eig_all_real ; q2 <- q2*eig_all_real + cible*(1 - eig_all_real)."""
    m = dsl.HyperbolicModel("toypred_" + tag)
    q0, q1, q2 = m.conservative_vars("q0", "q1", "q2")
    m.set_flux(x=[q0, q1, q2], y=[0.5 * q0, 0.5 * q1, 0.5 * q2])
    m.set_eigenvalues(x=[dsl.Const(1.0)], y=[dsl.Const(0.5)])
    m.set_primitive_state("q0", "q1", "q2")
    m.set_conservative_from([q0, q1, q2])
    is_real = dsl.eig_all_real([[q0, -q1], [q1, q0]], im_tol=im_tol)
    m.projection([q0, q1, q2 * is_real + target * (1.0 - is_real)])
    return m, im_tol, target


def test_codegen():
    print("== (3) codegen : foncteur all_real abaisse sur EigBounds::all_real (PAS .max_im) ==")
    m, im_tol, _ = build_pred_model("cg")
    src = m.emit_cpp_brick(name="ToyPredCg")
    chk("#include <adc/numerics/dense_eig.hpp>" in src, "brique inclut dense_eig.hpp")
    chk("static ADC_HD adc::Real adc_eig_all_real_2x2(" in src,
        "foncteur nomme adc_eig_all_real_2x2 declare")
    chk("adc::real_eig_minmax(M).all_real(" in src,
        "le foncteur abaisse sur EigBounds::all_real (verrou de surete : converged)")
    chk(".max_im" not in src,
        "le predicat n'est PAS abaisse sur un comparatif .max_im (repli => 0.0, jamais reel)")
    chk(repr(float(im_tol)) in src, "im_tol passe en argument du foncteur (seuil relatif)")
    chk("[&]" not in src and "[=]" not in src, "aucune lambda etendue (device-clean)")
    chk("adc_eig_all_real_2x2(" in src.split("State project")[1], "project() appelle le foncteur")


def test_cse_im_tol():
    print("== (7) CSE : im_tol DIFFERENT -> locales distinctes ; im_tol identique -> partagee ==")
    q0 = dsl.Var("q0", "cons"); q1 = dsl.Var("q1", "cons")
    mat = [[q0, -q1], [q1, q0]]
    a = dsl.eig_all_real(mat, im_tol=1e-5)
    b = dsl.eig_all_real(mat, im_tol=1e-3)
    c = dsl.eig_all_real(mat, im_tol=1e-5)
    chk(dsl._key(a) != dsl._key(b), "cles CSE distinctes pour im_tol different")
    chk(dsl._key(a) == dsl._key(c), "cles CSE identiques pour im_tol identique")


def test_additive():
    print("== (6) extension ADDITIVE : projection SANS predicat inchangee (ADC-177) ==")
    m = dsl.HyperbolicModel("toyplain_pred")
    q0, q1 = m.conservative_vars("q0", "q1")
    m.set_flux(x=[q0, q1], y=[0.5 * q0, 0.5 * q1])
    m.set_eigenvalues(x=[dsl.Const(1.0)], y=[dsl.Const(0.5)])
    m.set_primitive_state("q0", "q1")
    m.set_conservative_from([q0, q1])
    m.projection([(q0 + dsl.abs_(q0)) / 2.0, q1])  # clamp ADC-177, aucun predicat
    src = m.emit_cpp_brick(name="ToyPlainPred")
    chk("dense_eig.hpp" not in src, "aucun include dense_eig sans predicat")
    chk("adc_eig_" not in src, "aucun foncteur eig sans predicat (additif)")


def test_fallback_conservative(cxx, tmp):
    print("== (4) [compilateur] non-convergence CONSERVATRICE : all_real() == false sous repli ==")
    main = os.path.join(tmp, "fallback_main.cpp")
    with open(main, "w") as f:
        f.write(
            "#include <cstdio>\n"
            "#include <adc/numerics/dense_eig.hpp>\n"
            "int main() {\n"
            # companion 3x3 plein (ne deflate pas en 1x1/2x2) -> cap QR 0 force le repli Gershgorin.
            "  const adc::Real A[3][3] = {{0,0,-6},{1,0,11},{0,1,-6}};\n"
            "  const adc::EigBounds b = adc::real_eig_minmax(A, /*max_iter=*/0);\n"
            "  std::printf(\"%d %d %d\\n\", (int)b.converged,\n"
            "              (int)(b.max_im == adc::Real(0)), (int)b.all_real());\n"
            "  return 0;\n"
            "}\n")
    exe = os.path.join(tmp, "fallback_main")
    cp = subprocess.run([cxx, "-std=c++17", "-I", INCLUDE, main, "-o", exe],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        chk(False, "compilation du test de repli (voir stderr)")
        print(cp.stderr[:2000]); return
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
    converged, maxim_zero, all_real = (v == "1" for v in out)
    chk(not converged, "repli declenche (converged == false) au cap QR 0")
    chk(maxim_zero, "le piege : max_im == 0 sous repli (lirait reel pour un test .max_im brut)")
    chk(not all_real, "all_real() == false sous repli -> 0.0, JAMAIS reel (conservateur)")


def test_cpp_brick_vs_numpy(cxx, tmp):
    print("== (5) [compilateur] brique generee project(U) == reference numpy (cellule par cellule) ==")
    # q0 == 0 sur tout le champ -> scale = 1, seuil = im_tol : les deux branches exercees a coup sur.
    m, im_tol, target = build_pred_model("cpp", im_tol=0.5, target=9.0)
    hpp = os.path.join(tmp, "pred_brick.hpp")
    with open(hpp, "w") as f:
        f.write(m.emit_cpp_brick(name="ToyPredCpp"))

    rng = np.random.default_rng(3620)
    n = 64
    q0 = np.zeros(n)
    q1 = rng.uniform(-1.0, 1.0, n)  # |q1| traverse im_tol=0.5 -> reel et complexe
    q2_in = np.zeros(n)

    main = os.path.join(tmp, "pred_main.cpp")
    with open(main, "w") as f:
        f.write(
            "#include <cstdio>\n"
            "#include <adc/core/types.hpp>\n"
            "#include <adc/core/state.hpp>\n"
            "#include <adc/core/variables.hpp>\n"
            '#include "pred_brick.hpp"\n'
            "int main(int argc, char** argv) {\n"
            "  adc_generated::ToyPredCpp m;\n"
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
    exe = os.path.join(tmp, "pred_main")
    cp = subprocess.run([cxx, "-std=c++17", "-I", INCLUDE, main, "-o", exe],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        chk(False, "compilation de la brique generee (voir stderr)")
        print(cp.stderr[:2000]); return
    chk(True, "la brique generee compile contre les en-tetes adc")

    out = os.path.join(tmp, "q2.txt")
    args = [exe, out]
    for i in range(n):
        args += ["%.17g" % q0[i], "%.17g" % q1[i], "%.17g" % q2_in[i]]
    subprocess.run(args, check=True)
    q2_cpp = np.loadtxt(out)

    M = np.zeros((n, 2, 2))
    M[:, 0, 0] = q0; M[:, 0, 1] = -q1; M[:, 1, 0] = q1; M[:, 1, 1] = q0
    is_real = ref_all_real(M, im_tol)
    q2_ref = q2_in * is_real + target * (1.0 - is_real)
    d = float(np.max(np.abs(q2_cpp - q2_ref)))
    chk(np.allclose(q2_cpp, q2_ref, rtol=0.0, atol=TOL_CPP),
        "project(U) C++ == numpy sur %d cellules (ecart max %.2e)" % (n, d))
    chk(np.any(is_real > 0.5) and np.any(is_real < 0.5),
        "les deux branches (reel / complexe) exercees (%d reelles / %d)"
        % (int(np.sum(is_real > 0.5)), n))


def main():
    test_eval_real_vs_complex()
    test_im_tol_relative_asymmetry()
    test_codegen()
    test_cse_im_tol()
    test_additive()

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx and os.path.isdir(INCLUDE):
        tmp = tempfile.mkdtemp()
        try:
            test_fallback_conservative(cxx, tmp)
            test_cpp_brick_vs_numpy(cxx, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("== (4)+(5) skip : compilateur ou en-tetes adc absents ==")

    print("FAILS =", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
