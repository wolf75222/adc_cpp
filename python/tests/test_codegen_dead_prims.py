#!/usr/bin/env python3
"""Codegen DSL : elimination des primitives mortes + hoist OPT-IN de la reciproque (ADC-200).

emit_cpp_brick emettait TOUTES les primitives du modele dans CHAQUE methode (flux,
max_wave_speed, wave_speeds, to_primitive...). Sur un systeme de moments, to_primitive (copie
identite) emettait ainsi la fermeture complete et ses sqrt, morts ; max_wave_speed n'a besoin que
d'une poignee de primitives mais les emettait toutes. On verifie, sur un modele de moments construit
via adc.moments (fermeture gaussienne, AUCUNE dependance a adc_cases) :

 (a) FILTRAGE : to_primitive n'emet plus la fermeture morte (0 sqrt, 0 primitive) alors que la
     brique en contient ailleurs (sqrt vivants dans flux) ;
 (b) CORRECTION : la brique compile et son flux egale l'evaluation Python de reference (rtol 1e-13)
     -- le filtrage retire du code mort, les valeurs vivantes sont inchangees ;
 (c) HOIST OPT-IN : par defaut le source ne contient pas inv_ ; active, il contient inv_ et son
     flux colle au flux par defaut a rtol 1e-13 (l'arrondi change, ce n'est PAS bit-exact) ;
 (d) _is_zero : une fermeture rendant dsl.Const(0.0) ne fait pas emettre les primitives de
     reconstruction d'ordre superieur (meme comportement que le zero flottant 0.0).

Les points (b) et (c) compilent une brique AUTONOME (en-tetes adc header-only, sans Kokkos) sur le
modele de test_dsl_brick : auto-skip si compilateur ou en-tetes absents.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import adc.moments as M
from adc import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
ORDER = 3          # order=3 -> 10 moments ; top order 4 (pair) -> sqrt (sx, sy) VIVANTS dans le flux
NV = len(M.moment_names(ORDER))

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def _method_body(src, sig):
    """Corps (lignes) de la methode dont la signature contient @p sig, jusqu'a sa fermeture."""
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if sig in l:
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip() == "  }":
                    return lines[i:j + 1]
    raise AssertionError("methode introuvable : %s" % sig)


def _prim_def_count(body):
    """Nombre de definitions de primitive de moments (C.., S.., u, v) dans @p body."""
    pref = ("const adc::Real u =", "const adc::Real v =", "const adc::Real C",
            "const adc::Real S", "const adc::Real sx", "const adc::Real sy")
    return sum(1 for l in body if any(l.strip().startswith(p) for p in pref))


def realizable_states(nstates, seed):
    """Etats de moments REALISABLES : moments bruts d'echantillons gaussiens (M00>0, C20/C02>0)."""
    rng = np.random.default_rng(seed)
    idx = M.moment_indices(ORDER)
    out = []
    for _ in range(nstates):
        a = rng.normal(0.0, 0.5, size=(2, 2))
        cov = a @ a.T + 0.3 * np.eye(2)
        xy = rng.multivariate_normal(rng.normal(0.0, 0.6, size=2), cov, size=4000)
        rho = float(0.5 + rng.random())
        out.append([rho * float(np.mean(xy[:, 0] ** p * xy[:, 1] ** q)) for (p, q) in idx])
    return out


HARNESS = r"""
#include <adc/physics/bricks.hpp>
#include <adc/core/physical_model.hpp>
#include <adc/core/variables.hpp>
__DEFAULT__
__HOIST__
#include <cstdio>
int main(){
  gdef::MomDef D; ghoi::MomHoi H; adc::Aux aux{};
  const double S[][__NV__] = {
__STATES__
  };
  const int ns = sizeof(S)/sizeof(S[0]);
  for(int k=0;k<ns;++k){
    adc::StateVec<__NV__> u{}; for(int i=0;i<__NV__;++i) u[i]=S[k][i];
    for(int dir=0;dir<2;++dir){
      auto fd=D.flux(u,aux,dir); auto fh=H.flux(u,aux,dir);
      for(int i=0;i<__NV__;++i) printf("D %d %d %d %.17g\n", k, dir, i, (double)fd[i]);
      for(int i=0;i<__NV__;++i) printf("H %d %d %d %.17g\n", k, dir, i, (double)fh[i]);
    }
    auto p=D.to_primitive(u);            // exerce to_primitive (filtree)
    (void)p; (void)D.max_wave_speed(u,aux,0);
  }
  return 0;
}
"""


def check_a_filtrage(m):
    print("== (a) FILTRAGE : to_primitive sans fermeture morte ==")
    src = m.emit_cpp_brick(name="MomA")
    tp = _method_body(src, "to_primitive(")
    fx = _method_body(src, "State flux(")
    chk("".join(tp).count("std::sqrt") == 0, "to_primitive : aucun std::sqrt (fermeture morte retiree)")
    chk(_prim_def_count(tp) == 0, "to_primitive : aucune primitive emise (copie identite)")
    chk("".join(fx).count("std::sqrt") >= 2, "flux : sqrt VIVANTS conserves (sx, sy)")
    chk(src.count("std::sqrt") >= 2, "brique : sqrt presents ailleurs (filtrage local, pas global)")


def check_d_is_zero():
    print("== (d) _is_zero reconnait dsl.Const(0.0) ==")
    top = ["C%d%d" % (p, ORDER + 1 - p) for p in range(ORDER + 2)]

    def zconst(S):
        return {"S%d%d" % (p, ORDER + 1 - p): dsl.Const(0.0) for p in range(ORDER + 2)}

    def zfloat(S):
        return {"S%d%d" % (p, ORDER + 1 - p): 0.0 for p in range(ORDER + 2)}

    mc = M.build_moment_model("zc", ORDER, zconst)._m
    mf = M.build_moment_model("zf", ORDER, zfloat)._m
    const_top = [c for c in top if c in mc.prim_defs]
    chk(const_top == [], "fermeture Const(0.0) : reconstruction d'ordre sup. NON emise (%s)" % const_top)
    chk(list(mc.prim_defs) == list(mf.prim_defs), "Const(0.0) == 0.0 flottant (memes primitives)")


def check_bc_numerique(m, cxx):
    print("== (b)/(c) brique compile : flux == reference Python (rtol 1e-13), hoist OPT-IN ==")
    src_def = m.emit_cpp_brick(name="MomDef", namespace="gdef")
    src_hoi = m.emit_cpp_brick(name="MomHoi", namespace="ghoi", hoist_reciprocals=True)
    chk("inv_" not in src_def, "hoist OFF (defaut) : aucun inv_ dans le source")
    chk("inv_" in src_hoi, "hoist ON : reciproque hissee (inv_) emise")

    states = realizable_states(6, 20200)
    sl = ",\n".join("{" + ",".join("%.17g" % v for v in s) + "}" for s in states)
    prog = (HARNESS.replace("__DEFAULT__", src_def).replace("__HOIST__", src_hoi)
            .replace("__STATES__", sl).replace("__NV__", str(NV)))
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "h.cpp")
        exe = os.path.join(tmp, "h")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    fd = {}
    fh = {}
    for line in out.splitlines():
        tag, k, d, i, val = line.split()
        (fd if tag == "D" else fh)[(int(k), int(d), int(i))] = float(val)

    facade = M.build_moment_model("ref", ORDER, M.gaussian_closure(ORDER))  # eval_flux Python
    rel_ref = 0.0
    rel_hoi = 0.0
    for k, s in enumerate(states):
        U = np.array(s, dtype=float).reshape(NV, 1, 1)
        for d in range(2):
            ref = facade.eval_flux(U, {}, d).reshape(NV)
            cd = np.array([fd[(k, d, i)] for i in range(NV)])
            ch = np.array([fh[(k, d, i)] for i in range(NV)])
            den = np.maximum(np.abs(ref), 1e-300)
            rel_ref = max(rel_ref, float(np.max(np.abs(cd - ref) / den)))
            rel_hoi = max(rel_hoi, float(np.max(np.abs(ch - cd) / np.maximum(np.abs(cd), 1e-300))))
    chk(rel_ref < 1e-13, "flux compile (filtre) == eval_flux Python (rtol %.2e)" % rel_ref)
    chk(rel_hoi < 1e-13, "flux hoist == flux defaut (rtol %.2e, arrondi change mais < 1e-13)" % rel_hoi)


def main():
    m = M.build_moment_model("mom", ORDER, M.gaussian_closure(ORDER))._m
    check_a_filtrage(m)
    check_d_is_zero()
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx and os.path.isdir(INCLUDE):
        check_bc_numerique(M.build_moment_model("mom", ORDER, M.gaussian_closure(ORDER))._m, cxx)
    else:
        print("skip  (b)/(c) : compilateur ou en-tetes adc absents (%s)" % INCLUDE)
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
