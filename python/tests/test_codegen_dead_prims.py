#!/usr/bin/env python3
"""DSL codegen: dead-primitive elimination + OPT-IN reciprocal hoist (ADC-200).

emit_cpp_brick used to emit ALL the model primitives in EVERY method (flux,
max_wave_speed, wave_speeds, to_primitive...). On a moment system, to_primitive (identity
copy) thus emitted the full closure and its dead sqrt; max_wave_speed needs only
a handful of primitives but emitted them all. We check, on a moment model built
via adc.moments (Gaussian closure, NO dependency on adc_cases):

 (a) FILTERING: to_primitive no longer emits the dead closure (0 sqrt, 0 primitive) while the
     brick contains some elsewhere (live sqrt in flux);
 (b) CORRECTNESS: the brick compiles and its flux equals the Python reference evaluation (rtol 1e-13)
     -- filtering removes dead code, the live values are unchanged;
 (c) OPT-IN HOIST: by default the source contains no inv_; enabled, it contains inv_ and its
     flux matches the default flux to rtol 1e-13 (rounding changes, it is NOT bit-exact);
 (d) _is_zero: a closure returning dsl.Const(0.0) does not emit the higher-order
     reconstruction primitives (same behavior as the float zero 0.0).

Points (b) and (c) compile a STANDALONE brick (header-only adc headers, without Kokkos) on the
model of test_dsl_brick: auto-skip if the compiler or the headers are absent.
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
    """Body (lines) of the method whose signature contains @p sig, up to its closing brace."""
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if sig in l:
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip() == "  }":
                    return lines[i:j + 1]
    raise AssertionError("method not found: %s" % sig)


def _prim_def_count(body):
    """Number of moment primitive definitions (C.., S.., u, v) in @p body."""
    pref = ("const adc::Real u =", "const adc::Real v =", "const adc::Real C",
            "const adc::Real S", "const adc::Real sx", "const adc::Real sy")
    return sum(1 for l in body if any(l.strip().startswith(p) for p in pref))


def realizable_states(nstates, seed):
    """REALIZABLE moment states: raw moments of Gaussian samples (M00>0, C20/C02>0)."""
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
    auto p=D.to_primitive(u);            // exercises to_primitive (filtered)
    (void)p; (void)D.max_wave_speed(u,aux,0);
  }
  return 0;
}
"""


def check_a_filtrage(m):
    print("== (a) FILTERING: to_primitive without dead closure ==")
    src = m.emit_cpp_brick(name="MomA")
    tp = _method_body(src, "to_primitive(")
    fx = _method_body(src, "State flux(")
    chk("".join(tp).count("std::sqrt") == 0, "to_primitive: no std::sqrt (dead closure removed)")
    chk(_prim_def_count(tp) == 0, "to_primitive: no primitive emitted (identity copy)")
    chk("".join(fx).count("std::sqrt") >= 2, "flux: LIVE sqrt kept (sx, sy)")
    chk(src.count("std::sqrt") >= 2, "brick: sqrt present elsewhere (local filtering, not global)")


def check_d_is_zero():
    print("== (d) _is_zero recognizes dsl.Const(0.0) ==")
    top = ["C%d%d" % (p, ORDER + 1 - p) for p in range(ORDER + 2)]

    def zconst(S):
        return {"S%d%d" % (p, ORDER + 1 - p): dsl.Const(0.0) for p in range(ORDER + 2)}

    def zfloat(S):
        return {"S%d%d" % (p, ORDER + 1 - p): 0.0 for p in range(ORDER + 2)}

    mc = M.build_moment_model("zc", ORDER, zconst)._m
    mf = M.build_moment_model("zf", ORDER, zfloat)._m
    const_top = [c for c in top if c in mc.prim_defs]
    chk(const_top == [], "Const(0.0) closure: higher-order reconstruction NOT emitted (%s)" % const_top)
    chk(list(mc.prim_defs) == list(mf.prim_defs), "Const(0.0) == float 0.0 (same primitives)")


def check_bc_numerique(m, cxx):
    print("== (b)/(c) brick compiles: flux == Python reference (rtol 1e-13), OPT-IN hoist ==")
    src_def = m.emit_cpp_brick(name="MomDef", namespace="gdef")
    src_hoi = m.emit_cpp_brick(name="MomHoi", namespace="ghoi", hoist_reciprocals=True)
    chk("inv_" not in src_def, "hoist OFF (default): no inv_ in the source")
    chk("inv_" in src_hoi, "hoist ON: hoisted reciprocal (inv_) emitted")

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
    chk(rel_ref < 1e-13, "compiled flux (filtered) == Python eval_flux (rtol %.2e)" % rel_ref)
    chk(rel_hoi < 1e-13, "hoist flux == default flux (rtol %.2e, rounding changes but < 1e-13)" % rel_hoi)


def main():
    m = M.build_moment_model("mom", ORDER, M.gaussian_closure(ORDER))._m
    check_a_filtrage(m)
    check_d_is_zero()
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx and os.path.isdir(INCLUDE):
        check_bc_numerique(M.build_moment_model("mom", ORDER, M.gaussian_closure(ORDER))._m, cxx)
    else:
        print("skip  (b)/(c): compiler or adc headers absent (%s)" % INCLUDE)
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
