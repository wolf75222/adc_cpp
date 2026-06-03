"""Dispatch TYPE-ERASED a l'execution : un programme principal compile SANS connaitre le type de la
brique generee charge un .so qui lui rend un adc::IModel<4>*, et l'utilise polymorphiquement (vtable).

C'est l'item (a) du reste (cf. docs/ARCHITECTURE_CIBLE.md) : la brique GENEREE (EulerGen) est enrobee
dans adc::ModelAdapter, exposee via une fabrique extern "C", chargee a l'execution (dlopen), et
dispatchee par l'interface virtuelle. Le main ne voit QUE adc::IModel<4> + adc::Euler (l'oracle), pas
EulerGen. Chemin HOTE (les appels virtuels ne vont pas sur GPU) : pendant compile de adc.PythonFlux.
"""
import os
import platform
import shutil
import subprocess
import tempfile

from test_dsl_brick import build_euler_brick   # Euler en formules + prim_state + cons_from

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

LIB = r"""
#include <adc/runtime/dynamic_model.hpp>
#include <adc/core/variables.hpp>
%s
extern "C" adc::IModel<4>* adc_make_model() { return new adc::ModelAdapter<adc_generated::EulerGen>(); }
extern "C" void adc_destroy_model(adc::IModel<4>* p) { delete p; }
"""

MAIN = r"""
#include <adc/model/euler.hpp>
#include <adc/runtime/dynamic_model.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: %s lib.so\n", argv[0]); return 4; }
  void* h = dlopen(argv[1], RTLD_NOW);
  if (!h) { std::printf("dlopen: %s\n", dlerror()); return 2; }
  auto mk  = reinterpret_cast<adc::IModel<4>* (*)()>(dlsym(h, "adc_make_model"));
  auto del = reinterpret_cast<void (*)(adc::IModel<4>*)>(dlsym(h, "adc_destroy_model"));
  if (!mk || !del) { std::printf("dlsym fail\n"); return 3; }

  adc::IModel<4>* model = mk();    // type concret (EulerGen) INCONNU dans ce TU
  adc::Euler ref; ref.gamma = 1.4; adc::Aux a{};
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},{0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0}};
  double md = 0.0;
  for (const auto& s : S) {
    adc::StateVec<4> u{}; for (int i=0;i<4;++i) u[i]=s[i];
    for (int dir=0; dir<2; ++dir) {
      auto fr = ref.flux(u,a,dir); auto fg = model->flux(u,a,dir);   // dispatch a l'execution
      for (int i=0;i<4;++i) md = std::fmax(md, std::fabs(fr[i]-fg[i]));
      md = std::fmax(md, std::fabs(ref.max_wave_speed(u,a,dir) - model->max_wave_speed(u,a,dir)));
    }
  }
  del(model); dlclose(h);
  std::printf("%.17g\n", md);
  return md < 1e-12 ? 0 : 1;
}
"""


def main():
    brick = build_euler_brick().emit_cpp_brick(name="EulerGen")
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> dispatch a l'execution saute")
        print("test_dsl_dynamic : OK (rien a compiler)")
        return

    ldl = ["-ldl"] if platform.system() == "Linux" else []
    with tempfile.TemporaryDirectory() as tmp:
        lib_cpp = os.path.join(tmp, "libmodel.cpp")
        main_cpp = os.path.join(tmp, "main.cpp")
        so = os.path.join(tmp, "libmodel.so")
        exe = os.path.join(tmp, "main")
        with open(lib_cpp, "w") as f:
            f.write(LIB % brick)
        with open(main_cpp, "w") as f:
            f.write(MAIN)
        subprocess.run([cxx, "-shared", "-fPIC", "-std=c++20", "-O2", "-I", INCLUDE,
                        lib_cpp, "-o", so], check=True)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, main_cpp, "-o", exe] + ldl,
                       check=True)
        out = subprocess.run([exe, so], capture_output=True, text=True, check=True).stdout

    md = float(out.strip())
    assert md < 1e-12, "modele charge a l'execution != adc::Euler (ecart max %.2e)" % md
    print("OK  brique generee chargee a l'execution (dlopen -> IModel<4>) == adc::Euler (ecart %.1e)"
          % md)
    print("test_dsl_dynamic : tout est vert")


if __name__ == "__main__":
    main()
