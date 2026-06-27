"""Spec 4: the build-time codegen path imports without numpy.

The C++ build runs ``scripts/gen_solver_kernel.py`` under a bare interpreter (the job's
system python, no numpy) to emit the custom-solver kernel header. That tool imports
``pops.lib`` -> ``pops.ir``, so the symbolic IR and lib layers must import without numpy
installed: numpy backs only the host ``.eval()`` interpreter, never IR construction or
C++ emission.

This guards a regression that reddened the C++ jobs. The python PRs route to gate-python
and SKIP the C++ build, so a numpy import at module scope in ``pops.ir`` only surfaced on
the master push (the gen-solver build step). We re-run the real codegen tool in a
subprocess with numpy forced absent (``sys.modules['numpy'] = None``) and assert it still
emits the kernel -- so the regression is caught in gate-python, before the master push.

Source-only: needs neither _pops nor numpy to run.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEN_SOLVER = REPO_ROOT / "scripts" / "gen_solver_kernel.py"


def test_gen_solver_kernel_runs_without_numpy():
    assert GEN_SOLVER.exists(), "missing %s" % GEN_SOLVER
    out = os.path.join(tempfile.mkdtemp(), "pops_generated_solver.hpp")
    # Force `import numpy` to fail (sys.modules[name] = None raises ModuleNotFoundError), then run the
    # real build-time codegen exactly as CMake does. A numpy import anywhere in the pops.lib/pops.ir
    # import chain would abort it -- which is the regression this test guards.
    code = (
        "import sys; sys.modules['numpy'] = None; "
        "import runpy; sys.argv = ['gen_solver_kernel.py', %r]; "
        "runpy.run_path(%r, run_name='__main__')" % (out, str(GEN_SOLVER))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "gen_solver_kernel.py failed without numpy (pops.lib / pops.ir must import numpy-free; "
        "numpy is for the host .eval() interpreter only):\n%s" % proc.stderr
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0, "no kernel header emitted"
    assert "richardson_gen_solve" in pathlib.Path(out).read_text(), \
        "emitted header missing the generated kernel"
