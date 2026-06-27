"""Load the ``_pops`` C++ extension with the right dlopen flags (Spec-4 PR-F).

The "production" DSL backend loads a ``.so`` loader via dlopen ; that loader resolves C++
symbols exported by the ``_pops`` extension (System::install_block, grid_context,
ensure_aux_width, etc.). CPython normally loads extensions with RTLD_LOCAL on Unix/macOS ;
the symbols then stay invisible to the loader and add_native_block fails at dlopen ("symbol
not found in flat namespace"). So we load ``_pops`` with RTLD_GLOBAL, then restore the flags
for the following imports. The already-loaded module keeps its global scope.

This module is the SINGLE module-scope importer of ``_pops`` (alongside the runtime layer).
Importing it has the SIDE EFFECT of loading the extension and binding the C++ config / system
types (SystemConfig, ModelSpec, System, AmrSystemConfig, AmrSystem, abi_key) as attributes of
this module ; ``pops`` and ``pops.runtime`` then import those names from here.
"""

import os as _os
import sys as _sys


def _explain_missing_extension(exc):
    """Turn the raw ModuleNotFoundError on pops._pops into an ACTIONABLE message (recurring bug :
    the extension is pinned to the cpython-3XY ABI of the interpreter that built it ; under a
    different python, the import fails without saying why). We list the .so files present next to the package
    and compare their tag to the current interpreter."""
    import glob
    here = _os.path.dirname(__file__)
    sos = sorted(_os.path.basename(p) for p in glob.glob(_os.path.join(here, "_pops.*")))
    cur = "cpython-%d%d" % (_sys.version_info[0], _sys.version_info[1])
    if not sos:
        hint = ("no _pops.*.so extension in %s : the module is not built. Build with "
                "`cmake --preset python && cmake --build --preset python`, then PYTHONPATH=<build>/python."
                % here)
    elif not any(cur in s for s in sos):
        hint = ("extension(s) present : %s, but the current interpreter is %s (%s). Use the "
                "python that built the module (conda env `pops`), or rebuild with this interpreter "
                "(-DPython_EXECUTABLE=%s)." % (", ".join(sos), cur, _sys.executable, _sys.executable))
    else:
        hint = ("the extension %s matches the interpreter (%s) but its import fails : missing "
                "dependency or corrupt .so ; rerun the module build." % (", ".join(sos), cur))
    return ImportError("import pops._pops failed : %s\n(original cause : %s)" % (hint, exc))


if hasattr(_sys, "setdlopenflags") and hasattr(_sys, "getdlopenflags"):
    _pops_old_dlopenflags = _sys.getdlopenflags()
    _pops_global_dlopenflags = _pops_old_dlopenflags
    if hasattr(_os, "RTLD_NOW"):
        _pops_global_dlopenflags |= _os.RTLD_NOW
    if hasattr(_os, "RTLD_GLOBAL"):
        _pops_global_dlopenflags |= _os.RTLD_GLOBAL
    _sys.setdlopenflags(_pops_global_dlopenflags)
    try:
        from ._pops import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # module ABI key ("production" DSL path / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e
    finally:
        _sys.setdlopenflags(_pops_old_dlopenflags)
    del _pops_old_dlopenflags, _pops_global_dlopenflags
else:
    try:
        from ._pops import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # module ABI key ("production" DSL path / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e

del _os, _sys
