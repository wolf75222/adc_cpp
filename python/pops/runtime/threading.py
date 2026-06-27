"""Parallelism : a single runtime knob (Spec-4 PR-F).

The compute backend is COMPILED into _pops. Multi-threading (and the GPU) are possible ONLY if
_pops was built with -DPOPS_USE_KOKKOS=ON (OpenMP device). At runtime, Kokkos initializes
LAZILY at the creation of the 1st System/AmrSystem and reads OMP_NUM_THREADS at that exact moment.
pops.set_threads(n) writes OMP_NUM_THREADS BEFORE this init : a single call replaces the ritual
`OMP_NUM_THREADS=n python ...`. To be called right after `import pops`, before creating the 1st system.

``_first_system_built`` is the shared mutable flag : read here (set_threads / parallel_info) and by
``doctor``, and WRITTEN by ``System.__init__`` / ``AmrSystem.__init__`` via
``threading._first_system_built = True`` (a module attribute, not a cross-file ``global`` rebind).
All readers/writers live in ``pops.runtime``, so the flag never leaks across layers.
"""

_first_system_built = False


def has_kokkos():
    """True if _pops was compiled with Kokkos (multi-thread/GPU possible), False if SERIAL.

    None if the module is too old to expose the info (attribute __has_kokkos__ absent)."""
    from pops import _pops
    return getattr(_pops, "__has_kokkos__", None)


def set_threads(n=None):
    """Set the number of compute threads (Kokkos OpenMP backend) in ONE line.

    Equivalent to exporting OMP_NUM_THREADS=n before launching Python, but without touching the shell. Has
    an effect only if _pops was compiled with -DPOPS_USE_KOKKOS=ON (preset 'python-parallel'), and MUST
    be called BEFORE the 1st System/AmrSystem (Kokkos initializes lazily at that moment and
    reads OMP_NUM_THREADS only once) :

        import pops
        pops.set_threads(8)     # 8 threads
        pops.set_threads()      # all cores (os.cpu_count())
        sim = pops.System(n=256)

    A SERIAL module or a late call are flagged by a warning (without raising an exception)."""
    import os
    import warnings
    if n is None:                       # default : all available logical cores
        n = os.cpu_count() or 1
    n = int(n)
    if n < 1:
        raise ValueError("pops.set_threads : n must be >= 1")
    # Source of truth : the REAL state of the Kokkos runtime (covers ALL lazy init paths --
    # System, AmrSystem, DSL .so, direct use of _pops). The Python flag stays the fallback for
    # an old module without the binding.
    from pops import _pops
    _kokkos_started = getattr(_pops, "kokkos_is_initialized", lambda: _first_system_built)()
    if _kokkos_started or _first_system_built:
        warnings.warn(
            "pops.set_threads : called AFTER the runtime initialization (1st System/AmrSystem or "
            "1st allocation) -> NO EFFECT. Call set_threads right after `import pops`.",
            RuntimeWarning, stacklevel=2)
        return
    if has_kokkos() is False:
        warnings.warn(
            "pops.set_threads : _pops is SERIAL (compiled without -DPOPS_USE_KOKKOS=ON) -> the thread "
            "setting is ignored at compute time. Rebuild with -DPOPS_USE_KOKKOS=ON "
            "-DKokkos_ROOT=$CONDA_PREFIX for multi-threading.", RuntimeWarning, stacklevel=2)
    # We write the env even in case of doubt (harmless) : a DSL .so with backend='production' compiled with
    # Kokkos will also read OMP_NUM_THREADS at its initialization.
    # We set TWO variables to be agnostic to the backend that Kokkos was compiled with :
    #   - OMP_NUM_THREADS  : read by the OpenMP device (usual case) ;
    #   - KOKKOS_NUM_THREADS : read by Kokkos::initialize whatever the device (OpenMP OR Threads),
    #     useful if the installed Kokkos (e.g. conda-forge) uses the Threads backend and not OpenMP.
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["KOKKOS_NUM_THREADS"] = str(n)
    # OMP_PROC_BIND=false ONLY on macOS (avoids libomp warnings/oversubscription on
    # dev Macs). On Linux/cluster we impose NOTHING : disabling affinity there would degrade
    # NUMA scaling, and a SLURM job that exports OMP_PROC_BIND=close/spread stays in control (setdefault
    # would not override it anyway).
    import sys as _s
    if _s.platform == "darwin":
        os.environ.setdefault("OMP_PROC_BIND", "false")


def parallel_info():
    """Parallelism state : compiled backend, current OMP_NUM_THREADS, Kokkos init already done."""
    import os
    return {
        "has_kokkos": has_kokkos(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "first_system_built": _first_system_built,
    }
