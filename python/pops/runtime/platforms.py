"""pops.runtime.platforms -- typed execution-platform descriptors (Spec 5 sec.8.15 / criterion 22).

Spec 5 stabilises "every object that chooses a route is a typed descriptor that declares its
requirements / capabilities / options and answers ``available(context)`` with an EXPLAINABLE
status". The execution PLATFORM (the Kokkos device or MPI) is one such route. This module adds the
inert platform descriptors :class:`KokkosSerial` / :class:`KokkosOpenMP` / :class:`KokkosCuda` /
:class:`KokkosHIP` / :class:`MPI` so a backend can be told ``Production(platform=KokkosOpenMP())``
with an object, not a string.

Each descriptor is INERT: it declares its host / gpu / mpi capabilities and answers
:meth:`available` against the COMPILED ``_pops`` build flags (``__has_kokkos__`` / ``__has_mpi__``),
EXPLAINING the missing build flag when the device is not present (e.g. a Serial-only build cannot
offer the OpenMP device). It computes nothing: no Kokkos device is initialised here, no MPI
communicator is created -- the runtime selects and drives the device. Like
:mod:`pops.runtime.threading`, the only ``_pops`` touch is lazy inside :meth:`available`, so
``import pops.runtime.platforms`` is side-effect-free.
"""
from pops.descriptors import Availability, Descriptor


def _has_kokkos():
    """True/False/None: whether _pops was compiled with Kokkos (None if too old to expose it)."""
    # Lazy, mirroring pops.runtime.threading.has_kokkos: importing _pops at module scope would
    # make this inert authoring module pull the C extension at import time.
    from pops import _pops
    return getattr(_pops, "__has_kokkos__", None)


def _has_mpi():
    """True/False/None: whether _pops was compiled with MPI (None if too old to expose it)."""
    from pops import _pops
    return getattr(_pops, "__has_mpi__", None)


class _Platform(Descriptor):
    """Base of the typed execution-platform descriptors (category ``"platform"``).

    A platform descriptor names a Kokkos device or the MPI transport and declares its host / gpu /
    mpi capabilities. It is inert -- it initialises no device; :meth:`available` only READS the
    compiled ``_pops`` build flags and explains a missing one. Subclasses set
    :attr:`_device` (a short token) and the three capability booleans.
    """

    category = "platform"
    #: A short device token reported in options / lowering.
    _device = None
    #: Honest capability booleans (overridden per device).
    _host = False
    _gpu = False
    _mpi = False

    def options(self):
        return {"device": self._device}

    def capabilities(self):
        return {"host": self._host, "gpu": self._gpu, "mpi": self._mpi}

    def requirements(self):
        """What the platform NEEDS from the compiled module (the build flag it depends on)."""
        req = {}
        if self._device != "serial":
            req["kokkos"] = True
        return req

    def lower(self, context=None):
        """The inert lowering record: the device token + the declared capabilities (metadata only)."""
        return {"name": self.name, "category": self.category, "device": self._device,
                "capabilities": self.capabilities()}


class KokkosSerial(_Platform):
    """The Kokkos Serial host device (single-thread CPU). Always available with a Kokkos build.

    The Serial device needs only a Kokkos-enabled ``_pops``; it is the fallback host device when
    no OpenMP / GPU backend was compiled in.
    """

    _device = "serial"
    _host = True

    def requirements(self):
        # Serial is the baseline host device: a Kokkos build offers it unconditionally; a strictly
        # SERIAL (non-Kokkos) _pops still runs single-threaded host code, so do not hard-require the
        # Kokkos flag here -- available() reports the honest status.
        return {}

    def available(self, context=None):
        """Available on any build (host single-thread is always present)."""
        return Availability.yes("host serial device is always available")


class KokkosOpenMP(_Platform):
    """The Kokkos OpenMP host device (multi-thread CPU). Needs a Kokkos-enabled ``_pops``.

    ``Production(platform=KokkosOpenMP())``. Multi-threading is possible only when ``_pops`` was
    built with ``-DPOPS_USE_KOKKOS=ON``; otherwise :meth:`available` explains the missing flag.
    """

    _device = "openmp"
    _host = True

    def available(self, context=None):
        kokkos = _has_kokkos()
        if kokkos is None:
            return Availability.partial(
                "the loaded _pops is too old to report its Kokkos build flag (__has_kokkos__ "
                "absent); cannot confirm the OpenMP device", missing=["__has_kokkos__"])
        if not kokkos:
            return Availability.no(
                "the OpenMP device needs a Kokkos build: _pops is SERIAL (compiled without "
                "-DPOPS_USE_KOKKOS=ON); rebuild with -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=$CONDA_PREFIX",
                missing=["-DPOPS_USE_KOKKOS=ON"], alternatives=["KokkosSerial()"])
        return Availability.yes("Kokkos build present (OpenMP device selectable at runtime)")


class KokkosCuda(_Platform):
    """The Kokkos CUDA device (NVIDIA GPU). Needs a Kokkos+CUDA ``_pops`` build.

    Inert. The GPU path is built + validated on the cluster (e.g. ROMEO GH200); a CPU/macOS build
    has no CUDA backend, so :meth:`available` explains the missing build rather than overclaim a
    device that is not present.
    """

    _device = "cuda"
    _gpu = True

    def available(self, context=None):
        kokkos = _has_kokkos()
        if kokkos is None:
            return Availability.partial(
                "the loaded _pops is too old to report its Kokkos build flag (__has_kokkos__ "
                "absent); cannot confirm the CUDA device", missing=["__has_kokkos__"])
        if not kokkos:
            return Availability.no(
                "the CUDA device needs a Kokkos build with the CUDA backend: _pops is SERIAL "
                "(no -DPOPS_USE_KOKKOS=ON); build _pops with Kokkos CUDA on a GPU host",
                missing=["-DPOPS_USE_KOKKOS=ON", "Kokkos_ENABLE_CUDA"],
                alternatives=["KokkosOpenMP()", "KokkosSerial()"])
        # A Kokkos build is present but _pops exposes no CUDA-specific flag: do not overclaim a GPU
        # the loaded module may not have (a CPU-only Kokkos build). Report partial + the missing
        # introspection rather than a false "yes".
        return Availability.partial(
            "_pops reports a Kokkos build but exposes no CUDA-device flag; the CUDA backend is "
            "confirmable only on a Kokkos+CUDA build (e.g. a GH200 host)",
            missing=["Kokkos_ENABLE_CUDA introspection"],
            alternatives=["KokkosOpenMP()", "KokkosSerial()"])


class KokkosHIP(_Platform):
    """The Kokkos HIP device (AMD GPU). Needs a Kokkos+HIP ``_pops`` build.

    Inert. The AMD GPU counterpart of :class:`KokkosCuda`; :meth:`available` explains the missing
    Kokkos / HIP build rather than overclaim a device.
    """

    _device = "hip"
    _gpu = True

    def available(self, context=None):
        kokkos = _has_kokkos()
        if kokkos is None:
            return Availability.partial(
                "the loaded _pops is too old to report its Kokkos build flag (__has_kokkos__ "
                "absent); cannot confirm the HIP device", missing=["__has_kokkos__"])
        if not kokkos:
            return Availability.no(
                "the HIP device needs a Kokkos build with the HIP backend: _pops is SERIAL "
                "(no -DPOPS_USE_KOKKOS=ON); build _pops with Kokkos HIP on an AMD GPU host",
                missing=["-DPOPS_USE_KOKKOS=ON", "Kokkos_ENABLE_HIP"],
                alternatives=["KokkosOpenMP()", "KokkosSerial()"])
        return Availability.partial(
            "_pops reports a Kokkos build but exposes no HIP-device flag; the HIP backend is "
            "confirmable only on a Kokkos+HIP build",
            missing=["Kokkos_ENABLE_HIP introspection"],
            alternatives=["KokkosOpenMP()", "KokkosSerial()"])


class MPI(_Platform):
    """The MPI distributed transport. Needs an MPI-enabled ``_pops`` build.

    ``Production(platform=MPI())``. Declares the ``mpi`` capability; :meth:`available` reads the
    compiled ``__has_mpi__`` flag and explains a missing MPI build. Inert: no communicator is
    created here -- the distributed runtime owns the ranks.
    """

    _device = "mpi"
    _host = True
    _mpi = True

    def requirements(self):
        return {"mpi": True}

    def available(self, context=None):
        mpi = _has_mpi()
        if mpi is None:
            return Availability.partial(
                "the loaded _pops is too old to report its MPI build flag (__has_mpi__ absent); "
                "cannot confirm the MPI transport", missing=["__has_mpi__"])
        if not mpi:
            return Availability.no(
                "the MPI transport needs an MPI build: _pops was compiled without MPI; build "
                "_pops with the 'mpi' preset (-DPOPS_USE_MPI=ON) to run distributed",
                missing=["-DPOPS_USE_MPI=ON"], alternatives=["KokkosSerial()", "KokkosOpenMP()"])
        return Availability.yes("MPI build present (distributed runtime available)")


__all__ = ["KokkosSerial", "KokkosOpenMP", "KokkosCuda", "KokkosHIP", "MPI"]
