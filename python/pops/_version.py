"""Package version (Spec-4 PR-F).

Single source of the version number: the value baked into the extension by ``project(VERSION)``
in CMake. An old module without the attribute degrades to "unknown" rather than breaking the
import. ``pops.__version__`` re-exports this.
"""

try:
    from ._pops import __version__
except ImportError:
    __version__ = "unknown"
