"""pops.lib.time -- ready-made time-stepping scheme macros (Spec 4 adc.lib.time).

These functions build a pops.time.Program step.  They are extracted from
pops.time and placed here per Spec 4: scheme macros belong in adc.lib.time,
not in the time-language module (adc.time).

Public API
----------
Euler:
    forward_euler

SSPRK:
    ssprk2, ssprk3

Classic Runge-Kutta:
    rk4, rk, explicit_rk, ButcherTableau, RK4_TABLEAU, SSPRK2_TABLEAU

Splitting:
    strang, lie, condensed_schur

IMEX:
    imex_local, imex_local_linear

Multi-step:
    adams_bashforth, adams_bashforth2, bdf

Predictor-corrector:
    predictor_corrector_local_linear

The schemes are exposed by their explicit names (Spec 4 s7: the name ``std`` is banned -- no
catch-all bundle). Call e.g. ``pops.lib.time.ssprk3`` / ``pops.lib.time.strang`` directly.
"""

from .euler import forward_euler
from .ssprk import ssprk2, ssprk3
from .rk import rk4, rk, explicit_rk, ButcherTableau, RK4_TABLEAU, SSPRK2_TABLEAU
from .strang import strang, lie, condensed_schur
from .imex import imex_local, imex_local_linear
from .multistep import adams_bashforth, adams_bashforth2, bdf
from .predictor_corrector import predictor_corrector_local_linear

__all__ = [
    # Euler
    "forward_euler",
    # SSPRK
    "ssprk2",
    "ssprk3",
    # Classic RK
    "rk4",
    "rk",
    "explicit_rk",
    "ButcherTableau",
    "RK4_TABLEAU",
    "SSPRK2_TABLEAU",
    # Splitting
    "strang",
    "lie",
    "condensed_schur",
    # IMEX
    "imex_local",
    "imex_local_linear",
    # Multi-step
    "adams_bashforth",
    "adams_bashforth2",
    "bdf",
    # Predictor-corrector
    "predictor_corrector_local_linear",
]
