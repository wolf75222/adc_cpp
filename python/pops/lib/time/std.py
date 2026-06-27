"""pops.lib.time.std -- the standard library of ready time-stepping schemes.

``std`` is a ``SimpleNamespace`` bundling every ready scheme builder under stable
attribute names (Spec 4 s6 / s14: the ready schemes and their ``std`` bundle live in
``pops.lib.time``, not in the time-language module ``pops.time``). It mirrors the
attribute surface the historical ``pops.time.std`` exposed, so a caller migrates by
swapping ``pops.time.std.X`` for ``pops.lib.time.std.X`` with no other change.

Each entry BUILDS ``pops.time.Program`` IR via the shared builder ops; there is no
scheme-specific C++ stepper. The generated ``problem.so`` executes the lowered IR.
"""
import types

from .euler import forward_euler
from .imex import imex_local, imex_local_linear
from .multistep import adams_bashforth, adams_bashforth2, bdf
from .predictor_corrector import predictor_corrector_local_linear
from .rk import RK4_TABLEAU, SSPRK2_TABLEAU, ButcherTableau, explicit_rk, rk, rk4
from .ssprk import ssprk2, ssprk3
from .strang import condensed_schur, lie, strang

std = types.SimpleNamespace(forward_euler=forward_euler, ssprk2=ssprk2, ssprk3=ssprk3, rk4=rk4,
                            rk=rk, RK4_TABLEAU=RK4_TABLEAU, SSPRK2_TABLEAU=SSPRK2_TABLEAU,
                            ButcherTableau=ButcherTableau,
                            adams_bashforth=adams_bashforth, adams_bashforth2=adams_bashforth2,
                            strang=strang, lie=lie, imex_local=imex_local, bdf=bdf,
                            condensed_schur=condensed_schur,
                            predictor_corrector_local_linear=predictor_corrector_local_linear,
                            explicit_rk=explicit_rk, imex_local_linear=imex_local_linear)
