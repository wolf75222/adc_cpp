"""PythonFlux : host (numpy) prototyping backend for the Flux interface (Spec-4 PR-F)."""


class PythonFlux:
    """PROTOTYPING backend (host, numpy) for the Flux interface: the user provides the physical
    flux and the wave speed in Python, and PythonFlux assembles the residual -div(F*) by finite
    volumes (Rusanov, order 1, periodic domain) over the whole array at once.

    OUT of the GPU/MPI hot path: this is a pure HOST path (numpy), it NEVER goes through a Kokkos
    kernel. For production (GPU/MPI), compose a COMPILED flux (pops.CompressibleFlux brick,
    pops.ExB...). PythonFlux formalizes the pattern of the custom_scheme case: iterate quickly on a
    novel flux without recompiling (pops.System serving as Poisson oracle if needed).

    flux(U, dir) -> F: U and F are numpy (ncomp, n, n); dir = 0 (x) or 1 (y).
    max_wave_speed(U) -> float: bound for the Rusanov flux and the CFL.
    """

    def __init__(self, flux, max_wave_speed):
        self.flux = flux
        self.max_wave_speed = max_wave_speed

    def residual(self, U, dx, dy=None):
        """-div(F*) by Rusanov flux (order 1, periodic). U numpy (ncomp, n, n); returns dU/dt."""
        import numpy as np
        dy = dx if dy is None else dy
        a = float(self.max_wave_speed(U))
        res = np.zeros_like(U)
        for axis, h, d in ((2, dx, 0), (1, dy, 1)):  # x = axis 2, y = axis 1
            F = self.flux(U, d)
            UR = np.roll(U, -1, axis=axis)
            FR = np.roll(F, -1, axis=axis)
            face = 0.5 * (F + FR) - 0.5 * a * (UR - U)       # flux at the +d face of each cell
            res -= (face - np.roll(face, 1, axis=axis)) / h  # -div: (F_{i+1/2} - F_{i-1/2}) / h
        return res

    def cfl_dt(self, U, h, cfl=0.4):
        """Stable time step: dt = cfl * h / max_wave_speed(U)."""
        return cfl * h / max(float(self.max_wave_speed(U)), 1e-30)
