"""Generic moment-model builders, closures and sources (Spec 4 re-organization).

Public API re-exported VERBATIM from `adc/moments.py`:
  - moment_indices, moment_names: the canonical (p, q) ordering and 'M{p}{q}' names;
  - gaussian_closure: the generic Gaussian (Levermore) closure;
  - lorentz_sources, maxwellian_moments, bgk_source: generic Vlasov-Lorentz and BGK sources;
  - build_moment_model: the generic 2D moment-model generator.
"""
from adc.lib.moments.closures import gaussian_closure
from adc.lib.moments.model_builder import (
    build_moment_model,
    moment_indices,
    moment_names,
)
from adc.lib.moments.sources import (
    bgk_source,
    lorentz_sources,
    maxwellian_moments,
)

__all__ = [
    "moment_indices",
    "moment_names",
    "gaussian_closure",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
    "build_moment_model",
]
