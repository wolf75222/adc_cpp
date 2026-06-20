#pragma once
// Forwarding header (ADC-332): `amr_level.hpp` is the single-box MultiFab AMR test oracle and now
// lives in `numerics/time/reference/`. This stub keeps the historical include path
// `<adc/numerics/time/amr_level.hpp>` compiling (the live `amr_reflux_mf.hpp` aggregator still
// pulls it through this path). Prefer the canonical reference path for new code.
#include <adc/numerics/time/reference/amr_level.hpp>
