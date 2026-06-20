#pragma once
// Forwarding header (ADC-326): `coupler.hpp` now lives in `coupling/single/`.
// This stub keeps the historical include path `<adc/coupling/coupler.hpp>` compiling
// during the migration to the family layout. Prefer the canonical path for new code.
#include <adc/coupling/single/coupler.hpp>
