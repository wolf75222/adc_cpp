#pragma once
// Forwarding header (ADC-326): `system_coupler.hpp` now lives in `coupling/static_system/`.
// This stub keeps the historical include path `<adc/coupling/system_coupler.hpp>` compiling
// during the migration to the family layout. Prefer the canonical path for new code.
#include <adc/coupling/static_system/system_coupler.hpp>
