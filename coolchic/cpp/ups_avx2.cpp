
#include "common.h"
#include "frame-memory.h"
#include "ups_cpu.h"
#include <immintrin.h>

#define KS 7
#define UPSNAME ups_refine_ks7_avx2
#include "ups_refine_avx2.hpp"

#define UPSNAME ups_refine_ksX_avx2
#include "ups_refine_avx2.hpp"

#define KS 8
#define UPS_SRC_PRECISION ARM_PRECISION
#define UPSNAME ups_upsample_ks8_ARMPREC_avx2
#include "ups_upsample_avx2.hpp"

#define KS 8
#define UPS_SRC_PRECISION UPS_PRECISION
#define UPSNAME ups_upsample_ks8_UPSPREC_avx2
#include "ups_upsample_avx2.hpp"

#define UPSNAME ups_upsample_ksX_avx2
#include "ups_upsample_avx2.hpp"
