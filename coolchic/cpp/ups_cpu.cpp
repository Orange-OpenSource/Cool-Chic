
#include "common.h"
#include "frame-memory.h"
#include "ups_cpu.h"

#define KS 7
#define UPSNAME ups_refine_ks7_cpu
#include "ups_refine_cpu.hpp"

#define UPSNAME ups_refine_ksX_cpu
#include "ups_refine_cpu.hpp"

#define KS 8
#define UPSNAME ups_upsample_ks8_cpu
#include "ups_upsample_cpu.hpp"

#define UPSNAME ups_upsample_ksX_cpu
#include "ups_upsample_cpu.hpp"
