#ifndef KERNELS_STRAIGHTFORWARD_RUN_H
#define KERNELS_STRAIGHTFORWARD_RUN_H

#include <base/gpuannData.h>

void gpuann_fann_run_implementation(gpuann &data);

fann_type * gpuann_fann_run_device(gpuann &data, fann_type * d_input);

#endif // KERNELS_STRAIGHTFORWARD_RUN_H
