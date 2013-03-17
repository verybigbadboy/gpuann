#ifndef KERNELS_COMPUTEMSE_RUN_H
#define KERNELS_COMPUTEMSE_RUN_H

#include <base/gpuannData.h>

void gpuann_fann_compute_MSE_implementation_gpu(gpuann &data, fann_type * d_desired_output);

#endif // KERNELS_COMPUTEMSE_RUN_H
