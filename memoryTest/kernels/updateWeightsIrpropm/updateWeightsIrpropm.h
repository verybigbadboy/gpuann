#ifndef KERNELS_UPDATEWEIGHTSIRPROPM_UPDATEWEIGHTSIRPROPM_H
#define KERNELS_UPDATEWEIGHTSIRPROPM_UPDATEWEIGHTSIRPROPM_H

#include <base/gpuannData.h>

void gpuann_fann_update_weights_irpropm_implementation(gpuann &data, unsigned int first_weight, unsigned int past_end);

#endif // KERNELS_UPDATEWEIGHTSIRPROPM_UPDATEWEIGHTSIRPROPM_H