#ifndef KERNELS_UPDATEWEIGHTSQUICKPROP_UPDATEWEIGHTSQUICKPROP_H
#define KERNELS_UPDATEWEIGHTSQUICKPROP_UPDATEWEIGHTSQUICKPROP_H

#include <base/gpuannData.h>

void gpuann_fann_update_weights_quickprop_implementation(gpuann &data, unsigned int num_data, unsigned int first_weight, unsigned int past_end);

#endif // KERNELS_UPDATEWEIGHTSQUICKPROP_UPDATEWEIGHTSQUICKPROP_H

