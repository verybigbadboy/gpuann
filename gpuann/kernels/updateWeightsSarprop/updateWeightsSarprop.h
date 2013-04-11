#ifndef KERNELS_UPDATEWEIGHTSSARPROP_UPDATEWEIGHTSSARPROP_H
#define KERNELS_UPDATEWEIGHTSSARPROP_UPDATEWEIGHTSSARPROP_H

#include <base/gpuannData.h>

void gpuann_fann_update_weights_sarprop_implementation(gpuann &data, unsigned int epoch, unsigned int first_weight, unsigned int past_end);

#endif // KERNELS_UPDATEWEIGHTSSARPROP_UPDATEWEIGHTSSARPROP_H