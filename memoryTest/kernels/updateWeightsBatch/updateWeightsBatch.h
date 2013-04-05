#ifndef KERNELS_UPDATEWEIGHTSBATCH_UPDATEWEIGHTSBATCH_H
#define KERNELS_UPDATEWEIGHTSBATCH_UPDATEWEIGHTSBATCH_H

#include <base/gpuannData.h>

void gpuann_fann_update_weights_batch_implementation(gpuann &data, unsigned int num_data, unsigned int first_weight, unsigned int past_end);

#endif // KERNELS_UPDATEWEIGHTSBATCH_UPDATEWEIGHTSBATCH_H
