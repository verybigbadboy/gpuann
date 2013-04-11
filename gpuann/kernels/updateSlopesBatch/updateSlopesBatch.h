#ifndef KERNELS_UPDATESLOPESBATCH_UPDATESLOPESBATCH_H
#define KERNELS_UPDATESLOPESBATCH_UPDATESLOPESBATCH_H

#include <base/gpuannData.h>

void gpuann_fann_update_slopes_batch_implementation(gpuann &ann, fann_layer *layerBegin, fann_layer *layerEnd);

#endif // KERNELS_UPDATESLOPESBATCH_UPDATESLOPESBATCH_H
