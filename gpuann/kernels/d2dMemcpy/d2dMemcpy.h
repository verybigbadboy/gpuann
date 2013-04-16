#ifndef KERNELS_D2DMEMCPY_D2DMEMCPY_H
#define KERNELS_D2DMEMCPY_D2DMEMCPY_H

#include <base/gpuannData.h>

void gpuann_d2dMemcpy(fann_type *d_dst, fann_type *d_src,  unsigned int size);

#endif // KERNELS_D2DMEMCPY_D2DMEMCPY_H