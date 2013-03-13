#ifndef GPUANN_H
#define GPUANN_H

#include <fann.h>

fann_type * gpuann_fann_run(struct fann * ann, fann_type * input);

#endif // GPUANN_H
