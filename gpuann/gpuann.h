#ifndef GPUANN_H
#define GPUANN_H

#include <fann.h>

fann_type * gpuann_fann_run(struct fann * ann, fann_type * input);
void gpuann_fann_multirun(struct fann * ann, fann_type ** input, unsigned int instanceCount, fann_type ** output);
void gpuann_fann_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs);
void gpuann_fann_parallel_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs);

#endif // GPUANN_H
