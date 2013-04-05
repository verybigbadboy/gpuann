#ifndef GPUANN_H
#define GPUANN_H

#include <fann.h>

fann_type * gpuann_fann_run(struct fann * ann, fann_type * input);
void gpuann_fann_multirun(struct fann * ann, fann_type ** input, unsigned int instanceCount, fann_type ** output);

void gpuann_fann_train(struct fann *ann, fann_type * input, fann_type * desired_output);

void test(fann *ann, fann_train_data* train);

#endif // GPUANN_H
