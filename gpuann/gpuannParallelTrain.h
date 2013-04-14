#ifndef GPUANN_PARALEL_TRAIN_H
#define GPUANN_PARALEL_TRAIN_H

#include <base/gpuannData.h>

void gpuann_fann_parallel_train_on_data(gpuann &data, gpuannTrainData &trainData, unsigned int maxEpochs);
void gpuann_fann_parallel_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs);

#endif // GPUANN_PARALEL_TRAIN_H