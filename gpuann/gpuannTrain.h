#ifndef GPUANN_TRAIN_H
#define GPUANN_TRAIN_H

#include <base/gpuannData.h>

void gpuann_fann_train_epoch_quickprop(gpuann &data, gpuannTrainData &trainData);
void gpuann_fann_train_epoch_irpropm(gpuann &data, gpuannTrainData &trainData);
void gpuann_fann_train_epoch_sarprop(gpuann &data, gpuannTrainData &trainData);
void gpuann_fann_train_epoch_batch(gpuann &data, gpuannTrainData &trainData);
void gpuann_fann_train_epoch_incremental(gpuann &data, gpuannTrainData &trainData);
void gpuann_fann_train_epoch(gpuann &data, gpuannTrainData &trainData);

#endif // GPUANN_TRAIN_H
