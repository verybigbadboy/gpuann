#ifndef GPUDATACREATOR_H
#define GPUDATACREATOR_H

#include <base/gpuannData.h>

void creategpuann(gpuann& nn, const fann *ann, unsigned int instanceCount = 1);
void removegpuann(gpuann& nn);
void copygpuann(gpuann& to, gpuann& from, unsigned int fromInstance = 0, unsigned int toInstance = 0, unsigned int instanceCount = 1);

void copygpuannValues(gpuann& to, gpuann& from, unsigned int fromInstance = 0, unsigned int toInstance = 0, unsigned int instanceCount = 1);
void copygpuannWeights(gpuann& to, gpuann& from, unsigned int fromInstance = 0, unsigned int toInstance = 0, unsigned int instanceCount = 1);
void copygpuannSlopes(gpuann& to, gpuann& from, unsigned int fromInstance = 0, unsigned int toInstance = 0, unsigned int instanceCount = 1);

void loadgpuann(gpuann& nn, const fann *ann, unsigned int instanceIndex = 0);
void savegpuann(const gpuann& nn, fann *ann, unsigned int instanceIndex = 0);

void gpuann_loadInputs(gpuann& nn, fann_type *d_inputs, unsigned int instanceIndex = 0);
fann_type* gpuann_getOutputsDevicePointer(gpuann& nn, unsigned int instanceIndex = 0);

void createDump(gpuann &nn, debugGpuann &dnn);
void removeDump(debugGpuann &dnn);

void creategpuannTrainData(gpuannTrainData &trainData, fann_train_data *train);
void removegpuannTrainData(gpuannTrainData &trainData);

#endif // GPUDATACREATOR_H
