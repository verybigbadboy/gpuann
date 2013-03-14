#ifndef GPUDATACREATOR_H
#define GPUDATACREATOR_H

#include <base/gpuannData.h>

void creategpuann(gpuann& nn, const fann *ann, unsigned int instanceCount = 1);
void removegpuann(gpuann& nn);
void loadgpuann(gpuann& nn, const fann *ann, unsigned int instanceIndex = 0);
void savegpuann(const gpuann& nn, fann *ann, unsigned int instanceIndex = 0);

#endif // GPUDATACREATOR_H
