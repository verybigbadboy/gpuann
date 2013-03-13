#ifndef GPUDATACREATOR_H
#define GPUDATACREATOR_H

#include <base/gpuannData.h>

void loadgpuann(gpuann& nn, const fann *ann);
void savegpuann(const gpuann& nn, fann *ann);

#endif // GPUDATACREATOR_H
