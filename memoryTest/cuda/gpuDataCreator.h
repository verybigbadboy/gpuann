#ifndef GPUDATACREATOR_H
#define GPUDATACREATOR_H

#include <fann.h>

struct gpuData
{
  fann_type *weightsArray;
  fann_type *valuesArray;
  fann_type *sumArray;
  fann_type *d_weightsArray;
  fann_type *d_valuesArray;
  fann_type *d_sumArray;
};

void prepareData(struct fann * ann, fann_type * input, gpuData &data);

void unPrepareAndFreeData(gpuData &data, fann *ann);

#endif // GPUDATACREATOR_H
