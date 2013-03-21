
#ifndef GPUANNDATA_H
#define GPUANNDATA_H

#include <fann.h>

struct gpuann
{
  const fann * _fann;
  unsigned int _instanceCount;

  //only for 1 instance
  fann_type *h_tmp_valuesArray;
  fann_type *h_tmp_sumArray;

  fann_type *d_weightsArray;
  fann_type *d_valuesArray;
  fann_type *d_sumArray;
  fann_type *d_trainErrorsArray;
  fann_type *d_prevWeightsDeltas;

  unsigned int _weightsCountPerInstance;
  unsigned int _neuronsCountPerInstance;
};

struct debugGpuann: public gpuann
{
};


#endif // GPUANNDATA_H
