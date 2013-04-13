
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
  fann_type *d_trainSlopes;
  fann_type *d_prevTrainSlopes;
  fann_type *d_prevSteps;

  unsigned int _weightsCountPerInstance;
  unsigned int _neuronsCountPerInstance;
  unsigned int _sarpropEpoch;
};

struct debugGpuann: public gpuann
{
};

struct gpuannTrainData
{
  unsigned int _dataCount;
  unsigned int _inputCount;
  unsigned int _outputCount;
  fann_type * d_input;
  fann_type * d_output;
};


#endif // GPUANNDATA_H