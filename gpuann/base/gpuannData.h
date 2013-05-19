
#ifndef GPUANNDATA_H
#define GPUANNDATA_H

#include <fann.h>

struct gpuann
{
  const fann * _fann;
  unsigned int _instanceCount; //current active instance count
  unsigned int _maxInstanceCount; //allocated instance count

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
  unsigned int _trainingAlgorithm;
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

enum gpuann_fann_train_enum
{
  GPUANN_FANN_TRAIN_INCREMENTAL = 0,
  GPUANN_FANN_TRAIN_BATCH,
  GPUANN_FANN_TRAIN_RPROP,
  GPUANN_FANN_TRAIN_QUICKPROP,
  GPUANN_FANN_TRAIN_SARPROP,
  GPUANN_FANN_TRAIN_BATCH_PARRALEL,
  GPUANN_FANN_TRAIN_RPROP_PARRALEL,
  GPUANN_FANN_TRAIN_QUICKPROP_PARRALEL,
  GPUANN_FANN_TRAIN_SARPROP_PARRALEL
};


#endif // GPUANNDATA_H
