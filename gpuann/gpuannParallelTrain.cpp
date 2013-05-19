#include <gpuannParallelTrain.h>
#include <configuration.h>
#include <base/gpuannDataCreator.h>

#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <kernels/computeMSE/run.h>
#include <kernels/mergeSlopes/mergeSlopes.h>
#include <kernels/straightforward/straightforward.h>
#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <kernels/updateWeights/updateWeights.h>
#include <kernels/updateWeightsBatch/updateWeightsBatch.h>
#include <kernels/updateWeightsIrpropm/updateWeightsIrpropm.h>
#include <kernels/updateWeightsQuickprop/updateWeigthsQuickprop.h>
#include <kernels/updateWeightsSarprop/updateWeightsSarprop.h>

void update_weights(gpuann &data, unsigned int trainDataCount)
{
  switch (data._trainingAlgorithm)
  {
    case GPUANN_FANN_TRAIN_QUICKPROP:
    {
      gpuann_fann_update_weights_quickprop_implementation(data, trainDataCount, 0, data._weightsCountPerInstance);
      break;
    }
    case GPUANN_FANN_TRAIN_RPROP:
    {
      gpuann_fann_update_weights_irpropm_implementation(data, 0, data._weightsCountPerInstance);
      break;
    }
    case GPUANN_FANN_TRAIN_BATCH:
    {
      gpuann_fann_update_weights_batch_implementation(data, trainDataCount, 0, data._weightsCountPerInstance);
      break;
    }
    case GPUANN_FANN_TRAIN_SARPROP:
    {
      gpuann_fann_update_weights_sarprop_implementation(data, data._sarpropEpoch, 0, data._weightsCountPerInstance);

      ++(data._sarpropEpoch);
      break;
    }
  }
}

void gpuann_fann_parallel_train_epoch(gpuann &multidata, gpuann &data, gpuannTrainData &trainData)
{
  const fann *ann = multidata._fann;
  unsigned int trainDataCount = trainData._dataCount;
  unsigned int multidataSize = multidata._maxInstanceCount;

  copygpuannWeightsToMultidata(multidata, data);

  for(unsigned int trainDataIt = 0; trainDataIt < trainDataCount; trainDataIt += multidataSize)
  {
    unsigned int dataCount = multidataSize;
    if(dataCount > (trainDataCount - trainDataIt))
      dataCount = trainDataCount - trainDataIt;

    copygpuannInputsToMultidata(multidata, trainData.d_input, trainDataIt, dataCount);
    multidata._instanceCount = dataCount;

    gpuann_fann_run_implementation(multidata);
    gpuann_fann_compute_MSE_implementation_gpu(multidata, trainData.d_output + trainData._outputCount * trainDataIt);
    gpuann_fann_backpropagate_MSE_implementation_gpu(multidata);
    gpuann_fann_update_slopes_batch_implementation(multidata, ann->first_layer + 1, ann->last_layer - 1);
  }

  multidata._instanceCount = multidata._maxInstanceCount;
  gpuann_merge_slopes_implementation(multidata);
  copygpuannSlopes(data, multidata);
  update_weights(data, trainDataCount);
}

void gpuann_fann_parallel_train_on_data(gpuann &data, gpuannTrainData &trainData, unsigned int maxEpochs)
{
  unsigned int trainDataCount = trainData._dataCount;
  unsigned int instanceCount = trainDataCount;
  const fann *ann = data._fann;

  gpuann multidata;
  if(trainDataCount > parallelTrainInstanceCountMax)
    instanceCount = parallelTrainInstanceCountMax;
  creategpuann(multidata, ann, instanceCount);

  copygpuannValuesToMultidata(multidata, data); // biases values should be 1. TODO, optimize it.

  for(unsigned int i = 0; i < maxEpochs; ++i)
  {
    gpuann_fann_parallel_train_epoch(multidata, data, trainData);
  }

  removegpuann(multidata);
}
