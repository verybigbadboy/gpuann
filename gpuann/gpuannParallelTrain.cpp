#include <gpuannParallelTrain.h>
#include <gpuannTrain.h>
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

void gpuann_fann_train_parralel_multi_update_slopes(gpuann &multidata, gpuannTrainData &trainData)
{
  const fann *ann = multidata._fann;
  
  gpuann_fann_run_implementation(multidata);
  gpuann_fann_compute_MSE_implementation_gpu(multidata, trainData.d_output);
  gpuann_fann_backpropagate_MSE_implementation_gpu(multidata);
  gpuann_fann_update_slopes_batch_implementation(multidata, ann->first_layer + 1, ann->last_layer - 1);
}

#include <cuda.h>
#include <cuda_runtime.h>

void gpuann_load_multidata(gpuann &multidata, gpuann &data, gpuannTrainData &trainData)
{
  unsigned int instanceCount = trainData._dataCount;

  for(unsigned int i = 0; i < instanceCount; i++)
  {
    copygpuannWeights(multidata, data, 0, i);
    gpuann_loadInputs(multidata, trainData.d_input + trainData._inputCount * i, i);
  }
}

void gpuann_fann_train_parralel_update_slopes(gpuann &multidata, gpuann &data, gpuannTrainData &trainData)
{
  gpuann_load_multidata(multidata, data, trainData);
  gpuann_fann_train_parralel_multi_update_slopes(multidata, trainData);
  gpuann_merge_slopes_implementation(multidata);
  copygpuannSlopes(data, multidata);
}

void update_weights(gpuann &data, gpuannTrainData &trainData)
{
  switch (data._trainingAlgorithm)
  {
    case GPUANN_FANN_TRAIN_QUICKPROP:
    {
      gpuann_fann_update_weights_quickprop_implementation(data, trainData._dataCount, 0, data._weightsCountPerInstance);
      break;
    }
    case GPUANN_FANN_TRAIN_RPROP:
    {
      gpuann_fann_update_weights_irpropm_implementation(data, 0, data._weightsCountPerInstance);
      break;
    }
    case GPUANN_FANN_TRAIN_BATCH:
    {
      gpuann_fann_update_weights_batch_implementation(data, trainData._dataCount, 0, data._weightsCountPerInstance);
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

float gpuann_fann_parallel_train_epoch(gpuann &multidata, gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);

  gpuann_fann_train_parralel_update_slopes(multidata, data, trainData);

  update_weights(data, trainData);

  return gpuann_fann_get_MSE(data);
}

void gpuann_fann_parallel_train_on_data(gpuann &data, gpuannTrainData &trainData, unsigned int maxEpochs)
{
  unsigned int instanceCount = trainData._dataCount;
  const fann *ann = data._fann;

  gpuann multidata;
  creategpuann(multidata, ann, instanceCount);

  for(unsigned int i = 0; i < instanceCount; i++)
    copygpuannValues(multidata, data, 0, i);

  for(unsigned int i = 0; i < maxEpochs; ++i)
  {
    gpuann_fann_parallel_train_epoch(multidata, data, trainData);
  }

  removegpuann(multidata);
}

void gpuann_fann_parallel_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs)
{
  float error;
  unsigned int i;

  gpuann data;
  gpuannTrainData trainData;

  creategpuannTrainData(trainData, train);

  (ann->first_layer->last_neuron - 1)->value = 1; ///bias input TODO!
  creategpuann(data, ann);
  loadgpuann(data, ann);

  gpuann_fann_parallel_train_on_data(data, trainData, maxEpochs);

  savegpuann(data, ann);

  removegpuann(data);
  removegpuannTrainData(trainData);
}