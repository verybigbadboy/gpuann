
#include <gpuannTrain.h>
#include <base/gpuannDataCreator.h>

#include <fann.h>

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

void gpuann_fann_train_epoch_base(gpuann &data, gpuannTrainData &trainData)
{
  const fann *ann = data._fann;

  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
}

void gpuann_fann_train_epoch_quickprop(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_train_epoch_base(data, trainData);
  gpuann_fann_update_weights_quickprop_implementation(data, trainData._dataCount, 0, data._weightsCountPerInstance);
}


void gpuann_fann_train_epoch_irpropm(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_train_epoch_base(data, trainData);
  gpuann_fann_update_weights_irpropm_implementation(data, 0, data._weightsCountPerInstance);
}

void gpuann_fann_train_epoch_sarprop(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_train_epoch_base(data, trainData);
  gpuann_fann_update_weights_sarprop_implementation(data, data._sarpropEpoch, 0, data._weightsCountPerInstance);
  ++(data._sarpropEpoch);
}

void gpuann_fann_train_epoch_batch(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_train_epoch_base(data, trainData);
  gpuann_fann_update_weights_batch_implementation(data, trainData._dataCount, 0, data._weightsCountPerInstance);
}

/*---------------SIMPLE-----------------*/

void gpuann_fann_train_epoch_incremental(gpuann &data, gpuannTrainData &trainData)
{
  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_weights_implementation(data);
  }
}

void gpuann_fann_train_epoch(gpuann &data, gpuannTrainData &trainData)
{
  switch (data._trainingAlgorithm)
  {
    case GPUANN_FANN_TRAIN_QUICKPROP:
      gpuann_fann_train_epoch_quickprop(data, trainData);
      break;
    case GPUANN_FANN_TRAIN_RPROP:
      gpuann_fann_train_epoch_irpropm(data, trainData);
      break;
    case GPUANN_FANN_TRAIN_SARPROP:
      gpuann_fann_train_epoch_sarprop(data, trainData);
      break;
    case GPUANN_FANN_TRAIN_BATCH:
      gpuann_fann_train_epoch_batch(data, trainData);
      break;
    case GPUANN_FANN_TRAIN_INCREMENTAL:
      gpuann_fann_train_epoch_incremental(data, trainData);
      break;
  }
}

