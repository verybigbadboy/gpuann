#include <gpuann.h>
#include <base/gpuannDataCreator.h>
#include <base/neuralNetworkTypeCheck.h>

#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <kernels/computeMSE/run.h>
#include <kernels/straightforward/run.h>
#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <kernels/updateWeights/updateWeights.h>
#include <kernels/updateWeightsBatch/updateWeightsBatch.h>
#include <kernels/updateWeightsIrpropm/updateWeightsIrpropm.h>
#include <kernels/updateWeightsQuickprop/updateWeigthsQuickprop.h>
#include <kernels/updateWeightsSarprop/updateWeightsSarprop.h>

void loadInputs(struct fann * ann, fann_type * input)
{
  struct fann_neuron *neuronArray = ann->first_layer->first_neuron;

  unsigned int num_input = ann->num_input;
  for(unsigned int i = 0; i != num_input; i++)
  {
    neuronArray[i].value = input[i];
  }

  //fixme
  (ann->first_layer->last_neuron - 1)->value = 1;
}

fann_type * gpuann_fann_run(struct fann * ann, fann_type * input)
{
  check(ann);
  loadInputs(ann, input);

  gpuann gann;
  creategpuann(gann, ann);
  loadgpuann(gann, ann);

  gpuann_fann_run_implementation(gann);
  savegpuann(gann, ann);

  fann_type *output = ann->output;
  unsigned int num_output = ann->num_output;
  fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
  for(unsigned int i = 0; i != num_output; i++)
  {
    output[i] = neurons[i].value;
  }
  removegpuann(gann);

  return ann->output;
}

//returns device pointer!
fann_type * gpuann_fann_run_device(gpuann &data, fann_type * d_input)
{
  gpuann_loadInputs(data, d_input);
  gpuann_fann_run_implementation(data);
  return gpuann_getOutputsDevicePointer(data);
}

void gpuann_fann_multirun(struct fann * ann, fann_type ** input, unsigned int instanceCount, fann_type ** output)
{
  check(ann);

  gpuann gann;
  creategpuann(gann, ann, instanceCount);

  for(unsigned int i = 0; i < instanceCount; ++i)
  {
    loadInputs(ann, input[i]);
    loadgpuann(gann, ann, i);
  }

  gpuann_fann_run_implementation(gann);

  unsigned int num_output = ann->num_output;
  for(unsigned int i = 0; i < instanceCount; ++i)
  {
    savegpuann(gann, ann, i);

    fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
    for(unsigned int j = 0; j < num_output; ++j)
    {
      output[i][j] = neurons[j].value;
    }
  }

  removegpuann(gann);
}

#include <cuda.h>
#include <cuda_runtime.h>

void gpuann_fann_train(struct fann *ann, fann_type * input, fann_type * desired_output)
{
  check(ann);
  loadInputs(ann, input);

  gpuann gann;
  creategpuann(gann, ann);
  loadgpuann(gann, ann);

  fann_type *d_desired_output;
  cudaMalloc((void **)&(d_desired_output), ann->num_output * sizeof(fann_type));
  cudaMemcpyAsync(d_desired_output, desired_output, ann->num_output * sizeof(fann_type), cudaMemcpyHostToDevice);

  gpuann_fann_run_implementation(gann);
  gpuann_fann_compute_MSE_implementation_gpu(gann, d_desired_output);
  gpuann_fann_backpropagate_MSE_implementation_gpu(gann);
  gpuann_fann_update_weights_implementation(gann);

  savegpuann(gann, ann);

  removegpuann(gann);
}

void gpuann_fann_reset_MSE(gpuann &data)
{
  //do something
}

void gpuann_fann_clear_train_arrays(gpuann &data)
{
  //clear
  
}

float gpuann_fann_get_MSE(gpuann &data)
{
  //
}

float gpuann_fann_train_epoch_quickprop(gpuann &data, gpuannTrainData *trainData)
{
  if(data.d_prevTrainSlopes == NULL)
  {
    gpuann_fann_clear_train_arrays(data);
  }
  
  gpuann_fann_reset_MSE(data);
  
  const fann *ann = data._fann;
  
  for(unsigned int i = 0; i < trainData->_dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData->d_input + trainData->_inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData->d_output + trainData->_outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  gpuann_fann_update_weights_quickprop_implementation(data, trainData->_dataCount, 0, data._weightsCountPerInstance);

  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_irpropm(gpuann &data, gpuannTrainData *trainData)
{
  if(data.d_prevTrainSlopes == NULL)
  {
    gpuann_fann_clear_train_arrays(data);
  }

  gpuann_fann_reset_MSE(data);

  const fann *ann = data._fann;

  for(unsigned int i = 0; i < trainData->_dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData->d_input + trainData->_inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData->d_output + trainData->_outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  gpuann_fann_update_weights_irpropm_implementation(data, 0, ann->total_connections);
  
  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_sarprop(gpuann &data, gpuannTrainData *trainData)
{
  if(data.d_prevTrainSlopes == NULL)
  {
    gpuann_fann_clear_train_arrays(data);
  }
  
  gpuann_fann_reset_MSE(data);
  
  const fann *ann = data._fann;
  
  for(unsigned int i = 0; i < trainData->_dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData->d_input + trainData->_inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData->d_output + trainData->_outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  
  gpuann_fann_update_weights_sarprop_implementation(data, data._sarpropEpoch, 0, ann->total_connections);
  
  ++(data._sarpropEpoch);

  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_batch(gpuann &data, gpuannTrainData *trainData)
{
  if(data.d_prevTrainSlopes == NULL)
  {
    gpuann_fann_clear_train_arrays(data);
  }

  gpuann_fann_reset_MSE(data);

  const fann *ann = data._fann;

  for(unsigned int i = 0; i < trainData->_dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData->d_input + trainData->_inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData->d_output + trainData->_outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }

  gpuann_fann_update_weights_batch_implementation(data, trainData->_dataCount, 0, ann->total_connections);

  return gpuann_fann_get_MSE(data);
}

/*
* Internal train function
*/
float gpuann_fann_train_epoch_incremental(gpuann &data, gpuannTrainData *trainData)
{
  gpuann_fann_reset_MSE(data);

  for(unsigned int i = 0; i != trainData->_dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData->d_input + trainData->_inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData->d_output + trainData->_outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_weights_implementation(data);
  }

  return gpuann_fann_get_MSE(data);
}
