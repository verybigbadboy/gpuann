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
  //do something TODO
}

float gpuann_fann_get_MSE(gpuann &data)
{
  //TODO
}

float gpuann_fann_train_epoch_quickprop(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);
  
  const fann *ann = data._fann;
  
  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  gpuann_fann_update_weights_quickprop_implementation(data, trainData._dataCount, 0, data._weightsCountPerInstance);

  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_irpropm(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);

  const fann *ann = data._fann;

  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  gpuann_fann_update_weights_irpropm_implementation(data, 0, ann->total_connections);
  
  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_sarprop(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);
  
  const fann *ann = data._fann;
  
  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }
  
  gpuann_fann_update_weights_sarprop_implementation(data, data._sarpropEpoch, 0, ann->total_connections);
  
  ++(data._sarpropEpoch);

  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_batch(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);

  const fann *ann = data._fann;

  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_slopes_batch_implementation(data, ann->first_layer + 1, ann->last_layer - 1);
  }

  gpuann_fann_update_weights_batch_implementation(data, trainData._dataCount, 0, ann->total_connections);

  return gpuann_fann_get_MSE(data);
}

float gpuann_fann_train_epoch_incremental(gpuann &data, gpuannTrainData &trainData)
{
  gpuann_fann_reset_MSE(data);

  for(unsigned int i = 0; i < trainData._dataCount; i++)
  {
    gpuann_fann_run_device(data, trainData.d_input + trainData._inputCount * i);
    gpuann_fann_compute_MSE_implementation_gpu(data, trainData.d_output + trainData._outputCount * i);
    gpuann_fann_backpropagate_MSE_implementation_gpu(data);
    gpuann_fann_update_weights_implementation(data);
  }

  return gpuann_fann_get_MSE(data);
}

void print2arrays(unsigned int size, fann_type *f, fann_type *s)
{
  printf("ololo\n");
  for(unsigned int i = 0; i < size; ++i)
  {
    printf("%10.3f %10.3f\n", f[i], s[i]);
  }
}

void fann_update_slopes_batch1(struct fann *ann, struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons;
  fann_type tmp_error;
  unsigned int i, num_connections;

  if(ann->train_slopes == NULL)
  {
    ann->train_slopes =
    (fann_type *) calloc(ann->total_connections_allocated, sizeof(fann_type));
    if(ann->train_slopes == NULL)
    {
      fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  }

  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  fann_type *error_begin = ann->train_errors;
  fann_type *slope_begin, *neuron_slope;

  slope_begin = ann->train_slopes;

  prev_neurons = first_neuron;

  for(; layer_begin <= layer_end; layer_begin++)
  {
    last_neuron = layer_begin->last_neuron;
    prev_neurons = (layer_begin - 1)->first_neuron;

    for(neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      tmp_error = error_begin[neuron_it - first_neuron];
      neuron_slope = slope_begin + neuron_it->first_con;
      num_connections = neuron_it->last_con - neuron_it->first_con;
      for(i = 0; i != num_connections; i++)
      {
        neuron_slope[i] += tmp_error * prev_neurons[i].value;
      }
    }
  }
}

float fann_train_epoch_batch(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;

  fann_reset_MSE(ann);

  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch1(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_batch(ann, data->num_data, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

float fann_train_epoch_sarprop(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;

  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_sarprop(ann, ann->sarprop_epoch, 0, ann->total_connections);

  ++(ann->sarprop_epoch);

  return fann_get_MSE(ann);
}

float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;

  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_irpropm(ann, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

float fann_train_epoch_quickprop(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;

  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  fann_update_weights_quickprop(ann, data->num_data, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

void test(fann *ann, fann_train_data* train)
{
  gpuann data;
  gpuannTrainData trainData;
  debugGpuann dump[10];

  creategpuannTrainData(trainData, train);

  (ann->first_layer->last_neuron - 1)->value = 1; ///TODO WHY?
  creategpuann(data, ann);
  loadgpuann(data, ann);

  createDump(data, dump[0]);

/*
  gpuann_fann_train_epoch_incremental(data, trainData);
  createDump(data, dump[1]);

  fann_train(ann, train->input[0], train->output[0]);
  fann_train(ann, train->input[1], train->output[1]);
  fann_train(ann, train->input[2], train->output[2]);
  fann_train(ann, train->input[3], train->output[3]);

  print2arrays(data._neuronsCountPerInstance, ann->train_errors, dump[1].d_trainErrorsArray);
  print2arrays(data._weightsCountPerInstance, ann->weights, dump[1].d_weightsArray);
*/

/*
  fann_train_epoch_batch(ann, train);
  gpuann_fann_train_epoch_batch(data, trainData);
  createDump(data, dump[2]);
  print2arrays(data._weightsCountPerInstance, ann->weights, dump[2].d_weightsArray);
  //print2arrays(data._weightsCountPerInstance, ann->train_slopes, dump[2].d_trainSlopes);
*/

/*
  gpuann_fann_train_epoch_sarprop(data, trainData);
  createDump(data, dump[3]);
  fann_train_epoch_sarprop(ann, train);
  print2arrays(data._weightsCountPerInstance, ann->weights, dump[3].d_weightsArray);
*/

/*
  gpuann_fann_train_epoch_irpropm(data, trainData);
  createDump(data, dump[4]);
  fann_train_epoch_irpropm(ann, train);
  print2arrays(data._weightsCountPerInstance, ann->weights, dump[4].d_weightsArray);
*/
/*
  gpuann_fann_train_epoch_quickprop(data, trainData);
  createDump(data, dump[5]);
  fann_train_epoch_quickprop(ann, train);
  print2arrays(data._weightsCountPerInstance, ann->weights, dump[5].d_weightsArray);
*/
  removegpuann(data);
  removegpuannTrainData(trainData);
}




