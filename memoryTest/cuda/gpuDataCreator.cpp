#include "gpuDataCreator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

void loadInputs(struct fann * ann, fann_type * input)
{
  struct fann_neuron *neuronArray = ann->first_layer->first_neuron;

  unsigned int num_input = ann->num_input;
  for(unsigned int i = 0; i != num_input; i++)
  {
    neuronArray[i].value = input[i];
  }

  (ann->first_layer->last_neuron - 1)->value = 1;
}

void prepareData(struct fann * ann, fann_type * input, gpuData &data)
{
  loadInputs(ann, input);

  unsigned int neuronCount = ann->total_neurons;
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;

  data.valuesArray  = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  data.sumArray     = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  data.weightsArray = ann->weights;

  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  struct fann_neuron * last_neuron = layer_it->last_neuron;
  struct fann_neuron * neuron_it   = layer_it->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    data.valuesArray[currentNeuronShift] = neuron_it->value;
    data.sumArray[currentNeuronShift]    = neuron_it->sum;
  }

  for(layer_it += 1; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it != last_neuron; neuron_it++)
    {
      unsigned int currentNeuronShift  = neuron_it - neuronsArray;
      if(neuron_it->last_con - neuron_it->first_con == 0)
        neuron_it->value = 1;
      data.valuesArray[currentNeuronShift] = neuron_it->value;
      data.sumArray[currentNeuronShift]    = neuron_it->sum;
    }
  }

  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  /*for(int i = 0; i < weightsCount; ++i)
    data.weightsArray[i] = 1;*/

  cudaMalloc((void **)&(data.d_sumArray),     neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(data.d_valuesArray),  neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(data.d_weightsArray), weightsCount * sizeof(fann_type));

  checkCudaErrors(cudaMemcpy(data.d_sumArray,     data.sumArray,     neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice));
  cudaMemcpy(data.d_valuesArray,  data.valuesArray,  neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpy(data.d_weightsArray, data.weightsArray, weightsCount * sizeof(fann_type), cudaMemcpyHostToDevice);
}

void unPrepareAndFreeData(gpuData &data, fann *ann)
{
  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpy(data.sumArray,     data.d_sumArray,     neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpy(data.valuesArray,  data.d_valuesArray,  neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpy(data.weightsArray, data.d_weightsArray, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);

  cudaFree(data.d_sumArray);
  cudaFree(data.d_valuesArray);
  cudaFree(data.d_weightsArray);

  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  for(; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it != last_neuron; neuron_it++)
    {
      unsigned int currentNeuronShift  = neuron_it - neuronsArray;
      neuron_it->value = data.valuesArray[currentNeuronShift];
      neuron_it->sum = data.sumArray[currentNeuronShift];
    }
  }

  free(data.sumArray);
  free(data.valuesArray);
}
