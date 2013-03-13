#include <base/gpuannDataCreator.h>

#include <cuda.h>
#include <cuda_runtime.h>


void loadgpuann(gpuann& nn, const fann *ann)
{
  nn._fann = ann;

  unsigned int neuronCount = ann->total_neurons;
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;

  nn.valuesArray  = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  nn.sumArray     = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  nn.weightsArray = ann->weights;

  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  struct fann_neuron * last_neuron = layer_it->last_neuron;
  struct fann_neuron * neuron_it   = layer_it->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    nn.valuesArray[currentNeuronShift] = neuron_it->value;
    nn.sumArray[currentNeuronShift]    = neuron_it->sum;
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
      nn.valuesArray[currentNeuronShift] = neuron_it->value;
      nn.sumArray[currentNeuronShift]    = neuron_it->sum;
    }
  }

  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

//  for(int i = 0; i < weightsCount; ++i)
//   nn.weightsArray[i] = 1;

  cudaMalloc((void **)&(nn.d_sumArray),     neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_valuesArray),  neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_weightsArray), weightsCount * sizeof(fann_type));

  cudaMemcpy(nn.d_sumArray,     nn.sumArray,     neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpy(nn.d_valuesArray,  nn.valuesArray,  neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpy(nn.d_weightsArray, nn.weightsArray, weightsCount * sizeof(fann_type), cudaMemcpyHostToDevice);
}

void savegpuann(const gpuann& nn, fann *ann)
{
  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpy(nn.sumArray,     nn.d_sumArray,     neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.valuesArray,  nn.d_valuesArray,  neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.weightsArray, nn.d_weightsArray, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);

  cudaFree(nn.d_sumArray);
  cudaFree(nn.d_valuesArray);
  cudaFree(nn.d_weightsArray);

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
      neuron_it->value = nn.valuesArray[currentNeuronShift];
      neuron_it->sum = nn.sumArray[currentNeuronShift];
    }
  }

  free(nn.sumArray);
  free(nn.valuesArray);
}
