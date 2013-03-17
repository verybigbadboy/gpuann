#include <base/gpuannDataCreator.h>

#include <cuda.h>
#include <cuda_runtime.h>

void creategpuann(gpuann& nn, const fann *ann, unsigned int instanceCount)
{
  nn.d_sumArray     = 0;
  nn.d_valuesArray  = 0;
  nn.d_weightsArray = 0;
  nn._instanceCount = instanceCount;

  nn._fann = ann;

  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  nn.h_tmp_valuesArray  = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  nn.h_tmp_sumArray     = (fann_type *)malloc(neuronCount * sizeof(fann_type));

  cudaMalloc((void **)&(nn.d_sumArray),          instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_valuesArray),       instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_trainErrorsArray),  instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_weightsArray),      instanceCount * weightsCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_prevWeightsDeltas), instanceCount * weightsCount * sizeof(fann_type));
  

  nn._neuronsCountPerInstance = neuronCount;
  nn._weightsCountPerInstance = weightsCount;
}

void removegpuann(gpuann& nn)
{
  free(nn.h_tmp_sumArray);
  free(nn.h_tmp_valuesArray);

  cudaFree(nn.d_sumArray);
  cudaFree(nn.d_valuesArray);
  cudaFree(nn.d_weightsArray);
  cudaFree(nn.d_trainErrorsArray);
  cudaFree(nn.d_prevWeightsDeltas);
}

void loadgpuann(gpuann& nn, const fann *ann, unsigned int instanceIndex)
{
  unsigned int neuronCount = ann->total_neurons;
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;

  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  struct fann_neuron * last_neuron = layer_it->last_neuron;
  struct fann_neuron * neuron_it   = layer_it->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    nn.h_tmp_valuesArray[currentNeuronShift] = neuron_it->value;
    nn.h_tmp_sumArray[currentNeuronShift]    = neuron_it->sum;
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
      nn.h_tmp_valuesArray[currentNeuronShift] = neuron_it->value;
      nn.h_tmp_sumArray[currentNeuronShift]    = neuron_it->sum;
    }
  }

  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpyAsync(nn.d_sumArray + neuronCount * instanceIndex,     nn.h_tmp_sumArray,    neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(nn.d_valuesArray + neuronCount * instanceIndex,  nn.h_tmp_valuesArray, neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(nn.d_weightsArray + weightsCount * instanceIndex, ann->weights,        weightsCount * sizeof(fann_type), cudaMemcpyHostToDevice);
}

void savegpuann(const gpuann& nn, fann *ann, unsigned int instanceIndex)
{
  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpyAsync(nn.h_tmp_sumArray,     nn.d_sumArray + neuronCount * instanceIndex,      neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(nn.h_tmp_valuesArray,  nn.d_valuesArray + neuronCount * instanceIndex,   neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(ann->weights,          nn.d_weightsArray + weightsCount * instanceIndex, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

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
      neuron_it->value = nn.h_tmp_valuesArray[currentNeuronShift];
      neuron_it->sum = nn.h_tmp_sumArray[currentNeuronShift];
    }
  }
}
