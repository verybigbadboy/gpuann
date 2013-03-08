#include "fannCuda.h"
#include "cuda/gpuDataCreator.h"
#include "cuda/neuralNetworkTypeCheck.h"

#include <cuda.h>
#include <cuda_runtime.h>

template <unsigned int blockSize, unsigned int layerActivationFunction>
__global__ void runGpuKernel(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  __shared__ fann_type local[256];
  if(tid < neuronInputCount)
  {
    local[tid] = (fann_type) (inputArray[tid] * weightsArray[neuronInputCount * blockIdx.x + tid]);
  }
  else
    local[tid] = 0;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s>32; s>>=1)
  {
    if (tid < s)
    {
      local[tid] += local[tid + s];
    }
    __syncthreads();
  }
  
  if (tid < 32)
  {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile fann_type *smem = local;
    
    if (blockSize >=  64)
    {
      smem[tid] += smem[tid + 32];
    }
    
    if (blockSize >=  32)
    {
      smem[tid] += smem[tid + 16];
    }
    
    if (blockSize >=  16)
    {
      smem[tid] += smem[tid +  8];
    }
    
    if (blockSize >=   8)
    {
      smem[tid] += smem[tid +  4];
    }
    
    if (blockSize >=   4)
    {
      smem[tid] += smem[tid +  2];
    }
    
    if (blockSize >=   2)
    {
      smem[tid] += smem[tid +  1];
    }
  }
  
  if (tid == 0)
  {
    fann_type neuron_sum = local[0];
    neuron_sum *= layerSteepness;
    
    fann_type max_sum = 150 / layerSteepness;
    if(neuron_sum > max_sum)
      neuron_sum = max_sum;
    else
      if(neuron_sum < -max_sum)
        neuron_sum = -max_sum;

    sumArray[blockIdx.x] = neuron_sum;

    fann_activation_switch(layerActivationFunction, neuron_sum, outputArray[blockIdx.x]);
  }
}


#define runGpuCase(X) case X: \
  runGpuKernel <blockSize, X> <<<dimGrid, dimBlock>>>(neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness); \
  break;

template <unsigned int blockSize>
void runGpu(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount)
{
  unsigned int threads = 256;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(neuronCount, 1, 1);

  switch(layerActivationFunction)
  {
    runGpuCase(0);
    runGpuCase(1);
    runGpuCase(2);
    runGpuCase(3);
    runGpuCase(4);
    runGpuCase(5);
    runGpuCase(6);
    runGpuCase(7);
    runGpuCase(8);
    runGpuCase(9);
    runGpuCase(10);
    runGpuCase(11);
    runGpuCase(12);
    runGpuCase(13);
    runGpuCase(14);
    runGpuCase(15);
  }
}

void run(struct fann * ann, gpuData &data)
{
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  struct fann_layer *last_layer = ann->last_layer;
  
  for(struct fann_layer *layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;

    fann_type    layerSteepness = neuron_it->activation_steepness;
    unsigned int layerActivationFunction = neuron_it->activation_function;
    unsigned int layerNeuronInputCount = neuron_it->last_con - neuron_it->first_con;
    unsigned int inputNeuronArrayShift = (layer_it - 1)->first_neuron - neuronsArray;
    unsigned int currentNeuronArrayShift = neuron_it - neuronsArray;
    unsigned int weightsArrayShift = neuron_it->first_con;

    runGpu<256> (layerNeuronInputCount
               , data.d_valuesArray + inputNeuronArrayShift
               , data.d_weightsArray + weightsArrayShift
               , data.d_sumArray + currentNeuronArrayShift
               , data.d_valuesArray + currentNeuronArrayShift
               , layerSteepness
               , layerActivationFunction
               , last_neuron - neuron_it);
  }
}


fann_type * fann_run1(struct fann * ann, fann_type * input)
{
  check(ann);
  
  gpuData data;
  prepareData(ann, input, data);
  for(int i = 0; i < 1e6; ++i)
    run(ann, data);
  unPrepareAndFreeData(data, ann);
  
  fann_type *output = ann->output;
  unsigned int num_output = ann->num_output;
  fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
  for(unsigned int i = 0; i != num_output; i++)
  {
    output[i] = neurons[i].value;
  }
  return ann->output;
}