#include "fannCuda.h"
#include "cuda/gpuDataCreator.h"
#include "cuda/neuralNetworkTypeCheck.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

template <unsigned int blockSize, unsigned int layerActivationFunction>
__global__ void runGpuKernel(const unsigned int neuronInputCount, const fann_type * inputArray, const fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, const fann_type layerSteepness)
{
  __shared__ fann_type local[blockSize];

  unsigned int tid = threadIdx.x;

  fann_type l_summ = 0;

  if(tid < neuronInputCount)
  {
    l_summ = (fann_type) (inputArray[tid] * weightsArray[neuronInputCount * blockIdx.x + tid]);
    if((tid + blockSize) < neuronInputCount)
      l_summ += (fann_type) (inputArray[tid + blockSize] * weightsArray[neuronInputCount * blockIdx.x + tid + blockSize]);
  }

  local[tid] = l_summ;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      local[tid] = l_summ = l_summ + local[tid + 256];
    }

    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      local[tid] = l_summ = l_summ + local[tid + 128];
    }

    __syncthreads();
  }

  if (blockSize >= 128)
  {
    if (tid <  64)
    {
      local[tid] = l_summ = l_summ + local[tid + 64];
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
      smem[tid] = l_summ = l_summ + smem[tid + 32];
    }

    if (blockSize >=  32)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 16];
    }

    if (blockSize >=  16)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 8];
    }

    if (blockSize >=   8)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 4];
    }

    if (blockSize >=   4)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 2];
    }

    if (blockSize >=   2)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 1];
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


#define runGpuActivatedCase(X) case X: \
  runGpuKernel <blockSize, X> <<<dimGrid, dimBlock>>>(neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness); \
  break;

template <unsigned int blockSize>
void runGpuActivated(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(neuronCount - 1, 1, 1);

  switch(layerActivationFunction)
  {
    runGpuActivatedCase(0);
    runGpuActivatedCase(1);
    runGpuActivatedCase(2);
    runGpuActivatedCase(3);
    runGpuActivatedCase(4);
    runGpuActivatedCase(5);
    runGpuActivatedCase(6);
    runGpuActivatedCase(7);
    runGpuActivatedCase(8);
    runGpuActivatedCase(9);
    runGpuActivatedCase(10);
    runGpuActivatedCase(11);
    runGpuActivatedCase(12);
    runGpuActivatedCase(13);
    runGpuActivatedCase(14);
    runGpuActivatedCase(15);
  }
}

inline int pow2roundup (int x)
{
  if (x < 0)
    return 0;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x+1;
}

#define runGpuThreadsCase(X) case X: \
  runGpuActivated <X> (neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, layerActivationFunction, neuronCount); \
  break;

void runGpu(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount)
{
  unsigned int threadsCount = pow2roundup(neuronInputCount) / 2;
  if(threadsCount < 4)
    threadsCount = 4;
  else
    if(threadsCount > 256)
      throw std::string("too many inputs");

  switch (threadsCount)
  {
    runGpuThreadsCase(4);
    runGpuThreadsCase(8);
    runGpuThreadsCase(16);
    runGpuThreadsCase(32);
    runGpuThreadsCase(64);
    runGpuThreadsCase(128);
    runGpuThreadsCase(256);
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
    __int64 inputNeuronArrayShift = (layer_it - 1)->first_neuron - neuronsArray;
    __int64 currentNeuronArrayShift = neuron_it - neuronsArray;
    __int64 weightsArrayShift = neuron_it->first_con;

    runGpu(layerNeuronInputCount
        , &(data.d_valuesArray[inputNeuronArrayShift])
        , &(data.d_weightsArray[weightsArrayShift])
        , &(data.d_sumArray[currentNeuronArrayShift])
        , &(data.d_valuesArray[currentNeuronArrayShift])
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