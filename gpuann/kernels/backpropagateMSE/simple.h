#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <base/derived.h>
#include <string>

#include <kernels/backpropagateMSE/parallel.h>
#include <kernels/backpropagateMSE/multiNeuron.h>

//prevlayer block

template <unsigned int blockSize, unsigned int prevActivationFunction>
__global__ void fann_backpropagate_MSE_simple_gpu_kernel(const unsigned int prevNeuronsCount,
                                                         const unsigned int neuronsCount,
                                                         fann_type *weights,
                                                         fann_type *trainErrors,
                                                         fann_type *prevTrainErrors,
                                                         fann_type *prevValue,
                                                         fann_type *prevSum,
                                                         fann_type prevSteepness,
                                                         const unsigned int totalNeuronsCount,
                                                         const unsigned int totalWeightsCount)
{
  const unsigned int tid                      = threadIdx.x;
  const unsigned int instance                 = blockIdx.y;
  const unsigned int weightPerNeuronCount     = prevNeuronsCount;
  unsigned int neuronIndex                    = tid;
  const unsigned int prevLayerNeuron          = blockIdx.x;
  const unsigned int prevLayerNeuronInstanced = prevLayerNeuron + instance * totalNeuronsCount;


  __shared__ fann_type sum[blockSize];

  fann_type mySum = 0;

  while (neuronIndex < neuronsCount)
  {
    const unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;
    const unsigned int weightBeginIndex     = neuronIndex * weightPerNeuronCount + instance * totalWeightsCount;

    mySum       += trainErrors[neuronIndexInstanced] * weights[weightBeginIndex + prevLayerNeuron];
    neuronIndex += blockSize;
  }

  if(tid < blockSize)
    sum[tid] = mySum;


  __syncthreads();

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      sum[tid] = mySum = mySum + sum[tid + 256];
    }

    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      sum[tid] = mySum = mySum + sum[tid + 128];
    }

    __syncthreads();
  }

  if (blockSize >= 128)
  {
    if (tid <  64)
    {
      sum[tid] = mySum = mySum + sum[tid +  64];
    }

    __syncthreads();
  }

  if (tid < 32)
  {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile fann_type *smem = sum;

    if (blockSize >=  64)
    {
      smem[tid] = mySum = mySum + smem[tid + 32];
    }

    if (blockSize >=  32)
    {
      smem[tid] = mySum = mySum + smem[tid + 16];
    }

    if (blockSize >=  16)
    {
      smem[tid] = mySum = mySum + smem[tid +  8];
    }

    if (blockSize >=   8)
    {
      smem[tid] = mySum = mySum + smem[tid +  4];
    }

    if (blockSize >=   4)
    {
      smem[tid] = mySum = mySum + smem[tid +  2];
    }

    if (blockSize >=   2)
    {
      smem[tid] = mySum = mySum + smem[tid +  1];
    }
  }

  __syncthreads();

  if(tid == 0)
    prevTrainErrors[prevLayerNeuronInstanced] = sum[0] * gpuann_fann_activation_derived<prevActivationFunction>(prevSteepness, prevValue[prevLayerNeuronInstanced], prevSum[prevLayerNeuronInstanced]);
}

template <unsigned int blockSize>
void fann_backpropagate_MSE_activationFunction(unsigned int instanceCount, unsigned int prevActivationFunction, unsigned int prevNeuronsCount, unsigned int neuronsCount, fann_type *weights, fann_type *trainErrors, fann_type *prevTrainErrors, fann_type *prevValue, fann_type *prevSum, fann_type prevSteepness
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(prevNeuronsCount, instanceCount, 1);

#define fann_backpropagate_MSE_simple_gpu_kernel_case(X)   case X: \
fann_backpropagate_MSE_simple_gpu_kernel<blockSize, X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
break;

  switch(prevActivationFunction)
  {
    fann_backpropagate_MSE_simple_gpu_kernel_case(0);
    fann_backpropagate_MSE_simple_gpu_kernel_case(1);
    fann_backpropagate_MSE_simple_gpu_kernel_case(2);
    fann_backpropagate_MSE_simple_gpu_kernel_case(3);
    fann_backpropagate_MSE_simple_gpu_kernel_case(4);
    fann_backpropagate_MSE_simple_gpu_kernel_case(5);
    fann_backpropagate_MSE_simple_gpu_kernel_case(6);
    fann_backpropagate_MSE_simple_gpu_kernel_case(7);
    fann_backpropagate_MSE_simple_gpu_kernel_case(8);
    fann_backpropagate_MSE_simple_gpu_kernel_case(9);
    fann_backpropagate_MSE_simple_gpu_kernel_case(10);
    fann_backpropagate_MSE_simple_gpu_kernel_case(11);
    fann_backpropagate_MSE_simple_gpu_kernel_case(12);
    fann_backpropagate_MSE_simple_gpu_kernel_case(13);
    fann_backpropagate_MSE_simple_gpu_kernel_case(14);
    fann_backpropagate_MSE_simple_gpu_kernel_case(15);
  }
}

void fann_backpropagate_MSE_simple_implementation(unsigned int instanceCount,
                                                  unsigned int prevActivationFunction,
                                                  unsigned int prevNeuronsCount,
                                                  unsigned int neuronsCount,
                                                  fann_type *weights,
                                                  fann_type *trainErrors,
                                                  fann_type *prevTrainErrors,
                                                  fann_type *prevValue,
                                                  fann_type *prevSum,
                                                  fann_type prevSteepness,
                                                  unsigned int totalNeuronsCount,
                                                  unsigned int totalWeightsCount)
{
  unsigned int threadsCount = pow2roundup(neuronsCount);

  if(threadsCount < 32)
    threadsCount = 32;

  threadsCount /= 2;

#define fann_backpropagate_MSE_gpu_kernel_activationFunction_case(X)   case X: \
  fann_backpropagate_MSE_activationFunction<X> (instanceCount, prevActivationFunction, prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
break;

  switch (threadsCount)
  {
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(32);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(64);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(128);
  default:
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(256);
  }
}

void fann_backpropagate_MSE_select_implementation(unsigned int instanceCount,
                                                  unsigned int prevActivationFunction,
                                                  unsigned int prevNeuronsCount,
                                                  unsigned int neuronsCount,
                                                  fann_type *weights,
                                                  fann_type *trainErrors,
                                                  fann_type *prevTrainErrors,
                                                  fann_type *prevValue,
                                                  fann_type *prevSum,
                                                  fann_type prevSteepness,
                                                  unsigned int totalNeuronsCount,
                                                  unsigned int totalWeightsCount)
{
  if(neuronsCount < 32)
  {
    fann_backpropagate_MSE_multineuron_implementation(instanceCount,
                                                      prevActivationFunction,
                                                      prevNeuronsCount,
                                                      neuronsCount,
                                                      weights,
                                                      trainErrors,
                                                      prevTrainErrors,
                                                      prevValue,
                                                      prevSum,
                                                      prevSteepness,
                                                      totalNeuronsCount,
                                                      totalWeightsCount);
  }
  else
  {
    if(1)
      fann_backpropagate_MSE_parallel_implementation(instanceCount,
                                                     prevActivationFunction,
                                                     prevNeuronsCount,
                                                     neuronsCount,
                                                     weights,
                                                     trainErrors,
                                                     prevTrainErrors,
                                                     prevValue,
                                                     prevSum,
                                                     prevSteepness,
                                                     totalNeuronsCount,
                                                     totalWeightsCount);
    else
      fann_backpropagate_MSE_simple_implementation(instanceCount,
                                                   prevActivationFunction,
                                                   prevNeuronsCount,
                                                   neuronsCount,
                                                   weights,
                                                   trainErrors,
                                                   prevTrainErrors,
                                                   prevValue,
                                                   prevSum,
                                                   prevSteepness,
                                                   totalNeuronsCount,
                                                   totalWeightsCount);

  }
}

void gpuann_fann_backpropagate_MSE_implementation_gpu(gpuann &data)
{
  const fann *ann = data._fann;
  unsigned int instanceCount = data._instanceCount;
  fann_layer *lastLayer = ann->last_layer;
  fann_layer *secondLayer = ann->first_layer + 1;
  fann_neuron *firstNeuron = ann->first_layer->first_neuron;
  fann_layer *layerIt;

  for(layerIt = lastLayer - 1; layerIt != secondLayer; --layerIt)
  {
    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronShift = layerIt->first_neuron - firstNeuron;
    fann_layer *prevLayer = layerIt - 1;
    fann_neuron *prevLayerFirstNeuron = prevLayer->first_neuron;
    unsigned int prevLayerNeuronShift = prevLayerFirstNeuron - firstNeuron;
    unsigned int prevLayerSize = prevLayer->last_neuron - prevLayerFirstNeuron;
    unsigned int prevActivationFunction = prevLayerFirstNeuron->activation_function;
    fann_type prevSteepness = prevLayerFirstNeuron->activation_steepness;

    fann_backpropagate_MSE_select_implementation(
      instanceCount,
      prevActivationFunction,
      prevLayerSize,
      layerSize - 1, //because bias not connected to any
      &(data.d_weightsArray[layerIt->first_neuron->first_con]),
      &(data.d_trainErrorsArray[layerNeuronShift]),
      &(data.d_trainErrorsArray[prevLayerNeuronShift]),
      &(data.d_valuesArray[prevLayerNeuronShift]),
      &(data.d_sumArray[prevLayerNeuronShift]),
      prevSteepness,
      data._neuronsCountPerInstance,
      data._weightsCountPerInstance);
  }
}
