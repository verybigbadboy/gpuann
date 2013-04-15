#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <base/derived.h>
#include <string>

//prevlayer block

template <unsigned int blockSize, unsigned int prevActivationFunction>
__device__ inline void fann_backpropagate_MSE_gpu_kernel(unsigned int prevNeuronsCount,
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
  unsigned int tid                      = threadIdx.x;
  unsigned int instance                 = blockIdx.y;
  unsigned int weightPerNeuronCount     = prevNeuronsCount;
  unsigned int neuronIndex              = tid + instance * totalNeuronsCount;
  unsigned int prevLayerNeuron          = blockIdx.x;
  unsigned int prevLayerNeuronInstanced = prevLayerNeuron + instance * totalNeuronsCount;
  unsigned int weightBeginIndex         = tid * weightPerNeuronCount + instance * totalWeightsCount;

  __shared__ fann_type sum[blockSize];

  if(tid < blockSize)
    sum[tid] = 0;
  __syncthreads();

  fann_type mySum = 0;

  if(tid < neuronsCount)
  {
    mySum = trainErrors[neuronIndex] * weights[weightBeginIndex + prevLayerNeuron];
    sum[tid] = mySum;
  }

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

  unsigned int localMemorySize = blockSize / 2;
  if(localMemorySize > 32)
    localMemorySize = 32;

  if (tid < localMemorySize)
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
    prevTrainErrors[prevLayerNeuronInstanced] = mySum * gpuann_fann_activation_derived<prevActivationFunction>(prevSteepness, prevValue[prevLayerNeuronInstanced], prevSum[prevLayerNeuronInstanced]);
}

#define fann_backpropagate_MSE_gpu_kernel_case(X)   case X: \
fann_backpropagate_MSE_gpu_kernel<blockSize, X>(prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
break;

template <unsigned int blockSize>
__global__ void fann_backpropagate_MSE_gpu_kernel_activationFunction(unsigned int prevActivationFunction, unsigned int prevNeuronsCount, unsigned int neuronsCount, fann_type *weights, fann_type *trainErrors, fann_type *prevTrainErrors, fann_type *prevValue, fann_type *prevSum, fann_type prevSteepness
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  switch(prevActivationFunction)
  {
    fann_backpropagate_MSE_gpu_kernel_case(0);
    fann_backpropagate_MSE_gpu_kernel_case(1);
    fann_backpropagate_MSE_gpu_kernel_case(2);
    fann_backpropagate_MSE_gpu_kernel_case(3);
    fann_backpropagate_MSE_gpu_kernel_case(4);
    fann_backpropagate_MSE_gpu_kernel_case(5);
    fann_backpropagate_MSE_gpu_kernel_case(6);
    fann_backpropagate_MSE_gpu_kernel_case(7);
    fann_backpropagate_MSE_gpu_kernel_case(8);
    fann_backpropagate_MSE_gpu_kernel_case(9);
    fann_backpropagate_MSE_gpu_kernel_case(10);
    fann_backpropagate_MSE_gpu_kernel_case(11);
    fann_backpropagate_MSE_gpu_kernel_case(12);
    fann_backpropagate_MSE_gpu_kernel_case(13);
    fann_backpropagate_MSE_gpu_kernel_case(14);
    fann_backpropagate_MSE_gpu_kernel_case(15);
  }
}

#define fann_backpropagate_MSE_gpu_kernel_activationFunction_case(X)   case X: \
fann_backpropagate_MSE_gpu_kernel_activationFunction<X> <<<dimGrid, dimBlock>>> (prevActivationFunction, prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
break;

void fann_backpropagate_MSE_gpu_kernel_blockSize(unsigned int instanceCount, unsigned int prevActivationFunction, unsigned int prevNeuronsCount, unsigned int neuronsCount, fann_type *weights, fann_type *trainErrors, fann_type *prevTrainErrors, fann_type *prevValue, fann_type *prevSum, fann_type prevSteepness
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  unsigned int threadsCount = pow2roundup(neuronsCount);

  if(threadsCount < 4)
    threadsCount = 4;
  else
    if(threadsCount > 512)
      throw std::string("too many inputs");

  dim3 dimBlock(threadsCount, 1, 1);
  dim3 dimGrid(prevNeuronsCount, instanceCount, 1);
    
  switch (threadsCount)
  {
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(4);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(8);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(16);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(32);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(64);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(128);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(256);
    fann_backpropagate_MSE_gpu_kernel_activationFunction_case(512);
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

    fann_backpropagate_MSE_gpu_kernel_blockSize(
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

/*
void fann_backpropagate_MSE(struct fann *ann)
{
  fann_type tmp_error;
  unsigned int i;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it, *last_neuron;
  struct fann_neuron **connections;

  fann_type *error_begin = ann->train_errors;
  fann_type *error_prev_layer;
  fann_type *weights;
  const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  const struct fann_layer *second_layer = ann->first_layer + 1;
  struct fann_layer *last_layer = ann->last_layer;

  for(layer_it = last_layer - 1; layer_it > second_layer; --layer_it)
  {
    last_neuron = layer_it->last_neuron;


    error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);

    neuron_it = layer_it->first_neuron;
    unsigned int prevLayerNeuronShift = neuron_it->first_con;
    unsigned int connectionsCount = neuron_it->last_con - neuron_it->first_con;
    for(unsigned int i = 0; i < connectionsCount; ++i)
    {
      for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
      {
        tmp_error = error_begin[neuron_it - first_neuron];
        weights = ann->weights + neuron_it->first_con;
        error_prev_layer[i] += tmp_error * weights[i];
      }
    }

    error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
    last_neuron = (layer_it - 1)->last_neuron;

    for(neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      *error_prev_layer *= fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness, neuron_it->value, neuron_it->sum);
      error_prev_layer++;
    }
  }
}
*/