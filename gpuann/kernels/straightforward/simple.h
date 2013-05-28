#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>
#include <base/gpuannData.h>
#include <configuration.h>

template <unsigned int blockSize, unsigned int layerActivationFunction>
__global__ void gpuann_fann_run_gpu_kernel(const unsigned int neuronInputCount,
                                           fann_type         *inputArray,
                                           fann_type         *weightsArray,
                                           fann_type         *sumArray,
                                           fann_type         *outputArray,
                                           fann_type          layerSteepness,
                                           const unsigned int totalNeuronsCount,
                                           const unsigned int totalWeightsCount)
{
  const unsigned int tid                  = threadIdx.x;
  const unsigned int instance             = blockIdx.y;
  unsigned int inputIndex                 = tid;
  const unsigned int neuronIndex          = blockIdx.x;
  const unsigned int neuronIndexInstanced = neuronIndex + totalNeuronsCount * instance;

  __shared__ fann_type local[blockSize];

  fann_type l_summ = 0;

  unsigned int inputIndexInstanced;
  unsigned int weightIndex;
  unsigned int weightIndexInstanced;

  while(inputIndex < neuronInputCount)
  {
    inputIndexInstanced  = inputIndex + totalNeuronsCount * instance;
    weightIndex          = neuronInputCount * neuronIndex + inputIndex;
    weightIndexInstanced = weightIndex + totalWeightsCount * instance;

    l_summ     += (fann_type) (inputArray[inputIndexInstanced] * weightsArray[weightIndexInstanced]);
    inputIndex += blockSize;
  }

  if(tid < blockSize)
  {
    local[tid] = l_summ;
  }

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

    if (blockSize >=  32 && tid < 16)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 16];
    }

    if (blockSize >=  16 && tid < 8)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 8];
    }

    if (blockSize >=   8 && tid < 4)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 4];
    }

    if (blockSize >=   4 && tid < 2)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 2];
    }

    if (blockSize >=   2 && tid < 1)
    {
      smem[tid] = l_summ = l_summ + smem[tid + 1];
    }
  }

  //TODO: why  remove this line causes random value
  __syncthreads();

  if (tid == 0)
  {
    fann_type neuron_sum = l_summ;
    neuron_sum *= layerSteepness;

    fann_type max_sum = 150 / layerSteepness;
    if(neuron_sum > max_sum)
      neuron_sum = max_sum;
    else
      if(neuron_sum < -max_sum)
        neuron_sum = -max_sum;

    sumArray[neuronIndexInstanced] = neuron_sum;

    fann_activation_switch(layerActivationFunction, neuron_sum, outputArray[neuronIndexInstanced]);
  }
}

template <unsigned int blockSize>
void gpuann_fann_run_gpu_kernel_activated(unsigned int neuronInputCount,
                                          fann_type   *inputArray,
                                          fann_type   *weightsArray,
                                          fann_type   *sumArray,
                                          fann_type   *outputArray,
                                          fann_type    layerSteepness,
                                          unsigned int layerActivationFunction,
                                          unsigned int neuronCount,
                                          unsigned int instanceCount,
                                          unsigned int totalNeuronsCount,
                                          unsigned int totalWeightsCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((neuronCount - 1), instanceCount, 1);

#define gpuann_fann_run_gpu_kernelCase(X) case X: \
gpuann_fann_run_gpu_kernel <blockSize, X> <<<dimGrid, dimBlock>>>(neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, totalNeuronsCount, totalWeightsCount); \
break;

  switch(layerActivationFunction)
  {
    gpuann_fann_run_gpu_kernelCase(0);
    gpuann_fann_run_gpu_kernelCase(1);
    gpuann_fann_run_gpu_kernelCase(2);
    gpuann_fann_run_gpu_kernelCase(3);
    gpuann_fann_run_gpu_kernelCase(4);
    gpuann_fann_run_gpu_kernelCase(5);
    gpuann_fann_run_gpu_kernelCase(6);
    gpuann_fann_run_gpu_kernelCase(7);
    gpuann_fann_run_gpu_kernelCase(8);
    gpuann_fann_run_gpu_kernelCase(9);
    gpuann_fann_run_gpu_kernelCase(10);
    gpuann_fann_run_gpu_kernelCase(11);
    gpuann_fann_run_gpu_kernelCase(12);
    gpuann_fann_run_gpu_kernelCase(13);
    gpuann_fann_run_gpu_kernelCase(14);
    gpuann_fann_run_gpu_kernelCase(15);
  }
}

void gpuann_fann_run_simple_implementation(unsigned int neuronInputCount,
                                           fann_type   *inputArray,
                                           fann_type   *weightsArray,
                                           fann_type   *sumArray,
                                           fann_type   *outputArray,
                                           fann_type    layerSteepness,
                                           unsigned int layerActivationFunction,
                                           unsigned int neuronCount,
                                           unsigned int instanceCount,
                                           unsigned int totalNeuronsCount,
                                           unsigned int totalWeightsCount)
{
  unsigned int threadsCount = pow2roundup(neuronInputCount) / 2;
  if(threadsCount < 32)
    threadsCount = 32;
  
  if(!minimalThreadCountPerBlockOptimization)
    threadsCount = 256;

#define gpuann_fann_run_gpu_kernel_activatedCase(X) case X: \
gpuann_fann_run_gpu_kernel_activated <X> (neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, layerActivationFunction, neuronCount, instanceCount, totalNeuronsCount, totalWeightsCount); \
break;

  switch (threadsCount)
  {
    gpuann_fann_run_gpu_kernel_activatedCase(32);
    gpuann_fann_run_gpu_kernel_activatedCase(64);
    gpuann_fann_run_gpu_kernel_activatedCase(128);
  default:
    gpuann_fann_run_gpu_kernel_activatedCase(256);
  }
}


//for neurons with < 32 inputs
//always 32 threads, one input = one thread
template <unsigned int blockSize, unsigned int layerActivationFunction, const unsigned int neuronInputCount>
__global__ void gpuann_fann_run_multineuron_gpu_kernel(
                                                       const unsigned int neuronCount,
                                                       fann_type         *inputArray,
                                                       fann_type         *weightsArray,
                                                       fann_type         *sumArray,
                                                       fann_type         *outputArray,
                                                       fann_type          layerSteepness,
                                                       const unsigned int totalNeuronsCount,
                                                       const unsigned int totalWeightsCount)
{
  const unsigned int threadCount          = blockSize;
  const unsigned int tid                  = threadIdx.x;
  const unsigned int instance             = blockIdx.y;
  const unsigned int inputIndex           = tid % neuronInputCount;
  const unsigned int inputIndexInstanced  = inputIndex + totalNeuronsCount * instance;
  const unsigned int neuronIndexInKernel  = tid / neuronInputCount;
  const unsigned int neuronsPerKernel     = threadCount / neuronInputCount;
  const unsigned int neuronIndex          = blockIdx.x * neuronsPerKernel + neuronIndexInKernel;
  const unsigned int weightIndex          = neuronInputCount * neuronIndex + inputIndex;
  const unsigned int neuronIndexInstanced = neuronIndex + totalNeuronsCount * instance;
  const unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;

  __shared__ fann_type local[threadCount];

  fann_type l_summ = 0;

  if(tid < neuronsPerKernel * neuronInputCount && neuronIndex < neuronCount)
  {
    l_summ = (fann_type) (inputArray[inputIndexInstanced] * weightsArray[weightIndexInstanced]);
    local[tid] = l_summ;
  }

  __syncthreads();

  if (tid < neuronsPerKernel * neuronInputCount && neuronIndex < neuronCount)
  {
    volatile fann_type *smem = local;

    if(neuronInputCount > 16)
      if(inputIndex < 16)
        if(inputIndex + 16 < neuronInputCount)
          smem[tid] = l_summ = l_summ + smem[tid + 16];

    if(neuronInputCount > 8)
      if(inputIndex < 8)
        if(inputIndex + 8 < neuronInputCount)
          smem[tid] = l_summ = l_summ + smem[tid + 8];

    if(neuronInputCount > 4)
      if(inputIndex < 4)
        if(inputIndex + 4 < neuronInputCount)
          smem[tid] = l_summ = l_summ + smem[tid + 4];

    if(neuronInputCount > 2)
      if(inputIndex < 2)
        if(inputIndex + 2 < neuronInputCount)
          smem[tid] = l_summ = l_summ + smem[tid + 2];

    if(neuronInputCount > 1)
      if(inputIndex < 1)
        if(inputIndex + 1 < neuronInputCount)
          smem[tid] = l_summ = l_summ + smem[tid + 1];
  }

  //TODO: why  remove this line causes random value
  __syncthreads();

  if (inputIndex == 0 && neuronIndex < neuronCount && neuronIndexInKernel < neuronsPerKernel)
  {
    fann_type neuron_sum = local[tid];
    neuron_sum *= layerSteepness;

    fann_type max_sum = 150 / layerSteepness;
    if(neuron_sum > max_sum)
      neuron_sum = max_sum;
    else
      if(neuron_sum < -max_sum)
        neuron_sum = -max_sum;

    sumArray[neuronIndexInstanced] = neuron_sum;
    fann_activation_switch(layerActivationFunction, neuron_sum, outputArray[neuronIndexInstanced]);
  }
}

template <unsigned int neuronInputCount>
void gpuann_fann_run_small_neurons_implementation_neuronInput(
                                                  fann_type * inputArray,
                                                  fann_type * weightsArray,
                                                  fann_type *sumArray,
                                                  fann_type * outputArray,
                                                  fann_type layerSteepness,
                                                  unsigned int layerActivationFunction,
                                                  unsigned int neuronCount,
                                                  unsigned int instanceCount,
                                                  unsigned int totalNeuronsCount,
                                                  unsigned int totalWeightsCount)
{
  neuronCount--; //bias
  const unsigned int blockSize = 256; //just for shared memory;
  unsigned int threadNeeded = pow2roundup(neuronInputCount * neuronCount);
  if(threadNeeded > 256)
    threadNeeded = 256;
  
  if(!minimalThreadCountPerBlockOptimization)
    threadNeeded = 256;

  const unsigned int neuronsPerBlock = threadNeeded / neuronInputCount;
  const unsigned int blocksNeeded    = neuronCount / neuronsPerBlock + 1;

  dim3 dimBlock(threadNeeded, 1, 1);
  dim3 dimGrid(blocksNeeded, instanceCount, 1); // TODO create bias if

#define gpuann_fann_run_multineuron_gpu_kernelCase(X) case X: \
  gpuann_fann_run_multineuron_gpu_kernel <blockSize, X, neuronInputCount> <<<dimGrid, dimBlock>>>(neuronCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, totalNeuronsCount, totalWeightsCount); \
break;

  switch(layerActivationFunction)
  {
    gpuann_fann_run_multineuron_gpu_kernelCase(0);
    gpuann_fann_run_multineuron_gpu_kernelCase(1);
    gpuann_fann_run_multineuron_gpu_kernelCase(2);
    gpuann_fann_run_multineuron_gpu_kernelCase(3);
    gpuann_fann_run_multineuron_gpu_kernelCase(4);
    gpuann_fann_run_multineuron_gpu_kernelCase(5);
    gpuann_fann_run_multineuron_gpu_kernelCase(6);
    gpuann_fann_run_multineuron_gpu_kernelCase(7);
    gpuann_fann_run_multineuron_gpu_kernelCase(8);
    gpuann_fann_run_multineuron_gpu_kernelCase(9);
    gpuann_fann_run_multineuron_gpu_kernelCase(10);
    gpuann_fann_run_multineuron_gpu_kernelCase(11);
    gpuann_fann_run_multineuron_gpu_kernelCase(12);
    gpuann_fann_run_multineuron_gpu_kernelCase(13);
    gpuann_fann_run_multineuron_gpu_kernelCase(14);
    gpuann_fann_run_multineuron_gpu_kernelCase(15);
  }
}

void gpuann_fann_run_small_neurons_implementation(unsigned int neuronInputCount,
  fann_type * inputArray,
  fann_type * weightsArray,
  fann_type *sumArray,
  fann_type * outputArray,
  fann_type layerSteepness,
  unsigned int layerActivationFunction,
  unsigned int neuronCount,
  unsigned int instanceCount,
  unsigned int totalNeuronsCount,
  unsigned int totalWeightsCount)
{
  #define gpuann_fann_run_small_neurons_implementation_neuronInputCase(X) case X: \
  gpuann_fann_run_small_neurons_implementation_neuronInput <X> (inputArray, weightsArray, sumArray, outputArray, layerSteepness, layerActivationFunction, neuronCount, instanceCount, totalNeuronsCount, totalWeightsCount); \
  break;
  
  switch(neuronInputCount)
  {
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(1);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(2);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(3);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(4);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(5);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(6);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(7);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(8);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(9);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(10);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(11);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(12);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(13);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(14);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(15);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(16);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(17);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(18);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(19);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(20);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(21);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(22);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(23);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(24);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(25);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(26);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(27);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(28);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(29);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(30);
    gpuann_fann_run_small_neurons_implementation_neuronInputCase(31);
  }
}

void gpuann_fann_run_select_best_implementation(unsigned int neuronInputCount,
                                                fann_type * inputArray,
                                                fann_type * weightsArray,
                                                fann_type *sumArray,
                                                fann_type * outputArray,
                                                fann_type layerSteepness,
                                                unsigned int layerActivationFunction,
                                                unsigned int neuronCount,
                                                unsigned int instanceCount,
                                                unsigned int totalNeuronsCount,
                                                unsigned int totalWeightsCount)
{
  if(neuronInputCount <= 32 && straightforwardSmallNeuronImplementationEnabled)
  {
    gpuann_fann_run_small_neurons_implementation(neuronInputCount,
                                                 inputArray,
                                                 weightsArray,
                                                 sumArray,
                                                 outputArray,
                                                 layerSteepness,
                                                 layerActivationFunction,
                                                 neuronCount,
                                                 instanceCount,
                                                 totalNeuronsCount,
                                                 totalWeightsCount);
  }
  else
  {
    gpuann_fann_run_simple_implementation(neuronInputCount,
                                          inputArray,
                                          weightsArray,
                                          sumArray,
                                          outputArray,
                                          layerSteepness,
                                          layerActivationFunction,
                                          neuronCount,
                                          instanceCount,
                                          totalNeuronsCount,
                                          totalWeightsCount);
  }
}

void gpuann_fann_run_implementation(gpuann &data)
{
  const fann * ann = data._fann;
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  struct fann_layer *lastLayer = ann->last_layer;

  for(struct fann_layer *layerIt = ann->first_layer + 1; layerIt != lastLayer; layerIt++)
  {
    struct fann_neuron * lastNeuron = layerIt->last_neuron;
    struct fann_neuron * neuronIt   = layerIt->first_neuron;

    unsigned int neuronsCount            = lastNeuron - neuronIt;
    fann_type    layerSteepness          = neuronIt->activation_steepness;
    unsigned int layerActivationFunction = neuronIt->activation_function;
    unsigned int layerNeuronInputCount   = neuronIt->last_con - neuronIt->first_con;
    unsigned int inputNeuronArrayShift   = (layerIt - 1)->first_neuron - neuronsArray;
    unsigned int currentNeuronArrayShift = neuronIt - neuronsArray;
    unsigned int weightsArrayShift       = neuronIt->first_con;

    gpuann_fann_run_select_best_implementation(layerNeuronInputCount,
                                               &(data.d_valuesArray[inputNeuronArrayShift]),
                                               &(data.d_weightsArray[weightsArrayShift]),
                                               &(data.d_sumArray[currentNeuronArrayShift]),
                                               &(data.d_valuesArray[currentNeuronArrayShift]),
                                               layerSteepness,
                                               layerActivationFunction,
                                               neuronsCount,
                                               data._instanceCount,
                                               data._neuronsCountPerInstance,
                                               data._weightsCountPerInstance);
  }
}

