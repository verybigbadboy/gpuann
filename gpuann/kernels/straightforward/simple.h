#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>
#include <base/gpuannData.h>


template <unsigned int blockSize, unsigned int layerActivationFunction>
__global__ void runGpuKernel(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness
                            , unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  __shared__ fann_type local[blockSize];

  unsigned int tid = threadIdx.x;
  unsigned int instance = blockIdx.y;

  fann_type l_summ = 0;

  if(tid < neuronInputCount)
  {
    l_summ = (fann_type) (inputArray[tid + totalNeuronsCount * instance] * weightsArray[neuronInputCount * blockIdx.x + tid + totalWeightsCount * instance]);
    if((tid + blockSize) < neuronInputCount)
      l_summ += (fann_type) (inputArray[tid + blockSize + totalNeuronsCount * instance] * weightsArray[neuronInputCount * blockIdx.x + tid + blockSize + totalWeightsCount * instance]);
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

  //avoid of access after local memory
  unsigned int localMemorySize = blockSize / 2;
  if(localMemorySize > 32)
    localMemorySize = 32;

  if (tid < localMemorySize)
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

  //TODO: why  remove this line causes random value
  __syncthreads();

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

    sumArray[blockIdx.x + totalNeuronsCount * instance] = neuron_sum;

    fann_activation_switch(layerActivationFunction, neuron_sum, outputArray[blockIdx.x + totalNeuronsCount * instance]);
  }
}

//for neurons with < 32 inputs
//always 32 threads, one input = one thread
template <unsigned int layerActivationFunction>
__global__ void gpuann_fann_run_multineuron_gpu_kernel(unsigned int neuronInputCount,
                                                       unsigned int neuronCount,
                                                       fann_type * inputArray,
                                                       fann_type * weightsArray,
                                                       fann_type * sumArray,
                                                       fann_type * outputArray,
                                                       fann_type layerSteepness,
                                                       unsigned int totalNeuronsCount,
                                                       unsigned int totalWeightsCount)
{
  const unsigned int threadCount = 256;
  //unsigned int threadCount          = blockDim.x;

  unsigned int tid                  = threadIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int inputIndex           = tid % neuronInputCount;
  unsigned int inputIndexInstanced  = inputIndex + totalNeuronsCount * instance;
  unsigned int neuronIndexInKernel  = tid / neuronInputCount;
  unsigned int neuronsPerKernel     = threadCount / neuronInputCount;
  unsigned int neuronIndex          = blockIdx.x * neuronsPerKernel + neuronIndexInKernel;
  unsigned int weightIndex          = neuronInputCount * neuronIndex + inputIndex;
  unsigned int neuronIndexInstanced = neuronIndex + totalNeuronsCount * instance;
  unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;
  

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

#define runGpuActivatedCase(X) case X: \
runGpuKernel <blockSize, X> <<<dimGrid, dimBlock>>>(neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, totalNeuronsCount, totalWeightsCount); \
break;

#define gpuann_fann_run_multineuron_gpu_kernelCase(X) case X: \
gpuann_fann_run_multineuron_gpu_kernel <X> <<<dimGrid, dimBlock>>>(neuronInputCount, neuronCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, totalNeuronsCount, totalWeightsCount); \
break;

template <unsigned int blockSize>
void runGpuActivated(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount, unsigned int instanceCount, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((neuronCount - 1), instanceCount, 1);
  
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

#define runGpuThreadsCase(X) case X: \
runGpuActivated <X> (neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, layerActivationFunction, neuronCount, instanceCount, totalNeuronsCount, totalWeightsCount); \
break;

void runGpu(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount, unsigned int instanceCount, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  if(neuronInputCount < 32)
  {
    neuronCount--; //bias
    unsigned int threadNeeded = pow2roundup(neuronInputCount * neuronCount);
    if(threadNeeded > 256)
      threadNeeded = 256;
    unsigned int neuronsPerBlock = threadNeeded / neuronInputCount;
    unsigned int blocksNeeded = neuronCount / neuronsPerBlock + 1;
    dim3 dimBlock(threadNeeded, 1, 1);
    dim3 dimGrid(blocksNeeded, instanceCount, 1); // TODO create bias if

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
  else
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
}

void gpuann_fann_run_implementation(gpuann &data)
{
  const fann * ann = data._fann;
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
    
    runGpu(layerNeuronInputCount
    , &(data.d_valuesArray[inputNeuronArrayShift])
    , &(data.d_weightsArray[weightsArrayShift])
    , &(data.d_sumArray[currentNeuronArrayShift])
    , &(data.d_valuesArray[currentNeuronArrayShift])
    , layerSteepness
    , layerActivationFunction
    , last_neuron - neuron_it
    , data._instanceCount
    , data._neuronsCountPerInstance
    , data._weightsCountPerInstance
    );
  }
}

