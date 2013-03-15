#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <unsigned int blockSize, unsigned int layerActivationFunction>
__global__ void runGpuKernel(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness
                            , unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  __shared__ fann_type local[blockSize];

  unsigned int tid = threadIdx.x;

  fann_type l_summ = 0;

  if(tid < neuronInputCount)
  {
    l_summ = (fann_type) (inputArray[tid + totalNeuronsCount * blockIdx.y] * weightsArray[neuronInputCount * blockIdx.x + tid + totalWeightsCount * blockIdx.y]);
    if((tid + blockSize) < neuronInputCount)
      l_summ += (fann_type) (inputArray[tid + blockSize + totalNeuronsCount * blockIdx.y] * weightsArray[neuronInputCount * blockIdx.x + tid + blockSize + totalWeightsCount * blockIdx.y]);
  }
  
  if(tid < blockSize)
  {
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

        sumArray[blockIdx.x + totalNeuronsCount * blockIdx.y] = neuron_sum;

      fann_activation_switch(layerActivationFunction, neuron_sum, outputArray[blockIdx.x + totalNeuronsCount * blockIdx.y]);
    }
  }
}
