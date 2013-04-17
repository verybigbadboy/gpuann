#include <kernels/mergeSlopes/mergeSlopes.h>

#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>



template <unsigned int blockSize>
__global__ void gpuann_merge_slopes_gpu_kernel(unsigned int weightsCount, fann_type *slopesBegin, unsigned int instanceCount)
{
  unsigned int tid         = threadIdx.x;
  //from last
  unsigned int weightIndex = blockIdx.x * blockSize + tid + weightsCount * (instanceCount - 1);

  fann_type sum = 0;

  if(blockIdx.x * blockSize + tid < weightsCount)
  {
    while(weightIndex >= weightsCount)
    {
      sum += slopesBegin[weightIndex];
      weightIndex -= weightsCount;
    }

    slopesBegin[weightIndex] += sum;
  }
}

void gpuann_merge_slopes_implementation(gpuann &data)
{
  unsigned int weightsCount = data._weightsCountPerInstance;
  unsigned int instanceCount = data._instanceCount;
  
  
  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsCount / threadCount + 1, 1, 1);

  gpuann_merge_slopes_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (weightsCount, data.d_trainSlopes, instanceCount);
}