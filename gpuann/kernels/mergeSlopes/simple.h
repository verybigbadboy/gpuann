#include <kernels/mergeSlopes/mergeSlopes.h>

#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>



template <unsigned int blockSize>
__global__ void gpuann_merge_slopes_gpu_kernel(unsigned int weightsCount, fann_type *toSlopes, fann_type *fromSlopes)
{
  unsigned int tid         = threadIdx.x;
  unsigned int weightIndex = blockIdx.x * blockSize + tid;

  if(weightIndex < weightsCount)
    toSlopes[weightIndex] += fromSlopes[weightIndex];
}

void gpuann_merge_slopes_implementation(gpuann &data)
{
  unsigned int weightsCount = data._weightsCountPerInstance;
  unsigned int instanceCount = data._instanceCount;
  
  
  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsCount / threadCount + 1, 1, 1);
  
  for(unsigned int instance = 1; instance < instanceCount; ++instance)
  {
    gpuann_merge_slopes_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (weightsCount, data.d_trainSlopes, &(data.d_trainSlopes[instance * weightsCount]));
  }
}