#include <kernels/mergeSlopes/mergeSlopes.h>

#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>



template <unsigned int blockSize>
__global__ void gpuann_merge_slopes_gpu_kernel(unsigned int weightsCount, fann_type *slopesBegin, unsigned int instanceCount)
{
  unsigned int tid         = threadIdx.x;
  //from last
  unsigned int weightIndex = blockIdx.x * blockSize * 2 + tid;
  unsigned int weightIndexInstance = weightIndex + weightsCount * (instanceCount - 1);

  fann_type sum[2] = {0, 0};

  if(weightIndex < weightsCount)
  {
    if(weightIndex + blockSize < weightsCount)
    {
      while(weightIndexInstance >= weightsCount)
      {
        sum[0] += slopesBegin[weightIndexInstance];
        __syncthreads();
        sum[1] += slopesBegin[weightIndexInstance + blockSize];
        __syncthreads();
        weightIndexInstance -= weightsCount;
      }

      slopesBegin[weightIndexInstance] += sum[0];
      __syncthreads();
      slopesBegin[weightIndexInstance + blockSize] += sum[1];
    }
    else
    {
      while(weightIndexInstance >= weightsCount)
      {
        sum[0] += slopesBegin[weightIndexInstance];
        __syncthreads();
        weightIndexInstance -= weightsCount;
      }

      slopesBegin[weightIndexInstance] += sum[0];
    }
  }
}

void gpuann_merge_slopes_implementation(gpuann &data)
{
  unsigned int weightsCount = data._weightsCountPerInstance;
  unsigned int instanceCount = data._instanceCount;
  
  
  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsCount / threadCount / 2 + 1, 1, 1);

  gpuann_merge_slopes_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (weightsCount, data.d_trainSlopes, instanceCount);
}