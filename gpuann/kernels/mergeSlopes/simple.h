#include <kernels/mergeSlopes/mergeSlopes.h>

#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>



template <unsigned int blockSize>
__global__ void gpuann_merge_slopes_gpu_kernel(unsigned int weightsCount, fann_type *slopesBegin, unsigned int instanceCount)
{
  unsigned int tid         = threadIdx.x;
  //from last
  unsigned int weightIndex = (blockIdx.x * blockSize  + tid) * 4;
  unsigned int weightIndexInstanced = weightIndex + weightsCount * (instanceCount - 1);

  if(weightIndex < weightsCount)
  {
    if(weightIndex + 4 > weightsCount)
    {
      fann_type sum[4] = {0,0,0,0};

      while(weightIndexInstanced >= weightsCount)
      {
        if(weightIndex + 0 < weightsCount)
          sum[0] += slopesBegin[weightIndexInstanced];
        if(weightIndex + 1 < weightsCount)
          sum[1] += slopesBegin[weightIndexInstanced + 1];
        if(weightIndex + 2 < weightsCount)
          sum[2] += slopesBegin[weightIndexInstanced + 2];
        weightIndexInstanced -= weightsCount;
      }

      if(weightIndex + 0 < weightsCount)
        slopesBegin[weightIndexInstanced]     += sum[0];
      if(weightIndex + 1 < weightsCount)
        slopesBegin[weightIndexInstanced + 1] += sum[1];
      if(weightIndex + 2 < weightsCount)
        slopesBegin[weightIndexInstanced + 2] += sum[2];
    }
    else
    {
      fann_type sum[4] = {0,0,0,0};

      while(weightIndexInstanced >= weightsCount)
      {
        sum[0] += slopesBegin[weightIndexInstanced];
        sum[1] += slopesBegin[weightIndexInstanced + 1];
        sum[2] += slopesBegin[weightIndexInstanced + 2];
        sum[3] += slopesBegin[weightIndexInstanced + 3];
        weightIndexInstanced -= weightsCount;
      }

      slopesBegin[weightIndexInstanced]     += sum[0];
      slopesBegin[weightIndexInstanced + 1] += sum[1];
      slopesBegin[weightIndexInstanced + 2] += sum[2];
      slopesBegin[weightIndexInstanced + 3] += sum[3];
    }
  }
}

void gpuann_merge_slopes_implementation(gpuann &data)
{
  unsigned int weightsCount = data._weightsCountPerInstance;
  unsigned int instanceCount = data._instanceCount;

  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsCount / threadCount / 4  + 1, 1, 1);

  gpuann_merge_slopes_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (weightsCount, data.d_trainSlopes, instanceCount);
}