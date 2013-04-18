#include <kernels/d2dMemcpy/d2dMemcpy.h>

#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>



template <unsigned int blockSize>
__global__ void gpuann_d2dMemcpy_gpu_kernel(fann_type *d_dst, fann_type *d_src, unsigned int size)
{
  unsigned int tid   = threadIdx.x;
  unsigned int index = blockIdx.x * blockSize + tid;

  if(index < size)
    d_dst[index] = d_src[index];
}

void gpuann_d2dMemcpy(fann_type *d_dst, fann_type *d_src,  unsigned int size)
{
  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(size / threadCount + 1, 1, 1);

  gpuann_d2dMemcpy_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (d_dst, d_src, size);
}

template <unsigned int blockSize>
__global__ void gpuann_d2dMemcpy_multi_gpu_kernel(fann_type *d_dst, fann_type *d_src, unsigned int size, unsigned int srcOffset, unsigned int dstOffset)
{
  unsigned int tid               = threadIdx.x;
  unsigned int index             = blockIdx.x * blockSize * 2 + tid;
  unsigned int instance          = blockIdx.y;
  unsigned int destinationOffset = dstOffset * instance;
  unsigned int sourceOffset      = srcOffset * instance;

  if(index < size)
  {
    d_dst[index + destinationOffset] = d_src[index + sourceOffset];
    if(index + blockSize < size)
      d_dst[index + destinationOffset + blockSize] = d_src[index + sourceOffset + blockSize];
  }
}

void gpuann_d2dMemcpyMulti(fann_type *d_dst, fann_type *d_src,  unsigned int size, unsigned int instanceCount, unsigned int srcOffset, unsigned int dstOffset)
{
  const unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(size / threadCount / 2 + 1, instanceCount, 1);

  gpuann_d2dMemcpy_multi_gpu_kernel<threadCount> <<<dimGrid, dimBlock>>> (d_dst, d_src, size, srcOffset, dstOffset);
}


