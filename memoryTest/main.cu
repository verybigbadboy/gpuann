/*#include <string>
#include <stdio.h>
#include <math.h>

#include <common.h>
#include <cuda_runtime.h>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
  __device__ inline operator       T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
  
  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
  __device__ inline operator       double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
  
  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
  T *sdata = SharedMemory<T>();

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

  while (i < n)
  {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i+blockSize];

    i += gridSize;
  }

  sdata[tid] = mySum;
  __syncthreads();

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();
  }

  if (blockSize >= 128)
  {
    if (tid <  64)
    {
      sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();
  }

  if (tid < 32)
  {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T *smem = sdata;

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

  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  //powerOf2
  if ((size & (size - 1)) == 0)
  {
    switch (threads)
    {
      case 512:
        reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256:
        reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128:
        reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 64:
        reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 32:
        reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 16:
        reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  8:
        reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  4:
        reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  2:
        reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  1:
        reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512:
        reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256:
        reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128:
        reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 64:
        reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 32:
        reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 16:
        reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  8:
        reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  4:
        reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  2:
        reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  1:
        reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
}

// Instantiate the reduction function for 3 types
template void reduce<int>(int size, int threads, int blocks, int *d_idata, int *d_odata);

template void reduce<float>(int size, int threads, int blocks, float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, double *d_idata, double *d_odata);



void reduceExample()
{
  printf("reduce example\n");
  int elementsCount = 1 << 24;
  int threads = 256;
  int blocks = (elementsCount + (threads * 2 - 1)) / (threads * 2);
  int resultElementsCount = blocks;
  
  size_t elementsCountSize = elementsCount * sizeof(float);
  size_t resultElementsCountSize = elementsCount * sizeof(float);
  
  float *h_a =      (float *)malloc(elementsCountSize);
  float *h_result = (float *)malloc(resultElementsCountSize);
  
  for(int i = 0; i < elementsCount; ++i)
  {
    h_a[i] = rand()/(float)RAND_MAX;
  }
  
  float *d_a = 0;
  float *d_result = 0;
  
  cudaMalloc((void **)&d_a, elementsCountSize);
  cudaMalloc((void **)&d_result, resultElementsCountSize);
  
  cudaMemcpy(d_a, h_a, elementsCountSize, cudaMemcpyHostToDevice);

  reduce <float> (elementsCount, threads, blocks, d_a, d_result);
  
  cudaMemcpy(h_result, d_result, resultElementsCountSize, cudaMemcpyDeviceToHost);

  double gpuSum = 0;

  for (int i = 0; i < resultElementsCount; ++i)
  {
    gpuSum += h_result[i];
  }
  
  double sum = 0;
  for (int i = 0; i < elementsCount; ++i)
  {
    sum += h_a[i];
  }

  printf("GPU = %f\nCPU = %f\n", gpuSum, sum);
  
  cudaFree(d_a);
  cudaFree(d_result);
  
  free(h_a);
  free(h_result);
  
  printf("done\n");
}


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    C[i] = A[i] + B[i];
  }
}

void vectorAddExample()
{
  printf("vector add example\n");
  int elementsCount = 1 << 21;
  size_t elementsCountSize = elementsCount * sizeof(float);
  float *h_a = (float *)malloc(elementsCountSize);
  float *h_b = (float *)malloc(elementsCountSize);
  float *h_c = (float *)malloc(elementsCountSize);

  for(int i = 0; i < elementsCount; ++i)
  {
    h_a[i] = rand()/(float)RAND_MAX;
    h_b[i] = rand()/(float)RAND_MAX;
  }

  float *d_a = 0;
  float *d_b = 0;
  float *d_c = 0;

  cudaMalloc((void **)&d_a, elementsCountSize);
  cudaMalloc((void **)&d_b, elementsCountSize);
  cudaMalloc((void **)&d_c, elementsCountSize);

  cudaMemcpy(d_a, h_a, elementsCountSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, elementsCountSize, cudaMemcpyHostToDevice);


  int threadsPerBlock = 256;
  int blocksPerGrid = (elementsCount + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd <<<blocksPerGrid, threadsPerBlock>>> (d_a, d_b, d_c, elementsCount);

  cudaMemcpy(h_c, d_c, elementsCountSize, cudaMemcpyDeviceToHost);

  for (int i = 0; i < elementsCount; ++i)
  {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  printf("done\n");
}

int main1()
{
  printf("Started\n");
  cudaDeviceInit();

  vectorAddExample();

  reduceExample();

  printf("Stoped\n");
  return 0;
}*/