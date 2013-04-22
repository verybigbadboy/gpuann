// includes, CUDA
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cuda.h>
#include <builtin_types.h>
#include <common/cuda/drvapi_error_string.h>
#include <cuda_runtime_api.h>

// Variables
static CUdevice cuDevice;
static CUcontext cuContext;
static CUmodule cuModule;


#define checkCudaErrors(err)  __checkCudaErrors ((CUresult)(err), __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
  if (CUDA_SUCCESS != err)
  {
    fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
            err, getCudaDrvErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

inline int cudaDeviceInit()
{
  int cuDevice = 0;
  int dev = 0;
  int deviceCount = 0;
  CUresult err = cuInit(0);

  if (CUDA_SUCCESS == err)
  {
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  }

  if (deviceCount == 0)
  {
    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  printf("CUDA Device count %d\n", deviceCount);
  checkCudaErrors(cuDeviceGet(&cuDevice, dev));
  cudaSetDevice(cuDevice);
  char name[100];
  cuDeviceGetName(name, 100, cuDevice);

  printf("> Using CUDA Device [%d]: %s\n", dev, name);
  
  return dev;
}

#endif // COMMON_H