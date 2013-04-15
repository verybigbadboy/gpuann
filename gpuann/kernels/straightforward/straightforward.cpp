
#include <kernels/straightforward/straightforward.h>
#include <base/gpuannDataCreator.h>
#include <cuda_runtime.h>

fann_type * gpuann_fann_run_device(gpuann &data, fann_type * d_input)
{
  gpuann_loadInputsAsync(data, d_input);
  cudaThreadSynchronize();
  gpuann_fann_run_implementation(data);
  return gpuann_getOutputsDevicePointer(data);
}