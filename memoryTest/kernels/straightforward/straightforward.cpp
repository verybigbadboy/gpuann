
#include <kernels/straightforward/straightforward.h>
#include <base/gpuannDataCreator.h>

fann_type * gpuann_fann_run_device(gpuann &data, fann_type * d_input)
{
  gpuann_loadInputs(data, d_input);
  gpuann_fann_run_implementation(data);
  return gpuann_getOutputsDevicePointer(data);
}