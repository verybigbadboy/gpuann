#include <testing/run.h>
#include <gpuann/gpuann.h>

bool runTest(struct fann *ann, fann_type * input)
{
  fann_type *calc_out_c;
  fann_type *calc_out_g;
  fann_type calc_out_gpu, calc_out_cpu;
  
  fann *gpunn = fann_copy(ann);
  fann *cpunn = fann_copy(ann);
  
  calc_out_g = gpuann_fann_run(gpunn, input);
  calc_out_c = fann_run(cpunn, input);
  
  calc_out_gpu = calc_out_g[0];
  calc_out_cpu = calc_out_c[0];
  
  bool success = (calc_out_cpu - calc_out_gpu) * (calc_out_cpu - calc_out_gpu) < 0.001;
  
  fann_destroy(cpunn);
  fann_destroy(gpunn);
  
  return success;
}

void testRunMethods(fann *ann)
{
  bool success = true;
  
  fann_type input[2];
  input[0] = -1;
  input[1] = -1;
  success &= runTest(ann, input);
  input[0] = -1;
  input[1] = 1;
  success &= runTest(ann, input);
  input[0] = 1;
  input[1] = -1;
  success &= runTest(ann, input);
  input[0] = 1;
  input[1] = 1;
  success &= runTest(ann, input);
  
  if(success)
    printf("runTests PASSED\n");
  else
    printf("runTests FAILED\n");
}