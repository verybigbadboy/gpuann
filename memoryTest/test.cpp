#include <stdio.h>
#include "floatfann.h"
#include "fannCuda.h"
#include "common.h"


int main()
{
  //cudaDeviceInit();
  
  fann_type *calc_out;
  fann_type input[2];
  
  struct fann *ann = fann_create_from_file("xor_float.net");
  
  input[0] = 1;
  input[1] = -1;
  calc_out = fann_run1(ann, input);

  printf("GPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  calc_out = fann_run(ann, input);
  printf("CPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  fann_destroy(ann);
  return 0;
}