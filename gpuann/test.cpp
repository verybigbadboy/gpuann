#include <stdio.h>
#include <gpuann.h>
#include <common/cuda/common.h>
#include <testing/testing.h>
#include <ctime>
#include <cstdio>

void runTestMultiRun(struct fann *ann)
{
  fann_type **input = new fann_type*[400];
  fann_type **output = new fann_type*[400];
  for(unsigned int i = 0; i < 400; ++i)
  {
    input[i]  = new fann_type[2];
    output[i] = new fann_type[1];
  }

  input[0][0] = 0;
  input[0][1] = 0;
  input[1][0] = 0;
  input[1][1] = 1;
  input[2][0] = 1;
  input[2][1] = 0;
  input[3][0] = 1;
  input[3][1] = 1;

  output[0][0] = 0;
  output[1][0] = 0;
  output[2][0] = 0;
  output[3][0] = 0;

  gpuann_fann_multirun(ann, (fann_type **) input, 400, (fann_type **)output);

  for(unsigned int i = 0; i < 400; ++i)
  {
    delete[] input[i];
    delete[] output[i];
  }
  delete [] input;
  delete [] output;
}

int main()
{
  cudaDeviceInit();

  fulltest();

  return 0;
}
