#include <stdio.h>
#include <gpuann.h>
#include <common/cuda/common.h>

void printann(struct fann *ann)
{
  struct fann_layer *last_layer = ann->last_layer;
  for(struct fann_layer *layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it < last_neuron; ++neuron_it)
    {
      if(neuron_it->last_con == neuron_it->first_con)
        printf("A");
      printf("%f ", neuron_it->sum);
    }
    printf("\n");
  }
}

void runTest(struct fann *ann, fann_type * input, const char * testName)
{
  printf("Test %s \n", testName);
  fann_type *calc_out;
  fann_type calc_out_gpu, calc_out_cpu;

  calc_out = gpuann_fann_run(ann, input);
  calc_out_gpu = calc_out[0];

  printf("GPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  //printann(ann);

  calc_out = fann_run(ann, input);
  calc_out_cpu = calc_out[0];

  printf("CPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  //printann(ann);

  if( (calc_out_cpu - calc_out_gpu) * (calc_out_cpu - calc_out_gpu) < 0.001 )
    printf("Passed\n");
  else
    printf("Failed!!!\n");
}

void runTestMultiRun(struct fann *ann)
{
  fann_type **input = new fann_type*[400];
  fann_type **output = new fann_type*[400];
  for(unsigned int i = 0; i < 400; ++i)
  {
    input[i]  = new fann_type[2];
    output[i] = new fann_type;
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

  for(unsigned int i = 0; i < 4; ++i)
  {
    printf("%f\n", output[i][0]);
  }  
}

void runTests(struct fann *ann)
{
  fann_type input[2];
  input[0] = 0;
  input[1] = 0;
  runTest(ann, input, "00");
  input[0] = 0;
  input[1] = 1;
  runTest(ann, input, "01");
  input[0] = 1;
  input[1] = 0;
  runTest(ann, input, "10");
  input[0] = 1;
  input[1] = 1;
  runTest(ann, input, "11");  
}

void gpubench(struct fann *ann)
{
  fann_type input[2];
  input[0] = 0;
  input[1] = 0;
  runTest(ann, input, "00");
}

void cpubench(struct fann *ann)
{
  fann_type input[2];
  input[0] = 0;
  input[1] = 0;
  for(int i = 0; i < 1e6; i++)
    fann_run(ann, input);
}

int main()
{
  cudaDeviceInit();

  struct fann *ann = fann_create_from_file("xor_float.net");

  //cpubench(ann);
  //gpubench(ann);
  //runTests(ann);
  runTestMultiRun(ann);

  fann_destroy(ann);
  return 0;
}
