#include <testing/testing.h>
#include <testing/train.h>
#include <gpuann.h>

fann *createSpecificTrainedFann(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer)
{
  if(num_hiden_layers > 7)
    return 0;
  const unsigned int num_input = 2;
  const unsigned int num_output = 1;
  const unsigned int num_layers = 2 + num_hiden_layers;
  const float desired_error = (const float) 0;
  const unsigned int max_epochs = 1000;
  const unsigned int epochs_between_reports = 0;

  struct fann *ann;
  struct fann_train_data *data;

  unsigned int p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
  p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = num_neurons_hidden_per_layer;
  switch (num_hiden_layers)
  {
    case 0:
      p1 = num_output;
      break;
    case 1:
      p2 = num_output;
      break;
    case 2:
      p3 = num_output;
      break;
    case 3:
      p4 = num_output;
      break;
    case 4:
      p5 = num_output;
      break;
    case 5:
      p6 = num_output;
      break;
    case 6:
      p7 = num_output;
      break;
    case 7:
      p8 = num_output;
      break;
    case 8:
      p9 = num_output;
      break;
    case 9:
      p10 = num_output;
      break;
  }

  ann = fann_create_standard(num_layers, num_input, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);

  data = fann_read_train_from_file("xor.data");

  fann_set_activation_steepness_hidden(ann, 1);
  fann_set_activation_steepness_output(ann, 1);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
  fann_set_bit_fail_limit(ann, 0.01f);

  fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);

  fann_init_weights(ann, data);

  testTrainMethods(ann, data);

  //fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
/*
  for(unsigned int i = 0; i < fann_length_train_data(data); i++)
  {
    fann_type *calc_out = fann_run(ann, data->input[i]);
    fann_type difference = fann_abs(calc_out[0] - data->output[i][0]);
    if(difference > 0.1)
      printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n", data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0], difference);
  }
*/
  fann_destroy_train(data);

  return ann;
}

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

void runTests(struct fann *ann, bool fullreport)
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

void fulltest()
{
  
  for(int i = 1; i < 4; ++i)
  {
    for(int j = 30; j < 1024; j *=2)
    {
      printf("Neural network type: %d %d\n", i, j);
      fann *ann = createSpecificTrainedFann(i, j);
      runTests(ann , false);
      fann_destroy(ann);
    }
  }

/*
  int i = 4, j = 240;
  printf("Neural network type: %d %d\n", i, j);
  fann *ann = createSpecificTrainedFann(i, j);
  runTests(ann , false);
  fann_destroy(ann);*/
}
