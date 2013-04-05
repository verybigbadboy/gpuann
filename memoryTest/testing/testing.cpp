#include <testing/testing.h>
#include <gpuann.h>

void printannsum(struct fann *ann)
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

void printannvalue(struct fann *ann)
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
      printf("%f ", neuron_it->value);
    }
    printf("\n");
  }
}

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

  fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

  fann_init_weights(ann, data);

  fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
  
  test(ann, data);

  fann_destroy_train(data);

  return ann;
}

bool runTest(struct fann *ann, fann_type * input, const char * testName, bool fullreport)
{
  if(!fullreport)
    printf("Test %s \n", testName);
  fann_type *calc_out;
  fann_type calc_out_gpu, calc_out_cpu;

  calc_out = gpuann_fann_run(ann, input);
  calc_out_gpu = calc_out[0];

  if(!fullreport)
    printf("GPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  //printann(ann);

  calc_out = fann_run(ann, input);
  calc_out_cpu = calc_out[0];

  if(!fullreport)
    printf("CPU xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

  bool success = (calc_out_cpu - calc_out_gpu) * (calc_out_cpu - calc_out_gpu) < 0.001;

  fann_type want = 0;
  gpuann_fann_train(ann, input, &want);

  calc_out = fann_run(ann, input);
  calc_out_cpu = calc_out[0];

  if(!fullreport)
    printf("gPU TRAIN xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);
  
  //printann(ann);

  
  if(!fullreport)
    if(success)
      printf("Passed\n");
    else
      printf("Failed!!!\n");
    return success;
}

void runTests(struct fann *ann, bool fullreport)
{
  bool success = true;

  fann_type input[2];
  input[0] = -1;
  input[1] = -1;
  success &= runTest(ann, input, "00", fullreport);
  input[0] = -1;
  input[1] = 1;
  success &= runTest(ann, input, "01", fullreport);
  input[0] = 1;
  input[1] = -1;
  success &= runTest(ann, input, "10", fullreport);
  input[0] = 1;
  input[1] = 1;
  success &= runTest(ann, input, "11", fullreport);

  if(success)
    printf("runTests Passed\n");
  else
    printf("runTests Failed!!!\n");
}

void fulltest()
{
  for(int i = 1; i < 7; ++i)
  {
    for(int j = 30; j < 512; j *=2)
    {
      fann *ann = createSpecificTrainedFann(i, j);
      runTests(ann);
      fann_destroy(ann);
    }
  }
}
