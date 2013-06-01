#include <common/fannHelper.h>
#include <configuration.h>

fann *createANN(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer, bool trained)
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
  
  if(trained)
  {
    fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
    
    for(unsigned int i = 0; i < fann_length_train_data(data); i++)
    {
      fann_type *calc_out = fann_run(ann, data->input[i]);
      fann_type difference = fann_abs(calc_out[0] - data->output[i][0]);
      if(difference > 0.1)
        printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n", data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0], difference);
    }
  }
  
  fann_destroy_train(data);
  
  return ann;
}

void printGPUANNConfig()
{
  printf("Parallel instances : %d\n", parallelTrainInstanceCountMax);
  printf("Backpropagation Multineuron ");
  if(backpropationMultiNeuronImplementationEnabled)
    printf("ENABLED\n");
  else
    printf("DISABLED\n");

  printf("Backpropagation Parallel ");
  if(backpropationParallelImplementationEnabled)
    printf("ENABLED from %d\n", backpropationParallelImplementationBeginCount);
  else
    printf("DISABLED\n");

  printf("Straighforward Smallneuron");
  if(straightforwardSmallNeuronImplementationEnabled)
    printf("ENABLED\n");
  else
    printf("DISABLED\n");

  printf("Update Slopes Batch Multineuron ");
  if(updateSlopesBatchMultiNeuronImplementationEnabled)
    printf("ENABLED from %d\n", updateSlopesBatchMultiNeuronImplementationBeginCount);
  else
    printf("DISABLED\n");

  printf("Update Slopes Bigneuron ");
  if(updateSlopesBatchBigNeuronImplementationEnabled)
    printf("ENABLED\n");
  else
    printf("DISABLED\n");

  printf("Minimal thread per block ");
  if(minimalThreadCountPerBlockOptimization)
    printf("ENABLED\n");
  else
    printf("DISABLED\n");
}
