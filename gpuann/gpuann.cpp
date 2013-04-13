#include <gpuann.h>
#include <base/gpuannDataCreator.h>
#include <base/neuralNetworkTypeCheck.h>

#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <kernels/computeMSE/run.h>
#include <kernels/straightforward/straightforward.h>
#include <kernels/updateWeights/updateWeights.h>
#include <gpuannTrain.h>

#include <cuda.h>
#include <cuda_runtime.h>

void loadInputs(struct fann * ann, fann_type * input)
{
  struct fann_neuron *neuronArray = ann->first_layer->first_neuron;

  unsigned int num_input = ann->num_input;
  for(unsigned int i = 0; i != num_input; i++)
  {
    neuronArray[i].value = input[i];
  }

  //fixme
  (ann->first_layer->last_neuron - 1)->value = 1;
}

fann_type * gpuann_fann_run(struct fann * ann, fann_type * input)
{
  check(ann);
  loadInputs(ann, input);

  gpuann gann;
  creategpuann(gann, ann);
  loadgpuann(gann, ann);

  gpuann_fann_run_implementation(gann);
  savegpuann(gann, ann);

  fann_type *output = ann->output;
  unsigned int num_output = ann->num_output;
  fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
  for(unsigned int i = 0; i != num_output; i++)
  {
    output[i] = neurons[i].value;
  }
  removegpuann(gann);

  return ann->output;
}

void gpuann_fann_multirun(struct fann * ann, fann_type ** input, unsigned int instanceCount, fann_type ** output)
{
  check(ann);

  gpuann gann;
  creategpuann(gann, ann, instanceCount);

  for(unsigned int i = 0; i < instanceCount; ++i)
  {
    loadInputs(ann, input[i]);
    loadgpuann(gann, ann, i);
  }

  gpuann_fann_run_implementation(gann);

  unsigned int num_output = ann->num_output;
  for(unsigned int i = 0; i < instanceCount; ++i)
  {
    savegpuann(gann, ann, i);

    fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
    for(unsigned int j = 0; j < num_output; ++j)
    {
      output[i][j] = neurons[j].value;
    }
  }

  removegpuann(gann);
}

void gpuann_fann_train(struct fann *ann, fann_type * input, fann_type * desired_output)
{
  check(ann);
  loadInputs(ann, input);

  gpuann gann;
  creategpuann(gann, ann);
  loadgpuann(gann, ann);

  fann_type *d_desired_output;
  cudaMalloc((void **)&(d_desired_output), ann->num_output * sizeof(fann_type));
  cudaMemcpy(d_desired_output, desired_output, ann->num_output * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  gpuann_fann_run_implementation(gann);
  gpuann_fann_compute_MSE_implementation_gpu(gann, d_desired_output);
  gpuann_fann_backpropagate_MSE_implementation_gpu(gann);
  gpuann_fann_update_weights_implementation(gann);

  savegpuann(gann, ann);

  removegpuann(gann);
}

void gpuann_fann_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs)
{
  float error;
  unsigned int i;

  gpuann data;
  gpuannTrainData trainData;

  creategpuannTrainData(trainData, train);

  (ann->first_layer->last_neuron - 1)->value = 1; ///bias input TODO!
  creategpuann(data, ann);
  loadgpuann(data, ann);

  for(i = 1; i <= maxEpochs; i++)
  {
    error = gpuann_fann_train_epoch(data, trainData);
    //TODO
    //desired_error_reached = fann_desired_error_reached(ann, desiredError);

    /*
    if(epochs_between_reports && (i % epochs_between_reports == 0 || i == max_epochs || i == 1 || desired_error_reached == 0))
    {
      if(ann->callback == NULL)
      {
        printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error, ann->num_bit_fail);
      }
      else if(((*ann->callback)(ann, data, max_epochs, epochs_between_reports, desired_error, i)) == -1)
      {
        break;
      }
    }

    if(desiredErrorReached == 0)
      break;
    */
  }

  savegpuann(data, ann);

  removegpuann(data);
  removegpuannTrainData(trainData);
}