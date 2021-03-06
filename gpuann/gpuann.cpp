#include <gpuann.h>
#include <base/gpuannDataCreator.h>
#include <base/neuralNetworkTypeCheck.h>

#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <kernels/computeMSE/run.h>
#include <kernels/straightforward/straightforward.h>
#include <kernels/updateWeights/updateWeights.h>
#include <gpuannTrain.h>
#include <gpuannParallelTrain.h>

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
    gpuann_fann_train_epoch(data, trainData);
  }

  savegpuann(data, ann);

  removegpuann(data);
  removegpuannTrainData(trainData);
}

void gpuann_fann_parallel_train_on_data(struct fann *ann, struct fann_train_data *train, unsigned int maxEpochs)
{
  float error;
  unsigned int i;

  gpuann data;
  gpuannTrainData trainData;

  creategpuannTrainData(trainData, train);

  (ann->first_layer->last_neuron - 1)->value = 1; ///bias input TODO!
  creategpuann(data, ann);
  loadgpuann(data, ann);

  gpuann_fann_parallel_train_on_data(data, trainData, maxEpochs);

  savegpuann(data, ann);

  removegpuann(data);
  removegpuannTrainData(trainData);
}
