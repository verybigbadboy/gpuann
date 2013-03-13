#include <gpuann.h>
#include <base/gpuannDataCreator.h>
#include <base/neuralNetworkTypeCheck.h>
#include <kernels/straightforward/run.h>

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
  loadgpuann(gann, ann);
  gpuann_fann_run_implementation(ann, gann);
  savegpuann(gann, ann);
  
  fann_type *output = ann->output;
  unsigned int num_output = ann->num_output;
  fann_neuron *neurons = (ann->last_layer - 1)->first_neuron;
  for(unsigned int i = 0; i != num_output; i++)
  {
    output[i] = neurons[i].value;
  }
  return ann->output;
}

