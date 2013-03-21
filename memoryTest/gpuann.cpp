#include <gpuann.h>
#include <base/gpuannDataCreator.h>
#include <base/neuralNetworkTypeCheck.h>
#include <kernels/straightforward/run.h>
#include <kernels/backpropagateMSE/backpropagateMSErun.h>
#include <kernels/computeMSE/run.h>
#include <kernels/updateWeights/updateWeights.h>

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

#include <cuda.h>
#include <cuda_runtime.h>

void gpuann_fann_train(struct fann *ann, fann_type * input, fann_type * desired_output)
{
  check(ann);
  loadInputs(ann, input);

  debugGpuann dump1, dump2, dump3;
  gpuann gann;
  creategpuann(gann, ann);
  loadgpuann(gann, ann);

  fann_type *d_desired_output;
  cudaMalloc((void **)&(d_desired_output), ann->num_output * sizeof(fann_type));
  cudaMemcpyAsync(d_desired_output, desired_output, ann->num_output * sizeof(fann_type), cudaMemcpyHostToDevice);
  
  gpuann_fann_run_implementation(gann);
  gpuann_fann_compute_MSE_implementation_gpu(gann, d_desired_output);
  createDump(gann, dump1);
  gpuann_fann_backpropagate_MSE_implementation_gpu(gann);
  createDump(gann, dump2);
  gpuann_fann_update_weights_implementation(gann);
  createDump(gann, dump3);

  fann *cpuann = fann_copy(ann);

  fann_run(cpuann, input);

  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;
  
  fann_compute_MSE(cpuann, desired_output);

  for(int i = 0; i < neuronCount; ++i)
  {
    printf("\n%3d %f %f", i, cpuann->train_errors[i], dump1.d_trainErrorsArray[i]);
  }

  fann_backpropagate_MSE(cpuann);

  printf("\nBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB");

  for(int i = 0; i < neuronCount; ++i)
  {
    printf("\n%f %f", cpuann->train_errors[i], dump2.d_trainErrorsArray[i]);
  }
  
  fann_update_weights(cpuann);
  
  savegpuann(gann, ann);
  
  removegpuann(gann);
}