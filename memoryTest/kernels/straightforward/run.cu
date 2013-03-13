#include <kernels/straightforward/run.h>
#include <kernels/straightforward/simple.h>
#include <common/math.h>
#include <string>

#define runGpuActivatedCase(X) case X: \
runGpuKernel <blockSize, X> <<<dimGrid, dimBlock>>>(neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness); \
break;

template <unsigned int blockSize>
void runGpuActivated(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(neuronCount - 1, 1, 1);
  
  switch(layerActivationFunction)
  {
    runGpuActivatedCase(0);
    runGpuActivatedCase(1);
    runGpuActivatedCase(2);
    runGpuActivatedCase(3);
    runGpuActivatedCase(4);
    runGpuActivatedCase(5);
    runGpuActivatedCase(6);
    runGpuActivatedCase(7);
    runGpuActivatedCase(8);
    runGpuActivatedCase(9);
    runGpuActivatedCase(10);
    runGpuActivatedCase(11);
    runGpuActivatedCase(12);
    runGpuActivatedCase(13);
    runGpuActivatedCase(14);
    runGpuActivatedCase(15);
  }
}

#define runGpuThreadsCase(X) case X: \
runGpuActivated <X> (neuronInputCount, inputArray, weightsArray, sumArray, outputArray, layerSteepness, layerActivationFunction, neuronCount); \
break;

void runGpu(unsigned int neuronInputCount, fann_type * inputArray, fann_type * weightsArray, fann_type *sumArray, fann_type * outputArray, fann_type layerSteepness, unsigned int layerActivationFunction, unsigned int neuronCount)
{
  unsigned int threadsCount = pow2roundup(neuronInputCount) / 2;
  if(threadsCount < 4)
    threadsCount = 4;
  else
    if(threadsCount > 256)
      throw std::string("too many inputs");
    
    switch (threadsCount)
    {
      runGpuThreadsCase(4);
      runGpuThreadsCase(8);
      runGpuThreadsCase(16);
      runGpuThreadsCase(32);
      runGpuThreadsCase(64);
      runGpuThreadsCase(128);
      runGpuThreadsCase(256);
    }
}

fann_type * gpuann_fann_run_implementation(struct fann * ann, gpuann &data)
{
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  struct fann_layer *last_layer = ann->last_layer;
  
  for(struct fann_layer *layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    
    fann_type    layerSteepness = neuron_it->activation_steepness;
    unsigned int layerActivationFunction = neuron_it->activation_function;
    unsigned int layerNeuronInputCount = neuron_it->last_con - neuron_it->first_con;
    unsigned int inputNeuronArrayShift = (layer_it - 1)->first_neuron - neuronsArray;
    unsigned int currentNeuronArrayShift = neuron_it - neuronsArray;
    unsigned int weightsArrayShift = neuron_it->first_con;
    
    runGpu(layerNeuronInputCount
    , &(data.d_valuesArray[inputNeuronArrayShift])
    , &(data.d_weightsArray[weightsArrayShift])
    , &(data.d_sumArray[currentNeuronArrayShift])
    , &(data.d_valuesArray[currentNeuronArrayShift])
    , layerSteepness
    , layerActivationFunction
    , last_neuron - neuron_it);
  }
  return 0;
}
