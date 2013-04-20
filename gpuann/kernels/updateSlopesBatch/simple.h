
#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>

__global__ void gpuann_fann_update_slopes_batch_multineuron_gpu_kernel(unsigned int prevNeuronsCount,
                                                                       unsigned int neuronsCount,
                                                                       fann_type   *trainErrors,
                                                                       fann_type   *neuronSlopes,
                                                                       fann_type   *prevValue,
                                                                       unsigned int totalNeuronsCount,
                                                                       unsigned int totalWeightsCount)
{
  unsigned int tid                  = threadIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int threadCount          = blockDim.x;
  unsigned int neuronCountPerKernel = threadCount / prevNeuronsCount;
  unsigned int neuronIndex          = blockIdx.x * neuronCountPerKernel + tid / prevNeuronsCount;
  unsigned int actualPrevNeuron     = tid % prevNeuronsCount;
  unsigned int prevLayerNeuronIndex = actualPrevNeuron + instance * totalNeuronsCount;
  unsigned int slopesIndex          = actualPrevNeuron + prevNeuronsCount * neuronIndex + instance * totalWeightsCount;
  unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

  if(neuronIndex < (neuronsCount - 1)) //due to bias
  {
    fann_type error = trainErrors[neuronIndexInstanced];
    if(tid < prevNeuronsCount * neuronCountPerKernel)
      neuronSlopes[slopesIndex] += error * prevValue[prevLayerNeuronIndex];
  }
}

//neuronSlopes actually points to prev layer values
template <unsigned int blockSize>
__global__ void gpuann_fann_update_slopes_batch_gpu_kernel(unsigned int prevNeuronsCount,
                                                           unsigned int neuronsCount,
                                                           fann_type *trainErrors,
                                                           fann_type *neuronSlopes,
                                                           fann_type *prevValue,
                                                           unsigned int totalNeuronsCount,
                                                           unsigned int totalWeightsCount)
{
  unsigned int tid                  = threadIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int neuronIndex          = blockIdx.x;
  unsigned int prevLayerNeuronIndex = tid;
  unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

  fann_type error = trainErrors[neuronIndexInstanced];
  unsigned int prevLayerNeuronIndexInstanced;
  unsigned int slopesIndexInstanced;

  while(prevLayerNeuronIndex < prevNeuronsCount)
  {
    prevLayerNeuronIndexInstanced = prevLayerNeuronIndex + instance * totalNeuronsCount;
    slopesIndexInstanced          = prevLayerNeuronIndex + prevNeuronsCount * neuronIndex + instance * totalWeightsCount;

    neuronSlopes[slopesIndexInstanced] += error * prevValue[prevLayerNeuronIndexInstanced];
    prevLayerNeuronIndex += blockSize;
  }
}

template <unsigned int blockSize>
void gpuann_fann_update_slopes_batch_blockSize(unsigned int prevNeuronsCount,
                                               unsigned int neuronsCount,
                                               fann_type   *trainErrors,
                                               fann_type   *neuronSlopes,
                                               fann_type   *prevValue,
                                               unsigned int totalNeuronsCount,
                                               unsigned int totalWeightsCount,
                                               unsigned int instanceCount
                                              )
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(neuronsCount - 1, instanceCount, 1); // TODO create bias if

  gpuann_fann_update_slopes_batch_gpu_kernel <blockSize> <<<dimGrid, dimBlock>>>(prevNeuronsCount,
                                                                           neuronsCount,
                                                                           trainErrors,
                                                                           neuronSlopes,
                                                                           prevValue,
                                                                           totalNeuronsCount,
                                                                           totalWeightsCount);
}

void gpuann_fann_update_slopes_batch_simple_implementation(unsigned int prevNeuronsCount,
                                                           unsigned int neuronsCount,
                                                           fann_type   *trainErrors,
                                                           fann_type   *neuronSlopes,
                                                           fann_type   *prevValue,
                                                           unsigned int totalNeuronsCount,
                                                           unsigned int totalWeightsCount,
                                                           unsigned int instanceCount)
{
#define gpuann_fann_update_slopes_batch_blockSizeCase(X)   case X: \
gpuann_fann_update_slopes_batch_blockSize<X>(prevNeuronsCount, neuronsCount, trainErrors, neuronSlopes, prevValue, totalNeuronsCount, totalWeightsCount, instanceCount); \
break;

unsigned int threadCount = pow2roundup(prevNeuronsCount) / 2;
  if(threadCount > 256)
    threadCount = 256;
  switch (threadCount)
  {
    gpuann_fann_update_slopes_batch_blockSizeCase(16);
    gpuann_fann_update_slopes_batch_blockSizeCase(32);
    gpuann_fann_update_slopes_batch_blockSizeCase(64);
    gpuann_fann_update_slopes_batch_blockSizeCase(128);
    gpuann_fann_update_slopes_batch_blockSizeCase(256);
  }
}

void gpuann_fann_update_slopes_batch_implementation(gpuann &data, fann_layer *layerBegin, fann_layer *layerEnd)
{
  const fann *ann = data._fann;
  unsigned int instanceCount = data._instanceCount;
  fann_neuron *firstNeuron = ann->first_layer->first_neuron;
  fann_layer *layerIt;

  for(layerIt = layerBegin; layerIt <= layerEnd; ++layerIt)
  {
    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronShift = layerIt->first_neuron - firstNeuron;
    unsigned int layerWeightShift = layerIt->first_neuron->first_con;

    fann_layer  *prevLayer = layerIt - 1;
    fann_neuron *prevLayerFirstNeuron = prevLayer->first_neuron;
    unsigned int prevLayerNeuronShift = prevLayerFirstNeuron - firstNeuron;
    unsigned int prevLayerSize = prevLayer->last_neuron - prevLayerFirstNeuron;

    if(prevLayerSize < 32)
    {
      unsigned int threadNeeded = pow2roundup((layerSize - 1) * prevLayerSize);
      if(threadNeeded > 256)
        threadNeeded = 256;
      unsigned int neuronsPerBlock = threadNeeded / prevLayerSize;
      unsigned int blocksNeeded = (layerSize - 1) / neuronsPerBlock + 1;
      dim3 dimBlock(threadNeeded, 1, 1);
      dim3 dimGrid(blocksNeeded, instanceCount, 1); // TODO create bias if

      gpuann_fann_update_slopes_batch_multineuron_gpu_kernel<<<dimGrid, dimBlock>>>(prevLayerSize,
        layerSize,
        &(data.d_trainErrorsArray[layerNeuronShift]),
        &(data.d_trainSlopes[layerWeightShift]),
        &(data.d_valuesArray[prevLayerNeuronShift]),
        data._neuronsCountPerInstance,
        data._weightsCountPerInstance
        );
    }
    else
    {
      gpuann_fann_update_slopes_batch_simple_implementation(prevLayerSize,
                                                            layerSize,
                                                            &(data.d_trainErrorsArray[layerNeuronShift]),
                                                            &(data.d_trainSlopes[layerWeightShift]),
                                                            &(data.d_valuesArray[prevLayerNeuronShift]),
                                                            data._neuronsCountPerInstance,
                                                            data._weightsCountPerInstance,
                                                            instanceCount
                                                           );
    }
  }
}

/* ORIGINAL
void fann_update_slopes_batch(struct fann *ann, struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons;
  fann_type tmp_error;
  unsigned int i, num_connections;

  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  fann_type *error_begin = ann->train_errors;
  fann_type *slope_begin, *neuron_slope;

  slope_begin = ann->train_slopes;

  prev_neurons = first_neuron;

  for(; layer_begin <= layer_end; layer_begin++)
  {
    last_neuron = layer_begin->last_neuron;
    prev_neurons = (layer_begin - 1)->first_neuron;

    for(neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      tmp_error = error_begin[neuron_it - first_neuron];
      neuron_slope = slope_begin + neuron_it->first_con;
      num_connections = neuron_it->last_con - neuron_it->first_con;
      for(i = 0; i != num_connections; i++)
      {
        neuron_slope[i] += tmp_error * prev_neurons[i].value;
      }
    }
  }
}
*/
