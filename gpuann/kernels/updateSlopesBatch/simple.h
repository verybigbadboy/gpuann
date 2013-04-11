
#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>

//neuronSlopes actually points to prev layer values
__global__ void gpuann_fann_update_slopes_batch_gpu_kernel(unsigned int prevNeuronsCount, unsigned int neuronsCount, fann_type *trainErrors, fann_type *neuronSlopes, fann_type *prevValue
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  unsigned int tid      = threadIdx.x;
  unsigned int instance = blockIdx.y;
  unsigned int neuronIndex     = blockIdx.x + instance * totalNeuronsCount;
  unsigned int prevLayerNeuronIndex = tid + instance * totalNeuronsCount;
  unsigned int slopesIndex = tid + prevNeuronsCount * blockIdx.x + instance * totalWeightsCount;

  fann_type error = trainErrors[neuronIndex];
  if(tid < prevNeuronsCount)
    neuronSlopes[slopesIndex] += error * prevValue[prevLayerNeuronIndex];
}


#define gpuann_fann_update_slopes_batch_gpu_kernel_blockSize_case(X)   case X: \
gpuann_fann_update_slopes_batch_gpu_kernel<X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, trainErrors, neuronSlopes, prevValue, totalNeuronsCount); \
break;

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

    dim3 dimBlock(prevLayerSize, 1, 1);
    dim3 dimGrid(layerSize - 1, instanceCount, 1); // TODO create bias if

    gpuann_fann_update_slopes_batch_gpu_kernel<<<dimGrid, dimBlock>>>(prevLayerSize,
                                               layerSize,
                                               &(data.d_trainErrorsArray[layerNeuronShift]),
                                               &(data.d_trainSlopes[layerWeightShift]),
                                               &(data.d_valuesArray[prevLayerNeuronShift]),
                                               data._neuronsCountPerInstance,
                                               data._weightsCountPerInstance
                                              );
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
